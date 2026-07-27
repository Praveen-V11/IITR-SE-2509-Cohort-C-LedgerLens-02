from decimal import Decimal
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, RateLimitError
from pydantic import ValidationError

from app.models.schemas import InvoiceSchema
from app.services import extraction as extraction_service
from app.services.extraction import (
    ExtractionError,
    ExtractionRateLimitedError,
    ExtractionServiceError,
    ExtractionTimeoutError,
)


def _patch_client(monkeypatch, side_effect):
    fake_client = MagicMock()
    fake_client.beta.chat.completions.parse.side_effect = side_effect
    monkeypatch.setattr(extraction_service, "_client", lambda: fake_client)
    return fake_client


def test_rate_limit_raises_clear_extraction_error(monkeypatch, tiny_png_bytes):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request)
    _patch_client(monkeypatch, RateLimitError("rate limited", response=response, body=None))

    with pytest.raises(ExtractionRateLimitedError):
        extraction_service.extract_invoice(tiny_png_bytes)


def test_timeout_raises_clear_extraction_error(monkeypatch, tiny_png_bytes):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    _patch_client(monkeypatch, APITimeoutError(request=request))

    with pytest.raises(ExtractionTimeoutError):
        extraction_service.extract_invoice(tiny_png_bytes)


def test_connection_failure_raises_clear_extraction_error(monkeypatch, tiny_png_bytes):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    _patch_client(monkeypatch, APIConnectionError(request=request))

    with pytest.raises(ExtractionServiceError):
        extraction_service.extract_invoice(tiny_png_bytes)


def test_currency_formatted_amounts_are_cleaned_up_instead_of_rejected():
    """The model sometimes writes a monetary value back with its currency
    symbol and/or thousands separators still attached (e.g. "₹23,134.75")
    despite the prompt asking for plain numbers - this used to raise
    pydantic.ValidationError straight out of the OpenAI SDK's response
    parsing (an unhandled 500 in production). It should now be normalized
    to a plain Decimal instead of treated as an error."""
    payload = {
        "document_type": "invoice",
        "vendor_name": "Acme",
        "vendor_confidence": 0.9,
        "currency": "INR",
        "line_items": [
            {
                "description": "Widget",
                "quantity": 1,
                "unit_price": "₹23,134.75",
                "amount": "₹23,134.75",
                "confidence": 0.9,
            }
        ],
        "subtotal": "₹23,134.75",
        "subtotal_confidence": 0.9,
        "tax": "₹4,164.25",
        "tax_confidence": 0.9,
        "total": "₹27,299.00",
        "total_confidence": 0.9,
    }
    invoice = InvoiceSchema.model_validate(payload)
    assert invoice.subtotal == Decimal("23134.75")
    assert invoice.tax == Decimal("4164.25")
    assert invoice.total == Decimal("27299.00")
    assert invoice.line_items[0].unit_price == Decimal("23134.75")
    assert invoice.reconciles()


def test_genuinely_unparseable_amount_still_raises_clear_extraction_error(monkeypatch, tiny_png_bytes):
    """Formatting cleanup only covers currency symbols/thousands separators
    - actual garbage (or a missing required field) should still fail
    validation, and the SDK raising pydantic.ValidationError for that must
    still come back as a clean ExtractionError rather than a 500."""
    bad_payload = {
        "document_type": "invoice",
        "vendor_name": "Acme",
        "vendor_confidence": 0.9,
        "currency": "INR",
        "line_items": [],
        "subtotal": "not a number at all",
        "subtotal_confidence": 0.9,
        "tax": "0",
        "total": "0",
    }
    try:
        InvoiceSchema.model_validate(bad_payload)
    except ValidationError as exc:
        validation_error = exc
    else:
        pytest.fail("expected InvoiceSchema to reject genuinely unparseable input")

    _patch_client(monkeypatch, validation_error)

    with pytest.raises(ExtractionError):
        extraction_service.extract_invoice(tiny_png_bytes)
