from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.models.schemas import InvoiceSchema, LineItem
from tests.conftest import make_invoice


def test_round_trip_preserves_data(valid_invoice):
    """A schema that can't survive its own serialize/deserialize cycle is
    the single worst thing to ship - this is the test the CI gate lives on."""
    dumped = valid_invoice.model_dump_json()
    restored = InvoiceSchema.model_validate_json(dumped)
    assert restored == valid_invoice


def test_confidence_bounds_are_enforced():
    with pytest.raises(ValidationError):
        make_invoice(vendor_confidence=1.5)

    with pytest.raises(ValidationError):
        make_invoice(vendor_confidence=-0.1)


def test_line_item_rejects_negative_amount():
    with pytest.raises(ValidationError):
        LineItem(description="Bad row", quantity=1, unit_price="-5.00", amount="-5.00", confidence=0.5)


def test_field_confidences_includes_every_line_item(valid_invoice):
    scores = valid_invoice.field_confidences()
    line_item_keys = [k for k in scores if k.startswith("line_item[")]
    assert len(line_item_keys) == len(valid_invoice.line_items)


def test_overall_confidence_is_average_of_all_fields(valid_invoice):
    scores = list(valid_invoice.field_confidences().values())
    expected = round(sum(scores) / len(scores), 4)
    assert valid_invoice.overall_confidence() == expected


def test_reconciles_true_for_balanced_invoice(valid_invoice):
    assert valid_invoice.reconciles() is True


def test_reconciles_false_when_total_is_off(valid_invoice):
    broken = make_invoice(total="999.00")
    assert broken.reconciles() is False


def test_missing_optional_fields_do_not_break_validation():
    minimal = make_invoice(invoice_number=None, document_date=None)
    assert minimal.invoice_number is None
    assert minimal.model_dump_json()  # should not raise


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("₹23,134.75", Decimal("23134.75")),
        ("$1,234.5", Decimal("1234.50")),
        ("1,234", Decimal("1234.00")),
        ("23134.75 INR", Decimal("23134.75")),
        (Decimal("10.00"), Decimal("10.00")),
    ],
)
def test_currency_formatted_header_amounts_are_normalized_not_rejected(raw, expected):
    """The model doesn't always follow the prompt's "plain decimal only"
    rule - a currency symbol or thousands separator still slipping through
    should be cleaned up rather than blowing up schema validation."""
    invoice = make_invoice(subtotal=raw)
    assert invoice.subtotal == expected


def test_currency_formatted_line_item_amounts_are_normalized_not_rejected():
    item = LineItem(description="Widget", quantity=1, unit_price="₹1,000.00", amount="₹1,000.00", confidence=0.9)
    assert item.unit_price == Decimal("1000.00")
    assert item.amount == Decimal("1000.00")


def test_genuinely_non_numeric_amount_still_fails_validation():
    with pytest.raises(ValidationError):
        make_invoice(subtotal="not a number at all")
