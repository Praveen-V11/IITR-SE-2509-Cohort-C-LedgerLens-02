import io
import os
import tempfile
from pathlib import Path

# Test-only environment, set before any app.* module gets imported so that
# app.config's class-level os.getenv(...) defaults pick these up instead of
# whatever the real .env has. A plain file-based sqlite db (not :memory:)
# because the default sqlite engine opens a fresh connection per request,
# and an in-memory db would reset between calls without extra pooling
# config that isn't worth adding just for the test suite.
_TEST_DIR = Path(tempfile.gettempdir()) / "ledgerlens_test"
_TEST_DIR.mkdir(parents=True, exist_ok=True)
_TEST_DB_PATH = _TEST_DIR / "ledgerlens_test.db"
_TEST_DB_PATH.unlink(missing_ok=True)

os.environ.setdefault("LEDGERLENS_DB_URL", f"sqlite:///{_TEST_DB_PATH}")
os.environ.setdefault("LEDGERLENS_UPLOAD_DIR", str(_TEST_DIR / "uploads"))
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-a-real-secret")

import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from app.models.schemas import DocumentType, InvoiceSchema, LineItem  # noqa: E402


@pytest.fixture
def tiny_png_bytes() -> bytes:
    """A minimal in-memory PNG so tests never touch the filesystem."""
    img = Image.new("RGB", (64, 64), color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_invoice(**overrides) -> InvoiceSchema:
    """Builds a fully-populated, reconciling InvoiceSchema for tests, with
    the option to override individual fields per test case."""
    defaults = dict(
        document_type=DocumentType.invoice,
        vendor_name="Acme Supplies",
        vendor_confidence=0.95,
        invoice_number="INV-1001",
        invoice_number_confidence=0.9,
        document_date="2026-06-01",
        date_confidence=0.9,
        currency="USD",
        line_items=[
            LineItem(description="Widget A", quantity=2, unit_price="10.00", amount="20.00", confidence=0.92),
            LineItem(description="Widget B", quantity=1, unit_price="15.00", amount="15.00", confidence=0.88),
        ],
        subtotal="35.00",
        subtotal_confidence=0.93,
        tax="3.50",
        tax_confidence=0.9,
        total="38.50",
        total_confidence=0.91,
    )
    defaults.update(overrides)
    return InvoiceSchema(**defaults)


@pytest.fixture
def valid_invoice() -> InvoiceSchema:
    return make_invoice()
