from app.models.schemas import ReviewStatus
from app.services import confidence as confidence_service
from tests.conftest import make_invoice


def test_high_confidence_reconciling_invoice_is_auto_approved(valid_invoice):
    decision = confidence_service.route(valid_invoice, threshold=0.75)
    assert decision.status == ReviewStatus.auto_approved
    assert decision.flagged_fields == []


def test_single_low_confidence_field_forces_review():
    invoice = make_invoice(tax_confidence=0.3)
    decision = confidence_service.route(invoice, threshold=0.75)
    assert decision.status == ReviewStatus.pending_review
    assert "tax" in decision.flagged_fields


def test_non_reconciling_invoice_is_flagged_even_if_all_fields_confident():
    invoice = make_invoice(total="999.00")  # every field confidence still high
    decision = confidence_service.route(invoice, threshold=0.75)
    assert decision.status == ReviewStatus.pending_review
    assert "__reconciliation__" in decision.flagged_fields


def test_threshold_is_configurable_per_call():
    invoice = make_invoice(tax_confidence=0.6)
    lenient = confidence_service.route(invoice, threshold=0.5)
    strict = confidence_service.route(invoice, threshold=0.8)
    assert lenient.status == ReviewStatus.auto_approved
    assert strict.status == ReviewStatus.pending_review


def test_line_item_confidence_participates_in_routing():
    invoice = make_invoice()
    invoice.line_items[0].confidence = 0.2
    decision = confidence_service.route(invoice, threshold=0.75)
    assert decision.status == ReviewStatus.pending_review
    assert any(key.startswith("line_item[0]") for key in decision.flagged_fields)
