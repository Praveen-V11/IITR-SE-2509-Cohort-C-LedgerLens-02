"""
Confidence routing.

This is the piece that keeps the pipeline honest: we never auto-accept an
extraction just because the model returned *something*. Each field's
confidence is checked against a threshold, and the record as a whole is
only auto-approved if every field clears the bar AND the numbers
reconcile (line items + tax ~= total). One low-confidence field is enough
to send the whole document to the review queue - partial auto-approval
of a single invoice would just move the trust problem into the database
instead of solving it.
"""

from dataclasses import dataclass

from app.config import get_settings
from app.models.schemas import InvoiceSchema, ReviewStatus


@dataclass
class RoutingDecision:
    status: ReviewStatus
    flagged_fields: list[str]
    overall_confidence: float
    reconciliation_ok: bool


def route(extracted: InvoiceSchema, threshold: float | None = None) -> RoutingDecision:
    settings = get_settings()
    cutoff = threshold if threshold is not None else settings.review_threshold

    flagged = [name for name, score in extracted.field_confidences().items() if score < cutoff]

    reconciles = extracted.reconciles()
    if not reconciles:
        # A document can have every individual field above threshold and
        # still not add up - that's exactly the kind of thing a human
        # should glance at, so treat it the same as a low-confidence flag.
        flagged.append("__reconciliation__")

    status = ReviewStatus.pending_review if flagged else ReviewStatus.auto_approved

    return RoutingDecision(
        status=status,
        flagged_fields=flagged,
        overall_confidence=extracted.overall_confidence(),
        reconciliation_ok=reconciles,
    )
