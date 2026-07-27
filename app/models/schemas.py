"""
Pydantic contracts for LedgerLens.

InvoiceSchema is the shape we force GPT-4o's structured output into via
`response_format`. Every extracted value is paired with a confidence float
so the routing layer (services/confidence.py) has something concrete to
threshold against instead of trusting the model blindly.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

# Despite the extraction prompt asking for plain numbers, the vision model
# occasionally writes a monetary value back with its currency symbol and/or
# thousands separators still attached (e.g. "₹23,134.75") - stripping
# everything but digits/./- and rounding to cents lets that parse cleanly
# instead of failing schema validation over formatting the model itself put
# there. Non-numeric-looking input passes through untouched so Pydantic's
# own Decimal coercion still raises its normal, honest error on real garbage.
_MONEY_STRIP_RE = re.compile(r"[^\d.\-]")


def _coerce_money(value: object) -> object:
    if not isinstance(value, str):
        return value
    cleaned = _MONEY_STRIP_RE.sub("", value.strip())
    if not cleaned or cleaned in {"-", "."}:
        return value
    try:
        amount = Decimal(cleaned)
    except InvalidOperation:
        return value
    return amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


class DocumentType(str, Enum):
    invoice = "invoice"
    receipt = "receipt"


class ReviewStatus(str, Enum):
    auto_approved = "auto_approved"
    pending_review = "pending_review"
    approved = "approved"
    rejected = "rejected"


class ScoredField(BaseModel):
    """A single extracted value plus how sure the model was about it."""
    value: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class LineItem(BaseModel):
    description: str
    quantity: float = Field(gt=0)
    unit_price: Decimal
    amount: Decimal
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("unit_price", "amount", mode="before")
    @classmethod
    def clean_money(cls, v: object) -> object:
        return _coerce_money(v)

    @field_validator("unit_price", "amount")
    @classmethod
    def no_negative_money(cls, v: Decimal) -> Decimal:
        if v < 0:
            raise ValueError("monetary fields cannot be negative")
        return v


class InvoiceSchema(BaseModel):
    """
    The structured-output contract passed as `response_format` to the
    vision model. Header fields carry their own confidence score;
    line items carry theirs individually since a nine-row invoice usually
    has one or two rows the model is genuinely unsure about, not all nine.
    """
    document_type: DocumentType
    vendor_name: str
    vendor_confidence: float = Field(ge=0.0, le=1.0)

    invoice_number: Optional[str] = None
    invoice_number_confidence: float = Field(ge=0.0, le=1.0, default=0.0)

    document_date: Optional[date] = None
    date_confidence: float = Field(ge=0.0, le=1.0, default=0.0)

    currency: str = Field(default="USD", max_length=8)

    line_items: list[LineItem] = Field(default_factory=list)

    subtotal: Decimal = Field(default=Decimal("0"))
    subtotal_confidence: float = Field(ge=0.0, le=1.0, default=0.0)

    tax: Decimal = Field(default=Decimal("0"))
    tax_confidence: float = Field(ge=0.0, le=1.0, default=0.0)

    total: Decimal = Field(default=Decimal("0"))
    total_confidence: float = Field(ge=0.0, le=1.0, default=0.0)

    @field_validator("subtotal", "tax", "total", mode="before")
    @classmethod
    def clean_money(cls, v: object) -> object:
        return _coerce_money(v)

    def field_confidences(self) -> dict[str, float]:
        """Flat map of field name -> confidence, used by the router."""
        scores = {
            "vendor_name": self.vendor_confidence,
            "invoice_number": self.invoice_number_confidence,
            "document_date": self.date_confidence,
            "subtotal": self.subtotal_confidence,
            "tax": self.tax_confidence,
            "total": self.total_confidence,
        }
        for idx, item in enumerate(self.line_items):
            scores[f"line_item[{idx}].{item.description[:20]}"] = item.confidence
        return scores

    def overall_confidence(self) -> float:
        scores = list(self.field_confidences().values())
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    def reconciles(self, tolerance: Decimal = Decimal("1.00")) -> bool:
        """Cheap arithmetic sanity check: do the line items + tax roughly equal total?"""
        computed = sum((li.amount for li in self.line_items), Decimal("0"))
        if self.subtotal:
            computed = self.subtotal
        return abs((computed + self.tax) - self.total) <= tolerance


class ModerationVerdict(str, Enum):
    allow = "allow"
    block = "block"


class ModerationResult(BaseModel):
    verdict: ModerationVerdict
    reason: Optional[str] = None
    raw_scores: dict[str, float] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: UUID = Field(default_factory=uuid4)
    status: ReviewStatus
    overall_confidence: float
    flagged_fields: list[str] = Field(default_factory=list)
    moderation: ModerationResult


class ReviewQueueItem(BaseModel):
    document_id: UUID
    filename: str
    status: ReviewStatus
    flagged_fields: list[str]
    extracted: InvoiceSchema
    created_at: str


class ApproveRequest(BaseModel):
    document_id: UUID
    corrected_fields: dict[str, str] = Field(
        default_factory=dict,
        description="field_name -> corrected value, only for fields the reviewer actually touched",
    )
    reviewer: str = Field(default="unknown", description="who approved this, for the audit trail")
