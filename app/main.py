"""
LedgerLens API.

Three endpoints do the real work:
  POST /ingest   - upload an image, run it through the full pipeline
  GET  /review   - list documents currently waiting on a human
  POST /approve  - commit a reviewer's corrections and close the record

Everything else (/health, /metrics) is plumbing for the demo and the
Grafana dashboard.
"""

import io
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import UUID

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import DocumentRecord, get_session, init_db
from app.metrics import (
    documents_auto_approved_total,
    documents_blocked_total,
    documents_extraction_failed_total,
    documents_flagged_total,
    documents_ingested_total,
    estimate_cost_usd,
    extraction_latency_seconds,
    moderation_latency_seconds,
    review_queue_size,
    token_cost_usd_total,
)
from app.models.schemas import (
    ApproveRequest,
    IngestResponse,
    InvoiceSchema,
    ModerationVerdict,
    ReviewQueueItem,
    ReviewStatus,
)
from app.services import confidence as confidence_service
from app.services import extraction as extraction_service
from app.services import moderation as moderation_service
from app.services import redaction as redaction_service
from app.services import watermark as watermark_service
from app.services.extraction import (
    ExtractionError,
    ExtractionRateLimitedError,
    ExtractionServiceError,
    ExtractionTimeoutError,
)
from app.services.moderation import (
    ModerationError,
    ModerationRateLimitedError,
    ModerationServiceError,
    ModerationTimeoutError,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ledgerlens.api")

@asynccontextmanager
async def _lifespan(app: FastAPI):
    init_db()
    _sync_review_queue_gauge()
    yield


def _sync_review_queue_gauge() -> None:
    """review_queue_size is a Gauge, not a delta counter someone has to keep
    balanced - it's always set to the actual count of pending_review rows in
    the DB. Anything else (inc/dec on ingest/approve) drifts the moment the
    process restarts, since the DB is persistent but the in-memory gauge
    isn't: a restart zeroes the gauge while real pending documents are still
    sitting there, and an unconditional dec() on approve can walk it
    negative. Called on startup and after every ingest/approve so it can
    never be stale for longer than one request."""
    session = get_session()
    try:
        count = (
            session.query(DocumentRecord)
            .filter(DocumentRecord.status == ReviewStatus.pending_review.value)
            .count()
        )
        review_queue_size.set(count)
    finally:
        session.close()


app = FastAPI(
    title="LedgerLens",
    description="Confidence-aware document intelligence for receipts and invoices.",
    version="0.1.0",
    lifespan=_lifespan,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def db_session():
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def _resize_if_needed(image_bytes: bytes) -> bytes:
    """Cost control: a phone photo can easily be 4000px+ on the long edge,
    which is more resolution than the model needs and just burns tokens."""
    settings = get_settings()
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="uploaded file is not a readable image") from exc
    longest = max(img.size)
    if longest <= settings.max_image_dimension_px:
        return image_bytes

    ratio = settings.max_image_dimension_px / longest
    new_size = (int(img.width * ratio), int(img.height * ratio))
    img = img.resize(new_size, Image.LANCZOS)

    out = io.BytesIO()
    img.convert("RGB").save(out, format="PNG")
    return out.getvalue()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...), session: Session = Depends(db_session)):
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="empty upload")

    # --- stage 1: moderation gate, before anything else touches the LLM ---
    t0 = time.perf_counter()
    try:
        mod_result = moderation_service.screen_image(raw_bytes)
    except ModerationRateLimitedError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ModerationTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except ModerationError as exc:
        logger.error("moderation call failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        moderation_latency_seconds.observe(time.perf_counter() - t0)

    if mod_result.verdict == ModerationVerdict.block:
        documents_blocked_total.inc()
        logger.warning("blocked upload: %s", redaction_service.redact(mod_result.reason or ""))
        raise HTTPException(status_code=422, detail={"blocked_reason": mod_result.reason})

    # --- stage 2: preprocess ---
    resized = _resize_if_needed(raw_bytes)

    # --- stage 3: vision extraction ---
    t0 = time.perf_counter()
    try:
        extraction_result = extraction_service.extract_invoice(resized)
    except ExtractionRateLimitedError as exc:
        documents_extraction_failed_total.inc()
        raise HTTPException(status_code=429, detail={"extraction_failed_reason": str(exc)}) from exc
    except ExtractionTimeoutError as exc:
        documents_extraction_failed_total.inc()
        raise HTTPException(status_code=504, detail={"extraction_failed_reason": str(exc)}) from exc
    except ExtractionServiceError as exc:
        documents_extraction_failed_total.inc()
        logger.error("extraction service failed: %s", exc)
        raise HTTPException(status_code=502, detail={"extraction_failed_reason": str(exc)}) from exc
    except ExtractionError as exc:
        documents_extraction_failed_total.inc()
        logger.info("extraction failed: %s", exc)
        raise HTTPException(status_code=422, detail={"extraction_failed_reason": str(exc)}) from exc
    finally:
        extraction_latency_seconds.observe(time.perf_counter() - t0)

    cost = estimate_cost_usd(extraction_result.prompt_tokens, extraction_result.completion_tokens)
    token_cost_usd_total.inc(cost)

    invoice = extraction_result.invoice

    # --- stage 4: confidence routing ---
    decision = confidence_service.route(invoice)

    # --- stage 5: watermark + persist ---
    watermarked = watermark_service.apply_watermark(resized, document_id=file.filename or "unknown")

    record = DocumentRecord(
        filename=file.filename or "upload.png",
        status=decision.status.value,
        overall_confidence=decision.overall_confidence,
        flagged_fields_json=json.dumps(decision.flagged_fields),
        extracted_json=invoice.model_dump_json(),
        image_path=None,  # kept in memory / local disk for the demo; see BUILD_NOTE.md
        created_at=_utcnow(),
    )
    session.add(record)
    session.commit()
    session.refresh(record)

    settings = get_settings()
    from pathlib import Path

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    image_path = upload_dir / f"{record.id}.png"
    image_path.write_bytes(watermarked)
    record.image_path = str(image_path)
    session.commit()

    documents_ingested_total.inc()
    if decision.status == ReviewStatus.auto_approved:
        documents_auto_approved_total.inc()
    else:
        documents_flagged_total.inc()
        _sync_review_queue_gauge()

    logger.info(
        "ingested doc_id=%s status=%s confidence=%.2f flagged=%s",
        record.id,
        decision.status.value,
        decision.overall_confidence,
        decision.flagged_fields,
    )

    return IngestResponse(
        document_id=UUID(record.id),
        status=decision.status,
        overall_confidence=decision.overall_confidence,
        flagged_fields=decision.flagged_fields,
        moderation=mod_result,
    )


@app.get("/review", response_model=list[ReviewQueueItem])
def list_review_queue(session: Session = Depends(db_session)):
    rows = (
        session.query(DocumentRecord)
        .filter(DocumentRecord.status == ReviewStatus.pending_review.value)
        .order_by(DocumentRecord.created_at.asc())
        .all()
    )
    items = []
    for row in rows:
        items.append(
            ReviewQueueItem(
                document_id=UUID(row.id),
                filename=row.filename,
                status=ReviewStatus(row.status),
                flagged_fields=row.flagged_fields(),
                extracted=InvoiceSchema.model_validate_json(row.extracted_json),
                created_at=row.created_at.isoformat(),
            )
        )
    return items


@app.post("/approve")
def approve(request: ApproveRequest, session: Session = Depends(db_session)):
    row = session.query(DocumentRecord).filter(DocumentRecord.id == str(request.document_id)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="document not found")

    invoice = InvoiceSchema.model_validate_json(row.extracted_json)

    # Apply the reviewer's corrections onto a plain dict copy - we don't
    # touch the original extracted_json so there's still a record of what
    # the model actually said versus what a human confirmed.
    reviewed = invoice.model_dump(mode="json")
    for field_name, corrected_value in request.corrected_fields.items():
        if field_name in reviewed:
            reviewed[field_name] = corrected_value

    row.reviewed_json = json.dumps(reviewed)
    row.status = ReviewStatus.approved.value
    row.reviewer = request.reviewer
    row.reviewed_at = _utcnow()
    session.commit()

    _sync_review_queue_gauge()

    logger.info("approved doc_id=%s reviewer=%s", row.id, request.reviewer)
    return {"document_id": str(row.id), "status": row.status}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
