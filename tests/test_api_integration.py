from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.models.schemas import ModerationResult, ModerationVerdict
from app.services.extraction import (
    ExtractionError,
    ExtractionRateLimitedError,
    ExtractionServiceError,
    ExtractionTimeoutError,
)
from app.services.moderation import (
    ModerationRateLimitedError,
    ModerationServiceError,
    ModerationTimeoutError,
)
from tests.conftest import make_invoice


@pytest.fixture
def client(monkeypatch, tiny_png_bytes):
    # Imported here, not at module scope, so conftest's env vars are
    # already in place before app.main (and therefore app.db) gets loaded.
    import app.main as main_module

    fake_extraction = MagicMock()
    fake_extraction.invoice = make_invoice()
    fake_extraction.prompt_tokens = 1200
    fake_extraction.completion_tokens = 300
    monkeypatch.setattr(main_module.extraction_service, "extract_invoice", lambda *a, **k: fake_extraction)

    monkeypatch.setattr(
        main_module.moderation_service,
        "screen_image",
        lambda *a, **k: ModerationResult(verdict=ModerationVerdict.allow, raw_scores={}),
    )

    # Context-manager form so the startup event (init_db) actually fires -
    # a bare TestClient(app) never triggers lifespan/startup handlers.
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ingest_auto_approves_a_clean_high_confidence_document(client, tiny_png_bytes):
    resp = client.post("/ingest", files={"file": ("receipt.png", tiny_png_bytes, "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "auto_approved"
    assert body["flagged_fields"] == []


def test_ingest_blocked_document_never_reaches_review_queue(client, tiny_png_bytes, monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(
        main_module.moderation_service,
        "screen_image",
        lambda *a, **k: ModerationResult(verdict=ModerationVerdict.block, reason="flagged categories: violence"),
    )
    resp = client.post("/ingest", files={"file": ("bad.png", tiny_png_bytes, "image/png")})
    assert resp.status_code == 422
    assert "violence" in resp.json()["detail"]["blocked_reason"]


def test_low_confidence_document_lands_in_review_queue_then_can_be_approved(client, tiny_png_bytes, monkeypatch):
    import app.main as main_module

    flagged = make_invoice(tax_confidence=0.2)
    fake_extraction = MagicMock()
    fake_extraction.invoice = flagged
    fake_extraction.prompt_tokens = 900
    fake_extraction.completion_tokens = 250
    monkeypatch.setattr(main_module.extraction_service, "extract_invoice", lambda *a, **k: fake_extraction)

    ingest_resp = client.post("/ingest", files={"file": ("flagged.png", tiny_png_bytes, "image/png")})
    assert ingest_resp.status_code == 200
    body = ingest_resp.json()
    assert body["status"] == "pending_review"
    assert "tax" in body["flagged_fields"]

    review_resp = client.get("/review")
    assert review_resp.status_code == 200
    queue = review_resp.json()
    assert any(item["document_id"] == body["document_id"] for item in queue)

    approve_resp = client.post(
        "/approve",
        json={
            "document_id": body["document_id"],
            "corrected_fields": {"tax": "3.50"},
            "reviewer": "praveen",
        },
    )
    assert approve_resp.status_code == 200
    assert approve_resp.json()["status"] == "approved"

    review_resp_after = client.get("/review")
    remaining_ids = [item["document_id"] for item in review_resp_after.json()]
    assert body["document_id"] not in remaining_ids


def test_ingest_non_invoice_image_returns_422_not_500(client, tiny_png_bytes, monkeypatch):
    """A blank/non-invoice photo makes the vision model unable to satisfy
    the schema. This used to raise ValueError straight out of the endpoint
    and surface as an unhandled 500 - it must now come back as a clean 422."""
    import app.main as main_module

    def _raise(*args, **kwargs):
        raise ExtractionError("document could not be read as an invoice or receipt")

    monkeypatch.setattr(main_module.extraction_service, "extract_invoice", _raise)

    resp = client.post("/ingest", files={"file": ("not_an_invoice.png", tiny_png_bytes, "image/png")})
    assert resp.status_code == 422
    assert "extraction_failed_reason" in resp.json()["detail"]


def test_ingest_rejects_unreadable_file_with_400(client):
    resp = client.post("/ingest", files={"file": ("bad.png", b"not an image at all", "image/png")})
    assert resp.status_code == 400


def test_ingest_rejects_empty_upload_with_400(client):
    resp = client.post("/ingest", files={"file": ("empty.png", b"", "image/png")})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "empty upload"


@pytest.mark.parametrize(
    "exc_cls, expected_status",
    [
        (ExtractionRateLimitedError, 429),
        (ExtractionTimeoutError, 504),
        (ExtractionServiceError, 502),
    ],
)
def test_ingest_maps_extraction_upstream_failures_to_clear_status_codes(
    client, tiny_png_bytes, monkeypatch, exc_cls, expected_status
):
    import app.main as main_module

    def _raise(*args, **kwargs):
        raise exc_cls("upstream trouble")

    monkeypatch.setattr(main_module.extraction_service, "extract_invoice", _raise)

    resp = client.post("/ingest", files={"file": ("receipt.png", tiny_png_bytes, "image/png")})
    assert resp.status_code == expected_status
    assert "extraction_failed_reason" in resp.json()["detail"]


@pytest.mark.parametrize(
    "exc_cls, expected_status",
    [
        (ModerationRateLimitedError, 429),
        (ModerationTimeoutError, 504),
        (ModerationServiceError, 502),
    ],
)
def test_ingest_maps_moderation_upstream_failures_to_clear_status_codes(
    client, tiny_png_bytes, monkeypatch, exc_cls, expected_status
):
    import app.main as main_module

    def _raise(*args, **kwargs):
        raise exc_cls("upstream trouble")

    monkeypatch.setattr(main_module.moderation_service, "screen_image", _raise)

    resp = client.post("/ingest", files={"file": ("receipt.png", tiny_png_bytes, "image/png")})
    assert resp.status_code == expected_status
    assert resp.json()["detail"]


def test_metrics_endpoint_exposes_prometheus_text_format(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "ledgerlens_documents_ingested_total" in resp.text


def _review_queue_gauge_value(client):
    from app.metrics import review_queue_size

    return next(iter(review_queue_size.collect())).samples[0].value


def test_review_queue_gauge_tracks_db_state_through_ingest_and_approve(client, tiny_png_bytes, monkeypatch):
    """review_queue_size must always match the real count of pending_review
    rows in the DB, not a hand-maintained inc/dec delta - a delta drifts
    the moment the process restarts (in-memory gauge resets to 0 while the
    DB still has real pending documents) or an approve() fires for a
    document that was never actually counted."""
    import app.main as main_module

    flagged = make_invoice(tax_confidence=0.2)
    fake_extraction = MagicMock()
    fake_extraction.invoice = flagged
    fake_extraction.prompt_tokens = 900
    fake_extraction.completion_tokens = 250
    monkeypatch.setattr(main_module.extraction_service, "extract_invoice", lambda *a, **k: fake_extraction)

    ingest_resp = client.post("/ingest", files={"file": ("flagged.png", tiny_png_bytes, "image/png")})
    assert ingest_resp.status_code == 200
    assert _review_queue_gauge_value(client) == 1

    doc_id = ingest_resp.json()["document_id"]
    approve_resp = client.post(
        "/approve",
        json={"document_id": doc_id, "corrected_fields": {}, "reviewer": "praveen"},
    )
    assert approve_resp.status_code == 200
    assert _review_queue_gauge_value(client) == 0


def test_review_queue_gauge_self_heals_from_stale_in_memory_state(client, tiny_png_bytes, monkeypatch):
    """Simulates the drift a process restart used to cause: the gauge is
    manually knocked out of sync with the DB, then the next request must
    resync it from ground truth rather than trusting the stale value."""
    import app.main as main_module
    from app.metrics import review_queue_size

    review_queue_size.set(999)  # pretend the in-memory gauge is stale/wrong

    flagged = make_invoice(tax_confidence=0.2)
    fake_extraction = MagicMock()
    fake_extraction.invoice = flagged
    fake_extraction.prompt_tokens = 900
    fake_extraction.completion_tokens = 250
    monkeypatch.setattr(main_module.extraction_service, "extract_invoice", lambda *a, **k: fake_extraction)

    resp = client.post("/ingest", files={"file": ("flagged.png", tiny_png_bytes, "image/png")})
    assert resp.status_code == 200

    queue = client.get("/review").json()
    assert _review_queue_gauge_value(client) == len(queue)
