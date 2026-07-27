from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, RateLimitError

from app.models.schemas import ModerationVerdict
from app.services import moderation as moderation_service
from app.services.moderation import (
    ModerationRateLimitedError,
    ModerationServiceError,
    ModerationTimeoutError,
)


def _fake_moderation_response(scores: dict[str, float]):
    """Builds a stand-in for the OpenAI moderation response shape, just
    deep enough for screen_image() to read what it needs."""
    category_scores = SimpleNamespace(model_dump=lambda: scores)
    result = SimpleNamespace(category_scores=category_scores)
    return SimpleNamespace(results=[result])


@pytest.fixture
def all_clear_scores():
    return {"sexual": 0.01, "violence": 0.0, "self-harm": 0.0, "sexual/minors": 0.0, "harassment": 0.02}


@pytest.fixture
def flagged_scores():
    return {"sexual": 0.02, "violence": 0.87, "self-harm": 0.01, "sexual/minors": 0.0, "harassment": 0.1}


def _patch_client(monkeypatch, response):
    fake_client = MagicMock()
    fake_client.moderations.create.return_value = response
    monkeypatch.setattr(moderation_service, "_client", lambda: fake_client)
    return fake_client


def test_clean_image_is_allowed(monkeypatch, tiny_png_bytes, all_clear_scores):
    _patch_client(monkeypatch, _fake_moderation_response(all_clear_scores))
    result = moderation_service.screen_image(tiny_png_bytes)
    assert result.verdict == ModerationVerdict.allow
    assert result.reason is None


def test_flagged_image_is_blocked_and_reason_named(monkeypatch, tiny_png_bytes, flagged_scores):
    _patch_client(monkeypatch, _fake_moderation_response(flagged_scores))
    result = moderation_service.screen_image(tiny_png_bytes)
    assert result.verdict == ModerationVerdict.block
    assert "violence" in result.reason


def test_extraction_call_is_never_made_when_blocked(monkeypatch, tiny_png_bytes, flagged_scores):
    """The whole point of running moderation first is to never spend an
    extraction call on a blocked image - this asserts that boundary holds
    even at the unit level, not just by pipeline ordering in main.py."""
    fake_client = _patch_client(monkeypatch, _fake_moderation_response(flagged_scores))
    moderation_service.screen_image(tiny_png_bytes)
    fake_client.beta.chat.completions.parse.assert_not_called()


def test_rate_limit_raises_clear_moderation_error(monkeypatch, tiny_png_bytes):
    request = httpx.Request("POST", "https://api.openai.com/v1/moderations")
    response = httpx.Response(status_code=429, request=request)
    fake_client = MagicMock()
    fake_client.moderations.create.side_effect = RateLimitError("rate limited", response=response, body=None)
    monkeypatch.setattr(moderation_service, "_client", lambda: fake_client)

    with pytest.raises(ModerationRateLimitedError):
        moderation_service.screen_image(tiny_png_bytes)


def test_timeout_raises_clear_moderation_error(monkeypatch, tiny_png_bytes):
    request = httpx.Request("POST", "https://api.openai.com/v1/moderations")
    fake_client = MagicMock()
    fake_client.moderations.create.side_effect = APITimeoutError(request=request)
    monkeypatch.setattr(moderation_service, "_client", lambda: fake_client)

    with pytest.raises(ModerationTimeoutError):
        moderation_service.screen_image(tiny_png_bytes)


def test_connection_failure_raises_clear_moderation_error(monkeypatch, tiny_png_bytes):
    request = httpx.Request("POST", "https://api.openai.com/v1/moderations")
    fake_client = MagicMock()
    fake_client.moderations.create.side_effect = APIConnectionError(request=request)
    monkeypatch.setattr(moderation_service, "_client", lambda: fake_client)

    with pytest.raises(ModerationServiceError):
        moderation_service.screen_image(tiny_png_bytes)


def test_irrelevant_low_scores_do_not_trigger_block(monkeypatch, tiny_png_bytes):
    """A receipt photo can trip unrelated low-confidence categories (e.g.
    a barcode misread as something else); only the watched categories
    above threshold should ever block."""
    scores = {"sexual": 0.4, "violence": 0.4, "self-harm": 0.4, "sexual/minors": 0.4, "harassment": 0.9}
    _patch_client(monkeypatch, _fake_moderation_response(scores))
    result = moderation_service.screen_image(tiny_png_bytes)
    assert result.verdict == ModerationVerdict.allow
