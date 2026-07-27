"""
Moderation gate. Runs before the (more expensive) vision extraction call
so a flagged image never reaches GPT-4o at all - saves cost and keeps a
clean boundary between "this is unsafe" and "this is safe but low
confidence" (the latter is what the review queue is for, not this).
"""

import base64
import logging

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from app.config import get_settings
from app.models.schemas import ModerationResult, ModerationVerdict

logger = logging.getLogger("ledgerlens.moderation")


class ModerationError(Exception):
    """Raised when the moderation call itself fails - this is about the
    OpenAI request breaking, not about the image being flagged unsafe (that
    case is a normal `ModerationResult` with verdict=block, not an
    exception). Subclasses below narrow down the failure mode; catch those
    first if you need to tell them apart."""


class ModerationRateLimitedError(ModerationError):
    """OpenAI rejected the call for exceeding a rate/quota limit."""


class ModerationTimeoutError(ModerationError):
    """The call to OpenAI didn't come back in time."""


class ModerationServiceError(ModerationError):
    """OpenAI's API itself failed (connection drop, 5xx, auth, etc.)."""

# Category scores we actually care about for a document-upload use case.
# Deliberately narrower than "block on any nonzero score" - a receipt photo
# can trip low-confidence categories that have nothing to do with the
# actual risk we're screening for.
WATCHED_CATEGORIES = ("sexual", "violence", "self-harm", "sexual/minors")


def _client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key)


def screen_image(image_bytes: bytes, mime_type: str = "image/png") -> ModerationResult:
    """
    Sends the raw image bytes to the Moderation API and decides allow/block
    based on the category scores that matter for this use case.
    """
    settings = get_settings()
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{b64}"

    client = _client()
    try:
        response = client.moderations.create(
            model=settings.moderation_model,
            input=[{"type": "image_url", "image_url": {"url": data_uri}}],
        )
    except RateLimitError as exc:
        # Order matters: RateLimitError/APITimeoutError are both APIError
        # subclasses, so they must be caught before the generic APIError.
        logger.warning("moderation rate-limited by OpenAI: %s", exc)
        raise ModerationRateLimitedError("moderation service is rate-limited, try again shortly") from exc
    except APITimeoutError as exc:
        logger.warning("moderation call timed out: %s", exc)
        raise ModerationTimeoutError("moderation service timed out, try again") from exc
    except APIConnectionError as exc:
        logger.error("moderation call could not reach OpenAI: %s", exc)
        raise ModerationServiceError("could not reach the moderation service") from exc
    except APIError as exc:
        logger.error("moderation API call failed: %s", exc)
        raise ModerationServiceError(f"moderation service error: {exc}") from exc

    result = response.results[0]
    scores = {cat: float(score) for cat, score in result.category_scores.model_dump().items()}

    worst_watched = max((scores.get(cat, 0.0) for cat in WATCHED_CATEGORIES), default=0.0)

    if worst_watched >= settings.moderation_block_threshold:
        triggered = [c for c in WATCHED_CATEGORIES if scores.get(c, 0.0) >= settings.moderation_block_threshold]
        logger.warning("moderation blocked upload, categories=%s", triggered)
        return ModerationResult(
            verdict=ModerationVerdict.block,
            reason=f"flagged categories: {', '.join(triggered)}",
            raw_scores=scores,
        )

    return ModerationResult(verdict=ModerationVerdict.allow, raw_scores=scores)
