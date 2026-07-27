"""
Vision extraction. One GPT-4o call, structured output enforced via
response_format=InvoiceSchema, so we never hand-parse a JSON blob out of
free text - if the model can't satisfy the schema, the SDK raises rather
than letting a malformed record slip into the pipeline.
"""

import base64
import logging
from dataclasses import dataclass

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
    RateLimitError,
    OpenAI,
)
from pydantic import ValidationError

from app.config import get_settings
from app.models.schemas import InvoiceSchema

logger = logging.getLogger("ledgerlens.extraction")


class ExtractionError(Exception):
    """Raised whenever the vision model couldn't produce a usable
    InvoiceSchema for the uploaded image - e.g. it isn't an invoice/receipt
    at all, or is blank/unreadable. Kept distinct from ValueError so callers
    can catch it specifically without swallowing unrelated bugs. Base class
    for the more specific upstream-failure variants below - catch those
    first if you need to tell them apart, since they're all ExtractionError
    subclasses."""


class ExtractionRateLimitedError(ExtractionError):
    """OpenAI rejected the call for exceeding a rate/quota limit - the
    document itself is fine, this is a "try again shortly" situation."""


class ExtractionTimeoutError(ExtractionError):
    """The call to OpenAI didn't come back in time."""


class ExtractionServiceError(ExtractionError):
    """OpenAI's API itself failed (connection drop, 5xx, auth, etc.) for
    reasons unrelated to the uploaded document."""


@dataclass
class ExtractionResult:
    invoice: InvoiceSchema
    prompt_tokens: int
    completion_tokens: int

EXTRACTION_PROMPT = """\
You are reading a photo of a receipt or invoice. Extract every field you can
see into the given schema.

Rules:
- If a field is not visible or you are guessing, still fill it in but give it
  a LOW confidence score (below 0.5). Never omit a required field just
  because you're unsure - report your best guess and be honest about the
  confidence instead.
- confidence scores are your own calibrated certainty for that exact field,
  from 0.0 (pure guess) to 1.0 (printed clearly and unambiguously).
- line item amounts should be the value printed on that line, not a
  recalculation.
- currency should be a 3-letter code if you can tell (e.g. USD, INR, EUR);
  default to USD only if there is truly no signal either way.
- every monetary value (unit_price, amount, subtotal, tax, total) must be a
  plain decimal number with no currency symbol and no thousands separator -
  write 23134.75, never ₹23,134.75 or 23,134.75 or "23134.75 INR". Put the
  currency code in the `currency` field instead, not in any amount.
"""


def _client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key)


def extract_invoice(image_bytes: bytes, mime_type: str = "image/png") -> ExtractionResult:
    settings = get_settings()
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{b64}"

    client = _client()
    try:
        completion = client.beta.chat.completions.parse(
            model=settings.vision_model,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract this document into the schema."},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            response_format=InvoiceSchema,
        )
    except (LengthFinishReasonError, ContentFilterFinishReasonError) as exc:
        # The model started answering but couldn't finish in-schema - most
        # commonly because the image isn't actually an invoice/receipt (a
        # blank page, a screenshot, a photo of a cat), so it rambled instead
        # of filling the schema, or its content filter tripped mid-response.
        logger.info("extraction could not complete: %s", exc)
        raise ExtractionError("document could not be read as an invoice or receipt") from exc
    except RateLimitError as exc:
        # Order matters: RateLimitError is an APIStatusError which is an
        # APIError, so this must be caught before the generic APIError below.
        logger.warning("extraction rate-limited by OpenAI: %s", exc)
        raise ExtractionRateLimitedError("extraction service is rate-limited, try again shortly") from exc
    except APITimeoutError as exc:
        # APITimeoutError is an APIConnectionError is an APIError - same
        # ordering requirement as above.
        logger.warning("extraction call timed out: %s", exc)
        raise ExtractionTimeoutError("extraction service timed out, try again") from exc
    except APIConnectionError as exc:
        logger.error("extraction call could not reach OpenAI: %s", exc)
        raise ExtractionServiceError("could not reach the extraction service") from exc
    except APIError as exc:
        logger.error("extraction API call failed: %s", exc)
        raise ExtractionServiceError(f"extraction service error: {exc}") from exc
    except ValidationError as exc:
        # The SDK's own response_format parsing raises this directly out of
        # client.beta.chat.completions.parse() - it happens before we ever
        # see a `completion` object back, so it can't be caught below via
        # message.refusal/message.parsed. Most commonly the model wrote a
        # currency symbol or thousands separator into a Decimal field
        # despite the prompt rule above; treat it the same as any other
        # "couldn't produce a usable InvoiceSchema" case rather than letting
        # a raw pydantic error surface as a 500.
        logger.info("extraction produced a schema-invalid response: %s", exc)
        raise ExtractionError("document could not be read as an invoice or receipt") from exc

    message = completion.choices[0].message
    if message.refusal:
        logger.info("extraction refused by model: %s", message.refusal)
        raise ExtractionError(f"model declined to process this image: {message.refusal}")

    parsed = message.parsed
    if parsed is None:
        # Belt-and-braces - in practice the SDK raises before we get here if
        # the model couldn't satisfy the schema, but we don't want a silent
        # None flowing downstream if that behavior ever changes.
        raise ExtractionError("document could not be read as an invoice or receipt")

    usage = completion.usage
    logger.info(
        "extraction complete vendor=%s overall_confidence=%.2f",
        parsed.vendor_name,
        parsed.overall_confidence(),
    )
    return ExtractionResult(
        invoice=parsed,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )
