"""
PII redaction. Applied to any text right before it hits a log line -
never applied to the data we actually store, since the review workflow
needs the real values. This is a log-hygiene control, not a data-masking
feature.
"""

import re

_PATTERNS: dict[str, re.Pattern] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[\w.\-]+@[\w\-]+\.[a-zA-Z]{2,}\b"),
    "phone": re.compile(r"(?:\+?\d{1,2}[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b"),
    "card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}


def redact(text: str) -> str:
    if not text:
        return text
    redacted = text
    for label, pattern in _PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED_{label.upper()}]", redacted)
    return redacted


def redact_for_log(payload: dict) -> str:
    """Convenience wrapper: stringify a dict and redact it in one call, for logger.info(...) sites."""
    import json

    return redact(json.dumps(payload, default=str))
