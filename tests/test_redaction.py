from app.services import redaction


def test_ssn_is_redacted():
    out = redaction.redact("borrower ssn is 123-45-6789 on file")
    assert "123-45-6789" not in out
    assert "[REDACTED_SSN]" in out


def test_email_is_redacted():
    out = redaction.redact("contact vendor at billing@acme-supplies.com for questions")
    assert "billing@acme-supplies.com" not in out
    assert "[REDACTED_EMAIL]" in out


def test_phone_is_redacted():
    out = redaction.redact("call us at (415) 555-0199 anytime")
    assert "555-0199" not in out
    assert "[REDACTED_PHONE]" in out


def test_clean_text_is_left_untouched():
    text = "Acme Supplies invoice INV-1001 dated 2026-06-01 total 38.50"
    assert redaction.redact(text) == text


def test_redact_for_log_handles_a_dict_payload():
    payload = {"vendor": "Acme", "note": "reach me at jane.doe@example.com"}
    out = redaction.redact_for_log(payload)
    assert "jane.doe@example.com" not in out
    assert "[REDACTED_EMAIL]" in out
