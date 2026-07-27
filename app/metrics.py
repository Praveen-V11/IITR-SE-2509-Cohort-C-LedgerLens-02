"""
Prometheus metrics wiring. Four things the brief specifically asks the
Grafana dashboard to show: per-document cost, moderation latency,
extraction latency, and the auto-approval rate (derived from the two
counters below, not tracked directly).
"""

from prometheus_client import Counter, Gauge, Histogram

moderation_latency_seconds = Histogram(
    "ledgerlens_moderation_latency_seconds",
    "Time spent in the moderation gate",
)

extraction_latency_seconds = Histogram(
    "ledgerlens_extraction_latency_seconds",
    "Time spent in the GPT-4o vision extraction call",
)

token_cost_usd_total = Counter(
    "ledgerlens_token_cost_usd_total",
    "Cumulative estimated USD cost of extraction calls",
)

documents_ingested_total = Counter(
    "ledgerlens_documents_ingested_total",
    "Documents that made it past moderation and were processed",
)

documents_auto_approved_total = Counter(
    "ledgerlens_documents_auto_approved_total",
    "Documents that cleared every confidence threshold with no review needed",
)

documents_flagged_total = Counter(
    "ledgerlens_documents_flagged_total",
    "Documents routed to the human review queue",
)

documents_blocked_total = Counter(
    "ledgerlens_documents_blocked_total",
    "Uploads rejected at the moderation gate",
)

documents_extraction_failed_total = Counter(
    "ledgerlens_documents_extraction_failed_total",
    "Uploads that passed moderation but couldn't be extracted as an invoice/receipt",
)

review_queue_size = Gauge(
    "ledgerlens_review_queue_size",
    "Current number of documents pending human review",
)


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    from app.config import get_settings

    settings = get_settings()
    cost = (
        (prompt_tokens / 1000.0) * settings.cost_per_1k_input_tokens
        + (completion_tokens / 1000.0) * settings.cost_per_1k_output_tokens
    )
    return round(cost, 6)
