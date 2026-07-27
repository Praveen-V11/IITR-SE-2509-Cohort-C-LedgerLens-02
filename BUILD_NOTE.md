# Build Note

## What shipped (core scope, per the brief)

- Upload flow that runs every image through the OpenAI Moderation API
  (`omni-moderation-latest`) before any extraction call is made.
- GPT-4o vision extraction bound to a Pydantic schema via `response_format`
  - the model's output is either a valid `InvoiceSchema` or the call fails,
  there's no hand-rolled JSON parsing in between.
- Per-field confidence scoring on every header field and every line item,
  plus an arithmetic reconciliation check (line items + tax vs. total) as
  a second, independent signal that something's off even when every
  individual field looks confident.
- Threshold-based routing: a document is auto-approved only if *every*
  field clears the bar and the numbers reconcile. One weak field is
  enough to send the whole document to review - see "key decisions" below
  for why that's deliberate.
- A Streamlit reviewer UI (`st.data_editor`) that lists pending documents,
  lets a reviewer correct flagged fields inline, and commits the approval.
- Visible PIL watermarking (document id + UTC timestamp) burned into every
  archived image before it's written to disk.
- PII redaction (SSN/email/phone patterns) applied to anything that hits a
  log line - never applied to stored data, since the review workflow needs
  the real values.
- FastAPI service with `/ingest`, `/review`, `/approve`, `/health`,
  `/metrics`.
- Prometheus counters/histograms/gauges for moderation latency, extraction
  latency, cumulative token cost, ingested/auto-approved/flagged/blocked
  document counts, and review queue size - visualized in a pre-provisioned
  Grafana dashboard.
- Docker + docker-compose (api + streamlit + Prometheus + Grafana, one
  command).
- GitHub Actions CI: ruff lint → pytest (29 tests, all passing, no real
  API key required) → Docker build → optional Cloud Run deploy if repo
  secrets are configured.

## Core vs. stretch

Everything above is core scope from the brief. No stretch goals (visual
RAG search, voice summaries, LangGraph review workflow, streaming) were
attempted - the priority was a complete, reliable core pipeline over a
partially-built stretch feature.

## Key decisions

**All-or-nothing auto-approval, not per-field.** A document is only
auto-approved if every field clears the confidence threshold and the
totals reconcile. Auto-approving individual high-confidence fields while
flagging only the weak ones was considered, but it moves the trust problem
into the database instead of solving it - a partially-reviewed record is
harder to reason about later than a fully-reviewed one. The per-field
scores are still surfaced in the API response either way, so nothing is
lost.

**Reconciliation as its own signal.** A model can report high confidence
on every field and still produce numbers that don't add up (e.g. a
misread digit in the total). The reconciliation check treats that as a
distinct flag (`__reconciliation__`) rather than folding it into the
confidence average, so it can't be diluted by six other confident fields.

**OpenAI SDK over alternatives.** The brief's toolbox and suggested layers
specify the OpenAI SDK (GPT-4o + Moderation API) for this project, so that's
what's used throughout rather than substituting a different provider.

**Streamlit over a custom frontend.** Faster to build solo, and
`st.data_editor` maps directly onto the "reviewer corrects a flagged row"
requirement without custom component work.

**SQLite over Postgres for this scope.** Simpler local + docker-compose
story. The trade-off (see below) is documented rather than hidden.

## Known limitations

- **Cloud Run's local filesystem is ephemeral.** If deployed there as-is,
  the SQLite database and everything under `uploads/` reset on cold start
  or scale-to-zero. The fix is a GCS bucket for images and Cloud SQL (or
  a persisted SQLite file) for the database - not implemented here because
  it's infrastructure work orthogonal to the document-intelligence problem
  the brief is actually testing. The documented fallback is the local
  `docker compose up` path, which persists correctly via named volumes.
- **Single extraction call per document, no retry/backoff.** A transient
  OpenAI API error currently surfaces as a 500 rather than being retried.
  Fine for a demo; a production version would wrap the extraction call in
  a retry policy.
- **Confidence threshold is a single global value**, not per-document-type
  or per-field-type. The brief doesn't require per-field thresholds, and a
  single configurable value (`LEDGERLENS_REVIEW_THRESHOLD`) keeps the
  routing logic easy to reason about and test.
- **No batch upload endpoint.** Documents are processed one at a time
  through `/ingest`. Mentioned as a "sample feature to build" in the brief
  but not required for core scope.
- **Cost tracking is approximate.** `metrics.py` estimates USD cost from a
  fixed per-1K-token rate set in config, not a live pricing lookup - fine
  for the Grafana panel's purpose (relative cost trend), not meant to match
  an actual invoice.
