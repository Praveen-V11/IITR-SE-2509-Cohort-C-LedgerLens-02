"""
Runtime configuration for LedgerLens.

Everything that might need to change between a laptop run and a Cloud Run
deploy lives here, pulled from environment variables with sane local
defaults. Nothing in this file should ever hold a real secret - the actual
OpenAI key is injected at runtime via the environment (see .env.example).
"""

import os
from functools import lru_cache


class Settings:
    # --- OpenAI ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    vision_model: str = os.getenv("LEDGERLENS_VISION_MODEL", "gpt-4o")
    moderation_model: str = os.getenv("LEDGERLENS_MODERATION_MODEL", "omni-moderation-latest")

    # --- Confidence routing ---
    # Fields at or above this score are auto-approved without a human touch.
    # Kept as a class attribute (not a constant module-level var) so tests
    # can monkeypatch it per-case without messing with global state.
    review_threshold: float = float(os.getenv("LEDGERLENS_REVIEW_THRESHOLD", "0.75"))

    # --- Moderation ---
    moderation_block_threshold: float = float(os.getenv("LEDGERLENS_MOD_BLOCK_THRESHOLD", "0.5"))

    # --- Storage ---
    database_url: str = os.getenv("LEDGERLENS_DB_URL", "sqlite:///./ledgerlens.db")
    upload_dir: str = os.getenv("LEDGERLENS_UPLOAD_DIR", "uploads")

    # --- Image pre-processing ---
    max_image_dimension_px: int = int(os.getenv("LEDGERLENS_MAX_IMG_DIM", "2048"))

    # --- Cost tracking (USD per 1K tokens - update if OpenAI pricing changes) ---
    # These are approximate figures used only for the Grafana cost panel, not
    # for billing. Kept here rather than hardcoded in metrics.py so a pricing
    # change is a one-line edit.
    cost_per_1k_input_tokens: float = float(os.getenv("LEDGERLENS_COST_INPUT", "0.0025"))
    cost_per_1k_output_tokens: float = float(os.getenv("LEDGERLENS_COST_OUTPUT", "0.01"))

    # --- Misc ---
    app_env: str = os.getenv("LEDGERLENS_ENV", "local")


@lru_cache
def get_settings() -> Settings:
    # lru_cache turns this into a cheap singleton without a global mutable
    # at import time - easier to reason about and still overridable in tests
    # via get_settings.cache_clear().
    return Settings()
