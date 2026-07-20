"""Shared OpenAI runtime configuration helpers for the public repository."""

from __future__ import annotations

import os
from typing import Dict, Tuple


def get_openai_runtime_config(model_default: str = "deepseek-v3.2") -> Tuple[str, str, str]:
    """Read OpenAI-compatible runtime settings from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    model = os.getenv("OPENAI_MODEL", model_default).strip() or model_default
    return api_key, base_url, model


def build_openai_client_kwargs(api_key: str, base_url: str = "", **extra_kwargs) -> Dict[str, object]:
    """Build keyword arguments for `openai.OpenAI` without hardcoding secrets."""
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for this script. "
            "Set it in the environment or in a local .env file that is not committed."
        )

    kwargs: Dict[str, object] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    kwargs.update(extra_kwargs)
    return kwargs
