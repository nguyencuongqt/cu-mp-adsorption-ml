"""Helpers for local LLM credentials and the independent judge configuration.

``OPENAI_API_KEY`` remains dedicated to generating C2/C3 explanations.  The
judge can be configured independently, so an Anthropic model can score those
explanations without provider self-preference.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
ENV_NAMES = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_WORKSPACE_ID",
    "LLM_JUDGE_PROVIDER",
    "LLM_JUDGE_MODEL",
}

DEFAULT_JUDGE_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}


def load_local_env(override: bool = True) -> None:
    """Load selected keys from project-local ``.env`` if present.

    By default, values in ``.env`` override inherited shell variables. This is
    deliberate for this repo because long-running IDE terminals often keep a
    stale API key alive after the user rotates it in the OpenAI dashboard.
    """
    if not ENV_FILE.exists():
        return
    for raw_line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in ENV_NAMES:
            continue
        value = value.split("#", 1)[0].strip().strip("'\"")
        if value and (override or key not in os.environ):
            os.environ[key] = value


def mask_key(value: str | None) -> str:
    if not value:
        return "missing"
    if len(value) <= 8:
        return value
    return f"{value[:8]}...{value[-4:]}"


def resolve_provider() -> tuple[str | None, str | None]:
    """Return the OpenAI credential used to generate C2/C3 explanations."""
    load_local_env()
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return "openai", openai_key
    return None, None


def resolve_judge_provider() -> tuple[str | None, str | None, str | None]:
    """Return ``(provider, api_key, model)`` for the LLM-as-Judge step.

    If no judge provider is configured explicitly, prefer Claude whenever an
    Anthropic key is present; this makes the independent-provider evaluation
    the safe default while preserving legacy OpenAI-only installations.
    """
    load_local_env()
    requested = os.getenv("LLM_JUDGE_PROVIDER", "").strip().lower()
    if requested and requested not in DEFAULT_JUDGE_MODELS:
        raise ValueError(
            "LLM_JUDGE_PROVIDER must be 'anthropic' or 'openai', "
            f"not {requested!r}."
        )

    if requested:
        provider = requested
    elif os.getenv("ANTHROPIC_API_KEY"):
        provider = "anthropic"
    elif os.getenv("OPENAI_API_KEY"):
        provider = "openai"
    else:
        return None, None, None

    key_name = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    api_key = os.getenv(key_name)
    model = os.getenv("LLM_JUDGE_MODEL", "").strip() or DEFAULT_JUDGE_MODELS[provider]
    return provider, api_key, model


def judge_label(provider: str | None, model: str | None) -> str:
    """Return a concise, report-safe label without exposing credentials."""
    if provider == "anthropic" and model == "claude-haiku-4-5-20251001":
        return "Claude Haiku 4.5 (Anthropic)"
    if provider == "openai" and model == "gpt-4o-mini":
        return "GPT-4o-mini (OpenAI)"
    if provider and model:
        return f"{model} ({provider})"
    return "None"
