"""Helpers for loading local OpenAI API keys."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
KEY_NAMES = ("OPENAI_API_KEY",)


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
        if key not in KEY_NAMES:
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
    """Return ``("openai", api_key)`` when OPENAI_API_KEY is available."""
    load_local_env()
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return "openai", openai_key
    return None, None
