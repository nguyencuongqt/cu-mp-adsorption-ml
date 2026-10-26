"""Thin wrappers around the LLM providers used for generating and judging explanations.

API keys are read from the project-local .env file (OPENAI_API_KEY, ANTHROPIC_API_KEY) into the
process environment; they are never printed or written anywhere.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
KEY_NAMES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

GENERATOR = ("openai", "gpt-4o-mini")          # same model as in the submitted version
JUDGES = {                                     # label -> (provider, model)
    "same_model_as_generator": ("openai", "gpt-4o-mini"),
    "independent_provider": ("anthropic", "claude-haiku-4-5-20251001"),
}


def _load_env() -> None:
    if not ENV_FILE.exists():
        return
    for raw in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.split("#", 1)[0].strip().strip("'\"")
        if key in KEY_NAMES.values() and value:
            os.environ[key] = value


def available(provider: str) -> bool:
    _load_env()
    return bool(os.getenv(KEY_NAMES[provider]))


def call_llm(provider: str, model: str, system: str, user: str, temperature: float = 0.0, max_tokens: int = 900,
             retries: int = 3) -> str:
    """One chat completion; retries on transient errors."""
    _load_env()
    last: Exception | None = None
    for attempt in range(retries):
        try:
            if provider == "openai":
                from openai import OpenAI

                resp = OpenAI().chat.completions.create(
                    model=model, temperature=temperature, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                return resp.choices[0].message.content or ""
            if provider == "anthropic":
                import anthropic

                # Anthropic SDK 1.x does not expose a sampling-temperature argument
                # for this endpoint; Claude therefore uses the provider default.
                resp = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")).messages.create(
                    model=model, max_tokens=max_tokens, system=system,
                    messages=[{"role": "user", "content": user}])
                return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
            raise ValueError(f"Unknown provider: {provider}")
        except Exception as exc:  # network and rate-limit errors
            last = exc
            if any(code in str(exc) for code in ("401", "403", "invalid_api_key", "authentication")):
                break
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{provider}/{model} call failed: {type(last).__name__}: {str(last)[:300]}")
