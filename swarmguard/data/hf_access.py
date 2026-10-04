"""Hugging Face authentication helpers for the gated AI Village dataset.

Token resolution order:
1. ``HF_TOKEN`` environment variable (also loaded from a local ``.env`` if python-dotenv is installed)
2. the local login created by ``hf auth login``

The token is never printed or logged.
"""
from __future__ import annotations

import os

REPO = "aidigestorg/ai-village"

GATED_HELP = (
    "This dataset is gated. Run `hf auth login` with the Hugging Face account that has been "
    f"granted access to {REPO}, or set HF_TOKEN."
)


class HFAccessError(RuntimeError):
    """Raised when the gated dataset cannot be accessed."""


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def hf_token() -> str | None:
    """Return the explicit HF_TOKEN if set, else None (meaning: use the local login)."""
    _load_dotenv()
    tok = os.environ.get("HF_TOKEN", "").strip()
    return tok or None


def token_kwarg() -> dict:
    """kwargs for datasets/huggingface_hub calls. Empty dict -> library uses the cached login."""
    tok = hf_token()
    return {"token": tok} if tok else {}


def whoami() -> str:
    """Return the authenticated username or raise HFAccessError."""
    from huggingface_hub import HfApi

    try:
        info = HfApi().whoami(**token_kwarg())
    except Exception as e:  # LocalTokenNotFoundError, HTTPError, ...
        raise HFAccessError(f"Not authenticated with Hugging Face ({type(e).__name__}). {GATED_HELP}") from e
    return str(info.get("name", "<unknown>"))


def explain_error(e: BaseException) -> str:
    """Turn a datasets/hub exception into an actionable message."""
    msg = str(e)
    low = msg.lower()
    if any(s in low for s in ("gated", "401", "403", "unauthorized", "authenticat", "access to it")):
        return GATED_HELP
    return f"{type(e).__name__}: {msg[:300]}"
