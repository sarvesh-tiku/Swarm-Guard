"""Dataset-independent normalization helpers (timestamps, URLs, text)."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import urlparse

import pandas as pd

URL_RE = re.compile(r"https?://[^\s<>\"'`)\]}]+", re.IGNORECASE)

# Domains that are infrastructure noise rather than meaningful shared channels.
_NOISE_DOMAINS = {"localhost", "127.0.0.1", "[redacted]"}


def parse_ts(value: Any) -> datetime | None:
    """Parse ISO strings, epoch seconds/ms, or datetimes into tz-aware UTC. Returns None on failure."""
    if value is None or value == "" or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        # Heuristic: values above 1e11 are milliseconds.
        secs = value / 1000.0 if value > 1e11 else float(value)
        return datetime.fromtimestamp(secs, tz=timezone.utc)
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def extract_urls(text: str | None, limit: int = 10) -> list[str]:
    if not text:
        return []
    out = []
    text = _USERINFO_ANYWHERE_RE.sub(r"\1", text)
    for m in URL_RE.findall(text):
        u = m.rstrip(".,;:!?*")
        if u not in out:
            out.append(u)
        if len(out) >= limit:
            break
    return out


_USERINFO_RE = re.compile(r"^(https?://)[^/@\s]*@", re.IGNORECASE)
# Userinfo may hold shell substitutions with spaces, e.g. oauth2:$(glab auth token)@host.
_USERINFO_ANYWHERE_RE = re.compile(r"(https?://)(?:\$\([^)]*\)|\$\{[^}]*\}|[^/@\s])*@", re.IGNORECASE)


def _strip_userinfo(url: str) -> str:
    """Drop ``user:token@`` from URLs (git remotes embed credentials); never store them."""
    return _USERINFO_RE.sub(r"\1", url)


def url_resource_key(url: str, granularity: str = "page") -> str | None:
    """Normalize a URL into a resource key.

    granularity="page" keeps host + first two path segments (e.g. a specific
    Google Doc or GitHub repo); "domain" keeps only the host.
    """
    try:
        p = urlparse(_strip_userinfo(url))
    except ValueError:
        return None
    host = (p.hostname or "").lower().removeprefix("www.")
    if not host or host in _NOISE_DOMAINS:
        return None
    if granularity == "domain":
        return host
    parts = [s for s in p.path.split("/") if s]
    segs = parts[:2]
    # Google Docs-style ids live in the 3rd segment: /document/d/<id>
    if host.startswith("docs.google.com"):
        segs = parts[:3]
    # Code hosts: identify the repository, not the shared org/group namespace.
    # GitLab allows nested groups (group/subgroup/repo); stop at "/-/" route markers.
    elif host in ("gitlab.com", "github.com"):
        if "-" in parts:
            parts = parts[: parts.index("-")]
        segs = parts[:3] if host == "gitlab.com" else parts[:2]
        segs = [s.removesuffix(".git") for s in segs]
    return "/".join([host, *segs]) if segs else host


def clip(text: str | None, n: int = 600) -> str | None:
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


def first_present(obj: dict, keys: Iterable[str], default: Any = None) -> Any:
    for k in keys:
        if k in obj and obj[k] not in (None, ""):
            return obj[k]
    return default
