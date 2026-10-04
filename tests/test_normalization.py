import math

import pandas as pd

from swarmguard.data.normalization import extract_urls, parse_ts, url_resource_key
from swarmguard.data.schemas import to_seconds


def test_parse_ts_variants():
    assert parse_ts("2026-07-06 15:59:00").tzinfo is not None
    assert parse_ts(None) is None and parse_ts(float("nan")) is None and parse_ts("") is None
    assert parse_ts(1_700_000_000_000).year == 2023  # epoch ms
    assert parse_ts("not a date") is None


def test_url_keys_identify_repo_not_group():
    a = url_resource_key("https://gitlab.com/ai-village-agents/village/quiet-rooms-gallery/-/blob/main/x.md")
    b = url_resource_key("https://gitlab.com/ai-village-agents/village/village-hub")
    assert a == "gitlab.com/ai-village-agents/village/quiet-rooms-gallery"
    assert a != b
    assert url_resource_key("https://github.com/org/repo/pull/5") == "github.com/org/repo"
    assert url_resource_key("https://www.example.com/a/b/c") == "example.com/a/b"
    assert url_resource_key("http://localhost:3000/x") is None


def test_extract_urls_strips_punctuation():
    assert extract_urls("see https://x.org/a. and (https://y.org/b)") == ["https://x.org/a", "https://y.org/b"]


def test_to_seconds_is_unit_independent():
    s_ns = pd.Series(pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:01:00"], utc=True))
    s_us = s_ns.astype("datetime64[us, UTC]")
    assert to_seconds(s_ns)[1] - to_seconds(s_ns)[0] == 60
    assert math.isclose(to_seconds(s_us)[1] - to_seconds(s_us)[0], 60)


def test_credentials_are_stripped_from_urls():
    u = "https://oauth2:[REDACTED]@gitlab.com/ai-village-agents/village/repo.git"
    assert extract_urls(f"git push {u}") == ["https://gitlab.com/ai-village-agents/village/repo.git"]
    assert url_resource_key(u) == "gitlab.com/ai-village-agents/village/repo"
    assert extract_urls("git remote set-url origin https://oauth2:$(glab auth token)@gitlab.com/g/v/r.git") == [
        "https://gitlab.com/g/v/r.git"]
