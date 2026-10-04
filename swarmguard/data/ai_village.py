"""AI Village (aidigestorg/ai-village) adapter.

Schema facts this adapter relies on (verified by scripts/inspect_ai_village.py,
documented in docs/ai_village_schema.md):

* All files are gzipped JSONL mirrors of DB tables, **ordered by uuid ``id``,
  not by time**. Time-window queries therefore need one full pass; we do that
  pass once and cache a time-sorted, slimmed Parquet copy.
* Timestamps are naive UTC strings (``2025-12-29 18:49:21.291984``).
* ``agents.id`` is referenced by ``chat_messages.agent_speaker_id``,
  ``computer_use_sessions.agent_id``, ``agent_goals.agent_id``, and
  ``events.data.agentId`` / ``events.data.speakerId``. Agent names are unique.
* ``events.data.actionType`` determines the payload. ``AGENT_TALK``/``USER_TALK``
  mirror ``chat_messages`` (via ``messageId``) and ``START_USING_COMPUTER``
  mirrors ``computer_use_sessions`` (via ``computerUseSessionId``), so we take
  chat from chat_messages, session starts from computer_use_sessions, and only
  the *other* action types from events (deduplication).
* Messages are posted to rooms (group chat); there is no recipient field.
  Directed addressing is inferred from agent-name mentions in the text.
* ``computer_use_turns.session_id`` -> ``computer_use_sessions.id``; turns are
  2.5 GB and only loaded for drill-down (see :func:`drilldown_turns`).

Modes:
  * ``download`` (default): hf_hub_download once into the HF cache, build the
    slim Parquet cache once, then every window query is a cheap Parquet filter.
  * ``stream``: ``load_dataset(..., streaming=True)`` and filter on the fly
    (no local copy of the raw files; one network pass per query).
"""
from __future__ import annotations

import gzip
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import pandas as pd
from tqdm.auto import tqdm

from ..embeddings.cache import cache_root
from .hf_access import REPO, HFAccessError, explain_error, token_kwarg
from .normalization import clip, extract_urls, parse_ts, url_resource_key
from .schemas import Event, events_to_frame

log = logging.getLogger(__name__)

REFERENCE_CONFIGS = ("agents", "agent_goals", "village_goals", "chat_rooms")
CORE_CONFIGS = ("chat_messages", "computer_use_sessions", "events")

# events.actionType -> canonical action_type. None = skip (duplicated elsewhere or pure noise).
EVENT_ACTIONS: dict[str, str | None] = {
    "AGENT_TALK": None,                 # from chat_messages
    "USER_TALK": None,                  # from chat_messages
    "START_USING_COMPUTER": None,       # from computer_use_sessions
    "STOP_USING_COMPUTER": "session_end",
    "CONSOLIDATE": "memory_write",
    "SEARCH_HISTORY": "memory_read",
    "OUTREACH_APPROVAL_REQUEST": "tool_use",
    "OUTREACH_APPROVAL_RESPONSE": "other",
    "REQUEST_HUMAN_HELPER": "tool_use",
    "CANCEL_REQUEST_FOR_HUMAN_HELPER": "tool_use",
    "STOP_HUMAN_USE_SESSION": "session_end",
    "ENTER_ROOM": "other",
    "WAIT": None,
    "PAUSE": None,
    "USER_NAME_CHANGE": None,
    "RESTARTING_AFTER_GOOGLE_SIGN_IN": None,
    "REQUEST_GOOGLE_SIGN_IN": None,
}
# Small text-ish fields kept from events.data in the slim cache (``output`` blobs are dropped).
_EVENT_KEEP = (
    "actionType", "agentId", "speakerId", "speakerName", "speakerType", "roomId", "roomName", "messageId",
    "computerUseSessionId", "summary", "nextSessionGoal", "nextShortDisplayedSessionGoal", "query",
    "answerToQuery", "medium", "recipient", "rationale", "messageContent", "approval", "adminComment",
    "outreachApprovalRequestId", "sessionGoal", "shortDisplayedSessionGoal", "humanConstraints",
    "endReason", "endComment", "previousRoomName", "userId", "newName", "oldName",
)


@dataclass
class Reference:
    """Small lookup tables loaded fully (all < 10 KB)."""

    agents: pd.DataFrame
    agent_goals: pd.DataFrame
    village_goals: pd.DataFrame
    rooms: pd.DataFrame
    agent_name: dict[str, str] = field(default_factory=dict)
    agent_model: dict[str, str] = field(default_factory=dict)
    room_name: dict[str, str] = field(default_factory=dict)
    user_name: dict[str, str] = field(default_factory=dict)
    changelog: list[tuple[datetime, str]] = field(default_factory=list)
    goal_intervals: dict[str, list[tuple[datetime | None, datetime | None, str]]] | None = None


# ---------------------------------------------------------------- raw access


def _download(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(REPO, filename, repo_type="dataset", **token_kwarg()))
    except Exception as e:
        raise HFAccessError(explain_error(e)) from e


def _iter_jsonl_gz(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_config(config: str, mode: str = "download") -> Iterator[dict]:
    """Yield raw rows of one config, either from the cached file or a HF stream."""
    if mode == "stream":
        from datasets import load_dataset

        try:
            ds = load_dataset(REPO, config, split="train", streaming=True, **token_kwarg())
        except Exception as e:
            raise HFAccessError(explain_error(e)) from e
        yield from ds
    else:
        yield from _iter_jsonl_gz(_download(f"{config}.jsonl.gz"))


def discover_configs() -> list[str]:
    from datasets import get_dataset_config_names

    try:
        return list(get_dataset_config_names(REPO, **token_kwarg()))
    except Exception as e:
        raise HFAccessError(explain_error(e)) from e


def validate_configs(required: Iterable[str] = REFERENCE_CONFIGS + CORE_CONFIGS) -> list[str]:
    available = set(discover_configs())
    missing = [c for c in required if c not in available]
    if missing:
        raise HFAccessError(f"AI Village configs missing from the hub: {missing}. Available: {sorted(available)}")
    return sorted(available)


# ---------------------------------------------------------------- reference


def _parse_changelog() -> list[tuple[datetime, str]]:
    try:
        text = _download("CHANGELOG.md").read_text()
    except Exception:
        return []
    out = []
    for m in re.finditer(r"^## (\d{4}-\d{2}-\d{2})[^\n]*\n(.*?)(?=^## |\Z)", text, re.M | re.S):
        d = datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        body = " ".join(m.group(2).split())
        out.append((d, body[:300]))
    return out


def load_reference(mode: str = "download") -> Reference:
    tables = {c: pd.DataFrame(list(iter_config(c, mode))) for c in REFERENCE_CONFIGS}
    ref = Reference(agents=tables["agents"], agent_goals=tables["agent_goals"],
                    village_goals=tables["village_goals"], rooms=tables["chat_rooms"])
    ref.agent_name = dict(zip(ref.agents["id"], ref.agents["name"]))
    ref.agent_model = dict(zip(ref.agents["id"], ref.agents["model_string"]))
    ref.room_name = dict(zip(ref.rooms["id"], ref.rooms["name"]))
    ref.changelog = _parse_changelog()
    return ref


# ---------------------------------------------------------------- slim cache


def _slim_event(r: dict) -> dict | None:
    d = r.get("data") or {}
    at = d.get("actionType")
    if EVENT_ACTIONS.get(at, "other") is None and at not in ("USER_TALK", "USER_NAME_CHANGE"):
        return None
    keep = {k: (clip(v, 4000) if isinstance(v, str) else v) for k, v in d.items() if k in _EVENT_KEEP}
    return {"id": r["id"], "created_at": r["created_at"], "event_index": r.get("event_index"),
            "payload": json.dumps(keep, ensure_ascii=False)}


def _slim_chat(r: dict) -> dict:
    return {"id": r["id"], "created_at": r["created_at"], "payload": json.dumps(
        {k: r.get(k) for k in ("agent_speaker_id", "user_speaker_id", "speaker_type", "content", "room_id")},
        ensure_ascii=False)}


def _slim_session(r: dict) -> dict:
    return {"id": r["id"], "created_at": r["created_at"], "payload": json.dumps({
        "agent_id": r.get("agent_id"), "session_goal": clip(r.get("session_goal"), 4000),
        "short_displayed_session_goal": r.get("short_displayed_session_goal"),
        "has_been_asked_to_stop": r.get("has_been_asked_to_stop")}, ensure_ascii=False)}


_SLIMMERS: dict[str, Callable[[dict], dict | None]] = {
    "events": _slim_event, "chat_messages": _slim_chat, "computer_use_sessions": _slim_session,
}


def slim_cache_dir() -> Path:
    d = cache_root() / "ai_village"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_slim_cache(configs: Iterable[str] = CORE_CONFIGS, force: bool = False) -> dict[str, Path]:
    """One pass over each core config -> time-sorted Parquet with small JSON payloads."""
    out = {}
    for c in configs:
        path = slim_cache_dir() / f"{c}.parquet"
        out[c] = path
        if path.exists() and not force:
            continue
        rows = []
        for r in tqdm(iter_config(c, "download"), desc=f"indexing {c}", unit="row", mininterval=2):
            s = _SLIMMERS[c](r)
            if s is not None:
                rows.append(s)
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        df.to_parquet(path, index=False)
        log.info("cached %s: %d rows -> %s", c, len(df), path)
    return out


def _read_window(config: str, start: datetime, end: datetime) -> list[dict]:
    path = build_slim_cache([config])[config]
    df = pd.read_parquet(path, filters=[("ts", ">=", pd.Timestamp(start)), ("ts", "<", pd.Timestamp(end))])
    rows = []
    for r in df.itertuples(index=False):
        p = json.loads(r.payload)
        row = {"id": r.id, "created_at": r.created_at}
        if config == "events":
            row["event_index"] = r.event_index
            row["data"] = p
        else:
            row.update(p)
        rows.append(row)
    return rows


def _stream_window(config: str, start: datetime, end: datetime, max_rows: int | None) -> list[dict]:
    rows = []
    for r in tqdm(iter_config(config, "stream"), desc=f"streaming {config}", unit="row", mininterval=2):
        ts = parse_ts(r.get("created_at"))
        if ts and start <= ts < end:
            s = _SLIMMERS[config](r)
            if s is None:
                continue
            row = {"id": r["id"], "created_at": r["created_at"]}
            if config == "events":
                row.update(event_index=r.get("event_index"), data=json.loads(s["payload"]))
            else:
                row.update(json.loads(s["payload"]))
            rows.append(row)
            if max_rows and len(rows) >= max_rows * 3:  # unordered file: over-collect then trim by time
                break
    return rows


def daily_activity() -> pd.DataFrame:
    """Messages / speakers per UTC day (from the slim chat cache); helps pick a window."""
    path = build_slim_cache(["chat_messages"])["chat_messages"]
    df = pd.read_parquet(path, columns=["ts", "payload"])
    p = df["payload"].map(json.loads)
    df["speaker"] = p.map(lambda x: x.get("agent_speaker_id") or x.get("user_speaker_id"))
    df["has_url"] = p.map(lambda x: "http" in (x.get("content") or ""))
    g = df.groupby(df["ts"].dt.date)
    return pd.DataFrame({"messages": g.size(), "speakers": g["speaker"].nunique(), "url_messages": g["has_url"].sum()})


# ---------------------------------------------------------------- normalization


class _Mentions:
    """Detect agent-name mentions in free text (full names + unique short aliases)."""

    def __init__(self, names: Iterable[str]) -> None:
        names = sorted(set(names), key=len, reverse=True)
        alias: dict[str, set[str]] = {}
        for n in names:
            forms = {n.lower()}
            parts = n.split()
            if len(parts) >= 2:
                forms.add(" ".join(parts[1:]).lower())      # "Opus 4.5", "3.5 Flash"
            if n.startswith("Claude ") and len(parts) == 2:  # "Claude Fable" -> "Fable"? keep exact only
                pass
            if len(parts) == 1 or n.lower().startswith(("o1", "o3", "o4")):
                forms.add(n.lower())
            for f in forms:
                alias.setdefault(f, set()).add(n)
        # Keep only unambiguous aliases of length >= 2 chars.
        self.alias = {a: next(iter(v)) for a, v in alias.items() if len(v) == 1 and len(a) >= 2}
        pats = sorted(self.alias, key=len, reverse=True)
        self.rx = re.compile(r"(?<![\w.\-])@?(" + "|".join(re.escape(p) for p in pats) + r")(?![\w\-]|\.\d)", re.I) if pats else None

    def find(self, text: str | None, exclude: str | None = None) -> list[str]:
        if not text or not self.rx:
            return []
        out = []
        for m in self.rx.finditer(text):
            n = self.alias.get(m.group(1).lower())
            if n and n != exclude and n not in out:
                out.append(n)
        return out


def _resources_from_text(*texts: str | None) -> tuple[list[str], list[str]]:
    urls, keys = [], []
    for t in texts:
        for u in extract_urls(t):
            k = url_resource_key(u)
            if k and k not in keys:
                keys.append(k)
                urls.append(u)
    return keys, urls


def _active_goal(ref: Reference, agent_id: str, ts: datetime) -> str | None:
    """Short name of the agent's individual goal active at ts (null start/end = open interval)."""
    if ref.goal_intervals is None:
        ref.goal_intervals = {}
        for r in ref.agent_goals.itertuples(index=False):
            ref.goal_intervals.setdefault(r.agent_id, []).append(
                (parse_ts(r.start_time), parse_ts(r.end_time), r.short_name))
    best = None
    for s, e, name in ref.goal_intervals.get(agent_id, []):
        if (s is None or s <= ts) and (e is None or ts < e):
            best = name
    return best


def normalize_chat(r: dict, ref: Reference, mentions: _Mentions) -> Event | None:
    ts = parse_ts(r["created_at"])
    if ts is None:
        return None
    if r.get("speaker_type") == "agent" and r.get("agent_speaker_id"):
        aid = r["agent_speaker_id"]
        actor, actor_type = ref.agent_name.get(aid, aid), "agent"
    else:
        uid = r.get("user_speaker_id") or "unknown"
        actor, actor_type, aid = f"human:{ref.user_name.get(uid, uid[:8])}", "human", uid
    content = r.get("content") or ""
    keys, urls = _resources_from_text(content)
    room = r.get("room_id")
    md = {"raw_id": r["id"], "raw_table": "chat_messages", "actor_raw_id": aid, "actor_name": actor,
          "room_name": ref.room_name.get(room, room), "resources": keys, "urls": urls,
          "mentions": mentions.find(content, exclude=actor)}
    if actor_type == "agent":
        md["goal_key"] = _active_goal(ref, aid, ts)
        md["model"] = ref.agent_model.get(aid)
    return Event(event_id=f"chat:{r['id']}", timestamp=ts, actor_id=actor, actor_type=actor_type,
                 action_type="message", target_id=room, target_type="room", content=clip(content, 2000),
                 source="ai_village:chat_messages", metadata=md)


def normalize_session(r: dict, ref: Reference, mentions: _Mentions) -> Event | None:
    ts = parse_ts(r["created_at"])
    aid = r.get("agent_id")
    if ts is None or not aid:
        return None
    actor = ref.agent_name.get(aid, aid)
    goal = r.get("session_goal") or ""
    short = r.get("short_displayed_session_goal") or ""
    keys, urls = _resources_from_text(goal)
    content = (short + " — " if short else "") + goal
    return Event(event_id=f"session:{r['id']}", timestamp=ts, actor_id=actor, actor_type="agent",
                 action_type="session_start", target_id=r["id"], target_type="session",
                 content=clip(content, 1500), source="ai_village:computer_use_sessions",
                 metadata={"raw_id": r["id"], "raw_table": "computer_use_sessions", "actor_raw_id": aid,
                           "actor_name": actor, "tool": "computer", "resources": keys, "urls": urls,
                           "mentions": mentions.find(goal, exclude=actor), "goal_key": _active_goal(ref, aid, ts),
                           "short_goal": short, "target_label": short or r["id"][:8]})


def normalize_event(r: dict, ref: Reference, mentions: _Mentions) -> Event | None:
    d = r.get("data") or {}
    at = d.get("actionType")
    action = EVENT_ACTIONS.get(at, "other")
    ts = parse_ts(r["created_at"])
    if action is None or ts is None:
        return None
    aid = d.get("agentId") or d.get("speakerId")
    actor, actor_type = (ref.agent_name.get(aid, aid), "agent") if aid else ("system", "system")
    target_id = target_type = None
    tool = None
    content = ""
    if at == "STOP_USING_COMPUTER":
        content = d.get("summary") or ""
        target_id, target_type = d.get("computerUseSessionId"), "session" if d.get("computerUseSessionId") else None
    elif at == "CONSOLIDATE":
        content = d.get("nextSessionGoal") or ""
        target_id, target_type = f"{actor}", "memory"
    elif at == "SEARCH_HISTORY":
        content = f"{d.get('query') or ''}\n→ {d.get('answerToQuery') or ''}"
        target_id, target_type, tool = "village_history", "memory", "search_history"
    elif at == "OUTREACH_APPROVAL_REQUEST":
        content = f"[outreach request to {d.get('recipient')} via {d.get('medium')}] {d.get('messageContent') or ''}"
        tool = "external_outreach"
    elif at == "OUTREACH_APPROVAL_RESPONSE":
        # The reviewer is a human admin; attribute to them, target the requesting agent.
        actor, actor_type = "human:outreach_reviewer", "human"
        verdict = "APPROVED" if d.get("approval") else "REJECTED"
        content = f"[outreach {verdict} for {ref.agent_name.get(aid, aid)}] {d.get('adminComment') or ''}"
        target_id, target_type = ref.agent_name.get(aid, aid), "agent"
    elif at in ("REQUEST_HUMAN_HELPER", "CANCEL_REQUEST_FOR_HUMAN_HELPER"):
        content = f"{d.get('sessionGoal') or ''} {d.get('humanConstraints') or ''}".strip()
        tool = "human_helper"
    elif at == "STOP_HUMAN_USE_SESSION":
        content = d.get("summary") or ""
        tool = "human_helper"
    elif at == "ENTER_ROOM":
        # Templated system text: kept out of `content` so it cannot create semantic matches.
        content = ""
        target_id, target_type = d.get("roomId"), "room"
    keys, urls = _resources_from_text(content, d.get("medium"))
    md = {"raw_id": r["id"], "raw_table": "events", "event_index": r.get("event_index"), "raw_action": at,
          "actor_raw_id": aid, "actor_name": actor, "resources": keys, "urls": urls, "tool": tool,
          "mentions": mentions.find(content, exclude=actor), "session_id": d.get("computerUseSessionId"),
          "room_name": ref.room_name.get(d.get("roomId"), d.get("roomId"))}
    if actor_type == "agent" and aid:
        md["goal_key"] = _active_goal(ref, aid, ts)
    if at.startswith("OUTREACH"):
        md.update(recipient=d.get("recipient"), medium=d.get("medium"), approval=d.get("approval"),
                  outreach_request_id=d.get("outreachApprovalRequestId"))
    return Event(event_id=f"event:{r['id']}", timestamp=ts, actor_id=actor, actor_type=actor_type,
                 action_type=action, target_id=target_id, target_type=target_type, content=clip(content, 2000),
                 source=f"ai_village:events:{at}", metadata=md)


def _goal_events(ref: Reference, start: datetime, end: datetime) -> list[Event]:
    out = []
    for r in ref.agent_goals.itertuples(index=False):
        s = parse_ts(r.start_time) or parse_ts(r.created_at)
        if s and start <= s < end:
            actor = ref.agent_name.get(r.agent_id, r.agent_id)
            out.append(Event(event_id=f"agent_goal:{r.id}", timestamp=s, actor_id=actor, actor_type="agent",
                             action_type="goal_set", target_id=r.short_name, target_type="goal",
                             content=f"{r.name} {r.description or ''}".strip(), source="ai_village:agent_goals",
                             metadata={"raw_id": r.id, "raw_table": "agent_goals", "actor_name": actor,
                                       "goal_key": r.short_name, "resources": [], "mentions": []}))
    for r in ref.village_goals.itertuples(index=False):
        s = parse_ts(r.start_time)
        if s and start <= s < end:
            out.append(Event(event_id=f"village_goal:{r.id}", timestamp=s, actor_id="village", actor_type="system",
                             action_type="goal_set", target_id="village_goal", target_type="goal", content=r.goal,
                             source="ai_village:village_goals",
                             metadata={"raw_id": r.id, "raw_table": "village_goals", "resources": [], "mentions": []}))
    return out


# ---------------------------------------------------------------- public API


@dataclass
class AIVillageSlice:
    events: pd.DataFrame
    reference: Reference
    start: datetime
    end: datetime
    context: dict[str, Any]


def load_window(
    start: datetime | str,
    end: datetime | str,
    max_events: int | None = 5000,
    mode: str = "download",
    include_humans: bool = True,
) -> AIVillageSlice:
    """Load and normalize a bounded AI Village time window.

    Args:
        start, end: UTC window (ISO strings accepted).
        max_events: cap on normalized events; the window is truncated in time
            order (keeps a contiguous prefix) so that propagation stays coherent.
        mode: "download" (cached Parquet) or "stream" (HF streaming).
    """
    start_dt, end_dt = parse_ts(start), parse_ts(end)
    if start_dt is None or end_dt is None or end_dt <= start_dt:
        raise ValueError(f"invalid window {start!r}..{end!r}")
    ref = load_reference("download" if mode == "download" else "stream")
    raw: dict[str, list[dict]] = {}
    for c in CORE_CONFIGS:
        raw[c] = _read_window(c, start_dt, end_dt) if mode == "download" else _stream_window(c, start_dt, end_dt, max_events)
    # Human display names come from USER_TALK events (users table is not exported).
    for r in raw["events"]:
        d = r["data"]
        if d.get("actionType") == "USER_TALK" and d.get("speakerId") and d.get("speakerName"):
            ref.user_name[d["speakerId"]] = d["speakerName"]
    active = {ref.agent_name.get(r.get("agent_speaker_id")) for r in raw["chat_messages"]} | \
             {ref.agent_name.get(r.get("agent_id")) for r in raw["computer_use_sessions"]}
    mentions = _Mentions(n for n in active if n)
    evs: list[Event] = []
    evs += [e for r in raw["chat_messages"] if (e := normalize_chat(r, ref, mentions))]
    evs += [e for r in raw["computer_use_sessions"] if (e := normalize_session(r, ref, mentions))]
    evs += [e for r in raw["events"] if (e := normalize_event(r, ref, mentions))]
    evs += _goal_events(ref, start_dt, end_dt)
    if not include_humans:
        evs = [e for e in evs if e.actor_type != "human"]
    df = events_to_frame(evs)
    truncated = False
    if max_events and len(df) > max_events:
        df = df.iloc[:max_events].reset_index(drop=True)
        truncated = True
    vg = [(parse_ts(r.start_time), parse_ts(r.end_time), r.goal) for r in ref.village_goals.itertuples(index=False)]
    active_vg = [g for s, e, g in vg if s and s <= end_dt and (e is None or e >= start_dt)]
    context = {
        "dataset": REPO, "mode": mode, "window_start": str(start_dt), "window_end": str(end_dt),
        "truncated_to_max_events": truncated, "raw_rows": {k: len(v) for k, v in raw.items()},
        "active_village_goals": active_vg,
        "nearby_scaffolding_changes": [(str(d.date()), t) for d, t in ref.changelog
                                       if start_dt - timedelta(days=3) <= d <= end_dt + timedelta(days=1)],
    }
    return AIVillageSlice(events=df, reference=ref, start=start_dt, end=end_dt, context=context)


def drilldown_turns(session_ids: Iterable[str], max_scan_rows: int | None = None, mode: str = "stream") -> pd.DataFrame:
    """Fetch computer_use_turns for specific sessions (expensive: the file is 2.5 GB and id-ordered).

    Streams the config once and keeps only matching rows; results are cached per session id.
    """
    wanted = set(session_ids)
    cache = slim_cache_dir() / "turns"
    cache.mkdir(exist_ok=True)
    found = {s: pd.read_parquet(cache / f"{s}.parquet") for s in wanted if (cache / f"{s}.parquet").exists()}
    todo = wanted - set(found)
    if todo:
        rows: dict[str, list[dict]] = {s: [] for s in todo}
        for n, r in enumerate(tqdm(iter_config("computer_use_turns", mode), desc="scanning turns", unit="row", mininterval=5)):
            if r.get("session_id") in todo:
                act = r.get("agent_action") or {}
                rows[r["session_id"]].append({
                    "turn_id": r["id"], "session_id": r["session_id"], "created_at": r["created_at"],
                    "action": json.dumps(act, ensure_ascii=False)[:2000], "output": clip(r.get("output"), 2000),
                    "error": clip(r.get("error"), 1000),
                })
            if max_scan_rows and n >= max_scan_rows:
                break
        for s, lst in rows.items():
            df = pd.DataFrame(lst, columns=["turn_id", "session_id", "created_at", "action", "output", "error"])
            df.to_parquet(cache / f"{s}.parquet", index=False)
            found[s] = df
    if not found:
        return pd.DataFrame()
    return pd.concat(found.values(), ignore_index=True).sort_values("created_at").reset_index(drop=True)
