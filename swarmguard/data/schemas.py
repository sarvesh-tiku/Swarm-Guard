"""Canonical event model shared by every SwarmGuard adapter.

Every dataset (AI Village, German Wiki, synthetic benchmark) is normalized into
``Event`` records. Downstream code (graph, propagation, detectors, interventions)
only ever sees this representation, usually as a pandas DataFrame produced by
:func:`events_to_frame`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable

import pandas as pd

# Canonical action vocabulary. Adapters map raw action names onto these so that
# detectors and the action-similarity term can compare across datasets.
ACTION_TYPES = (
    "message",          # agent/human posts to a chat room or another agent
    "session_start",    # agent starts a computer-use / tool session
    "session_end",
    "tool_use",         # a structured tool/computer action
    "visit",            # agent visits an external resource (URL/page)
    "edit",             # agent edits/writes an external artifact (wiki page, doc)
    "create",           # agent creates an artifact (incl. recreation after deletion)
    "delete",           # artifact removed (e.g. an admin deletion = a historical containment action)
    "memory_write",
    "memory_read",
    "goal_set",         # an agent/village goal becomes active
    "goal_end",
    "wait",
    "other",
)

# Node types used in the heterogeneous graph.
NODE_TYPES = ("agent", "human", "message", "memory", "goal", "tool", "resource", "session", "room")

FRAME_COLUMNS = [
    "event_id", "timestamp", "actor_id", "actor_type", "action_type",
    "target_id", "target_type", "content", "source", "metadata",
]


@dataclass
class Event:
    """One observable action by one actor.

    Attributes:
        event_id: Globally unique id. Adapters prefix the raw id with the source
            config (e.g. ``chat:1234``) so ids never collide across tables.
        timestamp: Timezone-aware UTC timestamp.
        actor_id: Who acted (agent name/id, wiki account, ...).
        actor_type: ``agent`` | ``human`` | ``system``.
        action_type: One of :data:`ACTION_TYPES`.
        target_id: What the action touched (room, URL, page, session, goal), if any.
        target_type: Node type of the target (``room``, ``resource``, ``session``...).
        content: Human-readable text used for semantic similarity and evidence.
        source: Dataset/config the event came from (``ai_village:chat_messages``).
        metadata: Raw ids and raw fields preserved for evidence drill-down. Must
            include ``raw_id``. ``resources`` (list[str]) lists extra resources
            (e.g. URLs mentioned in a message) the event references.
    """

    event_id: str
    timestamp: datetime
    actor_id: str
    actor_type: str = "agent"
    action_type: str = "other"
    target_id: str | None = None
    target_type: str | None = None
    content: str | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def events_to_frame(events: Iterable[Event]) -> pd.DataFrame:
    """Convert events into the canonical, time-sorted DataFrame."""
    rows = [e.to_dict() for e in events]
    df = pd.DataFrame(rows, columns=FRAME_COLUMNS)
    if df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset="event_id", keep="first")
    return df.sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)


def frame_resources(row: Any) -> list[str]:
    """All resource keys an event touches: its target (if a resource) plus metadata['resources']."""
    out: list[str] = []
    tt = getattr(row, "target_type", None) if not isinstance(row, dict) else row.get("target_type")
    tid = getattr(row, "target_id", None) if not isinstance(row, dict) else row.get("target_id")
    md = getattr(row, "metadata", None) if not isinstance(row, dict) else row.get("metadata")
    # Goals are deliberately excluded: a shared *assigned* goal is a common cause, not a channel.
    if tid is not None and isinstance(tid, str) and tt in ("resource", "room", "memory"):
        out.append(f"{tt}:{tid}")
    if isinstance(md, dict):
        for r in md.get("resources", []) or []:
            key = f"resource:{r}"
            if key not in out:
                out.append(key)
    return out


def to_seconds(ts: pd.Series) -> "np.ndarray":
    """Epoch seconds (float) for a tz-aware datetime Series, independent of its unit (ns/us/ms)."""
    import numpy as np

    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return ((pd.to_datetime(ts, utc=True) - epoch) / pd.Timedelta(seconds=1)).to_numpy(dtype=np.float64)
