"""Historical-response audit: compare SwarmGuard's recommendation with what operators actually did.

Applies when the data contains real containment actions, e.g. German Wiki admin deletions
(``action_type == "delete"`` by a human) and later recreations (``metadata.recreation_of``).
For each episode it reports:

* which of the episode's pages were deleted, when, and how long after the episode ended;
* whether the top-ranked SwarmGuard target was among them (agreement) and the time gap;
* recreations after deletion (persistence despite containment);
* writes to the page after the episode but before deletion (activity a timely,
  narrow control might have prevented).

These are observational comparisons, not proof that an earlier control would have worked.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def _hours(a: pd.Timestamp, b: pd.Timestamp) -> float:
    return round((b - a).total_seconds() / 3600.0, 1)


def historical_response(episode, events: pd.DataFrame, ranked: list | None = None) -> dict[str, Any] | None:
    deletes = events[(events.action_type == "delete") & (events.actor_type == "human")]
    if deletes.empty:
        return None
    sub = events[events.event_id.isin(set(episode.event_ids))]
    pages = set(sub.target_id.dropna()) | {r for r in episode.resources if r.startswith("wiki:")}
    pages = {p for p in pages if str(p).startswith("wiki:")}
    if not pages:
        return None
    on_pages = events[events.target_id.isin(pages)]
    dels = deletes[deletes.target_id.isin(pages)].sort_values("timestamp")
    first_del = dels.groupby("target_id")["timestamp"].min().to_dict()
    recreations = on_pages[(on_pages.action_type == "create") &
                           on_pages.metadata.map(lambda m: bool(isinstance(m, dict) and m.get("recreation_of")))]
    post = on_pages[(on_pages.timestamp > episode.end) & (on_pages.action_type.isin(["edit", "create"]))]
    writes_before_delete = sum(1 for r in post.itertuples(index=False)
                               if r.target_id in first_del and r.timestamp < first_del[r.target_id])
    # Base rate: if the operator removed nearly everything, "they also removed our page" is not evidence.
    written = set(events[events.action_type.isin(["edit", "create"]) & (events.actor_type == "agent")].target_id.dropna())
    base_rate = len(written & set(deletes.target_id)) / max(1, len(written))
    out: dict[str, Any] = {
        "operator_deletion_base_rate": round(base_rate, 3),
        "episode_pages": len(pages),
        "pages_deleted": len(first_del),
        "deleted_by": sorted(set(dels.actor_id)),
        "hours_from_episode_end_to_first_deletion": (_hours(episode.end, min(first_del.values())) if first_del else None),
        "recreations_after_deletion": int(len(recreations)),
        "writes_after_episode_before_deletion": int(writes_before_delete),
        "writes_after_episode_on_never_deleted_pages": int(sum(1 for r in post.itertuples(index=False)
                                                               if r.target_id not in first_del)),
    }
    if ranked:
        top = ranked[0].intervention
        tgt = top.target.removeprefix("resource:") if top.kind in ("restrict_writes", "isolate_memory") else None
        out["top_recommendation"] = top.label
        if tgt and tgt in pages:
            d = first_del.get(tgt)
            out["top_target_deleted"] = d is not None
            out["top_target_hours_to_deletion"] = _hours(episode.end, d) if d is not None else None
            out["agreement"] = (("operator later removed the same page SwarmGuard ranks first"
                                 if d is not None else "operator never removed the page SwarmGuard ranks first")
                                + f" (base rate: operator removed {base_rate:.0%} of agent-written pages, so agreement "
                                  "alone is weak evidence; timing and selectivity are the informative comparison)")
        else:
            out["agreement"] = "top recommendation is not a page-level control; compare manually"
    return out
