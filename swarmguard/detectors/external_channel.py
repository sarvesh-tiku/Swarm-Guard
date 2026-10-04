"""External-channel coordination: multiple agents start coordinating through the same
external resource (doc, repo, site, pastebin) rather than only through internal chat."""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from ..data.schemas import frame_resources
from .base import Detection


def detect_external_channel(episode, events: pd.DataFrame, min_agents: int = 3, min_pairs: int = 3) -> Detection:
    eset = set(episode.event_ids)
    sub = events[events["event_id"].isin(eset)]
    touch: dict[str, list[tuple[pd.Timestamp, str, str]]] = defaultdict(list)
    for r in sub.itertuples(index=False):
        for res in frame_resources(r):
            if res.startswith("resource:"):
                touch[res].append((r.timestamp, r.actor_id, r.event_id))
    pair_support: dict[str, int] = defaultdict(int)
    for lst in episode.pairs["shared_resources"]:
        for res in lst:
            if res.startswith("resource:"):
                pair_support[res] += 1
    channels = []
    for res, lst in touch.items():
        agents = list(dict.fromkeys(a for _, a, _ in sorted(lst)))
        if len(agents) >= min_agents and pair_support.get(res, 0) >= min_pairs:
            channels.append({"resource": res, "agents": agents, "first_actor": agents[0],
                             "n_touches": len(lst), "supporting_pairs": pair_support[res],
                             "event_ids": [e for _, _, e in sorted(lst)][:15]})
    channels.sort(key=lambda c: (-len(c["agents"]), -c["supporting_pairs"]))
    n_pairs = max(1, len(episode.pairs))
    ext_share = sum(1 for lst in episode.pairs["shared_resources"] if any(r.startswith("resource:") for r in lst)) / n_pairs
    triggered = bool(channels)
    top = channels[0] if channels else None
    conf = "low"
    if top:
        conf = "high" if len(top["agents"]) >= 4 and top["supporting_pairs"] >= 4 else "moderate"
    why = (f"{len(channels)} external resource(s) used by >= {min_agents} agents; top: {top['resource']} "
           f"({len(top['agents'])} agents, first used by {top['first_actor']}, {top['supporting_pairs']} candidate pairs). "
           f"{ext_share:.0%} of the episode's candidate pairs run through external resources.") if top else \
        "No external resource shared by enough agents."
    return Detection(
        name="external_channel_coordination", triggered=triggered, confidence=conf, why=why,
        does_not_prove=("Sharing a doc/repo/site is how agents normally collaborate. This flags where coordination "
                        "leaves internal chat (relevant for observability and containment), not misuse."),
        affected_agents=top["agents"] if top else [], affected_resources=[c["resource"] for c in channels[:5]],
        evidence_event_ids=top["event_ids"] if top else [], details={"channels": channels[:5], "external_pair_share": round(ext_share, 3)},
    )
