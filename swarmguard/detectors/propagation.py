"""Information / strategy propagation: content or an action pattern first appears in one
agent and subsequently in multiple other agents along time-respecting candidate paths."""
from __future__ import annotations

import pandas as pd

from ..graph.reach import TemporalGraph
from .base import Detection


def propagation_tree(episode) -> tuple[str | None, dict, TemporalGraph]:
    """Pick the candidate source with the largest time-respecting reach (ties -> earliest activity)."""
    kept = {(e.src, e.dst) for e in episode.agent_edges}
    p = episode.pairs
    # Only content/resource-bearing pairs count as "pattern" propagation.
    if len(p):
        p = p[[(s, d) in kept for s, d in zip(p.src_agent, p.dst_agent)]]
        p = p[(p["semantic"] > 0) | (p["resource"] > 0) | (p["direct"] > 0)]
    tg = TemporalGraph(p)
    first_seen = {}
    if len(p):
        for a, t in pd.concat([p[["src_agent", "src_ts"]].set_axis(["a", "t"], axis=1),
                               p[["dst_agent", "dst_ts"]].set_axis(["a", "t"], axis=1)]).groupby("a")["t"].min().items():
            first_seen[a] = t
    best, best_arr = None, {}
    for s in sorted(tg.nodes, key=lambda a: first_seen.get(a, pd.Timestamp.max)):
        arr = tg.earliest_arrival(s)
        if len(arr) > len(best_arr):
            best, best_arr = s, arr
    return best, best_arr, tg


def detect_propagation(episode, events: pd.DataFrame, min_downstream: int = 3, min_supported_share: float = 0.5) -> Detection:
    root, arr, tg = propagation_tree(episode)
    downstream = [a for a in arr if a != root]
    # Evidence: the edge pairs along the earliest-arrival tree, in time order.
    tree = sorted(((v, info) for v, info in arr.items() if info[1] is not None), key=lambda x: x[1][0])
    ev_ids, hops = [], []
    edge_conf = {(e.src, e.dst): e.confidence for e in episode.agent_edges}
    for v, (t, pred, pair) in tree:
        ev_ids.extend(pair)
        hops.append({"from": pred, "to": v, "evidence": list(pair), "confidence": edge_conf.get((pred, v), "weak")})
    confs = [h["confidence"] for h in hops]
    strong_share = sum(c != "weak" for c in confs) / max(1, len(confs))
    # A pattern must reach several agents along mostly non-weak hops to count.
    triggered = root is not None and len(downstream) >= min_downstream and strong_share >= min_supported_share
    conf = "high" if strong_share >= 0.75 and len(downstream) >= 3 else "moderate" if strong_share >= 0.5 else "low"
    # Competing roots: other agents with the same reach.
    competing = [s for s in tg.nodes if s != root and len(tg.earliest_arrival(s)) == len(arr)]
    why = (f"Pattern traced from {root} to {len(downstream)} downstream agent(s) via {len(hops)} time-respecting "
           f"candidate hop(s); {strong_share:.0%} of hops are moderate/strong.") if root else "No multi-hop pattern."
    if competing:
        why += f" {len(competing)} other agent(s) have equal reach, so the source is ambiguous: {', '.join(competing[:3])}."
    return Detection(
        name="information_propagation", triggered=triggered, confidence=conf if triggered else "low", why=why,
        does_not_prove=("The candidate source is the earliest observed carrier of the pattern, not necessarily its "
                        "origin (unobserved channels, shared prompts, or a human may have seeded several agents)."),
        affected_agents=[root, *downstream] if root else [], evidence_event_ids=list(dict.fromkeys(ev_ids)),
        details={"candidate_source": root, "hops": hops, "competing_sources": competing},
    )
