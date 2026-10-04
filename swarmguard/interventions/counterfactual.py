"""Structural counterfactuals: apply an intervention to a *copy* of the inferred
evidence and compare time-respecting reachability before and after.

Graph-based counterfactual estimate, not causal ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..graph.propagation import AgentEdge, PropagationConfig, collapse_agent_edges, rescore_pairs
from ..graph.reach import TemporalGraph
from .candidates import Intervention

DISCLAIMER = "Graph-based counterfactual estimate, not causal ground truth."


def kept_pairs(pairs: pd.DataFrame, edges: list[AgentEdge]) -> pd.DataFrame:
    """Pairs that belong to agent edges which survived the edge threshold."""
    if pairs.empty:
        return pairs
    keep = {(e.src, e.dst) for e in edges}
    return pairs[[(s, d) in keep for s, d in zip(pairs.src_agent, pairs.dst_agent)]]


@dataclass
class CounterfactualResult:
    intervention: Intervention
    paths_before: set[tuple[str, str]]
    paths_after: set[tuple[str, str]]
    edges_before: list[AgentEdge]
    edges_after: list[AgentEdge]
    disclaimer: str = DISCLAIMER
    removed_path_examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def removed_paths(self) -> set[tuple[str, str]]:
        return self.paths_before - self.paths_after

    @property
    def paths_removed_frac(self) -> float:
        return len(self.removed_paths) / max(1, len(self.paths_before))

    @property
    def removed_edges(self) -> list[AgentEdge]:
        after = {(e.src, e.dst) for e in self.edges_after}
        return [e for e in self.edges_before if (e.src, e.dst) not in after]


def simulate(
    intervention: Intervention,
    pairs: pd.DataFrame,
    edges: list[AgentEdge],
    ev_meta: dict[str, dict],
    config: PropagationConfig | None = None,
    n_examples: int = 5,
) -> CounterfactualResult:
    cfg = config or PropagationConfig()
    before_pairs = kept_pairs(pairs, edges)
    tg_before = TemporalGraph(before_pairs)
    paths_before = tg_before.reachable_pairs()

    edited = rescore_pairs(intervention.apply(pairs, ev_meta), cfg)
    edges_after = collapse_agent_edges(edited, cfg)
    tg_after = TemporalGraph(kept_pairs(edited, edges_after))
    paths_after = tg_after.reachable_pairs()

    res = CounterfactualResult(intervention, paths_before, paths_after, edges, edges_after)
    for s, t in sorted(res.removed_paths)[:n_examples]:
        hops = tg_before.path_to(s, t)
        res.removed_path_examples.append({"source": s, "target": t,
                                          "path": [f"{u}→{v}" for u, v, _ in hops],
                                          "evidence": [list(ev) for _, _, ev in hops]})
    return res
