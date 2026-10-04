"""Coordination burst: a sudden increase in cross-agent candidate interaction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import Detection


@dataclass
class BurstBaseline:
    """Per-hour rate of cross-agent candidate pairs over the whole loaded slice (active hours only)."""

    median_pairs_per_hour: float
    p90_pairs_per_hour: float
    median_agent_pairs_per_hour: float

    @classmethod
    def from_pairs(cls, pairs: pd.DataFrame) -> "BurstBaseline":
        if pairs is None or pairs.empty:
            return cls(0.0, 0.0, 0.0)
        hour = pairs["dst_ts"].dt.floor("h")
        per_hour = pairs.groupby(hour).size()
        agent_pairs = pairs.assign(h=hour).groupby("h").apply(
            lambda g: len(set(zip(g.src_agent, g.dst_agent))), include_groups=False)
        return cls(float(per_hour.median()), float(np.percentile(per_hour, 90)), float(agent_pairs.median()))


def detect_coordination_burst(episode, baseline: BurstBaseline, min_ratio: float = 2.0) -> Detection:
    st = episode.stats
    hours = max(st["duration_hours"], 0.25)  # avoid inflating rates for very short episodes
    rate = st["n_candidate_pairs"] / hours
    agent_pairs = len({(e.src, e.dst) for e in episode.agent_edges})
    ratio = rate / max(baseline.median_pairs_per_hour, 1.0)
    triggered = ratio >= min_ratio and st["n_agents"] >= 3 and agent_pairs >= 3
    conf = "high" if ratio >= 4 and rate >= baseline.p90_pairs_per_hour else "moderate" if ratio >= 3 else "low"
    why = (f"{rate:.0f} cross-agent candidate interactions/hour among {st['n_agents']} agents "
           f"({agent_pairs} directed agent pairs) vs. a slice median of {baseline.median_pairs_per_hour:.0f}/hour "
           f"(×{ratio:.1f}; p90={baseline.p90_pairs_per_hour:.0f}).")
    return Detection(
        name="coordination_burst", triggered=triggered, confidence=conf if triggered else "low", why=why,
        does_not_prove=("Coordination is usually benign (shared goals, scheduled events, a human prompt). A burst "
                        "says interaction intensified, not that it was unsafe or that any agent initiated it."),
        affected_agents=episode.agents,
        evidence_event_ids=list(episode.pairs.sort_values("score", ascending=False)["dst_event"].head(10)),
        details={"rate_per_hour": round(rate, 1), "ratio_vs_median": round(ratio, 2), "agent_pairs": agent_pairs},
    )
