"""Transparent intervention scoring.

    containment_score = paths_removed
                      - alpha * collateral
                      - beta  * uncertainty
                      - gamma * irreversibility

* paths_removed: share of inferred time-respecting propagation paths (agent
  pairs) in the episode that disappear in the counterfactual graph.
* collateral: share of *unrelated* activity (events outside the episode, in
  the loaded slice) that the control would also touch.
* uncertainty: 1 - mean confidence of the candidate edges the control removes
  (removing only weak edges = acting on thin evidence).
* irreversibility: 1 - reversibility of the control type.

We never report "risk reduced by X%": the claim is "removes X% of inferred
propagation paths".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..graph.propagation import PropagationConfig
from .candidates import Intervention, generate_candidates
from .counterfactual import CounterfactualResult, simulate

CONF_WEIGHT = {"strong": 0.9, "moderate": 0.6, "weak": 0.3}


@dataclass
class ScoringConfig:
    alpha: float = 1.5   # collateral penalty (prefer small, targeted controls)
    beta: float = 0.25   # uncertainty penalty
    gamma: float = 0.3   # irreversibility penalty


@dataclass
class RankedIntervention:
    intervention: Intervention
    paths_removed: float
    n_paths_before: int
    n_paths_removed: int
    affected_agents: list[str]
    collateral: float
    collateral_events: int
    uncertainty: float
    reversibility: float
    score: float
    confidence: str
    counterfactual: CounterfactualResult

    def headline(self) -> str:
        rev = "yes" if self.reversibility >= 0.8 else "partly" if self.reversibility >= 0.6 else "costly"
        return (f"{self.intervention.label} — removes {self.paths_removed:.0%} of inferred propagation paths "
                f"({self.n_paths_removed}/{self.n_paths_before}); directly affects {len(self.affected_agents)} agent(s); "
                f"touches {self.collateral:.1%} of unrelated activity; reversible: {rev}; confidence: {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervention": self.intervention.label, "kind": self.intervention.kind, "target": self.intervention.target,
            "paths_removed_pct": round(100 * self.paths_removed, 1), "n_paths_before": self.n_paths_before,
            "n_paths_removed": self.n_paths_removed, "affected_agents": self.affected_agents,
            "collateral_pct": round(100 * self.collateral, 2), "collateral_events": self.collateral_events,
            "uncertainty": round(self.uncertainty, 3), "reversibility": self.reversibility,
            "containment_score": round(self.score, 4), "confidence": self.confidence, "headline": self.headline(),
            "removed_path_examples": self.counterfactual.removed_path_examples,
        }


def event_meta(events: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for r in events.itertuples(index=False):
        md = r.metadata if isinstance(r.metadata, dict) else {}
        out[r.event_id] = {"action_type": r.action_type, "tool": md.get("tool"), "actor": r.actor_id}
    return out


def rank_interventions(
    episode,
    events: pd.DataFrame,
    prop_config: PropagationConfig | None = None,
    config: ScoringConfig | None = None,
    candidates: list[Intervention] | None = None,
    ev_meta: dict[str, dict] | None = None,
) -> list[RankedIntervention]:
    cfg = config or ScoringConfig()
    ev_meta = ev_meta or event_meta(events)
    cands = candidates if candidates is not None else generate_candidates(episode, events)
    eset = set(episode.event_ids)
    inside = events[events.event_id.isin(eset)]
    outside = events[~events.event_id.isin(eset)]
    n_out = max(1, len(outside))
    out: list[RankedIntervention] = []
    for iv in cands:
        cf = simulate(iv, episode.pairs, episode.agent_edges, ev_meta, prop_config)
        affected = sorted({r.actor_id for r in inside.itertuples(index=False) if iv.touches_event(r)})
        coll_n = sum(1 for r in outside.itertuples(index=False) if iv.touches_event(r))
        coll = coll_n / n_out
        removed = cf.removed_edges
        unc = 1.0 - (sum(CONF_WEIGHT[e.confidence] for e in removed) / len(removed)) if removed else 1.0
        score = cf.paths_removed_frac - cfg.alpha * coll - cfg.beta * unc - cfg.gamma * (1 - iv.reversibility)
        conf = "high" if unc <= 0.2 else "moderate" if unc <= 0.45 else "low"
        out.append(RankedIntervention(
            intervention=iv, paths_removed=cf.paths_removed_frac, n_paths_before=len(cf.paths_before),
            n_paths_removed=len(cf.removed_paths), affected_agents=affected, collateral=coll,
            collateral_events=coll_n, uncertainty=unc, reversibility=iv.reversibility, score=score,
            confidence=conf, counterfactual=cf,
        ))
    out.sort(key=lambda r: -r.score)
    return out
