"""Episode discovery: turn thousands of events into a handful of reviewable windows.

Algorithm (deterministic, no LLM):
  1. Split the timeline into activity segments at temporal gaps > ``gap_minutes``.
  2. Within each segment, union events connected by *strong* candidate pairs
     (score >= ``strong_pair_score``) -> connected components.
  3. Components with >= ``min_agents`` agents and >= ``min_events`` events are
     episodes; components longer than ``max_hours`` are split at their largest
     internal gap until they fit.
  4. Each episode gets its own collapsed agent graph, statistics and a label
     built from class-based TF-IDF terms + its dominant resource.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..data.schemas import frame_resources, to_seconds
from .propagation import AgentEdge, PropagationConfig, collapse_agent_edges


@dataclass
class EpisodeConfig:
    gap_minutes: float = 45.0
    strong_pair_score: float = 0.45
    min_agents: int = 3
    min_events: int = 4
    max_hours: float = 4.0
    max_episodes: int = 50
    max_agents: int = 40              # larger components are re-split at a stricter pair threshold


@dataclass
class Episode:
    episode_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    agents: list[str]
    resources: list[str]
    event_ids: list[str]
    pairs: pd.DataFrame
    agent_edges: list[AgentEdge]
    label: str
    stats: dict[str, Any] = field(default_factory=dict)
    detections: list[Any] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id, "label": self.label,
            "start": str(self.start), "end": str(self.end),
            "agents": self.agents, "resources": self.resources[:10],
            "n_events": len(self.event_ids), "n_candidate_edges": len(self.agent_edges),
            "stats": self.stats,
            "detectors": [d.name for d in self.detections if d.triggered],
        }


class _DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _split_long(idx: list[int], ts: np.ndarray, max_s: float) -> list[list[int]]:
    idx = sorted(idx)
    if ts[idx[-1]] - ts[idx[0]] <= max_s or len(idx) < 4:
        return [idx]
    gaps = np.diff(ts[idx])
    cut = int(np.argmax(gaps)) + 1
    return _split_long(idx[:cut], ts, max_s) + _split_long(idx[cut:], ts, max_s)


def _labels(docs: list[str], n_terms: int = 4) -> list[list[str]]:
    """Class-based TF-IDF: distinctive terms of each episode vs. the others."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not docs:
        return []
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2),
                              token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b", sublinear_tf=True,
                              max_df=0.8 if len(docs) > 3 else 1.0)
        X = vec.fit_transform(docs)
    except ValueError:
        return [[] for _ in docs]
    vocab = np.array(vec.get_feature_names_out())
    out = []
    for i in range(X.shape[0]):
        row = X[i].toarray().ravel()
        top = [vocab[t] for t in np.argsort(-row)[: n_terms * 3] if row[t] > 0]
        picked: list[str] = []
        for t in top:  # avoid "foo" + "foo bar" duplicates
            if not any(t in p or p in t for p in picked):
                picked.append(t)
            if len(picked) >= n_terms:
                break
        out.append(picked)
    return out


def discover_episodes(
    events: pd.DataFrame,
    pairs: pd.DataFrame,
    prop_config: PropagationConfig | None = None,
    config: EpisodeConfig | None = None,
) -> list[Episode]:
    cfg = config or EpisodeConfig()
    pcfg = prop_config or PropagationConfig()
    ev = events.reset_index(drop=True)
    if ev.empty or pairs is None or pairs.empty:
        return []
    idx = {eid: i for i, eid in enumerate(ev["event_id"])}
    ts = to_seconds(ev["timestamp"])

    # 1. activity segments
    seg = np.zeros(len(ev), dtype=int)
    if len(ev) > 1:
        seg[1:] = np.cumsum(np.diff(ts) > cfg.gap_minutes * 60)

    # 2. strong-pair components within segments
    dsu = _DSU(len(ev))
    strong = pairs[pairs["score"] >= cfg.strong_pair_score]
    for r in strong.itertuples(index=False):
        i, j = idx.get(r.src_event), idx.get(r.dst_event)
        if i is not None and j is not None and seg[i] == seg[j]:
            dsu.union(i, j)
    comps: dict[int, list[int]] = {}
    touched = set()
    for r in strong.itertuples(index=False):
        touched.add(idx[r.src_event])
        touched.add(idx[r.dst_event])
    for i in touched:
        comps.setdefault(dsu.find(i), []).append(i)

    # 2b. oversized components (dense broadcast activity) are re-split with stricter pairs
    def resplit(members: list[int], thr: float) -> list[list[int]]:
        n_ag = len(set(ev.loc[members, "actor_id"]))
        if n_ag <= cfg.max_agents or thr >= 0.85:
            return [members]
        mset = set(members)
        sub = strong_idx[(strong_idx.score >= thr)]
        d2 = _DSU(len(ev))
        touched2 = set()
        for i, j in zip(sub.i, sub.j):
            if i in mset and j in mset:
                d2.union(i, j)
                touched2.update((i, j))
        parts: dict[int, list[int]] = {}
        for i in touched2:
            parts.setdefault(d2.find(i), []).append(i)
        return [q for part in parts.values() for q in resplit(part, thr + 0.05)]

    strong_idx = pd.DataFrame({"i": strong.src_event.map(idx), "j": strong.dst_event.map(idx), "score": strong.score})
    comps = {k: v for k, v in enumerate(q for m in comps.values() for q in resplit(m, cfg.strong_pair_score + 0.05))}

    # 3. filter + split long components
    groups: list[list[int]] = []
    for members in comps.values():
        for part in _split_long(members, ts, cfg.max_hours * 3600):
            agents = set(ev.loc[part, "actor_id"])
            if len(agents) >= cfg.min_agents and len(part) >= cfg.min_events:
                groups.append(part)

    # 4. materialize
    actor_name = {}
    for r in ev.itertuples(index=False):
        if isinstance(r.metadata, dict) and r.metadata.get("actor_name"):
            actor_name[r.actor_id] = r.metadata["actor_name"]
    episodes: list[Episode] = []
    docs: list[str] = []
    for part in groups:
        eids = ev.loc[part, "event_id"].tolist()
        eset = set(eids)
        p = pairs[pairs["src_event"].isin(eset) & pairs["dst_event"].isin(eset)].reset_index(drop=True)
        edges = collapse_agent_edges(p, pcfg)
        sub = ev.loc[sorted(part)]
        res = Counter(r for row in sub.itertuples(index=False) for r in frame_resources(row) if not r.startswith("room:"))
        agents = sorted(set(sub["actor_id"]))
        dur_h = max((ts[max(part)] - ts[min(part)]) / 3600.0, 1 / 60)
        stats = {
            "n_events": len(part),
            "n_agents": len(agents),
            "duration_hours": round(dur_h, 2),
            "events_per_hour": round(len(part) / dur_h, 1),
            "n_candidate_pairs": int(len(p)),
            "n_agent_edges": len(edges),
            "n_strong_edges": sum(e.confidence == "strong" for e in edges),
            "action_mix": dict(Counter(sub["action_type"]).most_common(6)),
            "top_resources": res.most_common(5),
        }
        episodes.append(Episode(
            episode_id="", start=sub["timestamp"].min(), end=sub["timestamp"].max(), agents=agents,
            resources=[r for r, _ in res.most_common()], event_ids=eids, pairs=p, agent_edges=edges,
            label="", stats=stats,
        ))
        docs.append(" ".join(sub["content"].fillna("").astype(str).str.slice(0, 2000)))
    terms = _labels(docs)
    for ep, t in zip(episodes, terms):
        res_hint = f" @ {ep.resources[0].split(':', 1)[1][:50]}" if ep.resources else ""
        ep.label = (" / ".join(t) or "unlabeled activity") + res_hint
        ep.stats["agent_names"] = [actor_name.get(a, a) for a in ep.agents]
    # Rank: many edges, many agents, then recency.
    episodes.sort(key=lambda e: (-(e.stats["n_agent_edges"] + 2 * e.stats["n_strong_edges"]), -e.stats["n_agents"]))
    episodes = episodes[: cfg.max_episodes]
    for k, ep in enumerate(episodes, 1):
        ep.episode_id = f"EP{k:03d}"
    return episodes
