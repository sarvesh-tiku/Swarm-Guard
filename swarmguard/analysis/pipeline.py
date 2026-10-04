"""End-to-end SwarmGuard pipeline: events -> embeddings -> candidate pairs ->
agent edges -> episodes -> detectors -> ranked interventions."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..detectors import BurstBaseline, run_all
from ..embeddings.encoder import Encoder, get_encoder
from ..graph.episodes import Episode, EpisodeConfig, discover_episodes
from ..graph.propagation import AgentEdge, PropagationConfig, collapse_agent_edges, score_pairs
from ..interventions.scoring import RankedIntervention, ScoringConfig, event_meta, rank_interventions
from .historical import historical_response

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    episodes: EpisodeConfig = field(default_factory=EpisodeConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    encoder: str = "auto"              # auto | minilm | tfidf
    rank_top_episodes: int = 10        # interventions are ranked for the top-N episodes only


@dataclass
class AnalysisResult:
    events: pd.DataFrame
    embeddings: np.ndarray | None
    pairs: pd.DataFrame
    agent_edges: list[AgentEdge]
    episodes: list[Episode]
    interventions: dict[str, list[RankedIntervention]]
    baseline: BurstBaseline
    context: dict[str, Any]
    timings: dict[str, float]
    encoder_name: str

    def episode(self, episode_id: str) -> Episode:
        return next(e for e in self.episodes if e.episode_id == episode_id)


def embed_events(events: pd.DataFrame, encoder: Encoder) -> np.ndarray:
    texts = events["content"].fillna("").astype(str).tolist()
    return encoder.encode(texts)


def run_pipeline(
    events: pd.DataFrame,
    config: PipelineConfig | None = None,
    context: dict[str, Any] | None = None,
    encoder: Encoder | None = None,
    progress: bool = True,
) -> AnalysisResult:
    cfg = config or PipelineConfig()
    t: dict[str, float] = {}
    t0 = time.time()
    enc = encoder or get_encoder(cfg.encoder)
    emb = embed_events(events, enc)
    t["embed"] = time.time() - t0

    t0 = time.time()
    pairs = score_pairs(events, emb, cfg.propagation, progress=progress)
    edges = collapse_agent_edges(pairs, cfg.propagation)
    t["propagation"] = time.time() - t0

    t0 = time.time()
    episodes = discover_episodes(events, pairs, cfg.propagation, cfg.episodes)
    baseline = BurstBaseline.from_pairs(pairs)
    for ep in episodes:
        run_all(ep, events, baseline)
    t["episodes+detectors"] = time.time() - t0

    t0 = time.time()
    meta = event_meta(events)
    ranked = {ep.episode_id: rank_interventions(ep, events, cfg.propagation, cfg.scoring, ev_meta=meta)
              for ep in episodes[: cfg.rank_top_episodes]}
    for ep in episodes:
        ep.stats["historical_response"] = historical_response(ep, events, ranked.get(ep.episode_id))
    t["interventions"] = time.time() - t0
    log.info("pipeline timings: %s", {k: round(v, 1) for k, v in t.items()})
    return AnalysisResult(events=events, embeddings=emb, pairs=pairs, agent_edges=edges, episodes=episodes,
                          interventions=ranked, baseline=baseline, context=context or {}, timings=t,
                          encoder_name=getattr(enc, "name", type(enc).__name__))
