"""Lightweight evaluation against the synthetic benchmark."""
from __future__ import annotations

from typing import Any

from ..data.synthetic import SyntheticBenchmark, make_benchmark
from ..embeddings.encoder import Encoder, get_encoder
from .pipeline import AnalysisResult, PipelineConfig, run_pipeline


def evaluate(result: AnalysisResult, bench: SyntheticBenchmark) -> dict[str, Any]:
    # Edge P/R: an edge is "asserted" if it is moderate/strong and not flagged with a common cause.
    asserted = {(e.src, e.dst) for e in result.agent_edges
                if e.confidence in ("moderate", "strong") and not e.common_cause_events}
    any_edge = {(e.src, e.dst) for e in result.agent_edges}
    structural = ("possibly indirect", "shared upstream", "common cause", "same origin")
    primary = {(e.src, e.dst) for e in result.agent_edges if e.confidence in ("moderate", "strong")
               and not any(a.startswith(structural) for a in e.alternatives)}
    tp_primary = len(primary & bench.true_edges)
    tp = len(asserted & bench.true_edges)
    precision = tp / len(asserted) if asserted else 0.0
    recall = tp / len(bench.true_edges)
    recall_any = len(any_edge & bench.true_edges) / len(bench.true_edges)

    # Episode recovery: best Jaccard of any episode with the true chain.
    best_ep, best_j = None, 0.0
    for ep in result.episodes:
        s = set(ep.event_ids)
        j = len(s & bench.episode_event_ids) / len(s | bench.episode_event_ids)
        if j > best_j:
            best_ep, best_j = ep, j

    source = None
    rank, top = None, None
    top_collateral = None
    leaked_parallel = False
    if best_ep is not None:
        prop = next(d for d in best_ep.detections if d.name == "information_propagation")
        source = prop.details.get("candidate_source")
        leaked_parallel = bool(set(best_ep.agents) & bench.parallel_cluster)
        ranked = result.interventions.get(best_ep.episode_id, [])
        for k, ri in enumerate(ranked, 1):
            if (ri.intervention.kind, ri.intervention.target) == bench.best_intervention:
                rank = k
                break
        if ranked:
            top = ranked[0].intervention.label
            top_collateral = round(ranked[0].collateral, 4)

    conf_edge = next((e for e in result.agent_edges if (e.src, e.dst) == bench.confounded_pair), None)
    confounder_handled = conf_edge is None or bool(conf_edge.common_cause_events) or conf_edge.confidence == "weak"
    return {
        "encoder": result.encoder_name,
        "edge_precision": round(precision, 3), "edge_recall": round(recall, 3),
        "edge_recall_any_confidence": round(recall_any, 3),
        "primary_edge_precision": round(tp_primary / len(primary), 3) if primary else 0.0,
        "primary_edge_recall": round(tp_primary / len(bench.true_edges), 3),
        "primary_edges": sorted(primary),
        "asserted_edges": sorted(asserted), "true_edges": sorted(bench.true_edges),
        "episode_jaccard": round(best_j, 3), "episode_includes_parallel_cluster": leaked_parallel,
        "source_recovered": source == bench.true_source, "candidate_source": source,
        "best_intervention_rank": rank, "top_intervention": top, "top_intervention_collateral": top_collateral,
        "confounder_handled": confounder_handled,
        "confounder_edge": None if conf_edge is None else
            {"confidence": conf_edge.confidence, "common_cause": conf_edge.common_cause_events, "alternatives": conf_edge.alternatives},
    }


def run_benchmark(encoder: Encoder | str = "auto", seed: int = 7, config: PipelineConfig | None = None) -> dict[str, Any]:
    bench = make_benchmark(seed=seed)
    enc = get_encoder(encoder) if isinstance(encoder, str) else encoder
    res = run_pipeline(bench.events, config=config, encoder=enc, progress=False)
    return evaluate(res, bench)
