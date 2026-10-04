"""End-to-end on the synthetic benchmark with the dependency-free TF-IDF encoder."""
from swarmguard.analysis.evaluation import run_benchmark
from swarmguard.analysis.pipeline import run_pipeline
from swarmguard.analysis.reports import build_report, render_markdown
from swarmguard.data.synthetic import make_benchmark
from swarmguard.embeddings.encoder import TfidfEncoder


def test_benchmark_recovers_structure_and_ranks_targeted_control_first():
    m = run_benchmark(TfidfEncoder())
    assert m["best_intervention_rank"] == 1
    assert m["confounder_handled"]
    assert m["episode_jaccard"] >= 0.7 and not m["episode_includes_parallel_cluster"]
    assert m["edge_recall_any_confidence"] == 1.0
    assert m["primary_edge_precision"] >= 0.8


def test_report_contains_evidence_and_caveats():
    b = make_benchmark()
    res = run_pipeline(b.events, encoder=TfidfEncoder(), progress=False)
    rep = build_report(res)
    assert rep["episodes"] and rep["episodes"][0]["interventions"]
    iv = rep["episodes"][0]["interventions"][0]
    assert "removes" in iv["headline"] and "risk" not in iv["headline"].lower()
    md = render_markdown(rep)
    assert "not causal ground truth" in md and "Does not prove" in md
    for d in rep["episodes"][0]["detectors"]:
        assert d["does_not_prove"]
