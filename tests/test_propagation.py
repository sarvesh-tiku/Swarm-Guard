import numpy as np

from conftest import ev
from swarmguard.embeddings.encoder import TfidfEncoder
from swarmguard.graph.propagation import PropagationConfig, collapse_agent_edges, score_pairs, temporal_score


def _run(frame, events, emb=True):
    df = frame(events)
    E = TfidfEncoder().encode(df["content"].fillna("").tolist()) if emb else None
    pairs = score_pairs(df, E, PropagationConfig(), progress=False)
    return df, pairs, collapse_agent_edges(pairs)


def test_temporal_score_bounds():
    assert temporal_score(0, 60) == 1.0
    assert temporal_score(-1, 60) == 0.0 and temporal_score(61, 60) == 0.0
    assert 0 < temporal_score(30, 60) < 1


def test_temporal_cooccurrence_alone_is_not_evidence(frame):
    _, pairs, edges = _run(frame, [ev("a", 0, "A", "lunch plans today"), ev("b", 1, "B", "kernel compile finished")])
    assert pairs.empty and edges == []


def test_direct_reply_creates_edge_with_breakdown(frame):
    events = [ev("a", 0, "A", "@B please mirror the deploy script to the shared repo", mentions=["B"],
                 resources=["gitlab.com/g/r"]),
              ev("b", 4, "B", "@A mirrored the deploy script to the shared repo", mentions=["A"],
                 resources=["gitlab.com/g/r"])]
    _, pairs, edges = _run(frame, events)
    e = next(e for e in edges if (e.src, e.dst) == ("A", "B"))
    assert e.breakdown["direct"] == 1.0 and e.breakdown["resource"] > 0
    assert ("a", "b") in e.evidence_pairs and "A" in e.explanation
    assert not any((x.src, x.dst) == ("B", "A") for x in edges)  # no backwards edge


def test_common_cause_flagged(frame):
    events = [ev("h", 0, "op", "everyone update the weekly status spreadsheet with progress", actor_type="human"),
              ev("i", 3, "I", "updating the weekly status spreadsheet with my progress", resources=["s.com/x"]),
              ev("j", 5, "J", "updated the weekly status spreadsheet with my progress", resources=["s.com/x"]),
              *[ev(f"n{k}", 20 + k, "Z", f"unrelated note {k} about gardening tomatoes") for k in range(5)]]
    _, pairs, edges = _run(frame, events)
    ij = pairs[(pairs.src_agent == "I") & (pairs.dst_agent == "J")]
    assert len(ij) and ij.iloc[0]["common_cause"] == ["h"]
    # Either no agent edge is asserted, or it carries the confounder as an alternative explanation.
    e = next((e for e in edges if (e.src, e.dst) == ("I", "J")), None)
    assert e is None or any(a.startswith("common cause") for a in e.alternatives)


def test_humans_are_not_propagation_nodes_by_default(frame):
    events = [ev("h", 0, "op", "use the shared doc", actor_type="human", resources=["d.com/1"]),
              ev("a", 2, "A", "using the shared doc", resources=["d.com/1"])]
    _, pairs, _ = _run(frame, events)
    assert "op" not in set(pairs.src_agent) if len(pairs) else True
