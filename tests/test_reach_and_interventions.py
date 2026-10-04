from datetime import datetime, timedelta, timezone

import pandas as pd

from swarmguard.graph.propagation import collapse_agent_edges
from swarmguard.graph.reach import TemporalGraph
from swarmguard.interventions.candidates import Intervention
from swarmguard.interventions.counterfactual import simulate

T = datetime(2026, 1, 1, tzinfo=timezone.utc)


def pair(s, d, se, de, t0, t1, resource=0.0, shared=(), direct=0.0, semantic=0.6):
    return {"src_agent": s, "dst_agent": d, "src_event": se, "dst_event": de,
            "src_ts": T + timedelta(minutes=t0), "dst_ts": T + timedelta(minutes=t1), "dt_minutes": t1 - t0,
            "score": 0.6, "temporal": 0.9, "semantic": semantic, "resource": resource, "direct": direct,
            "goal": 0.0, "action": 0.3, "shared_resources": list(shared), "common_cause": [], "common_cause_actors": []}


def test_reachability_is_time_respecting():
    # B->C happens BEFORE A->B, so A cannot reach C through B.
    p = pd.DataFrame([pair("B", "C", "b1", "c1", 0, 5), pair("A", "B", "a1", "b2", 10, 15)])
    tg = TemporalGraph(p)
    assert "C" not in tg.earliest_arrival("A")
    p2 = pd.DataFrame([pair("A", "B", "a1", "b2", 0, 5), pair("B", "C", "b3", "c1", 10, 15)])
    tg2 = TemporalGraph(p2)
    assert set(tg2.earliest_arrival("A")) == {"A", "B", "C"}
    assert [(u, v) for u, v, _ in tg2.path_to("A", "C")] == [("A", "B"), ("B", "C")]


def _chain_pairs():
    R = "resource:paste/x"
    return pd.DataFrame([
        pair("A", "B", "a1", "b1", 0, 5, direct=1.0),
        pair("B", "C", "b2", "c1", 6, 10, resource=1.0, shared=[R], semantic=0.0),
        pair("B", "D", "b2", "d1", 6, 12, resource=1.0, shared=[R], semantic=0.0),
    ]), R


def test_restrict_writes_cuts_resource_paths_and_keeps_input_intact():
    pairs, R = _chain_pairs()
    edges = collapse_agent_edges(pairs)
    before = pairs.copy()
    meta = {e: {"action_type": "message", "tool": None} for e in set(pairs.src_event) | set(pairs.dst_event)}
    cf = simulate(Intervention("restrict_writes", R), pairs, edges, meta)
    assert ("A", "B") in cf.paths_after                        # direct edge untouched
    assert ("B", "C") not in cf.paths_after and ("A", "D") not in cf.paths_after
    assert cf.paths_removed_frac > 0.5
    pd.testing.assert_frame_equal(pairs, before)               # counterfactual works on a copy


def test_quarantine_removes_all_paths_through_agent():
    pairs, _ = _chain_pairs()
    edges = collapse_agent_edges(pairs)
    meta = {e: {"action_type": "message", "tool": None} for e in set(pairs.src_event) | set(pairs.dst_event)}
    cf = simulate(Intervention("quarantine_agent", "B"), pairs, edges, meta)
    assert cf.paths_after == set() and cf.paths_removed_frac == 1.0


def test_block_channel_only_drops_message_borne_pairs():
    pairs, _ = _chain_pairs()
    edges = collapse_agent_edges(pairs)
    meta = {e: {"action_type": "message", "tool": None} for e in set(pairs.src_event) | set(pairs.dst_event)}
    cf = simulate(Intervention("block_channel", "A->B"), pairs, edges, meta)
    assert ("A", "B") not in cf.paths_after and ("B", "C") in cf.paths_after
