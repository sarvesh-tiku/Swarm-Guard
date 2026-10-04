"""Candidate propagation inference.

For an earlier event ``a`` by agent A and a later event ``b`` by agent B, we
compute an *association* score

    score(a -> b) = w_time*temporal + w_semantic*semantic + w_resource*resource
                  + w_direct*direct + w_goal*goal + w_action*action

and keep the pair only if at least one non-temporal signal is present (two
agents merely being active at the same time is not evidence). Event-pair
candidates are then collapsed into agent->agent ``candidate_propagation`` edges
that carry their score decomposition, evidence event ids, a confidence class,
alternative explanations, and a plain-language explanation.

Nothing here is a causal claim: an edge means "B's later behavior is
consistent with having been influenced by A's earlier behavior, given these
observable signals".

Candidate generation avoids O(N^2):
  * events are processed in time order with a sliding window;
  * within the window we only score (i) events sharing a resource with ``b``,
    (ii) messages that directly address B / are addressed by b, and
    (iii) the top-k semantic neighbours of ``b`` (one mat-vec per event over a
    window capped at ``max_window_events``).
"""
from __future__ import annotations

import bisect
import re
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..data.schemas import frame_resources, to_seconds

COMPONENTS = ("temporal", "semantic", "resource", "direct", "goal", "action")
READ_ONLY_ACTIONS = frozenset({"memory_read", "visit"})

PAIR_COLUMNS = [
    "src_agent", "dst_agent", "src_event", "dst_event", "src_ts", "dst_ts", "dt_minutes",
    "score", *COMPONENTS, "shared_resources", "common_cause", "common_cause_actors", "same_origin",
]


@dataclass
class PropagationConfig:
    """Tunable weights and thresholds. Weights need not sum to 1 but the defaults do."""

    window_minutes: float = 120.0
    w_time: float = 0.15
    w_semantic: float = 0.30
    w_resource: float = 0.25
    w_direct: float = 0.15
    w_goal: float = 0.05
    w_action: float = 0.10
    semantic_floor: float = 0.5       # cosine below this -> 0; above it is rescaled to (0, 1]
    semantic_top_k: int = 5           # semantic neighbours considered per event
    max_window_events: int = 1500     # cap on look-back window size
    min_pair_score: float = 0.35      # event-pair candidates below this are dropped
    min_edge_score: float = 0.40      # agent edges below this are dropped
    max_pairs_per_agent_edge: int = 25  # evidence pairs kept per agent edge
    resource_df_cap: float = 0.5      # resources touched by > this share of agents are "ubiquitous" and down-weighted
    common_cause_sim: float = 0.6     # third-party event similar to both a and b at >= this -> confounder
    include_actor_types: tuple[str, ...] = ("agent",)
    template_min_actors: int = 3       # identical text from >= this many actors = template  # who can be a propagation source/target

    def weights(self) -> dict[str, float]:
        return {
            "temporal": self.w_time, "semantic": self.w_semantic, "resource": self.w_resource,
            "direct": self.w_direct, "goal": self.w_goal, "action": self.w_action,
        }


@dataclass
class AgentEdge:
    """A collapsed agent->agent candidate propagation edge."""

    src: str
    dst: str
    score: float
    breakdown: dict[str, float]
    support: int
    first_ts: pd.Timestamp
    last_ts: pd.Timestamp
    evidence_pairs: list[tuple[str, str]]
    shared_resources: list[str]
    confidence: str
    alternatives: list[str]
    explanation: str
    common_cause_events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["first_ts"] = str(self.first_ts)
        d["last_ts"] = str(self.last_ts)
        return d


def temporal_score(dt_minutes: float, window: float) -> float:
    """Exponential decay in (0, 1]; 0 outside [0, window]."""
    if dt_minutes < 0 or dt_minutes > window:
        return 0.0
    return float(math.exp(-dt_minutes / max(1.0, window / 3.0)))


def _action_score(a: Any, b: Any) -> float:
    if a.action_type != b.action_type:
        return 0.0
    ta = (a.metadata or {}).get("tool")
    tb = (b.metadata or {}).get("tool")
    if ta and tb and ta == tb:
        return 1.0
    # Messages are the default action in chat logs; matching on them says little.
    return 0.3 if a.action_type == "message" else 0.6


def _direct_score(a: Any, b: Any) -> float:
    """Structural direct links (explicit recipient / reply id). Mention-based links are handled in score_pairs."""
    if a.target_type == "agent" and a.target_id == b.actor_id:
        return 1.0
    if (b.metadata or {}).get("reply_to") == a.event_id:
        return 1.0
    return 0.0


def _same_origin(a: Any, b: Any) -> str | None:
    """Shared origin hint (e.g. the same /16 address block) under two different actor ids."""
    oa = (a.metadata or {}).get("origin")
    return oa if oa and oa == (b.metadata or {}).get("origin") else None


def _goal_score(a: Any, b: Any, sem: float) -> float:
    """Goal transition of B that echoes A's earlier content, or same explicit goal key."""
    ga = (a.metadata or {}).get("goal_key")
    gb = (b.metadata or {}).get("goal_key")
    if b.action_type == "goal_set" and sem > 0:
        return 1.0
    if ga and gb and ga == gb:
        return 0.5
    return 0.0


def score_pairs(
    events: pd.DataFrame,
    embeddings: np.ndarray | None,
    config: PropagationConfig | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Generate and score candidate event pairs (a earlier, b later, different agents).

    Args:
        events: canonical event frame (time-sorted).
        embeddings: L2-normalized matrix aligned with ``events`` rows (or None to disable semantics).
    Returns:
        DataFrame with :data:`PAIR_COLUMNS`.
    """
    cfg = config or PropagationConfig()
    W = cfg.weights()
    ev = events.reset_index(drop=True)
    n = len(ev)
    if n < 2:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    ts = to_seconds(ev["timestamp"])
    rows = list(ev.itertuples(index=False))
    eligible = np.array([r.actor_type in cfg.include_actor_types for r in rows])
    # Templated text (identical content posted by several actors, e.g. placeholders or
    # canned status lines) carries no propagation signal: exclude it from semantics.
    norm = [re.sub(r"\s+", " ", str(r.content or "")).strip().lower() for r in rows]
    text_actors: dict[str, set] = defaultdict(set)
    for r, t in zip(rows, norm):
        if t:
            text_actors[t].add(r.actor_id)
    has_text = np.array([bool(t) and len(text_actors[t]) < cfg.template_min_actors for t in norm])
    actors = ev["actor_id"].to_numpy()
    window_s = cfg.window_minutes * 60.0

    # Resource index and inverse "agent frequency" weighting so ubiquitous
    # resources (e.g. the main group-chat room) carry little evidence.
    res_of = [frame_resources(r) for r in rows]
    res_agents: dict[str, set] = defaultdict(set)
    res_events: dict[str, list[int]] = defaultdict(list)
    for i, rs in enumerate(res_of):
        for r in rs:
            res_agents[r].add(actors[i])
            res_events[r].append(i)
    n_agents = max(1, len(set(actors[eligible])))
    res_weight = {}
    for r, ags in res_agents.items():
        share = len(ags) / n_agents
        res_weight[r] = 1.0 if share <= cfg.resource_df_cap else max(0.1, 1.0 - share)
        # Absolute crowding: a page touched by hundreds of identities is a broadcast board.
        res_weight[r] *= min(1.0, 3.0 / math.log2(len(ags) + 1)) if len(ags) > 7 else 1.0
        # Chat rooms are broadcast channels; co-presence in them is weak evidence.
        if r.startswith("room:"):
            res_weight[r] *= 0.2

    out: list[dict] = []
    last_by_actor: dict[str, int] = {}
    it = range(n)
    if progress and n > 2000:
        it = tqdm(it, desc="scoring candidate pairs", unit="ev")
    for j in it:
        if not eligible[j]:
            continue
        b = rows[j]
        lo = bisect.bisect_left(ts, ts[j] - window_s, 0, j)
        lo = max(lo, j - cfg.max_window_events)
        if lo >= j:
            last_by_actor[b.actor_id] = j
            continue
        cand: dict[int, None] = {}
        # (i) shared resources
        for r in res_of[j]:
            lst = res_events[r]
            k = bisect.bisect_left(lst, lo)
            for i in lst[k: bisect.bisect_left(lst, j)][-50:]:
                cand[i] = None
        # (ii) direct addressing. Only two narrow cases count, because agents in a
        # shared chat name each other constantly:
        #   reply:   b mentions A -> A's most recent event before b;
        #   inbox:   a mentions B and was posted after B's previous action, so b is
        #            B's first action after being addressed.
        direct_set: dict[int, float] = {}
        for m in (b.metadata or {}).get("mentions") or []:
            k = last_by_actor.get(m)
            if k is not None and k >= lo:
                direct_set[k] = 1.0
        prev_b = last_by_actor.get(b.actor_id, -1)
        for i in range(max(lo, prev_b + 1, j - 400), j):
            if b.actor_id in ((rows[i].metadata or {}).get("mentions") or []):
                direct_set[i] = 1.0
        for i in direct_set:
            cand[i] = None
        # (iii) semantic neighbours
        sims_window = None
        if embeddings is not None and has_text[j]:
            sims_window = embeddings[lo:j] @ embeddings[j]
            k = min(cfg.semantic_top_k * 3, len(sims_window))
            top = np.argpartition(-sims_window, k - 1)[:k] if k > 0 else []
            added = 0
            for t in sorted(top, key=lambda t: -sims_window[t]):
                i = lo + int(t)
                if actors[i] != b.actor_id and sims_window[t] >= cfg.semantic_floor:
                    cand[i] = None
                    added += 1
                    if added >= cfg.semantic_top_k:
                        break

        last_by_actor[b.actor_id] = j
        for i in cand:
            a = rows[i]
            if not eligible[i] or a.actor_id == b.actor_id or i >= j:
                continue
            dt_min = (ts[j] - ts[i]) / 60.0
            t_s = temporal_score(dt_min, cfg.window_minutes)
            sem = 0.0
            if sims_window is not None and has_text[i]:
                cos = float(sims_window[i - lo])
                # Rescale above the floor: in a single-community chat, cos≈0.5 is "same topic", not echo.
                sem = (cos - cfg.semantic_floor) / (1.0 - cfg.semantic_floor) if cos >= cfg.semantic_floor else 0.0
            # A shared resource is only a channel if the earlier event put something into it;
            # two agents merely *reading* the same object cannot influence each other through it.
            shared = [] if a.action_type in READ_ONLY_ACTIONS else [r for r in res_of[i] if r in set(res_of[j])]
            res_s = max((res_weight[r] for r in shared), default=0.0)
            direct = direct_set.get(i, 0.0) or _direct_score(a, b)
            goal = _goal_score(a, b, sem)
            act = _action_score(a, b)
            # Gate: need at least one non-temporal, non-action signal.
            if sem == 0.0 and res_s < 0.15 and direct == 0.0 and goal == 0.0:
                continue
            comp = {"temporal": t_s, "semantic": sem, "resource": res_s, "direct": direct, "goal": goal, "action": act}
            score = sum(W[c] * comp[c] for c in COMPONENTS)
            if score < cfg.min_pair_score:
                continue
            out.append({
                "src_agent": a.actor_id, "dst_agent": b.actor_id,
                "src_event": a.event_id, "dst_event": b.event_id,
                "src_ts": a.timestamp, "dst_ts": b.timestamp, "dt_minutes": round(dt_min, 2),
                "score": round(score, 4), **{c: round(v, 4) for c, v in comp.items()},
                "shared_resources": shared, "common_cause": [], "common_cause_actors": [],
                "same_origin": _same_origin(a, b),
            })
    pairs = pd.DataFrame(out, columns=PAIR_COLUMNS)
    if len(pairs) and embeddings is not None:
        _flag_common_causes(pairs, ev, embeddings, cfg)
    return pairs


def _flag_common_causes(pairs: pd.DataFrame, ev: pd.DataFrame, emb: np.ndarray, cfg: PropagationConfig) -> None:
    """Attach third-party events (often human/broadcast messages) similar to both a and b.

    If C said something before both a and b that both resemble, "A influenced B"
    competes with "C influenced both" (a common-cause confounder).
    """
    idx = {eid: i for i, eid in enumerate(ev["event_id"])}
    ts = to_seconds(ev["timestamp"])
    actors = ev["actor_id"].to_numpy()
    has_text = ev["content"].fillna("").astype(str).str.strip().ne("").to_numpy()
    window_s = cfg.window_minutes * 60.0
    cc_col, cc_act = [], []
    for r in pairs.itertuples(index=False):
        if r.semantic == 0.0:
            cc_col.append([])
            cc_act.append([])
            continue
        i, j = idx[r.src_event], idx[r.dst_event]
        lo = bisect.bisect_left(ts, ts[i] - window_s, 0, i)
        if lo >= i:
            cc_col.append([])
            cc_act.append([])
            continue
        sa = emb[lo:i] @ emb[i]
        sb = emb[lo:i] @ emb[j]
        mask = (sa >= cfg.common_cause_sim) & (sb >= cfg.common_cause_sim) & has_text[lo:i]
        mask &= (actors[lo:i] != r.src_agent) & (actors[lo:i] != r.dst_agent)
        hits = np.nonzero(mask)[0]
        cc_col.append([ev.at[lo + int(h), "event_id"] for h in hits[-3:]])
        cc_act.append([str(actors[lo + int(h)]) for h in hits[-3:]])
    pairs["common_cause"] = cc_col
    pairs["common_cause_actors"] = cc_act


def rescore_pairs(pairs: pd.DataFrame, config: PropagationConfig | None = None) -> pd.DataFrame:
    """Recompute scores after components were edited (used by counterfactuals); re-applies gate + threshold."""
    cfg = config or PropagationConfig()
    if pairs.empty:
        return pairs
    W = cfg.weights()
    p = pairs.copy()
    p["score"] = sum(W[c] * p[c] for c in COMPONENTS).round(4)
    gate = (p["semantic"] > 0) | (p["resource"] >= 0.15) | (p["direct"] > 0) | (p["goal"] > 0)
    return p[gate & (p["score"] >= cfg.min_pair_score)].reset_index(drop=True)


def _confidence(score: float, support: int, n_signals: int, has_cc: bool) -> str:
    level = 0
    if score >= 0.55 and n_signals >= 2 and support >= 2:
        level = 2
    elif score >= 0.45 or (n_signals >= 2 and support >= 2):
        level = 1
    if has_cc:
        level -= 1
    return "strong" if level >= 2 else "moderate" if level == 1 else "weak"


def _explain(src: str, dst: str, bd: dict[str, float], support: int, shared: list[str], dt: float) -> str:
    parts = []
    if bd["direct"] > 0:
        parts.append(f"{src} and {dst} addressed each other directly")
    if bd["resource"] > 0 and shared:
        parts.append(f"both touched {', '.join(shared[:2])}")
    if bd["semantic"] > 0:
        parts.append(f"{dst}'s later content is semantically similar to {src}'s (cos≈{bd['semantic']:.2f})")
    if bd["goal"] > 0:
        parts.append("goal overlap / goal transition after exposure")
    if bd["action"] >= 0.6:
        parts.append("same action/tool type")
    body = "; ".join(parts) or "temporal co-occurrence only"
    return f"{dst} acted after {src} (median lag {dt:.0f} min, {support} supporting event pair(s)): {body}."


def collapse_agent_edges(pairs: pd.DataFrame, config: PropagationConfig | None = None) -> list[AgentEdge]:
    """Collapse event-pair candidates into agent->agent edges.

    Edge score = mean of the top-3 pair scores (robust to a single lucky
    coincidence, and does not grow without bound with chat volume).
    """
    cfg = config or PropagationConfig()
    if pairs is None or pairs.empty:
        return []
    # Agents with direct/resource evidence of reaching X are X's *upstream* in a chain;
    # resembling both ends of X->Y is then expected, so they are not counted as confounders.
    upstream: dict[str, set[str]] = defaultdict(set)
    for r in pairs[(pairs["direct"] > 0) | (pairs["resource"] >= 0.15)][["src_agent", "dst_agent"]].itertuples(index=False):
        upstream[r.dst_agent].add(r.src_agent)
    # Vectorized aggregation over all (src, dst) groups; Python objects only for edges that pass.
    p = pairs.sort_values(["src_agent", "dst_agent", "score"], ascending=[True, True, False], kind="stable")
    p = p.reset_index(drop=True)
    p["_rank"] = p.groupby(["src_agent", "dst_agent"], sort=False).cumcount()
    top = p[p["_rank"] < 3]
    agg = top.groupby(["src_agent", "dst_agent"], sort=False)[["score", *COMPONENTS]].mean()
    agg = agg[agg["score"] >= cfg.min_edge_score]
    if agg.empty:
        return []
    grp = p.groupby(["src_agent", "dst_agent"], sort=False)
    stats = grp.agg(support=("score", "size"), first_ts=("src_ts", "min"), last_ts=("dst_ts", "max"),
                    dt_med=("dt_minutes", "median")).loc[agg.index].to_dict("index")
    origin_share = top.assign(_o=top["same_origin"].notna()).groupby(["src_agent", "dst_agent"], sort=False)["_o"].mean().to_dict() \
        if "same_origin" in top else None
    idx = grp.indices
    cols = {c: p[c].tolist() for c in ("src_event", "dst_event", "shared_resources", "common_cause",
                                        "common_cause_actors", "same_origin") if c in p}
    edges: list[AgentEdge] = []
    for (s, d), row in agg.iterrows():
        rows = idx[(s, d)]                      # positions, already sorted by score desc
        st = stats[(s, d)]
        score = float(row["score"])
        bd = {c: round(float(row[c]), 4) for c in COMPONENTS}
        n_signals = sum(1 for c in ("semantic", "resource", "direct", "goal") if bd[c] > 0)
        shared = sorted({r for i in rows for r in cols["shared_resources"][i]})
        # Confounders are judged on the edge's best evidence pair only, so one
        # boilerplate pair cannot discredit an otherwise well-supported edge.
        b = rows[0]
        acts = (cols.get("common_cause_actors") or [None] * len(p))[b] or [None] * len(cols["common_cause"][b])
        cc = sorted({e for e, a in zip(cols["common_cause"][b], acts) if a not in upstream[s]})
        dt_med = float(st["dt_med"])
        alts = []
        if cc:
            alts.append(f"common cause: {len(cc)} earlier third-party event(s) resemble both sides ({', '.join(cc[:2])})")
        if bd["direct"] == 0 and bd["resource"] == 0:
            alts.append("semantic similarity alone: both agents may be following the same goal or prompt")
        if shared and all(r.startswith("room:") for r in shared):
            alts.append("shared broadcast room: co-presence rather than targeted influence")
        same_origin = bool(origin_share is not None and origin_share.get((s, d), 0) >= 0.5)
        if same_origin:
            o = next(cols["same_origin"][i] for i in rows if cols["same_origin"][i])
            alts.append(f"same origin ({o}) on both sides: possibly one operator under two "
                        "identities (self-propagation, not cross-agent influence)")
        if bd["goal"] > 0:
            alts.append("shared assigned goal: both agents were given the same individual goal (common cause)")
        if dt_med < 2:
            alts.append("near-simultaneous: could be a shared trigger rather than A→B")
        k = rows[: cfg.max_pairs_per_agent_edge]
        support = int(st["support"])
        edges.append(AgentEdge(
            src=str(s), dst=str(d), score=round(score, 4), breakdown=bd, support=support,
            first_ts=st["first_ts"], last_ts=st["last_ts"],
            evidence_pairs=[(cols["src_event"][i], cols["dst_event"][i]) for i in k],
            shared_resources=shared, confidence=_confidence(score, support, n_signals, bool(cc) or same_origin),
            alternatives=alts, explanation=_explain(str(s), str(d), bd, support, shared, dt_med),
            common_cause_events=cc,
        ))
    _mark_indirect(edges)
    edges.sort(key=lambda e: -e.score)
    return edges


_DOWNGRADE = {"strong": "moderate", "moderate": "weak", "weak": "weak"}


def _mark_indirect(edges: list[AgentEdge]) -> None:
    """Downgrade edges with a structural alternative explanation (no direct addressing):

    * indirect: A->C explained by a time-consistent path A->B->C (transitive shortcut);
    * shared upstream: C->D where some X reached both C and D first (sibling co-exposure).
    """
    idx = {(e.src, e.dst): e for e in edges}
    out_of: dict[str, list[AgentEdge]] = defaultdict(list)
    in_to: dict[str, list[AgentEdge]] = defaultdict(list)
    for e in edges:
        out_of[e.src].append(e)
        in_to[e.dst].append(e)
    for e in edges:
        if e.breakdown["direct"] > 0:
            continue
        for ab in out_of[e.src]:
            bc = idx.get((ab.dst, e.dst))
            if bc is None or ab.dst == e.dst:
                continue
            if ab.first_ts <= bc.last_ts and min(ab.score, bc.score) >= 0.9 * e.score:
                e.alternatives.append(f"possibly indirect: explained by {e.src}→{ab.dst}→{e.dst}")
                e.confidence = _DOWNGRADE[e.confidence]
                break
        else:
            # Sibling rule: some X reached both src and dst (with direct/resource evidence)
            # before this edge's evidence. Then src and dst may simply share an upstream.
            for xs in in_to[e.src]:
                xd = idx.get((xs.src, e.dst))
                if (xd is None or xs.src == e.dst or
                        not (xs.breakdown["direct"] > 0 or xs.breakdown["resource"] > 0) or
                        not (xd.breakdown["direct"] > 0 or xd.breakdown["resource"] > 0)):
                    continue
                if xs.first_ts <= e.first_ts and xd.score >= 0.9 * e.score:
                    e.alternatives.append(f"shared upstream: {xs.src} reached both {e.src} and {e.dst}")
                    e.confidence = _DOWNGRADE[e.confidence]
                    break


def edges_frame(edges: list[AgentEdge]) -> pd.DataFrame:
    return pd.DataFrame([{
        "src": e.src, "dst": e.dst, "score": e.score, "confidence": e.confidence, "support": e.support,
        **{f"c_{k}": v for k, v in e.breakdown.items()},
        "shared_resources": ", ".join(e.shared_resources[:3]), "alternatives": " | ".join(e.alternatives),
        "explanation": e.explanation,
    } for e in edges])
