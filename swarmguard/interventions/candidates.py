"""Candidate containment interventions.

Each intervention is a *structural edit* of the inferred evidence: it drops or
weakens candidate event pairs that the real-world control would have
prevented. Nothing here controls a live system; these are recommendations an
operator could apply in the actual runtime (permissions, approval gates,
channel filters, agent pause).

Kinds:
  restrict_writes     shared resource R becomes read-only (except its owner)
  approval_gate       all external writes in the episode require human approval
  isolate_memory      shared memory object M is isolated per agent
  block_channel       A's messages are not delivered to / addressed at B
  limit_tool          agent X loses tool T
  quarantine_agent    agent X is paused (most disruptive)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..data.schemas import frame_resources

# Reversibility in [0, 1]: how cleanly the control can be undone without losing work/state.
REVERSIBILITY = {
    "restrict_writes": 0.95, "approval_gate": 0.9, "isolate_memory": 0.8,
    "block_channel": 0.9, "limit_tool": 0.85, "quarantine_agent": 0.5,
}
# Human-facing labels; stay concrete and do not imply live control.
VERBS = {
    "restrict_writes": "Restrict writes to shared resource",
    "approval_gate": "Require approval for external writes",
    "isolate_memory": "Isolate shared memory object",
    "block_channel": "Block communication edge",
    "limit_tool": "Limit tool for agent",
    "quarantine_agent": "Quarantine agent",
}


# Platforms the swarm uses but does not administer. Restricting "writes" there means an
# egress/posting gate on the agents, not a permission change on the resource.
THIRD_PARTY_HOSTS = ("youtube.com", "youtu.be", "twitter.com", "x.com", "reddit.com", "news.ycombinator.com",
                     "linkedin.com", "facebook.com", "instagram.com", "tiktok.com", "bsky.app", "manifold.markets",
                     "medium.com", "discord.com", "lichess.org", "wikipedia.org")


def is_third_party(resource: str) -> bool:
    host = resource.split("/", 1)[0]
    return any(host == h or host.endswith("." + h) for h in THIRD_PARTY_HOSTS)


@dataclass(frozen=True)
class Intervention:
    kind: str
    target: str                 # resource key, agent name, "A->B", "agent|tool", or "*external*"
    note: str = ""

    @property
    def label(self) -> str:
        t = self.target.replace("resource:", "").replace("memory:", "")
        if self.kind == "restrict_writes" and is_third_party(t):
            # The operator cannot make YouTube read-only; the real control is an egress/posting gate.
            return f"Gate agent posts/links to third-party resource: {t}"
        return f"{VERBS[self.kind]}: {t}"

    @property
    def reversibility(self) -> float:
        return REVERSIBILITY[self.kind]

    # -- structural effect on candidate pairs ------------------------------------
    def apply(self, pairs: pd.DataFrame, ev_meta: dict[str, dict]) -> pd.DataFrame:
        """Return edited pairs (components zeroed / rows dropped). Caller re-scores."""
        p = pairs.copy()
        if p.empty:
            return p
        if self.kind == "quarantine_agent":
            return p[(p.src_agent != self.target) & (p.dst_agent != self.target)]
        if self.kind == "block_channel":
            a, b = self.target.split("->")
            hit = (p.src_agent == a) & (p.dst_agent == b)
            is_msg = p.src_event.map(lambda e: ev_meta.get(e, {}).get("action_type") == "message")
            # Messages from A no longer reach B: drop message-borne pairs; others lose the direct term.
            p = p[~(hit & is_msg)].copy()
            p.loc[(p.src_agent == a) & (p.dst_agent == b), "direct"] = 0.0
            return p
        if self.kind == "limit_tool":
            agent, tool = self.target.split("|")
            uses = p.src_event.map(lambda e: ev_meta.get(e, {}).get("tool") == tool)
            return p[~((p.src_agent == agent) & uses)]
        if self.kind in ("restrict_writes", "isolate_memory", "approval_gate"):
            def keep_res(r: str) -> bool:
                if self.kind == "approval_gate":
                    return not r.startswith("resource:")
                return r != self.target
            new_shared = p.shared_resources.map(lambda lst: [r for r in lst if keep_res(r)])
            lost = new_shared.map(len) < p.shared_resources.map(len)
            p["shared_resources"] = new_shared
            p.loc[lost & (new_shared.map(len) == 0), "resource"] = 0.0
            if self.kind == "approval_gate":
                # An approval gate delays rather than blocks: content still flows if approved.
                # Model as removing the resource term only (semantic/direct echoes remain).
                pass
            return p
        raise ValueError(self.kind)

    # -- which observed events would this control touch? (for collateral) --------
    def touches_event(self, row: Any) -> bool:
        md = row.metadata if isinstance(row.metadata, dict) else {}
        if self.kind == "quarantine_agent":
            return row.actor_id == self.target
        if self.kind == "block_channel":
            a, b = self.target.split("->")
            return row.actor_id == a and row.action_type == "message" and b in (md.get("mentions") or [])
        if self.kind == "limit_tool":
            agent, tool = self.target.split("|")
            return row.actor_id == agent and md.get("tool") == tool
        res = frame_resources(row)
        if self.kind == "approval_gate":
            return any(r.startswith("resource:") for r in res)
        return self.target in res


def generate_candidates(episode, events: pd.DataFrame, max_per_kind: int = 4) -> list[Intervention]:
    """Propose interventions targeted at the structure of one episode."""
    p = episode.pairs
    cands: list[Intervention] = []
    if p.empty:
        return cands
    kept = {(e.src, e.dst) for e in episode.agent_edges}
    pk = p[[(s, d) in kept for s, d in zip(p.src_agent, p.dst_agent)]]
    res_support = Counter(r for lst in pk.shared_resources for r in lst if not r.startswith("room:"))
    for r, _ in res_support.most_common(max_per_kind * 2):
        if r.startswith("memory:"):
            cands.append(Intervention("isolate_memory", r))
        elif r.startswith("resource:"):
            cands.append(Intervention("restrict_writes", r))
    cands = cands[: max_per_kind * 2]
    if any(r.startswith("resource:") for r in res_support):
        cands.append(Intervention("approval_gate", "*external*"))
    # Agents: by out-strength in the episode graph (sum of outgoing edge scores).
    out_strength = Counter()
    for e in episode.agent_edges:
        out_strength[e.src] += e.score
    for a, _ in out_strength.most_common(max_per_kind):
        cands.append(Intervention("quarantine_agent", a))
    for e in sorted(episode.agent_edges, key=lambda e: -e.score * e.support)[:max_per_kind]:
        if e.breakdown.get("direct", 0) > 0 or e.breakdown.get("semantic", 0) > 0:
            cands.append(Intervention("block_channel", f"{e.src}->{e.dst}"))
    # Tools used as sources by the top agents.
    eset = set(episode.event_ids)
    sub = events[events.event_id.isin(eset)]
    tool_use = Counter()
    for r in sub.itertuples(index=False):
        t = (r.metadata or {}).get("tool") if isinstance(r.metadata, dict) else None
        if t and r.actor_id in out_strength:
            tool_use[(r.actor_id, t)] += 1
    for (a, t), _ in tool_use.most_common(2):
        cands.append(Intervention("limit_tool", f"{a}|{t}"))
    return list(dict.fromkeys(cands))
