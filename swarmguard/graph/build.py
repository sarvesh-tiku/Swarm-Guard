"""Heterogeneous temporal graph construction (networkx.MultiDiGraph).

Node ids are namespaced ``<type>:<id>``. Node types: agent, human, message,
memory, goal, tool, resource, session, room.

Observed edges (one per event, keyed by event_id):
  agent -sent-> message -posted_in-> room       (chat)
  message/session -referenced-> resource        (URLs mentioned)
  agent -used-> session                         (computer-use session start)
  agent -wrote-> memory                         (memory consolidation)
  agent -followed-> goal                        (goal assignment)
  agent -edited/created/read-> resource         (wiki edits, page visits)
  agent -used-> tool                            (tool calls)

Derived edges:
  agent -shares_resource_with-> agent           (both touched a non-broadcast resource)
  agent -candidate_propagation-> agent          (from graph.propagation)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import networkx as nx
import pandas as pd

from .propagation import AgentEdge

_ACTION_TO_REL = {
    "edit": "edited", "create": "created", "visit": "read", "memory_read": "read",
    "memory_write": "wrote", "tool_use": "used", "session_start": "used",
    "goal_set": "followed", "message": "sent",
}


def _actor_node(row) -> str:
    kind = "human" if row.actor_type == "human" else "agent"
    return f"{kind}:{row.actor_id}"


def build_heterogeneous_graph(
    events: pd.DataFrame,
    agent_edges: Iterable[AgentEdge] = (),
    message_nodes: bool = True,
) -> nx.MultiDiGraph:
    """Build the observed + derived heterogeneous graph for an event slice.

    Args:
        message_nodes: create one node per message. Disable for very large
            slices; messages then link actor -> room directly.
    """
    G = nx.MultiDiGraph()
    res_touch: dict[str, set[str]] = defaultdict(set)
    for row in events.itertuples(index=False):
        actor = _actor_node(row)
        G.add_node(actor, node_type=actor.split(":", 1)[0], label=row.metadata.get("actor_name", row.actor_id)
                   if isinstance(row.metadata, dict) else row.actor_id)
        md = row.metadata if isinstance(row.metadata, dict) else {}
        attrs = dict(event_id=row.event_id, timestamp=row.timestamp, action_type=row.action_type, source=row.source)
        hub = actor
        if row.action_type == "message" and message_nodes:
            msg = f"message:{row.event_id}"
            G.add_node(msg, node_type="message", label=(row.content or "")[:60], timestamp=row.timestamp)
            G.add_edge(actor, msg, key=row.event_id, rel="sent", **attrs)
            hub = msg
            if row.target_id:
                room = f"room:{row.target_id}"
                G.add_node(room, node_type="room", label=md.get("room_name", row.target_id))
                G.add_edge(msg, room, key=row.event_id + ":room", rel="posted_in", **attrs)
            for m in md.get("mentions") or []:
                G.add_edge(msg, f"agent:{m}", key=row.event_id + f":to:{m}", rel="received", **attrs)
        elif row.target_id is not None and isinstance(row.target_id, str):
            ttype = row.target_type or "resource"
            tnode = f"{ttype}:{row.target_id}"
            G.add_node(tnode, node_type=ttype, label=md.get("target_label", row.target_id))
            G.add_edge(actor, tnode, key=row.event_id, rel=_ACTION_TO_REL.get(row.action_type, "used"), **attrs)
            if ttype in ("session", "memory"):
                hub = tnode
            if ttype in ("resource",):
                res_touch[tnode].add(actor)
        if md.get("tool"):
            tool = f"tool:{md['tool']}"
            G.add_node(tool, node_type="tool", label=md["tool"])
            G.add_edge(actor, tool, key=row.event_id + ":tool", rel="used", **attrs)
        for r in md.get("resources") or []:
            rnode = f"resource:{r}"
            G.add_node(rnode, node_type="resource", label=r)
            G.add_edge(hub, rnode, key=row.event_id + f":ref:{r}", rel="referenced", **attrs)
            res_touch[rnode].add(actor)

    for rnode, actors in res_touch.items():
        agents = sorted(a for a in actors if a.startswith("agent:"))
        if 2 <= len(agents) <= 12:
            for i, a in enumerate(agents):
                for b in agents[i + 1:]:
                    G.add_edge(a, b, key=f"share:{rnode}", rel="shares_resource_with", resource=rnode)

    for e in agent_edges:
        G.add_edge(f"agent:{e.src}", f"agent:{e.dst}", key=f"prop:{e.src}->{e.dst}", rel="candidate_propagation",
                   score=e.score, confidence=e.confidence, breakdown=e.breakdown, support=e.support,
                   evidence_pairs=e.evidence_pairs, explanation=e.explanation)
    return G


def agent_propagation_graph(edges: Iterable[AgentEdge], min_score: float = 0.0) -> nx.DiGraph:
    """Simple weighted DiGraph over agents from candidate propagation edges."""
    D = nx.DiGraph()
    for e in edges:
        if e.score >= min_score:
            D.add_edge(e.src, e.dst, weight=e.score, confidence=e.confidence, edge=e)
    return D
