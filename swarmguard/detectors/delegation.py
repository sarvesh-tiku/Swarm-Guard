"""Emergent delegation (experimental): several agents take up *different* sub-tasks of the
same theme in the episode without an explicit assignment message addressing them."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .base import Detection

_ASSIGN = re.compile(r"\b(?:can you|could you|please (?:take|handle|do)|you take|you handle|assign(?:ed|ing)? to|"
                     r"i'll take|i will take|i'll handle|i'll do|i'm taking|claim(?:ing)?)\b", re.I)


def detect_emergent_delegation(episode, events: pd.DataFrame) -> Detection:
    from sklearn.feature_extraction.text import TfidfVectorizer

    sub = events[events["event_id"].isin(set(episode.event_ids))]
    starts = sub[sub["action_type"] == "session_start"].drop_duplicates("actor_id")
    msgs = sub[sub["action_type"] == "message"]
    explicit = msgs[msgs["content"].fillna("").str.contains(_ASSIGN)]
    assigned = set()
    for r in explicit.itertuples(index=False):
        assigned.add(r.actor_id)  # self-claim
        assigned.update((r.metadata or {}).get("mentions") or [])
    cand = starts[~starts["actor_id"].isin(assigned)]
    triggered, why, details = False, "Fewer than 3 agents started distinct unassigned sessions.", {}
    if len(cand) >= 3:
        texts = cand["content"].fillna("").tolist()
        try:
            X = TfidfVectorizer(stop_words="english").fit_transform(texts)
            S = (X @ X.T).toarray()
            off = S[~np.eye(len(texts), dtype=bool)]
            mean_sim = float(off.mean())
        except ValueError:
            mean_sim = 1.0
        # Divided labour: goals are distinct from each other (low mutual similarity).
        triggered = mean_sim < 0.25
        why = (f"{len(cand)} agents started sessions with mutually distinct goals (mean TF-IDF sim {mean_sim:.2f}) "
               f"inside one coordination episode, without an explicit assignment/claim message naming them.")
        details = {"session_goals": dict(zip(cand["actor_id"], [t[:160] for t in texts])), "mean_goal_similarity": mean_sim}
    return Detection(
        name="emergent_delegation", triggered=triggered, confidence="low", why=why,
        does_not_prove=("Experimental heuristic. Assignment may have happened in an unobserved channel, in an "
                        "earlier window, or via standing roles; distinct goals may simply be unrelated work."),
        affected_agents=cand["actor_id"].tolist() if triggered else [], evidence_event_ids=cand["event_id"].tolist()[:10],
        details=details,
    )
