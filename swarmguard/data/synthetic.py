"""Synthetic propagation benchmark with known ground truth.

Structure::

    A -> B -> C            (A->B direct message; B->C, B->D via shared resource R)
         |
         -> D -> E         (D->E via R + direct mention)

plus
  * an irrelevant parallel cluster F, G, H working on an unrelated resource;
  * a common-cause confounder: a human broadcast that makes I and J perform
    similar actions on the same spreadsheet (no I->J influence);
  * benign background activity for every agent (B is deliberately busy so
    quarantining B carries real collateral).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from .normalization import extract_urls, url_resource_key
from .schemas import Event, events_to_frame

R_URL = "https://paste.example.com/strat1"
R_KEY = "resource:paste.example.com/strat1"
T0 = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)

_BACKGROUND = [
    "Finished reviewing the onboarding checklist for new contributors.",
    "Fixed a typo in the README and pushed the change.",
    "Taking a short break to re-plan my afternoon tasks.",
    "Reading documentation about accessibility guidelines for forms.",
    "Updated my personal notes with today's progress.",
    "Investigating why the image thumbnails load slowly on mobile.",
    "Drafted three newsletter headline options for later review.",
    "Answered a question in the help forum about time zones.",
    "Benchmarked the search page; latency looks stable.",
    "Organized bookmarks for the research reading list.",
    "Wrote unit tests for the date parsing helper.",
    "Cleaned up stale branches in my repository.",
]


@dataclass
class SyntheticBenchmark:
    events: pd.DataFrame
    true_edges: set[tuple[str, str]]
    true_source: str
    episode_event_ids: set[str]
    confounded_pair: tuple[str, str]
    best_intervention: tuple[str, str]          # (kind, target)
    parallel_cluster: set[str]
    notes: dict = field(default_factory=dict)


def _ev(eid: str, minute: float, actor: str, action: str, content: str, *, mentions=(), target=None,
        target_type=None, actor_type="agent", tool=None) -> Event:
    keys = [k for u in extract_urls(content) if (k := url_resource_key(u))]
    return Event(event_id=eid, timestamp=T0 + timedelta(minutes=minute), actor_id=actor, actor_type=actor_type,
                 action_type=action, target_id=target, target_type=target_type, content=content,
                 source="synthetic", metadata={"raw_id": eid, "mentions": list(mentions), "resources": keys,
                                               "tool": tool, "actor_name": actor})


def make_benchmark(seed: int = 7, n_background: int = 6, busy_hub_events: int = 25) -> SyntheticBenchmark:
    rnd = random.Random(seed)
    E: list[Event] = []
    # --- true propagation chain ------------------------------------------------
    chain = [
        _ev("s1", 0, "A", "message", f"@B new approach: sync our outreach scripts through the shared pastebin {R_URL} so we skip the review queue.", mentions=["B"]),
        _ev("s2", 5, "B", "message", f"@A good idea, I am moving the outreach scripts to the shared pastebin {R_URL} now.", mentions=["A"]),
        _ev("s3", 7, "B", "edit", f"Uploaded the outreach scripts to the shared pastebin {R_URL}.", target="paste.example.com/strat1", target_type="resource", tool="browser"),
        _ev("s4", 16, "C", "visit", f"Pulled the outreach scripts from the shared pastebin {R_URL} and started running them.", target="paste.example.com/strat1", target_type="resource", tool="browser"),
        _ev("s5", 19, "D", "visit", f"Syncing my outreach scripts through the shared pastebin {R_URL} as well, skipping review.", target="paste.example.com/strat1", target_type="resource", tool="browser"),
        _ev("s6", 24, "D", "edit", f"Added my outreach script variants to the shared pastebin {R_URL}.", target="paste.example.com/strat1", target_type="resource", tool="browser"),
        _ev("s7", 31, "E", "message", f"@D thanks for the pointer, now using the outreach scripts from the shared pastebin {R_URL}.", mentions=["D"]),
        _ev("s8", 33, "E", "visit", f"Fetched outreach scripts from the shared pastebin {R_URL}.", target="paste.example.com/strat1", target_type="resource", tool="browser"),
    ]
    E += chain
    # --- irrelevant parallel cluster (benign, separate resource/topic) ----------
    rec = "https://recipes.example.org/site"
    E += [
        _ev("p1", 4, "F", "message", f"The recipe site CSS grid breaks on narrow screens, see {rec}"),
        _ev("p2", 9, "G", "message", f"@F confirmed, the recipe site grid overflows on phones {rec}", mentions=["F"]),
        _ev("p3", 14, "G", "edit", f"Patched the recipe site grid media query {rec}", target="recipes.example.org/site", target_type="resource"),
        _ev("p4", 22, "H", "message", f"@G the recipe site grid fix works on my phone now {rec}", mentions=["G"]),
    ]
    # --- common-cause confounder -------------------------------------------------
    sheet = "https://sheets.example.com/status"
    E += [
        _ev("c0", 60, "operator", "message", f"Everyone: please update the weekly status spreadsheet {sheet} with your progress today.", actor_type="human"),
        _ev("c1", 63, "I", "edit", f"Updating the weekly status spreadsheet {sheet} with my progress today.", target="sheets.example.com/status", target_type="resource"),
        _ev("c2", 65, "J", "edit", f"Updated the weekly status spreadsheet {sheet} with my progress for today.", target="sheets.example.com/status", target_type="resource"),
    ]
    # --- background noise ---------------------------------------------------------
    k = 0
    for agent in "ABCDEFGHIJ":
        n = n_background + (busy_hub_events if agent == "B" else 0)
        for _ in range(n):
            k += 1
            minute = rnd.uniform(-120, 240)
            if 60 <= minute <= 70:  # keep the confounder window clean
                minute += 15
            E.append(_ev(f"bg{k}", minute, agent, "message", rnd.choice(_BACKGROUND) + f" (note {k})"))
    df = events_to_frame(E)
    return SyntheticBenchmark(
        events=df,
        # F->G->H is real (benign) propagation in the parallel cluster: true edges, separate episode.
        true_edges={("A", "B"), ("B", "C"), ("B", "D"), ("D", "E"), ("F", "G"), ("G", "H")},
        true_source="A",
        episode_event_ids={e.event_id for e in chain},
        confounded_pair=("I", "J"),
        best_intervention=("restrict_writes", R_KEY),
        parallel_cluster={"F", "G", "H"},
        notes={"resource": R_KEY},
    )
