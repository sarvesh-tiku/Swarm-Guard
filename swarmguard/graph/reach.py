"""Time-respecting reachability over candidate propagation pairs.

A path s -> ... -> u -> v is only valid if the evidence pair used for u -> v
starts *after* u was reached (earliest-arrival semantics). This prevents chat
back-and-forth from making every agent "reach" every other agent and is what
makes source recovery meaningful.
"""
from __future__ import annotations

import bisect
import heapq
from collections import defaultdict

import pandas as pd

from ..data.schemas import to_seconds


class TemporalGraph:
    """Pre-indexed pairs: for each (u, v), src times sorted with suffix-min of dst times."""

    def __init__(self, pairs: pd.DataFrame) -> None:
        self.out: dict[str, dict[str, tuple[list[float], list[float], list[tuple[str, str]]]]] = defaultdict(dict)
        self.nodes: set[str] = set()
        if pairs is None or pairs.empty:
            return
        p = pairs.assign(
            _s=to_seconds(pairs["src_ts"]), _d=to_seconds(pairs["dst_ts"])
        ).sort_values("_s")
        for (u, v), g in p.groupby(["src_agent", "dst_agent"], sort=False):
            s = g["_s"].tolist()
            d = g["_d"].tolist()
            ev = list(zip(g["src_event"], g["dst_event"]))
            # suffix min over dst times with argmin evidence
            suf_d, suf_e = [0.0] * len(d), [("", "")] * len(d)
            best, best_e = float("inf"), ("", "")
            for k in range(len(d) - 1, -1, -1):
                if d[k] < best:
                    best, best_e = d[k], ev[k]
                suf_d[k], suf_e[k] = best, best_e
            self.out[u][v] = (s, suf_d, suf_e)
            self.nodes.update((u, v))

    def earliest_arrival(self, source: str) -> dict[str, tuple[float, str | None, tuple[str, str] | None]]:
        """Return {agent: (arrival_time, predecessor, evidence_pair)} for agents reachable from source."""
        arr: dict[str, tuple[float, str | None, tuple[str, str] | None]] = {source: (float("-inf"), None, None)}
        pq = [(float("-inf"), source)]
        done = set()
        while pq:
            t, u = heapq.heappop(pq)
            if u in done:
                continue
            done.add(u)
            for v, (s, suf_d, suf_e) in self.out.get(u, {}).items():
                k = bisect.bisect_left(s, t)
                if k >= len(s):
                    continue
                cand = suf_d[k]
                if v not in arr or cand < arr[v][0]:
                    arr[v] = (cand, u, suf_e[k])
                    heapq.heappush(pq, (cand, v))
        return arr

    def reachable_pairs(self) -> set[tuple[str, str]]:
        """All (source, target) agent pairs connected by a time-respecting path."""
        out = set()
        for s in self.nodes:
            for t in self.earliest_arrival(s):
                if t != s:
                    out.add((s, t))
        return out

    def path_to(self, source: str, target: str) -> list[tuple[str, str, tuple[str, str]]]:
        arr = self.earliest_arrival(source)
        if target not in arr:
            return []
        path = []
        cur = target
        while cur != source:
            _, pred, ev = arr[cur]
            path.append((pred, cur, ev))
            cur = pred
        return list(reversed(path))
