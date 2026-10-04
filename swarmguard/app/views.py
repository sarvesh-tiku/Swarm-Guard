"""Episode views: the spread map, who-used-what, and evidence pairs.

Design goals: every chart answers one question, labels never sit on top of each
other, and every piece of evidence shows the passage that matters, with what the
two sides share highlighted.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd
import plotly.graph_objects as go

from swarmguard.app import style as S

ROW_H = 40  # px per agent row


def common_words(events: pd.DataFrame, share: float = 0.015) -> set[str]:
    """Words in more than `share` of log lines, plus agent-name fragments: not distinctive evidence."""
    df = Counter()
    for t in events["content"].fillna("").astype(str):
        df.update({w.lower() for w in S._TOKEN.findall(t) if "://" not in w})
    n = max(1, len(events))
    common = {w for w, c in df.items() if c / n > share}
    names = {part.lower() for a in events["actor_id"].astype(str).unique() for part in a.replace("@", " ").split()}
    return common | names


def _arrival(ep, events: pd.DataFrame):
    """(order of agents, arrival time per agent, tree hops) from the propagation check."""
    sub = events[events.event_id.isin(set(ep.event_ids))]
    first = sub.groupby("actor_id")["timestamp"].min()
    prop = next(d for d in ep.detections if d.name == "information_propagation")
    hops = prop.details.get("hops", []) or []
    ev_t = sub.set_index("event_id")["timestamp"]
    arrival = {}
    root = prop.details.get("candidate_source")
    if root in first:
        arrival[root] = first[root]
    for h in hops:
        t = ev_t.get(h["evidence"][1], first.get(h["to"]))
        arrival.setdefault(h["to"], t)
    for a, t in first.items():
        arrival.setdefault(a, t)
    order = sorted(arrival, key=lambda a: (a not in {root, *[h["to"] for h in hops]}, arrival[a]))
    return order, arrival, hops, root, sub


def _why_short(e) -> str:
    bd = e.breakdown
    parts = []
    if bd.get("direct", 0) > 0:
        parts.append("reply")
    if bd.get("resource", 0) > 0:
        parts.append("page")
    if bd.get("semantic", 0) > 0:
        parts.append("wording")
    return ", ".join(parts) or "timing"


def spread_map(ep, events: pd.DataFrame, removed: set[tuple[str, str]] | None = None) -> tuple[go.Figure, str]:
    """One row per agent, ordered by when they joined. Arrows: who may have picked it up from whom."""
    order, arrival, hops, root, sub = _arrival(ep, events)
    y = {a: i for i, a in enumerate(order)}
    edge_by = {(e.src, e.dst): e for e in ep.agent_edges}
    ev_t = sub.set_index("event_id")["timestamp"]
    fig = go.Figure()
    # each agent's own actions in the episode: small grey ticks on its row
    fig.add_scatter(x=sub.timestamp, y=[y.get(a) for a in sub.actor_id], mode="markers",
                    marker=dict(size=9, color=S.RULE, symbol="line-ns-open", line=dict(width=1.6, color="#a3a397")),
                    hovertext=[f"{a}: {S.esc(str(c)[:140])}" for a, c in zip(sub.actor_id, sub.content)],
                    hoverinfo="text", name="each logged action", showlegend=True)
    # arrows along the earliest-arrival tree
    for h in hops:
        a_t = ev_t.get(h["evidence"][0], arrival.get(h["from"]))
        b_t = ev_t.get(h["evidence"][1], arrival.get(h["to"]))
        gone = removed is not None and (h["from"], h["to"]) in removed
        e = edge_by.get((h["from"], h["to"]))
        width = {"strong": 2.6, "moderate": 1.8, "weak": 1.1}.get(h["confidence"], 1.2)
        fig.add_annotation(x=b_t, y=y[h["to"]], ax=a_t, ay=y[h["from"]], xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2 if gone else width,
                           arrowcolor=S.RULE if gone else S.AMBER, standoff=6, startstandoff=4)
        if gone:
            fig.add_annotation(x=a_t + (b_t - a_t) / 2, y=(y[h["from"]] + y[h["to"]]) / 2, text="✕", showarrow=False,
                               font=dict(size=15, color=S.INK), bgcolor=S.CREAM)
        # labels live in their own column right of the timeline, so arrows never cross text
        label = f"from {h['from'][:22]}" + (f"  ({_why_short(e)})" if e is not None else "")
        fig.add_annotation(x=1.0, xref="paper", y=y[h["to"]], text=label, showarrow=False, xanchor="left", xshift=8,
                           font=dict(family=S.SANS, size=10.5, color=S.RULE if gone else S.AMBER_INK))
    # arrival dots, labelled on the row axis
    fig.add_scatter(x=[arrival[a] for a in order], y=[y[a] for a in order], mode="markers",
                    marker=dict(size=[14 if a == root else 10 for a in order],
                                color=[S.SLATE if a in {root, *[h["to"] for h in hops]} else S.CREAM for a in order],
                                line=dict(width=1.6, color=S.SLATE)),
                    hovertext=[f"{a}: first involved {arrival[a]:%H:%M:%S}" for a in order], hoverinfo="text",
                    name="when the agent first took part")
    fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=S.AMBER, width=2),
                    name="may have picked it up from (inferred)")
    if removed:
        fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=S.RULE, width=2), name="cut by this option ✕")
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(order))),
                     ticktext=[("▸ " if a == root else "") + a for a in order], autorange="reversed",
                     tickfont=dict(family=S.SERIF, size=13, color=S.INK), showgrid=True, gridcolor=S.HAIR)
    t0, t1 = sub.timestamp.min(), sub.timestamp.max()
    pad = (t1 - t0) * 0.04
    # room on the right for the row labels
    fig.update_xaxes(showgrid=False, tickformat="%H:%M", title=None, range=[t0 - pad, t1 + pad])
    S.style_fig(fig, max(320, 110 + ROW_H * len(order)))
    fig.update_layout(margin=dict(l=10, r=250, t=70, b=30))
    cap = (f"Rows are agents in the order they joined; ▸ marks the earliest observed carrier ({root}). "
           f"Amber arrows follow the {len(hops)} time-ordered links SwarmGuard considers most likely; the column on "
           "the right says who each agent may have picked it up from, and what the two share (reply, same page, "
           "similar wording). Grey ticks are "
           "each agent's other actions. Hollow dots took part but are not on a chain.")
    return fig, cap


def who_used_what(ep, events: pd.DataFrame, max_resources: int = 8) -> tuple[go.Figure, str] | None:
    """Agents on the left, shared pages/sites on the right; a line means the agent really touched it."""
    sub = events[events.event_id.isin(set(ep.event_ids))]
    touch = Counter()
    for r in sub.itertuples(index=False):
        res = list((r.metadata or {}).get("resources") or [])
        if isinstance(r.target_id, str) and r.target_type == "resource":
            res.append(r.target_id)
        for x in set(res):
            touch[(r.actor_id, x)] += 1
    by_res = Counter()
    for (a, x), n in touch.items():
        by_res[x] += 1
    res = [x for x, n in by_res.most_common(max_resources) if n >= 2] or [x for x, _ in by_res.most_common(3)]
    if not res:
        return None
    agents = sorted({a for (a, x) in touch if x in res}, key=lambda a: -sum(touch[(a, x)] for x in res))
    ya = {a: i for i, a in enumerate(agents)}
    span = max(len(agents) - 1, 1)
    yr = {x: (i + 0.5) * span / max(len(res), 1) for i, x in enumerate(res)}
    fig = go.Figure()
    for (a, x), n in touch.items():
        if x in yr and a in ya:
            fig.add_scatter(x=[0, 1], y=[ya[a], yr[x]], mode="lines", hoverinfo="skip", showlegend=False,
                            line=dict(color=S.GREEN, width=min(1 + n * 0.6, 4)), opacity=0.55)
    fig.add_scatter(x=[0] * len(agents), y=[ya[a] for a in agents], mode="markers+text", text=agents,
                    textposition="middle left", textfont=dict(family=S.SERIF, size=13, color=S.INK),
                    marker=dict(size=9, color=S.SLATE), hoverinfo="text", showlegend=False)
    lab = [f"{S.resource_name(x)[:40]}  · {by_res[x]} agents" for x in res]
    fig.add_scatter(x=[1] * len(res), y=[yr[x] for x in res], mode="markers+text", text=lab,
                    textposition="middle right", textfont=dict(family=S.SANS, size=11, color=S.INK),
                    marker=dict(size=11, symbol="square-open", color=S.INK, line=dict(width=1.6)),
                    hoverinfo="text", showlegend=False)
    fig.update_xaxes(visible=False, range=[-0.85, 2.35])
    fig.update_yaxes(visible=False, autorange="reversed")
    S.style_fig(fig, max(260, 70 + 26 * len(agents)), legend=False)
    cap = ("Left: agents. Right: the shared pages and sites they used. A green line means the logs show that agent "
           "touching that page (thicker = more often). A page many lines converge on is a natural place to intervene.")
    return fig, cap


def _reasons(pair_row, e) -> list[str]:
    out = []
    if pair_row is not None:
        if pair_row.get("direct", 0) > 0:
            out.append("the later agent replied to, or was addressed by, the earlier one")
        shared = [S.resource_name(r) for r in (pair_row.get("shared_resources") or []) if not str(r).startswith("room:")]
        if shared:
            out.append("both used " + ", ".join(f"<i>{S.esc(x)}</i>" for x in shared[:2]))
        if pair_row.get("semantic", 0) > 0:
            out.append("similar wording (highlighted)")
        dt = pair_row.get("dt_minutes")
        if dt is not None:
            out.append(f"{dt:.0f} min apart" if dt >= 1 else "under a minute apart")
    elif e is not None:
        out.append(S.esc(S.clean_text(e.explanation)))
    return out or ["close in time"]


def evidence_pair(a_id: str, b_id: str, events_idx: pd.DataFrame, pairs_idx: dict, edge=None, title: str = "") -> str:
    if a_id not in events_idx.index or b_id not in events_idx.index:
        return ""
    a, b = events_idx.loc[a_id], events_idx.loc[b_id]
    row = pairs_idx.get((a_id, b_id))
    reasons = _reasons(row, edge)
    extra = []
    if row is not None:  # make the shared page/URL itself highlightable
        for r in row.get("shared_resources") or []:
            r = str(r).removeprefix("resource:")
            if r.startswith("wiki:"):
                extra.append(r.split("~", 1)[-1])
            elif not r.startswith(("room:", "memory:")):
                extra.append(r.split("/")[0] if len(r) > 40 else r)
    meta = lambda r: f"{r.timestamp:%b %d %H:%M:%S} UTC · " + S.ACTION_VERB.get(r.action_type, r.action_type)
    return S.pair_html(meta(a), a.actor_id, str(a.content or ""), meta(b), b.actor_id, str(b.content or ""),
                       reasons, extra, title)


def pairs_index(pairs: pd.DataFrame, event_ids: set[str]) -> dict:
    """{(src_event, dst_event): row-dict} restricted to one episode (fast lookups)."""
    p = pairs[pairs.src_event.isin(event_ids) & pairs.dst_event.isin(event_ids)]
    return {(r["src_event"], r["dst_event"]): r for r in p.to_dict("records")}


# ------------------------------------------------------------------ the takeaway
def episode_shape(ep) -> tuple[str, str, dict]:
    """Classify an episode from measured link shares. Returns (shape, plain sentence, measurements)."""
    edges = ep.agent_edges
    n = max(1, len(edges))
    via_page = sum(1 for e in edges if e.breakdown.get("resource", 0) > 0) / n
    replies = sum(1 for e in edges if e.breakdown.get("direct", 0) > 0) / n
    det = {d.name: d for d in ep.detections}
    chain = det["information_propagation"].triggered
    channel = det["external_channel_coordination"]
    top_res = (ep.stats.get("top_resources") or [[None, 0]])[0]
    # share of the episode's agents that touched the most-used page/site
    res_agents = 0
    if channel.details.get("channels"):
        res_agents = len(channel.details["channels"][0]["agents"])
    board_share = res_agents / max(1, len(ep.agents))
    m = {"via_page": via_page, "replies": replies, "chain": chain, "board_share": board_share,
         "board": channel.details["channels"][0]["resource"] if channel.details.get("channels") else top_res[0]}
    if chain:
        src = det["information_propagation"].details.get("candidate_source")
        return ("chain", f"a <b>chain</b>: a pattern passed hop by hop, starting with the earliest observed carrier "
                         f"<b>{S.esc(src)}</b>", m)
    if channel.triggered and board_share >= 0.5:
        also = "; much of the exchange around it is also direct replies" if replies >= 0.5 else ""
        return ("board", f"a <b>broadcast board</b>: {res_agents} of {len(ep.agents)} agents wrote to or used "
                         f"<i>{S.esc(S.resource_name(m['board']))}</i>, and most links run through it rather than "
                         f"from one agent to the next{also}", m)
    if replies >= 0.5:
        return ("conversation", "a <b>conversation</b>: most links are direct replies between agents", m)
    return ("cluster", "a <b>loose cluster</b>: agents working on related things at the same time, without one "
                       "dominant channel or chain", m)


def confidence_verdict(ep, shape: str = "", m: dict | None = None) -> tuple[str, str]:
    edges = ep.agent_edges
    n = max(1, len(edges))
    good = sum(1 for e in edges if e.confidence in ("moderate", "strong")) / n
    alt = sum(1 for e in edges if any(a.startswith(("common cause", "possibly indirect", "shared upstream", "same origin"))
                                      for a in e.alternatives)) / n
    prop = next(d for d in ep.detections if d.name == "information_propagation")
    ambiguous = len(prop.details.get("competing_sources") or [])
    if good >= 0.5 and alt < 0.3:
        v = "solid"
    elif good < 0.25 or alt >= 0.6:
        v = "thin"
    else:
        v = "mixed"
    if shape == "board" and m:
        # Two different claims: "they coordinated through this page" is read straight off the logs;
        # "who influenced whom on it" is the inferred, pairwise part.
        finding = (f"That they coordinated through the page is <b>solid</b>: it is read straight off the logs "
                   f"({m['board_share']:.0%} of the agents wrote to or used it). <b>Who influenced whom</b> on it is "
                   f"<b>{v}</b>.")
        how = (f"the board claim needs no inference. For the pairwise claim, {good:.0%} of the {len(edges)} links are "
               f"plausible or well supported and {alt:.0%} have a competing explanation, usually that everyone may "
               "simply have read the page itself rather than the agent before them. "
               "Solid = at least half the links plausible+ and under 30% with competing explanations; thin = under a "
               "quarter plausible+ or most links have one.")
        return finding, how
    how = (f"{good:.0%} of the {len(edges)} links are plausible or well supported (two or more independent kinds of "
           f"evidence, repeated); {alt:.0%} have a competing explanation SwarmGuard could not rule out"
           + (f"; {ambiguous} other agent(s) could equally be the source" if ambiguous else "") + ". "
           "Solid = at least half the links are plausible+ and under 30% have competing explanations; "
           "thin = under a quarter plausible+ or most links have one.")
    return f"The evidence is <b>{v}</b>.", how


def takeaway_html(ep, events: pd.DataFrame, ranked: list | None) -> str:
    sub = events[events.event_id.isin(set(ep.event_ids))]
    shape, shape_txt, m = episode_shape(ep)
    verdict, how_sure = confidence_verdict(ep, shape, m)
    mins = round(ep.stats["duration_hours"] * 60)
    rows = []

    def row(label, finding, how):
        rows.append(f'<div class="tk-row"><div class="tk-lbl">{label}</div><div class="tk-body"><p class="tk-find">{finding}</p>'
                    f'<p class="tk-how"><b>How we gauge it:</b> {how}</p></div></div>')

    top = (ep.stats.get("top_resources") or [[None, 0]])[0][0]
    row("What happened",
        f"{len(ep.agents)} agents took {len(sub)} logged actions in {mins} minutes"
        + (f", centred on <i>{S.esc(S.resource_name(top))}</i>." if top else "."),
        "counted directly from the log lines in this window; nothing inferred.")
    row("What shape it is", shape_txt + ".",
        f"{m['via_page']:.0%} of links run through a shared page or site, {m['replies']:.0%} are direct replies, "
        f"and the hop-by-hop spread check {'fired' if m['chain'] else 'did not fire'}"
        + (f"; the busiest page was used by {m['board_share']:.0%} of the agents" if m['board_share'] else "") + ".")
    row("How sure we are", verdict, how_sure)
    if ranked:
        r0 = ranked[0]
        undo = "easy to undo" if r0.reversibility >= 0.8 else "partly reversible" if r0.reversibility >= 0.6 else "costly to undo"
        alt = next((r for r in ranked if r.intervention.kind == "quarantine_agent"), None)
        cmp_txt = (f" Pausing the most connected agent instead would cut {alt.paths_removed:.0%} while touching "
                   f"{alt.collateral:.1%}." if alt and alt is not r0 else "")
        row("Where to step in",
            f"<i>{S.esc(S.intervention_label(r0))}</i>: removes {r0.paths_removed:.0%} of the inferred spread, touches "
            f"{r0.collateral:.1%} of unrelated activity, {undo}.{cmp_txt}",
            "every option is applied to a copy of the inferred graph; we count the agent-to-agent paths that disappear "
            "and the actions outside this episode the same control would also hit, then rank by spread removed minus "
            "penalties for side-effects, thin evidence and being hard to undo.")
    hr = ep.stats.get("historical_response")
    if hr and hr.get("hours_from_episode_end_to_first_deletion") is not None:
        row("What really happened next",
            f"The operator deleted {hr['pages_deleted']} of {hr['episode_pages']} pages, the first "
            f"{hr['hours_from_episode_end_to_first_deletion']} hours after the episode; {hr['writes_after_episode_before_deletion']} "
            "more writes landed first.",
            "read from the real deletion log in the data, not inferred.")
    row("What it does not tell us",
        "Whether this behaviour was harmful, or that any agent caused another's action.",
        "SwarmGuard measures structure and evidence, not intent or harm; read the quoted evidence below and decide.")
    return f'<div class="tk"><div class="tk-head">What we learned from this episode</div>{"".join(rows)}</div>'
