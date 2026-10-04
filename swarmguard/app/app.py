"""SwarmGuard Streamlit UI.

    streamlit run swarmguard/app/app.py

Five tabs, written to be read top to bottom: Summary · Episodes · Investigate ·
Intervene · How it works. Nothing here controls a live swarm: "simulate" runs a
structural counterfactual on the inferred graph.
"""
from __future__ import annotations

import faulthandler
import json
import signal
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from swarmguard.analysis.pipeline import AnalysisResult, PipelineConfig, run_pipeline
from swarmguard.analysis.reports import CAVEATS, build_report, render_markdown
from swarmguard.app import blog
from swarmguard.app import style as S
from swarmguard.data.hf_access import HFAccessError
from swarmguard.graph.propagation import PropagationConfig, edges_frame
from swarmguard.interventions.counterfactual import DISCLAIMER
from swarmguard.interventions.scoring import ScoringConfig

st.set_page_config(page_title="SwarmGuard", layout="centered", page_icon="🛡️", initial_sidebar_state="expanded")
try:  # `kill -USR1 <pid>` dumps every thread's Python stack to the server log (debugging hangs)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
except (ValueError, AttributeError):
    pass
st.markdown(S.CSS, unsafe_allow_html=True)

WIKI_DIR = Path(__file__).resolve().parents[2] / "datasets" / "german_wiki"
CONF_WIDTH = {"strong": 3.2, "moderate": 2.0, "weak": 1.0}
GROUP = {"message": "talk", "memory_write": "memory", "memory_read": "memory", "goal_set": "memory"}
GROUP_SYMBOL = {"talk": "circle", "action": "square", "memory": "diamond"}

PRESETS = {
    "AI Village: the Village Hub day": dict(source="ai", day=date(2026, 7, 6), hours=(15, 24)),
    "German wiki swarm (May–July 2026)": dict(source="wiki"),
    "Synthetic test (known right answer)": dict(source="synthetic"),
    "Custom…": dict(source="custom"),
}


@st.cache_resource(show_spinner="Reading the logs and reconstructing what spread where… (about a minute the first time)")
def analyze(source: str, params_json: str) -> AnalysisResult:
    p = json.loads(params_json)
    cfg = PipelineConfig(
        propagation=PropagationConfig(window_minutes=p["window"], w_time=p["w_time"], w_semantic=p["w_semantic"],
                                      w_resource=p["w_resource"], w_direct=p["w_direct"], w_goal=p["w_goal"],
                                      w_action=p["w_action"]),
        scoring=ScoringConfig(alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"]), encoder=p["encoder"])
    if source == "synthetic":
        from swarmguard.data.synthetic import make_benchmark

        return run_pipeline(make_benchmark().events, cfg, context={"dataset": "Synthetic test with a known answer"},
                            progress=False)
    if source == "ai":
        from swarmguard.data.ai_village import load_window

        sl = load_window(p["start"], p["end"], max_events=p["max_events"], mode=p["hf_mode"])
        sl.context["dataset"] = "AI Village"
        return run_pipeline(sl.events, cfg, context=sl.context, progress=False)
    from swarmguard.data.german_wiki import load_dir

    sl = load_dir(p["wiki_dir"], max_events=p["max_events"])
    sl.context["dataset"] = "German wiki agent swarm"
    return run_pipeline(sl.events, cfg, context=sl.context, progress=False)


# ------------------------------------------------------------------ sidebar: what to look at
st.sidebar.markdown("## SwarmGuard")
st.sidebar.markdown('<p class="sg-dek" style="font-size:1rem">Which logs should we read?</p>', unsafe_allow_html=True)
preset = st.sidebar.radio("Data", list(PRESETS), label_visibility="collapsed")
pr = PRESETS[preset]
params: dict = {"hf_mode": "download"}
src = pr["source"]
if src == "custom":
    src = {"AI Village (any day)": "ai", "German wiki files": "wiki"}[
        st.sidebar.selectbox("Source", ["AI Village (any day)", "German wiki files"])]
    if src == "ai":
        d = st.sidebar.date_input("Day (UTC)", value=date(2026, 8, 13), min_value=date(2025, 4, 2))
        h = st.sidebar.slider("Hours (UTC)", 0, 24, (15, 24))
        pr = dict(day=d, hours=h)
        params["hf_mode"] = st.sidebar.selectbox("Load mode", ["download", "stream"],
                                                 help="download: cached copy, fast after the first run. "
                                                      "stream: read from Hugging Face each time (slow).")
    else:
        params["wiki_dir"] = st.sidebar.text_input("Folder with the wiki export", str(WIKI_DIR))
if src == "ai":
    start = datetime.combine(pr["day"], time(0), tzinfo=timezone.utc) + timedelta(hours=pr["hours"][0])
    params.update(start=str(start), end=str(start + timedelta(hours=pr["hours"][1] - pr["hours"][0])))
if src == "wiki":
    params.setdefault("wiki_dir", str(WIKI_DIR))
    if not Path(params["wiki_dir"]).exists():
        st.info("Put the German wiki export in `datasets/german_wiki/` (or point Custom at its folder).")
        st.stop()
params["max_events"] = 100_000 if src == "wiki" else 5000

with st.sidebar.expander("Advanced: how links are scored"):
    st.caption("How much each kind of evidence counts when SwarmGuard decides whether one agent's action "
               "may have led to another's.")
    params["window"] = st.slider("Look back this far (minutes)", 10, 360, 120, 10)
    for k, v, lbl in [("w_time", .15, "Close in time"), ("w_semantic", .30, "Similar wording"),
                      ("w_resource", .25, "Same page / site / file"), ("w_direct", .15, "Addressed each other"),
                      ("w_goal", .05, "Goal overlap"), ("w_action", .10, "Same kind of action")]:
        params[k] = st.slider(lbl, 0.0, 1.0, v, 0.05)
    params["encoder"] = st.selectbox("Text similarity model", ["auto", "minilm", "tfidf"],
                                     help="auto = MiniLM sentence embeddings if installed, else TF-IDF")
with st.sidebar.expander("Advanced: how interventions are ranked"):
    st.caption("score = share of spread removed − α·side-effects − β·uncertainty − γ·hard-to-undo")
    params["alpha"] = st.slider("α  penalty for side-effects", 0.0, 3.0, 1.5, 0.1)
    params["beta"] = st.slider("β  penalty for thin evidence", 0.0, 1.0, 0.25, 0.05)
    params["gamma"] = st.slider("γ  penalty for being hard to undo", 0.0, 1.0, 0.3, 0.05)

try:
    R = analyze(src, json.dumps(params, sort_keys=True, default=str))
except HFAccessError as e:
    st.error(str(e))
    st.stop()
except Exception as e:  # adapter errors as one readable line
    st.error(f"{type(e).__name__}: {e}")
    st.stop()

ev_idx = R.events.set_index("event_id", drop=False)
ids = [e.episode_id for e in R.episodes]
if st.session_state.get("episode") not in ids:
    st.session_state.episode = ids[0] if ids else None


def quote(eid: str, n: int = 420) -> str:
    if eid not in ev_idx.index:
        return ""
    r = ev_idx.loc[eid]
    md = r.metadata if isinstance(r.metadata, dict) else {}
    text = str(r.content or "").strip() or "(no text — " + S.ACTION_VERB.get(r.action_type, r.action_type) + ")"
    where = S.resource_name(r.target_id) if isinstance(r.target_id, str) and r.target_type == "resource" else ""
    return S.quote_html(r.timestamp.strftime("%b %d %H:%M:%S UTC"), r.actor_id,
                        S.ACTION_VERB.get(r.action_type, r.action_type) + (f" {where}" if where else ""),
                        text[:n] + ("…" if len(text) > n else ""), f"{md.get('raw_table')}:{str(md.get('raw_id'))[:28]}")


def episode_graph(ep, edges, removed: set[tuple[str, str]] | None = None, max_resources: int = 6) -> go.Figure:
    G = nx.DiGraph()
    for e in edges:
        G.add_edge(e.src, e.dst)
    touches = Counter()
    sub = R.events[R.events.event_id.isin(set(ep.event_ids))]
    for r in sub.itertuples(index=False):
        res = list((r.metadata or {}).get("resources") or [])
        if isinstance(r.target_id, str) and r.target_type == "resource":
            res.append(r.target_id)
        for x in res:
            touches[(r.actor_id, x)] += 1
    top_res = [x for x, _ in Counter(k[1] for k in touches.elements()).most_common(max_resources)]
    for (a, x), _ in touches.items():
        if x in top_res and a in ep.agents:
            G.add_edge(a, "R::" + x, kind="touch")
    for a in ep.agents:
        G.add_node(a)
    pos = nx.spring_layout(G, seed=7, k=1.3 / max(1, len(G) ** 0.5))
    fig = go.Figure()
    xs, ys = [], []
    for u, v, d in G.edges(data=True):
        if d.get("kind"):
            xs += [pos[u][0], pos[v][0], None]
            ys += [pos[u][1], pos[v][1], None]
    fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=S.GREEN, width=1, dash="dot"), hoverinfo="skip",
                    name="observed: agent touched this page/site", opacity=0.7)
    for e in edges:
        gone = removed is not None and (e.src, e.dst) in removed
        fig.add_annotation(x=pos[e.dst][0], y=pos[e.dst][1], ax=pos[e.src][0], ay=pos[e.src][1], xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1,
                           arrowwidth=1.2 if gone else CONF_WIDTH[e.confidence],
                           arrowcolor=S.RULE if gone else S.AMBER, opacity=0.9 if gone else 0.8, standoff=9, startstandoff=9)
    fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=S.AMBER, width=2.5),
                    name="inferred: may have spread A → B (thicker = better supported)")
    if removed:
        fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=S.RULE, width=2),
                        name="removed by this intervention")
    agents = [n for n in G if not str(n).startswith("R::")]
    res = [n for n in G if str(n).startswith("R::")]
    fig.add_scatter(x=[pos[n][0] for n in agents], y=[pos[n][1] for n in agents], mode="markers+text",
                    marker=dict(size=13, color=S.SLATE, line=dict(width=2, color=S.CREAM)), text=agents,
                    textposition="top center", textfont=dict(size=11, color=S.INK), name="agent", hoverinfo="text")
    if res:
        lab = [S.resource_name(n[3:])[:34] for n in res]
        fig.add_scatter(x=[pos[n][0] for n in res], y=[pos[n][1] for n in res], mode="markers+text",
                        marker=dict(size=12, symbol="square-open", color=S.INK, line=dict(width=1.6)),
                        text=lab, textposition="bottom center", textfont=dict(size=11, color=S.DULL, family=S.SANS),
                        name="shared page / site", hovertext=lab, hoverinfo="text")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return S.style_fig(fig, 540)


def _span(a, b) -> str:
    return f"{a:%b %d %Y}" if a.date() == b.date() else f"{a:%b %d} – {b:%b %d %Y}"


# ------------------------------------------------------------------ header
ctx = R.context
n_agents = R.events.loc[R.events.actor_type == "agent", "actor_id"].nunique()
st.markdown("# SwarmGuard")
st.markdown('<p class="sg-dek">Reconstructs how behaviour spreads between AI agents and the pages, sites and memory '
            'they share, then points to the smallest, most reversible place an operator could step in.</p>',
            unsafe_allow_html=True)
st.markdown(f'<div class="sg-byline">{S.esc(ctx.get("dataset", ""))}<span class="sep">·</span>'
            f'{_span(R.events.timestamp.min(), R.events.timestamp.max())}<span class="sep">·</span>'
            f'{len(R.events):,} logged actions<span class="sep">·</span>{n_agents:,} agents</div>', unsafe_allow_html=True)
st.markdown(S.key_html(), unsafe_allow_html=True)

t_sum, t_eps, t_inv, t_int, t_how, t_blog = st.tabs(["Summary", "Episodes", "Investigate", "Intervene", "How it works", "Blog"])

# ------------------------------------------------------------------ Summary
with t_sum:
    if R.episodes:
        e0 = R.episodes[0]
        top0 = R.interventions.get(e0.episode_id, [])
        lede = (f"In this window, <b>{n_agents:,} agents</b> took <b>{len(R.events):,} logged actions</b>. "
                f"SwarmGuard grouped the activity where agents appear to be influencing each other into "
                f"<b>{len(R.episodes)} episodes</b>. The most prominent: {S.esc(S.episode_sentence(e0, R.events))}")
        if top0:
            lede += (f" The smallest effective place to step in would be to <i>{S.esc(S.intervention_label(top0[0]).lower())}</i>, "
                     f"which removes {top0[0].paths_removed:.0%} of the inferred spread while touching "
                     f"{top0[0].collateral:.1%} of unrelated activity.")
        st.markdown(f'<p class="sg-lede">{lede}</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="sg-lede">No episodes found: in this window, no group of agents shows enough linked '
                    'activity to review.</p>', unsafe_allow_html=True)

    c = st.columns(4)
    c[0].metric("Logged actions", f"{len(R.events):,}")
    c[1].metric("Agents", f"{n_agents:,}")
    c[2].metric("Possible links", f"{len(R.agent_edges):,}",
                help="Pairs of agents where one's action may have influenced the other's. Inferred, not proven.")
    c[3].metric("Episodes", len(R.episodes), help="Bursts of linked activity worth reviewing.")

    tl = R.events.assign(bucket=R.events.timestamp.dt.floor("10min" if src != "wiki" else "6h"))
    agg = tl.groupby("bucket").size()
    fig = go.Figure(go.Bar(x=agg.index, y=agg.values, marker_color=S.SLATE, marker_line_width=0,
                           hovertemplate="%{x|%b %d %H:%M}: %{y} actions<extra></extra>", name="actions"))
    labelled: list = []
    span_gap = (R.events.timestamp.max() - R.events.timestamp.min()) / 12
    for ep in R.episodes[:6]:
        # label a band only if it is not crowded by an already-labelled one
        show = all(abs(ep.start - t) > span_gap for t in labelled)
        if show:
            labelled.append(ep.start)
        label = dict(annotation_text=ep.episode_id, annotation_position="top left",
                     annotation_font=dict(family=S.SANS, size=10, color=S.AMBER_INK)) if show else {}
        fig.add_vrect(x0=ep.start, x1=ep.end, fillcolor=S.AMBER, opacity=0.12, line_width=0, **label)
    fig.update_layout(title=f"Logged actions per {'10 minutes' if src != 'wiki' else '6 hours'} "
                            "(amber bands: the top episodes)", bargap=0.15)
    st.plotly_chart(S.style_fig(fig, 280, legend=False), use_container_width=True, config={"displayModeBar": False})

    notes = []
    if ctx.get("active_village_goals"):
        notes.append("<b>Shared goals in force:</b> " + "; ".join(f"<i>{S.esc(g)}</i>" for g in ctx["active_village_goals"])
                     + ". Agents acting on the same goal can look linked without influencing each other.")
    if ctx.get("observability"):
        notes.append(f"<b>What we can see:</b> {S.esc(ctx['observability'])}.")
    if ctx.get("identity_note"):
        notes.append(f"<b>Who is who:</b> {S.esc(ctx['identity_note'])}. Two names may be one operator.")
    if ctx.get("coverage_gap_sites"):
        notes.append(f"<b>Coverage:</b> {ctx.get('coverage_sites')} sites collected; {ctx['coverage_gap_sites']} have known gaps.")
    for d_, t in (ctx.get("nearby_scaffolding_changes") or [])[:3]:
        notes.append(f"<b>Scaffolding change {S.esc(d_)}:</b> {S.esc(t[:180])}")
    if notes:
        st.markdown("### Before you read the episodes")
        for n_ in notes:
            st.markdown(f'<p class="sg-note">{n_}</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ Episodes
with t_eps:
    st.markdown('<p class="sg-lede">An <b>episode</b> is a short window in which several agents\' actions appear linked: '
                'same page, similar wording, replies, close timing. Most are ordinary collaboration; the point is to '
                'know where to look.</p>', unsafe_allow_html=True)
    for ep in R.episodes[:20]:
        with st.container(border=True):
            a, b = st.columns([5, 1])
            trig = [d for d in ep.detections if d.triggered]
            a.markdown(f'<span class="sg-label">{ep.episode_id} · {ep.start:%b %d, %H:%M} UTC</span>', unsafe_allow_html=True)
            a.markdown(f"**{S.esc(S.episode_sentence(ep, R.events))}**")
            pills = "".join(f'<span class="sg-pill on">{S.esc(S.DETECTOR_PILL[d.name])}</span>'
                            for d in trig) or '<span class="sg-pill">no checks raised</span>'
            a.markdown(pills, unsafe_allow_html=True)
            a.caption("Agents: " + ", ".join(ep.agents[:8]) + (f" and {len(ep.agents) - 8} more" if len(ep.agents) > 8 else ""))
            if b.button("Investigate", key=f"inv_{ep.episode_id}"):
                st.session_state.episode = ep.episode_id
                st.success("Selected. Open the **Investigate** tab above.")

# ------------------------------------------------------------------ Investigate
with t_inv:
    if not R.episodes:
        st.info("No episodes to investigate.")
    else:
        sel = st.selectbox("Episode", ids, index=ids.index(st.session_state.episode),
                           format_func=lambda i: f"{i} — {S.episode_sentence(R.episode(i), R.events)}")
        st.session_state.episode = sel
        ep = R.episode(sel)
        sub = R.events[R.events.event_id.isin(set(ep.event_ids))]
        st.markdown(f"## {S.esc(S.episode_sentence(ep, R.events))}")
        st.markdown(f'<div class="sg-byline">{ep.episode_id}<span class="sep">·</span>{ep.start:%b %d %Y, %H:%M}–{ep.end:%H:%M} UTC'
                    f'<span class="sep">·</span>keywords: {S.esc(ep.label)}</div>', unsafe_allow_html=True)

        st.markdown("### 1. What the logs show")
        st.caption("The first and most connected actions in this episode, verbatim (green = straight from the logs).")
        shown = set()
        prop = next(d for d in ep.detections if d.name == "information_propagation")
        seq = [e for h in prop.details.get("hops", []) for e in h["evidence"]] or list(sub.event_id[:6])
        for eid in seq:
            if eid not in shown and len(shown) < 6:
                shown.add(eid)
                st.markdown(quote(eid), unsafe_allow_html=True)
        with st.expander(f"Show all {len(sub)} actions in this episode"):
            for eid in sub.event_id[:120]:
                st.markdown(quote(eid, 260), unsafe_allow_html=True)

        st.markdown("### 2. Who and what was involved")
        st.plotly_chart(episode_graph(ep, ep.agent_edges), use_container_width=True, config={"displayModeBar": False})
        st.caption("Slate dots are agents; open squares are the pages or sites they used. Green dotted lines are "
                   "observed (the agent really touched that page). Amber arrows are SwarmGuard's inference that one "
                   "agent's action may have led to another's.")
        sub2 = sub.assign(group=sub.action_type.map(GROUP).fillna("action"))
        fig = go.Figure()
        for g, sym in GROUP_SYMBOL.items():
            s = sub2[sub2.group == g]
            if len(s):
                fig.add_scatter(x=s.timestamp, y=s.actor_id, mode="markers", name={"talk": "message", "action": "action on a page / tool", "memory": "memory or goal"}[g],
                                marker=dict(size=9, color=S.SLATE, symbol=sym, line=dict(width=1.5, color=S.CREAM)),
                                hovertext=[f"{r.actor_id}: {S.esc(str(r.content)[:140])}" for r in s.itertuples()], hoverinfo="text")
        fig.update_layout(title="When each agent acted")
        st.plotly_chart(S.style_fig(fig, max(260, 26 * len(ep.agents))), use_container_width=True, config={"displayModeBar": False})

        st.markdown("### 3. What SwarmGuard infers")
        if prop.details.get("hops"):
            src_ = prop.details["candidate_source"]
            amb = prop.details.get("competing_sources") or []
            body = (f"The earliest observed carrier is <b>{S.esc(src_)}</b>; from there a time-ordered chain reaches "
                    f"{len(prop.details['hops'])} other agent(s).")
            if amb:
                body += f" But {len(amb)} other agent(s) could equally be the source ({S.esc(', '.join(amb[:3]))})."
            st.markdown(S.infer_html("Possible spread", body), unsafe_allow_html=True)
        for e in ep.agent_edges[:5]:
            st.markdown(S.infer_html(f"{e.src} → {e.dst} · {S.CONFIDENCE_WORD[e.confidence]}",
                                     S.esc(S.clean_text(e.explanation)), [S.clean_text(x) for x in e.alternatives]),
                        unsafe_allow_html=True)
        with st.expander("Score details for every link in this episode"):
            ef = edges_frame(ep.agent_edges)
            if len(ef):
                ef = ef.rename(columns={"c_temporal": "time", "c_semantic": "wording", "c_resource": "same page",
                                        "c_direct": "addressed", "c_goal": "goal", "c_action": "same action"})
                st.dataframe(ef[["src", "dst", "score", "confidence", "support", "time", "wording", "same page",
                                 "addressed", "goal", "same action", "alternatives"]], hide_index=True, use_container_width=True)
                pick = st.selectbox("Show the evidence pairs behind one link", range(len(ep.agent_edges)),
                                    format_func=lambda k: f"{ep.agent_edges[k].src} → {ep.agent_edges[k].dst}")
                for a_, b_ in ep.agent_edges[pick].evidence_pairs[:4]:
                    st.markdown(quote(a_, 240) + quote(b_, 240), unsafe_allow_html=True)
                    st.markdown("<hr style='border-top:1px solid #e8e8df'>", unsafe_allow_html=True)

        st.markdown("### 4. The checks")
        for d in ep.detections:
            verdict = "Yes" if d.triggered else "No"
            st.markdown(f"**{S.DETECTOR_QUESTION[d.name]}** {verdict}"
                        + (f" (confidence {S.CONFIDENCE_WORD.get(d.confidence, d.confidence)})." if d.triggered else "."))
            st.markdown(f'<p style="margin:.1rem 0 .1rem 0">{S.esc(d.why)}</p>'
                        f'<p class="sg-alt" style="color:#666">This does not prove: {S.esc(d.does_not_prove)}</p>',
                        unsafe_allow_html=True)

        hr = ep.stats.get("historical_response")
        if hr:
            st.markdown("### 5. What the operators actually did")
            body = (f"{hr['pages_deleted']} of {hr['episode_pages']} pages in this episode were later deleted by "
                    f"{S.esc(', '.join(hr['deleted_by']))}, the first one <b>{hr['hours_from_episode_end_to_first_deletion']} hours</b> "
                    f"after the episode ended. {hr['writes_after_episode_before_deletion']} more writes landed in between; "
                    f"{hr['recreations_after_deletion']} page(s) came back after deletion.")
            st.markdown(S.observed_html("From the deletion log", body), unsafe_allow_html=True)
            if hr.get("agreement"):
                st.caption(hr["agreement"])

        sess = sub[sub.action_type == "session_start"]
        if src == "ai" and len(sess):
            with st.expander("Go deeper: load the computer-use steps behind these sessions (slow, several minutes)"):
                choice = st.multiselect("Sessions", sess.target_id.tolist(),
                                        format_func=lambda s: f"{s[:8]} · {sess.set_index('target_id').loc[s, 'actor_id']}")
                if choice and st.button("Load steps"):
                    from swarmguard.data.ai_village import drilldown_turns

                    with st.spinner("Streaming computer_use_turns…"):
                        st.dataframe(drilldown_turns(choice), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ Intervene
with t_int:
    if not R.episodes or st.session_state.episode not in R.interventions:
        st.info("Pick one of the top ten episodes on the Investigate tab.")
    else:
        ep = R.episode(st.session_state.episode)
        ranked = R.interventions[ep.episode_id]
        st.markdown(f"## If you could change one thing")
        st.markdown(f'<p class="sg-lede">For {ep.episode_id} ({S.esc(S.episode_sentence(ep, R.events).rstrip("."))}), these are the '
                    'controls that would cut the most inferred spread while disturbing the least unrelated activity. '
                    'They are recommendations for whoever runs the system; SwarmGuard controls nothing itself.</p>',
                    unsafe_allow_html=True)
        for k, ri in enumerate(ranked[:3], 1):
            with st.container(border=True):
                st.markdown(f'<span class="sg-label">Option {k}</span>', unsafe_allow_html=True)
                st.markdown(f"### {S.esc(S.intervention_label(ri))}")
                st.markdown(f'<p style="margin-top:-.3rem">{S.esc(S.INTERVENTION_PRACTICE[ri.intervention.kind])}</p>',
                            unsafe_allow_html=True)
                c = st.columns(4)
                c[0].metric("Spread removed", f"{ri.paths_removed:.0%}",
                            help=f"{ri.n_paths_removed} of {ri.n_paths_before} inferred agent-to-agent paths disappear.")
                c[1].metric("Side-effects", f"{ri.collateral:.1%}", help="Share of unrelated activity this would also touch.")
                c[2].metric("Undo", "easy" if ri.reversibility >= 0.8 else "partly" if ri.reversibility >= 0.6 else "costly")
                c[3].metric("Evidence", {"high": "solid", "moderate": "fair", "low": "thin"}[ri.confidence])

        fig = go.Figure(go.Scatter(x=[ri.collateral * 100 for ri in ranked], y=[ri.paths_removed * 100 for ri in ranked],
                                   mode="markers+text", text=[str(i + 1) for i in range(len(ranked))],
                                   textposition="top center", textfont=dict(family=S.SANS, size=11, color=S.SLATE),
                                   marker=dict(size=11, color=S.SLATE, line=dict(width=2, color=S.CREAM)),
                                   hovertext=[S.intervention_label(ri) for ri in ranked], hoverinfo="text"))
        fig.update_layout(title="Every option: spread removed vs. side-effects (top-left is best)",
                          xaxis_title="unrelated activity touched (%)", yaxis_title="inferred spread removed (%)")
        st.plotly_chart(S.style_fig(fig, 340, legend=False), use_container_width=True, config={"displayModeBar": False})
        with st.expander("All options as a table"):
            st.dataframe(pd.DataFrame([{"#": i + 1, "option": S.intervention_label(ri),
                                        "spread removed %": round(100 * ri.paths_removed, 1),
                                        "side-effects %": round(100 * ri.collateral, 2),
                                        "agents affected": len(ri.affected_agents), "reversibility": ri.reversibility,
                                        "evidence": ri.confidence, "score": round(ri.score, 3)} for i, ri in enumerate(ranked)]),
                         hide_index=True, use_container_width=True)

        st.markdown("### Simulate one option")
        k = st.selectbox("Option", range(len(ranked)), format_func=lambda i: f"{i + 1}. {S.intervention_label(ranked[i])}")
        ri = ranked[k]
        cf = ri.counterfactual
        st.markdown(f'<p>With <i>{S.esc(S.intervention_label(ri).lower())}</i> in place, '
                    f'<b>{ri.n_paths_removed} of {ri.n_paths_before}</b> inferred paths between agents disappear. '
                    f'Removed links are drawn in grey.</p>', unsafe_allow_html=True)
        st.plotly_chart(episode_graph(ep, cf.edges_before, {(e.src, e.dst) for e in cf.removed_edges}),
                        use_container_width=True, key=f"before_{k}", config={"displayModeBar": False})
        if cf.removed_path_examples:
            st.markdown("**Examples of spread that would no longer connect:** " + "; ".join(
                f"{S.esc(x['source'])} → {S.esc(x['target'])}" for x in cf.removed_path_examples[:5]))
        st.caption(DISCLAIMER + " It shows which inferred links an option would cut, not what the agents would "
                                "have done instead.")
        st.download_button("Download the full report (Markdown)", render_markdown(build_report(R)),
                           file_name="swarmguard_report.md")

# ------------------------------------------------------------------ How it works
with t_how:
    st.markdown("## How SwarmGuard reads a swarm")
    st.markdown('<p class="sg-lede">A single-agent monitor asks whether one action is bad. In a swarm the worrying thing '
                'is often a pattern no single action reveals: an idea that jumps from one agent to five, a shared page '
                'everyone writes to, coordination that leaves the channel you watch. SwarmGuard reconstructs that '
                'structure from the logs and asks where a small, reversible change would cut it.</p>', unsafe_allow_html=True)
    st.markdown("""
1. **Normalise.** Every log line (chat message, wiki edit, computer session, memory update, admin deletion) becomes
   one action: *who*, *when*, *what*, *where*.
2. **Link.** For each action, look back up to two hours for earlier actions by *other* agents that could explain it:
   same page or site, similar wording, a direct reply, close timing. Timing alone never counts.
3. **Group.** Strongly linked actions that happen close together form an **episode**.
4. **Check.** Four simple checks flag bursts, spread, outside channels and unassigned division of labour, each with
   a note on what it does *not* prove.
5. **Intervene on paper.** Remove a page, a tool, a channel or an agent from a copy of the inferred graph and count
   how much of the spread disappears, versus how much unrelated activity the change would also hit.
""")
    st.markdown("### Words used on this page")
    st.markdown("""<dl class="sg-gloss">
<dt>Possible link (A → B)</dt><dd>B's action came after A's and shares something specific with it. It is a lead, not a proof of influence.</dd>
<dt>Well supported / plausible / thin</dt><dd>How much independent evidence backs a link: several kinds of signal, repeated, versus one coincidence.</dd>
<dt>Could also be</dt><dd>Competing explanations SwarmGuard checks for: a third party prompted both, the same goal was assigned to both, A only reached C through B, or two names are one operator.</dd>
<dt>Spread removed</dt><dd>Share of inferred agent-to-agent paths (respecting time order) that disappear when an option is applied to the graph.</dd>
<dt>Side-effects</dt><dd>Share of actions <i>outside</i> the episode that the same option would also block or touch.</dd>
<dt>Episode</dt><dd>A burst of linked activity among several agents, usually minutes to a few hours long.</dd>
</dl>""", unsafe_allow_html=True)
    st.markdown("### Limits")
    for c_ in CAVEATS:
        st.markdown(f"- {c_}")

# ------------------------------------------------------------------ Blog
with t_blog:
    blog.render()
