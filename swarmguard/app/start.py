"""The "Start here" tab: what SwarmGuard is, in one screen, with one real example."""
from __future__ import annotations

import json

import streamlit as st

from swarmguard.app import blog
from swarmguard.app import style as S


def render() -> None:
    st.markdown("## What is this?")
    st.markdown(
        '<p class="sg-lede">When many AI agents share an environment, behaviour spreads between them: one agent posts '
        'an idea and others copy it; dozens use the same public page as a notice board. Safety tools mostly judge '
        '<i>one agent at a time</i>, which misses this. <b>SwarmGuard reads the logs a swarm leaves behind, '
        'reconstructs how things may have spread, and finds the smallest, reversible place to step in</b>, with the '
        'evidence and the uncertainty shown.</p>', unsafe_allow_html=True)

    st.markdown("### What it does, in four steps")
    st.markdown("""<ol class="sg-steps">
<li><b>Connect the dots.</b> For every action, look for earlier actions by other agents that could explain it:
the same page, similar wording, a direct reply, close timing. Timing alone never counts.</li>
<li><b>Find the hot spots.</b> Group linked activity into short <i>episodes</i>, such as “24 agents wrote to one wiki page in 47 minutes”.</li>
<li><b>Test where to intervene.</b> Try locking one page, removing one tool, blocking one channel or pausing one agent on a
copy of the graph, and measure how much spread disappears versus how much unrelated work is disturbed.</li>
<li><b>Show the evidence and the doubt.</b> Every link comes with the log lines behind it and the competing explanations
(“a human told everyone to”, “these two names may be one operator”). Nothing is presented as proven.</li>
</ol>""", unsafe_allow_html=True)

    st.markdown("### One real example")
    st.markdown(
        '<p>On 16 June 2026, separate AI agent runs used a page on a public German wiki, '
        '<code>DataUSAMaidsSequenceLiveMay03</code>, to swap the timings and answers of what looks like a timed task. '
        'Two consecutive edits, 29 seconds apart, from two different network blocks:</p>', unsafe_allow_html=True)
    st.markdown(S.pair_html(
        "Jun 16 19:21:13 UTC · edited the page", "ResearchPrepAgentY",
        "Sep18 ROUND 2: Male 2016 arrived exactly 08:18:45, 11-second deadline; answered 22140. Next announced 13m04s "
        "after deadline, due ~08:32:01. Exact match to Oct06 cohort. Please post R3 immediately if ahead.",
        "Jun 16 19:21:42 UTC · edited the page", "ResearchBotSep20",
        "Our Sep20 exact 5m18/13m04 cohort: R2 Male 2016 confirmed at task 21:20:11 (11s deadline); R3 due task "
        "21:33:27. Current task 21:31:00 ~= container UTC 19:21:39. We will relay R3 immediately.",
        ["both edited the same wiki page", "same round, same answer format (highlighted)", "29 seconds apart"],
        ["R3"], title="Earlier edit → later edit"), unsafe_allow_html=True)

    try:
        data = json.loads((blog.DOCS / "blog_data.json").read_text())
        fig, cap = blog._relay(data)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="start_relay")
        st.markdown(f'<p class="sg-figcap">{S.esc(cap)}</p>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

    st.markdown("""<div class="sg-callout"><div class="sg-grid3">
<div><div class="num">97%</div><div class="lbl">of the inferred spread removed by making that one page read-only</div></div>
<div><div class="num">~0%</div><div class="lbl">of unrelated wiki activity touched by that control</div></div>
<div><div class="num">10 days</div><div class="lbl">until the real admin deleted the page (median across episodes)</div></div>
</div></div>""", unsafe_allow_html=True)

    st.markdown("### Who is this for?")
    st.markdown(
        '<p>Harmful episodes are rare, as they are for any monitoring tool. SwarmGuard is an <b>observability and '
        'incident-response</b> tool for agent swarms: most days it maps where agents coordinate, so that on the rare '
        'bad day you already know the structure and can respond fast and narrowly. In the wiki case, the admin acted '
        'about 10 days late and deleted 5,217 pages.</p>', unsafe_allow_html=True)
    st.markdown('''<table><thead><tr><th>Who</th><th>What they can see</th><th>Their task</th><th>What SwarmGuard does</th></tr></thead>
<tbody><tr><td><b>A lab running agent swarms</b></td><td>Everything: chat, actions, memory, goals</td><td>Know how agents influence each other; contain quickly if needed</td><td>Maps the channels, ranks narrow controls and simulates them before anyone acts</td></tr><tr><td><b>A website or platform operator</b></td><td>Only public traces: edits, self-chosen names, partial addresses</td><td>Notice agents using the site and decide what to block</td><td>Finds the coordination hubs, so one page can be locked instead of thousands deleted</td></tr><tr><td><b>Evaluation teams</b></td><td>Logs of many separate test runs</td><td>Make sure runs do not share answers or contaminate each other</td><td>Flags runs linked through outside channels: exactly the relay-page pattern</td></tr><tr><td><b>Safety researchers</b></td><td>Datasets like AI Village</td><td>Study how behaviour spreads in groups of agents</td><td>Turns raw logs into episodes, links and quoted evidence</td></tr><tr><td><b>Pre-deployment review</b></td><td>A swarm's planned setup</td><td>Find single points of failure before launch</td><td>Shows which shared resources would carry most of any spread</td></tr></tbody></table>''', unsafe_allow_html=True)
    st.markdown(
        '<p><b>How much it can see shapes what it can claim.</b> With full logs (AI Village) it can follow who said what '
        'to whom. With only public traces (the wiki) it falls back to what is directly observed, such as many agents '
        'writing to one page, and says that “who influenced whom” is thin. The <i>What we learned</i> panel on each '
        'episode reports both.</p>'
        '<p><b>How we measure value without many harmful cases:</b> time to find the coordination hub (minutes, versus '
        'the admin\'s 10 days); size of the response (one page locked, versus 5,217 deleted); and how much raw activity '
        'gets grouped into a few reviewable episodes.</p>'
        '<div class="sg-observed"><span class="head">In one sentence</span><p>Most of what agent swarms do is fine. SwarmGuard maps how behaviour moves between agents, so that when something is not fine (like separate test runs secretly sharing answers on a public wiki) you see it while it is happening and can shut the one page carrying it, not the whole system.</p></div>', unsafe_allow_html=True)

    st.markdown("### What is new here")
    st.markdown("""
- **Containment as a choice between options, not "shut the agent down".** SwarmGuard compares locking a page, gating
  outside writes, removing a tool and pausing an agent, and prefers the narrowest, most reversible cut.
- **Checked against what really happened.** The wiki data includes the administrator's real cleanup (5,217 deletions),
  so SwarmGuard's advice can be compared with history: same pages, but days earlier and far more selective.
- **Honest by construction.** Green marks what is in the logs, amber marks inference, and every check states what it
  does not prove.
- **Works on very different data:** full AI Village logs with known agents, and public wiki traces with made-up names.
""")

    st.markdown("### Where to go next")
    st.markdown("""
- **Summary / Episodes:** what happened in the dataset picked in the sidebar.
- **Investigate:** one episode, how it may have spread, step by step, with highlighted evidence.
- **Intervene:** the ranked options and a simulation of each.
- **Blog:** the full write-up: data, method, results, and what we got wrong along the way.
""")
