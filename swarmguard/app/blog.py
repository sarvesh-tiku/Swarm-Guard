"""Renders docs/blog.md inside the app, with figures built from docs/blog_data.json."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from swarmguard.app import style as S

DOCS = Path(__file__).resolve().parents[2] / "docs"


def _relay(d: dict) -> go.Figure:
    df = pd.DataFrame(d["relay"])
    df["t"] = pd.to_datetime(df["t"])
    df = df[(df.action != "delete") & (df.t.dt.date == df.t.min().date())]
    order = list(dict.fromkeys(df.sort_values("t").actor))
    fig = go.Figure(go.Scatter(
        x=df.t, y=df.actor, mode="markers", marker=dict(size=9, color=S.SLATE, line=dict(width=1.5, color=S.CREAM)),
        hovertext=[f"{a} ({o}.x.x): {S.esc(t[:160])}" for a, o, t in zip(df.actor, df.origin, df.text)], hoverinfo="text"))
    fig.update_yaxes(categoryorder="array", categoryarray=order[::-1], tickfont=dict(size=10))
    fig.update_layout(title="Every edit to the relay page on 16 June (UTC), one row per name")
    cap = (f"{d['relay_page']}: {df.actor.nunique()} names (including the page's creator at 18:48) writing from "
           f"{df.origin.nunique()} different /16 network blocks. Hover a dot to read the edit. The administrator "
           "deleted the page ten days later.")
    return S.style_fig(fig, 560, legend=False), cap


def _wiki_daily(d: dict) -> go.Figure:
    w = pd.Series(d["wiki_daily_writes"]).sort_index()
    x = pd.Series(d["wiki_daily_deletes"]).sort_index()
    fig = go.Figure()
    fig.add_bar(x=pd.to_datetime(w.index), y=w.values, name="agent writes per day", marker_color=S.SLATE, marker_line_width=0)
    fig.add_bar(x=pd.to_datetime(x.index), y=x.values, name="admin deletions per day", marker_color="#9fb3c1", marker_line_width=0)
    fig.update_layout(barmode="group", bargap=0.2, title="Agent writes and admin deletions per day")
    cap = ("Agent writes peak on 18 June. The cleanup begins in earnest the next day and continues in large "
           "batches until 14 July, well after the last agent write on 2 July.")
    return S.style_fig(fig, 320), cap


def _audit(d: dict) -> go.Figure:
    lags = pd.Series(d["audit_top_lag_hours"]) / 24
    med = float(lags.median())
    fig = go.Figure(go.Histogram(x=lags, xbins=dict(start=-1, end=22, size=1), marker_color=S.SLATE,
                                 marker_line=dict(width=1, color=S.CREAM),
                                 hovertemplate="%{x} days: %{y} episodes<extra></extra>"))
    fig.add_vline(x=med, line=dict(color=S.INK, width=1, dash="dot"))
    fig.add_annotation(x=med, y=1, yref="paper", text=f"median {med:.0f} days", showarrow=False, xanchor="left",
                       xshift=4, font=dict(family=S.SANS, size=11, color=S.DULL))
    fig.update_layout(title="How long until the admin deleted the page SwarmGuard ranks first",
                      xaxis_title="days after the episode ended", yaxis_title="episodes", bargap=0.05)
    cap = (f"One bar per day of delay, across the {len(lags)} episodes whose top recommendation is a single page "
           "that the administrator later deleted. Negative values mean the page was deleted while the episode was "
           "still running.")
    return S.style_fig(fig, 300, legend=False), cap


FIGS = {"relay": _relay, "wiki_daily": _wiki_daily, "audit": _audit}


def render() -> None:
    md = (DOCS / "blog.md").read_text()
    data_p = DOCS / "blog_data.json"
    data = json.loads(data_p.read_text()) if data_p.exists() else None
    parts = re.split(r"<!-- fig:(\w+) -->", md)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part, unsafe_allow_html=True)
        elif data is not None and part in FIGS:
            fig, cap = FIGS[part](data)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"blog_{part}")
            st.markdown(f'<p class="sg-figcap">{S.esc(cap)}</p>', unsafe_allow_html=True)


STATIC = Path(__file__).resolve().parent / "static" / "blog.html"


def render_page() -> None:
    """Embed the standalone write-up (scripts/build_blog.py) and link to its full-width version."""
    import streamlit.components.v1 as components

    if not STATIC.exists():
        st.info("Build the write-up first: `python scripts/build_blog.py`. Showing the plain version instead.")
        render()
        return
    # Reading mode: while this tab is visible, hide the sidebar and app header and let the write-up fill the
    # window; leaving the tab restores them. (The component iframe is same-origin, so it can reach the page.)
    components.html("""<script>
const me = window.frameElement, doc = window.parent.document, win = window.parent;
function tick() {
  const on = !!(me && me.offsetParent !== null);
  doc.body.classList.toggle('sg-blog-mode', on);
  const blog = [...doc.querySelectorAll('iframe')].find(f => f !== me && (f.srcdoc || '').includes('Where would you step in?'));
  if (blog && on) {
    const h = Math.max(500, win.innerHeight - blog.getBoundingClientRect().top - 4) + 'px';
    if (blog.style.height !== h) { blog.style.height = h; if (blog.parentElement) blog.parentElement.style.height = h; }
  }
}
setInterval(tick, 200); tick();
</script>""", height=0)
    components.html(STATIC.read_text(), height=900, scrolling=True)
