"""Visual language and plain-English helpers for the SwarmGuard UI.

The look follows rubyhack.ai (the write-up of the same wiki agent swarm): ET Book
serif on cream, one reading column, hairline rules, and a two-tier color scheme:

* SEMANTIC: green = observed directly in the logs (raw evidence); amber =
  SwarmGuard's inference (candidate links, which can be wrong). Spent only where
  that claim is the point.
* STRUCTURAL: slate = the tool's own machinery (controls, links, chart marks).
"""
from __future__ import annotations

import html
import re

import plotly.graph_objects as go

CREAM = "#fffff8"
WASH = "#f6f6ee"
INK = "#111111"
DULL = "#666666"
HAIR = "#e8e8df"
RULE = "#d6d6cc"
GREEN = "#2A623D"        # observed in the logs
GREEN_TINT = "#e6efe9"
AMBER = "#c77400"        # inferred by SwarmGuard (rules/dots only)
AMBER_INK = "#7a4b00"    # amber when it must be text
AMBER_WASH = "#fff8ea"
SLATE = "#31566f"        # machinery
SLATE_TINT = "#e9edf1"
SERIF = 'et-book, Palatino, "Palatino Linotype", "Book Antiqua", Georgia, serif'
SANS = '-apple-system, BlinkMacSystemFont, "avenir next", avenir, "helvetica neue", helvetica, arial, sans-serif'
MONO = 'SFMono-Regular, Menlo, Consolas, Monaco, monospace'

_FONT = "https://cdn.jsdelivr.net/gh/edwardtufte/tufte-css@gh-pages/et-book"

CSS = f"""
<style>
@font-face {{ font-family: "et-book"; font-style: normal; font-weight: 400; font-display: swap;
  src: url("{_FONT}/et-book-roman-line-figures/et-book-roman-line-figures.woff") format("woff"); }}
@font-face {{ font-family: "et-book"; font-style: italic; font-weight: 400; font-display: swap;
  src: url("{_FONT}/et-book-display-italic-old-style-figures/et-book-display-italic-old-style-figures.woff") format("woff"); }}
@font-face {{ font-family: "et-book"; font-style: normal; font-weight: 700; font-display: swap;
  src: url("{_FONT}/et-book-bold-line-figures/et-book-bold-line-figures.woff") format("woff"); }}

html, body, .stMarkdown, .stText, p, li, label, input, textarea, button, select, [data-baseweb="select"] div {{
  font-family: {SERIF} !important; }}
/* keep Streamlit's icon ligature font (expander arrows, help icons) */
[data-testid="stIconMaterial"], .material-symbols-rounded, [data-testid="stExpanderToggleIcon"] {{
  font-family: "Material Symbols Rounded" !important; }}
.stApp {{ background: {CREAM}; color: {INK}; }}
[data-testid="stSidebar"] {{ background: {WASH}; border-right: 1px solid {HAIR}; }}
.block-container {{ max-width: 900px !important; padding-top: 2.2rem !important; }}
.stMarkdown p, .stMarkdown li {{ font-size: 1.12rem; line-height: 1.55; }}
h1, h2, h3, h4 {{ font-family: {SERIF} !important; font-weight: 400 !important; letter-spacing: -.005em; color: {INK}; }}
h1 {{ font-size: 2.9rem !important; line-height: 1.02 !important; }}
h2 {{ font-size: 2.05rem !important; margin-top: 2.2rem !important; }}
h3 {{ font-size: 1.55rem !important; }}
code, pre, kbd, .sg-mono {{ font-family: {MONO} !important; font-size: .82em; }}
.stMarkdown code {{ color: {INK} !important; background: #f0f0f0 !important; padding: 0 .3em; border-radius: 2px; }}
.stMarkdown a, .stMarkdown a:visited {{ color: {INK} !important; text-decoration: underline; text-decoration-thickness: 1px;
  text-underline-offset: 2px; }}
.stMarkdown a:hover {{ color: {SLATE} !important; }}
.sg-figcap {{ font-size: 1rem; line-height: 1.45; color: {DULL}; font-style: italic; margin: -.4rem 0 1.4rem; }}
a {{ color: inherit; text-decoration: underline; text-decoration-thickness: 1.5px; text-underline-offset: 1.5px; }}

/* tabs: quiet text, slate underline */
[data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid {RULE}; }}
[data-baseweb="tab"] p {{ font-size: 1.08rem !important; }}
[data-baseweb="tab-highlight"] {{ background-color: {SLATE} !important; }}

/* buttons: hairline, slate */
.stButton > button, .stDownloadButton > button {{ background: transparent; color: {SLATE}; border: 1px solid {SLATE};
  border-radius: 2px; font-family: {SANS} !important; font-size: .82rem; letter-spacing: .02em; }}
.stButton > button:hover {{ background: {SLATE_TINT}; color: {SLATE}; border-color: {SLATE}; }}

/* containers as cards on cream */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {HAIR} !important; border-radius: 2px !important;
  background: {CREAM}; }}
[data-testid="stExpander"] details {{ border: 1px solid {HAIR}; border-radius: 2px; background: {CREAM}; }}
[data-testid="stExpander"] summary p {{ font-size: 1rem !important; color: {SLATE}; }}
[data-testid="stMetricValue"] {{ font-family: {SERIF} !important; font-size: 2.1rem !important; }}
[data-testid="stMetricLabel"] p {{ font-family: {SANS} !important; font-size: .78rem !important; color: {DULL};
  text-transform: uppercase; letter-spacing: .05em; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: {DULL} !important; font-style: italic; }}

/* essay pieces */
.sg-dek {{ font-size: 1.25rem; line-height: 1.45; color: {DULL}; font-style: italic; margin: .2rem 0 .6rem; }}
.sg-byline {{ font-family: {SANS}; font-size: .85rem; color: {DULL}; margin-bottom: 1.2rem; }}
.sg-byline .sep {{ margin: 0 .45em; }}
.sg-label {{ font-family: {SANS}; font-size: .72rem; text-transform: uppercase; letter-spacing: .07em; color: {DULL}; }}
.sg-lede {{ font-size: 1.2rem; line-height: 1.55; }}
.sg-key {{ display: flex; gap: 1.4rem; flex-wrap: wrap; font-family: {SANS}; font-size: .8rem; color: {DULL};
  border-top: 1px solid {HAIR}; border-bottom: 1px solid {HAIR}; padding: .55rem 0; margin: .4rem 0 1.4rem; }}
.sg-key span {{ display: inline-flex; align-items: center; gap: .45em; }}
.sg-swatch {{ display: inline-block; width: 3px; height: 1.05em; }}
table {{ border-collapse: collapse; margin: 1rem 0; font-size: 1rem; }}
th, td {{ text-align: left; vertical-align: top; padding: 6px 10px; border-bottom: 1px solid {HAIR}; }}
th {{ font-family: {SANS}; font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; color: {DULL}; font-weight: 400; }}
blockquote.sg-quote {{ margin-left: 0; margin-right: 0; }}
.sg-quote {{ margin: .9rem 0; padding: .15rem 0 .15rem 1rem; border-left: 3px solid {GREEN}; }}
.sg-quote .meta {{ font-family: {MONO}; font-size: .74rem; color: {DULL}; display: block; margin-bottom: .2rem; }}
.sg-quote .who {{ color: {SLATE}; }}
.sg-quote .body {{ font-size: 1.05rem; line-height: 1.5; }}
.sg-infer {{ margin: .9rem 0; padding: .55rem .9rem; border-left: 3px solid {AMBER}; background: {AMBER_WASH}; }}
.sg-infer .head {{ font-family: {SANS}; font-size: .74rem; color: {AMBER_INK}; text-transform: uppercase; letter-spacing: .06em; }}
.sg-infer p {{ margin: .25rem 0 0; }}
.sg-alt {{ color: {AMBER_INK}; font-size: .98rem; font-style: italic; }}
.sg-observed {{ margin: .9rem 0; padding: .55rem .9rem; border-left: 3px solid {GREEN}; background: {GREEN_TINT}; }}
.sg-observed .head {{ font-family: {SANS}; font-size: .74rem; color: {GREEN}; text-transform: uppercase; letter-spacing: .06em; }}
.sg-note {{ font-size: 1rem; color: {DULL}; border-top: 1px solid {HAIR}; padding-top: .6rem; margin-top: 1rem; }}
.sg-pill {{ font-family: {SANS}; font-size: .72rem; padding: .08rem .45rem; border: 1px solid {RULE}; border-radius: 2px;
  color: {SLATE}; margin-right: .35rem; white-space: nowrap; }}
.sg-pill.on {{ border-color: {SLATE}; background: {SLATE_TINT}; }}
.sg-big {{ font-size: 2.4rem; line-height: 1; }}
.sg-stat {{ font-family: {SANS}; font-size: .78rem; color: {DULL}; text-transform: uppercase; letter-spacing: .05em; }}
dl.sg-gloss dt {{ font-weight: 700; margin-top: .9rem; }}
dl.sg-gloss dd {{ margin: .15rem 0 0 0; }}
</style>
"""


def esc(s: object) -> str:
    return html.escape(str(s if s is not None else ""))


def key_html() -> str:
    return (f'<div class="sg-key">'
            f'<span><i class="sg-swatch" style="background:{GREEN}"></i>Observed in the logs (raw evidence)</span>'
            f'<span><i class="sg-swatch" style="background:{AMBER}"></i>SwarmGuard\'s inference (can be wrong)</span>'
            f'<span><i class="sg-swatch" style="background:{SLATE}"></i>Controls and links</span></div>')


def quote_html(when: str, who: str, what: str, text: str, ref: str = "") -> str:
    return (f'<div class="sg-quote"><span class="meta">{esc(when)} · <span class="who">{esc(who)}</span> · {esc(what)}'
            f'{" · " + esc(ref) if ref else ""}</span><span class="body">{esc(text)}</span></div>')


def infer_html(head: str, body: str, alts: list[str] | None = None) -> str:
    alt = "".join(f'<p class="sg-alt">Could also be: {esc(a)}</p>' for a in (alts or []))
    return f'<div class="sg-infer"><span class="head">{esc(head)}</span><p>{body}</p>{alt}</div>'


def observed_html(head: str, body: str) -> str:
    return f'<div class="sg-observed"><span class="head">{esc(head)}</span><p>{body}</p></div>'


# ------------------------------------------------------------------ plain-English vocabulary

DETECTOR_QUESTION = {
    "coordination_burst": "Did cross-agent activity suddenly spike?",
    "information_propagation": "Did one agent's pattern spread to several others, hop by hop?",
    "external_channel_coordination": "Did agents coordinate through an outside site rather than internal chat?",
    "emergent_delegation": "Did agents split up work without anyone assigning it? (experimental)",
}
DETECTOR_PILL = {
    "coordination_burst": "burst of cross-agent activity",
    "information_propagation": "pattern spread agent to agent",
    "external_channel_coordination": "coordinating through an outside page or site",
    "emergent_delegation": "work split without an assignment",
}
CONFIDENCE_WORD = {"strong": "well supported", "moderate": "plausible", "weak": "thin",
                   "high": "high", "low": "low"}
ACTION_VERB = {"edit": "edited", "create": "created", "message": "posted", "session_start": "started a session",
               "memory_write": "updated memory", "memory_read": "searched history", "visit": "visited",
               "delete": "deleted", "tool_use": "used a tool", "session_end": "ended a session", "goal_set": "got a goal"}
INTERVENTION_PRACTICE = {
    "restrict_writes": "Make this resource read-only (or require approval to change it) for the agents involved.",
    "approval_gate": "Hold every external post or edit by these agents for human approval.",
    "isolate_memory": "Stop agents from reading each other's entries in this shared memory.",
    "block_channel": "Stop messages from the first agent reaching the second.",
    "limit_tool": "Take this one tool away from this one agent.",
    "quarantine_agent": "Pause this agent entirely: the bluntest option.",
}


def resource_name(r: str) -> str:
    """Human-readable resource names: 'resource:wiki:dse~Foo' -> 'wiki page Foo'."""
    r = r.removeprefix("resource:")
    if r.startswith("wiki:"):
        return "wiki page " + r.split("~", 1)[-1]
    if r.startswith("memory:village_history"):
        return "the shared village history"
    if r.startswith("memory:"):
        return f"{r.split(':', 1)[1]}'s memory"
    if r.startswith("room:"):
        return "a chat room"
    return r


def minutes_phrase(hours: float) -> str:
    m = round(hours * 60)
    return f"{m} minutes" if m < 120 else f"{hours:.1f} hours"


def episode_sentence(ep, events) -> str:
    """One plain sentence: who, what, where, how fast."""
    st = ep.stats
    n, dur = st["n_agents"], minutes_phrase(st["duration_hours"])
    top = st.get("top_resources") or []
    mix = st.get("action_mix") or {}
    dom = max(mix, key=mix.get) if mix else "message"
    where = resource_name(top[0][0]) if top else None
    if dom in ("edit", "create"):
        # Name the page they actually edited, not one they merely linked to.
        sub = events[events.event_id.isin(set(ep.event_ids)) & events.action_type.isin(["edit", "create"])]
        if len(sub) and sub.target_id.notna().any():
            where = resource_name(sub.target_id.value_counts().index[0])
    if dom in ("edit", "create") and where:
        verb = f"wrote to {where}"
    elif dom == "message" and where:
        verb = f"discussed and linked {where}"
    elif where:
        verb = f"converged on {where}"
    else:
        verb = "interacted"
    return f"{n} agents {verb} within {dur} ({st['n_events']} events)."


def intervention_label(ri) -> str:
    iv = ri.intervention
    t = resource_name(iv.target) if iv.kind in ("restrict_writes", "isolate_memory") else iv.target
    if iv.kind == "limit_tool":
        a, tool = iv.target.split("|")
        t = f"{tool.replace('proxy:', '')} for {a}"
    if iv.kind == "block_channel":
        t = iv.target.replace("->", " → ")
    names = {"restrict_writes": "Restrict writes to", "approval_gate": "Require approval for all external writes",
             "isolate_memory": "Isolate", "block_channel": "Block messages", "limit_tool": "Remove tool",
             "quarantine_agent": "Pause agent"}
    return names[iv.kind] if iv.kind == "approval_gate" else f"{names[iv.kind]} {t}"


# ------------------------------------------------------------------ charts

def style_fig(fig: go.Figure, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=6, r=6, t=64 if legend else 40, b=6), paper_bgcolor=CREAM, plot_bgcolor=CREAM,
        font=dict(family=SERIF, size=14, color=INK), title_font=dict(family=SERIF, size=16, color=INK),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(family=SANS, size=11, color=DULL)),
        title=dict(y=0.985, yanchor="top", x=0, xanchor="left", pad=dict(l=4)),
        hoverlabel=dict(bgcolor=CREAM, bordercolor=RULE, font=dict(family=SERIF, size=13, color=INK)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=RULE, tickfont=dict(family=SANS, size=11, color=DULL))
    fig.update_yaxes(showgrid=True, gridcolor=HAIR, zeroline=False, tickfont=dict(family=SANS, size=11, color=DULL))
    return fig


def strip_md(s: str) -> str:
    return re.sub(r"[*_`]", "", s or "")


_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


def clean_text(s: str) -> str:
    """Replace internal keys (resource:..., room:<uuid>) with readable names in explanations."""
    s = re.sub(r"room:" + _UUID, "a chat room", s or "")
    s = re.sub(r"resource:(wiki:[^\s,;)]+|memory:[^\s,;)]+|[^\s,;)]+)", lambda m: resource_name(m.group(1)), s)
    return s.replace("event pair(s)", "pairs of actions")
