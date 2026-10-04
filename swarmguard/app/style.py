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
mark.sg-hl {{ background: {GREEN_TINT}; color: {INK}; padding: 0 .12em; border-bottom: 1px solid {GREEN}; }}
.sg-pair {{ margin: 1.8rem 0 2.2rem; }}
.sg-pair-title {{ font-family: {SANS}; font-size: .74rem; text-transform: uppercase; letter-spacing: .07em; color: {DULL};
  margin-bottom: .2rem; }}
.sg-pair .sg-quote {{ margin: .35rem 0; }}
.sg-why {{ display: flex; gap: .55rem; align-items: baseline; margin: .25rem 0 .25rem .2rem; padding: .3rem .7rem;
  border-left: 3px solid {AMBER}; background: {AMBER_WASH}; font-size: .98rem; color: {AMBER_INK}; }}
.sg-why .arrow {{ font-family: {SANS}; color: {AMBER}; }}
.sg-why b {{ font-weight: 400; font-family: {SANS}; font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; }}
.sg-steps {{ counter-reset: step; list-style: none; padding: 0; margin: 1.2rem 0 1.8rem; }}
.sg-steps li {{ counter-increment: step; position: relative; padding: .1rem 0 .9rem 2.6rem; }}
.sg-steps li::before {{ content: counter(step); position: absolute; left: 0; top: .05rem; width: 1.8rem; height: 1.8rem;
  border: 1px solid {SLATE}; border-radius: 50%; color: {SLATE}; font-family: {SANS}; font-size: .85rem;
  display: flex; align-items: center; justify-content: center; }}
.sg-steps b {{ font-weight: 700; }}
.sg-callout {{ border-top: 1px solid {RULE}; border-bottom: 1px solid {RULE}; padding: 1rem 0; margin: 1.6rem 0; }}
.sg-callout .num {{ font-size: 2.6rem; line-height: 1; }}
.sg-callout .lbl {{ font-family: {SANS}; font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; color: {DULL}; }}
.sg-grid3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.2rem; }}
.sg-section-gap {{ height: 1.6rem; }}
.tk {{ border-top: 2px solid {INK}; border-bottom: 1px solid {RULE}; margin: 1.2rem 0 2.4rem; }}
.tk-head {{ font-family: {SANS}; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: {INK};
  padding: .7rem 0 .3rem; }}
.tk-row {{ display: grid; grid-template-columns: 9.5rem 1fr; gap: 1.2rem; padding: .85rem 0; border-top: 1px solid {HAIR}; }}
.tk-lbl {{ font-family: {SANS}; font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; color: {DULL}; padding-top: .3rem; }}
.tk-find {{ font-size: 1.22rem !important; line-height: 1.45 !important; margin: 0 0 .3rem !important; }}
.tk-how {{ font-size: .98rem !important; line-height: 1.45 !important; color: {DULL}; margin: 0 !important; }}
.tk-how b {{ font-family: {SANS}; font-weight: 400; font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; }}
/* reading mode (Blog tab): no sidebar, no app header, full width */
body.sg-blog-mode [data-testid="stSidebar"], body.sg-blog-mode [data-testid="stSidebarCollapsedControl"],
body.sg-blog-mode [data-testid="stHeader"], body.sg-blog-mode .sg-apphead {{ display: none !important; }}
body.sg-blog-mode .block-container {{ max-width: 100% !important; padding: .4rem 0 0 !important; }}
body.sg-blog-mode [data-baseweb="tab-list"], body.sg-blog-mode [role="tablist"] {{ padding-left: 2rem !important; }}
body.sg-blog-mode .block-container {{ padding-top: 0 !important; }}
body.sg-blog-mode [data-testid="stMain"] {{ margin-left: 0 !important; }}
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


# ------------------------------------------------------------------ evidence highlighting
# Shared content between an earlier and a later action is what makes a link worth reading, so we
# show the passage around it and mark it (green tint = observed in both log lines).

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _STOP

_EXTRA_STOP = {"please", "thanks", "thank", "just", "now", "here", "also", "will", "can", "like", "really", "great",
               "today", "know", "think", "want", "need", "make", "sure", "going", "let", "ll", "ve", "re", "http", "https",
               "www", "com", "agent", "agents", "village", "ai", "page", "new", "one", "two", "time", "work"}
_URL_TOK = r"https?://[^\s<>\"')\]]+[^\s<>\"')\].,;:!?]"
_TOKEN = re.compile(_URL_TOK + r"|[A-Za-z0-9][A-Za-z0-9_\-\.]*[A-Za-z0-9]|[A-Za-z0-9]")


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text or "")]


# Words too common in the loaded dataset to be evidence (set per dataset by the app).
COMMON: set[str] = set()


def shared_terms(a: str, b: str, extra: list[str] | None = None, limit: int = 10) -> list[str]:
    """Distinctive tokens / two-word phrases that appear in both texts (case-insensitive).

    Links, page names and anything with a digit always count; plain words count only if they are
    rare in the dataset (not in COMMON), so highlights point at what is specific to this pair."""
    ta, tb = _tokens(a), _tokens(b)
    la, lb = [t.lower() for t in ta], {t.lower() for t in tb}

    def keep(t: str) -> bool:
        tl = t.lower()
        if "://" in tl:
            return True
        if tl in _STOP or tl in _EXTRA_STOP or tl in COMMON:
            return False
        if any(c.isdigit() for c in tl):
            return len(tl) >= 2
        return len(tl) >= 4

    lb = {t.rstrip("/") for t in lb}
    singles = [t for t, tl in zip(ta, la) if tl.rstrip("/") in lb and keep(t)]
    a_low, b_low = (a or "").lower(), (b or "").lower()
    # two-word phrases count only if they appear verbatim (single space) in both texts
    pairs = [f"{ta[i]} {ta[i + 1]}" for i in range(len(ta) - 1)
             if f"{la[i]} {la[i + 1]}" in a_low and f"{la[i]} {la[i + 1]}" in b_low
             and keep(ta[i]) and keep(ta[i + 1])]
    terms = list(dict.fromkeys([*(extra or []), *sorted(pairs, key=len, reverse=True), *singles]))
    # drop singles already covered by a kept phrase
    out: list[str] = []
    for t in terms:
        if t and not any(t.lower() in o.lower() for o in out):
            out.append(t)
        if len(out) >= limit:
            break
    return out


def snippet(text: str, terms: list[str], width: int = 360) -> str:
    """The passage around the first shared term, rather than the start of the message."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= width:
        return text
    low = text.lower()
    hits = [low.find(t.lower()) for t in terms if t and low.find(t.lower()) >= 0]
    pos = min(hits) if hits else 0
    start = max(0, min(pos - width // 3, len(text) - width))
    cut = text[start:start + width]
    return ("…" if start > 0 else "") + cut + ("…" if start + width < len(text) else "")


def highlight(text: str, terms: list[str]) -> str:
    """HTML-escape text and wrap each shared term in a green-tint mark."""
    out = esc(text)
    for t in sorted({t for t in terms if t}, key=len, reverse=True):
        et = re.escape(esc(t))
        if "://" in t:  # URLs: match with or without a trailing slash
            et = re.escape(esc(t.rstrip("/"))) + "/?"
        out = re.sub(rf"(?<![\w>/])({et})(?![\w<])", r'<mark class="sg-hl">\1</mark>', out, flags=re.I)
    return out


def _shorten_urls(text: str, keep: list[str], n: int = 70) -> str:
    """Long query-string URLs drown the prose; shorten them unless the URL itself is the shared evidence."""
    keep_l = [k.lower().rstrip("/") for k in keep if "://" in k]

    def sub(m):
        u = m.group(0)
        if len(u) <= n or any(u.lower().rstrip("/").startswith(k) for k in keep_l):
            return u
        return u[: n - 18] + "…"
    return re.sub(_URL_TOK, sub, text)


def quote_hl_html(meta: str, who: str, text: str, terms: list[str], ref: str = "", width: int = 360) -> str:
    body = highlight(snippet(_shorten_urls(text, terms), terms, width), terms)
    return (f'<div class="sg-quote"><span class="meta">{esc(meta)} · <span class="who">{esc(who)}</span>'
            f'{" · " + esc(ref) if ref else ""}</span><span class="body">{body}</span></div>')


def why_line(reasons: list[str]) -> str:
    return (f'<div class="sg-why"><span class="arrow">↓</span><span><b>Why these are linked:</b> '
            f'{" · ".join(reasons)}</span></div>')


def pair_html(a_meta: str, a_who: str, a_text: str, b_meta: str, b_who: str, b_text: str,
              reasons: list[str], extra_terms: list[str] | None = None, title: str = "") -> str:
    terms = shared_terms(a_text, b_text, extra_terms)
    head = f'<div class="sg-pair-title">{esc(title)}</div>' if title else ""
    return (f'<div class="sg-pair">{head}{quote_hl_html(a_meta, a_who, a_text, terms)}{why_line(reasons)}'
            f'{quote_hl_html(b_meta, b_who, b_text, terms)}</div>')
