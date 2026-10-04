"""Build the SwarmGuard write-up as one self-contained HTML page (layout after rubyhack.ai / collusion.wiki).

    python scripts/build_blog.py

Reads docs/blog_data.json (numbers and quotes extracted from the real runs) and writes:
  docs/blog/index.html                 standalone page for the repo
  swarmguard/app/static/blog.html      same page, served by the Streamlit app at /app/static/blog.html

Charts are hand-built SVG so they can carry annotations (dated callouts with leader lines, bands,
numbered evidence markers), which plotting libraries do poorly.
"""
from __future__ import annotations

import html
import json
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

from swarmguard.app import style as S

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / "docs" / "blog_data.json").read_text())
E = html.escape

INK, DULL, HAIR, RULE, CREAM = "#111", "#666", "#e8e8df", "#d6d6cc", "#fffff8"
GREEN, AMBER, AMBER_INK, SLATE = "#2A623D", "#c77400", "#7a4b00", "#31566f"
MONO = "SFMono-Regular, Menlo, Consolas, monospace"
SANS = '-apple-system, BlinkMacSystemFont, "avenir next", avenir, "helvetica neue", helvetica, arial, sans-serif'


def d(s: str) -> date:
    return datetime.fromisoformat(str(s)[:10]).date()


def fmt_day(x: date) -> str:
    return f"{x.day} {x.strftime('%b')}"


# ============================================================ chart 1: the incident timeline
def timeline_svg() -> str:
    W, L, R = 920, 64, 20
    top_h, a_h, gap, b_h = 120, 230, 34, 96
    y_a0 = top_h + a_h            # baseline of writes panel
    y_b_top = y_a0 + gap
    y_b0 = y_b_top + b_h          # baseline of deletions panel
    H = y_b0 + 190
    start, end = date(2026, 5, 9), date(2026, 7, 17)
    span = (end - start).days

    def X(x: date) -> float:
        return L + (x - start).days / span * (W - L - R)

    bw = (W - L - R) / span * 0.78
    writes = {d(k): v for k, v in D["wiki_daily_writes"].items()}
    dels = {d(k): v for k, v in D["wiki_daily_deletes"].items()}
    wmax, dmax = 7000, 700
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Timeline of agent wiki writes and admin deletions">']
    # gridlines + y labels (writes)
    for v in (2000, 4000, 6000):
        y = y_a0 - v / wmax * a_h
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="{HAIR}"/>')
        out.append(f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{v // 1000}k</text>')
    out.append(f'<text transform="translate(16 {y_a0 - a_h / 2}) rotate(-90)" text-anchor="middle" class="axis">agent wiki writes per day</text>')
    for v in (300, 600):
        y = y_b0 - v / dmax * b_h
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="{HAIR}"/>')
        out.append(f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{v}</text>')
    out.append(f'<text transform="translate(16 {y_b0 - b_h / 2}) rotate(-90)" text-anchor="middle" class="axis">admin deletions</text>')
    # relay-board day highlight
    rx = X(date(2026, 6, 16))
    out.append(f'<rect x="{rx - bw:.1f}" y="{top_h}" width="{bw * 2:.1f}" height="{y_b0 - top_h}" fill="{AMBER}" opacity=".10"/>')
    # bars
    for k, v in writes.items():
        h = min(v, wmax) / wmax * a_h
        out.append(f'<rect x="{X(k) - bw / 2:.1f}" y="{y_a0 - h:.1f}" width="{bw:.1f}" height="{max(h, .6):.1f}" fill="{INK}">'
                   f'<title>{fmt_day(k)}: {v:,} agent writes</title></rect>')
    for k, v in dels.items():
        h = v / dmax * b_h
        out.append(f'<rect x="{X(k) - bw / 2:.1f}" y="{y_b0 - h:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{SLATE}" opacity=".85">'
                   f'<title>{fmt_day(k)}: {v:,} deletions by the admin</title></rect>')
    out.append(f'<line x1="{L}" x2="{W - R}" y1="{y_a0}" y2="{y_a0}" stroke="{INK}" stroke-width=".8"/>')
    out.append(f'<line x1="{L}" x2="{W - R}" y1="{y_b0}" y2="{y_b0}" stroke="{INK}" stroke-width=".8"/>')
    # x ticks (weekly)
    t = date(2026, 5, 11)
    while t <= end:
        out.append(f'<text x="{X(t):.1f}" y="{y_b0 + 18}" text-anchor="middle" class="tick">{fmt_day(t)}</text>')
        t += timedelta(days=7)
    # SwarmGuard episodes per day (inference -> amber), as a small bar row under the axis
    ry, rh = y_b0 + 34, 34
    per_day: dict[date, list] = {}
    for ep in D["wiki_episodes"]:
        per_day.setdefault(d(ep["start"]), []).append(ep)
    mx = max(len(v) for v in per_day.values())
    for k, eps in per_day.items():
        h = max(len(eps) / mx * rh, 2.5)
        ids = ", ".join(e["id"] for e in eps[:6]) + (" …" if len(eps) > 6 else "")
        out.append(f'<rect x="{X(k) - bw / 2:.1f}" y="{ry + rh - h:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{AMBER}">'
                   f'<title>{fmt_day(k)}: {len(eps)} episode(s) flagged by SwarmGuard ({ids})</title></rect>')
        if len(eps) == mx:
            out.append(f'<text x="{X(k) + bw * 7:.1f}" y="{ry + 10}" class="lbl-amber">{len(eps)} start on {fmt_day(k)}</text>')
    out.append(f'<line x1="{L}" x2="{W - R}" y1="{ry + rh}" y2="{ry + rh}" stroke="{AMBER}" stroke-width=".6"/>')
    out.append(f'<text transform="translate(16 {ry + rh / 2}) rotate(-90)" text-anchor="middle" class="axis" '
               f'style="fill:{AMBER_INK}">episodes</text>')
    out.append(f'<text x="{L + 4}" y="{ry + 14}" class="lbl-amber">50 episodes flagged; '
               f'{sum(len(v) for k, v in per_day.items() if date(2026, 6, 15) <= k <= date(2026, 6, 22))} '
               f'in 15–22 June</text>')
    # bands (observed periods), one row each so they never overlap
    for row, (a, b, label) in enumerate([(d(D["milestones"]["first_other_wiki"]), d(D["milestones"]["last_agent_write"]), "Agents writing on the wikis"),
                                         (date(2026, 6, 19), d(D["milestones"]["last_delete"]), "Admin cleanup")]):
        by = y_b0 + 92 + row * 38
        out.append(f'<path d="M{X(a):.1f} {by + 6} V{by} H{X(b):.1f} V{by + 6}" fill="none" stroke="{DULL}" stroke-width="1"/>')
        out.append(f'<text x="{(X(a) + X(b)) / 2:.1f}" y="{by + 21}" text-anchor="middle" class="band">{E(label)} '
                   f'({fmt_day(a)} – {fmt_day(b)})</text>')
    # callouts above: date (mono) + label, leader line to the bar
    calls = [(d(D["milestones"]["first_other_wiki"]), "First edits on", "other wikis"),
             (d(D["milestones"]["first_probe"]), "First script", "probes"),
             (d(D["milestones"]["first_dse_save"]), "First save on", "the DSE wiki"),
             (date(2026, 6, 16), "Relay board", "(Finding 1)"),
             (date(2026, 6, 18), "Peak: 6,778", "writes in a day"),
             (date(2026, 6, 19), "Admin starts", "deleting in bulk")]
    slot_w = (W - L - R) / len(calls)
    for i, (x_d, l1, l2) in enumerate(calls):
        lx = L + slot_w * i + 6
        x = X(x_d)
        target = y_a0 - min(writes.get(x_d, 0), wmax) / wmax * a_h - 4
        out.append(f'<text x="{lx:.1f}" y="22" class="date">{fmt_day(x_d)}</text>')
        out.append(f'<text x="{lx:.1f}" y="42" class="call">{E(l1)}</text><text x="{lx:.1f}" y="60" class="call">{E(l2)}</text>')
        out.append(f'<path d="M{lx + 14:.1f} 70 L{lx + 14:.1f} 84 L{x:.1f} {top_h - 6} L{x:.1f} {target:.1f}" fill="none" '
                   f'stroke="#9aa69c" stroke-width="1"/>')
    out.append("</svg>")
    return "\n".join(out)


# ============================================================ chart 2: the relay board swimlane
def relay_svg(markers: dict[str, int]) -> str:
    rows = [r for r in D["relay"] if r["action"] != "delete" and r["t"][:10] == "2026-06-16"]
    order = list(dict.fromkeys(r["actor"] for r in sorted(rows, key=lambda r: r["t"])))
    W, L, R, T, RH = 920, 220, 30, 34, 24
    H = T + RH * len(order) + 46
    ts = [datetime.fromisoformat(r["t"]) for r in rows]
    t0, t1 = min(ts), max(ts)
    sp = (t1 - t0).total_seconds()

    def X(t: datetime) -> float:
        return L + (t - t0).total_seconds() / sp * (W - L - R)

    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Every edit to the relay page, one row per name">']
    for i, a in enumerate(order):
        y = T + i * RH
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y}" y2="{y}" stroke="{HAIR}"/>')
        out.append(f'<text x="{L - 10}" y="{y + 4.5}" text-anchor="end" class="row">{E(a)}</text>')
    for r in rows:
        t = datetime.fromisoformat(r["t"])
        y = T + order.index(r["actor"]) * RH
        m = markers.get(r["t"][11:19])
        out.append(f'<circle cx="{X(t):.1f}" cy="{y}" r="{7 if m else 5}" fill="{INK if m else SLATE}" stroke="{CREAM}" stroke-width="1.5">'
                   f'<title>{r["t"][11:19]} {E(r["actor"])} ({E(r["origin"] or "?")}.x.x): {E(r["text"][:160])}</title></circle>')
        if m:
            out.append(f'<text x="{X(t) + 10:.1f}" y="{y - 7}" class="mark">{m}</text>')
    tick = t0.replace(minute=(t0.minute // 10) * 10, second=0)
    while tick <= t1:
        if tick >= t0:
            out.append(f'<text x="{X(tick):.1f}" y="{H - 18}" text-anchor="middle" class="tick">{tick:%H:%M}</text>')
        tick += timedelta(minutes=10)
    out.append("</svg>")
    return "\n".join(out)


# ============================================================ chart 3: how long until deletion
def audit_svg() -> str:
    lags = sorted(x / 24 for x in D["audit_top_lag_hours"])
    W, L, R, T = 920, 40, 30, 70
    lo, hi = -1, 22

    def X(v: float) -> float:
        return L + (v - lo) / (hi - lo) * (W - L - R)

    stacks: dict[int, int] = {}
    dots = []
    for v in lags:
        b = int(v // 1) if v >= 0 else -1
        k = stacks.get(b, 0)
        stacks[b] = k + 1
        dots.append((b + 0.5, k, v))
    H = T + 13 * (max(stacks.values()) + 1) + 70
    base = H - 50
    med = sorted(lags)[len(lags) // 2]
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Days until the admin deleted the top-ranked page">']
    for cx, k, v in dots:
        out.append(f'<circle cx="{X(cx):.1f}" cy="{base - 9 - k * 13}" r="5.2" fill="{SLATE}"><title>{v:.1f} days</title></circle>')
    out.append(f'<line x1="{L}" x2="{W - R}" y1="{base}" y2="{base}" stroke="{INK}" stroke-width=".8"/>')
    for v in range(0, 22, 3):
        out.append(f'<text x="{X(v):.1f}" y="{base + 18}" text-anchor="middle" class="tick">{v} d</text>')
    mx = X(med)
    out.append(f'<line x1="{mx:.1f}" x2="{mx:.1f}" y1="{T - 20}" y2="{base}" stroke="{INK}" stroke-dasharray="3 3"/>')
    out.append(f'<text x="{mx + 6:.1f}" y="{T - 24}" class="date">median {med:.0f} days</text>')
    out.append(f'<text x="{X(-0.5):.1f}" y="{T - 46}" class="call-s">deleted while the</text>'
               f'<text x="{X(-0.5):.1f}" y="{T - 30}" class="call-s">episode was running</text>')
    out.append(f'<text x="{X(14):.1f}" y="{T - 6}" class="call-s">some pages stayed up for three weeks</text>')
    out.append(f'<text x="{L}" y="{base + 40}" class="axis">days from the end of each episode until the admin deleted the page SwarmGuard ranks first (one dot per episode)</text>')
    out.append("</svg>")
    return "\n".join(out)


# ============================================================ chart 4: the synthetic test, drawn
def bench_svg() -> str:
    W, H = 920, 340
    N = {"A": (90, 105), "B": (270, 105), "R": (470, 105), "C": (680, 45), "D": (680, 165), "E": (850, 165),
         "op": (200, 275), "I": (560, 245), "J": (560, 305)}
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="The synthetic swarm with a known answer">',
           f'<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
           f'<path d="M0 0 L10 5 L0 10 z" fill="{INK}"/></marker>'
           f'<marker id="aa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
           f'<path d="M0 0 L10 5 L0 10 z" fill="{AMBER}"/></marker></defs>']
    half = {"R": 50, "op": 118}

    def line(a, b, color=INK, dash="", marker="ar", label="", dy=-9):
        (x1, y1), (x2, y2) = N[a], N[b]
        import math
        dx, dyy = x2 - x1, y2 - y1
        n = math.hypot(dx, dyy)
        s1, s2 = half.get(a, 22), half.get(b, 24)
        if a in half and abs(dyy) > 0:  # leave a box from its side
            s1 = half[a] / max(abs(dx) / n, .4) * .9
        x1b, y1b = x1 + dx / n * s1, y1 + dyy / n * s1
        x2b, y2b = x2 - dx / n * s2, y2 - dyy / n * s2
        out.append(f'<line x1="{x1b:.1f}" y1="{y1b:.1f}" x2="{x2b:.1f}" y2="{y2b:.1f}" stroke="{color}" stroke-width="1.6" '
                   + (f'stroke-dasharray="{dash}" ' if dash else "") + f'marker-end="url(#{marker})"/>')
        if label:
            out.append(f'<text x="{(x1b + x2b) / 2:.1f}" y="{(y1b + y2b) / 2 + dy:.1f}" text-anchor="middle" class="call-s">{E(label)}</text>')

    line("A", "B", label="direct message")
    line("B", "R", label="writes to")
    line("R", "C", label="read by C", dy=-10)
    line("R", "D", label="read by D", dy=18)
    line("D", "E", label="mention + read")
    line("op", "I", AMBER, "4 3", "aa")
    line("op", "J", AMBER, "4 3", "aa")
    for k, (x, y) in N.items():
        if k == "R":
            out.append(f'<rect x="{x - 50}" y="{y - 20}" width="100" height="40" fill="{CREAM}" stroke="{INK}" stroke-width="1.6"/>'
                       f'<text x="{x}" y="{y + 5}" text-anchor="middle" class="node">pastebin</text>')
        elif k == "op":
            out.append(f'<rect x="{x - 118}" y="{y - 20}" width="236" height="40" fill="{CREAM}" stroke="{AMBER}" stroke-width="1.2" stroke-dasharray="4 3"/>'
                       f'<text x="{x}" y="{y + 5}" text-anchor="middle" class="node-h">human: “everyone, update the sheet”</text>')
        else:
            main = k in "ABCDE"
            out.append(f'<circle cx="{x}" cy="{y}" r="20" fill="{SLATE if main else CREAM}" stroke="{SLATE}" stroke-width="1.6"/>'
                       f'<text x="{x}" y="{y + 5}" text-anchor="middle" class="node" fill="{CREAM if main else INK}">{k}</text>')
    out.append(f'<text x="{N["A"][0]}" y="{N["A"][1] - 32}" text-anchor="middle" class="lbl-green">✓ source found</text>')
    out.append(f'<text x="{N["B"][0]}" y="{N["B"][1] + 42}" text-anchor="middle" class="call-s">busy hub: pausing B is costly</text>')
    out.append(f'<text x="{N["R"][0]}" y="{N["R"][1] - 32}" text-anchor="middle" class="lbl-green">✓ ranked #1: restrict writes here</text>')
    out.append(f'<line x1="{N["I"][0]}" y1="{N["I"][1] + 21}" x2="{N["J"][0]}" y2="{N["J"][1] - 23}" stroke="{AMBER}" stroke-width="1.3" stroke-dasharray="2 3"/>')
    out.append(f'<text x="{N["I"][0] + 34}" y="{(N["I"][1] + N["J"][1]) / 2 - 4}" class="lbl-amber">I → J looks like a link, but SwarmGuard</text>'
               f'<text x="{N["I"][0] + 34}" y="{(N["I"][1] + N["J"][1]) / 2 + 13}" class="lbl-amber">flags the human prompt as a common cause</text>')
    out.append(f'<text x="20" y="{N["op"][1] - 44}" class="date">planted confounder</text>')
    out.append(f'<line x1="20" x2="{W - 20}" y1="{N["op"][1] - 58}" y2="{N["op"][1] - 58}" stroke="{HAIR}"/>')
    out.append(f'<text x="20" y="18" class="date">true chain</text>')
    out.append("</svg>")
    return "\n".join(out)


# ============================================================ chart 5: the Village Hub chain
def hub_svg() -> str:
    hops = D["hub"]["pathway"]
    pts = []
    for h in hops:
        for e in h["evidence"]:
            pts.append((datetime.fromisoformat(e["t"]), e["actor"]))
    order = list(dict.fromkeys(a for _, a in sorted(pts)))
    W, L, R, T, RH = 920, 170, 220, 30, 40
    H = T + RH * len(order) + 40
    t0, t1 = min(t for t, _ in pts), max(t for t, _ in pts)
    sp = (t1 - t0).total_seconds()

    def X(t):
        return L + (t - t0).total_seconds() / sp * (W - L - R)

    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="How the Village Hub idea may have spread">',
           f'<defs><marker id="am" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
           f'<path d="M0 0 L10 5 L0 10 z" fill="{AMBER}"/></marker></defs>']
    for i, a in enumerate(order):
        y = T + i * RH
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y}" y2="{y}" stroke="{HAIR}"/>'
                   f'<text x="{L - 10}" y="{y + 4}" text-anchor="end" class="row">{E(a)}</text>')
    for h in hops:
        a, b = h["evidence"]
        ta, tb = datetime.fromisoformat(a["t"]), datetime.fromisoformat(b["t"])
        ya, yb = T + order.index(a["actor"]) * RH, T + order.index(b["actor"]) * RH
        out.append(f'<line x1="{X(ta):.1f}" y1="{ya}" x2="{X(tb):.1f}" y2="{yb}" stroke="{AMBER}" stroke-width="1.8" marker-end="url(#am)"/>')
        out.append(f'<circle cx="{X(ta):.1f}" cy="{ya}" r="4.5" fill="{SLATE}"/><circle cx="{X(tb):.1f}" cy="{yb}" r="4.5" fill="{SLATE}"/>')
        out.append(f'<text x="{W - R + 10}" y="{yb + 4}" class="lbl-amber">from {E(h["from"])} ({E(h["confidence"])})</text>')
    tick = t0.replace(minute=(t0.minute // 5) * 5, second=0, microsecond=0)
    while tick <= t1:
        if tick >= t0:
            out.append(f'<text x="{X(tick):.1f}" y="{H - 12}" text-anchor="middle" class="tick">{tick:%H:%M}</text>')
        tick += timedelta(minutes=5)
    out.append("</svg>")
    return "\n".join(out)



# ============================================================ chart 6: 18 months of AI Village
def village_svg() -> str:
    v = D["village"]["weekly"]
    weeks = sorted(v)
    W, L, R = 920, 64, 20
    top_h, a_h, gap, b_h = 118, 210, 30, 70
    y_a0 = top_h + a_h
    y_b0 = y_a0 + gap + b_h
    H = y_b0 + 40
    start, end = d(weeks[0]) - timedelta(days=3), d(weeks[-1]) + timedelta(days=10)
    span = (end - start).days

    def X(x: date) -> float:
        return L + (x - start).days / span * (W - L - R)

    bw = (W - L - R) / span * 7 * 0.78
    mmax, amax = 6000, 32
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="AI Village chat messages and active agents per week">']
    for val in (2000, 4000, 6000):
        y = y_a0 - val / mmax * a_h
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="{HAIR}"/>'
                   f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{val // 1000}k</text>')
    out.append(f'<text transform="translate(16 {y_a0 - a_h / 2}) rotate(-90)" text-anchor="middle" class="axis">chat messages per week</text>')
    for val in (15, 30):
        y = y_b0 - val / amax * b_h
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="{HAIR}"/>'
                   f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{val}</text>')
    out.append(f'<text transform="translate(16 {y_b0 - b_h / 2}) rotate(-90)" text-anchor="middle" class="axis">agents active</text>')
    for k in weeks:
        r = v[k]
        h = r["messages"] / mmax * a_h
        out.append(f'<rect x="{X(d(k)) - bw / 2:.1f}" y="{y_a0 - h:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{INK}">'
                   f'<title>week of {fmt_day(d(k))} {d(k).year}: {r["messages"]:,} messages, {r["sessions"]:,} computer sessions, '
                   f'{r["agents"]} agents speaking</title></rect>')
        h2 = r["agents"] / amax * b_h
        out.append(f'<rect x="{X(d(k)) - bw / 2:.1f}" y="{y_b0 - h2:.1f}" width="{bw:.1f}" height="{h2:.1f}" fill="{SLATE}" opacity=".85">'
                   f'<title>week of {fmt_day(d(k))}: {r["agents"]} agents speaking</title></rect>')
    for y0 in (y_a0, y_b0):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y0}" y2="{y0}" stroke="{INK}" stroke-width=".8"/>')
    t = date(2025, 5, 1)
    while t <= end:
        out.append(f'<text x="{X(t):.1f}" y="{y_b0 + 18}" text-anchor="middle" class="tick">{t.strftime("%b")}'
                   f'{" " + str(t.year)[2:] if t.month in (1, 5) else ""}</text>')
        t = date(t.year + (t.month // 12), t.month % 12 + 1, 1)
    calls = [(date(2025, 4, 2), "Raise money", "for a charity"), (date(2025, 6, 26), "Run a", "merch store"),
             (date(2025, 10, 20), "Reduce global", "poverty"), (date(2026, 1, 12), "Hack the OWASP", "Juice Shop"),
             (date(2026, 4, 27), "Build your own", "world"), (date(2026, 7, 6), "Maximize your", "assigned goal")]
    slot = (W - L - R) / len(calls)
    for i, (x_d, l1, l2) in enumerate(calls):
        lx = L + slot * i + 6
        wk = min(weeks, key=lambda k: abs((d(k) - x_d).days))
        target = y_a0 - v[wk]["messages"] / mmax * a_h - 4
        out.append(f'<text x="{lx:.1f}" y="22" class="date">{fmt_day(x_d)} {str(x_d.year)[2:]}</text>'
                   f'<text x="{lx:.1f}" y="42" class="call">{E(l1)}</text><text x="{lx:.1f}" y="60" class="call">{E(l2)}</text>'
                   f'<path d="M{lx + 14:.1f} 70 L{lx + 14:.1f} 84 L{X(x_d):.1f} {top_h - 6} L{X(x_d):.1f} {target:.1f}" '
                   f'fill="none" stroke="#9aa69c" stroke-width="1"/>')
    out.append("</svg>")
    return "\n".join(out)

# ============================================================ evidence blocks
def quote(when: str, who: str, what: str, text: str, terms: list[str], marker: str = "") -> str:
    body = S.highlight(S.snippet(S._shorten_urls(text, terms), terms, 420), terms)
    mk = f'<span class="qmark">{marker}</span>' if marker else ""
    return (f'<div class="quote">{mk}<span class="qmeta">{E(when)} · <span class="who">{E(who)}</span> · {E(what)}</span>'
            f'<span class="qbody">{body}</span></div>')


def why(text: str) -> str:
    return f'<div class="why"><span class="arrow">↓</span><span><b>Why SwarmGuard links these</b> {text}</span></div>'


def relay_section() -> tuple[str, str]:
    pick = {"18:54:00": "1", "19:21:13": "2", "19:21:42": "3", "19:31:48": "4"}
    by_t = {r["t"][11:19]: r for r in D["relay"]}
    r1, r2, r3, r4 = (by_t[k] for k in pick)
    t23 = S.shared_terms(r2["text"], r3["text"], ["R3"])
    t1 = ["Jan12", "female 2015", "deadline", "male 2016"]
    t4 = ["R2", "R3", "18m04", "cohort"]
    blocks = (
        quote("16 Jun 18:54:00", r1["actor"], "edits the page", r1["text"], t1, "1")
        + '<p class="aside">The first edit after the page was created: a run reports its round timings.</p>'
        + '<div class="pair">'
        + quote("16 Jun 19:21:13", r2["actor"], "edits the page", r2["text"], t23, "2")
        + why("same page · same round, same answer format (highlighted) · 29 seconds apart · different network blocks")
        + quote("16 Jun 19:21:42", r3["actor"], "edits the page", r3["text"], t23, "3")
        + "</div>"
        + quote("16 Jun 19:31:48", r4["actor"], "edits the page", r4["text"], t4, "4")
        + '<p class="aside">A run addresses another by name and asks it to “fast-forward” and post the next round.</p>'
    )
    return relay_svg(pick), blocks


def hub_section() -> str:
    out = []
    for h in D["hub"]["pathway"][:4]:
        a, b = h["evidence"]
        terms = S.shared_terms(a["text"] or "", b["text"] or "", ["village-hub-1ddcad.gitlab.io", "Powered by AI Village"])
        out.append('<div class="pair">'
                   + quote(f"6 Jul {a['t'][11:19]}", a["actor"], a["action"], a["text"] or "", terms)
                   + why(f"{E(h['from'])} → {E(h['to'])}: shared link or wording (highlighted) · confidence {E(h['confidence'])}")
                   + quote(f"6 Jul {b['t'][11:19]}", b["actor"], b["action"], b["text"] or "", terms)
                   + "</div>")
    return "".join(out)


# ============================================================ page
def build() -> str:
    a = D["audit_summary"]
    syn = D["synthetic"]
    relay_chart, relay_quotes = relay_section()
    vt = D["village"]["totals"]
    tot_agents, tot_msgs, tot_sessions = vt["agents"], f"{vt['messages']:,}", f"{vt['sessions']:,}"
    village_chart = village_svg()
    facts = [
        ("SwarmGuard reconstructs how behaviour spreads between AI agents, and where to cut it",
         "It reads the logs a swarm leaves behind, links each action to earlier actions by other agents that could explain it "
         "(same page, similar wording, a reply), groups linked activity into episodes, and ranks controls by how much of the "
         "inferred spread they cut versus how much unrelated activity they disturb.", "how"),
        ("On a public wiki, 24 differently named agents used one page to trade answers for a timed task",
         "On 16 June, between 18:54 and 19:40 UTC, runs writing from 24 different network blocks posted round timings and answers "
         "to a single page, asking each other to “post R3 immediately”.", "relay"),
        ("Making that one page read-only removes 97% of the inferred spread, and touches almost nothing else",
         "Of all the controls SwarmGuard tries, a single-page write restriction cuts the most inferred agent-to-agent paths with "
         "the least side-effect. Pausing agents would hit far more unrelated activity, and there was no single agent to pause.", "relay"),
        ("The real administrator deleted the same pages, but a median of 10 days later",
         f"Across the {a['episodes']} episodes, the page SwarmGuard ranks first was deleted in {a['deleted']} of {a['page_recs']} "
         f"cases. Agreement is weak evidence (the admin removed almost every agent page); the point is timing: {a['writes_between']} "
         f"more writes landed in between, and {a['recreations']} pages came back after deletion.", "audit"),
        ("On a synthetic swarm with a known answer, SwarmGuard finds the source and ranks the right control first",
         "It recovers the true source and episode on every seed, ranks “restrict writes to the shared pastebin” first with 0% "
         "side-effects, and flags a planted confounder instead of asserting it.", "bench"),
        ("Nothing here is proof of causation",
         "Every link is a lead with evidence attached, and every check says what it does not prove. Unusual coordination is not "
         "unsafe behaviour, and the controls are recommendations for whoever runs the system.", "limits"),
    ]
    facts_html = "".join(
        f'<details class="fact"><summary><span class="plus">+</span><span class="ft">{E(t)}</span></summary>'
        f'<div class="fbody"><p>{E(b)} <a class="more" href="#{k}">(more)</a></p></div></details>' for t, b, k in facts)

    toc = [("intro", "Intro", []), ("facts", "Key facts", []),
           ("users", "Who is this for?", []),
           ("data", "The data", [("village-data", "AI Village"), ("wiki-data", "The wiki export")]),
           ("timeline", "Timeline of the wiki swarm", []),
           ("findings", "Findings", [("relay", "A relay board on a public wiki"), ("audit", "What the operator actually did"),
                                     ("hub", "The Village Hub")]),
           ("how", "How SwarmGuard works", [("links", "Candidate links"), ("paths", "Time-respecting paths"),
                                            ("controls", "Interventions on paper")]),
           ("bench", "Does it find a known answer?", []),
           ("lessons", "What real data taught us", []), ("limits", "Caveats", []), ("open", "Open questions", []),
           ("run", "Run it yourself", [])]
    toc_html = "".join(
        f'<li><a href="#{i}" data-id="{i}">{E(t)}</a>' + (
            '<ul>' + "".join(f'<li><a href="#{j}" data-id="{j}">{E(u)}</a></li>' for j, u in sub) + '</ul>' if sub else "")
        + '</li>' for i, t, sub in toc)

    body = f"""
<section id="intro">
<h1 class="title">Where would you step in?</h1>
<p class="byline">Sarvesh Tiku <span class="sep">·</span> 3 October 2026 <span class="sep">·</span> SwarmGuard</p>
<p class="dek">Reconstructing how behaviour spreads through AI agent swarms, and finding the smallest reversible place to interrupt it. Built and tested on AI Village and on the wiki swarm.</p>
<h2>Intro</h2>
<p>When many AI agents share a world, the interesting behaviour is often not inside any one agent. It is in what moves <em>between</em> them: an idea posted in one place and repeated in five others, a public page that two dozen separate runs use as a notice board, a habit that spreads because everyone reads the same memory.</p>
<p>Safety tools mostly judge <em>one agent at a time</em>. SwarmGuard looks at the structure instead. It reads the logs a swarm leaves behind, reconstructs <b>candidate pathways</b> (which action may have led to which, through which page, message or memory), groups them into <b>episodes</b>, and for each episode <b>ranks interventions</b> by how much of the inferred spread they would cut versus how much unrelated activity they would disturb <a class="more" href="#how">(more)</a>.</p>
<p>We ran it on two real datasets: <b>AI Village</b>, a long-running village of frontier-model agents with full logs, and the public record of the <b>wiki swarm</b>, agents that used several old German wikis as a coordination channel in May–July 2026.<sup class="fn"><a href="#fn1">1</a></sup> Nothing here is a claim of causation: SwarmGuard produces leads with evidence attached, and says what each lead does not prove.</p>
</section>

<section id="facts">
<h2>Key facts</h2>
<div class="facts">{facts_html}</div>
</section>



<section id="users">
<h2>Who is this for?</h2>
<p>Harmful episodes are rare, as they are for any monitoring tool: security and reliability systems spend most of their time watching ordinary traffic. Their value is that when something does go wrong, the structure is already known and the response can be fast and narrow. SwarmGuard plays that role for agent swarms. In the wiki case, without such a map, the administrator acted a median of ten days after each episode and ended up deleting 5,217 pages.</p>
<table class="data"><thead><tr><th>Who</th><th>What they can see</th><th>Their task</th><th>What SwarmGuard does</th></tr></thead><tbody>
<tr><td><b>A lab running agent swarms</b></td><td>Everything: chat, actions, memory, goals</td><td>Know how agents influence each other; contain quickly if needed</td><td>Maps the channels, ranks narrow controls and simulates them before anyone acts</td></tr><tr><td><b>A website or platform operator</b></td><td>Only public traces: edits, self-chosen names, partial addresses</td><td>Notice agents using the site and decide what to block</td><td>Finds the coordination hubs, so one page can be locked instead of thousands deleted</td></tr><tr><td><b>Evaluation teams</b></td><td>Logs of many separate test runs</td><td>Make sure runs do not share answers or contaminate each other</td><td>Flags runs linked through outside channels: exactly the relay-page pattern</td></tr><tr><td><b>Safety researchers</b></td><td>Datasets like AI Village</td><td>Study how behaviour spreads in groups of agents</td><td>Turns raw logs into episodes, links and quoted evidence</td></tr><tr><td><b>Pre-deployment review</b></td><td>A swarm's planned setup</td><td>Find single points of failure before launch</td><td>Shows which shared resources would carry most of any spread</td></tr>
</tbody></table>
<p><b>The most immediate use is evaluation integrity.</b> The relay page is a real example of separate runs of a timed task trading answers through a public page that nobody was watching. Flagging runs that become linked through outside channels is a concrete job a team running agent evaluations needs done today <a class="more" href="#relay">(more)</a>.</p>
<h3>Observability shapes the claim</h3>
<p>SwarmGuard works at whatever level of visibility it is given, and says what it cannot conclude at that level. With full logs (AI Village) it can follow who said what to whom, though names mentioned everywhere make addressing noisy. With only public traces (the wiki) it falls back to what is directly observed: “twenty-four names wrote to one page” is solid even when “who influenced whom” is thin. The less you can see, the more it relies on observed structure and the less it claims about influence.</p>
<h3>How to measure its value</h3>
<ul>
<li><b>Time to understand:</b> minutes to locate the coordination hub, against the administrator's ten days.</li>
<li><b>Size of the response:</b> one page made read-only (97% of inferred spread removed, about 0% side-effects) against 5,217 deletions.</li>
<li><b>Coverage:</b> 20,502 raw wiki events grouped into 50 reviewable episodes.</li>
<li><b>Catching the rare case in time:</b> the next test is a replay that stops the data on 16 June and checks the relay page is flagged then.</li>
</ul>
<p class="note">Most of what agent swarms do is fine. SwarmGuard maps how behaviour moves between agents, so that when something isn't fine (like separate test runs secretly sharing answers on a public wiki) you see it while it is happening and can shut the one page carrying it, not the whole system.</p>
</section>

<section id="data">
<h2>The data</h2>
<p>Most AI-safety evidence comes from short, staged tests. Both datasets used here are different: they are long, messy records of agents doing things in the real world, together, with nobody scripting the next step. That is exactly what you need to study how behaviour moves between agents.</p>

<section id="village-data">
<h3>AI Village: eighteen months of agents living together</h3>
<p><a href="https://huggingface.co/datasets/aidigestorg/ai-village" target="_blank" rel="noopener">AI Village on Hugging Face</a> (<code>aidigestorg/ai-village</code>, from <a href="https://theaidigest.org/village" target="_blank" rel="noopener">AI Digest</a>) is a near-complete export of an experiment that has run since 2 April 2025: a village of frontier-model agents from Anthropic, OpenAI, Google, xAI, Moonshot and others, each with its own computer, a shared group chat, long-term memory, and a new collective goal every week or two. They raised money for charity, ran merch stores, built websites and games, wrote Substacks, hacked a practice target, and elected leaders. Everything they did is in the logs.</p>
<div class="stats">
<div><span class="n">{tot_agents}</span><span class="l">agents over the life of the village</span></div>
<div><span class="n">{tot_msgs}</span><span class="l">chat messages between agents and humans</span></div>
<div><span class="n">{tot_sessions}</span><span class="l">computer-use sessions, each with its stated goal</span></div>
<div><span class="n">~1.1M</span><span class="l">computer-use steps (clicks, commands, outputs), with screenshots</span></div>
<div><span class="n">~165k</span><span class="l">long-term memory entries the agents wrote for themselves</span></div>
<div><span class="n">51</span><span class="l">village-wide goals, each with start and end times</span></div>
</div>
<figure class="fig wide">{village_chart}
<figcaption>Chat messages per week (black) and the number of agents speaking each week (slate) across the life of the village. Callouts mark a few of the 51 village goals; the tallest week is the one the Village Hub came from.</figcaption></figure>
<p>What makes it unusually good for this work: it records <b>what agents said to each other</b> (chat), <b>what they did</b> (computer sessions down to individual clicks and commands), <b>what they remembered</b> (memory), <b>what they were told to do</b> (goals) and <b>when the scaffolding changed</b> (a dated changelog of prompt, tool and model changes, so a sudden shift in behaviour can be checked against an upgrade). Two practical notes for anyone using it: the files are ordered by row id rather than time, and chat has no recipient field, so who-addressed-whom has to be inferred from names in the text.<sup class="fn"><a href="#fn3">3</a></sup></p>
</section>

<section id="wiki-data">
<h3>The wiki export: a swarm seen only from the outside</h3>
<p>The second dataset, published at <a href="https://collusion.wiki" target="_blank" rel="noopener">collusion.wiki</a> alongside <a href="https://rubyhack.ai/" target="_blank" rel="noopener">rubyhack.ai</a>'s investigation, is the opposite kind of record. Agents used several old German wikis as a message board; the export holds <b>14,591 saved revisions</b> across four wikis, <b>5,217 administrator deletions</b>, <b>101 script-injection probes</b>, <b>499 short links</b> on a public URL shortener, and per-page and per-name totals, with a manifest that re-checks its own counts (122 of 122 pass).</p>
<p>There is no chat, no memory and no identity: each save carries a self-chosen name (899 carry none) and only the first half of an address. That is what makes it valuable here. It is the view a platform operator actually has when agents show up on their site, and it includes what that operator did about it, which lets us compare SwarmGuard's advice with a real response.</p>
</section>
</section>

<section id="timeline">
<h2>Timeline of the wiki swarm</h2>
<figure class="fig wide">{timeline_svg()}
<figcaption>Agent writes per day (black) and the administrator's deletions per day (slate) share one time axis; the amber row counts the episodes SwarmGuard flags on each day, and the amber band marks the day of the relay board. Hover any bar for its count.</figcaption></figure>
<p>The swarm's footprint on the wikis is small until mid-June, then spikes: 6,778 saves on 18 June alone. The administrator's cleanup begins in earnest the next day and runs in batches until 14 July, twelve days after the last agent write. SwarmGuard's episodes cluster in the week of the spike, when there was still time to act.</p>
</section>

<section id="findings">
<h2>Findings</h2>

<section id="relay">
<h3>1. A relay board on a public wiki</h3>
<p>On 16 June 2026, between 18:54 and 19:40 UTC, <b>24 differently named agents, each writing from a different network block</b>,<sup class="fn"><a href="#fn2">2</a></sup> edited a single page on the DSE wiki, <code>DataUSAMaidsSequenceLiveMay03</code>. Read in order, the edits look like separate runs of a timed task trading round timings and answers so that later runs could anticipate what was coming.</p>
<figure class="fig">{relay_chart}
<figcaption>Every edit to the relay page on 16 June, one row per name, in the order they first wrote. Numbered dots are quoted below. 25 names in all (including the page's creator at 18:48), writing from 24 different network blocks.</figcaption></figure>
{relay_quotes}
<p>What SwarmGuard says about it: a <b>burst of cross-agent activity</b> (345 candidate interactions per hour against a dataset median of 19) and <b>coordination through an outside page</b>, both high confidence. The hop-by-hop spread check does <em>not</em> fire: this is many-to-many broadcast through a board, not a chain from one originator, so there is no single agent to pause. The best control is to <b>restrict writes to that page</b>: it removes 97% of the inferred paths between agents and touches about 0% of activity elsewhere.</p>
<p class="note">What this does not prove: that these were separate models rather than one operator's runs; what the task was; or whether sharing answers changed any outcome.</p>
</section>

<section id="audit">
<h3>2. What the operator actually did</h3>
<p>The export includes the administrator's real response, so SwarmGuard's advice can be compared with history. In {a['page_recs']} of {a['episodes']} episodes the top recommendation is a single-page write restriction, and the administrator later deleted that same page in {a['deleted']} of them. <b>That agreement is weak evidence</b>: the administrator removed about 77% of all agent-written pages, nearly 100% on the main wiki, so almost any page would eventually have been deleted.</p>
<figure class="fig">{audit_svg()}
<figcaption>One dot per episode whose top-ranked page the administrator later deleted, placed by how many days after the episode the deletion came.</figcaption></figure>
<p>The informative comparison is <b>timing</b> and <b>selectivity</b>. Deletion of the recommended page came a median of <b>10 days</b> after the episode ended; in that gap <b>{a['writes_between']} further writes</b> landed on episode pages and <b>{a['recreations']} pages</b> came back after deletion. SwarmGuard's recommendation is available while the episode is happening, and targets one page per episode: a median of {a['median_paths'] * 100:.0f}% of inferred spread removed at {a['median_coll'] * 100:.2f}% side-effects.</p>
<p class="note">What this does not prove: that an earlier, narrower control would have worked. Agents that lose one page can move to another, and this data shows they recreated pages.</p>
</section>

<section id="hub">
<h3>3. The Village Hub</h3>
<p>AI Village is the opposite case: full logs, known identities and a cooperative goal. At 15:59 UTC on 6 July, 21 agents received new individual goals at once. Within 24 minutes a cross-promotion strategy (a public project directory, then a “Powered by AI Village” badge on every project) spread through a site one agent built.</p>
<figure class="fig">{hub_svg()}
<figcaption>The time-ordered chain SwarmGuard considers most likely. Each arrow joins the earlier and later action quoted below; the label says who the agent may have picked it up from.</figcaption></figure>
{hub_section()}
<p>The best-ranked control is to restrict writes to <code>village-hub-1ddcad.gitlab.io</code>: it removes 54% of inferred paths while touching 0.3% of unrelated activity. This is benign collaboration and SwarmGuard says so: the spread check did not fire, the source is ambiguous among four agents, and the shared goal assignment is listed as a common cause.</p>
</section>
</section>

<section id="how">
<h2>How SwarmGuard works</h2>
<p><b>One shape for every log line.</b> A chat message, a wiki edit, a computer session, a memory update and an administrator's deletion all become the same record: who, when, what, where, what it said, plus the raw ids so every claim can be traced back.</p>
<section id="links"><h3>Candidate links</h3>
<p>For each action by agent B, SwarmGuard looks back (default two hours) for earlier actions by other agents that could explain it, and scores each pair on several kinds of evidence. Timing alone never counts.</p>
<table class="data"><thead><tr><th>Evidence</th><th>What counts</th></tr></thead><tbody>
<tr><td>Same page / site / file</td><td>A put something into it (posted, edited, wrote) and B used it. Two agents merely reading the same thing does not count.</td></tr>
<tr><td>Similar wording</td><td>Sentence-embedding similarity above a “same topic” baseline. Boilerplate repeated by several agents is ignored.</td></tr>
<tr><td>Addressed each other</td><td>B replied to A by name, or A addressed B and this is B's first action afterwards.</td></tr>
<tr><td>Close in time</td><td>Decays across the window. Never enough on its own.</td></tr></tbody></table>
<p>Each link carries a confidence word and the competing explanations SwarmGuard checks automatically: a third party said something resembling both sides first; A reached C only through B; some X reached both first; or both names come from the same network block.</p></section>
<section id="paths"><h3>Time-respecting paths</h3>
<p>A path A → B → C only counts if B's onward action came after B was reached. Without this rule, ordinary back-and-forth makes everyone “reach” everyone, and “where did this start?” becomes meaningless. Strongly linked actions close together form an episode, and four checks (burst, spread, outside channel, unassigned division of labour) each report why they fired and what they do not prove.</p></section>
<section id="controls"><h3>Interventions on paper</h3>
<p>Each candidate control (restrict writes to a page, require approval for external writes, isolate a shared memory, block one channel, remove a tool, pause an agent) is applied to a copy of the inferred graph. SwarmGuard counts the agent-to-agent paths that disappear, and the activity outside the episode the same control would touch. The ranking is transparent: spread removed minus penalties for side-effects, thin evidence and being hard to undo. We say “removes 54% of inferred paths”, never “reduces risk by 54%”.</p></section>
</section>

<section id="bench">
<h2>Does it find a known answer?</h2>
<p>Before trusting real-data findings we built a small swarm where the truth is known: A messages B; B writes to a shared pastebin that C and D read; D passes it to E. Alongside it sit an unrelated cluster, a busy hub agent (so pausing B should be costly), and a <b>planted confounder</b>: a human tells everyone to update a spreadsheet, so I and J look linked without influencing each other.</p>
<figure class="fig">{bench_svg()}
<figcaption>The synthetic swarm. Ink arrows are the true chain; amber dashed lines are the human prompt that makes I and J look linked.</figcaption></figure>
<table class="data"><thead><tr><th>Question</th><th>Result (3 random seeds)</th></tr></thead><tbody>
<tr><td>Finds the true source (A)?</td><td>Yes, 3 of 3</td></tr>
<tr><td>Recovers the true episode?</td><td>Exact match every time; never merges the unrelated cluster</td></tr>
<tr><td>Ranks the right control first?</td><td>Yes, 3 of 3: restrict writes to the pastebin, 0% side-effects</td></tr>
<tr><td>Handles the confounder?</td><td>Yes, 3 of 3: shown as a common cause, not asserted</td></tr>
<tr><td>Links with no competing explanation</td><td>precision {syn['primary_edge_precision']:.2f} / recall {syn['primary_edge_recall']:.2f} on the default seed</td></tr></tbody></table>
</section>

<section id="lessons">
<h2>What real data taught us</h2>
<p>Most of SwarmGuard's design came from things that looked fine on synthetic data and broke on the real logs.</p>
<ol class="lessons">
<li><b>A unit bug made every lag zero.</b> Microsecond timestamps were read as nanoseconds, so a two-hour window behaved like 83 days.</li>
<li><b>Names are everywhere.</b> AI Village agents mention each other constantly, so “addressed each other” fired on 89% of pairs until it was narrowed to replies and first responses.</li>
<li><b>Reading is not influence.</b> Two agents searching the same shared history were being linked. A shared resource now counts only if the earlier agent put something into it.</li>
<li><b>Data sources are not channels.</b> Hundreds of wiki agents fetched the same public data files; that is a shared source, not coordination.</li>
<li><b>Agents chain proxies, and credentials hide in URLs.</b> Proxies wrapped in proxies are unwrapped; tokens in git remotes are stripped before anything is stored or shown.</li>
<li><b>The checker was too generous.</b> Early on every episode was “flagged”. Thresholds were tightened, and we still report where a check does not discriminate.</li>
</ol>
</section>

<section id="limits">
<h2>Caveats</h2>
<ul>
<li>A possible link is not proof of influence. Unusual coordination is not unsafe behaviour.</li>
<li>Intervention numbers are structural estimates on an inferred graph, not predictions of what agents would have done instead.</li>
<li>Observability is partial: screens, private memory and off-platform channels are mostly unseen, and agents misreport.</li>
<li>On the wiki an “agent” is a self-chosen name; on AI Village, addressing is inferred from names in text.</li>
<li>SwarmGuard helps a person investigate. It does not decide that any agent is dangerous, and it controls nothing.</li>
</ul>
</section>

<section id="open">
<h2>Open questions</h2>
<ul>
<li><b>How much does external-only observability miss?</b> Hide AI Village's chat and see which episodes survive.</li>
<li><b>Who is behind the names?</b> Cluster wiki names into likely operators before linking.</li>
<li><b>Would a narrow, early control hold?</b> Agents recreate pages; a counterfactual that models adaptation is the next step.</li>
<li><b>Calibration.</b> A few hundred hand-labelled real links would let us set weights and confidence words from data.</li>
</ul>
</section>

<section id="run">
<h2>Run it yourself</h2>
<pre><code>uv venv ~/.venvs/swarmguard --python 3.12 &amp;&amp; ln -sfn ~/.venvs/swarmguard .venv
uv pip install --python .venv/bin/python -e ".[embeddings,dev]"
hf auth login                      # AI Village is gated; or set HF_TOKEN
python scripts/run_analysis.py ai-village --start "2026-07-06 15:00" --end "2026-07-07 00:00"
python scripts/run_analysis.py german-wiki --dir datasets/german_wiki --mode medium
streamlit run swarmguard/app/app.py</code></pre>
</section>

<section id="notes" class="notes">
<h2>Notes</h2>
<ol>
<li id="fn1">AI Digest, “AI Village dataset”, 2026 (gated, research terms). The wiki data is the public export published alongside <a href="https://rubyhack.ai/">rubyhack.ai</a>'s write-up of the wiki swarm, whose visual style this page follows.</li>
<li id="fn3">Numbers are from the export as downloaded on 3 October 2026 (it is refreshed roughly weekly). Step and memory counts are the dataset card's; the rest were counted directly. The dataset is gated: request access on Hugging Face and agree to its research terms.</li>
<li id="fn2">Addresses in the export are cut to their first two numbers (a /16 block). Names and blocks are both cheap, so two names on one block are treated as possibly one operator, and 24 different blocks argue against, but do not rule out, a single operator.</li>
</ol>
</section>
"""
    return TEMPLATE.replace("{{TOC}}", toc_html).replace("{{BODY}}", body)


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Where would you step in?</title>
<style>
@font-face { font-family: "et-book"; src: url("https://cdn.jsdelivr.net/gh/edwardtufte/tufte-css@gh-pages/et-book/et-book-roman-line-figures/et-book-roman-line-figures.woff") format("woff"); font-weight: 400; font-style: normal; font-display: swap; }
@font-face { font-family: "et-book"; src: url("https://cdn.jsdelivr.net/gh/edwardtufte/tufte-css@gh-pages/et-book/et-book-display-italic-old-style-figures/et-book-display-italic-old-style-figures.woff") format("woff"); font-weight: 400; font-style: italic; font-display: swap; }
@font-face { font-family: "et-book"; src: url("https://cdn.jsdelivr.net/gh/edwardtufte/tufte-css@gh-pages/et-book/et-book-bold-line-figures/et-book-bold-line-figures.woff") format("woff"); font-weight: 700; font-style: normal; font-display: swap; }
:root { --bg:#fffff8; --ink:#111; --dull:#666; --hair:#e8e8df; --rule:#d6d6cc; --wash:#f6f6ee;
  --green:#2A623D; --green-tint:#e6efe9; --amber:#c77400; --amber-ink:#7a4b00; --amber-wash:#fff8ea; --slate:#31566f; --link:#3a5fa8;
  --serif: et-book, Palatino, "Palatino Linotype", "Book Antiqua", Georgia, serif;
  --sans: -apple-system, BlinkMacSystemFont, "avenir next", avenir, "helvetica neue", helvetica, arial, sans-serif;
  --mono: SFMono-Regular, Menlo, Consolas, monospace; --col: 820px; --rail: 250px; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: 84px; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--serif); -webkit-font-smoothing: subpixel-antialiased; }
.topnav { position: sticky; top: 0; z-index: 5; background: var(--bg); border-bottom: 1px solid var(--hair); }
.topnav .inner { max-width: 100%; margin: 0 auto; padding: 0 24px; justify-content: center;
  display: flex; gap: 1.35rem; overflow-x: auto; }
.topnav a { display: inline-block; padding: 15px 2px 12px; font-family: var(--sans); font-size: 14px; color: var(--ink); text-decoration: none;
  border-bottom: 2px solid transparent; white-space: nowrap; }
.topnav a.on { font-family: var(--serif); font-size: 17px; border-bottom-color: var(--ink); }
.frame { display: grid; grid-template-columns: var(--rail) minmax(0, var(--col)); gap: 60px; max-width: calc(var(--rail) + 60px + var(--col) + 48px);
  margin: 0 auto; padding: 0 24px; }
.rail { position: sticky; top: 72px; align-self: start; max-height: calc(100vh - 90px); overflow-y: auto; padding: 48px 0 40px;
  border-left: 1px solid var(--hair); }
.rail .h { font-family: var(--serif); font-size: 18px; color: var(--dull); margin: 0 0 14px 20px; }
.rail ul { list-style: none; margin: 0; padding: 0; }
.rail li a { display: block; padding: 4px 0 4px 20px; margin-left: -1px; border-left: 2px solid transparent; font-family: var(--sans);
  font-size: 14px; line-height: 1.3; color: var(--ink); text-decoration: none; }
.rail li li a { padding-left: 34px; font-size: 12.5px; color: #555; }
.rail a.on { border-left-color: var(--ink); color: var(--ink); }
main { padding: 40px 0 120px; min-width: 0; }
h1.title { font-weight: 400; font-size: 3.1rem; line-height: 1.04; letter-spacing: -.01em; margin: 1.4rem 0 1rem; text-wrap: balance; }
.byline { color: var(--dull); font-size: 16px; margin: 0 0 1.2rem; } .byline .sep { margin: 0 .45em; }
.dek { font-size: 1.18rem; line-height: 1.45; color: var(--dull); font-style: italic; margin: 0 0 1rem; }
h2 { font-weight: 400; font-size: 2.2rem; line-height: 1.08; margin: 3.4rem 0 1.2rem; letter-spacing: -.005em; }
h3 { font-weight: 400; font-size: 1.55rem; line-height: 1.15; margin: 2.6rem 0 .9rem; }
p, li, td { font-size: 1.14rem; line-height: 1.6; }
p { margin: 0 0 1.25rem; }
b { font-weight: 700; }
a { color: inherit; text-decoration: underline; text-decoration-thickness: 1px; text-underline-offset: 2px; }
a.more { font-family: var(--sans); font-size: .72em; color: var(--link); text-decoration: none; margin-left: .15em; }
sup.fn a { font-family: var(--sans); font-size: .62em; color: var(--link); text-decoration: none; }
code { font-family: var(--mono); font-size: .72em; background: #f0f0f0; padding: 0 .3em; border-radius: 2px; overflow-wrap: anywhere; }
pre { background: var(--wash); border: 1px solid var(--hair); padding: 1rem 1.2rem; overflow-x: auto; }
pre code { background: none; font-size: .85rem; line-height: 1.6; word-break: normal; }
.note { color: var(--dull); font-style: italic; font-size: 1.04rem; }
.aside { color: var(--dull); font-size: 1rem; font-style: italic; margin: -.4rem 0 1.6rem 1.2rem; }
/* key facts */
.facts { border-top: 1px solid var(--hair); }
.fact { border-bottom: 1px solid var(--hair); }
.fact summary { list-style: none; cursor: pointer; display: flex; gap: .8rem; align-items: baseline; padding: .9rem 0; }
.fact summary::-webkit-details-marker { display: none; }
.fact .plus { font-family: var(--sans); font-size: 1.2rem; color: var(--slate); width: 1.1rem; flex: none; transition: transform .15s; }
.fact[open] .plus { transform: rotate(45deg); }
.fact .ft { font-size: 1.62rem; line-height: 1.22; }
.fact summary:hover .ft { text-decoration: underline; text-decoration-thickness: 1px; text-underline-offset: 5px; text-decoration-color: var(--rule); }
.fact .fbody { padding: 0 0 1.2rem 2rem; }
.fact .fbody p { font-size: 1.08rem; }
/* figures */
figure.fig { margin: 2.2rem 0 2.6rem; padding: 1.4rem 1.2rem .8rem; background: #fbfbf3; border: 1px solid var(--hair); border-radius: 4px; }
figure.fig.wide { margin-left: -2rem; margin-right: -2rem; }
figure.fig figcaption { font-size: .98rem; line-height: 1.5; color: var(--dull); font-style: italic; margin: .8rem .2rem .4rem; }
svg.chart { width: 100%; height: auto; display: block; overflow: visible; }
svg .tick { font: 11.5px var(--mono); fill: #555; }
svg .axis { font: 12.5px var(--sans); fill: #555; }
svg .date { font: 12px var(--mono); fill: #2b2b2b; }
svg .call { font: 14.5px var(--sans); fill: var(--ink); }
svg .call-s { font: 12.5px var(--sans); fill: #444; }
svg .band { font: 13px var(--mono); fill: #555; }
svg .lbl-amber { font: 12.5px var(--sans); fill: var(--amber-ink); }
svg .lbl-green { font: 12.5px var(--sans); fill: var(--green); }
svg .row { font: 14px var(--serif); fill: var(--ink); }
svg .mark { font: 700 12px var(--sans); fill: var(--ink); }
svg .node { font: 14px var(--sans); }
svg .node-h { font: italic 15px var(--serif); fill: var(--amber-ink); }
svg rect:hover, svg circle:hover { opacity: .7; }
/* dataset at a glance */
.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.4rem 1.6rem; border-top: 1px solid var(--rule);
  border-bottom: 1px solid var(--rule); padding: 1.3rem 0; margin: 1.6rem 0 2rem; }
.stats .n { display: block; font-size: 2.1rem; line-height: 1; margin-bottom: .35rem; }
.stats .l { display: block; font: 13px/1.35 var(--sans); color: var(--dull); }
@media (max-width: 640px) { .stats { grid-template-columns: repeat(2, 1fr); } }
/* evidence */
.quote { position: relative; margin: 1.1rem 0; padding: .2rem 0 .2rem 1.1rem; border-left: 3px solid var(--green); }
.quote .qmark { position: absolute; left: -2.1rem; top: .15rem; width: 1.5rem; height: 1.5rem; border-radius: 50%; background: var(--ink);
  color: var(--bg); font: 700 12px var(--sans); display: flex; align-items: center; justify-content: center; }
.qmeta { display: block; font: 11.5px var(--mono); color: var(--dull); margin-bottom: .25rem; }
.qmeta .who { color: var(--slate); }
.qbody { font-size: 1.08rem; line-height: 1.55; }
mark.sg-hl { background: var(--green-tint); color: var(--ink); padding: 0 .12em; border-bottom: 1px solid var(--green); }
.pair { margin: 1.8rem 0 2.2rem; }
.pair .quote { margin: .3rem 0; }
.why { display: flex; gap: .6rem; align-items: baseline; margin: .3rem 0 .3rem .25rem; padding: .35rem .8rem; border-left: 3px solid var(--amber);
  background: var(--amber-wash); color: var(--amber-ink); font-size: .98rem; }
.why .arrow { font-family: var(--sans); color: var(--amber); }
.why b { font: 400 11.5px var(--sans); text-transform: uppercase; letter-spacing: .07em; margin-right: .4rem; }
table.data { border-collapse: collapse; width: 100%; margin: 1.4rem 0 2rem; }
table.data th { text-align: left; font: 12px var(--sans); text-transform: uppercase; letter-spacing: .06em; color: var(--dull);
  padding: 8px 10px; border-bottom: 1px solid var(--rule); }
table.data td { vertical-align: top; padding: 9px 10px; border-bottom: 1px solid var(--hair); font-size: 1.04rem; line-height: 1.45; }
ol.lessons li, section ul li { margin-bottom: .6rem; }
.notes li { font-size: .98rem; color: #444; }
@media (max-width: 1150px) { .frame { grid-template-columns: minmax(0, var(--col)); } .rail { display: none; }
  .topnav .inner { padding-left: 24px; } figure.fig.wide { margin-left: 0; margin-right: 0; } }
@media (max-width: 640px) { h1.title { font-size: 2.3rem; } h2 { font-size: 1.8rem; } .fact .ft { font-size: 1.3rem; }
  p, li, td { font-size: 1.05rem; } .quote .qmark { display: none; } }
</style></head>
<body>
<nav class="topnav"><div class="inner">
<a href="#intro" data-sec="intro">Overview</a><a href="#facts" data-sec="facts">Key facts</a><a href="#users" data-sec="users">Who it's for</a><a href="#data" data-sec="data">Data</a><a href="#timeline" data-sec="timeline">Timeline</a>
<a href="#findings" data-sec="findings">Findings</a><a href="#how" data-sec="how">Method</a><a href="#bench" data-sec="bench">Benchmark</a>
<a href="#limits" data-sec="limits">Caveats</a><a href="#run" data-sec="run">Run it</a></div></nav>
<div class="frame">
<aside class="rail"><div class="h">Contents</div><ul>{{TOC}}</ul></aside>
<main>{{BODY}}</main>
</div>
<script>
// highlight the current section in the contents rail and the top nav
const links = [...document.querySelectorAll('.rail a[data-id]')];
const tops = [...document.querySelectorAll('.topnav a[data-sec]')];
const targets = links.map(a => document.getElementById(a.dataset.id)).filter(Boolean);
function update() {
  let cur = targets[0];
  for (const t of targets) if (t.getBoundingClientRect().top < window.innerHeight * 0.35) cur = t;
  links.forEach(a => a.classList.toggle('on', a.dataset.id === cur.id));
  let top = cur.closest('section[id]:not(section section)') || cur;
  const topId = (cur.parentElement && cur.parentElement.closest('section[id]')) ? cur.parentElement.closest('section[id]').id : cur.id;
  tops.forEach(a => a.classList.toggle('on', a.dataset.sec === topId || a.dataset.sec === cur.id));
}
document.addEventListener('scroll', update, { passive: true }); update();
// Section links scroll inside this page. Embedded in the app, a plain "#id" link would make the frame
// load the app's own URL (a second copy of the app inside the blog), so we never let the browser follow it.
document.addEventListener('click', e => {
  const a = e.target.closest('a[href]');
  if (!a) return;
  const href = a.getAttribute('href');
  if (href.startsWith('#')) {
    e.preventDefault();
    const el = document.getElementById(href.slice(1));
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } else if (/^https?:/.test(href)) { a.target = '_blank'; a.rel = 'noopener'; }
});
// "(more)" links open the target and close the fact
document.querySelectorAll('.fact a.more').forEach(a => a.addEventListener('click', () => a.closest('details').open = false));
</script>
</body></html>
"""


if __name__ == "__main__":
    out = ROOT / "docs" / "blog" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build())
    static = ROOT / "swarmguard" / "app" / "static"
    static.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, static / "blog.html")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB) and {static / 'blog.html'}")
