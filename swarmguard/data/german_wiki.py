"""German Wiki (collusion.wiki export) adapter: the external-observability case.

Only public artifacts are visible here: wiki saves, admin deletions, script
probes, shortener links. No internal agent state is available. Schema verified
against the real export (manifest exporter ``explorer-schema-2``, cut
``revision.write_date >= 2026-05-01``) and documented in ``docs/german_wiki_schema.md``.

Files (a directory holding the ``.jsonl.gz`` downloads or the unzipped
``full-wiki-logs/`` with plain ``.jsonl``; both layouts work):

  revisions   one stored save each, with full body + diff hunks      -> edit / create events
  events      save | delete | revert | probe                         -> delete, revert, probe events
              (saves duplicate revisions and are skipped)
  pages       per-page totals (deletions, recreations, labels)       -> page metadata / context
  labels      per-label totals, ``is_human_handle``                  -> actor types
  shortener-logs.json.gz   499 rmn.re links made during the incident -> create events
  other-wikis.json.gz      8 pages on 3 other wikis (no labels)      -> edit events
  records / links / *.csv  cross-site text corpus and coverage (no timestamps) -> context only

Identity is weak by construction. ``label`` is a self-chosen per-save preference
name (899 saves have none), and addresses are cut to /16. Actors are
``label`` (or ``anon@<ip16>``); the /16 travels as ``metadata.origin`` so that
propagation can flag "same address block on both sides" as an alternative
explanation (one operator under two labels).
"""
from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

from .normalization import clip, extract_urls, parse_ts, url_resource_key
from .schemas import Event, events_to_frame

# Public fetch/CORS/markdown proxies the agents routed requests through: treated as tools.
PROXY_HOSTS = ("jqp.vercel.app", "r.jina.ai", "allorigins.hexlet.app", "api.allorigins.win", "md.succ.ai",
               "markdown.new", "pure.md", "jsonhero.io", "cors.bwa.workers.dev", "corsproxy.io", "r.jina.ai")
WIKI_HOSTS = ("wikiservice.at", "prowiki.org")
# UseMod/ProWiki CamelCase links. No nested quantifiers: the obvious ``(?:[A-Z][A-Za-z0-9]*)+``
# backtracks catastrophically on long tokens followed by "_".
_WIKIWORD = re.compile(r"\b([A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*)\b")
_BRACKET = re.compile(r"\[\[([^\]|#]+)")


def _find(directory: Path, stem: str) -> Path | None:
    for cand in (directory / f"{stem}.jsonl.gz", directory / f"{stem}.jsonl",
                 directory / "full-wiki-logs" / f"{stem}.jsonl", directory / f"{stem}.json.gz"):
        if cand.exists():
            return cand
    return None


def _iter(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _load_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def profile_dir(directory: str | Path, n: int = 2) -> None:
    """Print keys and sample rows for each file (schema check)."""
    d = Path(directory)
    for stem in ("pages", "revisions", "events", "labels", "records", "links"):
        p = _find(d, stem)
        if not p:
            continue
        rows = []
        for r in _iter(p):
            rows.append(r)
            if len(rows) >= 300:
                break
        print(f"\n[{p.name}] keys: {dict(Counter(k for r in rows for k in r))}")
        for r in rows[:n]:
            print("  ", {k: repr(v)[:100] for k, v in r.items()})


@dataclass
class WikiSlice:
    events: pd.DataFrame
    context: dict[str, Any]
    pages: pd.DataFrame


def page_key(wiki: str, name: str) -> str:
    return f"{wiki}~{name}"


# Default text the wiki inserts into a new page ("Describe the new page here."): boilerplate, not content.
PLACEHOLDERS = ("Beschreibe hier die neue Seite.",)


def _added_text(body: str | None, hunks: list | None) -> str:
    """Lines this revision added (from difflib hunks), minus wiki placeholder text."""
    text = _added_raw(body, hunks)
    return "\n".join(l for l in text.split("\n") if l.strip() not in PLACEHOLDERS).strip()


def _added_raw(body: str | None, hunks: list | None) -> str:
    if not body:
        return ""
    lines = body.split("\n")
    if not hunks:
        return body
    out = []
    for h in hunks:
        if h.get("op") in ("insert", "replace"):
            out.extend(lines[h.get("b0", 0): h.get("b1", 0)])
    return "\n".join(out)


def _wiki_url_page(url: str) -> str | None:
    """``https://www.wikiservice.at/dse/wiki.cgi?PageName`` -> ``dse~PageName``."""
    try:
        p = urlparse(url)
    except ValueError:
        return None
    host = (p.hostname or "").removeprefix("www.")
    if host not in WIKI_HOSTS or not p.path.endswith("wiki.cgi"):
        return None
    segs = [s for s in p.path.split("/") if s]
    wiki = segs[-2].lower() if len(segs) >= 2 else None
    q = unquote(p.query or "")
    if not wiki or not q:
        return None
    qs = parse_qs(q)
    name = (qs.get("id") or qs.get("page") or [None])[0]
    if name is None and "=" not in q.split("&")[0].split(";")[0]:
        name = q.split("&")[0].split(";")[0]
    return page_key(wiki, name) if name else None


def _proxied_target(url: str) -> str | None:
    """Target URL embedded in a proxy URL: ``?url=...`` / ``?u=...`` or a path like ``/https:/host/...``."""
    p = urlparse(url)
    qs = parse_qs(p.query)
    for k in ("url", "u", "target", "src"):
        if qs.get(k):
            t = unquote(qs[k][0])
            return t if t.startswith("http") else "https://" + t
    m = re.search(r"/(https?):/+(.+)$", p.path)
    if m:
        return f"{m.group(1)}://{m.group(2)}"
    m = re.match(r"^/((?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+/.*)$", p.path)  # md.succ.ai/www.sec.gov/...
    return f"https://{m.group(1)}" if m else None


class _Refs:
    """Resolve references in added text to known pages, external resources, and proxy tools."""

    def __init__(self, known_pages: set[str]) -> None:
        self.known = known_pages
        self.last_external: list[str] = []
        self.by_name: dict[str, list[str]] = {}
        for k in known_pages:
            self.by_name.setdefault(k.split("~", 1)[1], []).append(k)

    def __call__(self, text: str, wiki: str, self_key: str) -> tuple[list[str], str | None, list[str]]:
        # Channels = media agents can write and others read: wiki pages (and short links).
        # Public data URLs many agents fetch (sec.gov, datausa) are shared *sources*, not
        # channels; they are kept as ``external_refs`` evidence only.
        res: list[str] = []
        ext: list[str] = []
        tool = None
        names = set(_WIKIWORD.findall(text)) | {m.strip() for m in _BRACKET.findall(text)}
        for nm in names:
            cands = self.by_name.get(nm, [])
            k = page_key(wiki, nm) if page_key(wiki, nm) in self.known else (cands[0] if len(cands) == 1 else None)
            if k and k != self_key:
                res.append(f"wiki:{k}")
        urls = extract_urls(text, limit=25)
        for u in urls:
            wk = _wiki_url_page(u)
            if wk:
                if wk != self_key:
                    res.append(f"wiki:{wk}")
                continue
            host = (urlparse(u).hostname or "").removeprefix("www.")
            if host in PROXY_HOSTS:
                # The proxy is the tool; the resource is the URL it was asked to fetch.
                tool = tool or f"proxy:{host}"
                for _ in range(3):  # agents chain proxies (jqp -> allorigins -> sec.gov)
                    u = _proxied_target(u)
                    if u is None or (urlparse(u).hostname or "").removeprefix("www.") not in PROXY_HOSTS:
                        break
                if u is None:
                    continue
            key = url_resource_key(u)
            if key:
                ext.append(key)
        self.last_external = list(dict.fromkeys(ext))
        return list(dict.fromkeys(res)), tool, urls[:10]


def load_dir(directory: str | Path, start: str | None = None, end: str | None = None,
             max_events: int | None = 100_000, include_probes: bool = True) -> WikiSlice:
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(d)
    rev_p, ev_p, pages_p, labels_p = (_find(d, s) for s in ("revisions", "events", "pages", "labels"))
    if rev_p is None:
        raise FileNotFoundError(f"no revisions(.jsonl[.gz]) in {d} or {d / 'full-wiki-logs'}")
    pages = pd.DataFrame(list(_iter(pages_p))) if pages_p else pd.DataFrame()
    labels = {r["label"]: r for r in _iter(labels_p)} if labels_p else {}
    humans = {k for k, v in labels.items() if v.get("is_human_handle")}
    known = set(pages["page_key"]) if len(pages) else set()
    refs = _Refs(known)
    s_ts, e_ts = parse_ts(start), parse_ts(end)
    evs: list[Event] = []

    def actor_of(label: str | None, ip16: str | None) -> tuple[str, str]:
        if label:
            return label, ("human" if label in humans else "agent")
        return f"anon@{ip16 or '?'}", "agent"

    def keep(ts) -> bool:
        return ts is not None and (s_ts is None or ts >= s_ts) and (e_ts is None or ts < e_ts)

    # --- revisions: one stored save each ---------------------------------------
    for r in _iter(rev_p):
        ts = parse_ts(r.get("time"))
        if not keep(ts):
            continue
        actor, atype = actor_of(r.get("label"), r.get("ip16"))
        added = _added_text(r.get("body"), r.get("hunks"))
        res, tool, urls = refs(added, r["wiki"], r["page_key"])
        recreation = r.get("relation_type") == "first_recreation_of"
        action = "create" if (r.get("seq") == 1 or r.get("diff_base_reason") == "page_created" or recreation) else "edit"
        evs.append(Event(
            event_id=f"rev:{r['rev_id']}", timestamp=ts, actor_id=actor, actor_type=atype, action_type=action,
            target_id=f"wiki:{r['page_key']}", target_type="resource",
            content=clip(added, 1500), source=f"german_wiki:revisions:{r['wiki']}",
            metadata={"raw_id": r["rev_id"], "raw_table": "revisions", "actor_name": actor, "label": r.get("label"),
                      "origin": r.get("ip16"), "resources": res, "urls": urls, "tool": tool, "mentions": [],
                      "external_refs": refs.last_external, "wiki": r["wiki"], "page": r["name"], "seq": r.get("seq"), "time_grade": r.get("time_grade"),
                      "change_summary": r.get("change_summary"), "recreation_of": r.get("related_event_id") if recreation else None,
                      "target_label": r["name"]},
        ))
    # --- events: deletions (admin), reverts (recreations w/o stored revision), probes ----
    if ev_p:
        for r in _iter(ev_p):
            et = r.get("event_type")
            if et == "save":
                continue
            ts = parse_ts(r.get("time"))
            if not keep(ts) or (et == "probe" and not include_probes):
                continue
            actor, atype = actor_of(r.get("actor_label"), r.get("ip16"))
            pk = r.get("page_key")
            if et == "delete":
                action, content = "delete", f"[deleted page {r.get('page')}] {r.get('change_summary') or ''}".strip()
            elif et == "revert":
                action, content = "create", f"[recreated after deletion] {r.get('change_summary') or ''}".strip()
            else:  # probe: script-injection family request; no body
                action, content = "visit", ""
            evs.append(Event(
                event_id=f"ev:{r['event_id']}", timestamp=ts, actor_id=actor, actor_type=atype, action_type=action,
                target_id=f"wiki:{pk}" if pk else None, target_type="resource" if pk else None, content=content,
                source=f"german_wiki:events:{et}",
                metadata={"raw_id": r["event_id"], "raw_table": "events", "event_type": et, "actor_name": actor,
                          "origin": r.get("ip16"), "resources": [], "mentions": [], "page": r.get("page"),
                          "tool": "script_probe" if et == "probe" else None, "param_family": r.get("param_family"),
                          "recreation_of": r.get("related_event_id") if et == "revert" else None,
                          "source_refs": r.get("source_refs")},
            ))
    # --- shortener links (rmn.re) ---------------------------------------------------
    sp = d / "shortener-logs.json.gz"
    n_short = 0
    if sp.exists():
        for site in _load_json_gz(sp).get("sites", []):
            for k, l in enumerate(site.get("links", [])):
                ts = parse_ts(l.get("time"))
                if not keep(ts):
                    continue
                n_short += 1
                actor = f"anon@{l.get('ip16') or '?'}"
                dest = l.get("url") or ""
                res, tool, _ = refs(dest, "dse", "")
                evs.append(Event(
                    event_id=f"short:{site.get('site')}:{l.get('keyword')}:{k}", timestamp=ts, actor_id=actor,
                    actor_type="agent", action_type="create", target_id=f"{site.get('site')}/{l.get('keyword')}",
                    target_type="resource", content=clip(f"short link → {dest} {l.get('title') or ''}", 600),
                    source="german_wiki:shortener", metadata={
                        "raw_id": l.get("keyword"), "raw_table": "shortener-logs", "actor_name": actor,
                        "origin": l.get("ip16"), "resources": res, "tool": tool or "url_shortener", "mentions": [],
                        "clicks": l.get("clicks")}))
    # --- other wikis (no labels; MediaWiki anonymous edits) -----------------------------
    op = d / "other-wikis.json.gz"
    if op.exists():
        for pg in _load_json_gz(op).get("pages", []):
            for rv in pg.get("revisions", []):
                ts = parse_ts(rv.get("time"))
                if not keep(ts):
                    continue
                added = "\n".join(rv.get("added") or [])
                res, tool, urls = refs(added, pg["wiki"], pg["page_key"])
                actor = f"anon@{rv.get('ip16') or '?'}"
                evs.append(Event(
                    event_id=f"ow:{pg['page_key']}@{rv.get('seq')}", timestamp=ts, actor_id=actor, actor_type="agent",
                    action_type="edit", target_id=f"wiki:{pg['page_key']}", target_type="resource",
                    content=clip(added, 1500), source=f"german_wiki:other_wikis:{pg['wiki']}",
                    metadata={"raw_id": f"{pg['page_key']}@{rv.get('seq')}", "raw_table": "other-wikis",
                              "actor_name": actor, "origin": rv.get("ip16"), "resources": res, "urls": urls,
                              "tool": tool, "mentions": [], "wiki": pg["wiki"], "target_label": pg["name"]}))

    df = events_to_frame(evs)
    truncated = bool(max_events and len(df) > max_events)
    if truncated:
        df = df.iloc[:max_events].reset_index(drop=True)
    ctx: dict[str, Any] = {
        "dataset": f"German Wiki / collusion.wiki export ({d.name})",
        "observability": "external artifacts only: saves, admin deletions, probes, shortener links; no internal agent state",
        "window_start": str(df.timestamp.min()) if len(df) else None,
        "window_end": str(df.timestamp.max()) if len(df) else None,
        "truncated_to_max_events": truncated,
        "event_counts": df["source"].str.split(":").str[1].value_counts().to_dict() if len(df) else {},
        "n_pages": len(pages), "n_labels": len(labels), "human_handles": sorted(humans),
        "identity_note": "labels are self-chosen per save (spoofable, 899 blank); addresses cut to /16",
        "shortener_links": n_short,
    }
    mp = _find(d, "manifest") or (d / "full-wiki-logs" / "manifest.json")
    if mp and mp.exists():
        man = json.load(open(mp)) if mp.suffix == ".json" else _load_json_gz(mp)
        ctx["manifest"] = {"generated_at": man.get("generated_at"), "cut": man.get("cut"),
                           "checks_ok": sum(1 for c in man.get("checks", []) if c.get("ok")),
                           "checks_total": len(man.get("checks", []))}
    cov = d / "site-coverage.csv"
    gaps = d / "coverage-gaps.csv"
    if cov.exists():
        ctx["coverage_sites"] = sum(1 for _ in csv.DictReader(open(cov)))
    if gaps.exists():
        g = list(csv.DictReader(open(gaps)))
        ctx["coverage_gap_sites"] = len(g)
        ctx["coverage_gap_examples"] = [f"{x['site']} ({x['compilation_status']})" for x in g[:5]]
    return WikiSlice(events=df, context=ctx, pages=pages)
