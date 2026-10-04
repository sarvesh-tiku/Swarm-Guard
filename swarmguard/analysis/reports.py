"""JSON + Markdown reports with evidence, candidate pathways, interventions, and caveats."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..interventions.counterfactual import DISCLAIMER
from .pipeline import AnalysisResult

CAVEATS = [
    "Candidate propagation edges are associations under uncertainty (timing, content similarity, shared "
    "resources, addressing). They are not proof of causal influence.",
    "Unusual coordination is not unsafe behavior. Most detected episodes are ordinary collaboration.",
    "Intervention numbers are structural estimates on the inferred graph ('removes X% of inferred "
    "propagation paths'), not predicted risk reduction.",
    "Observability is partial: unobserved channels (screens, external sites, private memory) can create or "
    "hide influence. Agents also misreport; treat narration as a claim.",
    "SwarmGuard assists human investigation. It does not decide which agents are dangerous and does not "
    "control any live system.",
]


def _excerpt(ev: pd.DataFrame, eid: str, n: int = 280) -> dict[str, Any]:
    if eid not in ev.index:
        return {"event_id": eid}
    r = ev.loc[eid]
    md = r.metadata if isinstance(r.metadata, dict) else {}
    return {"event_id": eid, "raw_id": md.get("raw_id"), "raw_table": md.get("raw_table"),
            "timestamp": str(r.timestamp), "actor": r.actor_id, "action": r.action_type,
            "content": (str(r.content)[:n] if r.content else None)}


def build_report(result: AnalysisResult, top_episodes: int = 5, top_interventions: int = 5) -> dict[str, Any]:
    ev = result.events.set_index("event_id", drop=False)
    agents = result.events.loc[result.events.actor_type == "agent", "actor_id"]
    rep: dict[str, Any] = {
        "context": result.context,
        "summary": {
            "n_events": len(result.events), "n_agents": int(agents.nunique()),
            "time_range": [str(result.events.timestamp.min()), str(result.events.timestamp.max())],
            "n_candidate_pairs": len(result.pairs), "n_agent_edges": len(result.agent_edges),
            "edge_confidence": pd.Series([e.confidence for e in result.agent_edges]).value_counts().to_dict(),
            "n_episodes": len(result.episodes), "encoder": result.encoder_name,
            "timings_s": {k: round(v, 1) for k, v in result.timings.items()},
        },
        "episodes": [],
        "caveats": CAVEATS,
        "counterfactual_disclaimer": DISCLAIMER,
    }
    for ep in result.episodes[:top_episodes]:
        d = ep.summary()
        d["detectors"] = [{
            "name": x.name, "triggered": x.triggered, "confidence": x.confidence, "why": x.why,
            "does_not_prove": x.does_not_prove, "affected_agents": x.affected_agents,
            "affected_resources": x.affected_resources,
            "evidence": [_excerpt(ev, e) for e in x.evidence_event_ids[:6]],
        } for x in ep.detections]
        prop = next((x for x in ep.detections if x.name == "information_propagation"), None)
        if prop:
            d["candidate_pathway"] = [{
                "from": h["from"], "to": h["to"], "confidence": h["confidence"],
                "evidence": [_excerpt(ev, e) for e in h["evidence"]],
            } for h in prop.details.get("hops", [])]
        d["top_edges"] = [{
            "src": e.src, "dst": e.dst, "score": e.score, "confidence": e.confidence, "support": e.support,
            "breakdown": e.breakdown, "explanation": e.explanation, "alternatives": e.alternatives,
            "evidence": [[_excerpt(ev, a, 160), _excerpt(ev, b, 160)] for a, b in e.evidence_pairs[:2]],
        } for e in ep.agent_edges[:6]]
        d["interventions"] = [ri.to_dict() for ri in result.interventions.get(ep.episode_id, [])[:top_interventions]]
        rep["episodes"].append(d)
    return rep


def render_markdown(rep: dict[str, Any]) -> str:
    s = rep["summary"]
    c = rep["context"]
    L = ["# SwarmGuard report", ""]
    if c:
        L += [f"**Dataset:** {c.get('dataset', 'n/a')}  ", f"**Window:** {c.get('window_start')} → {c.get('window_end')}  "]
        if c.get("active_village_goals"):
            L.append(f"**Active village goals:** {'; '.join(c['active_village_goals'])}  ")
        if c.get("observability"):
            L.append(f"**Observability:** {c['observability']}  ")
        if c.get("identity_note"):
            L.append(f"**Identity caveat:** {c['identity_note']}  ")
        if c.get("coverage_gap_sites"):
            L.append(f"**Coverage:** {c.get('coverage_sites')} sites covered, {c['coverage_gap_sites']} with known gaps "
                     f"(e.g. {', '.join(c.get('coverage_gap_examples', [])[:3])})  ")
        if c.get("nearby_scaffolding_changes"):
            L.append("**Scaffolding changes near this window (alternative explanation for behavior shifts):**  ")
            L += [f"- {d}: {t[:160]}" for d, t in c["nearby_scaffolding_changes"]]
    L += ["", f"{s['n_events']:,} events · {s['n_agents']} agents · {s['n_candidate_pairs']:,} candidate pairs · "
          f"{s['n_agent_edges']} agent edges {s['edge_confidence']} · {s['n_episodes']} episodes · encoder `{s['encoder']}`", ""]
    for ep in rep["episodes"]:
        L += [f"## {ep['episode_id']}: {ep['label']}", "",
              f"{ep['start']} → {ep['end']} · {ep['stats']['n_agents']} agents · {ep['n_events']} events · "
              f"{ep['n_candidate_edges']} candidate edges", "",
              f"**Agents:** {', '.join(ep['agents'])}  ",
              f"**Top resources:** {', '.join(r for r, _ in ep['stats']['top_resources'][:4]) or '—'}", "",
              "### Detectors", ""]
        for d in ep["detectors"]:
            mark = "TRIGGERED" if d["triggered"] else "not triggered"
            L += [f"- **{d['name']}** ({mark}, confidence {d['confidence']}): {d['why']}",
                  f"  - *Does not prove:* {d['does_not_prove']}"]
        if ep.get("candidate_pathway"):
            L += ["", "### Candidate pathway (time-respecting)", ""]
            for h in ep["candidate_pathway"]:
                L.append(f"- **{h['from']} → {h['to']}** ({h['confidence']})")
                for x in h["evidence"]:
                    L.append(f"  - `{x.get('timestamp', '')[11:19]}` {x.get('actor')} [{x.get('action')}] "
                             f"{(x.get('content') or '').replace(chr(10), ' ')[:220]} (`{x.get('raw_table')}:{x.get('raw_id')}`)")
        L += ["", "### Strongest candidate edges", ""]
        for e in ep["top_edges"]:
            L.append(f"- **{e['src']} → {e['dst']}** score {e['score']:.2f} ({e['confidence']}, {e['support']} pairs). {e['explanation']}")
            if e["alternatives"]:
                L.append(f"  - *Alternative explanations:* {'; '.join(e['alternatives'])}")
        hr = ep["stats"].get("historical_response")
        if hr:
            L += ["", "### What operators actually did (historical audit)", ""]
            L.append(f"- {hr['pages_deleted']}/{hr['episode_pages']} episode pages were later deleted by "
                     f"{', '.join(hr['deleted_by']) or '—'}; first deletion "
                     f"{hr['hours_from_episode_end_to_first_deletion']} h after the episode ended.")
            L.append(f"- {hr['writes_after_episode_before_deletion']} further writes landed on those pages between the "
                     f"episode and their deletion; {hr['recreations_after_deletion']} recreation(s) after deletion.")
            if hr.get("agreement"):
                extra = (f" ({hr['top_target_hours_to_deletion']} h after the episode)"
                         if hr.get("top_target_hours_to_deletion") is not None else "")
                L.append(f"- Agreement: {hr['agreement']}{extra}.")
            L.append("- *Observational comparison; it does not show an earlier control would have worked.*")
        L += ["", "### Recommended interventions (structural estimates)", ""]
        for k, iv in enumerate(ep["interventions"], 1):
            L.append(f"{k}. {iv['headline']} · containment score {iv['containment_score']:.2f}")
            for ex in iv["removed_path_examples"][:2]:
                L.append(f"   - removes path {' , '.join(ex['path'])}")
        L.append("")
    L += ["## Caveats", ""] + [f"- {x}" for x in rep["caveats"]] + ["", f"*{rep['counterfactual_disclaimer']}*", ""]
    return "\n".join(L)


def write_report(result: AnalysisResult, out_dir: str | Path, stem: str = "report", **kw) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep = build_report(result, **kw)
    jp, mp = out / f"{stem}.json", out / f"{stem}.md"
    jp.write_text(json.dumps(rep, indent=2, default=str))
    mp.write_text(render_markdown(rep))
    return jp, mp
