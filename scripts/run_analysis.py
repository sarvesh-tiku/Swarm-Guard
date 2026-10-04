"""Run the SwarmGuard pipeline and write JSON + Markdown reports.

Examples:
    # synthetic benchmark (no network) + evaluation metrics
    python scripts/run_analysis.py synthetic

    # bounded real AI Village window (QUICK mode: <= 5k events)
    python scripts/run_analysis.py ai-village --start "2026-07-06 15:00" --end "2026-07-07 00:00"

    # MEDIUM mode, several days, streamed from the Hub instead of the cached download
    python scripts/run_analysis.py ai-village --start 2026-07-06 --end 2026-07-10 --mode medium --source stream

    # show daily activity to pick a window
    python scripts/run_analysis.py ai-village --list-days

    # German Wiki files
    python scripts/run_analysis.py german-wiki --dir path/to/german_wiki
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

MODES = {"quick": 5_000, "medium": 100_000, "streaming": None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["synthetic", "ai-village", "german-wiki"])
    ap.add_argument("--start", help="UTC start (ISO), AI Village / German Wiki")
    ap.add_argument("--end", help="UTC end (ISO)")
    ap.add_argument("--mode", choices=list(MODES), default="quick", help="event budget: quick<=5k, medium<=100k")
    ap.add_argument("--max-events", type=int, help="override the mode's event cap")
    ap.add_argument("--source", choices=["download", "stream"], default="download",
                    help="AI Village: cached download + Parquet index (default) or HF streaming")
    ap.add_argument("--dir", help="German Wiki directory with *.jsonl.gz files")
    ap.add_argument("--encoder", choices=["auto", "minilm", "tfidf"], default="auto")
    ap.add_argument("--window-minutes", type=float, default=120.0, help="propagation look-back window")
    ap.add_argument("--out", default="outputs", help="report directory")
    ap.add_argument("--list-days", action="store_true", help="AI Village: print busiest days and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    from swarmguard.analysis.pipeline import PipelineConfig, run_pipeline
    from swarmguard.analysis.reports import write_report
    from swarmguard.data.hf_access import HFAccessError
    from swarmguard.graph.propagation import PropagationConfig

    cap = args.max_events if args.max_events is not None else MODES[args.mode]
    cfg = PipelineConfig(propagation=PropagationConfig(window_minutes=args.window_minutes), encoder=args.encoder)

    if args.dataset == "synthetic":
        from swarmguard.analysis.evaluation import evaluate
        from swarmguard.data.synthetic import make_benchmark

        bench = make_benchmark()
        res = run_pipeline(bench.events, cfg, context={"dataset": "synthetic benchmark"})
        metrics = evaluate(res, bench)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "synthetic_eval.json").write_text(json.dumps(metrics, indent=2, default=str))
        print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, list)}, indent=2, default=str))
        jp, mp = write_report(res, out, "synthetic_report")
    elif args.dataset == "ai-village":
        from swarmguard.data import ai_village as av

        try:
            if args.list_days:
                print(av.daily_activity().sort_values("messages", ascending=False).head(25).to_string())
                return 0
            if not (args.start and args.end):
                ap.error("--start and --end are required for ai-village (try --list-days)")
            sl = av.load_window(args.start, args.end, max_events=cap, mode=args.source)
        except HFAccessError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        print(f"loaded {len(sl.events):,} events ({sl.context['raw_rows']}); truncated={sl.context['truncated_to_max_events']}")
        res = run_pipeline(sl.events, cfg, context=sl.context)
        stem = f"ai_village_{args.start}_{args.end}".replace(" ", "T").replace(":", "")
        jp, mp = write_report(res, args.out, stem)
    else:
        from swarmguard.data import german_wiki as gw

        if not args.dir:
            ap.error("--dir is required for german-wiki")
        sl = gw.load_dir(args.dir, start=args.start, end=args.end, max_events=cap)
        print(f"loaded {len(sl.events):,} events {sl.context.get('event_counts')}")
        res = run_pipeline(sl.events, cfg, context=sl.context)
        jp, mp = write_report(res, args.out, "german_wiki_report")

    print(f"\n{len(res.episodes)} episodes, {len(res.agent_edges)} candidate edges. Top episodes:")
    for ep in res.episodes[:5]:
        trig = [d.name for d in ep.detections if d.triggered]
        top = res.interventions.get(ep.episode_id, [])
        print(f"  {ep.episode_id} {ep.label[:70]} | {ep.stats['n_agents']} agents | {trig}")
        if top:
            print(f"      → {top[0].headline()}")
    print(f"\nreports: {jp}\n         {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
