"""Discover AI Village configs and schemas without downloading the dataset.

Streams the first N rows of each config, prints columns + representative rows,
and summarizes nested keys (e.g. events.data.actionType) over a larger sample.

Usage:
    python scripts/inspect_ai_village.py                       # all configs, 3 rows each
    python scripts/inspect_ai_village.py --configs events --profile-rows 5000
    python scripts/inspect_ai_village.py --out docs/raw_schema_dump.txt
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import io
import itertools
import sys

from swarmguard.data.hf_access import REPO, explain_error, token_kwarg

# Configs that are too large to profile deeply by default.
LARGE = {"computer_use_turns", "agent_memories", "claude_code_messages", "events"}


def _type_name(v: object) -> str:
    if v is None:
        return "null"
    if isinstance(v, dict):
        return "dict"
    if isinstance(v, list):
        return f"list[{_type_name(v[0]) if v else '?'}]"
    return type(v).__name__


def _walk(prefix: str, v: object, counter: collections.Counter, types: dict) -> None:
    counter[prefix] += 1
    types.setdefault(prefix, collections.Counter())[_type_name(v)] += 1
    if isinstance(v, dict):
        for k, vv in v.items():
            _walk(f"{prefix}.{k}", vv, counter, types)


def inspect_config(config: str, n_rows: int, profile_rows: int) -> None:
    from datasets import get_dataset_split_names, load_dataset

    splits = get_dataset_split_names(REPO, config, **token_kwarg())
    print(f"\n[{config}]\nsplits: {splits}")
    split = "train" if "train" in splits else splits[0]
    ds = load_dataset(REPO, config, split=split, streaming=True, **token_kwarg())
    it = iter(ds)
    rows = list(itertools.islice(it, max(n_rows, profile_rows)))
    if not rows:
        print("No rows found")
        return
    print("columns:", list(rows[0].keys()))
    # Representative rows: first, middle, last of the sample.
    picks = sorted({0, len(rows) // 2, len(rows) - 1})[:n_rows]
    for idx in picks:
        print(f"\nROW {idx}")
        for key, value in rows[idx].items():
            print(f"  {key}: {repr(value)[:1000]}")
    counter: collections.Counter = collections.Counter()
    types: dict = {}
    for r in rows:
        for k, v in r.items():
            _walk(k, v, counter, types)
    print(f"\nFIELD PROFILE over {len(rows)} streamed rows (path: present-count types)")
    for path, c in sorted(counter.items()):
        print(f"  {path}: {c} {dict(types[path])}")
    # Categorical-looking fields: show top values.
    for path in sorted(counter):
        vals = collections.Counter()
        for r in rows:
            v = r
            for part in path.split("."):
                v = v.get(part) if isinstance(v, dict) else None
            if isinstance(v, (str, bool, int)) and (not isinstance(v, str) or len(v) < 40):
                vals[v] += 1
        if 1 < len(vals) <= 40 and sum(vals.values()) > 0.5 * len(rows):
            print(f"  VALUES {path}: {dict(vals.most_common(25))}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", help="subset of configs (default: all)")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--profile-rows", type=int, default=300, help="rows streamed for field profiling")
    ap.add_argument("--out", help="also write output to this file")
    args = ap.parse_args()

    from datasets import get_dataset_config_names

    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s: str) -> int:
            sys.__stdout__.write(s)
            buf.write(s)
            return len(s)

    with contextlib.redirect_stdout(Tee()):
        try:
            configs = get_dataset_config_names(REPO, **token_kwarg())
        except Exception as e:
            print(explain_error(e))
            return 1
        print("CONFIGS\n" + "=" * 80)
        for c in configs:
            print(c)
        print("\nSCHEMAS\n" + "=" * 80)
        for c in args.configs or configs:
            try:
                prof = args.profile_rows if c not in LARGE or args.configs else min(args.profile_rows, 200)
                inspect_config(c, args.rows, prof)
            except Exception as e:
                print(f"\n[{c}] ERROR:\n{explain_error(e)}")
    if args.out:
        with open(args.out, "w") as f:
            f.write(buf.getvalue())
    return 0


if __name__ == "__main__":
    sys.exit(main())
