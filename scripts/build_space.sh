#!/usr/bin/env bash
# Assemble a ready-to-upload Hugging Face Space (nothing is uploaded by this script).
#
#   bash scripts/build_space.sh                  # builds ~/swarmguard-space
#   hf upload <your-username>/swarmguard ~/swarmguard-space . --repo-type space
#
# The bundle is built outside the project on purpose: ~/Desktop is iCloud-synced, and iCloud can
# offload files ("dataless"), which makes copies stall until they are downloaded again.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-$HOME/swarmguard-space}"

# Ask iCloud to bring back any offloaded source files before copying (macOS only; no-op elsewhere).
if command -v brctl >/dev/null; then
  for path in swarmguard .streamlit pyproject.toml deploy docs/blog_data.json datasets/german_wiki; do
    [ -e "$path" ] && brctl download "$path" >/dev/null 2>&1 || true
  done
  for i in $(seq 1 60); do
    if ! ls -lOR swarmguard .streamlit deploy datasets/german_wiki docs/blog_data.json pyproject.toml 2>/dev/null | grep -q dataless; then break; fi
    sleep 2
  done
fi

rm -rf "$OUT" && mkdir -p "$OUT/datasets/german_wiki"

# app code + config
cp -R swarmguard "$OUT/"
cp -R .streamlit "$OUT/"
cp pyproject.toml "$OUT/"
cp deploy/space/README.md deploy/space/Dockerfile deploy/space/requirements.txt "$OUT/"
mkdir -p "$OUT/docs" && cp docs/blog_data.json "$OUT/docs/"          # the app's Start-here chart reads this
find "$OUT" -name "__pycache__" -type d -prune -exec rm -rf {} +
rm -rf "$OUT/swarmguard.egg-info"

# public German wiki export: only the files the app reads (gzipped, ~4 MB)
W=datasets/german_wiki
for f in pages.jsonl.gz revisions.jsonl.gz events.jsonl.gz labels.jsonl.gz shortener-logs.json.gz \
         other-wikis.json.gz manifest.json.gz site-coverage.csv coverage-gaps.csv; do
  if [ -f "$W/$f" ]; then cp "$W/$f" "$OUT/datasets/german_wiki/"; else echo "missing: $W/$f" >&2; exit 1; fi
done

# never ship secrets or gated AI Village data
if grep -rIlE "hf_[A-Za-z0-9]{30,}" "$OUT" >/dev/null; then echo "token-like string found in $OUT; aborting" >&2; exit 1; fi
test ! -e "$OUT/.env"

echo "built $OUT ($(du -sh "$OUT" | cut -f1)):"
find "$OUT" -maxdepth 2 -not -path "*/swarmguard/*" | sed "s|^$OUT/||" | sort | head -40
