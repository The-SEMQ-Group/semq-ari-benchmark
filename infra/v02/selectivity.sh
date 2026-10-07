#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# Selectivity study (experiments/selectivity-confirmatory/PREREGISTRATION.md):
# capture this box's references and episodes, one fresh process each.
#   BOX=cpu|a10g|t4 bash infra/v02/selectivity.sh
export BOX=${BOX:?set BOX to cpu, a10g or t4}
export LABEL=${LABEL:-selectivity_$BOX}
source "$(dirname "$0")/common.sh"
# Second stack for google/embeddinggemma-2, which needs transformers 5.19.
VG=$HOME/venvgemma/bin
if [ ! -x "$VG/python" ]; then
  "$HOME/.local/bin/uv" venv -q -p 3.12 "$HOME/venvgemma"
  VIRTUAL_ENV=$HOME/venvgemma "$HOME/.local/bin/uv" pip install -q "torch==2.14.1" "torchvision==0.29.1" \
    "transformers==5.19.0" "sentence-transformers==6.1.0" pillow sentencepiece protobuf pip \
    || { echo "ERROR: gemma stack install failed"; exit 1; }
fi
VIRTUAL_ENV=$HOME/venvgemma "$HOME/.local/bin/uv" pip install -q -e . "semq==1.0.0" || { echo "ERROR: gemma harness install"; exit 1; }
$VG/python -c "import torch,transformers,sentence_transformers;print('gemma stack', torch.__version__, transformers.__version__, sentence_transformers.__version__, torch.cuda.is_available())" | tee -a "$OUT/versions.txt"
cd experiments/selectivity-confirmatory
mkdir -p "$OUT/vec"
$VP/python - > "$OUT/ids.txt" <<PY
import json; p = json.load(open("episodes.json"))
for r in p["references"]:
    if r["box"] == "$BOX": print(r["id"], r["model"])
for e in p["episodes"]:
    if e["box"] == "$BOX": print(e["id"], e["model"])
PY
echo "$(wc -l < "$OUT/ids.txt") captures on $BOX"
capture() {  # id model
  local py=$VP/python; [ "$2" = "google/embeddinggemma-2" ] && py=$VG/python
  [ -f "$OUT/vec/$1.npy" ] && return 0
  "$py" capture_episode.py --id "$1" --out "$OUT/vec" >> "$OUT/capture.log" 2>&1 || { echo "FAILED $1" >> "$OUT/failed.txt"; return 1; }
}
export -f capture; export VP VG OUT
# Copy finished vectors to S3 every 5 minutes, so a timer kill loses at most one interval.
( while sleep 300; do aws s3 sync --quiet --region us-east-2 "$OUT/vec" "$BUCKET/$LABEL/vec/"; done ) &
SYNC_PID=$!
# References first (episodes compare against them), then episodes.
grep '^ref-' "$OUT/ids.txt" | while read -r id model; do capture "$id" "$model"; done
PAR=${PAR:-1}
grep -v '^ref-' "$OUT/ids.txt" | xargs -P "$PAR" -L 1 bash -c 'capture "$0" "$1"'
[ -s "$OUT/failed.txt" ] && FAILED+=("$(wc -l < "$OUT/failed.txt") captures failed (failed.txt)")
echo "captured $(ls "$OUT"/vec/*.npy 2>/dev/null | wc -l) of $(wc -l < "$OUT/ids.txt")"
kill "$SYNC_PID" 2>/dev/null
cd "$REPO"; publish
