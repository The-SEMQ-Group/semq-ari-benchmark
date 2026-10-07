#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# Self-hosted CPU panel, day 0, for google/embeddinggemma-2 on its own stack
# (transformers 5.19). Same steps as cpu_panel.sh. Run on the panel host, so the
# `time` comparison >=24h later happens on the same machine.
export LABEL=${LABEL:-cpu_gemma}
source "$(dirname "$0")/common.sh"
VG=$HOME/venvgemma/bin
if [ ! -x "$VG/python" ]; then
  "$HOME/.local/bin/uv" venv -q -p 3.12 "$HOME/venvgemma"
  VIRTUAL_ENV=$HOME/venvgemma "$HOME/.local/bin/uv" pip install -q "torch==2.14.1" "torchvision==0.29.1" \
    "transformers==5.19.0" "sentence-transformers==6.1.0" pillow sentencepiece protobuf pip \
    || { echo "ERROR: gemma stack install failed"; exit 1; }
fi
VIRTUAL_ENV=$HOME/venvgemma "$HOME/.local/bin/uv" pip install -q -e ".[data]" "semq==1.0.0"
$VG/python -c "import torch,transformers,sentence_transformers;print('gemma stack', torch.__version__, transformers.__version__, sentence_transformers.__version__)" | tee -a "$OUT/versions.txt"
m=google/embeddinggemma-2; slug=${m//\//_}
step "report $m" bash -c "$VG/python ari/tools/run_report.py --agent bge --model '$m' --inputs '$INPUTS' && cp 'leaderboard/submissions/$slug.json' '$OUT/report_$slug.json'"
step "prec $m" $VG/python ari/tools/selfhosted_prec.py --model "$m" --inputs "$INPUTS" --out "$OUT/prec_$slug.json"
step "time-baseline $m" $VG/python ari/tools/selfhosted_conc_time.py --mode baseline --model "$m" --inputs "$INPUTS"
step "mach-ref $m" $VG/python ari/tools/gpu_capture.py capture --model "$m" --device cpu --dtype fp32 --tf32 off \
  --inputs "$INPUTS" --n 512 --outdir "$OUT/mach_ref/${slug}__cpu_fp32" --tag "${slug}__cpu_fp32" --conditions same
step "copy baseline" cp -r "$HOME/ari_selfhosted_conc_time/$slug" "$OUT/time_baseline_$slug"
publish
