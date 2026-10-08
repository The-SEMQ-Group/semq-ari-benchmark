#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# Case studies on one A10G: (1) retrieval quality vs code change across 3 BEIR corpora
# x 4 encoders at the 2-bin probe; (2) the `lib` condition, explicit fp32 vs the
# transformers 5 default load for popular half-precision checkpoints.
export LABEL=${LABEL:-casestudies}
source "$(dirname "$0")/common.sh"
VG=$HOME/venvgemma/bin
if [ ! -x "$VG/python" ]; then
  "$HOME/.local/bin/uv" venv -q -p 3.12 "$HOME/venvgemma"
  VIRTUAL_ENV=$HOME/venvgemma "$HOME/.local/bin/uv" pip install -q "torch==2.14.1" "torchvision==0.29.1" \
    "transformers==5.19.0" "sentence-transformers==6.1.0" pillow sentencepiece protobuf datasets pip \
    || { echo "ERROR: gemma stack install failed"; exit 1; }
fi
VIRTUAL_ENV=$HOME/venvgemma "$HOME/.local/bin/uv" pip install -q -e ".[data]" "semq==1.0.0"
# Copy finished results to S3 every 5 minutes; an earlier run lost everything to its timer
# because it published only at the end.
( while sleep 300; do
    cp experiments/regime-discrimination/results/regime_matrix_*_b2.json "$OUT/" 2>/dev/null
    aws s3 sync --quiet --region us-east-2 "$OUT" "$BUCKET/$LABEL/"
  done ) &
SYNC_PID=$!
for ds in scifact nfcorpus arguana; do
  for enc in sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-large-en-v1.5 intfloat/multilingual-e5-large google/embeddinggemma-2; do
    py=$VP/python; [ "$enc" = "google/embeddinggemma-2" ] && py=$VG/python
    step "retrieval $ds $enc" bash -c "cd experiments/regime-discrimination && ARI_R_DATASET=$ds ARI_R_ENCODER=$enc ARI_R_BINS=2 \
      ARI_R_CONDITIONS=reference,gpu_tf32_off,gpu_tf32_on,gpu_bf16,fp16 $py -u run_matrix.py 2>&1 | tee -a '$OUT/retrieval.log'; exit \${PIPESTATUS[0]}"
  done
done
step "copy retrieval" bash -c "cp experiments/regime-discrimination/results/regime_matrix_*_b2.json '$OUT/'"
step "lib default dtype" bash -c "$VG/python experiments/lib-default-dtype/run.py --device cuda --out '$OUT/lib_default_dtype.json' 2>&1 | tee '$OUT/lib.log'; exit \${PIPESTATUS[0]}"
kill "$SYNC_PID" 2>/dev/null
publish
