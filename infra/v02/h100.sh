#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# H100 (Capacity Block, p5.4xlarge): the batch-invariance matrix for the eight ARI-R
# encoders on the pinned stack, then EmbeddingGemma 2 on its own stack (transformers 5.19).
# Captures publish to S3 as each cell finishes. Powers off (terminating the instance) when done.
export LABEL=${LABEL:-h100}
source "$(dirname "$0")/common.sh"
VG=$HOME/venvgemma/bin
if [ ! -x "$VG/python" ]; then
  "$HOME/.local/bin/uv" venv -q -p 3.12 "$HOME/venvgemma"
  VIRTUAL_ENV=$HOME/venvgemma "$HOME/.local/bin/uv" pip install -q "torch==2.14.1" "torchvision==0.29.1" \
    "transformers==5.19.0" "sentence-transformers==6.1.0" pillow sentencepiece protobuf pip \
    || { echo "ERROR: gemma stack install failed"; exit 1; }
fi
VIRTUAL_ENV=$HOME/venvgemma "$HOME/.local/bin/uv" pip install -q -e ".[data]" "semq==1.0.0"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a "$OUT/versions.txt"
EIGHT=sentence-transformers/all-MiniLM-L6-v2,BAAI/bge-large-en-v1.5,BAAI/bge-m3,intfloat/multilingual-e5-large,sentence-transformers/all-mpnet-base-v2,nomic-ai/nomic-embed-text-v1.5,mixedbread-ai/mxbai-embed-large-v1,Snowflake/snowflake-arctic-embed-l
step "batch sweep (8 encoders)" bash -c "$VP/python experiments/batch-invariance/run_batch_sweep.py --label h100 --models '$EIGHT' \
  --n 1000 --out '$OUT/batch' --publish-s3 '$BUCKET/$LABEL' 2>&1 | tee '$OUT/batch.log'; exit \${PIPESTATUS[0]}"
step "batch sweep (EmbeddingGemma 2)" bash -c "$VG/python experiments/batch-invariance/run_batch_sweep.py --label h100 --models google/embeddinggemma-2 \
  --n 1000 --out '$OUT/batch_gemma' --publish-s3 '$BUCKET/$LABEL' 2>&1 | tee '$OUT/batch_gemma.log'; exit \${PIPESTATUS[0]}"
publish
sudo shutdown -h +5
