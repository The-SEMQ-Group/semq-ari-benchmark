#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# Second session in the H100 Capacity Block:
#   (1) the `lib` condition again, one process per model;
#   (2) the decoding matrix for the three 7-8B models;
#   (3) the TF32 x determinism 2x2 isolation with `same` and `proc` repeats (the A10G method).
# Results publish to S3 at the end of each step. Powers off when done.
export LABEL=${LABEL:-h100_extra}
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
sync_out() { aws s3 sync --quiet --region us-east-2 "$OUT" "$BUCKET/$LABEL/"; }

mkdir -p "$OUT/lib"
for m in Qwen/Qwen3-Embedding-0.6B ibm-granite/granite-embedding-small-english-r2 mixedbread-ai/mxbai-embed-large-v1 \
         intfloat/multilingual-e5-large-instruct Alibaba-NLP/gte-multilingual-base thenlper/gte-small google/embeddinggemma-2; do
  step "lib $m" bash -c "$VG/python experiments/lib-default-dtype/run.py --device cuda --models '$m' \
    --out '$OUT/lib/${m//\//_}.json' 2>&1 | tee -a '$OUT/lib.log'; exit \${PIPESTATUS[0]}"
done
sync_out

for spec in "NousResearch/Meta-Llama-3.1-8B-Instruct llama31_8b" "Qwen/Qwen2.5-7B-Instruct qwen25_7b" \
            "mistralai/Mistral-7B-Instruct-v0.3 mistral7b_v03"; do
  model=${spec% *}; tag=${spec#* }
  rm -f experiments/decoding-reproducibility/results/decoding_matrix.json
  step "decoding $tag" bash -c "cd experiments/decoding-reproducibility && ARI_D_MODEL='$model' ARI_D_DEVICE=cuda $VP/python -u run_matrix.py 2>&1 | tee '$OUT/decoding_$tag.log'; exit \${PIPESTATUS[0]}"
  step "copy decoding $tag" cp experiments/decoding-reproducibility/results/decoding_matrix.json "$OUT/decoding_matrix_${tag}_gpu.json"
  sync_out
done

step "isolation 2x2" bash -c "$VP/python experiments/gpu-determinism/isolation.py --n 512 --out '$OUT/iso' 2>&1 | tee '$OUT/iso.log'; exit \${PIPESTATUS[0]}"
publish
sudo shutdown -h +5
