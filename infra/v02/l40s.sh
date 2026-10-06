#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# L40S: decoding matrix for the three production models, then the SFR-Embedding-2_R row.
export LABEL=${LABEL:-l40s}
source "$(dirname "$0")/common.sh"
for spec in "NousResearch/Meta-Llama-3.1-8B-Instruct llama31_8b" "Qwen/Qwen2.5-7B-Instruct qwen25_7b" \
            "mistralai/Mistral-7B-Instruct-v0.3 mistral7b_v03"; do
  model=${spec% *}; tag=${spec#* }
  ( cd experiments/decoding-reproducibility && ARI_D_MODEL=$model ARI_D_DEVICE=cuda \
    $VP/python -u run_matrix.py ) 2>&1 | tee "$OUT/decoding_$tag.log"
  cp experiments/decoding-reproducibility/results/decoding_matrix.json "$OUT/decoding_matrix_${tag}_gpu.json"
done
$VP/python -u experiments/deployed-agent-panel/sfr_capture.py 2>&1 | tee "$OUT/sfr.log"
cp -r experiments/deployed-agent-panel/out "$OUT/sfr_out" 2>/dev/null
publish
