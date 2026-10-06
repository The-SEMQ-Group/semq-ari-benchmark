#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# L40S retry: the two steps of l40s.sh that failed on the first v0.2 run
# (Mistral needed sentencepiece/protobuf; SFR ran out of GPU memory).
export LABEL=${LABEL:-l40s_retry}
source "$(dirname "$0")/common.sh"
rm -f experiments/decoding-reproducibility/results/decoding_matrix.json
rm -rf experiments/decoding-reproducibility/results/cache/a99e33145f8dada1
step "decoding mistral7b_v03" bash -c "cd experiments/decoding-reproducibility && ARI_D_MODEL='mistralai/Mistral-7B-Instruct-v0.3' ARI_D_DEVICE=cuda $VP/python -u run_matrix.py 2>&1 | tee '$OUT/decoding_mistral7b_v03.log'; exit \${PIPESTATUS[0]}"
step "copy decoding mistral7b_v03" cp experiments/decoding-reproducibility/results/decoding_matrix.json "$OUT/decoding_matrix_mistral7b_v03_gpu.json"
rm -rf experiments/deployed-agent-panel/out
step "sfr" bash -c "$VP/python -u experiments/deployed-agent-panel/sfr_capture.py 2>&1 | tee '$OUT/sfr.log'; exit \${PIPESTATUS[0]}"
step "copy sfr" cp -r experiments/deployed-agent-panel/out "$OUT/sfr_out"
publish
