#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# A10G: SciFact GPU rows, then the GPU-determinism matrix and its 2x2 isolation.
LABEL=${LABEL:-a10g} source "$(dirname "$0")/common.sh"
( cd experiments/regime-discrimination && \
  ARI_R_CONDITIONS=reference,gpu_tf32_off,gpu_tf32_on,gpu_bf16,fp16 $VP/python -u run_matrix.py ) \
  2>&1 | tee "$OUT/regime_gpu.log"
cp experiments/regime-discrimination/results/regime_matrix.json "$OUT/regime_matrix_gpu.json"
$VP/python experiments/gpu-determinism/run_matrix.py --label a10g --tf32-available true --n 512 \
  --out "$OUT/gpu_det" --skip-7b 2>&1 | tee "$OUT/gpu_det.log"
$VP/python experiments/gpu-determinism/isolation.py --n 512 --out "$OUT/iso" 2>&1 | tee "$OUT/iso.log"
publish
