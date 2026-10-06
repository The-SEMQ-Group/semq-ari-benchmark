#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# T4: the GPU-determinism matrix (no TF32 on Turing).
export LABEL=${LABEL:-t4}
source "$(dirname "$0")/common.sh"
$VP/python experiments/gpu-determinism/run_matrix.py --label t4 --tf32-available false --n 512 \
  --out "$OUT/gpu_det" --skip-7b 2>&1 | tee "$OUT/gpu_det.log"
publish
