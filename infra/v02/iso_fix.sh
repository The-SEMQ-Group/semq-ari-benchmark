#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# TF32 x determinism isolation, rerun after the 2026-10-13 fix to gpu_capture's `proc` worker
# (it now applies the parent's TF32 and determinism flags).
#   LABEL=iso_fix_a10g bash infra/v02/iso_fix.sh
export LABEL=${LABEL:?set LABEL, e.g. iso_fix_a10g}
source "$(dirname "$0")/common.sh"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a "$OUT/versions.txt"
step "isolation 2x2" bash -c "$VP/python experiments/gpu-determinism/isolation.py --n 512 --out '$OUT/iso' 2>&1 | tee '$OUT/iso.log'; exit \${PIPESTATUS[0]}"
publish
sudo shutdown -h +5
