#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# Run the drift rank-profile experiment on the instance.
#
# Separate from run_gpu.sh because this needs the whole card to itself. Every
# arm holds a full fp32 copy of the model, and at 7B that is ~30 GB, so nothing
# else can be resident while it runs.
#
# Sizing is set by the fp32 reference, not by inference:
#
#   Qwen2.5-1.5B-Instruct   ~6 GB   fits g5.xlarge  (A10G, 24 GB)
#   Qwen2.5-7B-Instruct    ~30 GB   needs g6e.xlarge (L40S, 48 GB)
set -uo pipefail

VP=${VP:-$HOME/venv/bin}
REPO=${REPO:-$HOME/semq-ari-benchmark}
export DRIFT_MODEL="${DRIFT_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export DRIFT_DEVICE="${DRIFT_DEVICE:-cuda}"
# run.py lives in experiments/, but imports `ari` from the repo root. Running a
# script by path puts the script's own directory on sys.path, not the root.
export PYTHONPATH="$REPO"

banner() { printf '\n\033[1m===== %s =====\033[0m\n' "$*"; }

banner "environment"
$VP/python -c "
import torch
print('cuda   ', torch.cuda.is_available())
print('device ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
free, total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
print(f'vram    {free/2**30:.1f} GiB free of {total/2**30:.1f} GiB')
import peft, transformers
print('peft   ', peft.__version__)
print('transformers', transformers.__version__)
"
if ! $VP/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "ERROR: no CUDA device. A CPU run of a 7B fp32 reference is not the" >&2
  echo "same measurement and will take hours. Stopping." >&2
  exit 1
fi
if ! $VP/python -c "import peft" 2>/dev/null; then
  echo "ERROR: peft is missing, so every adapter arm would be skipped and the" >&2
  echo "run would look clean while measuring nothing. Install it first." >&2
  exit 1
fi

banner "drift rank profile on $DRIFT_MODEL"
( cd "$REPO" && $VP/python -u experiments/drift-rank-profile/run.py ) \
  2>&1 | grep -v "^Warning: You are sending unauthenticated" || true

banner "result"
ls -la "$REPO/experiments/drift-rank-profile/results/" 2>/dev/null

cat <<'EOF'

Done. Pull the result down from your laptop:

    ./infra/fetch_results.sh <instance-ip>

Then terminate the instance. Do not rely on the shutdown timer.
EOF
