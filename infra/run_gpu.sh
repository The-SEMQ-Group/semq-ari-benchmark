#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# Run the GPU-only conditions of both experiments, on the instance.
#
# ARI-R  regime discrimination: fp16, TF32 on/off, GPU bf16 against the CPU
#        fp32 reference. This supplies the retrieval half of the TF32 audit-gap
#        argument, which the gpu-determinism experiment asserted but never
#        measured.
#
# ARI-D  decoding reproducibility on a model people actually serve. Qwen2.5-3B
#        has a 151,936-token vocabulary, so this also exercises the chunked
#        SEMQ path that TinyLlama did not.
#
# Model choice is constrained by the fp32 reference, not by inference. A 3B
# model in fp32 is ~12 GB and fits an A10G's 24 GB. A 7B fp32 reference is
# ~28 GB and does NOT -- use g6e.xlarge (L40S, 48 GB) and set ARI_D_MODEL.
set -uo pipefail

VP=${VP:-$HOME/venv/bin}
REPO=${REPO:-$HOME/semq-ari-benchmark}
export ARI_D_MODEL="${ARI_D_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export ARI_D_DEVICE="${ARI_D_DEVICE:-cuda}"

banner() { printf '\n\033[1m===== %s =====\033[0m\n' "$*"; }

# Run an experiment, drop known-noisy lines, and keep the Python exit status.
#
# A pipe reports grep's status, not Python's, and `|| true` discarded what was
# left. That combination turned a download failure, an OOM, or a crash into a
# run that printed "Done" and then advertised whatever JSON happened to be on
# disk -- which, on a box that has run this before, is the previous model's
# results. PIPESTATUS[0] recovers the status of the command itself.
#
# Process substitution would also preserve the status, but it is asynchronous:
# output can arrive after the function returns, so a failing run loses the very
# lines that say why it failed. A pipe keeps the ordering.
run_experiment() {
  local dir="$1" filter="$2" status
  ( cd "$REPO/$dir" && $VP/python -u run_matrix.py ) 2>&1 | grep -v "$filter"
  status=${PIPESTATUS[0]}
  return "$status"
}

banner "environment"
$VP/python -c "
import torch
print('cuda', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
free, total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
print(f'vram free {free/2**30:.1f} GiB of {total/2**30:.1f} GiB')
"
if ! $VP/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "ERROR: no CUDA device. Nothing here will do what it claims." >&2
  exit 1
fi

banner "ARI-R -- regime discrimination (GPU conditions)"
# Only the reference and the GPU conditions. The CPU precision conditions are
# already measured on a CPU host, and this instance has AVX2 without AVX-512,
# so torch would run bf16 through a slow emulation path and measure the box.
export ARI_R_CONDITIONS="${ARI_R_CONDITIONS:-reference,gpu_tf32_off,gpu_tf32_on,gpu_bf16,fp16}"
failed=()
run_experiment experiments/regime-discrimination \
  "^Warning: You are sending unauthenticated" || failed+=("ARI-R")

banner "ARI-D -- decoding reproducibility on $ARI_D_MODEL"
echo "note: the int8 condition is CPU-only in torch and is skipped on GPU."
run_experiment experiments/decoding-reproducibility "max_new_tokens" || failed+=("ARI-D")

if (( ${#failed[@]} )); then
  echo >&2
  echo "ERROR: ${failed[*]} failed. Any JSON listed below is from an earlier run" >&2
  echo "       and does not describe this one." >&2
  exit 1
fi

banner "results"
find "$REPO/experiments" -name '*.json' -newermt '-6 hours' -print 2>/dev/null | while read -r f; do
  echo "  $f"
done

cat <<'EOF'

Done. Pull the results down from your laptop:

    ./infra/fetch_results.sh <instance-ip>

Then terminate the instance. Do not rely on the shutdown timer.
EOF
