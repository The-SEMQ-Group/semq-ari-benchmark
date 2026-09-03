#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# ARI-D fp32 reference session -- spec rollout item 4.
#
# Generates the yardstick the rehosted class is measured against: the
# completions the published weights produce, in fp32, under our control, over
# the frozen 100-prompt set. Then the exploratory half-precision passes that
# spec 13.4 asks for.
#
# Separate from run_gpu.sh and run_drift.sh because the node is different in
# kind. A 70B in fp32 is ~282 GB resident, so this is the only run here that
# needs a multi-GPU node (g6e.48xlarge, 8xL40S, 384 GB) -- and the only one
# where the node costs tens of dollars an hour rather than one or two.
#
#   MODELS=llama33      only the Together subject's reference
#   DTYPES=fp32         skip the exploratory passes
#   SMOKE_ONLY=1        the 3-prompt shakedown, then stop
#
# COST. Six passes over two 70B-class models is roughly 10-14 h of generation
# plus ~290 GB of download, so plan on $350-450 at g6e.48xlarge on-demand.
# The smoke pass exists so that a template, token or placement mistake costs
# five minutes instead of the first full pass.
#
# The spec says the half-precision passes are "near zero" marginal cost
# because the model is already loaded. That is not what the generator does,
# and deliberately: each dtype is a fresh load, never an in-place cast,
# because casting a loaded model crushes fp32-born buffers. So every dtype is
# a full pass, and the passes below are priced as such.
set -uo pipefail

VP=${VP:-$HOME/venv/bin}
REPO=${REPO:-$HOME/semq-ari-benchmark}
EXP="$REPO/experiments/arid-fp32-refs"
MODELS="${MODELS:-llama33 qwen25}"
DTYPES="${DTYPES:-fp32 bf16 fp16}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

banner() { printf '\n\033[1m===== %s =====\033[0m\n' "$*"; }
fail()   { echo "ERROR: $*" >&2; exit 1; }

banner "environment"
$VP/python - <<'PY'
import torch
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
print("cuda        ", torch.cuda.is_available())
print("gpus        ", n)
tot = 0
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    tot += p.total_memory
    print(f"  cuda:{i}      {p.name}  {p.total_memory/2**30:.0f} GiB")
print(f"total vram   {tot/2**30:.0f} GiB")
PY

$VP/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || fail "no CUDA device. A CPU fp32 pass over a 70B is not the same
       measurement and would not finish inside the shutdown timer."

# Placement is checked inside generate_refs.py, which refuses to publish a
# reference generated through a CPU or disk offload path. This is the cheaper
# version of the same check: catch an undersized node before the download.
VRAM=$($VP/python -c "
import torch
print(int(sum(torch.cuda.get_device_properties(i).total_memory
              for i in range(torch.cuda.device_count())) / 2**30))")
if [[ "$DTYPES" == *fp32* && "$VRAM" -lt 290 ]]; then
  fail "$VRAM GiB of VRAM. The largest fp32 pass here is Qwen2.5-72B at ~271
       GiB resident (Llama-3.3-70B is ~263), so this needs a node above 290
       GiB: p4d.24xlarge (8xA100 40GB, ~317 GiB usable) or g6e.48xlarge
       (8xL40S, ~358). Set DTYPES='bf16 fp16' to run only the half-precision
       passes on a smaller node."
fi

# Disk, before a 141 GB download discovers it the hard way.
AVAIL=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
NEED=$(( $(wc -w <<< "$MODELS") * 150 ))
[[ "$AVAIL" -lt "$NEED" ]] && fail "$AVAIL GB free, need ~$NEED GB for
       $(wc -w <<< "$MODELS") checkpoint(s). Relaunch with a larger VOLUME_GB."

[[ -n "${HF_TOKEN:-}" ]] || fail "HF_TOKEN is unset. Llama-3.3-70B is a gated
       repo; the download 401s partway through with the node billing. Export
       it in the invocation -- deliberately not baked into user-data."

banner "gated repo access"
# Metadata only, no weights: confirms the token is valid AND that the account
# has accepted the licence, which a valid token alone does not imply.
$VP/python - <<PY || exit 1
import sys
sys.path.insert(0, "$EXP")
from huggingface_hub import HfApi
from generate_refs import MODELS as M
api = HfApi()
for name, cfg in M.items():
    try:
        api.model_info(cfg["hf_id"], revision=cfg["revision"])
        print(f"  ok        {cfg['hf_id']} @ {cfg['revision'][:12]}")
    except Exception as e:
        print(f"  DENIED    {cfg['hf_id']}: {type(e).__name__}: {e}")
        sys.exit(1)
PY

banner "smoke: 3 prompts, fp32, first model"
FIRST=$(awk '{print $1}' <<< "$MODELS")
( cd "$EXP" && $VP/python -u generate_refs.py --model "$FIRST" --dtype fp32 --limit 3 ) \
  || fail "smoke pass failed. Nothing below would have worked either."
$VP/python - <<PY || fail "smoke output is empty or malformed"
import json, pathlib
f = pathlib.Path("$EXP/results/refs_${FIRST}_fp32_prefix3.jsonl")
rs = [json.loads(l) for l in f.read_text().splitlines()]
assert len(rs) == 3, f"{len(rs)} records, expected 3"
assert all(r["text"].strip() for r in rs), "a completion came back empty"
print(f"  {len(rs)} completions, finish_reasons={[r['finish_reason'] for r in rs]}")
print(f"  first: {rs[0]['text'][:100]!r}")
PY

if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo; echo "SMOKE_ONLY set. Stopping before the full passes."; exit 0
fi

banner "full passes"
echo "models: $MODELS"
echo "dtypes: $DTYPES"
echo "Each pass resumes if interrupted; rerunning this script is safe."
failed=()
for m in $MODELS; do
  for d in $DTYPES; do
    banner "$m / $d"
    start=$(date +%s)
    if ( cd "$EXP" && $VP/python -u generate_refs.py --model "$m" --dtype "$d" ); then
      echo "  done in $(( ($(date +%s) - start) / 60 )) min"
    else
      echo "  FAILED after $(( ($(date +%s) - start) / 60 )) min" >&2
      failed+=("$m/$d")
    fi
  done
done

banner "results"
ls -la "$EXP/results/" 2>/dev/null

if (( ${#failed[@]} )); then
  echo >&2
  echo "ERROR: these passes failed: ${failed[*]}" >&2
  echo "       Rerun this script -- completed prompts are not regenerated." >&2
  echo "       Any reference file listed above for a failed pass is PARTIAL." >&2
  exit 1
fi

cat <<'EOF'

All passes complete. From your laptop:

    ./infra/fetch_results.sh <instance-ip>

Then terminate the instance. Do not rely on the shutdown timer -- this node
is the expensive one.
EOF
