# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# Shared setup for the ARI v0.2 rerun drivers. Source it; do not run it.
# Checks out the rerun branch, installs the harness at the pinned SDK, records
# versions, and defines `publish` to copy outputs to S3.
set -uo pipefail
REPO=${REPO:-$HOME/semq-ari-benchmark}
# semq needs Python >= 3.11 and the Deep Learning AMI ships 3.10, so build a 3.12
# environment with uv once per box (uv fetches the interpreter).
VP=${VP:-$HOME/venv312/bin}
if [ ! -x "$VP/python" ]; then
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  export PATH="$HOME/.local/bin:$PATH"
  uv venv -q -p 3.12 "$(dirname "$VP")" || { echo "ERROR: could not create a Python 3.12 env"; exit 1; }
  export VIRTUAL_ENV="$(dirname "$VP")"
  uv pip install -q "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
fi
# Pinned on every run, so a box set up earlier converges. transformers 5 needs a newer
# torch than 2.5.1 and breaks nomic's remote code; 4.57.6 loads all eight encoders, the
# three decoders and SFR-Embedding-2_R. torch >= 2.6 is needed to load .bin-only checkpoints (bge-m3).
VIRTUAL_ENV="$(dirname "$VP")" "$HOME/.local/bin/uv" pip install -q "torch==2.6.0" \
  --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || true
VIRTUAL_ENV="$(dirname "$VP")" "$HOME/.local/bin/uv" pip install -q "transformers==4.57.6" \
  "sentence-transformers==4.1.0" datasets scikit-learn accelerate einops hf_transfer sentencepiece protobuf pip \
  || { echo "ERROR: dependency install failed"; exit 1; }
BRANCH=${BRANCH:-ilona/v02-reruns}
BUCKET=${BUCKET:-s3://semq-agent-memory-benchmark/ari-v02-reruns}
export LABEL=${LABEL:?set LABEL, e.g. a10g}
OUT=${OUT:-$HOME/v02_out/$LABEL}
mkdir -p "$OUT"
cd "$REPO" && git fetch -q origin "$BRANCH" && git checkout -q -B "$BRANCH" FETCH_HEAD || exit 1
export GIT_COMMIT=$(git rev-parse HEAD)
$VP/pip install -q -e ".[data]" "semq==1.0.0" || { echo "ERROR: harness install failed"; exit 1; }
# Fetched on the box through its instance role, so the token never passes through a command.
export HF_TOKEN=$(aws ssm get-parameter --region us-east-2 --name /semq-benchmarks/hf_token \
  --with-decryption --query Parameter.Value --output text 2>/dev/null)
{ echo "label=$LABEL commit=$GIT_COMMIT host=$(hostname) date=$(date -u +%FT%TZ)"
  $VP/python - <<'PY'
import platform, semq, torch, transformers, sentence_transformers
print("python", platform.python_version(), "semq", semq.__version__, semq.build_info().build_id)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("transformers", transformers.__version__, "sentence-transformers", sentence_transformers.__version__)
PY
} | tee "$OUT/versions.txt"
# Every step is run through `step`, so a failure is recorded rather than lost in a log,
# and `publish` uploads a STATUS file saying which steps failed. Results without
# "STATUS: ok" are not results.
FAILED=()
step() { local name="$1"; shift; echo ">>> step: $name"; "$@" || { echo "FAILED step: $name"; FAILED+=("$name"); }; }
publish() {
  if [ ${#FAILED[@]} -eq 0 ]; then echo "STATUS: ok" > "$OUT/STATUS"; else printf "STATUS: failed\n%s\n" "${FAILED[@]}" > "$OUT/STATUS"; fi
  cat "$OUT/STATUS"
  aws s3 cp --recursive --quiet --region us-east-2 "$OUT" "$BUCKET/$LABEL/" && echo "published -> $BUCKET/$LABEL/"
}
MODELS8=(BAAI/bge-large-en-v1.5 BAAI/bge-m3 intfloat/multilingual-e5-large
         sentence-transformers/all-mpnet-base-v2 sentence-transformers/all-MiniLM-L6-v2
         nomic-ai/nomic-embed-text-v1.5 mixedbread-ai/mxbai-embed-large-v1
         Snowflake/snowflake-arctic-embed-l)
INPUTS=$REPO/data/ari-bench-v0.1.jsonl
