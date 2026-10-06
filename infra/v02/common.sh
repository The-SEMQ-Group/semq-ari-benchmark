# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# Shared setup for the ARI v0.2 rerun drivers. Source it; do not run it.
# Checks out the rerun branch, installs the harness at the pinned SDK, records
# versions, and defines `publish` to copy outputs to S3.
set -uo pipefail
VP=${VP:-$HOME/venv/bin}
REPO=${REPO:-$HOME/semq-ari-benchmark}
BRANCH=${BRANCH:-ilona/v02-reruns}
BUCKET=${BUCKET:-s3://semq-agent-memory-benchmark/ari-v02-reruns}
LABEL=${LABEL:?set LABEL, e.g. a10g}
OUT=${OUT:-$HOME/v02_out/$LABEL}
mkdir -p "$OUT"
cd "$REPO" && git fetch -q origin "$BRANCH" && git checkout -q -B "$BRANCH" "origin/$BRANCH" || exit 1
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
publish() { aws s3 cp --recursive --quiet --region us-east-2 "$OUT" "$BUCKET/$LABEL/" && echo "published -> $BUCKET/$LABEL/"; }
MODELS8=(BAAI/bge-large-en-v1.5 BAAI/bge-m3 intfloat/multilingual-e5-large
         sentence-transformers/all-mpnet-base-v2 sentence-transformers/all-MiniLM-L6-v2
         nomic-ai/nomic-embed-text-v1.5 mixedbread-ai/mxbai-embed-large-v1
         Snowflake/snowflake-arctic-embed-l)
INPUTS=$REPO/data/ari-bench-v0.1.jsonl
