#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# Copy the working tree up to the instance. The counterpart of
# fetch_results.sh.
#
# bootstrap.sh tries to clone, but the repository is private and the instance
# holds no credentials -- deliberately, since user-data is readable from the
# metadata service for the life of the box. So the clone fails by design and
# the tree comes up over rsync instead, from a laptop that is already
# authenticated.
#
# This copies the WORKING TREE, not a commit: it is what lets an experiment
# run before its branch is pushed. Whatever is checked out and uncommitted is
# what will run, so check `git status` before relying on the result.
#
# Usage: ./push_tree.sh <instance-ip> [key-path]
set -euo pipefail

IP="${1:?usage: push_tree.sh <instance-ip> [key-path]}"
REGION="${AWS_REGION:-us-east-1}"
KEY="${2:-$HOME/.ssh/semq-ari-gpu-${REGION}.pem}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"

[[ -f "$KEY" ]] || { echo "ERROR: no key at $KEY" >&2; exit 1; }

echo "==> pushing $SRC -> $IP:semq-ari-benchmark"
echo "    HEAD $(git -C "$SRC" rev-parse --short HEAD) on $(git -C "$SRC" rev-parse --abbrev-ref HEAD)"
if [[ -n "$(git -C "$SRC" status --porcelain)" ]]; then
  echo "    tree is dirty; the uncommitted state is what will run"
fi

# --delete keeps the box a mirror rather than an accumulation of past runs.
# results/ is excluded from it: a resumed pass reads what is already on the
# box, and syncing over it would discard the prompts already generated.
rsync -az --delete --info=progress2 \
  -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.venv/' \
  --exclude='*.egg-info/' \
  --exclude='results/cache/' \
  --exclude='results/transcripts/' \
  --exclude='docs/paper/latex/*.pdf' \
  --filter='protect experiments/*/results/***' \
  "$SRC/" "ubuntu@$IP:semq-ari-benchmark/"

echo
echo "==> done. On the box:"
echo "    HF_TOKEN=hf_... SMOKE_ONLY=1 bash ~/semq-ari-benchmark/infra/run_refs.sh"
