#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# EC2 user-data. Runs once as root on first boot.
#
# The shutdown timer is armed FIRST, before anything that can fail or hang.
# If the rest of this script breaks, the instance still terminates itself.
set -uo pipefail
exec > >(tee -a /var/log/semq-bootstrap.log) 2>&1
echo "=== semq bootstrap $(date -u) ==="

# --- cost guard, armed before any work ------------------------------------
shutdown -h +__MAX_MINUTES__ &
echo "shutdown armed: +__MAX_MINUTES__ minutes"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip git >/dev/null

cd /home/ubuntu

echo "--- cloning ---"
sudo -u ubuntu git clone --depth 1 \
  https://github.com/The-SEMQ-Group/semq-ari-benchmark.git 2>&1 | tail -2 || \
  echo "clone failed (private repo?) -- copy the tree up with rsync instead"

echo "--- python env ---"
sudo -u ubuntu python3 -m venv /home/ubuntu/venv
VP=/home/ubuntu/venv/bin
sudo -u ubuntu $VP/pip install -q --upgrade pip wheel

# The Deep Learning AMI ships a CUDA driver; install a matching torch wheel.
sudo -u ubuntu $VP/pip install -q torch --index-url https://download.pytorch.org/whl/cu121
sudo -u ubuntu $VP/pip install -q \
  transformers sentence-transformers datasets scikit-learn accelerate peft einops

# semq carries the QBIN probe, and ari/probe.py imports it to compute codes, so
# an experiment that needs codes cannot run without it. It resolves from
# CodeArtifact, never from PyPI, which needs an auth step. Without the login
# below pip reports "No matching distribution found for semq". Because pip
# fails a whole install command when one requirement is unresolvable, semq used
# to take transformers down with it, and this script has no `set -e`, so the
# box reached the GPU with no transformers and still logged "bootstrap
# complete". Keep semq on its own line whatever happens to the login.
CA_DOMAIN=semq
CA_REPO=semq-sdk
CA_REGION=us-east-2
CA_OWNER=127348475353
if CA_TOKEN=$(aws codeartifact get-authorization-token --domain "$CA_DOMAIN" \
      --domain-owner "$CA_OWNER" --region "$CA_REGION" \
      --query authorizationToken --output text 2>/dev/null) && [ -n "$CA_TOKEN" ]; then
  CA_URL="https://aws:${CA_TOKEN}@${CA_DOMAIN}-${CA_OWNER}.d.codeartifact.${CA_REGION}.amazonaws.com/pypi/${CA_REPO}/simple/"
  # --extra-index-url, not --index-url: the latter REPLACES PyPI, and this
  # CodeArtifact repository serves only semq. semq depends on cffi, so an
  # --index-url install resolves semq and then fails on "No matching
  # distribution found for cffi".
  sudo -u ubuntu $VP/pip install -q --extra-index-url "$CA_URL" semq \
    && echo "semq installed from CodeArtifact" \
    || echo "WARNING: semq install failed; anything computing codes will not run"
else
  echo "WARNING: no CodeArtifact token (instance role missing codeartifact"
  echo "         permissions?); semq not installed and codes cannot be computed"
fi
# hf_transfer is the Rust download path. It matters only for the ARI-D
# reference session, where the checkpoint is ~141 GB and the default
# single-stream python downloader turns a 10-minute fetch into an hour of
# billed node time.
sudo -u ubuntu $VP/pip install -q "huggingface_hub[hf_transfer]"

# One cache location both the driver and generate_refs.py agree on, on the
# root volume (the only volume this instance has). Deliberately NOT setting a
# token here: user-data stays readable from the instance metadata service for
# the life of the box and shows in the console, so a gated-repo token belongs
# in the operator's invocation of run_refs.sh, not in this file.
cat > /etc/profile.d/semq-hf.sh <<'EOS'
export HF_HOME=/home/ubuntu/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
EOS

echo "--- versions ---"
sudo -u ubuntu $VP/python - <<'PY'
import torch, transformers
print("torch       ", torch.__version__)
print("cuda avail  ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device      ", torch.cuda.get_device_name(0))
    print("capability  ", torch.cuda.get_device_capability(0))
print("transformers", transformers.__version__)
try:
    import semq
    print("semq        ", getattr(semq, "__version__", "?"))
except Exception as e:
    # Loud, because ari/probe.py needs it and a silent absence is what let a
    # box look ready while it could not compute a single code.
    print("semq        FAILED:", e)
    print("*** codes cannot be computed on this box ***")
PY

chown -R ubuntu:ubuntu /home/ubuntu
echo "=== bootstrap complete $(date -u) ==="
