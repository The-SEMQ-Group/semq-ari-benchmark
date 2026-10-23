# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""Capture one episode or reference of the selectivity study, in its own process.

Encodes all frozen inputs with the settings episodes.json gives for the id, and
saves the raw float32 vectors plus a metadata record. Vectors are kept so every
pre-registered detector can be computed offline.

    python capture_episode.py --id proc-000 --out OUTDIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))


def _source_revision() -> str | None:
    """The harness commit, with "-dirty" if the tree has changes; None without git."""
    try:
        rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return rev + ("-dirty" if dirty else "")


def _snapshot_revision(path) -> str | None:
    """The Hugging Face commit a local snapshot directory belongs to, from its path
    (``.../snapshots/<40-hex commit>/...``); None if the path is not a hub snapshot."""
    m = re.search(r"/snapshots/([0-9a-f]{40})(?:/|$)", str(path or ""))
    return m.group(1) if m else None


def _revisions(model) -> dict:
    """The model and tokenizer revisions actually loaded. Raises if they differ: the
    study assumes both come from one snapshot."""
    first = model[0]
    model_rev = (_snapshot_revision(getattr(first.auto_model, "name_or_path", None))
                 or getattr(first.auto_model.config, "_commit_hash", None))
    tok_rev = (_snapshot_revision(getattr(first.tokenizer, "name_or_path", None))
               or getattr(first.tokenizer, "init_kwargs", {}).get("_commit_hash"))
    if model_rev and tok_rev and model_rev != tok_rev:
        raise RuntimeError(f"model revision {model_rev} and tokenizer revision {tok_rev} differ")
    return {"model_revision": model_rev, "tokenizer_revision": tok_rev}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    plan = json.loads((HERE / "episodes.json").read_text())
    spec = next((e for e in plan["episodes"] + plan["references"] if e["id"] == args.id), None)
    if spec is None:
        print(f"unknown id {args.id}")
        return 2

    import numpy as np
    import torch

    torch.set_num_threads(int(spec.get("threads", 4)))
    if spec["device"] == "cuda":
        on = spec.get("tf32") == "on"
        torch.backends.cuda.matmul.allow_tf32 = on
        torch.backends.cudnn.allow_tf32 = on

    from ari.inputs import load_ari_bench
    from ari.st_load import load_sentence_transformer

    from ari.inputs import InputSet

    full = load_ari_bench(REPO / plan["inputs"])
    inputs = InputSet(full.name, full.ids[: plan["n_inputs"]], full.texts[: plan["n_inputs"]])
    texts = inputs.texts
    t0 = time.time()
    model = load_sentence_transformer(spec["model"], device=spec["device"], dtype=spec.get("dtype", "fp32"),
                                      trust_remote_code=True, revision=spec.get("revision"))
    if spec.get("int8"):
        torch.backends.quantized.engine = ("qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines
                                           else "fbgemm")
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    t1 = time.time()
    v = model.encode(texts, batch_size=int(spec["batch"]), normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False)
    v = np.ascontiguousarray(np.asarray(v, dtype=np.float32))
    t2 = time.time()

    args.out.mkdir(parents=True, exist_ok=True)
    tmp = args.out / f"{args.id}.tmp.npy"
    np.save(tmp, v)
    tmp.rename(args.out / f"{args.id}.npy")
    import sentence_transformers
    import transformers
    meta = {"id": args.id, "spec": spec, "shape": list(v.shape), "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "sentence_transformers": sentence_transformers.__version__,
            "load_s": round(t1 - t0, 1), "encode_s": round(t2 - t1, 1),
            # Provenance, recorded since 2026-10-14: the model and tokenizer revisions actually
            # loaded (checked to be one snapshot), the inputs, the vectors and the harness. A
            # revision that cannot be resolved is recorded as null, not guessed.
            **_revisions(model),
            "inputs_content_hash": inputs.content_hash,
            "vectors_sha256": hashlib.sha256(v.tobytes()).hexdigest(),
            "source_revision": _source_revision(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (args.out / f"{args.id}.json").write_text(json.dumps(meta, indent=1))
    print(f"{args.id} {spec['model']} {spec.get('condition', spec.get('reference'))} {v.shape} "
          f"load {meta['load_s']}s encode {meta['encode_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
