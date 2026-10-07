# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0 and patent pending. The SDK is not covered by the
# Apache License.
"""The `lib` condition, measured: what a transformers upgrade does to a default load.

transformers 4 loads a checkpoint in fp32 unless told otherwise; transformers 5 loads
it in the dtype its config declares. For a model that ships half precision, the same
user code (`SentenceTransformer(model_id)`) changes precision on upgrade. This script
encodes the frozen inputs twice under transformers 5 — once in an explicit fp32 load,
once with the library default — and reports HER and rho between the two.

    python run.py --device cuda --out results/lib_default_dtype.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

# Popular embedding models whose config declares half precision (hf_dtype_scan.json).
MODELS = [
    "Qwen/Qwen3-Embedding-0.6B",
    "ibm-granite/granite-embedding-small-english-r2",
    "mixedbread-ai/mxbai-embed-large-v1",
    "intfloat/multilingual-e5-large-instruct",
    "Alibaba-NLP/gte-multilingual-base",
    "thenlper/gte-small",
    "google/embeddinggemma-2",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    from ari import metrics
    from ari.inputs import load_ari_bench
    from ari.probe import load_probe
    from ari.st_load import load_sentence_transformer

    texts = load_ari_bench(REPO / "data" / "ari-bench-v0.1.jsonl").texts
    rows = []
    for mid in MODELS:
        try:
            explicit = load_sentence_transformer(mid, device=a.device, dtype="fp32", trust_remote_code=True)
            v32 = explicit.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
            del explicit
            torch.cuda.empty_cache() if a.device == "cuda" else None
            default = SentenceTransformer(mid, device=a.device, trust_remote_code=True)   # what users write
            loaded = str(next(default.parameters()).dtype).replace("torch.", "")
            vd = default.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
            del default
            torch.cuda.empty_cache() if a.device == "cuda" else None
            v32, vd = np.asarray(v32, np.float32), np.asarray(vd, np.float32)
            probe = load_probe(v32)
            m = metrics.aggregate(probe.encode(v32), probe.encode(vd))
            cos = float(np.mean(np.sum(v32 * vd, axis=1)))
            rows.append({"model": mid, "dim": int(v32.shape[1]), "default_load_dtype": loaded,
                         "HER": m.HER, "rho": m.rho, "rho_ci": m.rho_ci, "mean_dot": cos})
            print(f"{mid}: default={loaded} HER={m.HER:.4f} rho={m.rho:.2e} dot={cos:.6f}", flush=True)
        except Exception as e:  # one model failing must not lose the others
            rows.append({"model": mid, "error": f"{type(e).__name__}: {e}"[:300]})
            print(f"{mid}: FAILED {e}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"transformers": transformers.__version__, "torch": torch.__version__,
                                 "device": a.device, "n": len(texts),
                                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                 "rows": rows}, indent=1))
    return 0 if all("error" not in r for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
