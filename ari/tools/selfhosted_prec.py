# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0 and patent pending. The SDK is not covered by the
# Apache License.
"""The `prec` diagnostic for a self-hosted model: fp32 against a fresh bf16 load, on CPU.

The bf16 model is loaded fresh in bf16, never cast in place: casting a loaded model
crushes fp32-born rotary buffers (deployed-agent-panel/RESULTS.md). Both loads go through
ari.st_load, which verifies the dtype.

  python selfhosted_prec.py --model BAAI/bge-large-en-v1.5 --inputs data/ari-bench-v0.1.jsonl --out prec.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS))

from ari import metrics                                   # noqa: E402
from ari.inputs import load_ari_bench                     # noqa: E402
from ari.probe import load_probe                          # noqa: E402
from ari.st_load import load_sentence_transformer         # noqa: E402


def encode(model_id: str, dtype: str, texts: list[str]) -> np.ndarray:
    m = load_sentence_transformer(model_id, device="cpu", dtype=dtype, trust_remote_code=True)
    return np.asarray(m.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                               batch_size=32), dtype=np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure the self-hosted `prec` diagnostic.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    texts = load_ari_bench(args.inputs).texts[:args.n]
    fp32 = encode(args.model, "fp32", texts)
    bf16 = encode(args.model, "bf16", texts)
    probe = load_probe(fp32)
    m = metrics.aggregate(probe.encode(fp32), probe.encode(bf16))
    out = {"model": args.model, "n": len(texts), "semq": probe.version, "s": probe.s,
           "prec_bf16": {"HER": round(m.HER, 6), "Hbar": round(m.Hbar, 6),
                         "HER_ci": [round(m.HER_ci[0], 6), round(m.HER_ci[1], 6)]}}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{args.model}: prec(bf16) HER={m.HER:.4f} H̄={m.Hbar:.3f} n={len(texts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
