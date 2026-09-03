"""Self-hosted contrast — a model *you* control, measured with the same SEMQ probe as the
API panel, to show the governance point: self-hosted is reproducible, the API is not.

For a self-hosted model the conditions differ from an API:

  same  — re-encode in the SAME process (within-process determinism control → 1.0)
  proc  — re-encode in a FRESH subprocess (the real cross-process reproducibility test;
          with BLAS threads pinned it should be 1.0; unpinned it can drift, and SEMQ
          catches that — the classic sub-microsigma BLAS non-determinism)

Runs locally, no external keys. Default model matches the ARI reference (bge-large-en-v1.5).

    python selfhosted_pilot.py --model BAAI/bge-large-en-v1.5 --n 64 --resamples 3
    python selfhosted_pilot.py --no-pin-threads   # demonstrate cross-process BLAS drift
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PIN = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
WORKER = Path(__file__).parent / "_encode_worker.py"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Self-hosted reproducibility contrast (bge).")
    ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--resamples", type=int, default=3)
    ap.add_argument("--probe-backend", choices=["auto", "semq", "mock"], default="semq")
    ap.add_argument("--pin-threads", dest="pin", action="store_true", default=True)
    ap.add_argument("--no-pin-threads", dest="pin", action="store_false")
    ap.add_argument("--out", type=Path, default=Path("selfhosted_pilot.json"))
    args = ap.parse_args(argv)

    if args.pin:
        os.environ.update(PIN)  # affects the parent's in-process encodes (before torch import)

    from ari import metrics                       # noqa: E402
    from ari.inputs import sample_inputs          # noqa: E402
    from ari.probe import load_probe              # noqa: E402
    from sentence_transformers import SentenceTransformer  # noqa: E402

    inputs = sample_inputs(args.n)
    model = SentenceTransformer(args.model, device="cpu", trust_remote_code=True)

    def encode_inproc(texts):
        return np.asarray(model.encode(texts, normalize_embeddings=True, convert_to_numpy=True),
                          dtype=np.float32)

    def encode_subproc(texts):
        with tempfile.TemporaryDirectory() as d:
            out = f"{d}/v.npy"
            pj = f"{d}/payload.json"
            json.dump({"model": args.model, "texts": texts, "out": out}, open(pj, "w"))
            env = {**os.environ, **(PIN if args.pin else {})}
            subprocess.run([sys.executable, str(WORKER), pj], check=True, env=env)
            return np.load(out)

    base = encode_inproc(inputs.texts)
    probe = load_probe(base, backend=args.probe_backend)
    base_codes = probe.encode(base)

    conditions = {}
    for kind, enc in (("same", encode_inproc), ("proc", encode_subproc)):
        hers, hbars = [], []
        for _ in range(args.resamples):
            m = metrics.aggregate(base_codes, probe.encode(enc(inputs.texts)))
            hers.append(m.HER); hbars.append(m.Hbar)
        conditions[kind] = {"HER_mean": round(st.mean(hers), 6),
                            "HER_min": round(min(hers), 6),
                            "Hbar_mean": round(st.mean(hbars), 4),
                            "per_resample_HER": [round(h, 6) for h in hers]}

    result = {
        "agent": args.model, "agent_class": "self_hosted", "n_inputs": len(inputs),
        "resamples": args.resamples, "probe_backend": probe.backend,
        "threads_pinned": args.pin, "conditions": conditions,
        "findings": {
            "within_process_deterministic": conditions["same"]["HER_mean"] >= 1 - 1e-9,
            "cross_process_reproducible": conditions["proc"]["HER_mean"] >= 1 - 1e-9,
        },
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"agent={args.model}  probe={probe.backend}  n={len(inputs)}  K={args.resamples}  pinned={args.pin}")
    print(f"  same HER = {conditions['same']['HER_mean']:.4f}  (within-process deterministic: {result['findings']['within_process_deterministic']})")
    print(f"  proc HER = {conditions['proc']['HER_mean']:.4f}  (cross-process reproducible: {result['findings']['cross_process_reproducible']})")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
