"""Drift-sensitivity sweep → the `(b, κ)` fingerprint for a self-hosted model.

Perturbs a model's embeddings with isotropic Gaussian noise across a σ grid, measures the
mean fraction of SEMQ code positions that flip, and fits the power law `Hamming = a·σ^b` on the
linear-regime window — then κ = a / √(2·dim/π). Reproduces the drift-sensitivity methodology
(experiments/drift-sensitivity/README.md) with the real SEMQ probe. BLAS threads pinned so the
measured drift is the injected σ, not host non-determinism.

  python drift_sweep.py --model sentence-transformers/all-MiniLM-L6-v2 --inputs data/ari-bench-v0.1.jsonl --n 1000
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS))

from ari.probe import load_probe                       # noqa: E402
# (Hamming is byte-level, computed inline — no metrics import needed)
from ari.inputs import InputSet, load_ari_bench        # noqa: E402

SIGMAS = np.logspace(-7, -2, 11)
FIT_LO, FIT_HI = 1e-5, 1e-3


def sweep_from_embeddings(X: np.ndarray, model_id: str, seed: int = 0) -> dict:
    """Run the σ-sweep on already-computed FP32 embeddings (source-agnostic: a
    SentenceTransformer or a black-box API both land here)."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    dim = int(X.shape[1])
    probe = load_probe(X, backend="semq")
    clean = probe.encode(X)
    mean_norm = float(np.linalg.norm(X, axis=1).mean())
    rng = np.random.default_rng(seed)

    curve = []
    for sigma in SIGMAS:
        noisy = X + rng.normal(0.0, sigma * mean_norm, size=X.shape).astype(np.float32)
        # Hamming = mean fraction of code *positions* (packed bytes) that disagree — the
        # per-row code-disagreement measure used by the reference drift-sensitivity methodology
        # (note: this is byte-position level, distinct from the bit-level popcount in ari.metrics).
        frac = float((clean != probe.encode(noisy)).mean())
        curve.append((float(sigma), frac))

    s = np.array([c[0] for c in curve]); h = np.array([c[1] for c in curve])
    mask = (s >= FIT_LO) & (s <= FIT_HI) & (h > 0)
    b, log_a = np.polyfit(np.log(s[mask]), np.log(h[mask]), 1)
    a = float(np.exp(log_a))
    kappa = a / np.sqrt(2 * dim / np.pi)
    return {"model": model_id, "dim": dim, "s": round(float(getattr(probe, "s", 0.0)), 6),
            "slope_b": round(float(b), 4), "prefactor_a": round(a, 4),
            "kappa": round(float(kappa), 3),
            "curve": [[round(x, 10), round(y, 6)] for x, y in curve]}


def sweep(model_id: str, texts, seed: int = 0) -> dict:
    """Self-hosted path — encode with SentenceTransformer (raw, normalized, no prefix, exactly
    as ari.agents.SentenceTransformerAgent does), then sweep."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_id, device="cpu", trust_remote_code=True)
    X = m.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return sweep_from_embeddings(X, model_id, seed=seed)


def sweep_api(agent_key: str, texts, seed: int = 0) -> dict:
    """API path — pull raw FP32 vectors from the black-box agent (one encode pass), then the
    σ-sweep is entirely offline. κ is a property of the embedding geometry, so a single clean
    encode is enough."""
    from ari.agents import DEFAULT_API_MODELS
    cls, model = DEFAULT_API_MODELS[agent_key]
    X = np.asarray(cls(model_id=model).encode(texts), dtype=np.float32)
    return sweep_from_embeddings(X, f"{agent_key}/{model}", seed=seed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Drift-sensitivity sweep → (b, κ) fingerprint.")
    g = ap.add_mutually_exclusive_group(required=True)
    from ari.agents import DEFAULT_API_MODELS
    g.add_argument("--model", help="a SentenceTransformer id (self-hosted)")
    g.add_argument("--agent", choices=list(DEFAULT_API_MODELS), help="a black-box API agent")
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="write the full result JSON here")
    args = ap.parse_args(argv)

    inputs = load_ari_bench(args.inputs)
    if args.n and len(inputs.ids) > args.n:
        inputs = InputSet(inputs.name, inputs.ids[:args.n], inputs.texts[:args.n])

    r = sweep_api(args.agent, inputs.texts, seed=args.seed) if args.agent \
        else sweep(args.model, inputs.texts, seed=args.seed)
    if args.out:
        args.out.write_text(json.dumps(r, indent=2))
    print(f"{r['model']}: dim={r['dim']} s={r['s']} b={r['slope_b']} kappa={r['kappa']}  "
          f"(a={r['prefactor_a']}, n={len(inputs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
