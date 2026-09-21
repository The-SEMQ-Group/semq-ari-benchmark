# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library that is subject to a
# commercial license owned by The SEMQ Group Inc. and is patent pending.
# The SDK is not covered by the Apache License.
"""Refit the noise response of one encoder in coordinate units.

The historical `(b, kappa)` registry fitted the packed-byte disagreement
rate. This tool runs the same sweep (README.md: canonical probe calibrated
once, Gaussian noise at `sigma` times the mean representation norm added
before any re-normalization, 11-point grid, fit window [1e-5, 1e-3]) and
records three rates per input per cell from `ari.code_metrics.code_diff`:
the coordinate change rate, the bit Hamming rate and the legacy byte rate.
The primary fit is on the coordinate rate; the other two are fitted the
same way so the unit conversion is measured, not assumed.

Outputs, under --out-dir:
  <model>.json   fits, bootstrap intervals, residuals, inverse validation
  <model>.npz    per-input rates per cell, for later resampling

  python experiments/drift-sensitivity/refit_coordinate.py \
      --model sentence-transformers/all-MiniLM-L6-v2
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS))

from ari.code_metrics import code_diff                                  # noqa: E402
from ari.inputs import InputSet, load_ari_bench                         # noqa: E402
from ari.probe import N_BINS, PERCENTILE, load_probe                    # noqa: E402
from ari.response_fit import (FIT_HI, FIT_LO, SIGMAS, bootstrap_power_law,  # noqa: E402
                              fit_power_law, sphere_prefactor, validate_inverse)

# Below the grid, where the registry's floor_limited note places the
# sigma-independent floor. A rate here is a boundary-degeneracy count, not
# a response to noise.
FLOOR_SIGMA = 1e-10
NEAR_ZERO = 1e-6

RATES = ("coordinate_change_rate", "bit_hamming_rate", "byte_change_rate")


def encode_inputs(model_id: str, texts: list[str], device: str) -> np.ndarray:
    """Same encode as ari.agents.SentenceTransformerAgent: normalized, no prefix."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id, device=device, trust_remote_code=True)
    return np.ascontiguousarray(
        model.encode(texts, normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)


def sweep(X: np.ndarray, sigmas: np.ndarray, seed: int) -> tuple[dict, dict]:
    """One calibration, then one noisy encode per sigma.

    Returns per-input rate matrices ``(n_sigma, n_inputs)`` keyed by rate
    name, and the calibration facts (scale, mean norm, floor probe).
    """
    dim = int(X.shape[1])
    probe = load_probe(X)
    clean = probe.encode(X)
    mean_norm = float(np.linalg.norm(X, axis=1).mean())
    rng = np.random.default_rng(seed)

    per_input = {k: np.zeros((len(sigmas), X.shape[0])) for k in RATES}
    equality = np.zeros(len(sigmas))
    for i, sigma in enumerate(sigmas):
        noisy = X + rng.normal(0.0, sigma * mean_norm, size=X.shape).astype(np.float32)
        d = code_diff(clean, probe.encode(noisy), n_bins=N_BINS, dim=dim)
        for k in RATES:
            per_input[k][i] = getattr(d, k)
        equality[i] = float(d.codes_equal.mean())

    # Floor probe: a separate draw after the grid so the grid's noise is the
    # same whether or not this cell is present.
    noisy = X + rng.normal(0.0, FLOOR_SIGMA * mean_norm, size=X.shape).astype(np.float32)
    floor = code_diff(clean, probe.encode(noisy), n_bins=N_BINS, dim=dim)
    floor_rates = {k: float(getattr(floor, k).mean()) for k in RATES}
    floor_per_input = {k: np.asarray(getattr(floor, k), dtype=float) for k in RATES}

    facts = {
        "dim": dim,
        "n_inputs": int(X.shape[0]),
        "n_bins": N_BINS,
        "calibration_percentile": PERCENTILE,
        "s": float(probe.s),
        "mean_norm": mean_norm,
        "n_bytes": int(clean.shape[1]),
        "bits_per_coordinate": floor.bits_per_coordinate,
        "exact_zero_coordinate_fraction": float((X == 0).mean()),
        # A coordinate this close to the sign boundary changes symbol under
        # any noise, so this fraction is the floor a sweep cannot go below.
        "near_zero_coordinate_fraction": float((np.abs(X) < NEAR_ZERO).mean()),
        "code_equality_rate": [float(v) for v in equality],
        "floor_probe": {"sigma": FLOOR_SIGMA, **floor_rates,
                        "code_equality_rate": float(floor.codes_equal.mean())},
    }
    if hasattr(probe, "close"):
        probe.close()
    return per_input, floor_per_input, facts


def fit_one(sigmas: np.ndarray, matrix: np.ndarray, dim: int, *, n_resamples: int,
            bootstrap_seed: int) -> dict:
    means = matrix.mean(axis=1)
    try:
        fit = fit_power_law(sigmas, means, dim, lo=FIT_LO, hi=FIT_HI)
    except ValueError as e:
        return {"error": str(e), "mean_rate": [float(v) for v in means]}
    boot = bootstrap_power_law(sigmas, matrix, dim, n_resamples=n_resamples,
                               seed=bootstrap_seed, lo=FIT_LO, hi=FIT_HI)
    # Residuals over the whole grid show where the power law bends; the
    # fit's own residuals cover the window only.
    pred = fit.predict(sigmas)
    grid_resid = [float(np.log(m) - np.log(p)) if m > 0 else None
                  for m, p in zip(means, pred)]
    return {
        "mean_rate": [float(v) for v in means],
        "fit": fit.to_dict(),
        "bootstrap": boot.to_dict(),
        "grid_residuals_log": grid_resid,
        "inverse": validate_inverse(sigmas, means, fit, bootstrap=boot,
                                    lo=FIT_LO, hi=FIT_HI),
    }


def fit_all(sigmas: np.ndarray, per_input: dict, floor_per_input: dict, dim: int, *,
            n_resamples: int, bootstrap_seed: int) -> dict:
    """Fit every rate the same way; the coordinate fit is the one that matters.

    ``coordinate_change_rate_floor_subtracted`` removes each input's own
    floor-probe rate from every cell before fitting. It is a diagnostic for
    a floor-limited model, not a registry-format fit: the registry fits the
    raw rate.
    """
    out = {k: fit_one(sigmas, per_input[k], dim, n_resamples=n_resamples,
                      bootstrap_seed=bootstrap_seed) for k in RATES}
    adjusted = per_input["coordinate_change_rate"] - floor_per_input["coordinate_change_rate"]
    out["coordinate_change_rate_floor_subtracted"] = fit_one(
        sigmas, adjusted, dim, n_resamples=n_resamples, bootstrap_seed=bootstrap_seed)
    return out


def unit_ratios(sigmas: np.ndarray, per_input: dict) -> list[dict]:
    """Per-cell means of all three rates and their ratios to the coordinate rate."""
    rows = []
    for i, sigma in enumerate(sigmas):
        coord = float(per_input["coordinate_change_rate"][i].mean())
        bit = float(per_input["bit_hamming_rate"][i].mean())
        byte = float(per_input["byte_change_rate"][i].mean())
        rows.append({
            "sigma": float(sigma),
            "coordinate_change_rate": coord,
            "bit_hamming_rate": bit,
            "byte_change_rate": byte,
            "byte_over_coordinate": byte / coord if coord > 0 else None,
            "bit_over_coordinate": bit / coord if coord > 0 else None,
        })
    return rows


def environment() -> dict:
    import semq
    import sentence_transformers
    import torch
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "semq": str(getattr(semq, "__version__", "unknown")),
        "sentence_transformers": sentence_transformers.__version__,
        "torch": torch.__version__,
        "threads": {k: os.environ.get(k) for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS")},
    }


def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def run(model_id: str, inputs: InputSet, *, seed: int, bootstrap_seed: int,
        n_resamples: int, device: str, out_dir: Path,
        embeddings: np.ndarray | None = None) -> dict:
    t0 = time.time()
    X = embeddings if embeddings is not None else encode_inputs(model_id, inputs.texts, device)
    if X.shape[0] != len(inputs):
        raise ValueError(f"{X.shape[0]} embeddings for {len(inputs)} inputs")
    sigmas = np.asarray(SIGMAS, dtype=float)
    per_input, floor_per_input, facts = sweep(X, sigmas, seed)
    fits = fit_all(sigmas, per_input, floor_per_input, facts["dim"],
                   n_resamples=n_resamples, bootstrap_seed=bootstrap_seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{slug(model_id)}.npz"
    np.savez_compressed(
        npz_path, sigmas=sigmas, input_ids=np.array(inputs.ids),
        model_id=model_id, noise_seed=seed, floor_sigma=FLOOR_SIGMA,
        **{k: per_input[k] for k in RATES},
        **{f"floor_{k}": floor_per_input[k] for k in RATES})

    result = {
        "model_id": model_id,
        "unit_of_primary_fit": "coordinate_change_rate",
        "inputs": {"name": inputs.name, "n": len(inputs), "content_hash": inputs.content_hash},
        "seeds": {"noise": seed, "bootstrap": bootstrap_seed},
        "sigma_definition": "per-coordinate Gaussian std as a fraction of the mean "
                            "representation norm; added before any re-normalization",
        "sigma_grid": [float(v) for v in sigmas],
        "fit_window": [FIT_LO, FIT_HI],
        "sphere_prefactor": sphere_prefactor(facts["dim"]),
        **facts,
        "cells": unit_ratios(sigmas, per_input),
        "fits": fits,
        "per_input_file": npz_path.name,
        "environment": environment(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / f"{slug(model_id)}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="a SentenceTransformer model id")
    ap.add_argument("--inputs", type=Path, default=HARNESS / "data" / "ari-bench-v0.1.jsonl")
    ap.add_argument("--n", type=int, default=0, help="use only the first n inputs (0 = all)")
    ap.add_argument("--seed", type=int, default=0, help="noise RNG seed")
    ap.add_argument("--bootstrap-seed", type=int, default=1)
    ap.add_argument("--bootstrap", type=int, default=1000, help="resamples over inputs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--embeddings", type=Path, default=None,
                    help="an .npy of precomputed FP32 embeddings in input order")
    ap.add_argument("--save-embeddings", type=Path, default=None,
                    help="write the FP32 embeddings here as .npy for later refits")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "results" / "coordinate")
    args = ap.parse_args(argv)

    inputs = load_ari_bench(args.inputs)
    if args.n and len(inputs) > args.n:
        inputs = InputSet(inputs.name, inputs.ids[:args.n], inputs.texts[:args.n])
    emb = np.load(args.embeddings) if args.embeddings else None
    if emb is None and args.save_embeddings:
        emb = encode_inputs(args.model, inputs.texts, args.device)
        np.save(args.save_embeddings, emb)

    r = run(args.model, inputs, seed=args.seed, bootstrap_seed=args.bootstrap_seed,
            n_resamples=args.bootstrap, device=args.device, out_dir=args.out_dir,
            embeddings=emb)
    for k, f in r["fits"].items():
        if "error" in f:
            print(f"{k}: {f['error']}")
            continue
        b_lo, b_hi = f["bootstrap"]["b_ci"]
        k_lo, k_hi = f["bootstrap"]["kappa_ci"]
        print(f"{k}: b={f['fit']['b']:.4f} [{b_lo:.4f}, {b_hi:.4f}]  "
              f"kappa={f['fit']['kappa']:.3f} [{k_lo:.3f}, {k_hi:.3f}]  "
              f"R2(log)={f['fit']['r2_log']:.4f}")
    print(f"floor probe at sigma={FLOOR_SIGMA:g}: "
          + ", ".join(f"{k}={v:.2e}" for k, v in r["floor_probe"].items() if k in RATES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
