# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library that is subject to a
# commercial license owned by The SEMQ Group Inc. and is patent pending.
# The SDK is not covered by the Apache License.
"""Measure which coordinates make up a floor-limited model's floor.

Encodes the frozen inputs as refit_coordinate.py does, replays its noise
draws so the floor cell is the same draw, and records the fraction of
coordinates below each magnitude threshold beside the fraction that
changed symbol at FLOOR_SIGMA. Output: <out-dir>/<model>.floor.json.

  python experiments/drift-sensitivity/floor_histogram.py \
      --model sentence-transformers/all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refit_coordinate import FLOOR_SIGMA, HARNESS, encode_inputs, slug  # noqa: E402
from ari.code_metrics import unpack_symbols                            # noqa: E402
from ari.inputs import load_ari_bench                                  # noqa: E402
from ari.probe import N_BINS, load_probe                               # noqa: E402
from ari.response_fit import SIGMAS                                    # noqa: E402

THRESHOLDS = (1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6)


def floor_histogram(X: np.ndarray, seed: int) -> dict:
    dim = int(X.shape[1])
    probe = load_probe(X)
    clean = unpack_symbols(probe.encode(X), N_BINS, dim)
    mean_norm = float(np.linalg.norm(X, axis=1).mean())
    rng = np.random.default_rng(seed)
    for sigma in SIGMAS:  # the sweep draws the grid first; replay it
        rng.normal(0.0, sigma * mean_norm, size=X.shape)
    noise = rng.normal(0.0, FLOOR_SIGMA * mean_norm, size=X.shape).astype(np.float32)
    noisy = X + noise
    flipped = unpack_symbols(probe.encode(noisy), N_BINS, dim) != clean
    if hasattr(probe, "close"):
        probe.close()
    sign_changed = np.sign(X) != np.sign(noisy)
    ax = np.abs(X)
    fx = ax[flipped]
    return {
        "floor_sigma": FLOOR_SIGMA,
        "noise_std": FLOOR_SIGMA * mean_norm,
        "n_coordinates": int(X.size),
        "coordinate_change_rate": float(flipped.mean()),
        "sign_change_rate": float(sign_changed.mean()),
        "flipped_equals_sign_changed": bool((flipped == sign_changed).all()),
        "exact_zero_fraction": float((X == 0).mean()),
        "below_threshold": {
            f"{t:g}": {"fraction": float((ax < t).mean()),
                       "flipped_fraction_within": float(flipped[ax < t].mean()) if (ax < t).any() else None}
            for t in THRESHOLDS},
        "flipped_abs_x": {"min": float(fx.min()), "median": float(np.median(fx)),
                          "p99": float(np.quantile(fx, 0.99)), "max": float(fx.max())},
        "unflipped_abs_x_min": float(ax[~flipped].min()),
        "per_input_below_1e-8": {"min": int((ax < 1e-8).sum(axis=1).min()),
                                 "max": int((ax < 1e-8).sum(axis=1).max())},
        "per_input_flipped": {"min": int(flipped.sum(axis=1).min()),
                              "max": int(flipped.sum(axis=1).max())},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", type=Path, default=HARNESS / "data" / "ari-bench-v0.1.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--embeddings", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "results" / "coordinate")
    args = ap.parse_args(argv)
    inputs = load_ari_bench(args.inputs)
    X = (np.load(args.embeddings) if args.embeddings
         else encode_inputs(args.model, inputs.texts, args.device))
    out = {"model_id": args.model, "inputs": {"name": inputs.name, "n": len(inputs),
                                              "content_hash": inputs.content_hash},
           "noise_seed": args.seed, **floor_histogram(X, args.seed)}
    path = args.out_dir / f"{slug(args.model)}.floor.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
