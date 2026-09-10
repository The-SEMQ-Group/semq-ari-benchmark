"""Measure floating-point non-associativity at embedding magnitudes.

The paper's first paragraph now rests on one claim: addition does not associate, so the order
a sum accumulates in decides its result. Everything else (reduction order, kernel selection,
batch shape, hardware, precision) acts through that. This script makes the claim a measurement
instead of an assertion, at the vector widths and magnitudes the encoders actually produce.

It reports two things per dtype:

  pairwise      the largest |(a+b)+c - a+(b+c)| over sampled triples
  permutation   the spread of one dot product summed in many different orders

The permutation number is the one that matters. A GPU does not reassociate three scalars, it
partitions a long sum across threads and combines the partials, which is exactly a permuted
accumulation of the same terms.

  python nonassociativity.py --out results/nonassociativity.json
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

# all-MiniLM-L6-v2 is 384-wide and the model behind the SciFact figure; bge-large is 1024.
# Coordinate scale follows the published QBIN scales, which sit near 0.1 for these encoders.
WIDTHS = (384, 1024)
COORD_SCALE = 0.1


def pairwise_worst(rng, dtype, trials):
    """Largest disagreement between (a+b)+c and a+(b+c) over random triples."""
    a = (rng.standard_normal(trials) * COORD_SCALE).astype(dtype)
    b = (rng.standard_normal(trials) * COORD_SCALE).astype(dtype)
    c = (rng.standard_normal(trials) * COORD_SCALE).astype(dtype)
    left = (a + b) + c
    right = a + (b + c)
    diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return float(diff.max()), int((diff > 0).sum()), trials


def permutation_spread(rng, dtype, width, orders):
    """Spread of one dot product accumulated in `orders` different term orders.

    Sequential summation in a permuted order stands in for the partial-sum tree a GPU builds.
    The terms are identical every time, so any spread comes from order alone.
    """
    x = (rng.standard_normal(width) * COORD_SCALE).astype(dtype)
    y = (rng.standard_normal(width) * COORD_SCALE).astype(dtype)
    terms = (x * y).astype(dtype)
    sums = []
    for _ in range(orders):
        idx = rng.permutation(width)
        acc = dtype(0)
        for t in terms[idx]:
            acc = dtype(acc + t)
        sums.append(float(acc))
    exact = float(np.dot(x.astype(np.float64), y.astype(np.float64)))
    return {
        "width": width,
        "distinct_results": len(set(sums)),
        "orders_tried": orders,
        "spread": max(sums) - min(sums),
        "max_abs_error_vs_float64": max(abs(s - exact) for s in sums),
        "mean_result": sum(sums) / len(sums),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("results/nonassociativity.json"))
    ap.add_argument("--trials", type=int, default=200_000)
    ap.add_argument("--orders", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260908)
    a = ap.parse_args()

    report = {"platform": platform.platform(), "numpy": np.__version__,
              "coord_scale": COORD_SCALE, "seed": a.seed, "dtypes": {}}

    for name, dtype in (("float32", np.float32), ("float16", np.float16)):
        rng = np.random.default_rng(a.seed)          # same draws for every dtype
        worst, n_diff, n = pairwise_worst(rng, dtype, a.trials)
        entry = {"pairwise": {"max_abs_diff": worst, "triples_that_differ": n_diff,
                              "triples_tried": n}, "permutation": []}
        for w in WIDTHS:
            entry["permutation"].append(permutation_spread(rng, dtype, w, a.orders))
        report["dtypes"][name] = entry

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")

    for name, e in report["dtypes"].items():
        p = e["pairwise"]
        print(f"{name}: {p['triples_that_differ']}/{p['triples_tried']} triples reassociate to "
              f"a different value, worst {p['max_abs_diff']:.3e}")
        for perm in e["permutation"]:
            print(f"  width {perm['width']}: {perm['distinct_results']} distinct sums from "
                  f"{perm['orders_tried']} orders, spread {perm['spread']:.3e}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
