"""ARI metrics: per-input hash-equality and Hamming drift, aggregated to HER / H̄ with
bootstrap confidence intervals.

Definitions (spec/report-schema.json, spec/condition-set.md):
- HE(x)  = 1 if the code under the condition equals the baseline code bit-exactly
- H(x)   = Hamming distance (number of differing code positions) between the two codes
- HER    = mean_x HE(x)         (Hash Equality Rate ∈ [0,1])
- H̄      = mean_x H(x)          (mean Hamming drift)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# bit popcount lookup for uint8 — SEMQ codes are bit-packed bytes, so Hamming is the
# number of differing *bits*, i.e. popcount(baseline XOR condition), not differing bytes.
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class ConditionMetrics:
    HER: float
    Hbar: float
    HER_ci: tuple[float, float]
    n: int


def per_input(baseline_codes: np.ndarray, condition_codes: np.ndarray):
    """Return (he, hamming) per input. `codes` are (n, k) uint8 (bit-packed) matrices.
    HE = codes equal bit-exact; Hamming = number of differing bits."""
    if baseline_codes.shape != condition_codes.shape:
        raise ValueError("baseline and condition code matrices must have the same shape")
    a = np.ascontiguousarray(baseline_codes, dtype=np.uint8)
    b = np.ascontiguousarray(condition_codes, dtype=np.uint8)
    xor = np.bitwise_xor(a, b)                        # (n, k)
    hamming = _POPCOUNT[xor].sum(axis=1).astype(float)  # differing bits per input
    he = (hamming == 0.0)                            # bit-exact equality
    return he, hamming


def _bootstrap_ci(values: np.ndarray, n_resamples: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = values[rng.integers(0, n, n)].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def aggregate(baseline_codes, condition_codes, n_resamples: int = 1000, seed: int = 0) -> ConditionMetrics:
    he, hamming = per_input(baseline_codes, condition_codes)
    return ConditionMetrics(
        HER=float(he.mean()),
        Hbar=float(hamming.mean()),
        HER_ci=_bootstrap_ci(he.astype(float), n_resamples, seed),
        n=len(he),
    )
