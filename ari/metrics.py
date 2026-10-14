# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""ARI metrics: per-input hash-equality and Hamming drift, aggregated to HER / H̄ with
bootstrap confidence intervals.

Definitions (spec/report-schema.json, spec/condition-set.md):
- HE(x)  = 1 if the code under the condition equals the baseline code bit-exactly
- H(x)   = Hamming distance in bits (popcount of the XOR) between the two codes
- HER    = mean_x HE(x)         (Hash Equality Rate ∈ [0,1])
- H̄      = mean_x H(x)          (mean Hamming drift)
- ρ      = mean_x (coordinates whose symbol changed / d)   (coordinate change rate)

HER asks whether the whole code still matches, so it falls as the dimension
grows at equal per-coordinate noise. ρ removes the dimension from the denominator:
under the v0.2 probe the bin edges sit at the same position relative to a
unit-norm vector's typical coordinate (range 2/sqrt(d)). It does not make
sensitivity model-independent: how often a symbol changes still depends on the
model's coordinate distribution, the mass near bin edges, and the structure of
the perturbation. ARI-R = 1 - mean ρ over the core.
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
    rho: float | None = None
    rho_ci: tuple[float, float] | None = None
    dim: int | None = None  # coordinates rho was computed over


def per_input(baseline_codes: np.ndarray, condition_codes: np.ndarray):
    """Return (he, hamming) per input. `codes` are (n, k) uint8 (bit-packed) matrices.

    HE = codes equal bit-exact; Hamming = number of differing bits. Bits, not
    bytes and not coordinates: QUANT packs several coordinates into one byte,
    so a byte or coordinate count is a different number. Padding bits are
    equal in both operands, so they never contribute.
    """
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


def coordinate_change(baseline_codes, condition_codes, n_bins: int, dim: int) -> np.ndarray:
    """Per input, the fraction of the ``dim`` coordinates whose QUANT symbol changed.

    ``dim`` is required: the packed width does not determine it. At 2 bits per
    coordinate, four dimensions share each byte width (385 to 388 all pack into
    97 bytes), and counting padding coordinates would bias the rate low.
    """
    from ari.code_metrics import bits_per_coordinate, unpack_symbols

    a = np.ascontiguousarray(baseline_codes, dtype=np.uint8)
    b = np.ascontiguousarray(condition_codes, dtype=np.uint8)
    width = -(-int(dim) * bits_per_coordinate(n_bins) // 8)
    if a.shape[1] != width:
        raise ValueError(f"codes are {a.shape[1]} bytes wide; dim={dim} at n_bins={n_bins} "
                         f"packs into {width}")
    sa = unpack_symbols(a, n_bins=n_bins, dim=dim)
    sb = unpack_symbols(b, n_bins=n_bins, dim=dim)
    return (sa != sb).mean(axis=1)


def aggregate(baseline_codes, condition_codes, n_resamples: int = 1000, seed: int = 0,
              n_bins: int = 2, *, dim: int) -> ConditionMetrics:
    he, hamming = per_input(baseline_codes, condition_codes)
    rho = coordinate_change(baseline_codes, condition_codes, n_bins, dim)
    return ConditionMetrics(
        HER=float(he.mean()),
        Hbar=float(hamming.mean()),
        HER_ci=_bootstrap_ci(he.astype(float), n_resamples, seed),
        n=len(he),
        rho=float(rho.mean()),
        rho_ci=_bootstrap_ci(rho, n_resamples, seed),
        dim=int(dim),
    )
