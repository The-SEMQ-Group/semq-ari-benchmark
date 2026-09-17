# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""Power-law fit of a code change rate against injected noise, with uncertainty.

The drift-sensitivity experiment models the mean change rate ``rho`` at
noise magnitude ``sigma`` as ``rho = a * sigma**b`` and reports
``kappa = a / sqrt(2 * dim / pi)``, the ratio to the uniform-on-sphere
prediction. The historical registry fitted ``rho`` measured as a packed
byte disagreement rate. This module is unit-agnostic: it fits whatever
per-input rate matrix it is given, so the same code fits coordinate,
bit and byte rates and the caller names the unit.

Everything here is plain numpy over rates that were already measured.
No SEMQ operator is involved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

SIGMAS = np.logspace(-7, -2, 11)
FIT_LO, FIT_HI = 1e-5, 1e-3


def sphere_prefactor(dim: int) -> float:
    """``sqrt(2 * dim / pi)``: the uniform-on-sphere prediction for ``a``."""
    return float(np.sqrt(2.0 * dim / np.pi))


def fit_window_mask(sigmas: np.ndarray, lo: float = FIT_LO, hi: float = FIT_HI) -> np.ndarray:
    """Cells inside the fit window. The bounds are inclusive with a relative
    tolerance, because ``logspace`` does not land exactly on 1e-5 and 1e-3."""
    s = np.asarray(sigmas, dtype=float)
    return (s >= lo * (1 - 1e-9)) & (s <= hi * (1 + 1e-9))


@dataclass(frozen=True)
class PowerLawFit:
    """``log rho = log a + b log sigma`` over the cells in ``sigmas_used``."""

    b: float
    log_a: float
    dim: int
    sigmas_used: list[float]
    rho_used: list[float]
    residuals_log: list[float]
    r2_log: float
    n_cells_dropped_zero: int

    @property
    def a(self) -> float:
        return float(np.exp(self.log_a))

    @property
    def kappa(self) -> float:
        return self.a / sphere_prefactor(self.dim)

    def predict(self, sigma) -> np.ndarray:
        return self.a * np.asarray(sigma, dtype=float) ** self.b

    def to_dict(self) -> dict:
        d = asdict(self)
        d["a"] = self.a
        d["kappa"] = self.kappa
        return d


def fit_power_law(sigmas: np.ndarray, rho: np.ndarray, dim: int, *,
                  lo: float = FIT_LO, hi: float = FIT_HI) -> PowerLawFit:
    """Least squares in log-log space over the window. Zero-rate cells have no
    logarithm and are dropped; their count is reported so a fit that quietly
    lost half its window is visible."""
    s = np.asarray(sigmas, dtype=float)
    r = np.asarray(rho, dtype=float)
    if s.shape != r.shape:
        raise ValueError(f"sigmas {s.shape} and rho {r.shape} differ in shape")
    window = fit_window_mask(s, lo, hi)
    usable = window & (r > 0)
    if usable.sum() < 2:
        raise ValueError(
            f"need at least two positive cells in [{lo}, {hi}], have {int(usable.sum())}")
    x, y = np.log(s[usable]), np.log(r[usable])
    b, log_a = np.polyfit(x, y, 1)
    resid = y - (log_a + b * x)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return PowerLawFit(
        b=float(b), log_a=float(log_a), dim=int(dim),
        sigmas_used=[float(v) for v in s[usable]],
        rho_used=[float(v) for v in r[usable]],
        residuals_log=[float(v) for v in resid],
        r2_log=float(r2),
        n_cells_dropped_zero=int((window & ~(r > 0)).sum()),
    )


@dataclass(frozen=True)
class BootstrapFit:
    """Percentile intervals from resampling inputs with replacement."""

    n_resamples: int
    n_inputs: int
    seed: int
    alpha: float
    b_samples: np.ndarray = field(repr=False)
    kappa_samples: np.ndarray = field(repr=False)
    n_failed: int

    def interval(self, samples: np.ndarray) -> tuple[float, float]:
        lo, hi = np.percentile(samples, [100 * self.alpha / 2, 100 * (1 - self.alpha / 2)])
        return float(lo), float(hi)

    def to_dict(self) -> dict:
        b_lo, b_hi = self.interval(self.b_samples)
        k_lo, k_hi = self.interval(self.kappa_samples)
        return {
            "n_resamples": self.n_resamples,
            "n_inputs": self.n_inputs,
            "seed": self.seed,
            "alpha": self.alpha,
            "n_failed": self.n_failed,
            "b_ci": [b_lo, b_hi],
            "b_se": float(self.b_samples.std(ddof=1)),
            "kappa_ci": [k_lo, k_hi],
            "kappa_se": float(self.kappa_samples.std(ddof=1)),
        }


def bootstrap_power_law(sigmas: np.ndarray, per_input: np.ndarray, dim: int, *,
                        n_resamples: int = 1000, seed: int = 1, alpha: float = 0.05,
                        lo: float = FIT_LO, hi: float = FIT_HI) -> BootstrapFit:
    """``per_input`` is ``(n_sigma, n_inputs)``: one rate per input per cell.

    Each resample draws inputs with replacement, takes the per-cell mean over
    the draw, and refits. Resampling inputs rather than cells keeps the
    dependence between cells (the same input appears at every sigma) intact.
    A resample whose window has fewer than two positive cells is counted in
    ``n_failed`` and left out of the intervals.
    """
    P = np.asarray(per_input, dtype=float)
    s = np.asarray(sigmas, dtype=float)
    if P.ndim != 2 or P.shape[0] != s.shape[0]:
        raise ValueError(f"per_input must be (n_sigma={s.shape[0]}, n_inputs), got {P.shape}")
    n = P.shape[1]
    rng = np.random.default_rng(seed)
    bs, ks, failed = [], [], 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means = P[:, idx].mean(axis=1)
        try:
            f = fit_power_law(s, means, dim, lo=lo, hi=hi)
        except ValueError:
            failed += 1
            continue
        bs.append(f.b)
        ks.append(f.kappa)
    if len(bs) < 2:
        raise ValueError(f"only {len(bs)} of {n_resamples} resamples could be fitted")
    return BootstrapFit(n_resamples=n_resamples, n_inputs=n, seed=seed, alpha=alpha,
                        b_samples=np.array(bs), kappa_samples=np.array(ks), n_failed=failed)


def invert_sigma(rho, b: float, kappa: float, dim: int) -> np.ndarray:
    """``sigma_hat = (rho / (kappa * sqrt(2 dim / pi)))**(1/b)``. Pure algebra on
    the fitted curve; whether the answer means anything is decided by
    :func:`validate_inverse`, which checks ``rho`` against the fitted range."""
    r = np.asarray(rho, dtype=float)
    return (r / (kappa * sphere_prefactor(dim))) ** (1.0 / b)


def validate_inverse(sigmas: np.ndarray, rho: np.ndarray, fit: PowerLawFit, *,
                     bootstrap: BootstrapFit | None = None,
                     lo: float = FIT_LO, hi: float = FIT_HI) -> dict:
    """Invert the observed mean rate at every grid cell and compare to the
    sigma that produced it.

    The calibrated range is the observed rate interval over the fit window.
    A cell whose rate lies inside it is ``in_range`` and its relative error is
    a fair test of the inverse. Cells outside are reported with the same
    numbers, flagged, so the out-of-range behaviour is on record and is not
    read as validation.
    """
    s = np.asarray(sigmas, dtype=float)
    r = np.asarray(rho, dtype=float)
    used = np.array(fit.rho_used)
    rho_lo, rho_hi = float(used.min()), float(used.max())
    cells = []
    for sigma, rate in zip(s, r):
        if rate <= 0:
            cells.append({"sigma": float(sigma), "rho": 0.0, "sigma_hat": None,
                          "relative_error": None, "in_range": False,
                          "note": "zero rate: no inverse"})
            continue
        sigma_hat = float(invert_sigma(rate, fit.b, fit.kappa, fit.dim))
        cell = {
            "sigma": float(sigma), "rho": float(rate), "sigma_hat": sigma_hat,
            "relative_error": sigma_hat / float(sigma) - 1.0,
            "in_range": bool(rho_lo <= rate <= rho_hi),
        }
        if bootstrap is not None:
            draws = invert_sigma(rate, bootstrap.b_samples, bootstrap.kappa_samples, fit.dim)
            cell["sigma_hat_ci"] = list(bootstrap.interval(draws))
        cells.append(cell)
    in_range = [c for c in cells if c["in_range"]]
    out = [c for c in cells if not c["in_range"] and c["sigma_hat"] is not None]
    return {
        "calibrated_rho_range": [rho_lo, rho_hi],
        "calibrated_sigma_window": [lo, hi],
        "max_abs_relative_error_in_range": (
            max(abs(c["relative_error"]) for c in in_range) if in_range else None),
        "max_abs_relative_error_out_of_range": (
            max(abs(c["relative_error"]) for c in out) if out else None),
        "cells": cells,
    }
