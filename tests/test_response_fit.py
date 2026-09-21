# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library that is subject to a
# commercial license owned by The SEMQ Group Inc. and is patent pending.
# The SDK is not covered by the Apache License.
"""Power-law fit, bootstrap and inverse on synthetic rates.

The fit is plain numpy, so these run without the SDK. The last test needs
the SDK and sentence-transformers and skips without them.
"""
import numpy as np
import pytest

from ari.response_fit import (FIT_HI, FIT_LO, SIGMAS, bootstrap_power_law,
                              fit_power_law, fit_window_mask, invert_sigma,
                              sphere_prefactor, validate_inverse)

DIM = 384


def _synthetic(b=0.97, kappa=2.5, n_inputs=200, seed=3, saturate=True):
    """Per-input rates whose means follow kappa*sqrt(2d/pi)*sigma^b, with
    multiplicative noise per input and a bend toward 1 at the top of the grid."""
    rng = np.random.default_rng(seed)
    mean = kappa * sphere_prefactor(DIM) * SIGMAS ** b
    if saturate:
        mean = 1 - np.exp(-mean)
    scale = rng.lognormal(0.0, 0.3, size=n_inputs)
    return mean[:, None] * scale[None, :] / scale.mean()


def test_fit_window_is_inclusive_of_the_logspace_endpoints():
    mask = fit_window_mask(SIGMAS)
    assert mask.sum() == 5
    assert SIGMAS[mask][0] == pytest.approx(FIT_LO)
    assert SIGMAS[mask][-1] == pytest.approx(FIT_HI)


def test_fit_recovers_the_generating_parameters():
    P = _synthetic(b=0.97, kappa=2.5, saturate=False)
    fit = fit_power_law(SIGMAS, P.mean(axis=1), DIM)
    assert fit.b == pytest.approx(0.97, abs=1e-6)
    assert fit.kappa == pytest.approx(2.5, rel=1e-6)
    assert fit.r2_log == pytest.approx(1.0, abs=1e-9)
    assert fit.n_cells_dropped_zero == 0
    assert len(fit.sigmas_used) == 5
    assert max(abs(r) for r in fit.residuals_log) < 1e-9


def test_saturation_outside_the_window_does_not_move_the_fit():
    P = _synthetic(b=0.97, kappa=2.5, saturate=True)
    fit = fit_power_law(SIGMAS, P.mean(axis=1), DIM)
    # 1 - exp(-x) = x - x^2/2 + ...; at the top of the window x ~ 0.04, so the
    # bend contributes about 2% to that cell and less below it.
    assert fit.b == pytest.approx(0.97, abs=0.02)
    assert fit.kappa == pytest.approx(2.5, rel=0.05)


def test_zero_cells_are_dropped_and_counted():
    rho = 2.5 * sphere_prefactor(DIM) * SIGMAS
    first_window_cell = np.flatnonzero(fit_window_mask(SIGMAS))[0]
    rho[first_window_cell] = 0.0
    fit = fit_power_law(SIGMAS, rho, DIM)
    assert fit.n_cells_dropped_zero == 1
    assert len(fit.sigmas_used) == 4
    assert fit.b == pytest.approx(1.0)


def test_fit_refuses_a_window_with_one_positive_cell():
    rho = np.zeros_like(SIGMAS)
    rho[-1] = 0.5
    rho[np.flatnonzero(fit_window_mask(SIGMAS))[0]] = 1e-3
    with pytest.raises(ValueError, match="at least two positive cells"):
        fit_power_law(SIGMAS, rho, DIM)


def test_bootstrap_intervals_cover_the_truth_and_are_deterministic():
    P = _synthetic(b=0.97, kappa=2.5, saturate=False)
    a = bootstrap_power_law(SIGMAS, P, DIM, n_resamples=200, seed=7)
    b = bootstrap_power_law(SIGMAS, P, DIM, n_resamples=200, seed=7)
    np.testing.assert_array_equal(a.b_samples, b.b_samples)
    d = a.to_dict()
    assert d["n_failed"] == 0
    assert d["b_ci"][0] <= 0.97 <= d["b_ci"][1]
    assert d["kappa_ci"][0] <= 2.5 <= d["kappa_ci"][1]
    # Per-input noise is multiplicative and shared across cells, so it moves
    # kappa but leaves the slope alone.
    assert d["b_se"] < 1e-9
    assert d["kappa_se"] > 0


def test_bootstrap_rejects_a_misshapen_matrix():
    with pytest.raises(ValueError, match="per_input must be"):
        bootstrap_power_law(SIGMAS, np.ones((3, 10)), DIM, n_resamples=5)


def test_inverse_is_the_algebraic_inverse_of_predict():
    P = _synthetic(b=0.95, kappa=2.0, saturate=False)
    fit = fit_power_law(SIGMAS, P.mean(axis=1), DIM)
    sigma_hat = invert_sigma(fit.predict(SIGMAS), fit.b, fit.kappa, DIM)
    np.testing.assert_allclose(sigma_hat, SIGMAS, rtol=1e-9)


def test_validate_inverse_flags_range_and_reports_out_of_range_error():
    P = _synthetic(b=0.97, kappa=2.5, saturate=True)
    means = P.mean(axis=1)
    fit = fit_power_law(SIGMAS, means, DIM)
    boot = bootstrap_power_law(SIGMAS, P, DIM, n_resamples=50, seed=2)
    v = validate_inverse(SIGMAS, means, fit, bootstrap=boot)
    cells = v["cells"]
    assert len(cells) == len(SIGMAS)
    in_range = [c for c in cells if c["in_range"]]
    assert [c["sigma"] for c in in_range] == pytest.approx(list(SIGMAS[fit_window_mask(SIGMAS)]))
    assert v["max_abs_relative_error_in_range"] < 0.05
    # The saturating top cell is out of range and inverts to a sigma well
    # below the one that produced it; the flag says not to read it.
    top = cells[-1]
    assert not top["in_range"]
    assert top["relative_error"] < -0.05
    assert v["max_abs_relative_error_out_of_range"] >= abs(top["relative_error"])
    assert all("sigma_hat_ci" in c for c in cells if c["sigma_hat"] is not None)
    assert v["calibrated_rho_range"] == [min(fit.rho_used), max(fit.rho_used)]


def test_validate_inverse_reports_a_zero_cell_without_inverting():
    means = 2.5 * sphere_prefactor(DIM) * SIGMAS
    means[0] = 0.0
    fit = fit_power_law(SIGMAS, means, DIM)
    v = validate_inverse(SIGMAS, means, fit)
    assert v["cells"][0]["sigma_hat"] is None
    assert v["cells"][0]["in_range"] is False


def test_registry_fit_basis_names_the_unit_of_every_fitted_row():
    import csv
    from pathlib import Path
    rows = list(csv.DictReader(
        (Path(__file__).resolve().parents[1] / "spec" / "fingerprints-v0.1.csv").open()))
    assert len(rows) == 15
    for r in rows:
        if r["b"]:
            assert r["kappa"] and r["fit_basis"] == "byte_change_rate", r
        else:
            assert not r["kappa"] and r["fit_basis"] == "", r


def test_sweep_rates_keep_their_denominators():
    """End to end on the SDK probe with random vectors: byte rate is four
    times the coordinate rate while changes are sparse, and each changed
    coordinate at n_bins=2 flips one of its two bits."""
    pytest.importorskip("semq")
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "experiments" / "drift-sensitivity" / \
        "refit_coordinate.py"
    spec = importlib.util.spec_from_file_location("refit_coordinate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, DIM)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    per_input, floor, facts = mod.sweep(X, SIGMAS, seed=0)
    assert facts["dim"] == DIM and facts["n_bins"] == 2
    assert facts["bits_per_coordinate"] == 2
    coord = per_input["coordinate_change_rate"].mean(axis=1)
    byte = per_input["byte_change_rate"].mean(axis=1)
    bit = per_input["bit_hamming_rate"].mean(axis=1)
    sparse = coord < 0.01
    assert sparse.any()
    np.testing.assert_allclose(byte[sparse], 4 * coord[sparse], rtol=0.05)
    np.testing.assert_allclose(bit, 0.5 * coord, rtol=1e-9)
    assert facts["floor_probe"]["coordinate_change_rate"] <= coord[0] + 1e-12
