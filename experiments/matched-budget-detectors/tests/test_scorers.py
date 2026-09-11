"""Tests for the detector scores.

The KL/JS cases exist because Appendix I claims those statistics are unusable
at small perturbations, on the evidence of one float32 implementation returning
a small negative value. These check the float64 log-space versions against an
exact oracle instead, at the perturbation sizes where the claim was made.
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scorers import (  # noqa: E402
    budget_ari,
    budget_margin_fp32,
    budget_projection,
    canonical_bytes,
    coord_mismatch,
    cosine_distance,
    js_div,
    kl_div,
    margin_delta,
    max_abs_diff,
    random_projection_scorer,
    rel_l2,
    sha256_rows,
    token_flip,
    topk_set_change,
)


# ---------------------------------------------------------------- canonical


def test_negative_zero_hashes_as_zero():
    """-0.0 == 0.0, so a hash that separated them would disagree with equality."""
    a = np.array([[0.0, 1.0]], dtype=np.float32)
    b = np.array([[-0.0, 1.0]], dtype=np.float32)
    assert canonical_bytes(a) == canonical_bytes(b)
    assert sha256_rows(a) == sha256_rows(b)


def test_nonfinite_is_rejected_not_hashed():
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            canonical_bytes(np.array([[bad, 1.0]], dtype=np.float32))


def test_hash_detects_single_ulp_change():
    a = np.ones((1, 8), dtype=np.float32)
    b = a.copy()
    b[0, 3] = np.nextafter(np.float32(1.0), np.float32(2.0))
    assert sha256_rows(a) != sha256_rows(b)


# ---------------------------------------------------------------- distances


def test_identical_vectors_score_zero_everywhere():
    rng = np.random.default_rng(0)
    r = rng.standard_normal((5, 32)).astype(np.float32)
    for fn in (max_abs_diff, rel_l2, coord_mismatch, cosine_distance):
        assert np.allclose(fn(r, r.copy()), 0.0), fn.__name__


def test_rel_l2_zero_norm_reference_falls_back_to_absolute():
    r = np.zeros((1, 4), dtype=np.float32)
    c = np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
    assert np.isclose(rel_l2(r, c)[0], 5.0)


def test_cosine_zero_norm_pairs():
    z = np.zeros((1, 4), dtype=np.float32)
    nz = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    assert cosine_distance(z, z.copy())[0] == 0.0   # both empty: no change
    assert cosine_distance(z, nz)[0] == 1.0         # undefined direction


def test_cosine_is_scale_invariant_but_max_abs_is_not():
    """The two answer different questions; a scaling shows which."""
    rng = np.random.default_rng(1)
    r = rng.standard_normal((3, 16)).astype(np.float32)
    c = (r * 2.0).astype(np.float32)
    assert np.allclose(cosine_distance(r, c), 0.0, atol=1e-6)
    assert (max_abs_diff(r, c) > 0).all()


# ---------------------------------------------------------------- logits


def _exact_kl(r_row, c_row) -> float:
    """KL in nats computed in exact rational arithmetic, then rounded once.

    Uses Fraction for the probabilities so the only floating-point operation is
    the final log, which makes this an independent oracle rather than a second
    implementation of the same float pipeline.
    """
    def probs(x):
        m = max(x)
        ex = [Fraction(math.exp(float(v - m))) for v in x]
        s = sum(ex)
        return [e / s for e in ex]

    p, q = probs(r_row), probs(c_row)
    return float(sum(float(pi) * math.log(float(pi) / float(qi)) for pi, qi in zip(p, q) if pi > 0))


def test_kl_matches_exact_oracle_at_tiny_perturbation():
    """The size at which Appendix I reports KL as unusable."""
    rng = np.random.default_rng(3)
    r = rng.standard_normal(64) * 5.0
    c = r + rng.standard_normal(64) * 1e-4
    got = float(kl_div(r[None, :], c[None, :])[0])
    want = _exact_kl(r, c)
    assert got >= 0.0, f"negative KL: {got}"
    assert abs(got - want) <= 1e-12 + 1e-6 * abs(want), (got, want)


def test_kl_is_zero_and_nonnegative_on_identical_logits():
    rng = np.random.default_rng(4)
    r = rng.standard_normal((8, 128)) * 3.0
    k = kl_div(r, r.copy())
    assert (k >= 0).all()
    assert np.allclose(k, 0.0, atol=1e-15)


def test_kl_survives_extreme_logits():
    """Large magnitudes are where a naive implementation overflows."""
    r = np.array([[800.0, -800.0, 0.0, 50.0]])
    c = r + 1e-3
    k = kl_div(r, c)
    assert np.isfinite(k).all() and (k >= 0).all()


def test_js_is_symmetric_bounded_and_nonnegative():
    rng = np.random.default_rng(5)
    a = rng.standard_normal((6, 64)) * 2.0
    b = a + rng.standard_normal((6, 64)) * 1e-3
    j1, j2 = js_div(a, b), js_div(b, a)
    assert np.allclose(j1, j2, rtol=1e-12)
    assert (j1 >= 0).all()
    assert (j1 <= math.log(2) + 1e-12).all()


def test_margin_and_token_flip_disagree_when_ranking_is_stable():
    """A margin can move a lot while the argmax does not, and that is the point."""
    r = np.array([[5.0, 1.0, 0.0]])
    c = np.array([[9.0, 1.0, 0.0]])
    assert token_flip(r, c)[0] == 0.0
    assert margin_delta(r, c)[0] == pytest.approx(4.0)


def test_topk_change_is_zero_when_topk_preserved():
    rng = np.random.default_rng(6)
    r = rng.standard_normal((1, 200)) * 3.0
    order = np.argsort(-r[0])
    c = r.copy()
    # perturb only outside the top-20, then verify the set really is preserved
    c[0, order[40:]] -= 5.0
    assert topk_set_change(r, c, k=20)[0] == 0.0
    assert set(np.argsort(-r[0])[:20]) == set(np.argsort(-c[0])[:20])


# ---------------------------------------------------------------- sketch


def test_projection_is_deterministic_for_a_seed():
    rng = np.random.default_rng(7)
    r = rng.standard_normal((4, 128)).astype(np.float32)
    c = r + 0.01
    s1 = random_projection_scorer(128, k=32, seed=11)
    s2 = random_projection_scorer(128, k=32, seed=11)
    assert np.allclose(s1(r, c), s2(r, c))
    s3 = random_projection_scorer(128, k=32, seed=12)
    assert not np.allclose(s1(r, c), s3(r, c))


# ---------------------------------------------------------------- budgets


def test_budgets_round_partial_bytes_up_and_charge_metadata():
    b = budget_ari(dim=385, bits_per_dim=2)      # 385*2/8 = 96.25 -> 97
    assert b.per_example == 97
    assert b.shared == 4                          # the scale is not free
    assert budget_margin_fp32().per_example == 4
    p = budget_projection(k=64)
    assert p.per_example == 256 and p.shared == 12


def test_amortized_budget_falls_with_n_but_never_below_per_example():
    b = budget_ari(dim=1024, bits_per_dim=4)
    assert b.total_at(1) > b.total_at(1000) > b.per_example
