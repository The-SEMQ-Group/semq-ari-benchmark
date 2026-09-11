"""The SDK is the authority on layout and calibration, not this repo.

Each test here pins a fact that a hand-rolled replica got wrong or that a
docstring asserted without checking.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("semq")

from ari.code_metrics import bits_per_coordinate, unpack_symbols  # noqa: E402
from ari.semq_compat import quant_context  # noqa: E402


def _codes(X, dim, bins, **kw):
    with quant_context(dim, n_bins=bins, **kw) as ctx:
        return np.asarray(ctx.batch_encode(np.ascontiguousarray(X, np.float32)))


def test_local_unpacking_agrees_with_the_core():
    """Symbols pack low-order-first. An inline big-endian reader disagrees."""
    rng = np.random.default_rng(0)
    dim, bins = 64, 8
    X = rng.standard_normal((8, dim)).astype(np.float32)
    with quant_context(dim, n_bins=bins, scale_max=3.0) as ctx:
        packed = np.asarray(ctx.batch_encode(X))
        assert np.array_equal(ctx.unpack_codes(packed, dim),
                              unpack_symbols(packed, dim=dim, n_bins=bins))


def test_symbol_order_within_a_byte_is_not_arbitrary():
    """The wrong order keeps the aggregate rate and corrupts every index.

    This is why it survived unnoticed: the error permutes coordinates the same
    way in both operands, so counts match and locations do not.
    """
    rng = np.random.default_rng(1)
    dim, bins = 64, 8
    R = rng.standard_normal((32, dim)).astype(np.float32)
    C = R + rng.standard_normal((32, dim)).astype(np.float32) * 0.05
    a, b = _codes(R, dim, bins, scale_max=3.0), _codes(C, dim, bins, scale_max=3.0)

    w = bits_per_coordinate(bins)
    good_a = unpack_symbols(a, dim=dim, n_bins=bins)
    good_b = unpack_symbols(b, dim=dim, n_bins=bins)
    swap = lambda p: (np.unpackbits(p, axis=1).reshape(len(p), -1, w)
                      * (1 << np.arange(w)[::-1])).sum(2).astype(np.uint8)

    assert np.allclose((good_a != good_b).mean(axis=1),
                       (swap(a) != swap(b)).mean(axis=1))
    assert not np.array_equal(np.flatnonzero(good_a[0] != good_b[0]),
                              np.flatnonzero(swap(a)[0] != swap(b)[0]))


def test_core_calibration_is_not_interchangeable_with_numpy_percentile():
    """semq_compat used to claim these give bit-identical codes. They do not."""
    rng = np.random.default_rng(7)
    dim, bins, pct = 384, 8, 0.999
    X = rng.standard_normal((200, dim)).astype(np.float32)

    with quant_context(dim, n_bins=bins) as ctx:
        core = float(ctx.calibrate(X, percentile=pct))
        a = np.asarray(ctx.batch_encode(X))
    hand = float(np.percentile(np.abs(X), pct * 100.0))
    b = _codes(X, dim, bins, scale_max=hand)

    assert core != hand
    assert not np.array_equal(a, b), "if this passes, relax the docstring again"
    assert (a != b).sum() < a.size * 0.01, "divergence should stay at the margin"


def test_compare_codes_separates_byte_rate_from_coordinate_rate():
    """The published score counts packed bytes; the baselines count symbols."""
    rng = np.random.default_rng(3)
    dim, bins = 128, 8
    R = rng.standard_normal((16, dim)).astype(np.float32)
    C = R + rng.standard_normal((16, dim)).astype(np.float32) * 0.1
    with quant_context(dim, n_bins=bins, scale_max=3.0) as ctx:
        cmp = ctx.compare_codes(np.asarray(ctx.batch_encode(R)),
                                np.asarray(ctx.batch_encode(C)), dim)
    # two coordinates share a byte at 4 bits, so a byte rate cannot be smaller
    assert (cmp.byte_change_rate >= cmp.coordinate_change_rate - 1e-12).all()
    assert (cmp.bit_hamming_rate <= cmp.coordinate_change_rate + 1e-12).all()
