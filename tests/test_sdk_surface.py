# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0 and patent pending. The SDK is not covered by the
# Apache License.
"""The SDK is the authority on layout and range, not this repo.

Each test here pins a fact that a hand-rolled replica got wrong or that a
docstring asserted without checking.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("semq")

from ari.code_metrics import bits_per_coordinate, unpack_symbols  # noqa: E402
from ari import semq_compat  # noqa: E402
from ari.probe import fixed_scale_codes, load_probe  # noqa: E402
from ari.semq_compat import encode_packed, unpack  # noqa: E402


def _codes(X, dim, bins):
    return encode_packed(X, bins)


def test_local_unpacking_agrees_with_the_core():
    """Symbols pack low-order-first. An inline big-endian reader disagrees."""
    rng = np.random.default_rng(0)
    dim, bins = 64, 8
    X = rng.standard_normal((8, dim)).astype(np.float32)
    packed = encode_packed(X, bins)
    assert np.array_equal(unpack(packed, dim, bins),
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
    a, b = _codes(R, dim, bins), _codes(C, dim, bins)

    w = bits_per_coordinate(bins)
    good_a = unpack_symbols(a, dim=dim, n_bins=bins)
    good_b = unpack_symbols(b, dim=dim, n_bins=bins)
    def swap(p):
        """Read each byte's symbols high-order first, as the old code did."""
        return (np.unpackbits(p, axis=1).reshape(len(p), -1, w)
                * (1 << np.arange(w)[::-1])).sum(2).astype(np.uint8)

    assert np.allclose((good_a != good_b).mean(axis=1),
                       (swap(a) != swap(b)).mean(axis=1))
    assert not np.array_equal(np.flatnonzero(good_a[0] != good_b[0]),
                              np.flatnonzero(swap(a)[0] != swap(b)[0]))


def test_the_range_is_fixed_by_the_dimension():
    """v0.2 has no calibration: s is 2/sqrt(dim) in float32, whatever the data."""
    for dim in (17, 384, 1024, 4096):
        assert semq_compat.max_magnitude(dim) == float(np.float32(2.0 / np.sqrt(dim)))
    rng = np.random.default_rng(5)
    assert load_probe(rng.standard_normal((4, 384))).s == load_probe(np.ones((9, 384))).s


def test_row_scale_does_not_change_the_code():
    """Rows are renormalised before encoding, so a provider's norm convention is moot."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((32, 384))
    scales = rng.uniform(0.1, 50.0, size=(32, 1))
    assert np.array_equal(encode_packed(X, 2), encode_packed(X * scales, 2))


def test_a_zero_row_is_rejected_by_index():
    X = np.ones((3, 8))
    X[1] = 0.0
    with pytest.raises(ValueError, match="row 1 is all zeros"):
        encode_packed(X, 2)


def test_a_v01_calibrated_baseline_scale_is_refused():
    """A stored v0.1 scale would compare calibrated codes with fixed-range codes."""
    rng = np.random.default_rng(13)
    X = rng.standard_normal((4, 384))
    s = semq_compat.max_magnitude(384)
    assert np.array_equal(fixed_scale_codes(X, s, 384), load_probe(X).encode(X))
    with pytest.raises(ValueError, match="calibrated by the v0.1 probe"):
        fixed_scale_codes(X, 0.1318197101354599, 384)
