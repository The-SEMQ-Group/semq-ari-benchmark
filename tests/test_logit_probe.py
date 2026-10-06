# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0 and patent pending. The SDK is not covered by the
# Apache License.
"""The ARI-D v0.2 logit probe: the properties spec/arid-logit-probe-v0.2.md promises."""

import numpy as np
import pytest

pytest.importorskip("semq")

from ari.code_metrics import chunked_code_diff  # noqa: E402
from ari.logit_probe import (centre_rows, encode_logit_chunks,  # noqa: E402
                             encode_logits, logit_widths)


def _logits(n=4, vocab=1000, seed=0):
    # Multiples of 2^-8 within +-64: adding a small integer and centring stay exact.
    rng = np.random.default_rng(seed)
    return (np.round(rng.standard_normal((n, vocab)) * 4 * 256) / 256).astype(np.float32)


def test_the_layout_depends_on_the_vocabulary_alone():
    assert logit_widths(32000) == [32000]
    assert logit_widths(65536) == [65536]
    assert logit_widths(151936) == [65536, 65536, 20864]


def test_a_constant_shift_does_not_change_the_code():
    """Softmax ignores a constant offset, and so does the probe."""
    X = _logits()
    a, _ = encode_logits(X)
    b, _ = encode_logits(X + np.float32(3.0))
    assert np.array_equal(a, b)


def test_a_uniform_rescale_does_not_change_the_code():
    """Rows are renormalised per chunk, so a temperature-style power-of-two scale is invisible."""
    X = _logits()
    assert np.array_equal(encode_logits(X)[0], encode_logits(X * np.float32(2.0))[0])


def test_centring_uses_the_full_vocabulary_mean():
    X = _logits(vocab=70000)
    C = centre_rows(X)
    assert np.allclose(C.astype(np.float64).mean(axis=1), 0.0, atol=1e-6)


def test_chunked_codes_split_back_at_the_same_boundaries():
    X = _logits(n=3, vocab=70001)
    Y = X.copy()
    Y[1, 70000] += np.float32(40.0)  # a large change in the last chunk only
    a, widths = encode_logits(X)
    b, _ = encode_logits(Y)
    d = chunked_code_diff(a, b, n_bins=8, widths=widths)
    assert widths == [65536, 4465]
    assert d.codes_equal.tolist() == [True, False, True]
    assert d.n_coordinates == 70001


def test_joined_and_unjoined_encodings_agree():
    X = _logits(n=2, vocab=70000)
    joined, widths = encode_logits(X)
    parts = encode_logit_chunks(X)
    assert [w for w, _ in parts] == widths
    assert np.array_equal(joined, np.concatenate([p for _, p in parts], axis=1))
