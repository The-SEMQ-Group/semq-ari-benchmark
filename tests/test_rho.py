# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0 and patent pending. The SDK is not covered by the
# Apache License.
"""rho, the coordinate change rate, and ARI-R, the dimension-normalized score."""

import numpy as np
import pytest

pytest.importorskip("semq")

from ari import metrics, report  # noqa: E402
from ari.semq_compat import encode_packed  # noqa: E402


def _unit(n, d, seed):
    X = np.random.default_rng(seed).standard_normal((n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_one_flipped_coordinate_is_one_over_d():
    X = _unit(4, 1024, 0)
    Y = X.copy()
    Y[2, 17] = -Y[2, 17]  # sign flip: same norm, one symbol changes
    m = metrics.aggregate(encode_packed(X, 2), encode_packed(Y, 2))
    assert m.HER == pytest.approx(0.75)
    assert m.rho == pytest.approx((1 / 1024) / 4)


@pytest.mark.parametrize("seed", [0, 1])
def test_rho_does_not_depend_on_dimension_but_her_does(seed):
    """Equal relative noise on unit vectors: HER falls with d, rho stays put."""
    eps = 3e-3
    out = {}
    for d in (384, 1024, 3072):
        X = _unit(400, d, seed)
        Y = X * (1 + eps * np.random.default_rng(seed + 100).standard_normal(X.shape))
        out[d] = metrics.aggregate(encode_packed(X, 2), encode_packed(Y, 2))
    rhos = [out[d].rho for d in out]
    assert max(rhos) / min(rhos) < 1.3, rhos
    assert out[384].HER > out[1024].HER > out[3072].HER


def test_report_carries_rho_and_ari_r():
    X = _unit(50, 384, 3)
    Y = X.copy()
    Y[0, 0] = -Y[0, 0]
    base, cur = encode_packed(X, 2), encode_packed(Y, 2)
    m = {c: metrics.aggregate(base, cur if c == "proc" else base) for c in ("proc", "conc", "time")}
    rep = report.build_report(agent_id="t", input_set="t", environment={}, metrics_by_condition=m,
                              codes_by_condition={c: base for c in m}, fingerprint={"dim": 384})
    semq = rep["results_per_condition"]["proc"]["detectors"]["semq"]
    assert semq["coordinate_change_rate"] == pytest.approx((1 / 384) / 50)
    assert rep["ARI_R"] == pytest.approx(1 - semq["coordinate_change_rate"] / 3)
