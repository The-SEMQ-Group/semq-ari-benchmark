# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0. The SDK is not covered by the Apache License.
"""rho, the coordinate change rate, and ARI-R, the dimension-normalized score."""

import numpy as np
import pytest

pytest.importorskip("semq")

from ari import metrics, report, run  # noqa: E402
from ari.inputs import InputSet  # noqa: E402
from ari.semq_compat import encode_packed  # noqa: E402


def _unit(n, d, seed):
    X = np.random.default_rng(seed).standard_normal((n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_one_flipped_coordinate_is_one_over_d():
    X = _unit(4, 1024, 0)
    Y = X.copy()
    Y[2, 17] = -Y[2, 17]  # sign flip: same norm, one symbol changes
    m = metrics.aggregate(encode_packed(X, 2), encode_packed(Y, 2), dim=1024)
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
        out[d] = metrics.aggregate(encode_packed(X, 2), encode_packed(Y, 2), dim=d)
    rhos = [out[d].rho for d in out]
    assert max(rhos) / min(rhos) < 1.3, rhos
    assert out[384].HER > out[1024].HER > out[3072].HER


def test_report_carries_rho_and_ari_r():
    X = _unit(50, 384, 3)
    Y = X.copy()
    Y[0, 0] = -Y[0, 0]
    base, cur = encode_packed(X, 2), encode_packed(Y, 2)
    m = {c: metrics.aggregate(base, cur if c == "proc" else base, dim=384)
         for c in ("proc", "conc", "time")}
    rep = report.build_report(agent_id="t", input_set="t", environment={}, metrics_by_condition=m,
                              codes_by_condition={c: base for c in m}, fingerprint={"dim": 384})
    semq = rep["results_per_condition"]["proc"]["detectors"]["semq"]
    assert semq["coordinate_change_rate"] == pytest.approx((1 / 384) / 50)
    assert rep["ARI_R"] == pytest.approx(1 - semq["coordinate_change_rate"] / 3)


def test_padding_coordinates_do_not_enter_the_denominator():
    """385 coordinates at 2 bits pack into 97 bytes, which also hold 388."""
    X = _unit(1, 385, 4)
    Y = X.copy()
    Y[0, 384] = -Y[0, 384]
    base, cur = encode_packed(X, 2), encode_packed(Y, 2)
    assert base.shape[1] == 97
    assert metrics.aggregate(base, cur, dim=385).rho == pytest.approx(1 / 385)
    with pytest.raises(TypeError):
        metrics.aggregate(base, cur)
    with pytest.raises(ValueError):
        metrics.aggregate(base, cur, dim=384)  # packs into 96 bytes, not 97

    codes = {"same": base, "proc": cur, "conc": base, "time": base}
    rep = run.aggregate_report(agent_id="t", input_set=InputSet("t", ["a"], ["a"]),
                               environment={}, codes_by_condition=codes,
                               fingerprint={"dim": 385})
    semq = rep["results_per_condition"]["proc"]["detectors"]["semq"]
    assert semq["n_coordinates"] == 385
    assert semq["coordinate_change_rate"] == pytest.approx(1 / 385)


def test_report_rejects_rho_over_a_different_dimension():
    X = _unit(2, 388, 5)
    base = encode_packed(X, 2)
    m = {c: metrics.aggregate(base, base, dim=388) for c in ("proc", "conc", "time")}
    with pytest.raises(ValueError, match="fingerprint.dim"):
        report.build_report(agent_id="t", input_set="t", environment={}, metrics_by_condition=m,
                            codes_by_condition={c: base for c in m}, fingerprint={"dim": 385})
