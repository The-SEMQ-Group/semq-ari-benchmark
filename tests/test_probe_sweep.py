"""The probe sweep's arithmetic and its development split.

The sweep's conclusions live in RESULTS.md; these pin the parts that
would silently change a conclusion — the split rule, the byte
accounting, and the two measures the decision rests on.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SWEEP = REPO / "experiments" / "probe-sweep" / "run_sweep.py"
pytest.importorskip("semq")


@pytest.fixture(scope="module")
def sweep():
    spec = importlib.util.spec_from_file_location("_sweep", SWEEP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_development_split_is_deterministic_and_disjoint(sweep):
    dev, reserved = sweep.split_development(5183)
    assert dev.size + reserved.size == 5183
    assert not set(dev.tolist()) & set(reserved.tolist())
    assert sweep.split_development(5183)[0].tolist() == dev.tolist()
    assert 0.55 < dev.size / 5183 < 0.65


def test_storage_is_not_proportional_to_bin_count(sweep):
    from ari.code_metrics import bits_per_coordinate
    import math
    got = [math.ceil(384 * bits_per_coordinate(b) / 8) for b in (2, 4, 8)]
    assert got == [96, 144, 192]


def test_gini_is_zero_when_changes_spread_evenly(sweep):
    assert sweep._gini(np.zeros(6)) == 0.0
    assert sweep._gini(np.full(6, 10)) == pytest.approx(0.0, abs=1e-12)
    concentrated = sweep._gini(np.array([0, 0, 0, 0, 0, 60]))
    assert concentrated > 0.7


def test_per_document_arrays_leave_the_json_and_survive_the_round_trip(sweep, tmp_path):
    """618k of the report's 618k lines were one integer per line. They moved."""
    report = {"cells": [
        {"operator": "semq_quant", "conditions": {
            "int8": {"her": 0.0, "per_document": {
                "changed": [3, 1, 4], "multi_region": [1, 0, 2],
                "unbounded_changed": [0, 1, 1]}}}},
    ]}
    sidecar = tmp_path / "s.per-document.npz"
    sweep.split_per_document(report, sidecar)

    cond = report["cells"][0]["conditions"]["int8"]
    assert "per_document" not in cond
    assert cond["per_document_fields"] == ["changed", "multi_region",
                                           "unbounded_changed"]
    assert report["per_document_sidecar"] == sidecar.name
    import json
    assert len(json.dumps(report).splitlines()) == 1

    back = sweep.load_per_document(sidecar, 0, "int8")
    assert back["changed"].tolist() == [3, 1, 4]
    assert back["multi_region"].tolist() == [1, 0, 2]
    assert back["unbounded_changed"].tolist() == [0, 1, 1]


@pytest.fixture(scope="module")
def decide():
    path = REPO / "experiments" / "probe-sweep" / "decide.py"
    spec = importlib.util.spec_from_file_location("_decide", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cell(conditions, n_bins=4, percentile=0.99):
    return {"operator": "semq_quant", "n_bins": n_bins, "percentile": percentile,
            "bits_per_coordinate": 3, "code_bytes": 144,
            "saturation_occupancy": 0.05, "conditions": conditions}


def test_a_missing_control_is_not_a_passed_control(decide, tmp_path, monkeypatch):
    """`all()` over an empty set is True, which would pass every cell."""
    report = {
        "data": {"near_null_conditions": ["proc", "threads1"],
                 "non_null_interventions": []},
        # the controls the rule names are absent from the scored conditions
        "cells": [_cell({"int8": {"her": 0.0}})],
    }
    monkeypatch.setattr(decide, "load_per_document", lambda *a, **k: {})
    out = decide.evaluate(report, tmp_path / "absent.npz")
    cell = out["cells"][0]
    assert cell["control_codes_intact"] is False
    assert cell["missing_controls"] == ["proc", "threads1"]
    assert out["selection_under_any_intervention"] is None


def test_a_disturbed_control_still_rejects_the_cell(decide, tmp_path, monkeypatch):
    report = {
        "data": {"near_null_conditions": ["proc"], "non_null_interventions": []},
        "cells": [_cell({"proc": {"her": 0.999}})],
    }
    monkeypatch.setattr(decide, "load_per_document", lambda *a, **k: {})
    cell = decide.evaluate(report, tmp_path / "absent.npz")["cells"][0]
    assert cell["control_codes_intact"] is False
    assert cell["missing_controls"] == []


def test_two_absent_selections_are_not_two_readings_agreeing(decide, tmp_path,
                                                             monkeypatch):
    report = {
        "data": {"near_null_conditions": ["proc"], "non_null_interventions": []},
        "cells": [_cell({"proc": {"her": 1.0}})],
    }
    monkeypatch.setattr(decide, "load_per_document", lambda *a, **k: {})
    out = decide.evaluate(report, tmp_path / "absent.npz")
    assert out["selection_under_any_intervention"] is None
    assert out["selection_under_all_interventions"] is None
    assert out["readings_agree"] is None       # not True


def test_the_operator_comparison_is_paired_on_documents(decide):
    """An unpaired interval carries corpus variation that cancels when paired."""
    rng = np.random.default_rng(0)
    n = 400
    changed = rng.integers(50, 400, size=n)
    # one shared per-document effect, plus a fixed 2-point advantage for a
    shared = rng.random(n) * 0.4
    a_num = (changed * (shared + 0.02)).astype(int)
    b_num = (changed * shared).astype(int)

    paired = decide.paired_difference(a_num, changed, b_num, changed)
    assert paired["point"] > 0
    assert paired["ci95"][0] > 0              # the shared variation cancels
    assert paired["n_documents"] == n

    # the same data compared without pairing: wider, and it straddles zero
    rng2 = np.random.default_rng(decide.SEED)
    idx_a = rng2.integers(0, n, size=(decide.N_BOOTSTRAP, n))
    idx_b = rng2.integers(0, n, size=(decide.N_BOOTSTRAP, n))
    unpaired = (np.asarray(a_num, float)[idx_a].sum(1) / changed[idx_a].sum(1)
                - np.asarray(b_num, float)[idx_b].sum(1) / changed[idx_b].sum(1))
    lo, hi = np.quantile(unpaired, [0.025, 0.975])
    assert (hi - lo) > (paired["ci95"][1] - paired["ci95"][0])


def test_a_paired_comparison_refuses_mismatched_documents(decide):
    with pytest.raises(ValueError, match="same documents"):
        decide.paired_difference([1, 2], [3, 4], [1], [3])
