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
