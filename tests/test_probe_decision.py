"""The decision rule, applied to a sweep report.

These pin the parts that decide what the sweep selects: which conditions count,
what a threshold on a point estimate does and does not establish, and that a
comparison between two operators is paired on the documents they share.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
pytest.importorskip("semq")


@pytest.fixture(scope="module")
def decide():
    path = REPO / "experiments" / "probe-sweep" / "decide.py"
    spec = importlib.util.spec_from_file_location("_decide", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _report(cells, near_null=("proc",), non_null=(), scored=None):
    """A sweep report shaped the way run_sweep.describe_data writes one."""
    if scored is None:
        scored = sorted({c for cell in cells for c in cell["conditions"]})
    return {"data": {"near_null_conditions": list(near_null),
                     "non_null_interventions": list(non_null),
                     "non_null_scored": [c for c in non_null if c in scored],
                     "conditions": list(scored)},
            "cells": cells}


def _cell(conditions, n_bins=4, percentile=0.99):
    return {"operator": "semq_quant", "n_bins": n_bins, "percentile": percentile,
            "bits_per_coordinate": 3, "code_bytes": 144,
            "saturation_occupancy": 0.05, "conditions": conditions}


def test_a_missing_control_is_not_a_passed_control(decide, tmp_path, monkeypatch):
    """`all()` over an empty set is True, which would pass every cell."""
    # the controls the rule names are absent from the scored conditions
    report = _report([_cell({"int8": {"her": 0.0}})],
                     near_null=("proc", "threads1"))
    monkeypatch.setattr(decide, "load_per_document", lambda *a, **k: {})
    out = decide.evaluate(report, tmp_path / "absent.npz")
    cell = out["cells"][0]
    assert cell["control_codes_intact"] is False
    assert cell["missing_controls"] == ["proc", "threads1"]
    assert out["selection_under_any_intervention"] is None


def test_a_disturbed_control_still_rejects_the_cell(decide, tmp_path, monkeypatch):
    report = _report([_cell({"proc": {"her": 0.999}})])
    monkeypatch.setattr(decide, "load_per_document", lambda *a, **k: {})
    cell = decide.evaluate(report, tmp_path / "absent.npz")["cells"][0]
    assert cell["control_codes_intact"] is False
    assert cell["missing_controls"] == []


def test_two_absent_selections_are_not_two_readings_agreeing(decide, tmp_path,
                                                             monkeypatch):
    report = _report([_cell({"proc": {"her": 1.0}})])
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


def test_an_unscored_intervention_cannot_satisfy_the_all_reading(decide,
                                                                 tmp_path,
                                                                 monkeypatch):
    """int8 qualifying is not both interventions qualifying when bf16 is absent."""
    cell = _cell({"proc": {"her": 1.0}, "int8": {"her": 0.0}})
    report = _report([cell], near_null=("proc",), non_null=("bf16", "int8"),
                     scored=["proc", "int8"])
    monkeypatch.setattr(decide, "load_per_document", lambda *a, **k: {
        "multi_region": np.full(200, 40), "changed": np.full(200, 100),
        "unbounded_changed": np.full(200, 5)})

    out = decide.evaluate(report, tmp_path / "absent.npz")
    scored_cell = out["cells"][0]
    assert scored_cell["qualifying_interventions"] == ["int8"]
    assert scored_cell["qualifies_any"] is True
    assert scored_cell["missing_interventions"] == ["bf16"]
    assert scored_cell["qualifies_all"] is False          # not 1 of 1
    assert out["selection_under_all_interventions"] is None
    assert out["declared_but_unscored"]["non_null"] == ["bf16"]


def test_every_declared_intervention_present_and_qualifying_does_pass(decide,
                                                                     tmp_path,
                                                                     monkeypatch):
    """The guard must not reject a grid that genuinely measured everything."""
    cell = _cell({"proc": {"her": 1.0}, "bf16": {"her": 0.0},
                  "int8": {"her": 0.0}})
    report = _report([cell], near_null=("proc",), non_null=("bf16", "int8"))
    monkeypatch.setattr(decide, "load_per_document", lambda *a, **k: {
        "multi_region": np.full(200, 40), "changed": np.full(200, 100),
        "unbounded_changed": np.full(200, 5)})

    out = decide.evaluate(report, tmp_path / "absent.npz")
    assert out["cells"][0]["missing_interventions"] == []
    assert out["cells"][0]["qualifies_all"] is True
    assert out["selection_under_all_interventions"] is not None
    assert out["declared_but_unscored"] == {"near_null": [], "non_null": []}


def test_the_minimum_screens_the_point_estimate_not_the_interval(decide, tmp_path,
                                                                 monkeypatch):
    """Qualifying does not establish that the true share clears the minimum.

    A share whose point estimate is over 1% but whose interval reaches below it
    passes the screen and fails the strict reading. Conflating the two would
    report a development screen as a bound.
    """
    cell = _cell({"proc": {"her": 1.0}, "int8": {"her": 0.0}})
    report = _report([cell], near_null=("proc",), non_null=("int8",))
    # ~1.5% of changed coordinates, spread so the interval reaches under 1%
    rng = np.random.default_rng(0)
    changed = np.full(60, 100)
    multi = rng.choice([0, 1, 2, 6], size=60, p=[0.55, 0.2, 0.15, 0.10])
    monkeypatch.setattr(decide, "load_per_document", lambda *a, **k: {
        "multi_region": multi, "changed": changed,
        "unbounded_changed": np.full(60, 5)})

    got = decide.evaluate(report, tmp_path / "absent.npz")["cells"][0]
    share = got["interventions"]["int8"]["multi_region_share"]
    assert share["point"] >= decide.MIN_MULTI_REGION_SHARE
    assert share["ci95"][0] < decide.MIN_MULTI_REGION_SHARE
    assert got["interventions"]["int8"]["qualifies"] is True
    assert got["interventions"]["int8"]["qualifies_strict"] is False
    assert got["qualifies_any"] is True
    assert got["qualifies_any_strict"] is False


def test_the_rule_says_in_the_report_that_the_threshold_is_a_screen(decide,
                                                                    tmp_path,
                                                                    monkeypatch):
    report = _report([_cell({"proc": {"her": 1.0}})])
    monkeypatch.setattr(decide, "load_per_document", lambda *a, **k: {})
    out = decide.evaluate(report, tmp_path / "absent.npz")
    assert "point estimate" in out["rule"]["threshold_is_a_screen"]
    assert "selection_under_any_intervention_strict" in out
