# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0. The SDK is not covered by the Apache License.
"""The selectivity analysis: a missing capture cannot move a cell between halves.

The split comes from the complete plan. A missing capture stops the analysis, and with
--allow-missing the missing episodes are dropped while every other episode keeps its half.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("semq")

_PATH = Path(__file__).resolve().parents[1] / "experiments/selectivity-confirmatory/analyze.py"
_spec = importlib.util.spec_from_file_location("selectivity_analyze", _PATH)
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)


def _plan_and_vectors(tmp_path):
    rng = np.random.default_rng(0)
    refs, eps = [], []
    for m in ("m1", "m2", "m3", "m4"):
        refs.append({"id": f"ref-{m}", "model": m})
        for cond, label in (("proc", "benign"), ("bf16", "material")):
            for j in range(2):
                eps.append({"id": f"{cond}-{m}-{j}", "model": m, "condition": cond, "label": label,
                            "reference": f"ref-{m}"})
    for e in refs + eps:
        x = rng.standard_normal((5, 64)).astype(np.float32)
        np.save(tmp_path / f"{e['id']}.npy", x / np.linalg.norm(x, axis=1, keepdims=True))
    return {"references": refs, "episodes": eps}


def _halves(res):
    return {s["id"]: s["half"] for s in res["episode_scores"]}


def test_missing_capture_stops_the_analysis(tmp_path):
    plan = _plan_and_vectors(tmp_path)
    (tmp_path / "proc-m1-0.npy").unlink()
    with pytest.raises(SystemExit, match="missing"):
        analyze.analyze(plan, tmp_path)


def test_allow_missing_keeps_the_split(tmp_path):
    plan = _plan_and_vectors(tmp_path)
    full = _halves(analyze.analyze(plan, tmp_path))
    # Remove a whole cell: under the old rule the remaining cells would be re-split.
    for j in range(2):
        (tmp_path / f"proc-m1-{j}.npy").unlink()
    res = analyze.analyze(plan, tmp_path, allow_missing=True)
    assert sorted(res["missing"]) == ["proc-m1-0", "proc-m1-1"]
    assert _halves(res) == {k: v for k, v in full.items() if not k.startswith("proc-m1-")}


def test_cell_wilson_counts_cells_against_the_detector():
    by_cell = {"a": np.array([1.0, 1.0]), "b": np.array([1.0, 0.0]), "c": np.array([0.0, 0.0])}
    assert analyze.cell_wilson(by_cell, "material")[0] == 1   # only cells flagged throughout
    assert analyze.cell_wilson(by_cell, "benign")[0] == 2     # any alarm counts
    k, (lo, hi) = analyze.cell_wilson({c: np.zeros(3) for c in range(30)}, "benign")
    assert k == 0 and lo == 0.0 and hi == pytest.approx(0.1135, abs=1e-3)


def _capture_module():
    path = Path(__file__).resolve().parents[1] / "experiments/selectivity-confirmatory/capture_episode.py"
    spec = importlib.util.spec_from_file_location("selectivity_capture", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_capture_records_model_and_tokenizer_revisions():
    from types import SimpleNamespace as NS

    cap = _capture_module()
    a, b = "a" * 40, "b" * 40
    snap = "/root/.cache/huggingface/hub/models--org--m/snapshots/{}"

    def model(model_path, tok_path):
        return [NS(auto_model=NS(name_or_path=model_path, config=NS()),
                   tokenizer=NS(name_or_path=tok_path, init_kwargs={}))]

    assert cap._revisions(model(snap.format(a), snap.format(a))) == {
        "model_revision": a, "tokenizer_revision": a}
    assert cap._revisions(model("org/m", "org/m")) == {
        "model_revision": None, "tokenizer_revision": None}
    with pytest.raises(RuntimeError, match="differ"):
        cap._revisions(model(snap.format(a), snap.format(b)))
