# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""Tests for the Open-SWE-Traces ARI-E experiment scripts.

The scripts live under experiments/, so they are loaded by path. Nothing here
touches the network: the fetch script is tested on its path-to-cell mapping and
the run script on small synthetic outcome tables.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "harness-effect"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fetch = _load("fetch_outcomes")
run = _load("run")


@pytest.mark.parametrize("path, cell", [
    ("data/sweagent/qwen35_122b/swe-rebench-v2/train-00000-of-00009.parquet",
     ("sweagent", "qwen35_122b")),
    ("data/openhands/minimax_m25/swe-rebench-v2/train-00017-of-00018.parquet",
     ("openhands", "minimax_m25")),
    # Layout before 2026-08-21: one directory per cell, "qwen35" for the model.
    ("data/qwen35_openhands_trajectories/train-00000-of-00023.parquet",
     ("openhands", "qwen35_122b")),
    ("data/minimax_m25_sweagent_trajectories/train-00022-of-00023.parquet",
     ("sweagent", "minimax_m25")),
])
def test_cell_of_maps_both_layouts(path, cell):
    assert fetch.cell_of(path) == cell


@pytest.mark.parametrize("path", [
    "data/minisweagent/qwen36_27b/scale-swe/train-00000-of-00017.parquet",
    "data/openhands/qwen36_27b/scale-swe/train-00000-of-00013.parquet",
    "data/openhands/deepseek_v4_flash/scale-swe/train-00000-of-00009.parquet",
    "data/minisweagent/qwen38_27b/swe-rebench-v2/train-00000-of-00022.parquet",
    "openhands_tools.json",
    "README.md",
])
def test_cell_of_ignores_other_cells_and_non_data(path):
    """'minisweagent' must not match 'sweagent', and other models are skipped."""
    assert fetch.cell_of(path) is None


def _table(tmp_path, rows):
    """Write a gzip outcome table plus a manifest that matches it."""
    text = "harness,model,instance_id,trajectory_id,resolved,source_file\n"
    text += "".join(",".join(map(str, r)) + ",f.parquet\n" for r in rows)
    csv = tmp_path / "outcomes.abc.csv.gz"
    with gzip.GzipFile(csv, "wb", mtime=0) as gz:
        gz.write(text.encode())
    manifest = tmp_path / "outcomes.abc.manifest.json"
    manifest.write_text(json.dumps({
        "revision": "abc", "output": {"sha256": hashlib.sha256(csv.read_bytes()).hexdigest()},
    }))
    return csv, manifest


def test_load_outcomes_checks_the_manifest_digest(tmp_path):
    csv, manifest = _table(tmp_path, [("sweagent", "qwen35_122b", "c1", "t1", 1)])
    cells, _ = run.load_outcomes(csv, manifest)
    assert cells[("sweagent", "qwen35_122b")] == [
        {"instance_id": "c1", "trajectory_id": "t1", "resolved": 1}]

    manifest.write_text(json.dumps({"revision": "abc", "output": {"sha256": "0" * 64}}))
    with pytest.raises(SystemExit):
        run.load_outcomes(csv, manifest)


def test_eligibility_needs_two_graded_runs_on_both_sides():
    a = [{"instance_id": c, "trajectory_id": f"t{i}", "resolved": r}
         for c, i, r in [("c1", 0, 1), ("c1", 1, 0), ("c2", 0, 1), ("c2", 1, -1),
                         ("c3", 0, 1), ("c3", 1, 1)]]
    b = [{"instance_id": c, "trajectory_id": f"u{i}", "resolved": r}
         for c, i, r in [("c1", 0, 1), ("c1", 1, 1), ("c2", 0, 0), ("c2", 1, 0),
                         ("c3", 0, 1)]]
    # c2 has one ungraded run on side a; c3 has one run on side b.
    assert run.shared_cases(run.graded(a), run.graded(b)) == {"c1"}


def test_permutation_null_centers_on_zero_and_reports_shapes():
    """A large synthetic contrast with a real gap: the shuffled effect is near zero."""
    rng = np.random.default_rng(3)
    a, b = [], []
    for i in range(600):
        p = rng.uniform(0.2, 0.8)
        for j in range(3):
            a.append({"instance_id": f"c{i}", "trajectory_id": f"a{j}",
                      "resolved": int(rng.random() < p)})
        for j in range(2 + (i % 2)):
            b.append({"instance_id": f"c{i}", "trajectory_id": f"b{j}",
                      "resolved": int(rng.random() < p - 0.3)})
    run.N_PERMUTATIONS, run.NULL_RESAMPLES, run.N_RESAMPLES = 5, 200, 300
    out = run.contrast(a, b, "A", "B", "scaffold effect, model=qwen35_122b",
                       np.random.default_rng(0))
    assert out["n_cases"] == 600
    assert out["repeat_shapes"] == {"3/2": 300, "3/3": 300}
    assert out["equal_repeats"]["n_cases"] == 300
    assert out["effect"] > 0.05 and out["detected"]
    null = out["permutation_null"]
    assert abs(null["mean"]) < 0.02
    assert len(null["effects"]) == len(null["intervals"]) == 5


def test_tex_rows_follow_the_paper_table_format():
    rows = run.tex_rows([{
        "title": "scaffold effect, model=qwen35_122b", "n_cases": 10788,
        "self_consistency": {"openhands": 0.8745, "sweagent": 0.8727},
        "cross_agreement": 0.7846, "effect": 0.0890, "effect_ci": [0.0833, 0.0937],
    }, {"title": "model effect, scaffold=openhands", "n_cases": 0}])
    assert rows == ("scaffold, model = Qwen3.5-122B & 10{,}788 & 0.874 & 0.785 "
                    "& \\textbf{+0.089} & [+0.083, +0.094] \\\\\n")
