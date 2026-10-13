# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0 and patent pending. The SDK is not covered by the
# Apache License.
"""EXPLORATORY, not pre-registered: SimHash as a matched-storage detector.

Added 2026-10-13, after both selectivity studies were analyzed, because SimHash (sign bits of
random projections) is the closest discrete alternative to the ARI-R code and neither study
tested it. Same rules as the pre-registered analyses:

- storage matched at B = d/4 bytes, so 2d sign bits from 2d Gaussian projections (fixed seed);
- per-input statistic: fraction of bits that flipped; episode score: its mean;
- threshold: the largest score on the first study's calibration benign cells (its own split);
- tested on the first study's test half and on every cell of the second study.

    python exploratory_simhash.py --vectors1 DIR --vectors2 DIR
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


study1 = _load("study1", HERE.parent / "selectivity-confirmatory" / "analyze.py")
study2 = _load("study2", HERE / "analyze.py")
_P: dict[int, np.ndarray] = {}


def det_simhash(ref, cur):
    d = ref.shape[1]
    if d not in _P:   # seed differs from the float sketch's (rng(d)), so the two are independent
        _P[d] = np.random.default_rng(10_000 + d).standard_normal((d, 2 * d)).astype(np.float32)
    P = _P[d]
    return ((ref @ P) > 0).astype(np.int8).__ne__((cur @ P) > 0).mean(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors1", type=Path, required=True)
    ap.add_argument("--vectors2", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "exploratory_simhash.json")
    a = ap.parse_args()

    # Study 1: its own split and threshold rule, with SimHash added next to the code.
    study1.DETECTORS = {**study1.DETECTORS, "simhash": det_simhash}
    r1 = study1.analyze(json.loads((HERE.parent / "selectivity-confirmatory" / "episodes.json").read_text()), a.vectors1)
    thr = r1["detectors"]["simhash"]["threshold"]

    # Study 2: every cell is a test cell; thresholds fixed (the code's from the frozen analysis).
    plan2 = json.loads((HERE / "episodes.json").read_text())
    dropped = {"Qwen/Qwen3-Embedding-0.6B"}  # dropped by the pre-registered rule
    cells, cache = [], {}
    for e in plan2["episodes"]:
        if e["model"] in dropped:
            continue
        if e["reference"] not in cache:
            cache.clear(); cache[e["reference"]] = np.load(a.vectors2 / f"{e['reference']}.npy")
        ref, cur = cache[e["reference"]], np.load(a.vectors2 / f"{e['id']}.npy")
        s_code = float(study1.det_code(ref, cur).mean())
        s_sim = float(det_simhash(ref, cur).mean())
        cells.append({"id": e["id"], "model": e["model"], "condition": e["condition"], "label": e["label"],
                      "code": s_code > study2.THRESHOLDS["code"], "simhash": s_sim > thr, "simhash_score": s_sim})
    benign = [c for c in cells if c["label"] == "benign"]
    material = [c for c in cells if c["label"] == "material"]
    b = sum(c["simhash"] and not c["code"] for c in benign)
    c_ = sum(c["code"] and not c["simhash"] for c in benign)
    out = {
        "status": "exploratory, not pre-registered",
        "simhash_threshold": thr,
        "study1_test": {k: {lab: r1["detectors"][k][lab]["rate"] for lab in ("material", "benign")}
                        for k in r1["detectors"]},
        "study1_thresholds_reproduce": {k: r1["detectors"][k]["threshold"] == study2.THRESHOLDS[k] for k in study2.THRESHOLDS},
        "study2": {
            "benign_cells": len(benign), "material_cells": len(material),
            "code_benign_alarms": sum(c["code"] for c in benign), "simhash_benign_alarms": sum(c["simhash"] for c in benign),
            "code_material_caught": sum(c["code"] for c in material), "simhash_material_caught": sum(c["simhash"] for c in material),
            "only_simhash": b, "only_code": c_, "p_one_sided_code_fewer": study2.sign_test_one_sided(b, c_),
            "simhash_benign_alarm_cells": [c["id"] for c in benign if c["simhash"]],
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
