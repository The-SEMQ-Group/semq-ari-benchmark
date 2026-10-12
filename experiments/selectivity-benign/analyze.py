# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0 and patent pending. The SDK is not covered by the
# Apache License.
"""The pre-registered analysis of the benign-alarm study (PREREGISTRATION.md).

Written and committed before any capture. Detectors are imported unchanged from the
first study's frozen analysis; thresholds are the first study's, fixed below.

    python analyze.py --vectors DIR [--out results.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("study1", HERE.parent / "selectivity-confirmatory" / "analyze.py")
study1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study1)
DETECTORS, wilson = study1.DETECTORS, study1.wilson

# Thresholds set in the first study (results/results.json, commit 76f407f), not refit here.
THRESHOLDS = {"code": 5.208333333333333e-06, "sha256": 1.0, "blocks": 1.0,
              "fp16_prefix": 4.5940744541894674e-08, "sketch": 1.959520798988491e-06,
              "cosine_full": 2.074719396283342e-12}
PRIMARY = ["sketch", "fp16_prefix"]      # matched-storage continuous detectors
SECONDARY = ["cosine_full"]               # full fp32, 16x the storage
ALPHA = 0.05


def sign_test_one_sided(b: int, c: int) -> float:
    """P(X >= b) for X ~ Binomial(b + c, 1/2): b = cells where only the other detector alarmed,
    c = cells where only the code alarmed. Small p means the code alarms on fewer cells."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(b, n + 1)) / 2 ** n


def holm(pvals: dict[str, float]) -> dict[str, float]:
    order = sorted(pvals, key=pvals.get)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvals[k]))
        adj[k] = running
    return adj


def analyze(plan, vec_dir: Path) -> dict:
    loaded = {}

    def vec(i):
        if i not in loaded:
            loaded[i] = np.load(vec_dir / f"{i}.npy")
        return loaded[i]

    cells, missing = [], []
    for e in plan["episodes"]:
        if not (vec_dir / f"{e['id']}.npy").exists() or not (vec_dir / f"{e['reference']}.npy").exists():
            missing.append(e["id"]); continue
        ref, cur = vec(e["reference"]), vec(e["id"])
        scores = {k: float(f(ref, cur).mean()) for k, f in DETECTORS.items()}
        cells.append({"id": e["id"], "model": e["model"], "condition": e["condition"], "label": e["label"],
                      "score": scores, "alarm": {k: scores[k] > THRESHOLDS[k] for k in DETECTORS},
                      "nonfinite": int((~np.isfinite(cur)).any(axis=1).sum())})
        loaded.pop(e["id"], None)

    # A model with any missing cell is dropped whole, so every detector sees the same cells.
    dropped = sorted({e["model"] for e in plan["episodes"] if e["id"] in missing})
    cells = [c for c in cells if c["model"] not in dropped]
    out = {"n_cells": len(cells), "missing": missing, "dropped_models": dropped,
           "thresholds": THRESHOLDS, "detectors": {}, "paired": {}}

    for label in ("benign", "material"):
        cs = [c for c in cells if c["label"] == label]
        for k in DETECTORS:
            a = sum(c["alarm"][k] for c in cs)
            d = out["detectors"].setdefault(k, {})
            d[label] = {"cells": len(cs), "alarms": a, "rate": a / len(cs) if cs else None,
                        "wilson95": wilson(a, len(cs))}
            d[f"{label}_by_condition"] = {
                cond: sum(c["alarm"][k] for c in cs if c["condition"] == cond)
                for cond in sorted({c["condition"] for c in cs})}
            d[f"{label}_alarm_cells"] = [c["id"] for c in cs if c["alarm"][k]]

    benign = [c for c in cells if c["label"] == "benign"]
    raw = {}
    for k in PRIMARY + SECONDARY:
        b = sum(c["alarm"][k] and not c["alarm"]["code"] for c in benign)
        c_ = sum(c["alarm"]["code"] and not c["alarm"][k] for c in benign)
        raw[k] = sign_test_one_sided(b, c_)
        out["paired"][k] = {"only_other_alarmed": b, "only_code_alarmed": c_, "p_one_sided": raw[k]}
    adj = holm({k: raw[k] for k in PRIMARY})
    for k in PRIMARY:
        out["paired"][k]["p_holm"] = adj[k]
        out["paired"][k]["code_fewer_benign_alarms"] = adj[k] < ALPHA
    out["primary_supported"] = {k: out["paired"][k]["code_fewer_benign_alarms"] for k in PRIMARY}
    cb = out["detectors"]["code"]["benign"]
    out["code_benign_upper_below_5pct"] = cb["wilson95"][1] < 0.05
    out["nonfinite_cells"] = [c["id"] for c in cells if c["nonfinite"]]
    out["cells"] = [{k: v for k, v in c.items()} for c in cells]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=HERE / "episodes.json")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "results.json")
    a = ap.parse_args()
    res = analyze(json.loads(a.plan.read_text()), a.vectors)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    for k, v in res["detectors"].items():
        print(f"{k:12s} benign {v['benign']['alarms']}/{v['benign']['cells']}  material {v['material']['alarms']}/{v['material']['cells']}")
    for k, v in res["paired"].items():
        print(f"paired code vs {k}: only {k} {v['only_other_alarmed']}, only code {v['only_code_alarmed']}, "
              f"p {v['p_one_sided']:.4g}" + (f", Holm {v['p_holm']:.4g}" if "p_holm" in v else ""))
    print("primary supported:", res["primary_supported"], "| code benign upper < 5%:", res["code_benign_upper_below_5pct"],
          "| dropped:", res["dropped_models"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
