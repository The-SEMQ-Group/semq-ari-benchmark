# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0. The SDK is not covered by the Apache License.
"""The pre-registered analysis of the selectivity study (PREREGISTRATION.md).

Written and committed before any episode was captured. Reads the vectors that
capture_episode.py wrote and episodes.json, and writes results.json.

    python analyze.py --vectors DIR [--out results.json] [--scores episode_scores.json] [--allow-missing]

Two changes were made on 2026-10-14, after the results were published. Neither
changes a pre-registered output on the complete data:
- The calibration/test split is derived from the complete plan, not from the
  episodes that happen to be present, and a missing capture stops the analysis
  unless --allow-missing is given. With --allow-missing, missing episodes are
  dropped and the split is unchanged.
- Each rate also gets a cell-level Wilson interval, ``ci_cell_wilson``, and the
  claim is re-checked with it (``claim_supported_cell_wilson``). These are
  labelled as not pre-registered. The pre-registered cluster bootstrap returns
  a zero-width interval whenever every test cell has the same outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

SPLIT_SEED = 20261007
BOOT = 10_000


# ----- detectors at matched storage B = d/4 bytes (PREREGISTRATION.md, "Detectors") ------------
def det_code(ref, cur):
    from ari.metrics import coordinate_change
    from ari.semq_compat import encode_packed
    return coordinate_change(encode_packed(ref, 2), encode_packed(cur, 2), n_bins=2, dim=ref.shape[1])


def det_sha256(ref, cur):
    return np.array([hashlib.sha256(a.tobytes()).digest() != hashlib.sha256(b.tobytes()).digest()
                     for a, b in zip(ref, cur)], dtype=float)


def det_blocks(ref, cur):
    d = ref.shape[1]
    n_blocks = max(1, (d // 4) // 32)            # B/32 digests of 32 bytes
    edges = np.linspace(0, d, n_blocks + 1).astype(int)
    out = np.zeros(len(ref))
    for lo, hi in zip(edges, edges[1:]):
        out += [hashlib.sha256(a[lo:hi].tobytes()).digest() != hashlib.sha256(b[lo:hi].tobytes()).digest()
                for a, b in zip(ref, cur)]
    return out / n_blocks


def det_fp16_prefix(ref, cur):
    k = (ref.shape[1] // 4) // 2                 # B/2 coordinates at 2 bytes each
    return np.abs(ref[:, :k].astype(np.float16).astype(np.float32)
                  - cur[:, :k].astype(np.float16).astype(np.float32)).mean(axis=1)


def det_sketch(ref, cur):
    d = ref.shape[1]
    k = (d // 4) // 4                            # B/4 fp32 projections
    P = np.random.default_rng(d).standard_normal((d, k)).astype(np.float32) / np.sqrt(k)
    return np.linalg.norm(ref @ P - cur @ P, axis=1)


def det_cosine(ref, cur):                         # reference: full fp32, not storage-matched
    a, b = ref.astype(np.float64), cur.astype(np.float64)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return 1.0 - cos


DETECTORS = {"code": det_code, "sha256": det_sha256, "blocks": det_blocks,
             "fp16_prefix": det_fp16_prefix, "sketch": det_sketch, "cosine_full": det_cosine}
MATCHED = ["code", "sha256", "blocks", "fp16_prefix", "sketch"]


# ----- statistics -------------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def cluster_ci(flags_by_cell, rng, boot=BOOT):
    """95% CI of the episode alarm rate, resampling cells with replacement."""
    cells = list(flags_by_cell.values())
    if not cells:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(boot):
        pick = [cells[i] for i in rng.integers(0, len(cells), len(cells))]
        allf = np.concatenate(pick)
        stats.append(allf.mean())
    return tuple(float(x) for x in np.quantile(stats, [0.025, 0.975]))


def cell_wilson(by_cell, label):
    """Wilson interval over cells, the unit of inference (not pre-registered).

    Each cell counts once, scored against the detector: a benign cell alarms if
    any of its episodes alarmed, and a material cell is detected only if every
    episode was flagged.
    """
    if label == "benign":
        k = sum(bool(f.any()) for f in by_cell.values())
    else:
        k = sum(bool(f.all()) for f in by_cell.values())
    return k, wilson(k, len(by_cell))


def split_cells(cells_by_condition):
    """Half the cells of each condition to calibration, half to test (fixed seed)."""
    rng = np.random.default_rng(SPLIT_SEED)
    calib, test = set(), set()
    for cond in sorted(cells_by_condition):
        cells = sorted(cells_by_condition[cond])
        rng.shuffle(cells)
        h = len(cells) // 2
        calib.update(cells[:h]); test.update(cells[h:])
    return calib, test


def analyze(plan, vec_dir: Path, allow_missing: bool = False) -> dict:
    refs = {r["id"]: r for r in plan["references"]}
    loaded = {}

    def vec(i):
        if i not in loaded:
            loaded[i] = np.load(vec_dir / f"{i}.npy")
        return loaded[i]

    rows, missing, digests = [], [], {}
    for e in plan["episodes"]:
        if not (vec_dir / f"{e['id']}.npy").exists() or not (vec_dir / f"{e['reference']}.npy").exists():
            missing.append(e["id"])
    if missing and not allow_missing:
        raise SystemExit(f"{len(missing)} planned captures are missing (first: {missing[0]}); "
                         "pass --allow-missing to drop them without changing the split")
    for e in plan["episodes"]:
        if e["id"] in missing:
            continue
        ref, cur = vec(e["reference"]), vec(e["id"])
        cell = (e["model"], e["condition"])
        h = hashlib.sha256(cur.tobytes()).hexdigest()
        dup = h in digests.setdefault(cell, set()); digests[cell].add(h)
        per_input = {k: f(ref, cur) for k, f in DETECTORS.items()}
        rows.append({"id": e["id"], "cell": cell, "label": e["label"], "condition": e["condition"],
                     "duplicate": dup, "score": {k: float(v.mean()) for k, v in per_input.items()},
                     "per_input": per_input})
        loaded.pop(e["id"], None)

    # The split comes from the complete plan, so a missing capture cannot move a cell
    # between calibration and test.
    cells_by_condition = {}
    for e in plan["episodes"]:
        cells_by_condition.setdefault(e["condition"], set()).add((e["model"], e["condition"]))
    calib, test = split_cells(cells_by_condition)
    rng = np.random.default_rng(SPLIT_SEED + 1)

    out = {"n_episodes": len(rows), "missing": missing,
           "duplicates": sum(r["duplicate"] for r in rows),
           "n_cells": {"benign": len({r["cell"] for r in rows if r["label"] == "benign"}),
                       "material": len({r["cell"] for r in rows if r["label"] == "material"})},
           "detectors": {}}
    for k in DETECTORS:
        cb = [r for r in rows if r["cell"] in calib and r["label"] == "benign"]
        thr = max(r["score"][k] for r in cb)                                   # zero calibration benign alarms
        thr_in = max(float(r["per_input"][k].max()) for r in cb)               # input-level unit
        res = {"threshold": thr, "matched_storage": k in MATCHED}
        for label in ("material", "benign"):
            tr = [r for r in rows if r["cell"] in test and r["label"] == label]
            flags = np.array([r["score"][k] > thr for r in tr], dtype=float)
            by_cell = {}
            for r, f in zip(tr, flags):
                by_cell.setdefault(r["cell"], []).append(f)
            by_cell = {c: np.array(v) for c, v in by_cell.items()}
            inputs = np.concatenate([r["per_input"][k] > thr_in for r in tr]) if tr else np.array([])
            k_cells, ci_cells = cell_wilson(by_cell, label)
            res[label] = {"episodes": len(tr), "cells": len(by_cell), "rate": float(flags.mean()) if len(tr) else None,
                          "ci_cluster": cluster_ci(by_cell, rng), "ci_wilson": wilson(int(flags.sum()), len(tr)),
                          "input_rate": float(inputs.mean()) if inputs.size else None,
                          "cells_counted": k_cells, "ci_cell_wilson": ci_cells}
        res["material_by_condition"] = {}
        for cond in sorted({r["condition"] for r in rows if r["label"] == "material"}):
            tr = [r for r in rows if r["cell"] in test and r["condition"] == cond]
            res["material_by_condition"][cond] = float(np.mean([r["score"][k] > thr for r in tr])) if tr else None
        out["detectors"][k] = res

    c = out["detectors"]["code"]["material"]["ci_cluster"]
    out["claim_supported"] = all(c[0] > out["detectors"][h]["material"]["ci_cluster"][1] for h in ("sha256", "blocks"))
    c = out["detectors"]["code"]["material"]["ci_cell_wilson"]
    out["claim_supported_cell_wilson"] = all(
        c[0] > out["detectors"][h]["material"]["ci_cell_wilson"][1] for h in ("sha256", "blocks"))
    out["not_preregistered"] = ["cells_counted", "ci_cell_wilson", "claim_supported_cell_wilson"]
    out["episode_scores"] = [{"id": r["id"], "model": r["cell"][0], "condition": r["condition"],
                              "label": r["label"], "half": "calibration" if r["cell"] in calib else "test",
                              "duplicate": r["duplicate"], "score": r["score"]} for r in rows]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=HERE / "episodes.json")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "results.json")
    ap.add_argument("--scores", type=Path, default=HERE / "results" / "episode_scores.json")
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    res = analyze(json.loads(a.plan.read_text()), a.vectors, a.allow_missing)
    # Every episode's score per detector, so the thresholds and rates can be checked
    # without the vectors.
    scores = res.pop("episode_scores")
    a.scores.write_text("[\n" + ",\n".join(json.dumps(s) for s in scores) + "\n]\n")
    a.out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    for k, v in res["detectors"].items():
        m, b = v["material"], v["benign"]
        print(f"{k:12s} material {m['rate']} {m['ci_cluster']}  benign {b['rate']}  by-cond {v['material_by_condition']}")
    print("claim supported:", res["claim_supported"], "| episodes", res["n_episodes"], "missing", len(res["missing"]),
          "duplicates", res["duplicates"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
