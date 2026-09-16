# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library that is subject to a
# commercial license owned by The SEMQ Group Inc. and is patent pending.
# The SDK is not covered by the Apache License.
"""Write the packed SEMQ codes behind regime_matrix.json to results/codes/.

Encoding needs the SEMQ SDK. Reading the codes back does not: every rate in
the results table can be recomputed from these files with ari.code_metrics,
which is plain numpy. The codes are the measurement; the SDK is only needed
to make new ones.

    python experiments/regime-discrimination/export_codes.py
    python experiments/regime-discrimination/export_codes.py --check   # no SDK
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

from ari.code_metrics import bits_per_coordinate, code_diff  # noqa: E402

CACHE = HERE / "results" / "cache"
CODES = HERE / "results" / "codes"
MATRIX = HERE / "results" / "regime_matrix.json"
REFERENCE = "reference"
CONDITIONS = ("proc", "threads1", "batch8", "batch128", "bf16", "int8")
QUANT_BINS = 8
CALIBRATION_PERCENTILE = 0.99


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _docs(name: str) -> np.ndarray:
    return np.ascontiguousarray(np.load(CACHE / f"{name}.npz")["docs"], np.float32)


def export() -> None:
    from ari.semq_compat import quant_context, sdk_version

    ref = _docs(REFERENCE)
    dim = ref.shape[1]
    ctx = quant_context(dim, n_bins=QUANT_BINS)
    try:
        # Calibrated once on the reference and frozen, as run_matrix.py does.
        scale = float(ctx.calibrate(ref, percentile=CALIBRATION_PERCENTILE))
        CODES.mkdir(parents=True, exist_ok=True)
        files = {}
        for name in (REFERENCE,) + CONDITIONS:
            codes = np.asarray(ctx.batch_encode(_docs(name)), dtype=np.uint8)
            out = CODES / f"{name}.npz"
            np.savez_compressed(out, codes=codes)
            files[name] = {"sha256": _sha256(out), "shape": list(codes.shape),
                           "source_sha256": _sha256(CACHE / f"{name}.npz")}
    finally:
        ctx.close()

    manifest = {
        "operator": "SEMQ QUANT",
        "n_bins": QUANT_BINS,
        "bits_per_coordinate": bits_per_coordinate(QUANT_BINS),
        "dim": dim,
        "calibration_percentile": CALIBRATION_PERCENTILE,
        "calibrated_on": REFERENCE,
        "scale_max": scale,
        "layout": ("one row per document; coordinates packed contiguously from "
                   "the least significant bit of byte 0, final byte zero-padded; "
                   "decode with ari.code_metrics.unpack_symbols"),
        "semq": sdk_version(),
        "files": files,
    }
    (CODES / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(files)} code files to {CODES}")


def check() -> int:
    """Recompute HER from the committed codes and compare with regime_matrix.json."""
    manifest = json.loads((CODES / "MANIFEST.json").read_text())
    rows = {r["condition"]: r for r in json.loads(MATRIX.read_text())["rows"]
            if r.get("host", "cpu") == "cpu"}
    ref = np.load(CODES / f"{REFERENCE}.npz")["codes"]
    bad = 0
    for name, entry in manifest["files"].items():
        path = CODES / f"{name}.npz"
        if _sha256(path) != entry["sha256"]:
            print(f"{name}: sha256 mismatch"); bad += 1
            continue
        if name == REFERENCE:
            continue
        cur = np.load(path)["codes"]
        diff = code_diff(ref, cur, n_bins=manifest["n_bins"], dim=manifest["dim"])
        her = float(diff.codes_equal.mean())
        want = rows[name]["semq_her"]
        ok = abs(her - want) < 5e-5
        bad += not ok
        print(f"{name:<10} HER {her:.4f}  table {want:.4f}  "
              f"coordinate change rate {diff.coordinate_change_rate.mean():.6f}  "
              f"{'ok' if ok else 'MISMATCH'}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify committed codes against regime_matrix.json")
    a = ap.parse_args()
    raise SystemExit(check() if a.check else export())
