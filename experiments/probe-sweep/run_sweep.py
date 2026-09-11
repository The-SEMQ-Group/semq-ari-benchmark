"""Bin count and calibration sweep, on development data only.

Scores the grid frozen in PREREGISTRATION.md over the cached SciFact
embeddings and their real serving conditions, and writes one
machine-readable record per cell.

The question is resolution against storage and saturation. Two magnitude
bins give four regions with both outermost ones unbounded, and adjacent
regions touch, so a positive lower bound on movement needs a move of two
or more regions. Whether a larger bin count produces such moves under
real serving drift is what decides the configuration.

  python run_sweep.py --out results/sweep.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from ari.bound_quality import summarize_bounds  # noqa: E402
from ari.code_metrics import bits_per_coordinate, code_diff  # noqa: E402
from ari.semq_compat import quant_context, sdk_version  # noqa: E402

DEFAULT_CACHE = REPO / "experiments" / "regime-discrimination" / "results" / "cache"
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
REFERENCE = "reference"
CONDITIONS = ["proc", "threads1", "batch8", "batch128", "bf16", "int8"]
# Declared roles. NEAR_NULL conditions hold the nominal configuration fixed and
# must leave the codes intact; a probe that fires on them is reporting noise.
# NON_NULL are the declared interventions a configuration has to resolve.
NEAR_NULL = ("proc", "threads1", "batch8", "batch128")
NON_NULL = ("bf16", "int8")

N_BINS_GRID = (2, 4, 8)
PERCENTILE_GRID = (0.99, 0.995, 0.999)
GROUP_SIZE = 64
DEV_MODULUS, DEV_KEEP = 5, 3          # doc_index % 5 < 3 is development


def load_docs(cache: Path, name: str) -> np.ndarray:
    return np.ascontiguousarray(
        np.load(cache / f"{name}.npz", allow_pickle=True)["docs"], np.float32)


def split_development(n: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    dev = idx[idx % DEV_MODULUS < DEV_KEEP]
    return dev, idx[idx % DEV_MODULUS >= DEV_KEEP]


def quant_cell(ref: np.ndarray, cur_by_cond: dict[str, np.ndarray],
               n_bins: int, percentile: float) -> dict:
    """Score one (n_bins, percentile) cell over every condition."""
    dim = ref.shape[1]
    ctx = quant_context(dim, n_bins=n_bins)
    try:
        scale = float(ctx.calibrate(ref, percentile=percentile))
        regions = ctx.quant_regions()
        t0 = time.perf_counter()
        ref_codes = np.asarray(ctx.batch_encode(ref))
        encode_seconds = time.perf_counter() - t0
        ref_sym = ctx.unpack_codes(ref_codes, dim)

        ranks = regions.ranks(ref_sym)
        top = regions.n_symbols - 1
        saturation = float(((ranks == 0) | (ranks == top)).mean())
        # The nominal outermost threshold from the calibration, against the
        # boundary the encoder actually uses. edges() is ascending over all
        # n_symbols + 1 edges, so the start of the outermost positive region
        # is at n_symbols - 1.
        nominal_edge = scale * (1.0 - 1.0 / n_bins)
        actual_edge = float(regions.edges()[regions.n_symbols - 1])

        conditions = {}
        for name, cur in cur_by_cond.items():
            cur_codes = np.asarray(ctx.batch_encode(cur))
            cur_sym = ctx.unpack_codes(cur_codes, dim)
            d = regions.displacement(ref_sym, cur_sym)
            diff = code_diff(ref_codes, cur_codes, n_bins=n_bins, dim=dim)
            q = summarize_bounds(d, over_changed_only=True)
            moved = d.regions_moved
            changed = int((moved > 0).sum())
            multi = int((moved >= 2).sum())
            counts = np.zeros(-(-dim // GROUP_SIZE), dtype=np.int64)
            for row in diff.changed_coordinates:
                if row.size:
                    np.add.at(counts, row // GROUP_SIZE, 1)
            # Per document, because coordinates inside one document are not
            # independent draws. Every rate below is pooled; the interval that
            # goes with it is resampled over documents in decide.py.
            per_doc_changed = (moved > 0).sum(axis=1).astype(int).tolist()
            per_doc_multi = (moved >= 2).sum(axis=1).astype(int).tolist()
            per_doc_unbounded = (
                np.isinf(d.max_abs_movement) & (moved > 0)
            ).sum(axis=1).astype(int).tolist()

            conditions[name] = {
                "her": float(diff.codes_equal.mean()),
                "coordinate_change_rate": float(diff.coordinate_change_rate.mean()),
                "n_changed": changed,
                "n_multi_region": multi,
                "per_document": {
                    "changed": per_doc_changed,
                    "multi_region": per_doc_multi,
                    "unbounded_changed": per_doc_unbounded,
                },
                # the precondition for any positive lower bound
                "multi_region_share_of_changed": (multi / changed) if changed else None,
                "n_lower_positive": q.n_lower_positive,
                "unbounded_share_of_changed": (
                    (q.n_changed - q.n_upper_finite_changed) / q.n_changed
                    if q.n_changed else None),
                "finite_width_quantiles": q.width_quantiles,
                # how concentrated the changes are across coordinate blocks
                "localization_gini": _gini(counts),
            }
    finally:
        ctx.close()

    bits = bits_per_coordinate(n_bins)
    return {
        "operator": "semq_quant",
        "n_bins": n_bins,
        "percentile": percentile,
        "bits_per_coordinate": bits,
        "scale_max": scale,
        "code_bytes": math.ceil(dim * bits / 8),
        "calibration_metadata_bytes": 4,        # one float32 scale
        "saturation_occupancy": saturation,
        "nominal_outermost_edge": nominal_edge,
        "encoder_outermost_edge": actual_edge,
        "encode_seconds": encode_seconds,
        "conditions": conditions,
    }


def uniform_cell(ref: np.ndarray, cur_by_cond: dict[str, np.ndarray],
                 bits: int, percentile: float) -> dict:
    """Ordinary uniform scalar quantization, same calibration opportunity."""
    dim = ref.shape[1]
    lo = float(np.percentile(ref, (1.0 - percentile) * 100.0))
    hi = float(np.percentile(ref, percentile * 100.0))
    levels = (1 << bits) - 1
    step = (hi - lo) / levels if hi > lo else 1.0

    def encode(x):
        return np.clip(np.round((x - lo) / step), 0, levels).astype(np.int32)

    t0 = time.perf_counter()
    ref_q = encode(ref)
    encode_seconds = time.perf_counter() - t0
    saturation = float(((ref_q == 0) | (ref_q == levels)).mean())

    conditions = {}
    for name, cur in cur_by_cond.items():
        cur_q = encode(cur)
        moved = np.abs(cur_q - ref_q)
        changed = int((moved > 0).sum())
        multi = int((moved >= 2).sum())
        conditions[name] = {
            "her": float((cur_q == ref_q).all(axis=1).mean()),
            "coordinate_change_rate": float((moved > 0).mean()),
            "n_changed": changed,
            "n_multi_region": multi,
            "multi_region_share_of_changed": (multi / changed) if changed else None,
            # Same documents as the QUANT cells, so a comparison between the
            # two operators can be paired rather than treated as independent.
            "per_document": {
                "changed": (moved > 0).sum(axis=1).astype(int).tolist(),
                "multi_region": (moved >= 2).sum(axis=1).astype(int).tolist(),
            },
        }
    return {
        "operator": "uniform_scalar",
        "n_bins": None,
        "percentile": percentile,
        "bits_per_coordinate": bits,
        "range": [lo, hi],
        "code_bytes": math.ceil(dim * bits / 8),
        "calibration_metadata_bytes": 8,        # two float32 endpoints
        "saturation_occupancy": saturation,
        "encode_seconds": encode_seconds,
        "conditions": conditions,
    }


def _gini(counts: np.ndarray) -> float:
    """Concentration of changes across coordinate blocks; 0 even, ->1 clumped."""
    c = np.sort(np.asarray(counts, dtype=np.float64))
    total = c.sum()
    if total <= 0:
        return 0.0
    n = c.size
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(c) / (n * total))


PER_DOCUMENT_FIELDS = ("changed", "multi_region", "unbounded_changed")


def per_document_key(cell_index: int, condition: str, field: str) -> str:
    return f"{cell_index}|{condition}|{field}"


def split_per_document(report: dict, sidecar: Path) -> None:
    """Move the per-document count arrays out of the report into an npz.

    One integer per document per condition per cell renders as one JSON line
    each under indent=2, which put 618k lines of data behind a 250-line result.
    The arrays are still needed -- decide.py bootstraps over them -- so they go
    to a compressed sidecar and the JSON keeps only what a reader reads.
    """
    arrays = {}
    for i, cell in enumerate(report["cells"]):
        for name, cond in cell["conditions"].items():
            pd = cond.pop("per_document", None)
            if not pd:
                continue
            cond["per_document_fields"] = sorted(pd)
            for field, values in pd.items():
                arrays[per_document_key(i, name, field)] = np.asarray(
                    values, dtype=np.int32)
    report["per_document_sidecar"] = sidecar.name
    np.savez_compressed(sidecar, **arrays)


def load_per_document(sidecar: Path, cell_index: int, condition: str) -> dict:
    """Counterpart of split_per_document, for decide.py."""
    with np.load(sidecar) as archive:
        return {f: archive[per_document_key(cell_index, condition, f)]
                for f in PER_DOCUMENT_FIELDS
                if per_document_key(cell_index, condition, f) in archive}


def describe_data(cache: Path, encoder: str, available: list[str], dim: int,
                  n_total: int, n_dev: int, n_reserved: int) -> dict:
    """The data block, including which declared roles were actually scored.

    The declared roles are recorded as declared. Narrowing them to whatever
    happened to be on disk is what would let a missing intervention read as a
    satisfied one: a rule of the form "every declared intervention qualifies"
    is trivially true over a list the reporter already pruned.
    """
    return {
        "source": _relative_source(cache),
        "encoder": encoder,
        "dim": dim,
        "n_documents_total": n_total,
        "n_development": n_dev,
        "n_reserved": n_reserved,
        "split_rule": f"doc_index % {DEV_MODULUS} < {DEV_KEEP}",
        "reserved_are_independent_episodes": False,
        "conditions": available,
        "conditions_missing": sorted(set(CONDITIONS) - set(available)),
        # declared, not discovered
        "near_null_conditions": list(NEAR_NULL),
        "non_null_interventions": list(NON_NULL),
        # discovered, so a reader can see the gap without recomputing it
        "near_null_scored": [c for c in NEAR_NULL if c in available],
        "non_null_scored": [c for c in NON_NULL if c in available],
        "uncertainty_unit": "document",
        "documents_are_independent_episodes": False,
    }


def _relative_source(cache: Path) -> str:
    """Repo-relative where possible; a cache outside the repo keeps its path."""
    try:
        return str(cache.relative_to(REPO))
    except ValueError:
        return str(cache)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(DEFAULT_CACHE),
                    help="directory of <condition>.npz files")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER,
                    help="label recorded with the results")
    ap.add_argument("--out", default=str(HERE / "results" / "sweep.json"))
    a = ap.parse_args()
    cache = Path(a.cache).resolve()

    available = [c for c in CONDITIONS if (cache / f"{c}.npz").exists()]
    ref_all = load_docs(cache, REFERENCE)
    dev, reserved = split_development(ref_all.shape[0])
    ref = np.ascontiguousarray(ref_all[dev])
    cur_by_cond = {c: np.ascontiguousarray(load_docs(cache, c)[dev])
                   for c in available}

    cells = []
    for n_bins in N_BINS_GRID:
        for pct in PERCENTILE_GRID:
            cells.append(quant_cell(ref, cur_by_cond, n_bins, pct))
    for n_bins in N_BINS_GRID:
        cells.append(uniform_cell(ref, cur_by_cond,
                                  bits_per_coordinate(n_bins), 0.99))

    report = {
        "preregistration": "PREREGISTRATION.md",
        "data": describe_data(cache, a.encoder, available, int(ref.shape[1]),
                              int(ref_all.shape[0]), int(dev.size),
                              int(reserved.size)),
        "environment": {
            "semq": sdk_version(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "cells": cells,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar = out.with_suffix(".per-document.npz")
    split_per_document(report, sidecar)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out} and {sidecar.name}  "
          f"({len(cells)} cells, {dev.size} development documents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
