"""Apply the configuration rule to a sweep, with intervals, in code.

The first version of this study carried its rule as prose and applied it
by eye. That is how a rule reading "a non-zero share" gets satisfied by
66 coordinates out of 246,058. The rule lives here now, so it is
evaluated rather than interpreted, and so a reader can disagree with the
constants rather than with a paragraph.

Three things the prose left undefined, fixed here:

**Which conditions count.** ``NEAR_NULL`` conditions hold the nominal
configuration fixed; a configuration that disturbs their codes is
rejected outright, whatever else it buys. ``NON_NULL`` are the declared
interventions a configuration has to resolve. The sweep records both
lists; this module uses them.

**How large a gain counts.** ``MIN_MULTI_REGION_SHARE`` is the smallest
share of changed coordinates that must move two or more regions before
the resolution is called real, since a multi-region move is the
precondition for any positive lower bound. A point estimate above it is
not enough: the lower end of the interval has to clear zero.

**What the uncertainty is over.** Coordinates inside one document are
not independent draws, so every interval here resamples **documents**.
Documents are not episodes: they come from one collection run and share
its process, machine and session, so these intervals cover sampling of
the corpus and nothing about run-to-run variation. An episode-level
interval needs the fresh collection in SEM-49.

The rule still has to say what to do when the declared interventions
disagree, and on this data they do. Both readings are evaluated and
reported rather than one being chosen after the fact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_sweep import load_per_document  # noqa: E402

MIN_MULTI_REGION_SHARE = 0.01     # 1% of changed coordinates
N_BOOTSTRAP = 2000
SEED = 47


def bootstrap_share(numer: list[int], denom: list[int], *, seed: int = SEED,
                    n: int = N_BOOTSTRAP) -> dict:
    """Pooled ratio with a document-resampled percentile interval.

    The statistic is ``sum(numer) / sum(denom)`` over documents, so a
    document contributes in proportion to how many changed coordinates
    it has, and the resampling carries that weighting with it.
    """
    a = np.asarray(numer, dtype=np.float64)
    b = np.asarray(denom, dtype=np.float64)
    total = b.sum()
    if total <= 0:
        return {"point": None, "ci95": [None, None], "n_documents": int(a.size)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n, a.size))
    sums_b = b[idx].sum(axis=1)
    draws = np.divide(a[idx].sum(axis=1), sums_b,
                      out=np.zeros(n), where=sums_b > 0)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"point": float(a.sum() / total), "ci95": [float(lo), float(hi)],
            "n_documents": int(a.size)}


def evaluate(report: dict, sidecar: Path) -> dict:
    """Score every cell against the rule and return the decision."""
    near_null = report["data"]["near_null_conditions"]
    non_null = report["data"]["non_null_interventions"]
    quant = [(i, c) for i, c in enumerate(report["cells"])
             if c["operator"] == "semq_quant"]

    scored = []
    for cell_index, cell in quant:
        conds = cell["conditions"]
        # A configuration that disturbs a near-null condition is rejected
        # whatever resolution it buys.
        control_clean = all(conds[c]["her"] == 1.0 for c in near_null if c in conds)

        per_intervention = {}
        for name in non_null:
            if name not in conds:
                continue
            pd = load_per_document(sidecar, cell_index, name)
            share = bootstrap_share(pd["multi_region"], pd["changed"])
            unbounded = bootstrap_share(pd["unbounded_changed"], pd["changed"])
            qualifies = (
                share["point"] is not None
                and share["point"] >= MIN_MULTI_REGION_SHARE
                and share["ci95"][0] > 0.0
            )
            per_intervention[name] = {
                "multi_region_share": share,
                "unbounded_share": unbounded,
                "qualifies": bool(qualifies),
            }

        qualifying = [k for k, v in per_intervention.items() if v["qualifies"]]
        scored.append({
            "n_bins": cell["n_bins"],
            "percentile": cell["percentile"],
            "code_bytes": cell["code_bytes"],
            "saturation_occupancy": cell["saturation_occupancy"],
            "control_codes_intact": control_clean,
            "interventions": per_intervention,
            "qualifies_any": bool(qualifying),
            "qualifies_all": len(qualifying) == len(per_intervention) and bool(qualifying),
            "qualifying_interventions": qualifying,
        })

    def smallest(key: str):
        ok = [c for c in scored if c["control_codes_intact"] and c[key]]
        if not ok:
            return None
        best = min(ok, key=lambda c: (c["n_bins"], c["code_bytes"],
                                      c["saturation_occupancy"]))
        return {"n_bins": best["n_bins"], "percentile": best["percentile"],
                "code_bytes": best["code_bytes"]}

    under_any, under_all = smallest("qualifies_any"), smallest("qualifies_all")
    return {
        "rule": {
            "min_multi_region_share": MIN_MULTI_REGION_SHARE,
            "interval": "95% percentile bootstrap over documents",
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
            "near_null_conditions": near_null,
            "non_null_interventions": non_null,
            "requires": ("near-null codes intact, share at or above the "
                         "minimum, and a bootstrap lower bound above zero"),
        },
        "uncertainty_unit": report["data"].get("uncertainty_unit"),
        "documents_are_independent_episodes":
            report["data"].get("documents_are_independent_episodes"),
        "cells": scored,
        "selection_under_any_intervention": under_any,
        "selection_under_all_interventions": under_all,
        "readings_agree": under_any == under_all,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--sweep", default=str(here / "results" / "sweep.json"))
    ap.add_argument("--out", default=str(here / "results" / "decision.json"))
    a = ap.parse_args()

    sweep = Path(a.sweep)
    report = json.loads(sweep.read_text())
    sidecar = sweep.parent / report.get(
        "per_document_sidecar", sweep.with_suffix(".per-document.npz").name)
    if not sidecar.exists():
        raise SystemExit(
            f"missing {sidecar}; regenerate it with run_sweep.py --cache ... "
            f"--out {sweep}")
    out = evaluate(report, sidecar)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {a.out}")
    print(f"  under any intervention: {out['selection_under_any_intervention']}")
    print(f"  under all interventions: {out['selection_under_all_interventions']}")
    print(f"  readings agree: {out['readings_agree']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
