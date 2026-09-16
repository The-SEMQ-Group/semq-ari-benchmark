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
COMPARATOR_PERCENTILE = 0.99      # the percentile uniform_cell is scored at
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


def paired_difference(a_num, a_den, b_num, b_den, *, seed: int = SEED,
                      n: int = N_BOOTSTRAP) -> dict:
    """Difference of two pooled ratios over the same documents.

    Both operators score the same corpus, so each bootstrap draw indexes both
    with one document sample. An unpaired interval would carry corpus
    variation that cancels here.
    """
    an, ad = np.asarray(a_num, float), np.asarray(a_den, float)
    bn, bd = np.asarray(b_num, float), np.asarray(b_den, float)
    if not (an.size == ad.size == bn.size == bd.size):
        raise ValueError("paired comparison needs the same documents on both sides")
    if ad.sum() <= 0 or bd.sum() <= 0:
        return {"point": None, "ci95": [None, None], "n_documents": int(an.size)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, an.size, size=(n, an.size))
    draws = (an[idx].sum(axis=1) / ad[idx].sum(axis=1)
             - bn[idx].sum(axis=1) / bd[idx].sum(axis=1))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"point": float(an.sum() / ad.sum() - bn.sum() / bd.sum()),
            "ci95": [float(lo), float(hi)], "n_documents": int(an.size)}


def compare_operators(report: dict, sidecar: Path) -> list[dict]:
    """QUANT against uniform scalar quantization at equal bit width.

    The comparator is scored at the current calibration percentile only, so
    this compares operators at a fixed calibration rather than at each
    operator's best one.
    """
    quant = {c["bits_per_coordinate"]: i for i, c in enumerate(report["cells"])
             if c["operator"] == "semq_quant"
             and c["percentile"] == COMPARATOR_PERCENTILE}
    uniform = {c["bits_per_coordinate"]: i for i, c in enumerate(report["cells"])
               if c["operator"] == "uniform_scalar"}

    out = []
    for bits in sorted(set(quant) & set(uniform)):
        for name in report["data"].get("non_null_scored",
                                       report["data"]["non_null_interventions"]):
            a = load_per_document(sidecar, quant[bits], name)
            b = load_per_document(sidecar, uniform[bits], name)
            if "multi_region" not in a or "multi_region" not in b:
                continue
            out.append({
                "bits_per_coordinate": bits,
                "intervention": name,
                "measure": "multi_region_share_of_changed",
                "quant_minus_uniform": paired_difference(
                    a["multi_region"], a["changed"],
                    b["multi_region"], b["changed"]),
            })
    return out


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
        # whatever resolution it buys. An absent near-null condition is a
        # missing control, not a passed one: `all()` over nothing is True, and
        # a cache short of its controls would have qualified every cell.
        missing_controls = [c for c in near_null if c not in conds]
        control_clean = (not missing_controls
                         and all(conds[c]["her"] == 1.0 for c in near_null))

        # A declared intervention that was never scored is not a satisfied one.
        # "every declared intervention qualifies" is trivially true over a list
        # that was pruned to what happened to be on disk, which is why the
        # roles arrive declared and the pruning is checked here.
        missing_interventions = [c for c in non_null if c not in conds]

        per_intervention = {}
        for name in non_null:
            if name not in conds:
                continue
            pd = load_per_document(sidecar, cell_index, name)
            share = bootstrap_share(pd["multi_region"], pd["changed"])
            unbounded = bootstrap_share(pd["unbounded_changed"], pd["changed"])
            # A screen, not an estimate. The point estimate clearing 1% with
            # an interval that excludes zero says the share is real and looks
            # large enough to carry forward. It does NOT say the true share is
            # at least 1% -- for that the interval's lower end has to clear the
            # threshold, which `qualifies_strict` reports separately so the two
            # readings can be compared rather than conflated.
            measured = share["point"] is not None
            qualifies = (measured
                         and share["point"] >= MIN_MULTI_REGION_SHARE
                         and share["ci95"][0] > 0.0)
            qualifies_strict = (measured
                                and share["ci95"][0] >= MIN_MULTI_REGION_SHARE)
            per_intervention[name] = {
                "multi_region_share": share,
                "unbounded_share": unbounded,
                "qualifies": bool(qualifies),
                "qualifies_strict": bool(qualifies_strict),
            }

        qualifying = [k for k, v in per_intervention.items() if v["qualifies"]]
        qualifying_strict = [k for k, v in per_intervention.items()
                             if v["qualifies_strict"]]
        scored.append({
            "n_bins": cell["n_bins"],
            "percentile": cell["percentile"],
            "code_bytes": cell["code_bytes"],
            "saturation_occupancy": cell["saturation_occupancy"],
            "control_codes_intact": control_clean,
            "missing_controls": missing_controls,
            "interventions": per_intervention,
            "qualifies_any": bool(qualifying),
            "qualifies_all": (not missing_interventions
                              and bool(qualifying)
                              and len(qualifying) == len(per_intervention)),
            "qualifying_interventions": qualifying,
            "qualifies_any_strict": bool(qualifying_strict),
            "missing_interventions": missing_interventions,
        })

    def smallest(key: str):
        ok = [c for c in scored if c["control_codes_intact"] and c[key]]
        if not ok:
            return None
        best = min(ok, key=lambda c: (c["n_bins"], c["code_bytes"],
                                      c["saturation_occupancy"]))
        return {"n_bins": best["n_bins"], "percentile": best["percentile"],
                "code_bytes": best["code_bytes"]}

    unscored = {
        "near_null": [c for c in near_null if c not in report["data"]["conditions"]],
        "non_null": [c for c in non_null if c not in report["data"]["conditions"]],
    }
    under_any, under_all = smallest("qualifies_any"), smallest("qualifies_all")
    under_any_strict = smallest("qualifies_any_strict")
    # Two absent selections are not two readings agreeing on one.
    readings_agree = (under_any == under_all) if under_any is not None else None
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
            "threshold_is_a_screen": (
                "The minimum is applied to the point estimate; the interval is "
                "only required to exclude zero. Qualifying therefore does not "
                "establish that the true share is at or above the minimum. "
                "selection_under_any_intervention_strict applies the minimum "
                "to the interval's lower end instead."),
        },
        "uncertainty_unit": report["data"].get("uncertainty_unit"),
        "documents_are_independent_episodes":
            report["data"].get("documents_are_independent_episodes"),
        "declared_but_unscored": unscored,
        "cells": scored,
        "operator_comparison": compare_operators(report, sidecar),
        "selection_under_any_intervention": under_any,
        "selection_under_any_intervention_strict": under_any_strict,
        "selection_under_all_interventions": under_all,
        "readings_agree": readings_agree,
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
    print(f"  under any, interval clears the minimum: "
          f"{out['selection_under_any_intervention_strict']}")
    print(f"  readings agree: {out['readings_agree']}")
    for role, names in out["declared_but_unscored"].items():
        if names:
            print(f"  WARNING: declared {role} conditions never scored: "
                  f"{', '.join(names)}")
    for row in out["operator_comparison"]:
        d = row["quant_minus_uniform"]
        if d["point"] is not None:
            print(f"  {row['intervention']} @{row['bits_per_coordinate']}b "
                  f"QUANT-uniform: {d['point'] * 100:+.2f}pp "
                  f"[{d['ci95'][0] * 100:+.2f}, {d['ci95'][1] * 100:+.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
