"""Migrate leaderboard/submissions/*.json from the v0.1 flat results_per_condition shape to
the multi-detector shape (spec/report-schema.json $defs/conditionResult).

    {"HER": 0.86, "Hbar": 1.2, "HER_ci": [0.84, 0.88]}
        ->
    {"detectors": {"semq": {"HER": 0.86, "Hbar": 1.2, "HER_ci": [0.84, 0.88],
                            "criterion_type": "exact",
                            "bytes_of_reference_state": <derived from encoder_fingerprint.dim>,
                            "unit": "code"}}}

Purely a reshape: no HER/Hbar/HER_ci value is recomputed or changed. Idempotent — a
condition entry that already has a `detectors` key is left alone.

Usage:
    python ari/tools/migrate_multidetector.py [glob ...]
    (defaults to leaderboard/submissions/*.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ari.rpc import semq_bytes_of_reference_state  # noqa: E402

DEFAULT_GLOB = "leaderboard/submissions/*.json"


def migrate_condition(entry: dict, bytes_of_reference_state: int) -> tuple[dict, bool]:
    """Return (possibly-migrated entry, changed?)."""
    if "detectors" in entry:
        return entry, False
    semq = {"HER": entry["HER"], "Hbar": entry["Hbar"]}
    if "HER_ci" in entry:
        semq["HER_ci"] = entry["HER_ci"]
    semq["criterion_type"] = "exact"
    semq["bytes_of_reference_state"] = bytes_of_reference_state
    semq["unit"] = "code"
    return {"detectors": {"semq": semq}}, True


def migrate_report(report: dict) -> tuple[dict, bool]:
    dim = (report.get("encoder_fingerprint") or {}).get("dim")
    if dim is None:
        raise ValueError(f"{report.get('agent_id')}: no encoder_fingerprint.dim to derive "
                         "bytes_of_reference_state from")
    bors = semq_bytes_of_reference_state(int(dim))

    rpc = report.get("results_per_condition", {})
    changed = False
    new_rpc = {}
    for cond, entry in rpc.items():
        new_entry, did_change = migrate_condition(entry, bors)
        new_rpc[cond] = new_entry
        changed = changed or did_change
    if changed:
        report = {**report, "results_per_condition": new_rpc}
    return report, changed


def main(argv=None) -> int:
    patterns = argv or [DEFAULT_GLOB]
    paths = sorted({p for pat in patterns for p in REPO.glob(pat)})
    if not paths:
        print(f"no files matched {patterns}")
        return 1

    n_migrated = 0
    for path in paths:
        report = json.loads(path.read_text())
        new_report, changed = migrate_report(report)
        if changed:
            path.write_text(json.dumps(new_report, indent=2) + "\n")
            print(f"migrated  {path.relative_to(REPO)}")
            n_migrated += 1
        else:
            print(f"unchanged {path.relative_to(REPO)} (already multi-detector)")
    print(f"\n{n_migrated}/{len(paths)} file(s) migrated to the multi-detector shape.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
