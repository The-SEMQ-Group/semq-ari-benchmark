# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""Validate an ARI report against the schema and check its `time` evidence.

    python -m ari.check_evidence report.json

Exit status 0 on a pass. See ari/README.md, "Capture the time condition".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .verify_report import check_time_condition, time_cell_without_evidence

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "spec" / "report-schema.json"


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def schema_violations(report: dict) -> list[str]:
    """Schema errors as `path: message` lines, ordered by path."""
    import jsonschema

    validator = jsonschema.Draft202012Validator(
        load_schema(), format_checker=jsonschema.FormatChecker())

    def leaves(err):
        # A oneOf failure (every condition result is one) says only "not valid
        # under any of the given schemas". The branch the instance was written
        # in is the one with the fewest errors; the other branch rejects the
        # whole shape. jsonschema's best_match prefers the shallower error and
        # picks the wrong branch here.
        if not err.context:
            yield err
            return
        by_branch: dict = {}
        for sub in err.context:
            by_branch.setdefault(sub.relative_schema_path[0], []).append(sub)
        fewest = min(len(v) for v in by_branch.values())
        for subs in by_branch.values():
            if len(subs) == fewest:
                for sub in subs:
                    yield from leaves(sub)

    lines = []
    for err in validator.iter_errors(report):
        for leaf in leaves(err):
            path = "/".join(str(p) for p in leaf.absolute_path) or "<root>"
            line = f"{path}: {leaf.message}"
            if line not in lines:
                lines.append(line)
    return sorted(lines)


LEGACY_TIME_WARNING = ("time cell carries no capture evidence; conformance to the "
                       ">24 h rule is unverified")


def check_report(report: dict, require_time_evidence: bool = True) -> list[str]:
    """Schema violations followed by time-evidence violations. Empty on a pass.

    With `require_time_evidence` false, a `time` cell that carries no
    `time_evidence` block is left to the caller to warn about; a block that is
    present is checked in full either way.
    """
    violations = schema_violations(report)
    if require_time_evidence or not time_cell_without_evidence(report):
        violations += check_time_condition(report)
    return violations


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report", type=Path)
    ap.add_argument("--require-time-evidence", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="fail a time result without a time_evidence block (default); "
                         "--no-require-time-evidence warns instead, as verify_report.py does")
    args = ap.parse_args(argv)
    try:
        report = json.loads(args.report.read_text())
    except (OSError, ValueError) as exc:
        print(f"cannot read {args.report}: {exc}", file=sys.stderr)
        return 1
    violations = check_report(report, args.require_time_evidence)
    for v in violations:
        print(f"  [FAIL] {v}")
    if violations:
        print(f"{args.report}: {len(violations)} violation(s)")
        return 1
    rpc = report.get("results_per_condition") or {}
    if "time" not in rpc:
        print(f"{args.report}: schema valid; no time result to check")
    elif time_cell_without_evidence(report):
        print(f"  [WARN] {LEGACY_TIME_WARNING}")
        print(f"{args.report}: schema valid; time result carries no evidence")
    else:
        gap = rpc["time"]["time_evidence"]["gap_hours"]
        print(f"{args.report}: schema valid; time result measured at a {gap} h gap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
