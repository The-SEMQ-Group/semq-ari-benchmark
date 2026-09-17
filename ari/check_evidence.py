# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""Check that an ARI report carries the evidence for what its cells claim.

Schema validation says a report is well formed. It cannot say whether a `time`
result was measured after the gap spec/condition-set.md requires, because the
`time_evidence` block is optional in the schema so that reports signed before
it existed still validate. This command closes that gap: it validates the
report against spec/report-schema.json, then rejects a `time` result that
lacks the block, whose timestamps give a gap of 24 hours or less, or whose
block disagrees with the report it sits in.

    python -m ari.check_evidence report.json

Exit status is 0 when the report passes and 1 when it does not. Each line of
output names the field at fault.

The time check itself lives in ari/verify_report.py so the standalone verifier
can run it without importing this package; it is imported from there, not
copied.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .verify_report import (MIN_GAP_HOURS, TIME_EVIDENCE_FIELDS,  # noqa: F401
                            check_time_condition)

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


def check_report(report: dict) -> list[str]:
    """Schema violations followed by time-evidence violations. Empty on a pass."""
    return schema_violations(report) + check_time_condition(report)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report", type=Path)
    args = ap.parse_args(argv)
    try:
        report = json.loads(args.report.read_text())
    except (OSError, ValueError) as exc:
        print(f"cannot read {args.report}: {exc}", file=sys.stderr)
        return 1
    violations = check_report(report)
    for v in violations:
        print(f"  [FAIL] {v}")
    if violations:
        print(f"{args.report}: {len(violations)} violation(s)")
        return 1
    rpc = report.get("results_per_condition") or {}
    if "time" in rpc:
        gap = rpc["time"]["time_evidence"]["gap_hours"]
        print(f"{args.report}: schema valid; time result measured at a {gap} h gap")
    else:
        print(f"{args.report}: schema valid; no time result to check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
