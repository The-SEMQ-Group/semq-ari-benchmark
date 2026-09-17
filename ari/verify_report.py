#!/usr/bin/env python3
# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""Check an ARI report against its attestation. Independent by construction.

This imports the standard library and `cryptography`. It does not import SEMQ,
and it does not import the rest of this package. A verifier that had to run the
publisher's code to confirm the publisher's claim would not be independent, so
running this file on its own is the whole point.

Copy it anywhere. It needs the report, the `.attestation.json` manifest, the
`.attestation.notary` sidecar, and whichever input files the manifest names.

    python verify_report.py harness_effect.json

Exit status is 0 when every check passes and 1 when any check fails.

What a pass means: the report and the named inputs are byte-for-byte what the
signer signed, and the signature belongs to the key in the sidecar. What it
does not mean: that the key belongs to anyone in particular, or that the inputs
are honest. Check the key against a registry you already trust. Without a
timestamp token, the recorded time is the signer's own clock.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

OK, BAD = "PASS", "FAIL"

# spec/condition-set.md: the `time` condition repeats after more than 24 hours.
MIN_GAP_HOURS = 24.0

TIME_EVIDENCE_TIMESTAMPS = (
    "baseline_started_at", "baseline_finished_at",
    "comparison_started_at", "comparison_finished_at",
)
TIME_EVIDENCE_FIELDS = TIME_EVIDENCE_TIMESTAMPS + (
    "gap_hours", "probe_calibration", "calibration_scale", "input_content_hash",
    "model_revision", "tokenizer_revision", "precision", "hardware",
    "sdk_version", "source_commit",
)
# Fields that must name an immutable revision. Mirrors ari.hub.NON_PINS, which
# this file cannot import.
_PINNED_FIELDS = ("model_revision", "tokenizer_revision", "source_commit")
_MOVING_REFS = frozenset({"", "unknown", "main", "master", "latest", "head"})
_TS_FORMATS = ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_utc(value):
    if not isinstance(value, str):
        return None
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def check_time_condition(report: dict) -> list[str]:
    """Violations in a report's `time` result, each naming the field at fault.

    Empty when the report has no `time` result, or when the result carries a
    complete `time_evidence` block whose timestamps give a gap above
    MIN_GAP_HOURS and agree with the recorded `gap_hours`. Cross-checks the
    block against the report's own input hash, probe id and precision, so a
    `time` cell measured on different inputs or at another precision is caught
    even when the block itself is complete.
    """
    rpc = report.get("results_per_condition")
    if not isinstance(rpc, dict) or "time" not in rpc:
        return []
    cell = rpc["time"]
    if not isinstance(cell, dict):
        return ["results_per_condition.time: not an object"]
    ev = cell.get("time_evidence")
    pre = "results_per_condition.time.time_evidence"
    if not isinstance(ev, dict):
        return [f"{pre}: missing; a time result must record both captures "
                f"(spec/condition-set.md, $defs/timeEvidence)"]

    out = [f"{pre}.{f}: missing" for f in TIME_EVIDENCE_FIELDS if f not in ev]

    stamps = {}
    for f in TIME_EVIDENCE_TIMESTAMPS:
        if f in ev:
            dt = _parse_utc(ev[f])
            if dt is None:
                out.append(f"{pre}.{f}: {ev[f]!r} is not a UTC ISO-8601 instant "
                           f"of the form YYYY-MM-DDTHH:MM:SSZ")
            else:
                stamps[f] = dt
    if len(stamps) == len(TIME_EVIDENCE_TIMESTAMPS):
        if stamps["baseline_finished_at"] < stamps["baseline_started_at"]:
            out.append(f"{pre}.baseline_finished_at: earlier than baseline_started_at")
        if stamps["comparison_finished_at"] < stamps["comparison_started_at"]:
            out.append(f"{pre}.comparison_finished_at: earlier than comparison_started_at")
        gap = ((stamps["comparison_started_at"] - stamps["baseline_finished_at"])
               .total_seconds() / 3600.0)
        if gap <= MIN_GAP_HOURS:
            out.append(f"{pre}.gap_hours: timestamps give {gap:.2f} h between "
                       f"baseline_finished_at and comparison_started_at; the time "
                       f"condition requires more than {MIN_GAP_HOURS:g} h")
        recorded = ev.get("gap_hours")
        if isinstance(recorded, (int, float)) and abs(recorded - gap) > 0.01:
            out.append(f"{pre}.gap_hours: recorded {recorded} but the timestamps "
                       f"give {gap:.2f}")

    recorded = ev.get("gap_hours", None)
    if "gap_hours" in ev and (not isinstance(recorded, (int, float))
                              or isinstance(recorded, bool)
                              or recorded <= MIN_GAP_HOURS):
        out.append(f"{pre}.gap_hours: {recorded!r} is not greater than {MIN_GAP_HOURS:g}")

    for f in _PINNED_FIELDS:
        v = ev.get(f)
        if f in ev and (not isinstance(v, str) or v.strip().lower() in _MOVING_REFS):
            out.append(f"{pre}.{f}: {v!r} does not name an immutable revision")
    for f in ("probe_calibration", "hardware", "sdk_version", "precision"):
        v = ev.get(f)
        if f in ev and (not isinstance(v, str) or not v.strip()):
            out.append(f"{pre}.{f}: {v!r} is empty")
    scale = ev.get("calibration_scale")
    if "calibration_scale" in ev and (not isinstance(scale, (int, float))
                                      or isinstance(scale, bool) or scale <= 0):
        out.append(f"{pre}.calibration_scale: {scale!r} is not a positive number")

    # The block must describe this report's measurement, not another one's.
    if "input_content_hash" in ev and ev["input_content_hash"] != report.get("input_content_hash"):
        out.append(f"{pre}.input_content_hash: {ev['input_content_hash']!r} differs "
                   f"from the report's input_content_hash")
    if "probe_calibration" in ev and ev["probe_calibration"] != report.get("probe_calibration"):
        out.append(f"{pre}.probe_calibration: {ev['probe_calibration']!r} differs "
                   f"from the report's probe_calibration")
    env_prec = (report.get("environment") or {}).get("precision")
    if "precision" in ev and ev["precision"] != env_prec:
        out.append(f"{pre}.precision: {ev['precision']!r} differs from "
                   f"environment.precision {env_prec!r}")
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def check(results, name, passed, detail=""):
    results.append((name, passed, detail))
    print(f"  [{OK if passed else BAD}] {name}" + (f"  {detail}" if detail else ""))
    return passed


def verify(report_path: Path, search_dir: Path | None = None,
           expect_key: str | None = None) -> bool:
    base = report_path.parent
    search_dir = search_dir or base
    stem = report_path.stem
    manifest_path = base / f"{stem}.attestation.json"
    sidecar_path = base / f"{stem}.attestation.notary"

    print(f"report    {report_path}")
    print(f"manifest  {manifest_path}")
    print(f"sidecar   {sidecar_path}\n")

    results: list[tuple[str, bool, str]] = []

    for p in (report_path, manifest_path, sidecar_path):
        if not p.exists():
            check(results, f"{p.name} exists", False, "missing")
            return False

    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    sidecar = json.loads(sidecar_path.read_text())

    # 1. The sidecar is a notary sidecar and not something else.
    check(results, "sidecar magic is NTRY", sidecar.get("magic") == "NTRY",
          str(sidecar.get("magic")))

    # 2. The manifest on disk is the one that was signed. Re-serialising it
    #    canonically must reproduce the bytes, or the digest cannot be checked.
    recomputed = canonical_json(manifest)
    check(results, "manifest is canonically serialised",
          recomputed == manifest_bytes,
          "re-serialising changes the bytes" if recomputed != manifest_bytes else "")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    # 3. The signature covers this manifest and no other.
    check(results, "sidecar digest matches the manifest",
          sidecar.get("snapshot_sha256") == manifest_sha,
          f"sidecar {str(sidecar.get('snapshot_sha256'))[:16]}... "
          f"manifest {manifest_sha[:16]}...")

    # 4. The report is the one the manifest names.
    want = manifest.get("report", {}).get("sha256")
    got = sha256_file(report_path)
    check(results, "report digest matches the manifest", want == got,
          f"manifest {str(want)[:16]}...  file {got[:16]}...")

    # 5. Every named input is present and unchanged. This is what stops the
    #    report being re-pointed at different data after signing.
    for item in manifest.get("inputs", []):
        p = search_dir / item["name"]
        if not p.exists():
            check(results, f"input {item['name']} present", False,
                  f"not found under {search_dir}")
            continue
        got = sha256_file(p)
        check(results, f"input {item['name']} unchanged", got == item["sha256"],
              f"manifest {item['sha256'][:16]}...  file {got[:16]}...")

    # 6. The Ed25519 signature verifies against the key in the sidecar.
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
        # A signature checks out against whatever key is in the sidecar, so a
        # forger who replaces both signature and key would pass. Naming the key
        # you expect is what turns this from an integrity check into a
        # statement about who signed.
        if expect_key:
            if sidecar["signer_public_key"].strip() != expect_key.strip():
                check(results, "signer key matches --expect-key", False,
                      f"sidecar carries {sidecar['signer_public_key'][:20]}...")
            else:
                check(results, "signer key matches --expect-key", True)
        pub = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(sidecar["signer_public_key"]))
        # The notary signs the digest as raw bytes, not as its hex text.
        pub.verify(base64.b64decode(sidecar["signature"]),
                   bytes.fromhex(sidecar["snapshot_sha256"]))
        check(results, "Ed25519 signature verifies", True,
              f"signer {sidecar.get('signer_identity') or 'unnamed'}")
    except InvalidSignature:
        check(results, "Ed25519 signature verifies", False, "bad signature")
    except Exception as e:  # noqa: BLE001 - report the reason, do not hide it
        check(results, "Ed25519 signature verifies", False, f"{type(e).__name__}: {e}")

    # 7. A `time` result must carry the evidence for its gap. Reports that are
    #    not ARI reports (a harness-effect summary, say) have no such cell and
    #    skip this. The check is stdlib-only and duplicated nowhere else:
    #    ari/check_evidence.py imports it from here.
    try:
        report = json.loads(report_path.read_text())
    except ValueError:
        report = None
    if isinstance(report, dict) and "time" in (report.get("results_per_condition") or {}):
        violations = check_time_condition(report)
        check(results, "time result carries its evidence", not violations,
              "; ".join(violations) if violations else
              f"gap {report['results_per_condition']['time']['time_evidence']['gap_hours']} h")

    # 8. Timestamp, if the signer asked for one.
    if sidecar.get("tsa_token"):
        print("\n  note: an RFC-3161 token is present. This script does not "
              "parse it.\n        Use an RFC-3161 verifier to recover the "
              "attested time.")
    else:
        print(f"\n  note: no timestamp token. created_at "
              f"({sidecar.get('created_at', '?')}) is the signer's own clock.")

    passed = all(p for _, p, _ in results)
    print(f"\n{'ALL CHECKS PASSED' if passed else 'VERIFICATION FAILED'}"
          f"  ({sum(p for _, p, _ in results)}/{len(results)})")
    if passed and not expect_key:
        print("\nThis proves the report and its inputs are what the key holder "
              "signed.\nIt does not prove WHO holds the key. Pass --expect-key "
              "with a public key\nyou already trust to check that too.")
    elif passed:
        print("\nThis proves the report and its inputs are what the named key "
              "signed.\nIt does not prove the inputs are honest.")
    return passed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report", type=Path)
    ap.add_argument("--inputs-dir", type=Path, default=None,
                    help="where the named input files live (default: "
                         "alongside the report)")
    ap.add_argument("--expect-key", default=None,
                    help="base64 Ed25519 public key you already trust; "
                         "without it, any key that signs consistently passes")
    args = ap.parse_args()
    return 0 if verify(args.report, args.inputs_dir, args.expect_key) else 1


if __name__ == "__main__":
    sys.exit(main())
