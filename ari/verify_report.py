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
import math
import re
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
    "gap_hours", "agent_id", "input_content_hash", "probe_calibration",
    "model_revision", "tokenizer_revision", "precision", "hardware",
    "machine_id", "sdk_version", "baseline_source_commit", "source_commit",
)
_COMMIT_FIELDS = ("baseline_source_commit", "source_commit")
# A decoding-layer report (layer "decoding", spec/arid-bench-v0.1.md) has no
# probe, dimension or SDK. Its block holds fixed the requested model, the
# protocol id and the request (by digest), and records the provider-reported
# model versions of each capture without holding them fixed: a backend change
# between the two runs is what the condition exists to show (§6).
DECODING_TIME_EVIDENCE_FIELDS = TIME_EVIDENCE_TIMESTAMPS + (
    "gap_hours", "agent_id", "input_content_hash", "protocol", "model",
    "request_sha256", "precision", "hardware", "baseline_model_versions",
    "model_versions", "baseline_source_commit", "source_commit",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# A self-hosted machine_id: the salted SHA-256 that ari/time_evidence.py records.
_MACHINE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
# The harness commits, and a self-hosted model or tokenizer revision, must be a
# full git commit id: 40 lowercase hex digits. A branch, a tag or a short id can
# move or become ambiguous, so it does not pin what ran.
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_REVISION_FIELDS = ("model_revision", "tokenizer_revision")
# agent_class values whose model someone else serves (spec/report-schema.json).
# Only these may record PROVIDER_INTERNAL, and only in _PROVIDER_FIELDS.
HOSTED_CLASSES = frozenset({"api", "vendor", "rehosted", "platform"})
PROVIDER_INTERNAL = "provider-internal"
_PROVIDER_FIELDS = _REVISION_FIELDS + ("precision", "hardware", "machine_id")
# Strings that identify nothing: moving refs, and the placeholders capture code
# writes when it does not know a value. Compared after strip() and lower().
# Extends ari.hub.NON_PINS, which this file cannot import.
NON_IDENTIFYING = frozenset({"", "unknown", "main", "master", "latest", "head",
                             "pinned", "floating", "none", "null", "n/a", "unavailable"})
# Under this probe the scale is 2/sqrt(dim), with no calibration
# (spec/ari-canonical-v0.2.md). Reports round s to 6 decimals.
FIXED_RANGE_PROBE = "ARI-Canonical-v0.2"
_TS_FORMATS = ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ")


def _finite_number(value) -> bool:
    """True for an int or float that is not a bool, NaN or infinite."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _parse_utc(value):
    if not isinstance(value, str):
        return None
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def identity_problem(field: str, value, hosted: bool) -> str | None:
    """Why `value` does not identify `field` of a capture, or None when it does.

    ari/time_evidence.py calls this at capture time, so a bad pin is refused
    before the 24-hour wait rather than after it.
    """
    if not isinstance(value, str) or value.strip().lower() in NON_IDENTIFYING:
        return f"{value!r} does not identify what ran"
    if value == PROVIDER_INTERNAL:
        if hosted and field in _PROVIDER_FIELDS:
            return None
        return (f"{PROVIDER_INTERNAL!r} is allowed only in {', '.join(_PROVIDER_FIELDS)} "
                f"of a hosted agent_class ({', '.join(sorted(HOSTED_CLASSES))})")
    if field == "machine_id" and not _MACHINE_ID_RE.match(value):
        return (f"{value!r} is not a machine id; expected 'sha256:' and 64 lowercase hex "
                f"digits, as ari/time_evidence.py records it")
    if field in _COMMIT_FIELDS or (field in _REVISION_FIELDS and not hosted):
        if not _COMMIT_RE.match(value):
            return (f"{value!r} does not name an immutable revision; expected a full "
                    f"40-character lowercase hex commit id")
    return None


def time_deviations(report: dict) -> list:
    """The `deviations` list of the `time` cell's evidence, or []."""
    cell = (report.get("results_per_condition") or {}).get("time")
    ev = cell.get("time_evidence") if isinstance(cell, dict) else None
    devs = ev.get("deviations") if isinstance(ev, dict) else None
    return devs if isinstance(devs, list) else []


def time_cell_without_evidence(report: dict) -> bool:
    """True when the `time` result is an object that carries no `time_evidence` key.

    Reports signed before the block existed look like this. They were not
    re-signed, so the verifier warns about them instead of failing.
    """
    cell = (report.get("results_per_condition") or {}).get("time")
    return isinstance(cell, dict) and "time_evidence" not in cell


def check_time_condition(report: dict, allow_deviations: bool = False) -> list[str]:
    """Violations in a report's `time` result, each naming the field at fault.

    Empty when the report has no `time` result, or when the result carries a
    complete `time_evidence` block whose timestamps give a gap above
    MIN_GAP_HOURS, whose `gap_hours` agrees with them within 0.01, whose
    identities pass identity_problem, and which describes this report: the
    same agent, inputs, probe, precision, hardware and SDK version (for a
    decoding report: protocol and requested model instead of probe and SDK,
    DECODING_TIME_EVIDENCE_FIELDS). Under
    FIXED_RANGE_PROBE the report's encoder_fingerprint.s must be 2/sqrt(dim).
    A non-empty `deviations` list is a violation unless `allow_deviations`:
    such a cell is not a clean canonical capture.
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

    decoding = report.get("layer") == "decoding"
    fields = DECODING_TIME_EVIDENCE_FIELDS if decoding else TIME_EVIDENCE_FIELDS
    out = [f"{pre}.{f}: missing" for f in fields if f not in ev]

    stamps = {}
    for f in TIME_EVIDENCE_TIMESTAMPS:
        if f in ev:
            dt = _parse_utc(ev[f])
            if dt is None:
                out.append(f"{pre}.{f}: {ev[f]!r} is not a UTC ISO-8601 instant "
                           f"of the form YYYY-MM-DDTHH:MM:SSZ")
            else:
                stamps[f] = dt
    recorded = ev.get("gap_hours")
    if "gap_hours" in ev and not _finite_number(recorded):
        out.append(f"{pre}.gap_hours: {recorded!r} is not a finite number")
    if len(stamps) == len(TIME_EVIDENCE_TIMESTAMPS):
        if stamps["baseline_finished_at"] < stamps["baseline_started_at"]:
            out.append(f"{pre}.baseline_finished_at: earlier than baseline_started_at")
        if stamps["comparison_finished_at"] < stamps["comparison_started_at"]:
            out.append(f"{pre}.comparison_finished_at: earlier than comparison_started_at")
        # The timestamps decide; gap_hours is a convenience that must agree.
        gap = ((stamps["comparison_started_at"] - stamps["baseline_finished_at"])
               .total_seconds() / 3600.0)
        if gap <= MIN_GAP_HOURS:
            out.append(f"{pre}.gap_hours: timestamps give {gap:.2f} h between "
                       f"baseline_finished_at and comparison_started_at; the time "
                       f"condition requires more than {MIN_GAP_HOURS:g} h")
        if _finite_number(recorded) and abs(recorded - gap) > 0.01 + 1e-9:
            out.append(f"{pre}.gap_hours: recorded {recorded} but the timestamps "
                       f"give {gap:.4f}")

    hosted = report.get("agent_class", "self_hosted") in HOSTED_CLASSES
    identities = (("agent_id", "model", "protocol", "precision", "hardware") if decoding else
                  ("agent_id",) + _REVISION_FIELDS + ("precision", "hardware", "machine_id",
                                                     "sdk_version"))
    for f in identities + _COMMIT_FIELDS:
        if f in ev:
            problem = identity_problem(f, ev[f], hosted)
            if problem:
                out.append(f"{pre}.{f}: {problem}")
    if decoding:
        if "request_sha256" in ev and not (isinstance(ev["request_sha256"], str)
                                           and _SHA256_RE.match(ev["request_sha256"])):
            out.append(f"{pre}.request_sha256: {ev['request_sha256']!r} is not a "
                       f"64-character lowercase hex SHA-256")
        for f in ("baseline_model_versions", "model_versions"):
            v = ev.get(f, [])
            if not isinstance(v, list) or any(identity_problem(f, x, hosted) for x in v):
                out.append(f"{pre}.{f}: {v!r} is not a list of reported model versions")

    # The block must describe this report's measurement, not another one's.
    env = report.get("environment") if isinstance(report.get("environment"), dict) else {}
    libs = env.get("library_versions") if isinstance(env.get("library_versions"), dict) else {}
    against = [("agent_id", report.get("agent_id"), "agent_id"),
               ("input_content_hash", report.get("input_content_hash"), "input_content_hash"),
               ("precision", env.get("precision"), "environment.precision"),
               ("hardware", env.get("hardware"), "environment.hardware")]
    if not decoding:
        against.append(("probe_calibration", report.get("probe_calibration"),
                        "probe_calibration"))
    for f, lib in (("sdk_version", "semq"),) if not decoding else (
            ("protocol", "protocol"), ("model", "model_requested")):
        if lib in libs:
            against.append((f, libs[lib], f"environment.library_versions.{lib}"))
    for f, theirs, where in against:
        if f in ev and ev[f] != theirs:
            out.append(f"{pre}.{f}: {ev[f]!r} differs from the report's {where} {theirs!r}")

    if not decoding and report.get("probe_calibration") == FIXED_RANGE_PROBE:
        fp = report.get("encoder_fingerprint")
        fp = fp if isinstance(fp, dict) else {}
        dim, s = fp.get("dim"), fp.get("s")
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            out.append(f"encoder_fingerprint.dim: {dim!r}; {FIXED_RANGE_PROBE} needs "
                       f"the dimension to check the scale")
        elif not _finite_number(s) or abs(s - 2 / math.sqrt(dim)) > 5.1e-7:
            out.append(f"encoder_fingerprint.s: {s!r} is not 2/sqrt({dim}) = "
                       f"{2 / math.sqrt(dim):.6f}, the fixed scale of {FIXED_RANGE_PROBE}")

    devs = ev.get("deviations")
    if devs and not allow_deviations:
        out.append(f"{pre}.deviations: {len(devs)} recorded; a cell with deviations "
                   f"is not a clean canonical capture")
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


def warn(name, detail=""):
    print(f"  [WARN] {name}" + (f"  {detail}" if detail else ""))


def verify(report_path: Path, search_dir: Path | None = None,
           expect_key: str | None = None, require_time_evidence: bool = False) -> bool:
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

    # 7. A `time` result that carries `time_evidence` must satisfy it. Reports
    #    that are not ARI reports (a harness-effect summary, say) have no such
    #    cell and skip this. A cell without the block predates it and was not
    #    re-signed (spec/condition-set.md), so it warns unless the caller asks
    #    for the block. A cell with `deviations` passes here with a warning;
    #    the conformance gate (ari/check_evidence.py) fails it. The check is
    #    stdlib-only and duplicated nowhere else: check_evidence imports it.
    try:
        report = json.loads(report_path.read_text())
    except ValueError:
        report = None
    if isinstance(report, dict) and "time" in (report.get("results_per_condition") or {}):
        if time_cell_without_evidence(report) and not require_time_evidence:
            warn("time cell carries no capture evidence; conformance to the "
                 ">24 h rule is unverified")
        else:
            violations = check_time_condition(report, allow_deviations=True)
            for dev in time_deviations(report):
                warn("time cell deviates from the procedure", dev)
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
    ap.add_argument("--require-time-evidence", action="store_true",
                    help="fail a time result that carries no time_evidence block "
                         "(default: warn, since reports signed before the block "
                         "existed were not re-signed)")
    args = ap.parse_args()
    return 0 if verify(args.report, args.inputs_dir, args.expect_key,
                       args.require_time_evidence) else 1


if __name__ == "__main__":
    sys.exit(main())
