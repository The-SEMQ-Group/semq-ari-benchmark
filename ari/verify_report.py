#!/usr/bin/env python3
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
from pathlib import Path

OK, BAD = "PASS", "FAIL"


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

    # 7. Timestamp, if the signer asked for one.
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
