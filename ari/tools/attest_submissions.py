#!/usr/bin/env python3
"""Sign every leaderboard submission, binding each report to the frozen input set.

    ARI_SIGNING_KEY=/path/to/ed25519.pem ARI_SIGNER="ARI leaderboard operator" \
        python ari/tools/attest_submissions.py

Writes `<report>.attestation.json` and `<report>.attestation.notary` beside each report in
`leaderboard/submissions/`. `leaderboard/scoring/score.py` picks the pair up and marks the
row `attested` — but only once the public key is listed in `spec/signers.json`. Adding the
key is a separate, deliberate act: a key that arrives with its own first signature proves
nothing to a verifier.

The manifest names `data/ari-bench-v0.1.jsonl`, so re-pointing a signed report at different
inputs breaks the signature. That binding is the whole reason to sign a report rather than
just publish a digest of it.

Requires `semq` (the notary writes into a SEMQ repo) and `cryptography`. Verification does
not: `ari/verify_report.py` needs only the standard library and `cryptography`, so a third
party never has to install our stack to check our claim.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSIONS = REPO_ROOT / "leaderboard" / "submissions"
# Each report is bound to the frozen set IT declares (a decoding report binds
# the ARI-D prompt set, not the embedding set).
INPUT_SETS = {
    "ARI-Bench-v0.1": REPO_ROOT / "data" / "ari-bench-v0.1.jsonl",
    "ARI-D-Bench-v0.1": REPO_ROOT / "data" / "arid-bench-v0.1.jsonl",
}
INPUT_SET = INPUT_SETS["ARI-Bench-v0.1"]  # default for the dry-run listing

sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--submissions", type=Path, default=SUBMISSIONS)
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be signed, touch nothing")
    ap.add_argument("--kms-key-id", default=os.environ.get("ARI_KMS_KEY_ID"),
                    help="AWS KMS key id or ARN (ECC_NIST_EDWARDS25519). The private key "
                         "never leaves the HSM. Defaults to $ARI_KMS_KEY_ID.")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION"),
                    help="AWS region for the KMS key (defaults to $AWS_REGION)")
    args = ap.parse_args()

    key_path = os.environ.get("ARI_SIGNING_KEY")
    if key_path and args.kms_key_id:
        print("Both ARI_SIGNING_KEY and --kms-key-id are set. Pick one: a board signed by "
              "two different keys is a board with no single signer to check.")
        return 1
    if not key_path and not args.kms_key_id and not args.dry_run:
        print("No signing key configured. Choose one:\n\n"
              "  --kms-key-id <arn>   an Ed25519 key in AWS KMS (ECC_NIST_EDWARDS25519).\n"
              "                       The private half never leaves the HSM.\n\n"
              "  ARI_SIGNING_KEY=...  a path to a local PEM Ed25519 private key:\n"
              "                         openssl genpkey -algorithm ed25519 -out ari.pem\n"
              "                       Keep that file out of the repository.")
        return 1

    reports = sorted(p for p in args.submissions.glob("*.json")
                     if not p.name.endswith(".attestation.json"))
    if not reports:
        print(f"no submissions under {args.submissions}")
        return 1

    if args.dry_run:
        print(f"would sign {len(reports)} report(s) against {INPUT_SET.name}:")
        for p in reports:
            print(f"  {p.name}")
        return 0

    from cryptography.hazmat.primitives import serialization

    from ari.attest import attest

    identity = os.environ.get("ARI_SIGNER", "unnamed")
    if "demo" in identity.lower():
        print(f"refusing to sign with signer identity {identity!r}: the leaderboard is not "
              f"the place for a demo key.")
        return 1

    key = kms = None
    if args.kms_key_id:
        from ari.kms_signer import KmsEd25519Signer

        kms = KmsEd25519Signer(args.kms_key_id, region_name=args.region,
                               identity=identity)
        # Fetch the public half first: it checks the key spec, so a key of the wrong type
        # fails here rather than after writing 13 unusable sidecars.
        pub = kms.public_key_raw()
        print(f"  KMS key {args.kms_key_id} is Ed25519 — signing {len(reports)} report(s)")
    else:
        key = serialization.load_pem_private_key(Path(key_path).read_bytes(), password=None)
        pub = key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    for report in reports:
        declared = json.loads(report.read_text()).get("input_set", "ARI-Bench-v0.1")
        input_set = INPUT_SETS.get(declared)
        if input_set is None:
            print(f"  SKIP {report.name}: input_set {declared!r} is not a frozen set here")
            continue
        att = attest(
            metric="ARI",
            report_path=report,
            input_paths=[input_set],
            repo_path=args.submissions / "attestation-repo",
            signing_key=key,
            signer=kms,
            signer_identity=identity,
            extra={"input_set": declared},
        )
        print(f"  signed {report.name} -> {att.sidecar_path.name}")

    import base64
    print(f"\nsigned {len(reports)} report(s) as {identity!r}.\n"
          f"Public key (add to spec/signers.json once it is published elsewhere too):\n"
          f"  {base64.b64encode(pub).decode()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
