# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""Re-sign existing attestations with the registered KMS key.

Each `<report>.attestation.json` manifest is kept byte for byte, so what it binds
(the report digest, the inputs, the references) does not change. Only the
`.attestation.notary` sidecar is replaced, with a signature by the key named in
spec/signers.json.

A manifest is re-signed only if it still verifies as a manifest: canonically
serialised, and naming the report digest that is on disk now. Signing a manifest
whose report has since changed would vouch for bytes nobody attested.

    ARI_KMS_KEY_ID=alias/semq-ari-attestation AWS_PROFILE=... \\
        python -m ari.tools.resign_attestations experiments/*/results/*.attestation.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ari.attest import canonical_json, sha256_bytes, sha256_file, write_sidecar
from ari.kms_signer import KmsEd25519Signer


def check_manifest(manifest_path: Path) -> str:
    """The manifest digest to sign, after checking it still describes the report on disk."""
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if canonical_json(manifest) != raw:
        raise ValueError(f"{manifest_path}: not canonically serialised")
    report = manifest_path.parent / manifest["report"]["name"]
    if sha256_file(report) != manifest["report"]["sha256"]:
        raise ValueError(f"{manifest_path}: {report.name} changed since it was attested")
    return sha256_bytes(raw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("manifests", nargs="+", type=Path)
    args = ap.parse_args(argv)

    key_id = os.environ.get("ARI_KMS_KEY_ID")
    if not key_id:
        print("set ARI_KMS_KEY_ID to the registered key (alias or ARN)", file=sys.stderr)
        return 2
    signer = KmsEd25519Signer(key_id, region_name=os.environ.get("ARI_KMS_REGION", "us-east-2"),
                              identity=os.environ.get("ARI_SIGNER"))

    # Check every manifest before signing any, so a failure leaves nothing half re-signed.
    digests = {m: check_manifest(m) for m in args.manifests}
    for manifest, digest in digests.items():
        sidecar = manifest.with_name(manifest.name.replace(".attestation.json",
                                                           ".attestation.notary"))
        write_sidecar(digest, signer, signer_identity=signer.identity, sidecar_path=sidecar)
        print(f"  re-signed {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
