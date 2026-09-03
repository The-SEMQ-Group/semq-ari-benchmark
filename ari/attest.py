"""Sign an ARI report so a third party can check it without trusting us.

An ARI number is a claim about someone's system, and often a claim they would
rather not be true. It is worth very little if the only evidence is that we say
we ran the measurement. This binds a report to the exact inputs it came from,
and signs that binding.

**What gets signed is a manifest, not the report alone.** The notary signs one
artifact at a time, so signing only the report would leave the inputs free to
change underneath it. The manifest names every input by digest, names the
report by digest, and records the metric version. Signing the manifest binds
all of them at once, and swapping any one of them breaks the check.

**The verifier does not import this package, or SEMQ.** `verify_report.py`
alongside this file uses the standard library and `cryptography` and nothing
else. A verifier that had to run our code to confirm our claim would not be
independent, so the split is the point rather than an implementation detail.

What this does and does not establish:

* It does establish that a given report came from given inputs, and that a
  holder of a given key vouched for the pair.
* It does not establish that the inputs describe reality. A dataset can be
  filtered before it reaches here. Attestation makes tampering after the fact
  detectable, and says nothing about the honesty of the run itself.
* Without a timestamp authority, `created_at` is the signer's local clock and
  is worth nothing against a determined signer. Pass `tsa_url` for a time that
  a third party can check.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

MANIFEST_SCHEMA = 2


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj) -> bytes:
    """Bytes a verifier can reproduce.

    Key order and separators are fixed, because a digest over JSON is only
    checkable if both sides serialise it the same way.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class Attestation:
    manifest_path: Path
    sidecar_path: Path
    manifest_sha256: str
    snap_id: str


# --- structured references: immutable inputs that are not local files -------
#
# The `inputs` list binds local files by digest. The real inputs of most ARI
# runs are not files: they are datasets and models on the Hugging Face Hub,
# pulled at whatever revision was current when the run happened. Binding a
# derived export instead (an early attempt bound `trajectories.jsonl`, a 4k-row
# sample) fails the one job of the manifest, because that file does not
# reconstruct the report. A reference names the input by its immutable Hub
# revision, so a verifier re-fetches the same bytes and re-derives the report.
#
# The constructors are permissive on purpose: an experiment may build its
# references block before it knows the revision, and an unsigned run must still
# work. The pin is enforced in `validate_references`, which runs only at sign
# time (inside `build_manifest`). So running a report is always allowed; signing
# one with an unpinned dataset or model fails loudly.


def dataset_ref(name: str, revision: Optional[str], *,
                config: Optional[str] = None,
                split: Optional[str] = None) -> dict:
    """One dataset input, named by its immutable Hub revision."""
    if not name:
        raise ValueError("a dataset reference needs a name")
    ref: dict = {"name": name, "revision": revision}
    if config is not None:
        ref["config"] = config
    if split is not None:
        ref["split"] = split
    return ref


def model_ref(model_id: str, revision: Optional[str], *,
              dtype: Optional[str] = None) -> dict:
    """One model input, named by its Hub revision, not a digest of its weights.

    The pin is the revision because a 7B checkpoint is gigabytes and its Hub
    commit already names those bytes immutably. Hashing the weights would bind
    the same fact at far greater cost.
    """
    if not model_id:
        raise ValueError("a model reference needs an id")
    ref: dict = {"id": model_id, "revision": revision}
    if dtype is not None:
        ref["dtype"] = str(dtype)
    return ref


def config_ref(config: dict) -> dict:
    """A digest of the experiment configuration over canonical JSON."""
    return {"sha256": sha256_bytes(canonical_json(config))}


def code_ref(*, git_commit: Optional[str] = None,
             packages: Sequence[str] = ()) -> dict:
    """The commit and the versions of the libraries whose output can move.

    `git_commit` defaults to $GIT_COMMIT so a CI run records it without a
    subprocess. Versions are read for the named packages only, because a full
    environment dump is noise a verifier cannot act on.
    """
    import os
    from importlib import metadata

    versions: dict = {}
    for pkg in sorted(set(packages)):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "git_commit": git_commit or os.environ.get("GIT_COMMIT"),
        "lib_versions": versions,
    }


def build_references(*, datasets: Sequence[dict] = (),
                     models: Sequence[dict] = (),
                     config: Optional[dict] = None,
                     code: Optional[dict] = None) -> dict:
    """Assemble a references block in a canonical order.

    Datasets and models are sorted by identity so the manifest digest does not
    depend on the order a call site happened to list them in.
    """
    refs: dict = {}
    if datasets:
        refs["datasets"] = sorted(
            datasets,
            key=lambda d: (d["name"], d.get("config") or "", d.get("split") or ""))
    if models:
        refs["models"] = sorted(models, key=lambda m: m["id"])
    if config is not None:
        refs["config"] = config
    if code is not None:
        refs["code"] = code
    return refs


def validate_references(references: dict) -> None:
    """Reject a references block that names an input it cannot pin.

    A dataset or model whose revision is missing, or is a moving ref like
    "main" or the "unknown" sentinel a failed resolve returns, cannot be
    re-fetched to a known state, so it reads as provenance and delivers none.
    Raising here makes an unpinned reference impossible to sign rather than
    merely discouraged.
    """
    from .hub import is_pinned

    problems = []
    for d in references.get("datasets", ()):
        if not is_pinned(d.get("revision")):
            problems.append(f"dataset {d.get('name')!r} revision "
                            f"{d.get('revision')!r} is not an immutable commit")
    for m in references.get("models", ()):
        if not is_pinned(m.get("revision")):
            problems.append(f"model {m.get('id')!r} revision "
                            f"{m.get('revision')!r} is not an immutable commit")
    if problems:
        raise ValueError(
            "unpinned references, so this report cannot be attested: "
            + "; ".join(problems)
            + ". Record the resolved Hub commit (ari.hub.hub_revision), or "
            "re-run on a machine that has the snapshot to establish one.")


def _binds_inputs(input_paths: Sequence, references: Optional[dict]) -> bool:
    """True when the manifest binds at least one input beyond the report.

    A `config` reference counts, because a simulation with no dataset or model
    (the power analysis, the baseline sweep) is fully determined by its
    parameters, and binding those is its honest provenance. A data experiment
    binds a dataset or model on top; that is a matter of wiring each call site,
    not something this backstop can enforce.
    """
    if input_paths:
        return True
    refs = references or {}
    return bool(refs.get("datasets") or refs.get("models") or refs.get("config"))


def build_manifest(
    *,
    metric: str,
    report_path,
    input_paths: Sequence,
    references: Optional[dict] = None,
    metric_version: str = "0.1",
    extra: Optional[dict] = None,
) -> dict:
    """The document that binds a report to its inputs."""
    report_path = Path(report_path)
    references = references or {}
    validate_references(references)
    return {
        "schema": MANIFEST_SCHEMA,
        "metric": metric,
        "metric_version": metric_version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "report": {
            "name": report_path.name,
            "sha256": sha256_file(report_path),
            "size_bytes": report_path.stat().st_size,
        },
        "inputs": [
            {
                "name": Path(p).name,
                "sha256": sha256_file(p),
                "size_bytes": Path(p).stat().st_size,
            }
            for p in sorted(input_paths, key=lambda x: Path(x).name)
        ],
        # Datasets and models on the Hub, pinned by revision. See the note above
        # build_manifest: this is where an ARI report's real inputs live.
        "references": references,
        # Recorded because a number that cannot be reproduced on another
        # machine is a different problem from a number that was tampered with,
        # and a verifier should be able to tell them apart.
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "implementation": sys.implementation.name,
        },
        "extra": extra or {},
    }


def _notarize_with_signer(manifest_bytes: bytes, signer, *,
                          signer_identity, sidecar_path) -> None:
    """Write a notary sidecar for a signer that is not an Ed25519PrivateKey.

    `semq.notary.notarize` requires a real `Ed25519PrivateKey`, which an
    HSM-backed key can never be — the private half does not exist outside the
    HSM. The signature is identical either way: PureEdDSA over the raw bytes of
    the manifest digest, which is exactly what `ari/verify_report.py`
    recomputes. This builds the same public `NotarizedSnapshot` rather than
    reaching into notary internals.
    """
    import base64

    from semq.notary import NotarizedSnapshot

    digest = sha256_bytes(manifest_bytes)
    ns = NotarizedSnapshot(
        magic="NTRY",
        schema=1,
        snapshot_sha256=digest,
        signer_public_key=base64.b64encode(signer.public_key_raw()).decode("ascii"),
        signature=base64.b64encode(signer.sign(bytes.fromhex(digest))).decode("ascii"),
        signer_identity=signer_identity,
        tsa_url=None,
        tsa_token=None,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    ns.write_sidecar(sidecar_path)


def attest(
    *,
    metric: str,
    report_path,
    input_paths: Sequence,
    repo_path,
    signing_key=None,
    signer=None,
    references: Optional[dict] = None,
    out_dir=None,
    metric_version: str = "0.1",
    signer_identity: Optional[str] = None,
    tsa_url: Optional[str] = None,
    extra: Optional[dict] = None,
) -> Attestation:
    """Store the report and its inputs, then sign a manifest binding them.

    Pass exactly one of:

    * `signing_key` — an `Ed25519PrivateKey` held locally.
    * `signer` — an object with `sign(bytes) -> bytes` and
      `public_key_raw() -> bytes`, such as `ari.kms_signer.KmsEd25519Signer`,
      for a key that never leaves its HSM. The sidecar format is identical and
      verifies with the same code.

    Generating a key per run defeats the purpose: a verifier checks the key
    against a registry of signers it already trusts, so the key has to outlive
    the report.

    `input_paths` binds local files; `references` binds datasets and models by
    revision. At least one must be present, because a signature over the report
    alone authenticates the output bytes and says nothing about where they came
    from.
    """
    from semq import Repo
    from semq.notary import notarize

    if (signing_key is None) == (signer is None):
        raise ValueError("pass exactly one of signing_key or signer")

    references = references or {}
    if not _binds_inputs(input_paths, references):
        raise ValueError(
            f"attestation for {metric} would bind no inputs: it would "
            "authenticate only the report bytes, not the dataset, model, or "
            "config it came from. Pass input_paths for local files, or "
            "references with datasets/models (data experiments) or a config "
            "(simulations).")

    report_path = Path(report_path)
    out_dir = Path(out_dir) if out_dir else report_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_path = Path(repo_path)
    repo = (Repo.open(repo_path) if (repo_path / "HEAD").exists()
            else Repo.init(repo_path))

    # Content-address the inputs and the report so the artifacts survive
    # alongside the digests that name them.
    for p in list(input_paths) + [report_path]:
        repo.snapshot(Path(p).read_bytes(), label=f"{metric}/{Path(p).name}")

    manifest = build_manifest(
        metric=metric, report_path=report_path, input_paths=input_paths,
        references=references, metric_version=metric_version, extra=extra)
    manifest_bytes = canonical_json(manifest)
    manifest_sha256 = sha256_bytes(manifest_bytes)

    snap = repo.snapshot(manifest_bytes, label=f"{metric}/manifest")
    sidecar = out_dir / f"{report_path.stem}.attestation.notary"
    if signer is not None:
        if tsa_url is not None:
            raise NotImplementedError(
                "the signer path does not request an RFC-3161 timestamp yet")
        _notarize_with_signer(manifest_bytes, signer,
                              signer_identity=signer_identity, sidecar_path=sidecar)
    else:
        notarize(repo, str(snap.id), signing_key,
                 signer_identity=signer_identity, tsa_url=tsa_url,
                 sidecar_path=sidecar)

    manifest_out = out_dir / f"{report_path.stem}.attestation.json"
    manifest_out.write_bytes(manifest_bytes)

    return Attestation(manifest_path=manifest_out, sidecar_path=sidecar,
                       manifest_sha256=manifest_sha256, snap_id=str(snap.id))


def sign_if_configured(
    *,
    metric: str,
    report_path,
    input_paths: Sequence = (),
    references: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> Optional[Attestation]:
    """Attest a report when a signing key is configured, otherwise say so.

    Two ways to configure a key, and at most one may be set:

    * `ARI_KMS_KEY_ID` — an Ed25519 key in AWS KMS (an ARN or alias), signed
      through `ari.kms_signer`. The private half never leaves the HSM. Region
      comes from `ARI_KMS_REGION` (default us-east-2). This is the project key.
    * `ARI_SIGNING_KEY` — a path to a local PEM Ed25519 private key.

    `ARI_SIGNER` labels the signer in the sidecar. Every experiment calls this
    at the end of its run so that signing is one line rather than a copied
    block.

    Generating a key here when none is configured would produce a signature
    that proves nothing, because a verifier checks the key against signers it
    already trusts. An unsigned report is the honest outcome instead.
    """
    import os

    report_path = Path(report_path)
    kms_key_id = os.environ.get("ARI_KMS_KEY_ID")
    key_path = os.environ.get("ARI_SIGNING_KEY")
    if kms_key_id and key_path:
        raise ValueError(
            "both ARI_KMS_KEY_ID and ARI_SIGNING_KEY are set; pick one, because a "
            "report can be signed by only one key")
    if not kms_key_id and not key_path:
        print(f"  not signed: set ARI_KMS_KEY_ID or ARI_SIGNING_KEY to sign "
              f"{report_path.name}")
        return None

    bound = [p for p in input_paths if Path(p).exists()]
    common = dict(
        metric=metric, report_path=report_path, input_paths=bound,
        references=references, repo_path=report_path.parent / "attestation-repo",
        signer_identity=os.environ.get("ARI_SIGNER"), extra=extra)

    if kms_key_id:
        from ari.kms_signer import KmsEd25519Signer

        signer = KmsEd25519Signer(
            kms_key_id, region_name=os.environ.get("ARI_KMS_REGION", "us-east-2"),
            identity=os.environ.get("ARI_SIGNER"))
        # The ARN is at least checkable against the registry; a free-text name is not.
        if common["signer_identity"] is None:
            common["signer_identity"] = signer.identity
        att = attest(signer=signer, **common)
    else:
        from cryptography.hazmat.primitives import serialization

        key = serialization.load_pem_private_key(
            Path(key_path).read_bytes(), password=None)
        if common["signer_identity"] is None:
            common["signer_identity"] = "unnamed"
        att = attest(signing_key=key, **common)

    print(f"  signed {report_path.name} -> {att.sidecar_path.name}")
    return att
