"""KMS-backed signing produces exactly what a local key produces.

Moving custody into an HSM must not move the format. These tests stand in a fake KMS
client backed by a local Ed25519 key, so the whole path runs without AWS: the request
shape, the DER unwrapping, and the resulting sidecar.

The request shape has its own test because AWS offers two Ed25519 algorithms and only one
matches. `ED25519_SHA_512` over `MessageType: RAW` is PureEdDSA. `ED25519_PH_SHA_512` is
HashEdDSA, and its signatures verify nowhere in this codebase. Nothing at runtime would
tell us apart, so a test does.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: E402
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kms_signer = _load("_kms_signer", REPO_ROOT / "ari" / "kms_signer.py")

def _fixture_report(path):
    """A minimal report to sign. Attestation binds the report's bytes and the input
    file it names, so these tests need no real submission — only an `input_set` the
    scorer recognizes as frozen."""
    path.write_text(json.dumps({"agent_id": "test/fixture-agent",
                                "input_set": "ARI-Bench-v0.1"}) + "\n")
    return path



class FakeKms:
    """A KMS client that keeps the private key locally instead of in an HSM.

    Records every request so a test can assert on the algorithm, which is the part that
    cannot be checked after the fact.
    """

    def __init__(self, key=None, key_spec=kms_signer.KEY_SPEC, algorithms=None):
        self.key = key or Ed25519PrivateKey.generate()
        self.key_spec = key_spec
        self.algorithms = (algorithms if algorithms is not None
                           else [kms_signer.SIGNING_ALGORITHM])
        self.sign_calls: list[dict] = []
        self.get_public_key_calls = 0

    def get_public_key(self, *, KeyId):
        self.get_public_key_calls += 1
        der = self.key.public_key().public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo)
        return {"KeyId": KeyId, "PublicKey": der, "KeySpec": self.key_spec,
                "SigningAlgorithms": self.algorithms}

    def sign(self, *, KeyId, Message, MessageType, SigningAlgorithm):
        self.sign_calls.append({"KeyId": KeyId, "Message": Message,
                                "MessageType": MessageType,
                                "SigningAlgorithm": SigningAlgorithm})
        return {"Signature": self.key.sign(Message), "KeyId": KeyId,
                "SigningAlgorithm": SigningAlgorithm}


@pytest.fixture
def fake():
    return FakeKms()


@pytest.fixture
def signer(fake):
    return kms_signer.KmsEd25519Signer("arn:aws:kms:eu-west-1:1:key/test", client=fake)


# -- the public half ------------------------------------------------------------------

def test_public_key_is_unwrapped_from_der_to_raw_32_bytes(signer, fake):
    raw = signer.public_key_raw()
    expected = fake.key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    assert raw == expected
    assert len(raw) == 32


def test_public_key_is_fetched_once_and_cached(signer):
    signer.public_key_raw()
    signer.public_key_raw()
    signer.public_key_b64()
    assert signer._kms.get_public_key_calls == 1


def test_a_non_ed25519_key_spec_is_refused():
    fake = FakeKms(key_spec="ECC_NIST_P256", algorithms=["ECDSA_SHA_256"])
    s = kms_signer.KmsEd25519Signer("arn:test", client=fake)
    with pytest.raises(ValueError, match="not 'ECC_NIST_EDWARDS25519'"):
        s.public_key_raw()


def test_a_key_that_cannot_offer_the_pure_algorithm_is_refused():
    fake = FakeKms(algorithms=["ED25519_PH_SHA_512"])
    s = kms_signer.KmsEd25519Signer("arn:test", client=fake)
    with pytest.raises(ValueError, match="does not offer ED25519_SHA_512"):
        s.public_key_raw()


# -- the request shape ----------------------------------------------------------------

def test_signing_uses_pure_eddsa_over_a_raw_message(signer, fake):
    """The PH trap. HashEdDSA would verify nowhere, and nothing else would notice."""
    message = hashlib.sha256(b"manifest").digest()
    signer.sign(message)

    assert len(fake.sign_calls) == 1
    call = fake.sign_calls[0]
    assert call["SigningAlgorithm"] == "ED25519_SHA_512"
    assert call["MessageType"] == "RAW"
    # The notary signs the digest as raw bytes, not as its hex text.
    assert call["Message"] == message
    assert len(call["Message"]) == 32


def test_a_signature_verifies_as_plain_ed25519(signer, fake):
    message = hashlib.sha256(b"manifest").digest()
    signature = signer.sign(message)
    pub = Ed25519PublicKey.from_public_bytes(signer.public_key_raw())
    pub.verify(signature, message)  # raises InvalidSignature on failure


def test_the_key_spec_is_checked_before_any_signature_is_produced():
    """A misconfigured key must fail before it can emit something that looks like proof."""
    fake = FakeKms(key_spec="RSA_4096", algorithms=["RSASSA_PSS_SHA_256"])
    s = kms_signer.KmsEd25519Signer("arn:test", client=fake)
    with pytest.raises(ValueError):
        s.sign(hashlib.sha256(b"x").digest())
    assert fake.sign_calls == []


def test_an_oversized_message_is_refused_before_the_api_call(signer, fake):
    with pytest.raises(ValueError, match="at most 4096"):
        signer.sign(b"x" * 4097)
    assert fake.sign_calls == []


# -- identity -------------------------------------------------------------------------

def test_identity_defaults_to_the_key_arn(fake):
    arn = "arn:aws:kms:eu-west-1:1:key/abc"
    assert kms_signer.KmsEd25519Signer(arn, client=fake).identity == arn


def test_identity_can_be_named(fake):
    s = kms_signer.KmsEd25519Signer("arn:test", client=fake, identity="The SEMQ Group")
    assert s.identity == "The SEMQ Group"


# -- end to end, against the real verifier ---------------------------------------------

def test_a_kms_signed_sidecar_verifies(tmp_path, fake):
    """The whole point: a KMS signature is indistinguishable to a verifier.

    This checks the shipped standalone verifier. That the leaderboard scorer then
    classifies the same sidecar as `attested` is the board's contract and is tested
    in the ari-leaderboard repo, which owns the scorer.
    """
    pytest.importorskip("semq")
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from ari.attest import attest

    report = _fixture_report(tmp_path / "fixture_report.json")
    data = REPO_ROOT / "data" / "ari-bench-v0.1.jsonl"

    signer = kms_signer.KmsEd25519Signer("arn:test", client=fake, identity="org key")
    attest(metric="ARI", report_path=report, input_paths=[data],
           repo_path=tmp_path / "repo", signer=signer, signer_identity="org key",
           extra={"input_set": "ARI-Bench-v0.1"})

    # The shipped standalone verifier accepts it.
    verifier = _load("_v", REPO_ROOT / "ari" / "verify_report.py")
    assert verifier.verify(report, search_dir=data.parent,
                           expect_key=signer.public_key_b64())


def test_kms_and_local_keys_sign_the_same_digest(tmp_path, fake):
    """Custody changes; the signed bytes do not."""
    pytest.importorskip("semq")
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from ari.attest import attest

    data = REPO_ROOT / "data" / "ari-bench-v0.1.jsonl"

    digests = {}
    for name, kwargs in (("kms", {"signer": kms_signer.KmsEd25519Signer(
                                      "arn:test", client=fake)}),
                         ("local", {"signing_key": fake.key})):
        d = tmp_path / name
        d.mkdir()
        report = _fixture_report(d / "fixture_report.json")
        attest(metric="ARI", report_path=report, input_paths=[data],
               repo_path=d / "repo", signer_identity="same", **kwargs)
        sidecar = json.loads((d / f"{report.stem}.attestation.notary").read_text())
        digests[name] = (sidecar["snapshot_sha256"], sidecar["signer_public_key"])

    assert digests["kms"] == digests["local"]


def test_attest_refuses_both_key_sources_at_once(tmp_path, fake):
    pytest.importorskip("semq")
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from ari.attest import attest

    report = tmp_path / "r.json"
    report.write_text("{}")
    with pytest.raises(ValueError, match="exactly one"):
        attest(metric="ARI", report_path=report, input_paths=[],
               repo_path=tmp_path / "repo", signing_key=fake.key,
               signer=kms_signer.KmsEd25519Signer("arn:test", client=fake))


def test_attest_refuses_no_key_at_all(tmp_path):
    pytest.importorskip("semq")
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from ari.attest import attest

    report = tmp_path / "r.json"
    report.write_text("{}")
    with pytest.raises(ValueError, match="exactly one"):
        attest(metric="ARI", report_path=report, input_paths=[],
               repo_path=tmp_path / "repo")
