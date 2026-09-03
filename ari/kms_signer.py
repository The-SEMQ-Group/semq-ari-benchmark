"""Sign ARI attestations with an Ed25519 key held in AWS KMS.

The point of KMS here is custody, not cryptography. The signature format does not change:
AWS KMS `ECC_NIST_EDWARDS25519` with `ED25519_SHA_512` and `MessageType: RAW` is PureEdDSA
over the message you hand it, which is exactly what `semq.notary` produces locally and what
`ari/verify_report.py` checks. A sidecar signed through KMS verifies with the same code, and
a verifier never learns where the private key lived.

    signer = KmsEd25519Signer(key_id="arn:aws:kms:eu-west-1:...:key/...")
    attest(metric="ARI", report_path=..., input_paths=[...], repo_path=..., signer=signer)

What this changes: the private key never exists outside the HSM, and every signature leaves
a CloudTrail record. What it does not change: a signature still says only that the holder of
that key vouched for the report. Who holds it is a question for the registry, not for this
file.

**The one trap.** AWS offers two Ed25519 signing algorithms and they are not
interchangeable. `ED25519_SHA_512` (MessageType RAW) is PureEdDSA and is the one that
matches. `ED25519_PH_SHA_512` (MessageType DIGEST) is HashEdDSA, a different algorithm whose
signatures will not verify with `Ed25519PublicKey.verify`. This module always sends the
former and refuses a key whose spec is anything but Ed25519, rather than quietly producing
signatures nobody can check.

Needs `boto3`, imported lazily so that nothing else in the package acquires an AWS
dependency. Verification never needs it.
"""

from __future__ import annotations

from typing import Optional

# The only key spec and algorithm this module will use. Anything else is refused rather
# than accommodated: a signature in the wrong algorithm is worse than no signature, because
# it looks like evidence and verifies nowhere.
KEY_SPEC = "ECC_NIST_EDWARDS25519"
SIGNING_ALGORITHM = "ED25519_SHA_512"
MESSAGE_TYPE = "RAW"
# KMS caps a RAW message at 4096 bytes. The message here is a 32-byte digest, so this
# never binds — it is asserted so that a future caller who passes a whole file finds out
# from us rather than from a truncated AWS error.
MAX_RAW_MESSAGE = 4096


class KmsEd25519Signer:
    """An Ed25519 signer backed by AWS KMS.

    Exposes the two operations `ari.attest` needs — `sign` and `public_key_raw` — so the
    attestation path does not care whether the key is a local file or an HSM.
    """

    def __init__(self, key_id: str, *, region_name: Optional[str] = None,
                 client=None, identity: Optional[str] = None):
        self.key_id = key_id
        self._identity = identity
        if client is not None:
            self._kms = client
        else:
            import boto3  # lazy: only a signer needs AWS, never a verifier

            self._kms = boto3.client("kms", region_name=region_name)
        self._public_raw: Optional[bytes] = None

    # -- identity ---------------------------------------------------------------

    @property
    def identity(self) -> str:
        """Label recorded in the sidecar. Defaults to the key ARN, which is at least
        checkable against the registry, unlike a free-text name."""
        return self._identity or self.key_id

    # -- public half ------------------------------------------------------------

    def public_key_raw(self) -> bytes:
        """The 32 raw Ed25519 public key bytes the notary sidecar records.

        `GetPublicKey` returns DER SubjectPublicKeyInfo, so it is unwrapped here. The key
        spec is checked at the same time: this is the moment we can still refuse, before
        any signature exists to be mistaken for evidence.
        """
        if self._public_raw is not None:
            return self._public_raw

        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        resp = self._kms.get_public_key(KeyId=self.key_id)

        spec = resp.get("KeySpec") or resp.get("CustomerMasterKeySpec")
        if spec != KEY_SPEC:
            raise ValueError(
                f"KMS key {self.key_id} has key spec {spec!r}, not {KEY_SPEC!r}. The ARI "
                f"notary format is Ed25519; signing with another algorithm would produce "
                f"a sidecar that no ARI verifier can check.")

        algorithms = resp.get("SigningAlgorithms") or []
        if algorithms and SIGNING_ALGORITHM not in algorithms:
            raise ValueError(
                f"KMS key {self.key_id} does not offer {SIGNING_ALGORITHM}; it offers "
                f"{algorithms}.")

        pub = serialization.load_der_public_key(resp["PublicKey"])
        if not isinstance(pub, Ed25519PublicKey):
            raise ValueError(
                f"KMS key {self.key_id} did not return an Ed25519 public key "
                f"(got {type(pub).__name__}).")

        self._public_raw = pub.public_bytes(encoding=serialization.Encoding.Raw,
                                            format=serialization.PublicFormat.Raw)
        return self._public_raw

    # -- signing ----------------------------------------------------------------

    def sign(self, message: bytes) -> bytes:
        """Sign `message` as PureEdDSA, the same thing a local Ed25519PrivateKey does.

        The notary hands us the raw bytes of a SHA-256 digest, not its hex text, and signs
        those. MessageType RAW keeps that meaning: KMS signs the bytes given, and does not
        treat them as a pre-computed digest.
        """
        if len(message) > MAX_RAW_MESSAGE:
            raise ValueError(
                f"message is {len(message)} bytes; KMS signs at most {MAX_RAW_MESSAGE} "
                f"with MessageType {MESSAGE_TYPE}.")
        # Reading the public key first also runs the key-spec check, so a misconfigured
        # key fails before it can emit a signature.
        self.public_key_raw()
        resp = self._kms.sign(
            KeyId=self.key_id,
            Message=message,
            MessageType=MESSAGE_TYPE,
            SigningAlgorithm=SIGNING_ALGORITHM,
        )
        return resp["Signature"]

    # -- convenience ------------------------------------------------------------

    def public_key_b64(self) -> str:
        import base64

        return base64.b64encode(self.public_key_raw()).decode("ascii")
