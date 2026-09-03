"""Attestation round-trip, and the tampering it has to catch.

A signature check that only passes is untested. Each test here changes one
thing after signing and asserts the verifier notices. The interesting cases are
the ones where the *report* is untouched and something else moved.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("semq")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ari.attest import (attest, build_manifest, build_references, config_ref,
                        dataset_ref, model_ref, validate_references)
from ari.verify_report import verify

VERIFIER = Path(__file__).resolve().parents[1] / "ari" / "verify_report.py"


@pytest.fixture
def signed(tmp_path):
    """A signed report with one input file beside it."""
    inp = tmp_path / "trajectories.jsonl"
    inp.write_text('{"case_id":"a","harness":"h1","run":0,"outcome":true}\n')
    report = tmp_path / "harness_effect.json"
    report.write_text(json.dumps({"effect": 0.089, "n_cases": 10788}) + "\n")

    att = attest(metric="ARI-E", report_path=report, input_paths=[inp],
                 repo_path=tmp_path / "repo",
                 signing_key=Ed25519PrivateKey.generate(),
                 signer_identity="test@example.com")
    return tmp_path, report, inp, att


def test_round_trip_verifies(signed):
    _, report, _, att = signed
    assert att.manifest_path.exists()
    assert att.sidecar_path.exists()
    assert verify(report) is True


def test_edited_report_fails(signed):
    """The obvious attack: change the number after signing."""
    _, report, _, _ = signed
    report.write_text(json.dumps({"effect": 0.5, "n_cases": 10788}) + "\n")
    assert verify(report) is False


def test_edited_input_fails_even_though_report_is_untouched(signed):
    """The attack the manifest exists to stop.

    Signing only the report would leave the inputs free to change. Here the
    report is byte-identical and the data underneath it is not.
    """
    _, report, inp, _ = signed
    inp.write_text('{"case_id":"a","harness":"h1","run":0,"outcome":false}\n')
    assert verify(report) is False


def test_missing_input_fails(signed):
    _, report, inp, _ = signed
    inp.unlink()
    assert verify(report) is False


def test_swapped_manifest_fails(signed, tmp_path):
    """A manifest from a different run must not validate this report."""
    _, report, _, att = signed
    other = tmp_path / "other"
    other.mkdir()
    inp2 = other / "trajectories.jsonl"
    inp2.write_text('{"case_id":"z","harness":"h9","run":0,"outcome":true}\n')
    report2 = other / "harness_effect.json"
    report2.write_text(json.dumps({"effect": 0.999}) + "\n")
    att2 = attest(metric="ARI-E", report_path=report2, input_paths=[inp2],
                  repo_path=other / "repo",
                  signing_key=Ed25519PrivateKey.generate())
    shutil.copy(att2.manifest_path, att.manifest_path)
    assert verify(report) is False


def test_forged_signature_fails(signed):
    """Re-signing the manifest with another key must not pass.

    A verifier trusts a key it already knows. This asserts the signature is
    actually checked, rather than the sidecar being taken at its word.
    """
    _, report, _, att = signed
    sidecar = json.loads(att.sidecar_path.read_text())
    rogue = Ed25519PrivateKey.generate()
    import base64
    sidecar["signature"] = base64.b64encode(
        rogue.sign(bytes.fromhex(sidecar["snapshot_sha256"]))).decode()
    att.sidecar_path.write_text(json.dumps(sidecar))
    # The public key still names the original signer, so the signature is wrong.
    assert verify(report) is False


def test_manifest_records_inputs_and_metric(signed):
    _, _, inp, att = signed
    m = json.loads(att.manifest_path.read_text())
    assert m["metric"] == "ARI-E"
    assert [i["name"] for i in m["inputs"]] == [inp.name]
    assert m["report"]["sha256"] and m["schema"] == 2


def test_verifier_runs_standalone_without_importing_semq(signed):
    """The property that makes verification independent.

    The script is executed as a subprocess with this package taken off the
    path, so an accidental import of `ari` or `semq` would fail the test rather
    than pass silently.
    """
    tmp, report, _, _ = signed
    copy = tmp / "standalone_verify.py"
    shutil.copy(VERIFIER, copy)
    r = subprocess.run([sys.executable, str(copy), str(report)],
                       capture_output=True, text=True, cwd=tmp)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ALL CHECKS PASSED" in r.stdout

    # Check the import statements, not the prose. The docstring mentions SEMQ
    # precisely because the file does not import it.
    import ast
    modules = set()
    for node in ast.walk(ast.parse(copy.read_text())):
        if isinstance(node, ast.Import):
            modules.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    assert "semq" not in modules, f"verifier imports semq: {sorted(modules)}"
    assert "ari" not in modules, f"verifier imports the package: {sorted(modules)}"
    assert modules <= {"argparse", "base64", "hashlib", "json", "sys",
                       "pathlib", "cryptography", "__future__", "ast"}, sorted(modules)


# ---------------------------------------------------------------------------
# Producer contract: a report's real inputs are datasets and models, not files.
# These pin the rule that finding 4 was about: an attestation must bind the
# inputs, and a reference it cannot pin is worse than none.
# ---------------------------------------------------------------------------

def test_validate_references_rejects_unpinned_dataset():
    refs = build_references(datasets=[dataset_ref("BeIR/scifact", None)])
    with pytest.raises(ValueError, match="immutable commit"):
        validate_references(refs)


def test_validate_references_rejects_unpinned_model():
    refs = build_references(models=[model_ref("meta-llama/Llama-3.1-8B", "")])
    with pytest.raises(ValueError, match="immutable commit"):
        validate_references(refs)


def test_validate_references_rejects_unknown_sentinel():
    """hub_revision returns 'unknown' when it cannot resolve; that is not a pin.

    The whole point of the references block is that a verifier re-fetches the
    exact commit, and 'unknown' (like 'main' or 'latest') names a moving target.
    """
    for moving in ("unknown", "main", "latest", "HEAD"):
        refs = build_references(datasets=[dataset_ref("BeIR/scifact", moving)])
        with pytest.raises(ValueError, match="immutable commit"):
            validate_references(refs)


def test_validate_references_accepts_pinned():
    refs = build_references(
        datasets=[dataset_ref("BeIR/scifact", "abc123", config="corpus")],
        models=[model_ref("all-MiniLM-L6-v2", "def456")])
    validate_references(refs)  # does not raise


def test_build_references_is_order_independent():
    a = build_references(models=[model_ref("z", "1"), model_ref("a", "2")])
    b = build_references(models=[model_ref("a", "2"), model_ref("z", "1")])
    assert a == b
    assert [m["id"] for m in a["models"]] == ["a", "z"]


def test_build_manifest_pins_go_into_the_manifest(tmp_path):
    report = tmp_path / "regime_matrix.json"
    report.write_text('{"rows": []}\n')
    refs = build_references(
        datasets=[dataset_ref("BeIR/scifact", "abc123", config="corpus")],
        models=[model_ref("all-MiniLM-L6-v2", "def456", dtype="float32")],
        config=config_ref({"top_k": 10}))
    m = build_manifest(metric="ARI-R", report_path=report, input_paths=[],
                       references=refs)
    assert m["schema"] == 2
    assert m["references"]["datasets"][0]["revision"] == "abc123"
    assert m["references"]["models"][0]["revision"] == "def456"


def test_build_manifest_refuses_unpinned(tmp_path):
    report = tmp_path / "r.json"
    report.write_text("{}\n")
    refs = build_references(datasets=[dataset_ref("BeIR/scifact", None)])
    with pytest.raises(ValueError, match="immutable commit"):
        build_manifest(metric="ARI-R", report_path=report, input_paths=[],
                       references=refs)


def test_attest_refuses_to_bind_nothing(tmp_path):
    """The finding-4 bug at its source: signing the report and nothing else."""
    report = tmp_path / "r.json"
    report.write_text("{}\n")
    with pytest.raises(ValueError, match="would bind no inputs"):
        attest(metric="ARI-R", report_path=report, input_paths=[],
               repo_path=tmp_path / "repo",
               signing_key=Ed25519PrivateKey.generate())


def test_attest_with_references_only_round_trips(tmp_path):
    """A report whose only inputs are a pinned dataset and model still signs."""
    report = tmp_path / "regime_matrix.json"
    report.write_text('{"rows": [], "n_docs": 5183}\n')
    refs = build_references(
        datasets=[dataset_ref("BeIR/scifact", "abc123", config="corpus")],
        models=[model_ref("all-MiniLM-L6-v2", "def456")])
    att = attest(metric="ARI-R", report_path=report, input_paths=[],
                 references=refs, repo_path=tmp_path / "repo",
                 signing_key=Ed25519PrivateKey.generate(),
                 signer_identity="test@example.com")
    assert att.manifest_path.exists()
    assert verify(report) is True
    manifest = json.loads(att.manifest_path.read_text())
    assert manifest["references"]["datasets"][0]["revision"] == "abc123"
