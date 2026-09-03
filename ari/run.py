"""ARI harness orchestrator.

Two stages, so the panel can fan out across instances:

  stage 1 (encode)    — produce SEMQ codes for the ARI-Bench inputs under one condition on
                        one environment. Runs wherever that environment lives (a given
                        instance / precision / library / region / API session).
  stage 2 (aggregate) — diff each condition's codes against the baseline, compute HER / H̄
                        with bootstrap CIs, and assemble the signed ARI report.

This module wires both for the **mock** agent so the pipeline runs end-to-end with no SDK
or credentials. For real agents, stage 1 is executed per environment (producing code
bundles) and stage 2 aggregates them; the seams are the `encode_condition` /
`aggregate_report` functions below.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import metrics, report
from .agents import MockAgent
from .inputs import InputSet, load_ari_bench, sample_inputs
from .probe import Probe, load_probe

# condition sets by agent class (deployed-agent-panel/README.md).
# The full canonical set lives in report.CANONICAL_CONDITIONS (single source of truth).
FULL_CONDITIONS = report.CANONICAL_CONDITIONS
API_CONDITIONS = ["same", "proc", "conc", "time"]

# illustrative drift used only by the mock agent to exercise the pipeline
MOCK_DRIFT = {"proc": 0.0, "mach": 0.02, "prec": 0.05, "lib": 0.005, "conc": 0.0, "time": 0.01}


def encode_condition(agent: MockAgent, probe: Probe, inputs: InputSet, condition: str) -> np.ndarray:
    """Stage 1: SEMQ codes for the inputs under one condition. (Mock path; real agents
    obtain vectors from the actual environment run for this condition.)"""
    vectors = agent.encode_under(inputs.texts, condition)
    return probe.encode(vectors)


def aggregate_report(
    *,
    agent_id: str,
    input_set: InputSet,
    environment: dict,
    codes_by_condition: dict[str, np.ndarray],
    agent_class: str = "self_hosted",
    n_resamples: int = 1000,
    seed: int = 0,
    fingerprint: dict | None = None,
) -> dict:
    """Stage 2: diff each condition against baseline `same` and build the report."""
    baseline = codes_by_condition["same"]
    metrics_by_condition = {
        cond: metrics.aggregate(baseline, codes, n_resamples=n_resamples, seed=seed)
        for cond, codes in codes_by_condition.items()
    }
    return report.build_report(
        agent_id=agent_id,
        input_set=input_set.name,
        input_content_hash=input_set.content_hash,
        environment=environment,
        metrics_by_condition=metrics_by_condition,
        codes_by_condition=codes_by_condition,
        agent_class=agent_class,
        fingerprint=fingerprint,
    )


def run_mock_panel(inputs: InputSet | None = None, dim: int = 256, probe_backend: str = "mock") -> dict:
    """Full end-to-end on the mock agent — the reference the test asserts against."""
    inputs = inputs or sample_inputs()
    agent = MockAgent(dim=dim, drift=MOCK_DRIFT)
    probe = load_probe(agent.encode(inputs.texts), backend=probe_backend)
    codes = {c: encode_condition(agent, probe, inputs, c) for c in FULL_CONDITIONS}
    environment = {
        "blas": "mock", "threads": 1, "hardware": "mock", "precision": "fp32",
        "library_versions": {"ari": report.ARI_VERSION, "probe_backend": probe.backend},
    }
    return aggregate_report(
        agent_id=agent.agent_id, input_set=inputs, environment=environment,
        codes_by_condition=codes,
        fingerprint={"dim": dim, "s": getattr(probe, "s", 0.0), "b": 0.98, "kappa": 2.5},
    )


def _validate_with_scorer(report_path: Path) -> int:
    scorer = Path(__file__).resolve().parents[1] / "leaderboard" / "scoring" / "score.py"
    if not scorer.exists():
        print(f"(scorer not found at {scorer} — skipping validation)")
        return 0
    return subprocess.call([sys.executable, str(scorer), str(report_path)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the ARI harness (mock end-to-end for now).")
    ap.add_argument("--inputs", type=Path, help="ARI-Bench JSONL; omit for the dry-run sample")
    ap.add_argument("--out", type=Path, default=Path("report.json"))
    ap.add_argument("--probe-backend", choices=["auto", "semq", "mock"], default="mock")
    ap.add_argument("--validate", action="store_true", help="run score.py on the output")
    args = ap.parse_args(argv)

    inputs = load_ari_bench(args.inputs) if args.inputs else sample_inputs()
    rep = run_mock_panel(inputs, probe_backend=args.probe_backend)
    report.write_report(args.out, rep)
    print(f"ARI = {rep['ARI']:.4f}  ({rep['agent_id']}, {len(inputs)} inputs)  -> {args.out}")
    print(f"input-set content hash: {inputs.content_hash[:16]}…")
    return _validate_with_scorer(args.out) if args.validate else 0


if __name__ == "__main__":
    raise SystemExit(main())
