"""ari — runs the ARI protocol on an agent and emits a signed ARI report.

See ../README.md for architecture. Public entry points:
    run.run_mock_panel   — end-to-end on the mock agent (no SDK/credentials)
    run.aggregate_report — stage 2: build a report from per-condition code bundles
    probe.load_probe     — the SEMQ QBIN probe (semq SDK, or a reference mock)
"""
# Note: `run` is the CLI entry point (python -m ari.run) and is intentionally NOT
# imported here, to avoid a double-import warning when executed as a module.
from . import agents, inputs, metrics, probe, report

__all__ = ["agents", "inputs", "metrics", "probe", "report"]
