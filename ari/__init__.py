# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""ari — runs the ARI protocol on an agent and emits a signed ARI report.

See ../README.md for architecture. Public entry points:
    run.run_mock_panel   — end-to-end on the mock agent (no credentials; SDK required)
    run.aggregate_report — stage 2: build a report from per-condition code bundles
    probe.load_probe     — the SEMQ QUANT probe (semq SDK; no substitute)
"""
# Note: `run` is the CLI entry point (python -m ari.run) and is intentionally NOT
# imported here, to avoid a double-import warning when executed as a module.
from . import agents, inputs, metrics, probe, report

__all__ = ["agents", "inputs", "metrics", "probe", "report"]
