# Probe sweep

This exploratory sweep evaluates QUANT probe bin counts and calibration percentiles
against storage, saturation, and displacement bounds. It selects a development
configuration; it is not a confirmatory comparison or an operational index study.

Run from the repository root:

```bash
python experiments/probe-sweep/run_sweep.py --out experiments/probe-sweep/results/sweep.json
```

The sweep uses the fixed split and conditions in [PREREGISTRATION.md](PREREGISTRATION.md).
Generated JSON and NPZ outputs are ignored by git; preserve the command, commit,
environment, and output hashes when publishing a run. Do not treat the
development split as independent episodes.

[RESULTS.md](RESULTS.md) holds the measured run. In short: more bins reduce
the unbounded share but cost storage and erase HER separation for small
precision changes. These findings require an independent, matched-budget
collection before supporting comparative or deployment claims.
