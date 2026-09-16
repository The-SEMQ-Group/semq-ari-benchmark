# Probe sweep

This exploratory sweep evaluates QUANT probe bin counts and calibration percentiles
against storage, saturation, and displacement bounds. It selects a development
configuration; it is not a confirmatory comparison or an operational index study.

Run from the repository root:

```bash
python experiments/probe-sweep/run_sweep.py --out experiments/probe-sweep/results/sweep.json
python experiments/probe-sweep/decide.py \
  --sweep experiments/probe-sweep/results/sweep.json \
  --out experiments/probe-sweep/results/decision.json
```

The sweep uses the fixed split and conditions in [PREREGISTRATION.md](PREREGISTRATION.md).
The decision script reports document-level bootstrap intervals and applies the
declared 1% development screen. Generated JSON and NPZ outputs are ignored by git; preserve the command, commit,
environment, and output hashes when publishing a run. Do not treat the
development split as independent episodes.

[RESULTS.md](RESULTS.md) holds the measured run and the decision it produced.
In short: more bins reduce the unbounded share but cost storage and erase HER
separation for small precision changes, QUANT has no consistent advantage over
uniform scalar quantization, and the implemented rule selects a cell the
tie-break defect drove it to. These findings require an independent,
matched-budget collection before supporting comparative or deployment claims.
