# ARI

ARI measures changes in model representations under different deployment conditions.
This repository contains the Python harness, frozen inputs, specifications, and experiment results.
The [leaderboard repository](https://github.com/The-SEMQ-Group/ari-leaderboard) maintains submission scoring and publication.

## Build and test

Use Python 3.12 to match CI. The package declares support for Python 3.9 and later.
Run these commands from the repository root on Linux or macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m ari.run --out /tmp/ari-report.json --validate
python -m pytest -q -rs
python -m build
```

The mock command writes `/tmp/ari-report.json` and checks its structure against the report schema.
It does not measure a real model or validate a leaderboard submission.
Tests must finish without failures. The `-rs` option lists skipped tests and their reasons.
Tests that require the separate `semq` SDK skip when that package is absent.
The build writes a wheel and source archive to `dist/`.

These commands require no API credentials, model downloads, or GPU.
Dependency installation requires access to a Python package index.
For Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell and use a local report path.

## Run a benchmark

To exercise the mock pipeline with all 1,000 frozen embedding inputs:

```bash
python -m ari.run --inputs data/ari-bench-v0.1.jsonl --out /tmp/ari-report.json --validate
```

For real captures, install the required provider or model dependencies and the canonical SEMQ probe.
Follow the [harness guide](ari/README.md). Provider calls incur charges. Model captures can require substantial memory and downloads.

## Measurements

| Term | Definition |
| --- | --- |
| ARI | Mean hash equality rate over the measured core conditions: `proc`, `conc`, and `time`. |
| HER | Fraction of inputs whose complete code matches the baseline code. |
| H̄ | Mean Hamming distance. Read the experiment's definition of its unit and normalization. |
| ARI-R | Representation measurement; the embedding report schema retains the name `ARI`. |
| ARI-D | Decoding measurement. The specification defines output comparisons; experiments also compare internal logits. |
| ARI-E | Harness effect after adjustment for disagreement within each harness. |

An ARI of `1.0` means complete code agreement on the measured inputs and conditions.
It does not establish answer quality, semantic equivalence, or reproducibility under unmeasured conditions.
Compare reports only when their inputs, probe versions, condition sets, and scoring rules match.
See the [condition specification](spec/condition-set.md) and [measurement scope](docs/proposals/ari-decomposition.md).

## Documentation

| Path | Purpose |
| --- | --- |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development, review requirements, documentation rules, and report submission. |
| [ari/README.md](ari/README.md) | Harness modules, provider captures, SDK installation, and validation. |
| [spec/README.md](spec/README.md) | Versioned measurement requirements and report contracts. |
| [data/README.md](data/README.md) | Input formats, provenance, licenses, and hash verification. |
| [experiments/README.md](experiments/README.md) | Experiment index and reproduction requirements. |
| [docs/findings.md](docs/findings.md) | Results overview and evidence limits. |
| [docs/retractions.md](docs/retractions.md) | Withdrawn claims and their replacements. |
| [infra/README.md](infra/README.md) | Remote GPU setup, execution, and result collection. |
| [docs/paper/latex/README.md](docs/paper/latex/README.md) | Paper source and build procedure. |

## License and citation

Code under `ari/` uses Apache-2.0. Specifications and benchmark selections use CC BY 4.0.
Source documents retain their original licenses. See [LICENSE](LICENSE) and [dataset licensing](data/README.md).
Use [CITATION.cff](CITATION.cff) for citation metadata.
