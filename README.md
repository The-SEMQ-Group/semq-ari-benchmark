# Agent Reproducibility Index (ARI)

ARI is a benchmark and reference implementation for measuring whether an AI system produces the same internal representation when it is run again under controlled deployment changes.

This repository contains the versioned ARI and ARI-D specifications, frozen benchmark inputs, a Python harness for comparing runs and building report JSON, and the experiments used to validate the measurement. The leaderboard and submission validator are maintained separately in [`ari-leaderboard`](https://github.com/The-SEMQ-Group/ari-leaderboard).

## What ARI measures

The embedding benchmark runs the same frozen inputs under a baseline and one or more conditions, then compares SEMQ QBIN codes produced from the embeddings.

| Metric | Meaning |
| --- | --- |
| ARI | Mean exact-code agreement over the comparable conditions. Higher is more reproducible; `1.0` means exact agreement for every measured input and condition. |
| HER | Hash equality rate for one condition: the fraction of inputs whose code exactly matches the baseline. |
| H̄ | Mean Hamming distance between baseline and condition codes. Lower is better; `0` means no changed bits. |

The comparable core is `{proc, conc, time}`: a fresh process or request context, concurrent load, and a run separated in time. Additional self-hosted diagnostics cover machine, precision, and library changes. See [`spec/condition-set.md`](spec/condition-set.md) for the normative definitions.

ARI measures representation-level reproducibility. It does not directly measure answer quality, retrieval quality, safety, or semantic equivalence.

## Published results

The current embedding panel contains 14 models measured on ARI-Bench v0.1: five hosted APIs and nine self-hosted models. The hosted APIs span ARI `0.169` to `1.000`; Gemini and all measured self-hosted models score `1.000` on the comparable core.

| Model | ARI | 95% CI |
| --- | ---: | ---: |
| `gemini/gemini-embedding-001` | 1.000 | [1.000, 1.000] |
| `openai/text-embedding-3-large` | 0.845 | [0.822, 0.866] |
| `mistral/mistral-embed` | 0.699 | [0.672, 0.727] |
| `voyage/voyage-4-large` | 0.500 | [0.472, 0.528] |
| `cohere/embed-v4.0` | 0.169 | [0.147, 0.194] |

Other measured results:

- The SEMQ perturbation response has slope `0.979 ± 0.028` across 13 embedding models, close to the expected linear response.
- On SciFact, a serving change altered every SEMQ code while Recall@10 remained statistically unchanged.
- With TF32 disabled, tested self-hosted GPU configurations were reproducible at PyTorch defaults; enabling TF32 reduced cross-process HER to `0.65–0.88`.
- In decoding experiments, TF32 changed every measured logit code while all 48 evaluated generations remained character-identical.

These numbers are snapshots of the measured models and environments, not permanent provider guarantees. Full methods, condition-level results, and caveats are in [`experiments/deployed-agent-panel/RESULTS.md`](experiments/deployed-agent-panel/RESULTS.md), [`experiments/gpu-determinism/RESULTS.md`](experiments/gpu-determinism/RESULTS.md), and [`docs/key-results.md`](docs/key-results.md).

## Installation

ARI requires Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[scoring]"
```

Install optional dependency groups as needed:

```bash
python -m pip install -e ".[apis]"        # hosted provider clients
python -m pip install -e ".[selfhosted]"  # PyTorch and sentence-transformers
python -m pip install -e ".[data]"        # rebuild the BEIR-derived input set
```

Computing canonical QBIN codes requires the separate `semq` SDK. The built-in mock probe is intended for development and tests; reports produced with it are not canonical benchmark submissions.

## Run the local pipeline

Run the complete mock pipeline without model downloads or API credentials:

```bash
python -m ari.run --out report.json
```

Run it against the frozen 1,000-item embedding input set:

```bash
python -m ari.run \
  --inputs data/ari-bench-v0.1.jsonl \
  --out report.json
```

The command writes a report containing condition metrics, bootstrap confidence intervals, environment metadata, the input-set content hash, and audit digests. The mock command exercises report generation; it does not provide a meaningful model score.

For canonical captures, provider configuration, concurrency/time captures, and submission steps, start with [`ari/README.md`](ari/README.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md). A basic provider capture has this form:

```bash
python ari/tools/run_report.py \
  --agent openai \
  --inputs data/ari-bench-v0.1.jsonl
```

Provider runs incur API charges. Self-hosted runs may download large model weights, and some experiment suites require a CUDA GPU.

## Run tests

Install the package in the active environment, then run:

```bash
python -m pytest -q
```

Some tests are skipped when optional packages such as `cryptography` or `semq` are unavailable.

## Repository layout

| Path | Contents |
| --- | --- |
| [`ari/`](ari/) | Harness, metrics, report generation, attestation, provider adapters, and capture tools |
| [`spec/`](spec/) | Normative probe, input-set, condition, fingerprint, signer, and report-schema definitions |
| [`data/`](data/) | Frozen ARI-Bench and ARI-D-Bench JSONL inputs |
| [`experiments/`](experiments/) | Experiment implementations, outputs, analysis, and result summaries |
| [`docs/`](docs/) | Methodology, findings, analysis notes, paper sources, and retraction ledger |
| [`infra/`](infra/) | Remote experiment and result-collection scripts |
| [`tests/`](tests/) | Tests for metrics, harness statistics, attestation, and signing |

The embedding dataset contains 1,000 domain-diverse BEIR records from NFCorpus, SciFact, and FiQA. The decoding dataset contains 100 prompts across five task categories. Hashes and provenance are documented in [`data/README.md`](data/README.md).

## Specifications and reproducibility

Reports are comparable only when they use the same versioned probe, inputs, conditions, and scoring rules. The current frozen artifacts are:

- [`ARI-Canonical-v0.1`](spec/ari-canonical-v0.1.md): SEMQ QBIN `n=2` with 99th-percentile calibration;
- [`ARI-Bench-v0.1`](spec/ari-bench-v0.1.md): frozen embedding inputs;
- [`ARI-D-Bench-v0.1`](spec/arid-bench-v0.1.md): frozen decoding prompts;
- [`report-schema.json`](spec/report-schema.json): machine-readable report contract.

Do not compare reports across incompatible spec versions or condition sets. Preserve the input content hash, environment manifest, model revision, precision, library versions, and per-condition audit digests when publishing a result.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for report submission and development guidance. Changes to frozen artifacts require a version bump; results should include confidence intervals and enough environment information for independent verification.

## License and citation

Code under `ari/` is licensed under Apache-2.0. Specifications and benchmark selections are CC BY 4.0; source documents in the BEIR-derived input set retain their original licenses. See [`LICENSE`](LICENSE) and [`data/README.md`](data/README.md).

Citation metadata is available in [`CITATION.cff`](CITATION.cff).
