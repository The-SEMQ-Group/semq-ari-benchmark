# ARI harness

The harness encodes inputs, compares condition codes, and writes report JSON.
Use the [root setup procedure](../README.md#build-and-test) before these commands.
Run commands from the repository root.

## Modules

| Module | Purpose |
| --- | --- |
| `agents.py` | Mock, embedding API, and local model adapters. |
| `probe.py` | Canonical SEMQ probe and development mock. |
| `inputs.py` | Input loading and content hashes. |
| `metrics.py` | Equality, Hamming distance, and bootstrap intervals. |
| `report.py` | Report assembly and condition digests. |
| `run.py` | Mock pipeline and local schema validation. |
| `harness.py` | Trajectory agreement and harness-effect measurement. |
| `attest.py`, `kms_signer.py` | Artifact attestation and signature creation. |
| `verify_report.py` | Signature and attestation verification. |

Capture and aggregation are separate stages.
Capture each condition in its required environment. Compare the resulting codes against a common baseline with a fixed calibration.
Report generation does not sign a report automatically.

## Validate the local pipeline

```bash
python -m ari.run --out /tmp/ari-report.json --validate
```

The command writes JSON and validates it against `spec/report-schema.json`.
A validation error returns a nonzero exit status.
This is a development check. Submit real reports through the separate leaderboard repository.

## Install the canonical probe

The public development dependencies do not include `semq`.
The repository's CI installs it from AWS CodeArtifact when an authorized role is configured.
With AWS CLI credentials that can read the SDK repository:

```bash
aws codeartifact login --tool pip --domain semq --repository semq-sdk --region us-east-2
python -m pip install semq
python -m pip show semq
```

The login command changes the local pip index configuration.
Record the installed SDK version with each capture. Use an approved wheel if you cannot access CodeArtifact.
Without SDK access, use the mock pipeline. Do not label mock reports as canonical measurements.
The capture backend requires the QBIN interface used in `probe.py`; SDK interface compatibility must be checked before a full capture.

## Capture a real model

Install the dependencies for the selected adapter:

```bash
python -m pip install -e ".[apis]"
```

Set the provider credential in your environment. For OpenAI, the variable is `OPENAI_API_KEY`.
Then run:

```bash
python ari/tools/run_report.py --agent openai --inputs data/ari-bench-v0.1.jsonl --probe-backend semq
```

This command makes billed API calls. It measures `same` and `proc`, then writes JSON under `leaderboard/submissions/`.
That local directory is a capture output location. It is not the separate leaderboard repository.
Do not use the legacy `--submit` option; it expects a scorer that is absent from this checkout.

For a local embedding model:

```bash
python -m pip install -e ".[selfhosted]"
python ari/tools/run_report.py --agent bge --model BAAI/bge-large-en-v1.5 --inputs data/ari-bench-v0.1.jsonl --probe-backend semq
```

This command downloads model weights and starts a second process for `proc`.
Check memory requirements before selecting a larger model.

## Complete the condition set

The basic capture does not measure `conc` or `time`.
Use the tools below with the [panel protocol](../experiments/deployed-agent-panel/README.md).
Read each tool's `--help` before execution.

| Tool | Purpose |
| --- | --- |
| `tools/proc_pilot.py` | Repeated API calls with persistent and fresh clients. |
| `tools/conc_probe.py` | API concurrency captures. |
| `tools/capture_time_baseline.py` | Baseline for later time comparisons. |
| `tools/selfhosted_conc_time.py` | Local model concurrency and time captures. |
| `tools/gpu_capture.py` | GPU condition captures. |

Preserve the input order, baseline calibration, model revision, environment, and audit digests.
Check the output conditions before comparing reports.
See [CONTRIBUTING.md](../CONTRIBUTING.md#submit-a-report) for submission steps.
