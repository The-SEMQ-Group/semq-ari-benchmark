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
| `verify_report.py` | Signature and attestation verification, and the time-evidence check. |
| `check_evidence.py` | Schema validation plus the time-evidence check for an unsigned report. |

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
The repository's CI installs it from a private AWS CodeArtifact repository when an authorized role is configured.
The domain, repository, region and account are operator-only and are not committed;
they live in `infra/operator.env` (see `infra/operator.env.example`).
With AWS CLI credentials that can read the SDK repository:

```bash
source infra/operator.env
aws codeartifact login --tool pip --domain "$CA_DOMAIN" --domain-owner "$CA_OWNER" \
    --repository "$CA_REPO" --region "$CA_REGION"
python -m pip install semq
python -m pip show semq
```

The login command changes the local pip index configuration.
Record the installed SDK version with each capture. Use an approved wheel if you cannot access the repository.
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

## Capture the time condition

The `time` condition compares two encodes separated by more than 24 hours ([condition set](../spec/condition-set.md#time-evidence)).
An immediate repeat is not a `time` measurement.
Prerequisites: the `semq` SDK, `python -m pip install -e ".[selfhosted]"`, a git checkout of this repository, and one machine that stays unchanged between the two commands.

1. Capture the baseline:

   ```bash
   python ari/tools/capture_time_baseline.py --mode capture --agent st \
       --model sentence-transformers/all-MiniLM-L6-v2 --inputs data/ari-bench-v0.1.jsonl
   ```

   The command writes `codes.npy` and `meta.json` under `~/ari_time_baseline/st_sentence-transformers_all-MiniLM-L6-v2/`.
   `meta.json` records the encode start and end in UTC, the frozen scale, the input hash, the model revision, the precision, the hardware, the SDK version, and the harness commit.
   Keep this directory. Do not change the SDK, the model cache, or the checkout before step 2.

2. After more than 24 hours, compare on the same machine:

   ```bash
   python ari/tools/capture_time_baseline.py --mode compare --agent st \
       --model sentence-transformers/all-MiniLM-L6-v2 --inputs data/ari-bench-v0.1.jsonl \
       --report leaderboard/submissions/sentence-transformers_all-MiniLM-L6-v2.json \
       --out leaderboard/submissions/sentence-transformers_all-MiniLM-L6-v2.with-time.json
   ```

   `--report` names the existing report for the same agent, inputs, and precision.
   The command re-encodes at the stored scale, adds the `time` cell with its `time_evidence`, recomputes ARI, and writes the merged report to `--out`.
   Without `--report`, it writes `time_cell.json` next to the baseline.
   The command exits with status 1 when the revision, precision, or SDK version differs from the baseline, or when the gap is 24 hours or less.

3. Check the merged report:

   ```bash
   python -m ari.check_evidence leaderboard/submissions/sentence-transformers_all-MiniLM-L6-v2.with-time.json
   ```

   The output names each missing or invalid field. Exit status 0 is the success check.
   `python ari/verify_report.py` runs the same check on a signed report.

Record with the result: the two command lines, the printed timestamps, the gap in hours, the model revision, and the `deviations` list if the output prints one.
A baseline written before `meta.json` carried timestamps produces a `deviations` entry. Such a cell is not a clean canonical capture.
For an API agent, replace `--agent st --model ...` with the API name, for example `--agent gemini`.
