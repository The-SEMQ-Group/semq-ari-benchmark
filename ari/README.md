# ARI harness

The harness encodes inputs, compares condition codes, and writes report JSON.
Use the [root setup procedure](../README.md#build-and-test) before these commands.
Run commands from the repository root.

## Modules

| Module | Purpose |
| --- | --- |
| `agents.py` | Mock, embedding API, and local model adapters. |
| `probe.py` | Canonical SEMQ probe. |
| `inputs.py` | Input loading and content hashes. |
| `metrics.py` | Equality, Hamming distance, and bootstrap intervals. |
| `report.py` | Report assembly and condition digests. |
| `run.py` | Mock-agent pipeline and local schema validation. |
| `harness.py` | Trajectory agreement and harness-effect measurement. |
| `attest.py`, `kms_signer.py` | Artifact attestation and signature creation. |
| `verify_report.py` | Signature and attestation verification, and the time-evidence check. |
| `check_evidence.py` | Schema validation plus the time-evidence check for an unsigned report. |
| `time_evidence.py` | Baseline record and `time_evidence` block for the `time` condition. |

Capture and aggregation are separate stages.
Capture each condition in its required environment. Compare the resulting codes against a common baseline with a fixed calibration.
Report generation does not sign a report automatically.

## Validate the local pipeline

```bash
python -m ari.run --out /tmp/ari-report.json --validate
```

The command needs the SEMQ SDK (see below); it fails with an error without it.
The command writes JSON and validates it against `spec/report-schema.json`.
A validation error returns a nonzero exit status.
This is a development check. Submit real reports through the separate leaderboard repository.

## Install the canonical probe

The probe is the public [SEMQ SDK](https://github.com/The-SEMQ-Group/semq), `semq` on PyPI.
The package declares it as a dependency, so `pip install -e .` installs it:

```bash
python -m pip install -e ".[dev]"
python -c "import semq; print(semq.__version__, semq.build_info().build_id)"
```

The harness is pinned to `semq` 1.0.0, so every capture uses the same build; any other version fails at import.
Earlier pre-release builds (1.2–1.5, historical) expose a different interface and are not supported.
Record the SDK version and build id with each capture; `ari.run` writes the version into the report environment.

The SDK is licensed under the PolyForm Noncommercial License 1.0.0; it is not covered by this repository's Apache License.
Production use by companies, and any other use that license does not permit, requires a separate license from The SEMQ Group Inc.

The v0.2 probe has no calibration. QUANT bins magnitudes over the fixed range `2 / sqrt(dim)`, and the harness renormalizes every row to unit length before encoding.
Codes from the v0.1 calibrated probe are not comparable with v0.2 codes; `fixed_scale_codes` refuses a stored v0.1 scale.
See [ARI-Canonical-v0.2](../spec/ari-canonical-v0.2.md).

## Capture a real model

Install the dependencies for the selected adapter:

```bash
python -m pip install -e ".[apis]"
```

Set the provider credential in your environment. For OpenAI, the variable is `OPENAI_API_KEY`.
Then run:

```bash
python ari/tools/run_report.py --agent openai --inputs data/ari-bench-v0.1.jsonl
```

This command makes billed API calls. It measures `same` and `proc`, then writes JSON under `leaderboard/submissions/`.
That local directory is a capture output location. It is not the separate leaderboard repository.
Do not use the legacy `--submit` option; it expects a scorer that is absent from this checkout.

For a local embedding model:

```bash
python -m pip install -e ".[selfhosted]"
python ari/tools/run_report.py --agent bge --model BAAI/bge-large-en-v1.5 --inputs data/ari-bench-v0.1.jsonl
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
| `tools/capture_time_baseline.py` | One API agent's time baseline and comparison. |
| `tools/selfhosted_conc_time.py` | Local model concurrency and time captures. |
| `tools/gpu_capture.py` | GPU condition captures. |

Preserve the input order, baseline calibration, model revision, environment, and audit digests.
Check the output conditions before comparing reports.
See [CONTRIBUTING.md](../CONTRIBUTING.md#submit-a-report) for submission steps.

## Capture the time condition

The `time` condition compares two encodes separated by more than 24 hours, with everything else held fixed ([time evidence](../spec/condition-set.md#time-evidence)).
An immediate repeat is not a `time` measurement.
Two tools produce `time` cells with their `time_evidence`:

- `tools/selfhosted_conc_time.py` for a self-hosted model.
- `tools/refresh_leaderboard.py` for the API panel. Each run compares against the baseline from the previous run and then records a new one.

Both store a baseline record next to the baseline codes: the encode's UTC start and end, the agent, the input hash, the dimension, the probe id, the model and tokenizer revisions, the precision, the hardware, the machine id, the SDK version, and the harness commit.
The harness commit is recorded for each capture, not held fixed.
They refuse a revision or harness commit that does not pin what ran when the baseline is captured, not after the wait.
They refuse a comparison whose baseline lacks one of these settings or records a different value.
`refresh_leaderboard.py` then records a new baseline and leaves the row unchanged, as on a first run.

For a self-hosted model:

1. Capture the baseline on the machine that will run the comparison:

   ```bash
   python ari/tools/selfhosted_conc_time.py --mode baseline \
       --model BAAI/bge-large-en-v1.5 --inputs data/ari-bench-v0.1.jsonl
   ```

   The command writes `baseline_codes.npy` and `baseline_meta.json` under `~/ari_selfhosted_conc_time/BAAI_bge-large-en-v1.5/`.
   `--revision <hub commit>` loads a specific commit; without it, the command records the commit the local cache resolves and exits if there is none.
   Keep the directory. Do not change the SDK, the model cache, the checkout, or the hardware before step 2, and run step 2 on the same machine: another instance of the same type is the `mach` condition, and the comparison is refused.

2. After more than 24 hours, run the same command with `--mode measure`.
   The command writes `conc_time_result.json`; its `time` entry carries `time_evidence`.
   It exits with status 1 and writes nothing when the gap is 24 hours or less or a held-fixed setting differs.

3. Copy `time_evidence` into the report's `time` cell.
   The report's `environment.hardware` must be the same hardware id; `run_report.py` records it.

For one API agent outside the weekly refresh, `tools/capture_time_baseline.py` captures a baseline and compares later:

```bash
python ari/tools/capture_time_baseline.py --mode capture --agent gemini --inputs data/ari-bench-v0.1.jsonl
# more than 24 hours later
python ari/tools/capture_time_baseline.py --mode compare --agent gemini --inputs data/ari-bench-v0.1.jsonl \
    --report gemini_gemini-embedding-001.json --out gemini_gemini-embedding-001.with-time.json
```

It shares the baseline store under `~/ari_time_baseline/` with `refresh_leaderboard.py`.
`--report` must be a report for the same agent, inputs, and dimension. The command adds the `time` cell, recomputes ARI and ARI-R, and writes the result to `--out`.
Without `--report`, it writes `time_cell.json` next to the baseline.

For a decoding-layer (ARI-D) row, `experiments/arid-dry-run/run_dry.py` records each run's start and end, harness commit, protocol id, request template and digest, and reported model versions in `results/manifest.json`.
Run the `same` condition twice, more than 24 hours apart, with `--label t2` on the second run.
`tools/build_arid_submission.py` builds the `time` cell's `time_evidence` from the two runs.
It exits with an error when either run lacks these fields. Runs captured before this change lack them and must be recaptured.

Check the report:

```bash
python -m ari.check_evidence report.json
```

The output names each missing or invalid field. Exit status 0 is the success check.
`python ari/verify_report.py` runs the same check on a signed report when the `time` cell carries `time_evidence`.
On a cell without the block it prints a warning and passes; add `--require-time-evidence` to fail instead.
