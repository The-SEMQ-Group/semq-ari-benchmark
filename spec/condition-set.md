# ARI embedding conditions

An environment is `E = (process, machine, precision, library version, concurrency, time, batch shape)`.
The baseline `E₀` records these settings.
Each condition changes the named setting while preserving the other controlled settings.
Hosted APIs can conceal changes to several internal settings.

| ID | Required comparison | Interpretation |
| --- | --- | --- |
| `same` | Repeat immediately in the same context. | Within-context agreement. |
| `proc` | Start a fresh process on the same machine. For an API, use a fresh client and connection. | Process or request-context variation. |
| `mach` | Use another machine with the same CPU model. | Machine variation. |
| `prec` | Compare fp32 with fp16 or bf16 on the same machine. | Precision variation. |
| `lib` | Change the library patch version on the same machine. | Library variation. |
| `conc` | Compare idle execution with execution under concurrent load. | Concurrency variation. |
| `time` | Repeat after more than 24 hours. | Variation across the recorded time interval. |
| `batch` | Change batch size while preserving inputs and process. | Batch-shape variation. |

These descriptions define the comparison, not a unique causal mechanism.
For example, an API concurrency change can expose routing, batching, and queueing differences.
Record the achieved settings and any differences from the intended protocol.

## Same-condition control

For a deterministic local agent with a fixed probe, `same` must have HER 1.000.
A failure requires investigation of both the input computation and the probe.
Test the probe on identical vectors before attributing a failure to the agent.

For an API, `same` is measured and can be below one.
The report's `agent_class` distinguishes `self_hosted` from `api`.
The leaderboard's local-agent control requirement does not apply to API rows.
Independently seeded quantizers are not equivalent to a fixed codebook; see [probe validation](../docs/analysis/probe-validation.md).

## Aggregation

The comparable core is `{proc, conc, time}`.
The embedding score is `ARI(A) = mean(HER(Eₖ))` over the present core conditions.
Report `same` separately and exclude it from the mean.
Report `mach`, `prec`, `lib`, and `batch` as diagnostics. Exclude them from the mean.

An API cannot expose every diagnostic condition.
This is why the aggregate uses the shared core.
A partial-core report is not equivalent to a report with all three core conditions.
Always show the measured condition set with the score.
The decoding specification has its own complete-core requirement.

## Diagnostic reporting

Report HER, H̄, uncertainty, and each available audit digest.
State the Hamming unit and normalization used by the implementation.
A missing batch condition and batch HER 1.000 are different results.
A precision change can alter codes while Recall@10 remains unchanged; retrieval lists can still change.
See [regime results](../experiments/regime-discrimination/RESULTS.md).

## Time evidence

A `time` result must carry a `time_evidence` object. The object is defined in [report-schema.json](report-schema.json) under `$defs/timeEvidence`.
It shows that the result compares two captures more than 24 hours apart with everything else held fixed.
Record the UTC start and end of the baseline encode and of the comparison encode.
The gap is the comparison start minus the baseline end, and it must be greater than 24 hours. The timestamps decide the gap.
`gap_hours` records it rounded up to 0.01 h and must agree with the timestamps within 0.01.
Record the agent id, the input content hash, the probe id, the model and tokenizer revisions, the precision, the hardware, the SDK version, and the harness commit.
Both captures must use the same values for these settings, and the dimension must match. A change in one of them is a different condition.
The capture tools refuse a comparison whose baseline does not record one of these settings or records a different value.

Hardware for a self-hosted model is the architecture, the CPU model, and the GPU names.
The hostname and the operating system version are not part of it: two machines of the same type are the same hardware, and an operating system update is not a hardware change.
For a hosted model, hardware is `provider-internal`.

The evidence check applies these rules:

- `gap_hours` must be a finite number. NaN and infinity are rejected.
- `source_commit` must be a full git commit id: 40 lowercase hex characters.
- For a `self_hosted` agent (or no `agent_class`), `model_revision` and `tokenizer_revision` must also be 40-character lowercase hex commit ids. Branches, tags, and short ids are rejected.
- For a hosted agent (`api`, `vendor`, `rehosted`, `platform`), `model_revision` is the provider's model id, with its snapshot id when one exists, or `provider-internal`.
- No field may hold a value that identifies nothing: a moving ref such as `main`, or a placeholder such as `unknown`, `pinned`, `floating`, or `none`.
- `provider-internal` is allowed only for a hosted agent, and only in `model_revision`, `tokenizer_revision`, `precision`, and `hardware`.
- The object must describe its report: `agent_id`, `input_content_hash`, and `probe_calibration` equal the report's; `precision` and `hardware` equal `environment.precision` and `environment.hardware`; `sdk_version` equals `environment.library_versions.semq` when the report records it.
- Under ARI-Canonical-v0.2 the scale is `2/sqrt(dim)`, with no calibration. The report must record `encoder_fingerprint.dim`, and `encoder_fingerprint.s` must equal `2/sqrt(dim)` to the 6 decimals reports record.
- A non-empty `deviations` list means the result is not a clean canonical capture.

`python -m ari.check_evidence <report>` is the conformance gate. It rejects a `time` result that lacks the object, whose gap is 24 hours or less, that records deviations, or that breaks one of these rules.
`python ari/verify_report.py <report>` checks the object when it is present and fails on a violation. It prints a warning for deviations.
When the object is absent, the verifier prints a warning and passes; `--require-time-evidence` makes the absence a failure.
The object is optional in the schema. Reports signed before it existed do not carry it and were not re-signed.
Those reports verify with the warning. Their `time` results are unverified against the 24-hour rule.
Read the `time` results of those reports with their panel documentation.

## Environment record

Record baseline and condition settings: BLAS implementation, thread count, hardware, precision, and library versions.
For APIs, record request parameters, endpoint, model identifier, and exposed version metadata.
Mark provider-internal settings as unobservable.
A hosted result describes observed agreement, not a controlled estimate of every internal source of drift.
