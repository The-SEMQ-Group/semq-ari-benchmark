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

## Environment record

Record baseline and condition settings: BLAS implementation, thread count, hardware, precision, and library versions.
For APIs, record request parameters, endpoint, model identifier, and exposed version metadata.
Mark provider-internal settings as unobservable.
A hosted result describes observed agreement, not a controlled estimate of every internal source of drift.
