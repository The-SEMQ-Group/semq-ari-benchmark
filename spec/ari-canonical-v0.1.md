# ARI-Canonical-v0.1

Status: frozen, v0.1-preview.
This specification fixes the embedding probe, calibration rule, and reference model.
Registry additions do not change the probe definition.

## Probe requirements

Use SEMQ QBIN with `n_bins = 2`, giving four symbols at approximately two bits per dimension.
Fix scale `s` at the 99th percentile of the reference distribution.
For v0.1, use the frozen ARI-Bench-v0.1 inputs as that distribution.
The reference model is `BAAI/bge-large-en-v1.5`, with `s = 0.0775` on ARI-Bench-v0.1.
Preserve the calibrated scale across compared conditions.

The probe must return identical codes for identical vectors.
The reconstruction property is `encode(reconstruct(c)) = c`.
These requirements do not exclude other fixed quantizers from separately defined protocols.
See [probe validation](../docs/analysis/probe-validation.md) and [retractions](../docs/retractions.md).

## Design rationale

A fixed probe and calibration make reports interpretable across runs.
The drift-sensitivity work fits a near-linear response within its tested perturbation range.
Its uniform-sphere reference prediction is modified by the empirical factor κ.
This model describes the reported synthetic sweep; it does not identify the cause of a deployment change.

## Cross-corpus calibration evidence

These historical measurements use independently calibrated retrieval corpora. Their scope is limited to the tested models and corpora.

| model | dim | s (99th pct, cross-corpus) | cross-corpus variation |
| --- | --- | --- | --- |
| BAAI/bge-large-en-v1.5 | 1024 | 0.0779 | 0.4% |
| BAAI/bge-m3 | 1024 | 0.0820 | 1.6% |
| intfloat/multilingual-e5-large | 1024 | 0.0760 | 2.8% |
| sentence-transformers/all-mpnet-base-v2 | 768 | 0.0992 | 2.6% |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 0.1321 | 0.6% |

## Fingerprint terms

The parameters summarize the measured response. They are not unconditional guarantees of corpus invariance.

| scalar | meaning | value / range | evidence |
| --- | --- | --- | --- |
| `s` | 99th-pct calibration scale | per-model (registry) | <3% across 3 corpora |
| `b` | linear-regime power-law slope | 0.979 ± 0.028 (≈ 1) | universal across 13 models |
| `κ` | angular-concentration multiplier | [1.48, 3.06] | [`../experiments/drift-sensitivity/`](../experiments/drift-sensitivity/) |

## Measured fingerprints

The table records the historical 13-model sensitivity registry. This is distinct from the expanded deployed-model panel.

| model | dim | s | b | κ |
| --- | --- | --- | --- | --- |
| BAAI/bge-large-en-v1.5 | 1024 | 0.0775 | 0.994 | 2.887 |
| BAAI/bge-m3 | 1024 | 0.0825 | 0.994 | 2.774 |
| intfloat/multilingual-e5-large | 1024 | 0.0762 | 0.991 | 2.893 |
| openai/text-embedding-3-large | 3072 | 0.0515 | 0.932 | 1.652 |
| voyage/voyage-4-large | 1024 | 0.0804 | 0.979 | 2.489 |
| cohere/embed-v4.0 | 1536 | 0.0699 | 0.995 | 2.657 |
| mistral/mistral-embed | 1024 | 0.0812 | 0.915 | 1.484 |
| gemini/gemini-embedding-001 | 3072 | 0.0510 | 0.990 | 2.547 |
| nomic-ai/nomic-embed-text-v1.5 | 768 | 0.0934 | 0.999 | 2.924 |
| mixedbread-ai/mxbai-embed-large-v1 | 1024 | 0.0783 | 0.977 | 2.546 |
| Snowflake/snowflake-arctic-embed-l | 1024 | 0.0823 | 1.007 | 3.058 |
| intfloat/e5-mistral-7b-instruct | 4096 | — | 0.960 | 2.065 |
| Alibaba-NLP/gte-Qwen2-7B-instruct | 3584 | — | 0.994 | 2.641 |

The machine-readable registry is [fingerprints-v0.1.csv](fingerprints-v0.1.csv).
A missing calibration value is not a measured zero.

## Calibration limits

The historical sweep found a perturbation floor for `all-mpnet-base-v2` and `all-MiniLM-L6-v2`.
Approximately one percent of code bytes changed under small synthetic perturbations, extending to σ = 1e-10.
Their response slopes could not support a reliable κ fit, so the registry labels them `floor_limited`.
This was model-specific: the 768-dimensional nomic model did not show the same floor.
The two affected models still had measured ARI 1.000 in the original panel.

Specify the probe and calibration together. A bare `n_bins=2` does not identify a comparable measurement.
Do not extrapolate a fitted sensitivity curve outside its measured regime without validation.

## Registry maintenance

Publish the model identifier, dimension, calibration scale, fit parameters, and coverage status.
The two sweep-only 7B embedders require an ARI-Bench calibration before a complete fingerprint can be reported.
Changes to frozen probe parameters require a new specification version.
