# Regime Discrimination. Can retrieval metrics tell you which configuration you are in?

**Status: run on CPU.** GPU conditions declared but deferred. Script:
[`run_matrix.py`](run_matrix.py). Machine-readable:
[`results/regime_matrix.json`](results/regime_matrix.json).

**Headline: a precision change moves every SEMQ code and 100% of retrieval result lists,
while changing Recall@10 by an amount indistinguishable from zero.** Benign variation. 
new process, different thread count, different batch size. Moves nothing at all.

This measures the half of the claim that the GPU-determinism experiment asserted without
testing. That document says these events are "invisible to cosine/retrieval" twice and
never measured a retrieval metric.

## Setup

BEIR SciFact: 5,183 documents, 300 queries with real relevance judgements, encoded with
`all-MiniLM-L6-v2`. The reference condition is fp32 on CPU at 4 threads and batch 32.
Every other condition changes exactly one thing and runs in a fresh process. SEMQ codes use
QUANT at 8 bins, calibrated once on the reference embeddings and frozen.

Real qrels are essential. On a clean corpus Recall@10 pins at 1.0 and any comparison
measures that degeneracy rather than the instrument. The mistake is recorded in
[the retraction ledger](../../docs/retractions.md).

## Results

| condition | axis | cos mean | cos p01 | R@10 | ΔR@10 | ΔR@10 95% CI | significant | top-10 lists identical | SEMQ HER |
| --- | --- | ---: | ---: | ---: | ---: | :--- | :---: | ---: | ---: |
| reference | `same` | 1.000000 | 1.000000 | 0.7833 | — | — | — | 100.00% | 1.0000 |
| proc | `proc` | 1.000000 | 1.000000 | 0.7833 | +0.0000 | [+0.0000, +0.0000] | no | 100.00% | 1.0000 |
| threads1 | `proc` | 1.000000 | 1.000000 | 0.7833 | +0.0000 | [+0.0000, +0.0000] | no | 100.00% | 1.0000 |
| batch8 | `batch` | 1.000000 | 1.000000 | 0.7833 | +0.0000 | [+0.0000, +0.0000] | no | 100.00% | 1.0000 |
| batch128 | `batch` | 1.000000 | 1.000000 | 0.7833 | +0.0000 | [+0.0000, +0.0000] | no | 100.00% | 1.0000 |
| **bf16** | `prec` | 1.001052 | 0.998050 | 0.7867 | **+0.0033** | [+0.0000, +0.0100] | **no** | **45.00%** | **0.0000** |
| **int8** | `prec` | 0.938093 | 0.904309 | 0.7772 | **−0.0061** | [−0.0278, +0.0156] | **no** | **0.00%** | **0.0000** |

CIs are paired bootstrap over queries, 10,000 resamples. "Significant" means the interval
excludes zero.

## Reading

**1. Aggregate retrieval quality is blind to both precision changes.** Neither bf16 nor
int8 produces a Recall@10 change distinguishable from zero at 300 queries. int8 changes the
embeddings by 6% mean cosine and *the measured recall goes down by 0.6 points*. Well
inside the noise band. A team watching a recall dashboard would see nothing. This is
exactly the reported industry failure mode: the endpoint returns 200, quality metrics look
normal, and nothing in the logs says the serving precision changed.

**2. SEMQ reads both as a hard zero.** HER = 0.0000 under bf16 and int8: not one document
in 5,183 keeps its code. There is no threshold to argue about and no dashboard to squint
at.

**3. The controls are clean, which is the part that makes (2) worth anything.** A new
process, single-threaded BLAS, and a 4× or 16× batch-size change all give HER = 1.0000 and
100% identical result lists. So HER = 0 on a precision change is not an instrument that
fires at everything. It discriminates. Without these rows the bf16 result would be
uninterpretable.

**4. The original claim needs narrowing.** "Invisible to cosine/retrieval" is true of
retrieval *quality metrics* and false of retrieval *result lists*: int8 changes the top-10
list for 100% of queries and bf16 for 55%. Comparing result lists does detect these events.

That refinement matters, and the honest version of the SEMQ advantage is narrower and more
practical:

- List comparison needs a stored reference top-*k* for every query, and it only covers the
  query distribution you happened to save. HER is computed corpus-side and needs no queries
  at all.
- List comparison gives a number that mixes real drift with tie-order churn. HER at
  matched calibration is exact.
- Quality metrics, the thing teams actually alert on, are blind either way.

**5. `cos mean` above 1.0 is not a bug, it is a finding.** bf16 shows mean cosine 1.001052
against unit-normalised references, because the bf16 model's own normalisation is
imprecise enough that its outputs are no longer unit vectors. Any monitor computing cosine
against a stored reference would need to notice a similarity slightly greater than 1 to
catch this, and most clamp or ignore it.

## What is deferred

Four conditions need a GPU and are declared but not run: `fp16`, `gpu_tf32_off`,
`gpu_tf32_on`, `gpu_bf16`. The TF32 pair is the most important remaining measurement,
because it is the scenario where a model audited at fp32/CPU is served at fp32/GPU with no
configuration change visible to the caller. The GPU-determinism experiment already has the
SEMQ side of that (HER 0.541–0.875). This experiment supplies the retrieval side.

fp16 was moved to the GPU set after a CPU attempt: PyTorch has no optimised CPU fp16
kernels, so a CPU fp16 run measures a fallback path rather than a serving regime.

## Caveats

- One encoder, one corpus, one query set of 300. The CIs measure the query-count limit.
  They do not cover encoder or corpus variation.
- Recall@10 at 300 queries cannot resolve a change smaller than roughly ±0.02. A precision
  change that degrades quality by 1 point would also read as "not significant" here. The
  claim is that quality metrics are underpowered for *detection*, not that quality is
  unaffected.
- `int8` is dynamic quantization of Linear layers via the `qnnpack` engine, which is not
  identical to the GPTQ/AWQ schemes a hosted provider would use.
- SEMQ HER of exactly 0.0000 means the calibration scale, frozen from the reference, is far
  from the perturbed distribution. A rebuilt calibration would give a different number. The
  frozen scale is the point, because a monitor holds it fixed.
