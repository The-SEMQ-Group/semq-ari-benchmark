# ARI findings

Everything measured in this cycle, with the results that went against us kept in
place. August 2026.

Each section gives the claim, the number, and the limit of the number. The detail
sits in the linked experiment. Reports are signed, and
[`../ari/verify_report.py`](../ari/verify_report.py) checks one without our software.

---

## 1. Summary

| index | what it measures | strongest result | state |
| --- | --- | --- | --- |
| **ARI-R** | representation | TF32 changes 47% of codes, retrieval quality does not move | measured |
| **ARI-D** | decoding | TF32 changes every reading, all output stays identical | measured on 5 models |
| **ARI-E** | harness | the scaffold decides more than the model, 1.4 to 1 | measured on 22,000 runs |

Two results go against the product. A cheaper statistic matches SEMQ at the decoding
layer, and ARI-E needs no quantizer at all. Sections 4 and 8 give both.

---

## 2. ARI-R: a serving change is invisible to retrieval quality

**Data.** BEIR SciFact. 5,183 documents, 300 queries with real relevance judgements,
`all-MiniLM-L6-v2`. Intervals are a paired bootstrap over queries.

| condition against fp32/CPU | Recall@10 change | 95% CI | top-10 lists same | SEMQ HER |
| --- | ---: | :--- | ---: | ---: |
| new process, 1 thread, batch 8, batch 128 | +0.0000 | [0.0000, 0.0000] | 100.00% | 1.0000 |
| GPU fp32, **TF32 off** | +0.0000 | [0.0000, 0.0000] | 100.00% | **0.9992** |
| GPU fp32, **TF32 on** | +0.0000 | [0.0000, 0.0000] | 96.33% | **0.5267** |
| GPU fp16 | +0.0000 | [0.0000, 0.0000] | 92.67% | 0.0000 |
| CPU int8 | -0.0061 | [-0.0278, +0.0156] | 0.00% | 0.0000 |

**The finding.** A model audited at fp32 on a CPU and served at fp32 on a GPU gets a
different code for 47% of documents. Recall@10 does not move. The interval is exactly
zero, not merely small.

**Why it is credible.** Turning TF32 off gives 0.9992 against the same reference. That
row rules out the GPU as the cause and leaves one switch.

**The limit.** Retrieval *quality* is blind. Retrieval *result lists* are not. int8
changes the top-10 list for every query. The advantage that survives is narrower: HER
needs no stored per-query reference and no query set.

Detail: [regime-discrimination](../experiments/regime-discrimination/RESULTS.md).

---

## 3. ARI-D: the logits move before the tokens

**Data.** Five models. TinyLlama-1.1B and Qwen2.5-3B on the first runs, then
Llama-3.1-8B, Qwen2.5-7B and Mistral-7B-v0.3 at 48 prompts and about 2,265 steps each.
Greedy decoding, so every difference comes from the infrastructure.

| model | condition | tokens same | SEMQ HER | steps: token same, code changed | whole generations identical |
| --- | --- | ---: | ---: | ---: | ---: |
| Llama-3.1-8B | TF32 off | 100.00% | 1.0000 | 0.00% | 48/48 |
| Llama-3.1-8B | **TF32 on** | **100.00%** | 0.0000 | **100.00%** | **48/48** |
| Qwen2.5-7B | TF32 on | 99.96% | 0.0000 | 99.96% | 48/48 |
| Mistral-7B | TF32 on | 100.00% | 0.0000 | 100.00% | 48/48 |

**The finding.** A provider turns on TF32. Every token stays the same. Every complete
generation stays the same. The instrument reads a change at every step.

**The second finding, which matters more commercially.** Small per-token drift is not a
small problem:

| model | tokens lost under bf16 | whole generations lost | amplification |
| --- | ---: | ---: | ---: |
| Llama-3.1-8B | 0.79% | 20.8% | 26x |
| Qwen2.5-7B | 0.79% | 35.4% | 45x |
| Mistral-7B | 0.48% | 20.8% | 43x |

One flipped token derails every token after it. A sub-1% step-level divergence becomes
a one-in-three chance of a different answer.

**Two design results.** H-bar is the metric and HER is not, because HER saturates at
zero for any real precision change. `Context` caps `max_dim` at 65,536, below the
vocabulary of most current models, so a longer vocabulary needs chunking.

**Checks that came back clean.** Mistral's 32,768-token vocabulary fits one context and
behaves like the two chunked models, so chunking does not create the result. Controls
give HER 1.0000 on all three models.

**The limit.** The generation column rests on 48 prompts while the others rest on 2,265
steps, so it carries an interval. An earlier 12-prompt run could not separate bf16 from
fp16 at all. Batch composition moves 19% of steps on Mistral and 67% on Qwen, so a
monitor tuned on one model would misread another.

Detail: [production models](../experiments/decoding-reproducibility/RESULTS-production-models.md).

---

## 4. ARI-D against cheaper alternatives: SEMQ mostly loses

This is the section to read before building on any of this.

| statistic | response at sigma 1e-3 | reference state per step | reproducible bit for bit |
| --- | ---: | ---: | :---: |
| SEMQ H-bar | 1.33e-3 | 16,000 bytes | yes |
| **top-2 margin** | **1.15e-3** | **8 bytes** | **yes** |
| KL divergence | 2.18e-7 | 128,000 bytes | no, 4.6% drift |
| Jensen-Shannon | 5.39e-8 | 128,000 bytes | no, 363% drift |

**KL is out.** It is second order in the perturbation. At sigma 1e-4 it reads -5.1e-10,
which is zero with the wrong sign. It is also not reproducible, so a published KL value
is not a number a third party can recompute.

**The top-2 margin wins on cost.** It gets 86% of the signal for one two-thousandth of
the storage, and it is equally reproducible. To detect a precision change, ship the
margin.

**What survives is coverage.** Move only the tail of the distribution and leave the top
ranks alone. SEMQ does not move, at 0.1289 against 0.1288 for a full perturbation. The
margin and the token statistic read exactly zero. They are structurally blind, not less
sensitive.

**The open question.** No one has shown that a real serving change produces
tail-confined drift. A precision change does not. A token-suppression filter, an
adapter, or a tokenizer change might. This decides whether the coverage argument is
commercially real or only true.

Detail: [baselines](../experiments/decoding-reproducibility/BASELINES.md).

---

## 5. ARI-E: the scaffold decides more than the model

**Data.** `nvidia/Open-SWE-Traces`. Two agent scaffolds, SWE-agent and OpenHands,
crossed with two models over about 18,000 SWE-bench-style instances, up to three
rollouts each. Neither scaffold was built here. An outcome comes from the repository's
own test suite, which the agent never saw.

| contrast | cases | self-consistency | cross agreement | effect | 95% CI |
| --- | ---: | ---: | ---: | ---: | :--- |
| scaffold, Qwen3.5-122B | 10,788 | 0.874 | 0.785 | **+0.089** | [+0.083, +0.094] |
| scaffold, Minimax-M2.5 | 12,195 | 0.899 | 0.849 | **+0.050** | [+0.044, +0.052] |
| model, SWE-agent | 10,674 | 0.880 | 0.839 | +0.041 | [+0.039, +0.046] |
| model, OpenHands | 11,933 | 0.892 | 0.833 | +0.059 | [+0.054, +0.062] |

Mean scaffold effect +0.069. Mean model effect +0.050. Every interval excludes zero.

**The control is worth more than the comparison.** Agents disagree with themselves 10 to
13% of the time. A report that diffs two scaffolds and stops would say the scaffold
changed 21.5% of outcomes on Qwen. The attributable figure is 8.9%. The naive number is
2.4 times too large, and across the four contrasts the control removes 59 to 74% of the
apparent effect.

**Scaffold and model interact.** Qwen3.5 scores 46.8% under SWE-agent and 33.4% under
OpenHands. Minimax moves 4.9 points between the same two. A harness effect quoted
without naming the model hides that.

**The limit.** About 22% of rows were ungraded and dropped, because an ungraded run is
not a failed one. If grading failure tracks task difficulty, the surviving set is
easier than the whole. That is the largest threat to these numbers.

Detail: [harness-effect](../experiments/harness-effect/RESULTS.md).

---

## 6. ARI-E power: the earlier plan would have failed

Simulated before spending the compute, not after.

| pass-rate gap | repeats | 25 cases | 100 cases | 200 cases |
| ---: | ---: | ---: | ---: | ---: |
| 0.00, the null | 2 | 8% | 6% | 6% |
| 0.20 | 2 | **8%** | 11% | 37% |
| 0.20 | 5 | 23% | 74% | 96% |
| 0.40 | 5 | 99% | 100% | 100% |

**The finding.** We planned to run 25 cases at two repeats. That design detects a
20-point gap 8% of the time. The published result that motivated the work, 24/25
against 19/25, is a 20-point gap. The plan would have missed it more than nine times in
ten.

**Two rules that follow.** Repeats buy more than cases, because repeats sharpen the
control that carries the effect. Below about 10 cases the false-positive
rate reaches 18% instead of 5%, so do not run ARI-E that small.

Detail: [harness-power](../experiments/harness-power/).

---

## 7. Probe verifiability: two claims retracted

Whitepaper section 3.1 argued twice that SEMQ is necessary for the index. Both
arguments failed our own testing.

| dimensions per subvector | assignments within 1e-6 of a tie | vectors reading differently | `encode(reconstruct(c)) = c` |
| ---: | ---: | ---: | :--- |
| 12, the standard setting | 0.045% | 0.00% | holds |
| 2 | 2.62% | 0.00% | holds |
| 1, degenerate | 91.19% | 4.64% | holds |

**The finding.** Crowding the centroids raises the near-tie rate by three orders of
magnitude and breaks nothing. The attractor property holds at every setting. The
failure reported earlier needed a constructed codebook with centroids 1e-6 apart, and
k-means does not produce that geometry. Two centroids that close compete for the same
Voronoi mass and merge.

**What replaced the claim.** SEMQ is not necessary for the index. A frozen product
quantizer works. What remains is an argument about verification cost: our guarantee
follows from the operator's form, so a verifier can check it in advance. Theirs needs a
fresh measurement for every codebook that ships.

Detail: [probe-verifiability](../experiments/probe-verifiability/RESULTS.md).

---

## 8. Where SEMQ sits in each index

An honest accounting, because the three layers do not use the product equally.

| index | SEMQ role | strength |
| --- | --- | --- |
| ARI-R | the instrument | load-bearing |
| ARI-D | the instrument, but a cheaper statistic matches it | contested |
| ARI-E | verification only, no quantizer | absent from the metric |

ARI-E compares verdicts and sequences of tool names. Nothing in it is a vector, so a
quantizer has no place there. Forcing one in would be decoration.

**Verification connects all three.** A signed manifest binds every report to its
inputs. Changing the report fails the check, and changing an input while the
report stays identical also fails it. Signing uses SEMQ. Verification does not require
SEMQ, because a check that needs the publisher's code is not independent.

A pass shows that the report and its inputs are what the named key signed. It does not
show who holds the key unless the reader supplies a key already trusted, and it does not
show that the inputs are honest.

---

## 9. Retractions

Our own adversarial testing withdrew each one.

| claim | state |
| --- | --- |
| SEMQ is about 300 times more sensitive than retrieval | retracted |
| SEMQ is necessary for the index, from seed stability | retracted |
| A frozen vector quantizer is not verifiable | retracted |
| Serving changes are invisible to retrieval | narrowed to quality metrics |
| SEMQ is the most sensitive decoding probe | withdrawn, the margin is cheaper |

One number remains weak, and the retraction ledger flags it. The 14x sensitivity gap
against sign-projection LSH comes from one synthetic run, and nobody has reproduced it.

---

## 10. Engineering results

These cost time and are worth recording.

- **`Context` caps `max_dim` at 65,536.** TinyLlama fits. Llama-3, Qwen2.5 and Gemma do
  not. A longer vocabulary is split into chunks with one scale for all of them.
  Calibrating each chunk on its own slice would make the joined code meaningless.
- **The published SDK and the experiments had drifted.** CodeArtifact ships semq 1.4.1,
  which names the operator `SEMQ_OP_QBIN`. Local builds renamed it `SEMQ_OP_QUANT`. The
  scripts failed on the instance after the encoding had finished.
  [`ari/semq_compat.py`](../ari/semq_compat.py) resolves either name.
- **A 16 GiB host cannot hold a 3B model in fp32.** `from_pretrained` builds the whole
  model in CPU memory before any move to the GPU. Use `device_map`, and a larger host
  for 7B and above.
- **A GPU host is the wrong place to measure a CPU precision condition.** The instance
  CPU had AVX2 and no AVX-512, so bf16 ran through an emulation path and one condition
  took over 13 minutes.

---

## 11. What is still open

1. **Is real serving drift ever tail-confined?** This decides whether SEMQ's coverage
   advantage at the decoding layer is commercially real.
2. **Does ARI-D separate infrastructure drift from sampling noise above temperature
   zero?** Every run so far is greedy.
3. **Re-derive the 14x LSH figure** before it goes in front of anyone.
4. **Run ARI-E on a harness pair we control**, to confirm the public-dataset result on
   runs whose grading we can see.
5. **Publish a signing key**, because `--expect-key` is worth nothing without somewhere
   to look the key up.
