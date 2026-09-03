# ARI: An Index for Measuring Reproducibility in Deployed ML Systems

The SEMQ Group · Workshop on ML for Systems, NeurIPS 2026

> **This file is a readable mirror. `latex/ari.tex` is the submission source of truth.**
> The built PDF is `latex/ari.pdf`: body ends on page 4, references and appendices follow.
>
> <!-- TODO: author list and affiliations. Anonymization is NOT required per the CFP. -->

---

## Abstract

Two identical calls to a commercial embedding API can return two different vectors, and nothing a provider publishes says which ones do it or how often. Measured under one protocol on one frozen input set, five commercial embedding APIs span a reproducibility index of 0.169 to 1.000: one is bit-identical across processes, concurrent load, and a 76-hour gap, and another returns a changed internal code on roughly 85% of repeated calls. Reproducibility is a property of each provider's serving stack, and hosted inference is neither uniformly safe nor uniformly broken. Operators miss this for a structural reason: aggregate output quality is a threshold detector reading a signal orders of magnitude below its resolution, so a precision or serving change moves 47-100% of internal codes while Recall@10 moves by an amount a paired bootstrap cannot distinguish from zero. ARI measures reproducibility on internal state instead: a deterministic quantizer over the representation, then hash-equality rate and mean Hamming drift under single-axis perturbations, on a frozen hash-pinned input set, in a report a third party can re-verify without the operator's infrastructure. What the output metric cannot resolve per call still compounds: drift that reads recall@10 = 0.988 diverts 11.5% of 15-hop retrieval trajectories, and 0.48-0.79% per-token divergence under bf16 changes 20.8-35.4% of complete greedy generations.

## 1. The measurement gap

The same model, on the same input, on the same hardware, does not always produce the same vector. Nondeterminism enters a deployed inference stack through BLAS reduction order under multithreading, CUDA atomics, mixed precision (fp16/bf16/TF32), library and kernel versions, hardware heterogeneity, and batch composition at decode time [7]. Each source contributes a perturbation of roughly 1e-7 to 1e-4 per representation component: small enough that the field treats it as numerical noise, and, as we show, large enough to change what a retrieval system stores and what a decoder emits.

Operators do not detect these events, and the reason is structural. For unit-normalized embeddings with minimum score gap `g_min`, a Gaussian perturbation of magnitude σ flips a top-K ranking with probability about `2K·Φ(−g_min/σ)`; at typical retrieval values (`g_min ~ 1e-2`, `σ ~ 1e-6`) that is `Φ(−10⁴) ≈ 0`. Float arithmetic followed by cosine ranking is a low-pass filter: sub-microsigma perturbations never reach top-K, so any quality metric computed on the ranking is a threshold detector with its threshold far above the signal.

The consequence is a silent failure class: a model audited in one configuration and served in another returns 200, holds the quality dashboard flat, and stores internal state that is no longer the state that was audited. Hosted APIs compound it, because the caller cannot set precision or determinism flags, so the drift is both invisible and unfixable from outside.

The phenomena are documented individually: nondeterministic hosted embeddings reported by users since 2023 [8], TF32 numerics, batch-invariant kernels for the decode-time case [7], and a benchmark of embedding reproducibility across precisions [1]. Our contribution is the measurement — a standardized, comparable, third-party-verifiable number on a frozen input set, with a bit-exact metric and bootstrap intervals, that works on systems opaque to their callers. Comparability is what makes it an index rather than an internal check: a team that controls its own stack can diff its own outputs, but a team choosing between vendors has no number to ask for, and the audit regimes that name reproducibility as a required property [2, 3, 4] define no measurement for it either.

## 2. The ARI protocol

**Definitions.** The system under test is `A: X → V`, mapping inputs to a vector-valued internal representation: an embedding, or the per-step logit vector during decoding. An environmental condition fixes process, machine, precision, library version, concurrency and time. A baseline run fixes all axes; each comparison condition perturbs exactly one, so observed drift is attributable to that axis.

| id | perturbs | isolates |
| --- | --- | --- |
| `same` | nothing; immediate repeat | probe purity / within-session determinism |
| `proc` | fresh process | thread schedule, per-process kernel autotuning |
| `conc` | concurrent burst | request routing under load |
| `time` | wall-clock gap > 24h | silent serving change |
| `prec` | fp32 → bf16/fp16 | precision-induced drift (diagnostic) |
| `mach`, `lib` | host, library patch version | hardware and version drift (diagnostics) |

**The probe.** Let `Q` be a deterministic quantizer and `P = Q ∘ A`. We use quantile binning at 2 bins, calibrated by one scalar fixed at the 99th percentile of a published reference distribution. Three properties matter, and retrieval quality is not one of them. `Q` must be bit-deterministic, so any observed difference belongs to the subject rather than the instrument; its response must be linear in the perturbation near zero, so it resolves the small-σ regime instead of cliffing; and its sensitivity Jacobian should be diagonal, so drift attributes back to specific dimensions (Appendix D). The response follows `E[Hamming] = κ·√(2·dim/π)·σ`, with fitted slope `b = 0.979 ± 0.028` across 13 architectures and a per-model prefactor `κ ∈ [1.48, 3.06]` that is stable across corpora. Publishing `(s, b, κ)` once per model gives the protocol a quantitative null hypothesis: drift above the prediction indicates a downstream noise source.

**Metrics.** Per input and condition, the reading is bit-exact code equality and, where codes differ, their Hamming distance: HER (hash-equality rate) in [0,1] and H̄ (mean Hamming drift) in bits, with block-bootstrap confidence intervals. The headline index averages HER over the comparable core `{proc, conc, time}`, the conditions measurable for any system class; `mach`, `prec` and `lib` are reported as diagnostics, since a hosted API cannot expose them and folding them in would penalize a self-hosted system for honesty. One consequence stated plainly: an API's headline is a measured lower bound on its true drift.

**Layers.** The same protocol applies at three layers: ARI-R reads the embedding, ARI-D reads the full-vocabulary logit vector at each greedy decoding step, and ARI-E reads end-to-end agent outcomes, where the metric is verdict agreement across repeated rollouts and no quantizer is involved.

**Verifiability.** Inputs are a frozen, hash-pinned 1,000-item slice of BEIR across medical, scientific, and financial domains. Each report carries per-condition HER, H̄, and CI, an environment manifest, and a SHA-256 audit digest per condition over the code matrix in fixed input order. An Ed25519 attestation binds the report to the exact input digest, so changing either breaks the check; the key is held in AWS KMS and listed in a public signer registry, and the verifier runs on the Python standard library without importing our code, because a verification that requires the publisher's software is not independent. <!-- TODO: confirm leaderboard wording; main registers the KMS key and the 13 submissions verify as `attested`, but leaderboard/README.md still says every row reads `self-reported`. -->

## 3. Results

### 3.1 Reproducibility is provider-specific and spans the full range

Five commercial embedding APIs, one protocol, the frozen input set, n = 1,000, `time` at a ~76h gap, `conc` as a 64-way burst against a calm pass:

| provider / model | `same` | `proc` | `conc` | `time` | **index** | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | :--- |
| gemini-embedding-001 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | [1.000, 1.000] |
| text-embedding-3-large | 0.859 | 0.862 | 0.851 | 0.822 | **0.845** | [0.822, 0.866] |
| mistral-embed | 0.734 | 0.753 | 0.791 | **0.554** | **0.699** | [0.672, 0.727] |
| voyage-4-large | 0.571 | 0.615 | **0.209** | 0.676 | **0.500** | [0.472, 0.528] |
| embed-v4.0 | 0.133 | 0.146 | 0.130 | 0.232 | **0.169** | [0.147, 0.194] |

The spread is the result. One provider is bit-reproducible and another drifts on roughly 85% of repeated calls under identical inputs, and a caller cannot infer which from anything providers publish today.

Three checks support that reading. The 1.000 is not a cache: 150 novel inputs, built by appending a unique nonce to each text, kept that provider at 1.000 while a control provider read 0.807 on the same inputs. The drift is real at the float level: of 30 texts embedded in two separated passes, 3 differed, and where they differed about 2,300 of 3,072 dimensions changed at magnitude ~1e-4 — a different replica's vector, not last-bit noise. And single axes carry distinct failures: only `conc` exposes one provider collapsing from 0.615 to 0.209 under burst, and only a multi-day `time` gap exposes another falling to 0.554, invisible at 19 hours.

Eight self-hosted open models read 1.000 across `proc`, `conc`, and `time`, including with BLAS threads unpinned, and every one collapses to HER ≤ 0.094 under a bf16 serving change. The defensible operational claim is: self-host and pin the precision.

### 3.2 Quality metrics are statistically blind to large internal drift

BEIR SciFact, 5,183 documents, 300 queries with real relevance judgments, all-MiniLM-L6-v2. The reference is fp32/CPU/4-thread/batch-32; every other condition changes one thing in a fresh process. Intervals are a paired bootstrap over queries, 10,000 resamples.

| condition | ΔR@10 | 95% CI | sig. | top-10 same | HER |
| --- | ---: | :--- | :---: | ---: | ---: |
| new proc / 1 thread / batch 8 / batch 128 | +0.0000 | [+0.0000, +0.0000] | no | 100.00% | 1.0000 |
| GPU fp32, TF32 off | +0.0000 | [+0.0000, +0.0000] | no | 100.00% | 0.9992 |
| GPU fp32, **TF32 on** | +0.0000 | [+0.0000, +0.0000] | no | 96.33% | **0.5267** |
| **bf16** | +0.0033 | [+0.0000, +0.0100] | no | 45.00% | **0.0000** |
| **int8** | −0.0061 | [−0.0278, +0.0156] | no | 0.00% | **0.0000** |

Neither precision change moves Recall@10 by an amount distinguishable from zero. int8 shifts the embeddings by 6% mean cosine and its 0.6-point recall drop sits well inside the noise band, while HER reads a hard zero: no document among 5,183 keeps its code. The control rows make that reading meaningful — a new process, a single thread, and a 4x or 16x batch change move nothing on any panel, so HER = 0 under a precision change is a discrimination, not an instrument that fires on everything.

The TF32 pair is the operationally important row. A model audited at fp32 on CPU and served at fp32 on GPU gets a different code for 47% of documents, with no configuration change visible to the caller. Turning TF32 off restores 0.9992 against the same reference, which leaves one switch; an isolation 2x2 over TF32 × `use_deterministic_algorithms` confirms it, with mean cross-process HER 1.000 with TF32 off and 0.749 with it on, and no effect from the determinism flag. TF32's 10-bit mantissa turns a ~1e-7 cross-process kernel difference into ~1e-3, above the quantizer floor.

One scope note, since the blanket claim is often overstated: quality metrics are blind, but result lists are not. int8 changes the top-10 list for every query and bf16 for 55%, so diffing stored lists does detect these events — at the cost of a stored reference top-k, coverage limited to the saved query distribution, and tie-order churn mixed in with real drift. HER is computed corpus-side, needs no queries, and is exact at matched calibration.

### 3.3 Small per-step divergence compounds into outcome divergence

A single retrieval rarely moves its top-K set, so the natural objection is that this drift does not matter. It compounds, and we measure the compounding at two layers.

**Iterative retrieval.** A self-avoiding nearest-neighbor walk over NFCorpus (3,633 documents) embeds the current query at each hop and steps to the nearest unvisited document: an LLM-free model of an iterative RAG loop. A per-hop perturbation σ mimics an API's per-call drift, over 15-hop walks, 100 seeds, 30 perturbed runs each, with σ set to the largest value that still looks reproducible to the standard check (recall@10 ≥ 0.98). At that point (σ = 1e-3, recall@10 = 0.988), 11.5% of trajectories diverge from the deterministic reference by depth 15, growing monotonically with depth (0.002 at depth 1): an 8.6x amplification of the per-hop top-1 flip rate. At σ = 0 divergence is exactly 0 at every depth, so all of it is attributable to per-call drift.

**Decoding.** The same protocol one layer up, quantizing the full-vocabulary logit vector at each greedy step (48 prompts, ~2,265 scored steps each, fp32 reference on the same GPU):

| model | condition | token agreement | HER | generations identical |
| --- | --- | ---: | ---: | ---: |
| Llama-3.1-8B | TF32 off (control) | 100.00% | 1.0000 | 48/48 |
| Llama-3.1-8B | **TF32 on** | 100.00% | 0.0000 | 48/48 |
| Llama-3.1-8B | **bf16** | 99.21% | 0.0000 | **38/48** |
| Qwen2.5-7B | **bf16** | 99.21% | 0.0000 | **31/48** |
| Mistral-7B-v0.3 | **bf16** | 99.52% | 0.0000 | **38/48** |

Two readings. A serving change can be total at the instrument and invisible at the output: under TF32 the code changes at every step while every token and all 48 complete generations stay character-identical, so an output-comparison monitor sees nothing. And small per-token loss is large per-outcome loss: 0.48-0.79% of tokens flipped under bf16 becomes 20.8-35.4% of complete generations changed, a 26x to 45x amplification, because one flipped token derails everything after it. Controls hold on all three models.

### 3.4 At the outcome layer, the harness is part of the system

ARI-E applies the same discipline — repeat, control, attribute — to agent outcomes, using `nvidia/Open-SWE-Traces`: two scaffolds (SWE-agent, OpenHands) crossed with two models over ~18,000 SWE-bench-style instances, graded by each repository's own test suite. Agents disagree with themselves on 10-13% of repeated rollouts, and that self-consistency control is the point: a naive diff of two scaffolds under Qwen3.5-122B reads a 21.5-point outcome difference, while the attributable scaffold effect after the control is +8.9 points [+8.3, +9.4]. Across all four contrasts the control removes 59-74% of the apparent effect, and the mean scaffold effect (+0.069) exceeds the mean model effect (+0.050). A benchmark delta measured without a self-consistency baseline overstates the cause under test by 2x or more.

**Limitations.** Every decoding run is greedy, which isolates infrastructure by removing sampling noise but leaves open whether the protocol separates infrastructure drift from sampling above temperature 0. The decoding conditions are serving changes we applied on one machine, not observations of a provider changing something. And reports are attested by the leaderboard operator's own key, so attestation proves integrity and origin without making the operator neutral. Appendix B lists the rest.

## 4. Conclusion

Reproducibility of a deployed ML system has no defined measurement, and the metrics operators watch cannot supply one, since they are threshold detectors reading a signal below their resolution. Measuring internal state with a deterministic quantizer, under single-axis perturbations, on a frozen input set, yields a number that is comparable across providers, verifiable by a third party, and usable on systems that are otherwise opaque. Under that protocol five commercial APIs span 0.169 to 1.000; a precision change rewrites half to all of a corpus's internal codes with no detectable movement in retrieval quality; and drift too small to see per step becomes 11.5% of retrieval trajectories and up to 35% of complete generations. The compounding is why the first two numbers matter.

---

## References

<!-- TODO: convert to the workshop's citation format; several entries need full
     bibliographic details before submission. -->

1. ReproRAG: reproducibility of retrieval-augmented generation embeddings. arXiv:2509.18869.
2. EU Artificial Intelligence Act, Article 15.
3. NIST AI Risk Management Framework, Manage 4.1.
4. US Treasury AI Risk Management Framework, February 2026.
5. Thakur, N. et al. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.
6. Muennighoff, N. et al. MTEB: Massive Text Embedding Benchmark.
7. Defeating Nondeterminism in LLM Inference (batch-invariant kernels).
8. openai/openai-python issue #868, community-reported nondeterministic embeddings since 2023.
9. NVIDIA. Open-SWE-Traces dataset (Hugging Face).
10. The SEMQ Group. ARI-Bench-v0.1 and the ARI reference implementation. Content hash `e9ec8b01c62635de…`.

---

## Appendix A. What our own testing took away

Three earlier claims did not survive adversarial testing, and the corrections bound what the index needs. At the decoding layer, an 8-byte top-2 logit margin recovers 86% of the quantizer's signal for 1/2000th of the stored reference state and is equally bit-reproducible; the quantizer's remaining advantage there is coverage of drift below the top ranks, demonstrated for token suppression (28.7x enrichment at ranks 3-10, with 100.00% token agreement, so output monitoring is blind to it) and unproven for serving changes generally. A frozen product-quantizer codebook also works as a probe: on every real k-means codebook we swept, `encode(reconstruct(c)) = c` held at every setting, including one with 91% of assignments within 1e-6 of a tie. What quantile binning offers is a verification-cost difference — a guarantee checkable from the operator's form in advance, and one scalar of calibration state instead of thousands of floats — and that is an argument about cost, not an impossibility result. And ARI-E uses no quantizer at all, since nothing in a verdict comparison is a vector.

## Appendix B. Remaining limitations

`mach` and `lib` are unmeasured for the self-hosted panel; one provider's `conc` ran at burst 16 rather than 64 due to rate limits; GPU coverage is two architectures. Drift rates are input-distribution dependent, which is exactly why the input set is frozen and hash-pinned. About 22% of ARI-E rows were ungraded and dropped; if grading failure tracks task difficulty, the surviving set is easier than the whole.

## Appendix C. Decoding controls

The model whose vocabulary fits one probe context behaves like the two that require chunking, so the chunked probe path introduces no artifact.

## Appendix D. Probe sensitivity spread

Measured influence spread is 1.00 for quantile binning versus about 16 for a block-microscaled format like NVFP4, whose shared per-block scale smears one perturbed dimension across its whole block. A diagonal sensitivity Jacobian is what lets drift attribute back to specific dimensions.

## Reproducibility statement

The specification (probe, calibration rule, condition set, report schema, per-model fingerprints), the frozen 1,000-item input set with its content hash, the reference implementation, the independent verifier, and every experiment's protocol with machine-readable results are in the accompanying repository; each experiment directory carries a reproduce command. Figures are generated from the experiments' own JSON output, so no figure recomputes a number independently of the table it illustrates. Code is Apache-2.0; the specification and input set are CC-BY-4.0, with BEIR text retaining its original licenses.
