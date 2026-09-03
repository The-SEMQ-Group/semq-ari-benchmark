# Measuring Reproducibility in Deployed AI Systems

The SEMQ Group · Workshop submission · August 2026

---

## Abstract

Reproducibility is named as a required property of AI systems by the EU AI Act (Art. 15), the NIST AI RMF (Manage 4.1) and the US Treasury AI RMF, and none of them defines how to measure it on a deployed system. The measurement practitioners reach for — aggregate output quality — cannot answer the question, because it is a threshold detector reading a signal several orders of magnitude below its resolution. We propose measuring reproducibility on the system's internal state instead: quantize the representation with a deterministic discretizer, then report hash-equality rate (HER) and mean Hamming drift under a published set of controlled environmental perturbations. The protocol is defined on a frozen, hash-pinned input set and emits a report a third party can re-verify without the operator's infrastructure. We report three results. Five commercial embedding APIs span **0.169 to 1.000** under an identical protocol, making reproducibility a provider-specific engineering property rather than a uniform property of hosted inference. A precision or serving change moves **47–100% of internal codes while Recall@10 changes by an amount a paired bootstrap cannot distinguish from zero**. And divergence that looks negligible per step compounds: **11.5%** of 15-hop retrieval trajectories diverge at a per-call drift where the standard check reads recall@10 = 0.988, and a 0.48–0.79% per-token divergence under bf16 becomes a **20.8–35.4%** chance of a different complete generation.

---

## 1. The problem

A deployed inference system is a deep stack of floating-point operations. Non-determinism enters it through BLAS reduction order under multi-threading, CUDA atomics in scatter operations, mixed precision (fp16/bf16/TF32), library and kernel patch versions, hardware heterogeneity, and decoding-level effects such as batch composition and tie-breaking. Each contributes a perturbation of roughly 10⁻⁷ to 10⁻⁴ per representation component.

Operators do not currently detect these events, and the reason is structural rather than a matter of insufficient monitoring. For unit-normalised embeddings with minimum score gap `g_min`, a Gaussian perturbation of magnitude σ flips a top-K ranking with probability ≈ `2K·Φ(−g_min/σ)`. At typical retrieval values (`g_min ~ 10⁻²`, `σ ~ 10⁻⁶`) that is `Φ(−10⁴) ≈ 0`. Float arithmetic plus cosine ranking is a **low-pass filter**: sub-microsigma perturbations are invisible to top-K by construction, so a quality metric computed on top of them is a binary threshold detector, not a continuous instrument.

The practical consequence is a class of silent failure. A model is audited in one configuration and served in another; the endpoint returns 200, the quality dashboard stays flat, and the internal state the system stores, compares and caches is not the state that was audited. Providers of hosted inference make this worse by being black boxes: the caller cannot set precision or determinism, so the drift is neither observable nor fixable from outside.

**What is and is not new here.** The phenomena are largely documented. Non-deterministic hosted embedding APIs have been reported by users since 2023 [8]; TF32's numerics are documented and it is off by default in current PyTorch; prior work benchmarks embedding reproducibility across precisions and finds embeddings bit-identical at framework defaults [1]. The contribution is the **measurement**: a standardized, comparable, third-party-verifiable number, on a frozen input set, with a bit-exact metric and bootstrap intervals, for systems that are opaque to their callers.

## 2. Method

**Definitions.** Let the system under test be `A: X → V`, mapping inputs to a vector-valued internal representation (an embedding, or a per-step logit vector). An **environmental condition** is a tuple `E = (process, machine, precision, library version, concurrency, time)`. A baseline run under a reference condition `E₀` fixes all of them; each comparison condition perturbs exactly one axis, so drift is attributable to that axis.

**The probe.** Let `Q` be a deterministic quantizer and `P = Q ∘ A`. We use quantile binning at `n_bins = 2`, calibrated by a **single scalar** `s` fixed at the 99th percentile of a published reference distribution. Three properties matter and none of them is retrieval quality. `Q` must be bit-deterministic, so that any observed difference belongs to the subject rather than the instrument. Its response must be *linear* in the perturbation near zero, so it resolves the small-σ regime rather than cliffing. And its sensitivity Jacobian `∂code/∂x` should be **diagonal**, so drift can be attributed back to specific dimensions — measured influence spread is 1.00 for this quantizer against ≈16 for a block-microscaled format such as NVFP4, whose shared per-block scale smears a single perturbed dimension across its whole block (attribution F1 1.000 vs 0.125–0.160 across five encoders).

The response follows a power law, `E[Hamming] = κ · √(2·dim/π) · σ`, with slope `b = 0.979 ± 0.028` across 13 architectures — the instrument's response depends on the operator, not the model. The per-model prefactor κ ∈ [1.48, 3.06] is an angular-concentration property of the embedding manifold, stable across corpora. Publishing `(s, b, κ)` once per model gives the protocol a **quantitative null hypothesis**: observed drift above the prediction is attributable to a downstream noise source.

**The metric.** Per input `x` and condition `Eₖ`, the reading is bit-exact equality of codes and, where they differ, their Hamming distance:

- **HER** (hash-equality rate) `= mean_x 1[P₀(x) = Pₖ(x)]` ∈ [0,1]
- **H̄** (mean Hamming drift) `= mean_x H(P₀(x), Pₖ(x))`, in bits

Confidence intervals are a block bootstrap over inputs. The headline index averages HER over a **comparable core** `{proc, conc, time}` — the conditions measurable for any system class. A hosted API cannot observe `mach`, `prec` or `lib`, so folding those in would penalise a self-hosted system that honestly exposes them relative to an API that merely hides them; they are reported as diagnostics instead. One consequence should be stated plainly: an API's headline is a **measured lower bound** on its true drift.

| id | perturbs | isolates |
| --- | --- | --- |
| `same` | nothing; immediate repeat | probe purity (self-hosted) / within-session determinism (API) |
| `proc` | fresh process | thread schedule, per-process kernel autotuning |
| `conc` | concurrent load | request routing under burst |
| `time` | wall-clock gap > 24h | silent serving change |
| `prec` | fp32 → bf16/fp16 | precision-induced drift *(diagnostic)* |
| `mach`, `lib` | host, library patch version | hardware and version drift *(diagnostics)* |

**Verifiability.** Inputs are a frozen, hash-pinned slice of BEIR — 1,000 items across medical, scientific and financial domains — drawn from a standard corpus rather than authored, so no reader can object that they were chosen to flatter or penalise a system. Each report carries per-condition HER, H̄ and CI, an environment manifest, and a **SHA-256 audit digest per condition** over the code matrix in fixed input order. Attestation signs a manifest binding report and inputs together, so changing either breaks the check, and the verifier runs on the standard library without importing the publisher's code — a check that needs the publisher's software is not independent.

## 3. Results

### 3.1 Reproducibility is provider-specific, and spans the full range

Five commercial embedding APIs, measured under one protocol on the frozen input set, n = 1,000, `time` at a canonical ~76h gap and `conc` as a 64-way burst against a calm pass:

| provider / model | `same` | `proc` | `conc` | `time` | **index** | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | :--- |
| `gemini/gemini-embedding-001` | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | [1.000, 1.000] |
| `openai/text-embedding-3-large` | 0.859 | 0.862 | 0.851 | 0.822 | **0.845** | [0.822, 0.866] |
| `mistral/mistral-embed` | 0.734 | 0.753 | 0.791 | **0.554** | **0.699** | [0.672, 0.727] |
| `voyage/voyage-4-large` | 0.571 | 0.615 | **0.209** | 0.676 | **0.500** | [0.472, 0.528] |
| `cohere/embed-v4.0` | 0.133 | 0.146 | 0.130 | 0.232 | **0.169** | [0.147, 0.194] |

The spread is the result. One provider is bit-reproducible and another drifts on roughly 85% of repeated calls, under identical inputs and an identical protocol — so reproducibility is a property of each serving stack rather than an inherent cost of hosted inference.

Three checks support reading it that way. **The 1.000 is not a cache.** We built 150 novel inputs by appending a unique nonce to each text, so no lookup cache can help; that provider read 1.000 on them while a control provider read 0.807 on the *same* novel inputs, and the ~76h `time` condition exceeds any plausible result-cache TTL. **The drift is genuine, verified below the probe.** Thirty texts embedded in two separated passes gave 3 of 30 differing at the float level, and where they differed, ~2,300 of 3,072 dimensions changed at magnitude ~1e-4 — a different replica's vector, not last-bit noise. **Individual axes carry distinct failures.** Only `conc` exposes one provider's collapse from 0.615 to 0.209 under burst, as concurrent requests reach backends that disagree; only a multi-day `time` gap exposes another's fall to 0.554, invisible at a 19h gap.

For contrast, eight self-hosted open models read 1.000 across `proc`, `conc` and `time`, including with BLAS threads unpinned under multi-threaded OpenBLAS — but every one collapses to HER ≤ 0.094 under a bf16 serving change. The defensible claim is *self-host and pin the precision*, not that self-hosting is inherently safe.

### 3.2 Quality metrics are statistically blind to large internal drift

BEIR SciFact, 5,183 documents, 300 queries with real relevance judgements, `all-MiniLM-L6-v2`. The reference is fp32/CPU/4-thread/batch-32; every other condition changes exactly one thing in a fresh process. Intervals are a paired bootstrap over queries, 10,000 resamples.

| condition | ΔRecall@10 | 95% CI | significant | top-10 lists identical | HER |
| --- | ---: | :--- | :---: | ---: | ---: |
| new process / 1 thread / batch 8 / batch 128 | +0.0000 | [+0.0000, +0.0000] | no | 100.00% | 1.0000 |
| GPU fp32, **TF32 off** | +0.0000 | [+0.0000, +0.0000] | no | 100.00% | 0.9992 |
| GPU fp32, **TF32 on** | +0.0000 | [+0.0000, +0.0000] | no | 96.33% | **0.5267** |
| **bf16** | +0.0033 | [+0.0000, +0.0100] | **no** | 45.00% | **0.0000** |
| **int8** | −0.0061 | [−0.0278, +0.0156] | **no** | 0.00% | **0.0000** |

Neither precision change produces a Recall@10 movement distinguishable from zero. int8 moves the embeddings by 6% mean cosine and the *measured* recall falls by 0.6 points — well inside the noise band. Meanwhile HER reads a hard zero: not one document in 5,183 keeps its code.

The four control rows are what make that worth anything. A new process, a single thread and a 4× or 16× batch change move nothing on any panel, so HER = 0 under a precision change is a discrimination rather than an instrument that fires at everything.

The TF32 pair is the operationally important row. **A model audited at fp32 on CPU and served at fp32 on GPU gets a different code for 47% of documents,** with no configuration change visible to the caller; turning TF32 off restores 0.9992 against the same reference, which rules out the GPU and leaves one switch. An isolation 2×2 over TF32 × `use_deterministic_algorithms` confirms the attribution: mean cross-process HER is 1.000 with TF32 off and 0.749 with it on, and the determinism flag has bit-identical effect in both columns — that is, none. The mechanism is that at true fp32 a cross-process cuBLAS kernel difference is ~1e-7 and stays below the quantizer floor, while TF32's ~19-bit mantissa turns the same difference into ~1e-3 and flips the code.

One scope correction belongs here, because it narrows a claim that is often overstated. Quality *metrics* are blind; result *lists* are not — int8 changes the top-10 list for every query and bf16 for 55%. Comparing lists does detect these events, but it requires a stored reference top-*k* per query, covers only the query distribution that was saved, and mixes real drift with tie-order churn. HER is computed corpus-side, needs no queries at all, and is exact at matched calibration.

### 3.3 Small per-step divergence compounds into outcome divergence

The obvious objection is that a single retrieval rarely moves its top-K set, so this drift may not matter. It compounds, and we measure the compounding in two independent settings.

**Iterative retrieval.** A multi-hop agent — a self-avoiding nearest-neighbour walk over NFCorpus (3,633 documents), each hop embedding the current query, retrieving the nearest unvisited document and stepping to it — is an LLM-free but faithful model of an iterative RAG loop. A per-hop query perturbation σ mimics a hosted API's per-call drift; 15-hop walks, 100 seed documents, 30 perturbed runs each. The operating σ is auto-selected as the largest that still *looks* reproducible by the standard check (recall@10 ≥ 0.98).

| σ | recall@10 (standard check) | top-1 flip rate | Hamming | trajectory divergence at depth 15 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1.0000 | 0.0000 | 0.00000 | **0.000** |
| 1e-4 | 0.9986 | 0.0007 | 0.00813 | 0.012 |
| **1e-3** (operating) | **0.9879** | 0.0133 | 0.07776 | **0.115** |
| 3e-3 | 0.9661 | 0.0435 | 0.21941 | 0.392 |

At a per-call drift where the standard check reads recall@10 = 0.988 — a practitioner would call this pipeline reproducible — **11.5% of trajectories diverge** from the deterministic reference, growing monotonically with depth (0.002 at depth 1 → 0.115 at depth 15). That is an 8.6× amplification of the per-hop top-1 flip rate. At σ = 0 divergence is exactly 0 at every depth, so all of it is attributable to per-call representation non-determinism.

**Decoding.** The same protocol applied one layer up, quantizing the full-vocabulary logit vector at each greedy decoding step, on three production-scale models (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B-v0.3; 48 prompts, ~2,265 scored steps each, fp32 reference on the same GPU):

| model | condition | token agreement | HER | complete generations identical | 95% CI |
| --- | --- | ---: | ---: | ---: | :--- |
| Llama-3.1-8B | TF32 off (control) | 100.00% | 1.0000 | 48/48 | [93%, 100%] |
| Llama-3.1-8B | **TF32 on** | **100.00%** | 0.0000 | **48/48** | [93%, 100%] |
| Llama-3.1-8B | **bf16** | 99.21% | 0.0000 | **38/48** | [66%, 88%] |
| Qwen2.5-7B | **bf16** | 99.21% | 0.0000 | **31/48** | [50%, 77%] |
| Mistral-7B-v0.3 | **bf16** | 99.52% | 0.0000 | **38/48** | [66%, 88%] |

Two readings. First, **a serving change can be total at the instrument and invisible at the output**: under TF32 the code changes at every step while every token and all 48 complete generations stay character-identical, so a monitor comparing outputs sees nothing. Second, **small per-token loss is not a small problem**: 0.48–0.79% of tokens lost under bf16 becomes **20.8–35.4% of complete generations lost**, an amplification of 26× to 45×, because one flipped token derails every token after it. Controls hold on all three models (fresh process, single thread, TF32 off → HER 1.0000, 48/48), and the model whose vocabulary fits a single probe context behaves like the two that require chunking, so the chunked path introduces no artifact.

The two settings agree in shape by different routes: divergence that is negligible per step is macroscopic per outcome in any system that iterates.

## 4. Discussion and limitations

**What the measurement is.** This is a *system* property — model plus BLAS plus threads plus hardware plus libraries — not a property of a model in isolation, which is why every report must carry an environment manifest. It does not detect semantic drift: a model swap, a fine-tune, or a prompt change all leave it silent.

**What we are not claiming about the instrument.** Quantile binning is not the only probe that could work here. A frozen, published product-quantizer codebook is bit-deterministic and, on every real k-means codebook we swept (5,000 SciFact abstracts, 256 centroids per subspace, five configurations), satisfies `encode(reconstruct(c)) = c` at every setting — including one with 91% of assignments within 1e-6 of a tie; the residual exposure is that two algebraically identical distance formulations disagree on 0% to ~5% of vectors. What quantile binning offers is a difference in how the guarantee is *obtained* — checkable from the operator's form in advance, against re-measured per shipped codebook — plus a one-scalar calibration state rather than thousands of floats to version and pin. That is an argument about verification cost, not an impossibility result. At the decoding layer an 8-byte top-2 logit margin recovers 86% of the signal for 1/2000th of the stored reference state and is equally bit-reproducible; the quantizer's remaining advantage there is coverage of drift below the top ranks, demonstrated for token suppression but not for serving changes generally.

**Coverage and scope.** `mach` and `lib` are unmeasured for the self-hosted panel; one provider's `conc` was measured at burst 16 rather than 64 owing to rate limits; GPU scope is two architectures at n = 512. Every decoding run is greedy, which isolates infrastructure by removing sampling noise but leaves untested whether the protocol separates infrastructure drift from sampling above temperature 0. The decoding runs are simulated serving changes on one machine rather than observations of a provider changing something. Rates are input-distribution dependent, which is precisely why the input set is frozen and hash-pinned. And the whole panel currently reads *self-reported*: the reports carry audit digests and input pins, but no durable signing key is registered, so none is attested.

## 5. Conclusion

Reproducibility is a required property of AI systems with no defined measurement, and the metrics operators watch cannot supply one — not because they are monitored badly but because they are threshold detectors reading a signal below their resolution. Measuring the internal state with a deterministic discretizer, under controlled single-axis environmental perturbations, on a frozen input set, produces a number that is comparable across providers, verifiable by a third party, and decision-grade for systems that are otherwise opaque. Under that protocol five commercial APIs span 0.169 to 1.000; a precision change moves half to all of a corpus's internal codes with no statistically detectable movement in retrieval quality; and divergence too small to see per step becomes 11.5% of retrieval trajectories and up to 35% of complete generations. The last of those is the reason the first two matter.

---

## References

1. ReproRAG: reproducibility of retrieval-augmented generation embeddings. arXiv:2509.18869.
2. EU Artificial Intelligence Act, Article 15 — accuracy, robustness and cybersecurity of high-risk AI systems.
3. NIST AI Risk Management Framework, Manage 4.1.
4. US Treasury AI Risk Management Framework, February 2026.
5. Thakur, N. et al. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.
6. Muennighoff, N. et al. MTEB: Massive Text Embedding Benchmark.
7. Defeating Nondeterminism in LLM Inference — batch-invariant kernels at roughly 60% throughput cost.
8. openai/openai-python issue #868 — non-deterministic embeddings, community-reported since 2023.
9. QwenLM/qwen-code PR #348 — precision pinning does not route as documented.
10. Dettmers, T. et al. QLoRA: Efficient Finetuning of Quantized LLMs (NF4).
11. The SEMQ Group. Canonical Quantization for Reproducible Semantic Retrieval (draft).
12. The SEMQ Group. ARI-Bench-v0.1 and the ARI reference implementation. Content hash `e9ec8b01c62635de…`.

## Reproducibility statement

The specification (probe, calibration rule, condition set, report schema, per-model fingerprints), the frozen 1,000-item input set with its content hash, the reference implementation, the independent verifier, and every experiment's protocol with its machine-readable results are in the accompanying repository. Each experiment directory carries a reproduce command. Figures are generated from the experiments' own JSON output, so no figure recomputes a number independently of the table it illustrates. Code is Apache-2.0; the specification and input set are CC-BY-4.0, with BEIR text retaining its original licenses.
