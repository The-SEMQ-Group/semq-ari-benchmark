# Measuring What Regulators Require: The Agent Reproducibility Index

**A three-layer, third-party-verifiable index of AI agent reproducibility, with a 13-model cross-provider panel**

The SEMQ Group · Draft, August 2026 · ARI v0.1-preview

---

## Abstract

Reproducibility is named as a required property of high-risk AI systems by the EU AI Act (Art. 15), the NIST AI RMF (Manage 4.1) and the US Treasury AI RMF. None of them defines how to measure it on a deployed system. Capability leaderboards measure what a model can do; none measure whether it does the same thing twice.

We define the **Agent Reproducibility Index (ARI)**: a protocol that measures how stably an agent's behaviour survives the operational noise of real deployment, and emits a report a third party can re-verify without access to the vendor's infrastructure. ARI decomposes into three layers — **ARI-R** (representation), **ARI-D** (decoding), and **ARI-E** (environment/harness).

The instrument at the R and D layers is SEMQ, a deterministic quantizer that turns sub-microsigma floating-point perturbations into a discrete Hamming signal. Its response follows a universal power law, `Hamming = a·σ^b` with `b = 0.979 ± 0.028` across 13 architectures, and each model carries a three-scalar fingerprint `(s, b, κ)` that predicts its drift without re-running the sweep.

We report four sets of measurements. **(1)** A deployed-agent panel of 13 embedding models — 5 commercial APIs and 8 self-hosted open models — on a frozen 1,000-item BEIR slice. The paid APIs span **ARI 0.169 to 1.000**: Gemini is bit-reproducible, Cohere `embed-v4.0` drifts on ~85% of repeated calls. This is, to our knowledge, the first standardized, comparable, third-party-verifiable reproducibility measurement across providers. **(2)** At the representation layer, a precision or TF32 change moves 47–100% of codes while Recall@10 moves by an amount a bootstrap cannot distinguish from zero. **(3)** At the decoding layer, on Llama-3.1-8B, Qwen2.5-7B and Mistral-7B, enabling TF32 changes the instrument's reading at **100% of steps while every token and every complete generation stays identical**; and a sub-1% per-step divergence under bf16 becomes a **20.8–35.4%** chance of a different complete answer — an amplification of 26× to 45×. **(4)** At the harness layer, over ~23,000 graded agent rollouts, the scaffold decides more of the outcome than the model does (+0.069 vs +0.050 mean effect), and a naive scaffold comparison that omits the self-consistency control overstates the effect by 2.4×.

We also report the measurements that go against our own instrument. An 8-byte top-2 margin statistic recovers 86% of SEMQ's decoding-layer signal for 1/2000th of the storage; a frozen product quantizer is a viable probe on every real codebook we tested; and the harness layer needs no quantizer at all. We therefore state the instrument claim narrowly — a class argument, a universal law, a decomposable sensitivity Jacobian, and a verification-cost preference, not an impossibility result.

---

## 1. Introduction

### 1.1 A required property with no defined measurement

A modern AI agent is a deep stack of floating-point operations. Non-determinism enters it through at least six channels:

1. **BLAS reduction order** — multi-threaded matmul over non-associative float addition.
2. **CUDA atomics** — warp scheduling in scatter operations.
3. **Mixed precision** — fp16/bf16/TF32 dynamic-range reductions.
4. **Library drift** — framework and kernel patch versions.
5. **Hardware heterogeneity** — GPU and CPU models, BLAS implementations.
6. **Decoding-level effects** — top-k tie-breaking, KV-cache layout, batch composition.

Each contributes a perturbation of roughly **10⁻⁷ to 10⁻⁴** per embedding component. Three regulatory frameworks require reproducibility of such systems and none says how to measure it. The measurement everyone reaches for — "does the agent return the same results?" — is blind to perturbations of that size by construction (§3).

### 1.2 Contributions

1. **An operational definition and protocol.** ARI: hash-equality rate (HER) and mean Hamming drift (H̄) of a deterministic discrete code, over a published environmental condition set, aggregated on a *comparable core* that keeps black-box APIs and self-hosted models apples-to-apples (§5).
2. **A three-layer decomposition.** ARI-R, ARI-D, ARI-E — separating what can be measured exactly from what is larger and harder, and making the uncovered layers a roadmap rather than a hole in the title (§5.1).
3. **A verifiable report artifact.** Every report is pinned to a hash-frozen input set, carries a per-condition SHA-256 audit digest, and can be attested by signing a manifest that binds report and inputs together. Verification runs on the standard library and does not import our code (§5.5).
4. **A cross-provider panel.** 13 embedding models on a frozen 1,000-item BEIR slice, with bootstrap confidence intervals (§6).
5. **A universal instrument law and a per-model fingerprint.** `(s, b, κ)`, published for 13 models, predictive of drift at any σ (§4.2).
6. **Layer-D and layer-E measurements** on production-scale models and on ~23,000 real agent rollouts (§7, §8).
7. **Two negative results against our own instrument**, reported in full rather than omitted (§7.3, §10).

### 1.3 What is and is not novel

The *phenomena* here are largely known. That hosted embedding APIs return different vectors for identical input has been community-documented since 2023 [11]. That TF32 changes numerical results is documented, and TF32 is off by default in modern PyTorch. That agent harnesses cause variance has two preprints in one quarter [7, 8].

What is new is the **measurement**: a standardized, comparable, third-party-verifiable number, on a frozen input set, with a bit-exact metric and confidence intervals, for systems that are black boxes to their callers. This is an MTEB-style contribution — not discovering a phenomenon, but making it the reference measurement.

---

## 2. Related work

**Capability leaderboards.** MTEB, HELM and AgentBench measure competence. Reproducibility is an orthogonal axis, and none of them reports it. ARI is designed as the companion axis: for each model on MTEB, an ARI on the same class of data.

**Embedding reproducibility.** ReproRAG [1] benchmarks embedding reproducibility across fp32/fp16/bf16/tf32 × determinism flags on BGE/E5/Qwen and reports embeddings bit-identical (L2 = 0.0). We reconcile rather than contradict: they measured same-GPU, same-config, at framework defaults, where we also read 1.000. The axes where drift appears — CPU↔GPU, and TF32-on cross-process — were not in their grid, and a raw-L2 metric reading "5.74e-4, reproducible" is exactly where a discrete-attractor metric reads a flipped code (§6.3).

**Inference non-determinism.** Public work on batch-invariant kernels shows temperature-0 determinism is recoverable at roughly a 60% throughput cost [10], which makes "are you paying for determinism, and did you get it?" a decision worth instrumenting. Provider routing surfaces exist for this — OpenRouter exposes a `quantizations` field and provider blacklisting — but the guidance for confirming they work is behavioural A/B testing, and one attempt to pin precision found the field did not route as documented [12]. There is a control surface with no verifier.

**Harness variance.** Zhang et al. [7] argue the harness "is often a stronger determinant of agent performance than the model it wraps", including cases of model ranking reversal. Harness-Bench [8] evaluates 5,194 trajectories over 106 sandboxed tasks and concludes capability should be reported at the model-harness configuration level. HAL [9] built standardized infrastructure over 21,730 rollouts explicitly to separate models from scaffolds. None defines an index. ARI-E (§8) proposes one and measures it with a self-consistency control that, as far as we can tell, the existing comparisons omit.

**Quantization as an instrument.** SEMQ's origin is a retrieval codec [14]. The instrument use here is distinct: we do not care that codes preserve nDCG, we care that they are a deterministic, decomposable function of the vector.

---

## 3. Why the obvious instrument fails

### 3.1 The analytical argument

For unit-normalised embeddings with minimum score gap `g_min`, a Gaussian perturbation of magnitude σ flips a top-K ranking with probability ≈ `2K·Φ(−g_min/σ)`. For typical retrieval (`g_min ~ 10⁻²`, `σ ~ 10⁻⁶`) that is `Φ(−10⁴) ≈ 0`.

> **FP32 + cosine retrieval is a low-pass filter.** Sub-microsigma perturbations are structurally invisible to top-K ranking. It is a binary threshold detector, not a continuous instrument.

| Representation | Observability vs σ | Class |
| --- | --- | --- |
| FP32 cosine ranking | `~ Φ(−g_min/σ)` — decays super-exponentially | low-pass |
| SEMQ Hamming | `~ dim·σ/s` — linear in σ | high-pass |

### 3.2 The empirical argument, with real relevance judgements

We measured this rather than asserting it. BEIR SciFact, 5,183 documents, 300 queries with real qrels, `all-MiniLM-L6-v2`, fp32/CPU/4-thread/batch-32 reference, one variable changed per condition in a fresh process. Confidence intervals are a paired bootstrap over queries, 10,000 resamples. Real qrels matter: on a clean corpus Recall@10 pins at 1.0 and any comparison measures that degeneracy instead of the instrument.

| condition | Recall@10 | ΔR@10 | 95% CI | significant | top-10 lists identical | SEMQ HER |
| --- | ---: | ---: | :--- | :---: | ---: | ---: |
| reference | 0.7833 | — | — | — | 100.00% | 1.0000 |
| new process | 0.7833 | +0.0000 | [+0.0000, +0.0000] | no | 100.00% | 1.0000 |
| 1 thread | 0.7833 | +0.0000 | [+0.0000, +0.0000] | no | 100.00% | 1.0000 |
| batch 8 | 0.7833 | +0.0000 | [+0.0000, +0.0000] | no | 100.00% | 1.0000 |
| batch 128 | 0.7833 | +0.0000 | [+0.0000, +0.0000] | no | 100.00% | 1.0000 |
| **bf16** | 0.7867 | **+0.0033** | [+0.0000, +0.0100] | **no** | **45.00%** | **0.0000** |
| **int8** | 0.7772 | **−0.0061** | [−0.0278, +0.0156] | **no** | **0.00%** | **0.0000** |

Adding the GPU arm from a separate run on the same corpus and encoder:

| condition against fp32/CPU | ΔRecall@10 | top-10 lists same | SEMQ HER |
| --- | ---: | ---: | ---: |
| GPU fp32, **TF32 off** | +0.0000 | 100.00% | **0.9992** |
| GPU fp32, **TF32 on** | +0.0000 | 96.33% | **0.5267** |

Four readings follow.

1. **Aggregate retrieval quality is blind to both precision changes.** Neither is distinguishable from zero at 300 queries. int8 moves the embeddings by 6% mean cosine and the measured recall *falls* by 0.6 points — well inside the noise band. A team watching a recall dashboard sees nothing.
2. **The controls are what make that worth anything.** A new process, a single thread and a 4× or 16× batch change move nothing on any panel. An instrument that fires at everything detects nothing.
3. **A model audited at fp32/CPU and served at fp32/GPU gets a different code for 47% of documents,** and turning TF32 off restores 0.9992 against the same reference — which rules out "the GPU" and leaves one switch.
4. **One published claim needs narrowing.** "Invisible to cosine/retrieval" is true of retrieval *quality metrics* and false of retrieval *result lists*: int8 changes the top-10 list for 100% of queries and bf16 for 55%. Comparing result lists does detect these events. The advantage that survives is narrower and more practical: list comparison needs a stored reference top-*k* per query and covers only the query distribution you saved; HER is computed corpus-side and needs no queries at all. Quality metrics, the thing teams actually alert on, are blind either way.

A fifth reading is an incidental finding worth recording: bf16 shows **mean cosine 1.001052** against unit-normalised references, because the bf16 model's own normalisation is imprecise enough that its outputs are no longer unit vectors. A monitor comparing cosine to a stored reference would have to notice a similarity slightly greater than 1 to catch this, and most clamp or ignore it.

---

## 4. The instrument

### 4.1 SEMQ as a high-pass probe

SEMQ is a deterministic quantizer. The ARI canonical probe is **QBIN with `n_bins = 2`**, calibrated by a single scalar `s` fixed at the 99th percentile of a published reference distribution. Codes are bit-packed; drift between a baseline code and a condition code is the popcount of their XOR.

In the linear regime,

```
E[Hamming] = κ · √(2·dim/π) · σ
```

equivalently `E[Hamming] = 4·dim·σ / (s·√(2π))`. The two forms are algebraically identical; κ makes the geometry explicit and comparable across models.

QBIN n=2 is canonical for four reasons. It is the only SEMQ operator with a **closed-form** drift law, which is what makes the fingerprint predictive rather than a curve fit. n=2 is the setting where that form is exact — the distance from a component to its nearest bin edge is uniform on `[0, s/2]`. The bin edge sits at the calibration threshold, so sensitivity is maximal. And the frozen state is **one scalar**, which a report can quote inline and a verifier can check by eye.

### 4.2 The universal law and the `(s, b, κ)` fingerprint

Primary sweep: 6 models (BERT-style 1024-d encoders, 7B decoder embedders, a closed-source API) × 11 σ on BEIR MS MARCO, plus a 132-cell corpus extension on NFCorpus and SciFact. The registry has since been extended to 13 models on the frozen ARI-Bench.

Three hypotheses were pre-registered before any data existed.

| # | Hypothesis | Gate | Result | Verdict |
| --- | --- | --- | --- | --- |
| H1 | SEMQ is a far more sensitive σ-detector than retrieval | Hamming slope ÷ Recall@10 slope ≥ 100× | min ratio 1.1 | **refuted (gate artifact)** |
| H2 | The curve matches first-principles theory | `(b, a)` within 5% of `(1, √(2·dim/π))` | slope matches; prefactor 1.65–2.89× | **refuted (prefactor only)** |
| H3 | The form is architecture-invariant | `std(b) ≤ 0.05` | std = 0.026 | **confirmed** |

**H1's gate was mis-designed, and we say so rather than restating it.** It is a ratio of two slopes, and Recall@10 is degenerate on a clean corpus — pinned at 1.0 for some models (ratio → ∞) and barely moving for others (ratio → ~1). Per-model ratios came out 1.1, 1.4, 3.9, 4.2 and ∞. The gate measures recall's degeneracy, not SEMQ's sensitivity. The robust phenomenon is **categorical**, and a slope ratio cannot express it.

**H2 fails only on the prefactor, and that failure is the discovery.** The slope matches theory (max error 0.068; 5 of 6 models within 4% of `b = 1`). The empirical prefactor exceeds the uniform-on-sphere prediction by a per-model factor. That factor is a real, measurable geometry property: κ, the angular-concentration multiplier. Theory assumes embeddings are uniform on the unit sphere; real embeddings concentrate.

Example response curve (`BAAI/bge-large-en-v1.5`):

| σ | SEMQ Hamming | Recall@10 |
| --- | --- | --- |
| 1e-7 | 9.4e-6 | 1.000 |
| **1e-6 (BLAS-typical)** | **8.0e-5** | **1.000** |
| 1e-5 | 7.9e-4 | 1.000 |
| 1e-3 | 7.6e-2 | 0.981 |
| 1e-2 | 5.8e-1 | 0.819 |

At the σ where production non-determinism actually lives, SEMQ shows a clean signal an order of magnitude above its noise floor while Recall@10 reads exactly 1.000.

Across the 13-model registry, `b = 0.979 ± 0.028` and `κ ∈ [1.48, 3.06]`. κ is corpus-stable: the 7B decoder embedders lock it to within a few percent across corpora, and the BERT family shows 8–12% spread inside its per-model band. `s` is corpus-invariant to <3% across MS MARCO, FiQA-2018 and Natural Questions. Publishing `(s, b, κ)` once per model lets any team predict expected drift at any σ without re-running the sweep — which gives ARI a **quantitative null hypothesis**: drift above the prediction is attributable to a downstream noise source.

Two of the 13 models, `all-mpnet-base-v2` and `all-MiniLM-L6-v2`, are **`floor_limited`**: roughly 1% of their code bytes flip under sub-ULP perturbation, σ-independent down to σ = 10⁻¹⁰, because those components sit exactly on a quantizer rounding boundary. This is not a low-dimension effect — `nomic-embed-text-v1.5` is also 768-d and fits κ cleanly at slope 0.999. The floor lives only in the synthetic sensitivity sweep, and both models remain bit-reproducible (ARI = 1.000). It does mean the canonical probe must fix **calibration jointly with a reference model**; a bare `n_bins=2` is underspecified.

The full registry is in Appendix A.

### 4.3 Choosing the probe: what is ruled out, and what is only a preference

A reproducibility probe has one hard requirement: **its reading must not depend on the machine or library that produced it.** If the instrument moves, its noise and the signal are inseparable.

It is tempting to turn that requirement into a claim that SEMQ is the *only* probe that meets it. We tested that claim adversarially and it does not hold. What survives is smaller and, we think, true.

**Ruled out: sign-projection LSH.** At matched bitrate it is roughly 14× less responsive to perturbation than SEMQ — an instrument that barely moves when the subject does. *This is the weakest number in the paper: one synthetic run on anisotropic vectors, never replicated. Treat it as indicative.*

**Not ruled out: a frozen product quantizer.** Freeze one codebook, publish it, ship it. Such a probe is bit-deterministic and satisfies `encode(reconstruct(c)) = c` exactly — a centroid's nearest centroid is itself — and at matched bitrate is at least as sensitive as SEMQ. We tested the mechanism that we thought separated them: floating-point reassociation in the nearest-centroid `argmin`. Sweeping real k-means codebooks over 5,000 SciFact abstracts, 256 centroids per subspace:

| dims per subvector | near-ties within 1e-6 | median margin | forms disagree (rows) | `encode(reconstruct(c)) = c` |
| ---: | ---: | ---: | ---: | :--- |
| 12 *(standard)* | 0.045% | 1.87e-03 | 0.00% | holds, both forms |
| 6 | 0.070% | 8.68e-04 | 0.04% | holds, both forms |
| 3 | 0.304% | 2.05e-04 | 0.00% | holds, both forms |
| 2 | 2.62% | 4.50e-05 | 0.00% | holds, both forms |
| 1 *(degenerate)* | 91.2% | 3.22e-07 | 4.64% | holds, both forms |

Near-tie density is real and rises three orders of magnitude as the split gets finer. It breaks nothing. `reconstruct(c)` returns the centroid, whose distance to itself is zero to within rounding, and that stays below the distance to any distinct centroid even when the margin is 3.2e-07. Two centroids 1e-6 apart do not survive a k-means fit — they compete for the same Voronoi mass and merge.

The mechanism *can* be made to bite, and we constructed the case that does it: a codebook pairing every centroid with a near-duplicate placed 1e-6 away makes the two formulations disagree on 39% of symbols and breaks `encode(reconstruct(c)) = c` outright under the expanded form. That result is real for that codebook and says nothing about shipped ones. Generalising from it would be the error the sweep above exists to prevent, and we flag it because it is the failure mode a reader should watch for elsewhere in this paper: an adversarial construction demonstrating that a mechanism exists is not evidence about the artifacts a fitting procedure actually produces.

What remains is small: the direct form `(b − c)²` and the expanded form `‖b‖² − 2b·c + ‖c‖²` disagree on **0% to ~5% of vectors** depending on configuration, roughly 1 in 2,500 at a plausible setting. Same input, same published codebook, same machine, different reading — but rarely.

**SEMQ has no such race.** Its codes come from threshold comparisons against a calibrated scale, not a nearest-centroid contest. There is no pair of candidates to reorder, so the assignment cannot depend on summation order, accumulator width or lane layout.

**The claim, stated precisely.** SEMQ is **not necessary** for ARI. A frozen VQ is a viable probe. What SEMQ offers is a difference in *how the guarantee is obtained*: SEMQ's invariance is checkable from the operator's form before any codebook exists; a frozen VQ's is contingent, varies from clean to ~5% of rows across configurations of the same algorithm on the same data, and nothing in a shipped codebook file announces which regime it is in. Two practical properties compound it: the calibration artifact is one scalar rather than thousands of floats (no supply-chain artifact to version and pin), and the tamper surface is correspondingly smaller — a large codebook can be nudged to flatter a subject in ways hard to detect by inspection; a scalar cannot.

That is an argument about **verification cost**, not an impossibility result.

### 4.4 The separator that does survive: a decomposable Jacobian

One structural differentiator does hold up against a strong modern baseline. NVFP4 (E2M1 elements with a per-16-block E4M3 microscale) is deterministic, ties SEMQ on probe purity (both read 0.000000), and is in fact *more* sensitive in raw Hamming. So determinism is necessary and not sufficient. The separator is the structure of the sensitivity Jacobian `∂code/∂x`, measured on 5 encoders spanning 384-d to 4096-d:

| encoder | dim | influence spread (SEMQ / NVFP4 / int4) | attribution F1 (SEMQ / NVFP4 / int4) |
| --- | ---: | --- | --- |
| bge-large-en-v1.5 | 1024 | 1.00 / 15.88 / 1.00 | 1.000 / 0.154 / 1.000 |
| bge-m3 | 1024 | 1.00 / 15.53 / 1.00 | 1.000 / 0.160 / 1.000 |
| multilingual-e5-large | 1024 | 1.00 / 15.88 / 1.00 | 1.000 / 0.138 / 1.000 |
| e5-mistral-7b-instruct | 4096 | 1.00 / 15.76 / 1.00 | 1.000 / 0.134 / 1.000 |
| gte-Qwen2-7B-instruct | 3584 | 1.00 / 16.00 / 1.00 | 1.000 / 0.125 / 1.000 |

SEMQ's Jacobian is **diagonal**: perturbing one input dimension changes exactly one output symbol. NVFP4's is **block-coupled** — spread ≈ 16, its microscaling block size — so a single perturbed dimension smears across the whole block through the shared scale. `int4_scalar`, a per-dim uniform format, is also diagonal, which isolates the coupling to NVFP4's block microscaling rather than to low-precision numerics generally. Only SEMQ can **attribute** injected drift back to the drifted dimensions (F1 1.000 vs 0.125–0.160).

This matters for a reproducibility index specifically. Detecting that something moved is one thing; saying *what* moved requires a per-dimension response. The structure is identical across BERT encoders and 7B decoder embedders, so it is a property of the quantizer, not of the encoder.

---

## 5. The ARI metric and protocol

### 5.1 The three-layer decomposition

An index named for *agent* reproducibility that measures only embeddings oversells itself, and the parts of the work that are sound get contaminated by the parts that are not. We therefore define ARI as a tuple.

| index | question | instrument | status |
| --- | --- | --- | --- |
| **ARI-R** | Does the same input produce the same internal representation across environments? | SEMQ over embeddings; HER + H̄ | measured, 13 deployed agents |
| **ARI-D** | Given the same representation, does the same token sequence come out? | SEMQ over per-step logits; **H̄** | measured, 5 models |
| **ARI-E** | Holding the model fixed, how much does the outcome depend on the scaffold? | outcome agreement with a self-consistency control; **no quantizer** | measured, ~23,000 rollouts |

**Why R is worth publishing even though E is probably larger.** At the D and E layers, stochasticity is *intended*. Temperature is a product decision, tool results depend on the network and the clock, a harness is supposed to make choices. A non-zero reading there is ambiguous by construction. At the representation layer, **any** variation is unintended — nobody ships an encoder meaning for it to return different vectors on Tuesday. So `HER = 1.000` is a clean statement and `HER = 0.749` is unambiguously a defect. R is the *controllable substrate*, which is what makes it certifiable in a way the other two layers are not.

**The honest liability, stated before anyone raises it in diligence:** on current public evidence and on our own ARI-E measurement (§8), the harness layer plausibly dominates the representation layer. Under a single index, that finding would refute ARI's premise. Under the decomposition it is a *result*.

### 5.2 Definitions

- **Agent under test (AUT):** a function `A: X → V` from inputs to vector-valued representations.
- **Environmental condition:** `E = (process, machine, precision, library version, concurrency, time)`.
- **Probe:** `P = Q ∘ A`, where `Q` is the canonical quantizer.
- **Reproducibility event:** for input `x` and conditions `E₁, E₂` — *hash-identical* if `P(x;E₁) = P(x;E₂)` bit-exactly, else *drifted* with magnitude `H(P(x;E₁), P(x;E₂))`.

Aggregates over the input set:

- **HER (Hash Equality Rate):** `HER(Eₖ) = mean_x 1[P₀(x) = Pₖ(x)]` ∈ [0,1]
- **Mean Hamming drift:** `H̄(Eₖ) = mean_x H(x, Eₖ)`, in bits
- **ARI(A):** HER averaged over the **comparable core** `{proc, conc, time}`

Confidence intervals are a block bootstrap over inputs.

### 5.3 The condition set

| id | description | isolates | expected on a reproducible agent |
| --- | --- | --- | --- |
| `same` | same context, repeated immediately | probe purity / within-session determinism | deterministic agent: 1.000; API: a measurement |
| `proc` | same machine, new process | BLAS thread schedule, kernel autotuning | < 1.0 iff cross-process non-determinism |
| `mach` | same CPU model, different machine | hardware drift | < 1.0 iff hardware-level float divergence |
| `prec` | fp16 vs fp32 vs bf16 | precision-induced drift | typically the largest drop |
| `lib` | library patch-version delta | version drift | < 1.0 iff a kernel changed |
| `conc` | under concurrent load | concurrency effects | usually near 1.0 |
| `time` | wall-clock gap > 24h | combined real-world | the cross-process replay failure mode |

**Why the headline averages only `{proc, conc, time}`.** These are the conditions measurable for *any* agent class. An API cannot observe `mach`, `prec` or `lib` — they are provider-internal — so folding them in would make a self-hosted agent that honestly *exposes* those axes score below an API that merely hides them. `mach`/`prec`/`lib` are reported per-condition as self-hosted **diagnostics**, read alongside the headline and not inside it.

**`same` is a control for one agent class and a measurement for the other.** For a deterministic self-hosted agent it must read 1.000, and a value below that means the *probe* is non-deterministic — exactly what disqualifies stochastic quantizers. For a black-box API we do not control the agent, so `same < 1.0` is the agent's within-session determinism, not probe impurity. Reports carry an `agent_class` and the scorer enforces `same = 1.0` only for `self_hosted`.

One consequence should be stated plainly: an API's headline is a **measured lower bound** on its true drift, because any `mach`/`prec`/`lib` drift it has is unobservable from outside. A self-hosted ARI and an API ARI are not strictly comparable even on the shared core.

### 5.4 The frozen input set

**ARI-Bench-v0.1** is a fixed, hash-pinned slice of BEIR: 334 NFCorpus (medical), 333 SciFact (scientific), 333 FiQA-2018 (financial), N = 1,000, each item title+body truncated to 512 characters. Content hash `e9ec8b01c62635…` over the ordered `input_id\0text\n` records. At N = 1,000 the HER standard error is ≤ 0.016.

ARI-Bench deliberately **introduces no new data**. ARI measures reproducibility, not capability, so the inputs need to be fixed, public and representative rather than novel. Reusing a recognised corpus removes any "you picked prompts to flatter model X" objection — which matters for a measurement meant to be cited in procurement — and makes ARI the companion axis to the leaderboard people already read. The slice is produced deterministically from `(corpora, n, seed)`, so the same parameters give a byte-identical file and the same hash on any machine.

### 5.5 The report artifact and its verification

A report is JSON conforming to a published schema: per-condition HER, H̄ and CI; the aggregate ARI; the environment manifest; the encoder fingerprint; the input-set content hash; and a **SHA-256 audit digest per condition** taken over the code matrix in fixed input order. A third party can re-run the protocol and compare digests without infrastructure access.

Attestation signs a **manifest, not the report alone**. The notary signs one artifact at a time, so signing only the report would leave the inputs free to change underneath it. The manifest names every input by digest, names the report by digest, and records the metric version; signing it binds all of them at once, and swapping any one breaks the check.

**The verifier does not import our package, or SEMQ.** It uses the standard library and `cryptography`. A verifier that had to run our code to confirm our claim would not be independent, so the split is the point rather than an implementation detail.

We are equally explicit about what attestation does *not* establish. A pass shows that the report and its inputs are what the named key signed. It does not show who holds the key unless the reader supplies a key they already trust. It does not show that the inputs describe reality — a dataset can be filtered before it reaches the signer. And without a timestamp authority, `created_at` is the signer's local clock and is worth nothing against a determined signer.

The leaderboard therefore carries a per-row `verification` field with exactly two values — `self-reported` (schema, invariants and arithmetic re-check, inputs pinned to the frozen set, per-condition numbers are the submitter's word) and `attested` (the above plus a verifying signature under a registered key). **All 13 current rows read `self-reported`,** because no durable signing key is registered yet. A sidecar that is present but does not hold up is a hard failure, not a downgrade. Rows also carry `input_pin` and `comparable`: a run on a *prefix* of the frozen set is accepted, recorded as `prefix:<n>`, and marked not comparable, because a 200-item run and a 1,000-item run are not the same measurement.

---

## 6. ARI-R: the deployed-agent panel

### 6.1 The panel

13 embedding models — 5 commercial APIs and 8 self-hosted open models — on the frozen ARI-Bench-v0.1. n = 1,000 for all except `mxbai` and `arctic` at n = 200 (deterministic; capped for CPU cost). `time` is measured at a canonical **~76h (3-day)** gap; `conc` is a 64-way concurrent burst against a calm pass.

| rank | agent | class | ARI | 95% CI |
| --- | --- | --- | --- | --- |
| 1 | `gemini/gemini-embedding-001` | **api** | **1.000** | [1.000, 1.000] |
| 1 | `BAAI/bge-large-en-v1.5` | self-hosted | 1.000 | [1.000, 1.000] |
| 1 | `BAAI/bge-m3` | self-hosted | 1.000 | [1.000, 1.000] |
| 1 | `intfloat/multilingual-e5-large` | self-hosted | 1.000 | [1.000, 1.000] |
| 1 | `sentence-transformers/all-MiniLM-L6-v2` | self-hosted | 1.000 | [1.000, 1.000] |
| 1 | `sentence-transformers/all-mpnet-base-v2` | self-hosted | 1.000 | [1.000, 1.000] |
| 1 | `nomic-ai/nomic-embed-text-v1.5` | self-hosted | 1.000 | [1.000, 1.000] |
| 1 | `mixedbread-ai/mxbai-embed-large-v1` | self-hosted | 1.000 | [1.000, 1.000] |
| 1 | `Snowflake/snowflake-arctic-embed-l` | self-hosted | 1.000 | [1.000, 1.000] |
| 10 | `openai/text-embedding-3-large` | api | 0.845 | [0.822, 0.866] |
| 11 | `mistral/mistral-embed` | api | 0.699 | [0.672, 0.727] |
| 12 | `voyage/voyage-4-large` | api | 0.500 | [0.472, 0.528] |
| 13 | `cohere/embed-v4.0` | api | 0.169 | [0.147, 0.194] |

Per-API breakdown, `proc / conc / time`, with `same` as a within-session diagnostic:

| agent | `same` | `proc` | `conc` | `time` (~76h) | ARI |
| --- | ---: | ---: | ---: | ---: | ---: |
| gemini-embedding-001 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| text-embedding-3-large | 0.859 | 0.862 | 0.851 | 0.822 | 0.845 |
| mistral-embed | 0.734 | 0.753 | 0.791 | **0.554** | 0.699 |
| voyage-4-large | 0.571 | 0.615 | **0.209** | 0.676 | 0.500 |
| embed-v4.0 | 0.133 | 0.146 | 0.130 | 0.232 | 0.169 |

### 6.2 What the panel shows

**1. Reproducibility is a real, provider-specific axis.** The paid APIs span 0.169 to 1.000. This is not anti-API bias: one API is perfectly reproducible, so the spread is a property of each provider's serving stack.

**2. Gemini's 1.000 is deterministic compute, not a cache.** This was the obvious objection and we tested it directly. We built 150 **novel** inputs — a unique nonce appended to each real text, so the provider cannot have cached them. Gemini read **1.000 on novel inputs**; the OpenAI control on the *same* novel inputs read 0.807, proving the probe detects non-determinism when it exists. The remaining short-TTL-cache hypothesis is settled by the `time` condition: Gemini holds 1.000 across a ~76h gap, far beyond any plausible result-cache TTL.

**3. The drift is mostly per-call — except Mistral, which accumulates over days.** For OpenAI, Voyage and Cohere, `time` ≈ `proc`: the noise is per-request, not time-dependent. Mistral is the exception. Its `time` drops to **0.554** against `proc` 0.753, with H̄ doubling, and a 19h gap read 0.753 — the drift only appears over multiple days. This is a silent model or infrastructure change, and it is exactly why the canonical `time` gap is >24h.

**4. Concurrency is a hidden axis.** A 64-way burst leaves OpenAI, Mistral and Cohere at their per-call rate and leaves Gemini bit-perfect. `voyage-4-large` collapses from `proc` 0.615 to `conc` **0.209**, with H̄ three times larger: under load, Voyage routes concurrent requests to backends that disagree. No other axis surfaced this, and it is what drops Voyage's overall ARI to 0.500. Note that `voyage-4-large` is the embedder Anthropic's documentation recommends, which makes its drift a practical concern for that stack.

**5. Ground truth, verified at the float level.** `same` HER < 1.0 for OpenAI contradicted the assumption that `same` is a 1.0 control, so we verified it below the probe. Four rapid repeats of one identical text returned **bit-identical** vectors (a short-window cache). Thirty texts embedded in two *separated* passes gave **3 of 30 differing at the float level**, and where they differed, **~2,300 of 3,072 dimensions** changed at magnitude **~1e-4**. That is a materially different vector from a different replica, not last-bit noise. The harness is correct; the differing codes trace to genuinely different embeddings.

**6. Self-hosted is bit-reproducible on the comparable core — with one condition attached.** All 8 open models read 1.000 under `proc`, `conc` and `time`, including with BLAS threads **unpinned** on x86 under multi-threaded OpenBLAS. We had expected unpinned multi-threaded BLAS to be the confound and it was not, so we make no thread-pinning caveat. But the `prec` diagnostic collapses every one of them:

| model | `same` | `prec` HER (bf16 vs fp32) | H̄ (bits) |
| --- | --- | --- | --- |
| BAAI/bge-large-en-v1.5 | 1.000 | 0.004 | 5.94 |
| BAAI/bge-m3 | 1.000 | 0.000 | 7.25 |
| intfloat/multilingual-e5-large | 1.000 | 0.004 | 6.77 |
| sentence-transformers/all-mpnet-base-v2 | 1.000 | 0.035 | 4.08 |
| sentence-transformers/all-MiniLM-L6-v2 | 1.000 | 0.094 | 2.34 |
| nomic-ai/nomic-embed-text-v1.5 | 1.000 | 0.008 | 4.36 |
| mixedbread-ai/mxbai-embed-large-v1 | 1.000 | 0.008 | 5.91 |
| Snowflake/snowflake-arctic-embed-l | 1.000 | 0.004 | 6.43 |

Higher-dimensional models flip more bits, having more coordinates to disagree on. The honest claim is *self-host **and pin your precision** → bit-reproducible*, not "self-hosted is always safe."

One caveat we record rather than hide: running 8 concurrent encodes on a **single shared model instance** (not thread-safe in PyTorch) produced a rare, reproducible ~0.1% single-input flip on bge-large — BLAS reduction order shifting under thread contention. It is a serving-implementation artifact rather than a reproducibility property, so `conc = 1.000` stands for the realistic batched or replica'd case. It is also the first direct sighting of BLAS-order code drift in this programme, and the instrument resolved it.

### 6.3 GPU determinism: the TF32 cliff

The self-hosted 1.000 above is measured on CPU. We re-ran the 8 encoders on GPU (A10G Ampere and T4 Turing, n = 512) to test whether it is a CPU artifact. It is, and the cause is specific.

An isolation 2×2 over TF32 × `use_deterministic_algorithms`, at fp32, cross-process, mean over the 8 encoders:

| | TF32 **off** / nondet | TF32 **off** / det | TF32 **on** / nondet | TF32 **on** / det |
| --- | --- | --- | --- | --- |
| **mean proc HER** | **1.000** | **1.000** | **0.749** | **0.749** |

**TF32 is the only variable that matters, and the determinism flag has exactly zero effect** — the two determinism columns are bit-identical per model (bge-large reads 0.8301 in both). The 2×2 is necessary because a single cell with TF32 off *and* determinism on would attribute the fix to either variable; only the crossing separates them.

The mechanism is worth stating because it is the high-pass thesis on a real configuration. In-process `same` is always 1.000. Across a fresh process, cuBLAS can select a different kernel through per-process autotuning. At true fp32 that difference is ~1e-7 — below the quantizer floor, so HER stays 1.0. TF32 truncates the mantissa to ~19 bits, so the *same* cross-process kernel difference becomes ~1e-3, crosses the floor, and flips the code. **TF32 amplifies cross-process kernel variation from invisible to code-flipping.**

Comparing GPU codes against a CPU-fp32 baseline at the same fixed scale:

| encoder | GPU fp32, **TF32 on**, vs CPU | GPU fp32, **TF32 off+det**, vs CPU | GPU **bf16** vs CPU |
| --- | --- | --- | --- |
| bge-large-en-v1.5 | 0.830 | 1.000 | 0.004 |
| bge-m3 | 0.641 | 1.000 | 0.000 |
| multilingual-e5-large | 0.670 | 1.000 | 0.000 |
| all-mpnet-base-v2 | 0.793 | 0.996 | 0.016 |
| all-MiniLM-L6-v2 | 0.875 | 0.998 | 0.131 |
| nomic-embed-text-v1.5 | 0.731 | 1.000 | 0.018 |
| **mxbai-embed-large-v1** | **0.541** | **0.537** | 0.004 |
| snowflake-arctic-embed-l | 0.660 | 0.998 | 0.004 |

"fp32 on GPU" runs matmul in TF32 by default on Ampere, so it disagrees with a true-fp32 CPU baseline on **12–46% of inputs**. **A model audited at fp32/CPU and served at fp32/GPU is not the same system** — silently, and invisibly to cosine or retrieval.

Cross-GPU is the axis that turned out *not* to be dangerous. With TF32 off and determinism forced, A10G↔T4 reads **mean 0.999** across the eight encoders. `mxbai-embed-large-v1` is the honest outlier: its two GPUs agree with each other at 1.000 but both disagree with CPU at 0.537. That is a CPU↔GPU kernel divergence, consistent across GPU architectures — the irreducible "audit on CPU, serve on GPU" gap made concrete for one model. **The dangerous axis is CPU↔GPU, not GPU-arch↔GPU-arch.**

Finally, a 7B decoder embedder (`intfloat/e5-mistral-7b-instruct`) at bf16, TF32 off, n = 128, reads `same` 1.000 and `proc` **1.000** cross-process. A bf16-served model does not route through TF32 and so sidesteps *that* cliff, while still paying the bf16-vs-fp32 precision cliff.

**How to read this section.** It is an instrument validation on a known phenomenon, not a discovery. TF32's numerics are documented and it is off by default in PyTorch 2.3. The value is that the discrete-attractor instrument gives a per-model, decision-grade number where a raw-L2 benchmark reads "5.74e-4, reproducible". The novel *data* in this paper is the API panel in §6.2.

---

## 7. ARI-D: the logits move before the tokens

### 7.1 Setup and the first measurement

The same high-pass argument applies one layer up. A logit vector is a vector; the interesting property is that logit perturbations are observable *before* they flip an `argmax`. We quantize the full-vocabulary logit vector at each generation step under greedy decoding, so every difference is infrastructure rather than sampling.

First run: TinyLlama-1.1B, 12 prompts, 48 new tokens, 539 steps, fp32/CPU/4-thread reference, teacher-forced so every condition is scored against an identical context.

| condition | token agree | HER | H̄ | early warning | median margin, flipped | median margin, survived | free-running exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference / proc / 1 thread / batched | 100.00% | 1.0000 | 0.0000 | 0.00% | — | 1.706 | 100.0% |
| **bf16** | **99.07%** | 0.0000 | **0.0343** | **99.07%** | **0.043** | **1.757** | 75.0% |
| **int8** | **74.21%** | 0.0000 | **0.8614** | **74.21%** | **0.588** | **2.681** | 0.0% |

*Early warning* = share of steps where the emitted token was identical but the code changed.

**The mechanism is confirmed directly rather than by analogy.** Under bf16 the steps that *did* flip sit at a median top-2 logit margin of **0.043**; the steps that survived sit at **1.757** — a 41× gap. Token flips concentrate almost entirely where the decision was nearly tied, while the distribution moved everywhere.

**Two design results follow.** First, **H̄ is the ARI-D metric and HER is not.** HER is exact-match over a 32,000-dimension code, so it saturates at 0.0000 for *both* precision changes and cannot rank them; H̄ separates them 0.0343 against 0.8614. This differs from ARI-R, where HER carries the signal. Second, an engineering prerequisite: SEMQ's `Context` caps `max_dim` at **65,536**, which fits TinyLlama's 32,000-token vocabulary but not Llama-3 (128,256), Qwen2.5 (151,936) or Gemma (256,000). Shipping ARI-D beyond toy models requires chunking, raising the cap, or a top-*k* slice — and the last changes what is being measured.

### 7.2 Production-scale models

TinyLlama is not a model anyone serves, so the first run could not support a claim about detecting a provider's serving change. We re-ran on three that are, on an L40S: 48 prompts, 48 new tokens each, ~2,265 scored steps per model, fp32 reference on the same GPU. Vocabularies were chunked across contexts with a single shared scale (calibrating each chunk separately would make the joined code meaningless).

| model | condition | tok agree | HER | H̄ | tok-same/code-changed | generations exact | 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :--- |
| **Llama-3.1-8B** | proc / 1 thread / TF32 off | 100.00% | 1.0000 | 0 | 0.00% | 48/48 | [93%, 100%] |
| | **TF32 on** | **100.00%** | 0.0000 | 3.1e-03 | **100.00%** | **48/48** | [93%, 100%] |
| | batched | 100.00% | 0.5267 | 1.0e-05 | 47.33% | 48/48 | [93%, 100%] |
| | fp16 | 99.91% | 0.0000 | 8.7e-03 | 99.91% | 47/48 | [89%, 100%] |
| | **bf16** | 99.21% | 0.0000 | 5.4e-02 | 99.21% | **38/48** | [66%, 88%] |
| **Qwen2.5-7B** | proc / 1 thread / TF32 off | 100.00% | 1.0000 | 0 | 0.00% | 48/48 | [93%, 100%] |
| | TF32 on | 99.96% | 0.0000 | 5.0e-03 | 99.96% | 48/48 | [93%, 100%] |
| | batched | 100.00% | 0.3347 | 1.6e-05 | 66.53% | 48/48 | [93%, 100%] |
| | fp16 | 99.74% | 0.0000 | 9.4e-03 | 99.74% | 45/48 | [83%, 98%] |
| | **bf16** | 99.21% | 0.0000 | 7.7e-02 | 99.21% | **31/48** | [50%, 77%] |
| **Mistral-7B-v0.3** | proc / 1 thread / TF32 off | 100.00% | 1.0000 | 0 | 0.00% | 48/48 | [93%, 100%] |
| | TF32 on | 100.00% | 0.0000 | 1.6e-03 | 100.00% | 48/48 | [93%, 100%] |
| | batched | 100.00% | 0.8065 | 1.3e-05 | 19.35% | 48/48 | [93%, 100%] |
| | fp16 | 99.87% | 0.0000 | 2.2e-02 | 99.87% | 43/48 | [78%, 95%] |
| | **bf16** | 99.52% | 0.0000 | 2.9e-02 | 99.52% | **38/48** | [66%, 88%] |

**A provider can turn on TF32 and change every reading the instrument takes while every token and every complete generation stays identical.** On Llama and Mistral all 48 generations are character-identical under TF32; on Qwen the tokens survive at 99.96% and the generations at 48/48. A monitor comparing outputs sees nothing on any of the three. The controls hold everywhere — new process, single thread, TF32 off gives HER 1.0000 and 48/48 — which is what makes this a discrimination rather than an instrument that fires at everything.

**Small per-token drift is not a small problem.** This is the result the 12-prompt run could not support:

| model | tokens lost under bf16 | complete generations lost | amplification |
| --- | ---: | ---: | ---: |
| Llama-3.1-8B | 0.79% | 20.8% | 26× |
| Qwen2.5-7B | 0.79% | 35.4% | 45× |
| Mistral-7B-v0.3 | 0.48% | 20.8% | 43× |

One flipped token derails every token after it, so a sub-1% per-step divergence becomes a one-in-three chance of a different answer. At 12 prompts the bf16 and fp16 intervals overlapped and this could not be claimed; at 48 they separate on Llama, [66%, 88%] against [89%, 100%].

Two checks came back clean. **Chunking introduces no artifact** — Mistral's 32,768-token vocabulary fits a single context and behaves like the two chunked models on every column that matters. And **batch composition varies more than anything else here**: the share of steps where the token held but the code moved runs 19% on Mistral, 47% on Llama and 67% on Qwen, so a monitor tuned on one model would misread another.

### 7.3 Against cheaper alternatives: SEMQ mostly loses

The ARI-D result is only interesting if a simpler statistic does not do the same job. We measured that, and the answer is largely no.

Response to Gaussian logit noise, and the reference state a monitor must retain per step:

| statistic | σ = 1e-4 | 1e-3 | 1e-2 | order | bytes/step | bit-reproducible |
| --- | ---: | ---: | ---: | :--- | ---: | :---: |
| **SEMQ H̄** | 1.33e-4 | 1.33e-3 | 1.33e-2 | linear | 16,000 | **yes** |
| **top-2 margin Δ** | 1.14e-4 | 1.15e-3 | 1.14e-2 | linear | **8** | **yes** |
| max \|Δlogit\| | 4.29e-4 | 4.28e-3 | 4.28e-2 | linear | 128,000 | yes |
| top-20 set change | 1.77e-4 | 7.07e-4 | 5.62e-3 | sub-linear | 80 | no |
| **KL** | **−5.1e-10** | 2.18e-7 | 2.23e-5 | **quadratic** | 128,000 | **no (4.6%)** |
| JS | −1.9e-10 | 5.39e-8 | 5.57e-6 | quadratic | 128,000 | no (363%) |
| token flip | 0 | 1.86e-3 | 2.97e-3 | threshold | 4 | no |

**KL is out, and the reason is structural.** It is second order in the perturbation, so a 10× rise in σ gives a 100× rise in KL. At σ = 1e-4 it reads −5.1e-10: zero to numerical precision, with the wrong sign. It is also not reproducible — regrouping the arithmetic algebraically moves it by 4.6%, and JS by more than 100% — so a published KL value is not a number a third party can recompute. That matters for attestation and not for detection. JS inherits the same defect.

**The top-2 margin is the result that goes against us, and it should not be softened.** It is linear in σ, tracks SEMQ within 14%, is equally bit-stable, and needs **8 bytes** of reference state against SEMQ's 16,000. **For detecting a precision change, ship the margin.** Anyone who runs this comparison will find the same thing.

**What survives is coverage, not sensitivity.** Every statistic above was measured under isotropic noise, which moves the top of the distribution and the tail together — which cannot separate a statistic that reads the whole vector from one that reads two entries. Perturbing *only* the tail (σ = 0.1):

| statistic | full vector | below rank 20 | below rank 100 |
| --- | ---: | ---: | ---: |
| **SEMQ H̄** | 0.1288 | **0.1289** | **0.1285** |
| top-20 set change | 0.0509 | 0.0335 | **0** |
| **top-2 margin Δ** | 0.1115 | **0** | **0** |
| **token flip** | 0.0289 | **0** | **0** |

SEMQ's response does not change. The margin and the token statistic read **exactly zero**. They are not less sensitive; they are structurally blind.

**The defensible posture is both, not either.** The margin is the right always-on tripwire at 8 bytes per step. SEMQ is the full-coverage, bit-reproducible record for what the tripwire cannot see, and for attestation, where KL's irreproducibility rules it out. Presenting SEMQ as a *replacement* for the cheap statistic would not survive scrutiny.

### 7.4 Does real drift ever land in the tail?

The coverage argument is only commercially real if some real serving change produces tail-confined drift. We asked that question directly, on Qwen2.5-1.5B-Instruct, CPU, the same 48 prompts and 2,265 steps, with both possible answers **pre-registered in the script before the run**.

The four conditions are a `logit_bias` on 40 refusal/hedging tokens (−5.0, the mechanism a `logit_bias` API parameter implements), bf16 as an expected-isotropic control, and two public third-party LoRA adapters merged into the weights. A tokenizer change is absent on purpose: two tokenizers do not share vocabulary indices, so there is no common space in which to measure a per-token change.

**The circularity this design avoids.** A bias placed on tokens chosen *by rank* would settle the question by construction. The 40 biased tokens were picked by meaning. Where they landed is a measurement: median rank **29,564**, lowest rank **2**.

Detection, and enrichment by rank band (share of total absolute logit change in a band ÷ share of vocabulary in that band; 1.00 is chance):

| condition | token agreement | top-2 margin Δ | SEMQ H̄ | ranks 1–2 | 3–10 | 11–100 | 101–1000 | 1001+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logit bias, refusal set | **100.00%** | **0.0019** | 5.14e-04 | 5.03× | **28.72×** | **21.86×** | **9.33×** | 0.94× |
| bf16 (control) | 98.90% | 0.0732 | 6.65e-02 | 1.42× | 1.29× | 1.20× | 1.07× | 1.00× |
| adapter Code-LoRA-r16 | 88.87% | 1.0517 | 8.94e-01 | 1.44× | 0.96× | 0.69× | 0.60× | 1.00× |
| adapter de-pii-redactor | 94.79% | 0.4610 | 3.94e-01 | 1.88× | 1.48× | 1.27× | 1.10× | 1.00× |

Enrichment rather than raw share is necessary because ranks 1001+ hold 99.34% of a 151,936-token vocabulary: an isotropic change puts 99.3% of its mass there and *looks* tail-confined. The bf16 row scores 99.288%, which is exactly that number.

**No condition refutes coverage.** The pre-registered refutation needed drift to concentrate in the top ranks; nothing does. **Token suppression is the one condition that concentrates** — 28.72× at ranks 3–10 and 21.86× at ranks 11–100, against a control at 1.29× and 1.20×. Adapter drift spreads across the vocabulary and is *below* chance in the middle bands for the code adapter.

**Output monitoring is blind to the bias arm and only to that arm.** Token agreement holds at 100.00%; the adapters drop to 88.87% and 94.79%, so an output monitor catches those with no instrument at all.

**And the margin is nearly blind to the bias, but not structurally blind.** This corrects an earlier reading of a 12-prompt run where the margin moved by exactly zero. At 48 prompts one biased token reaches rank 2 and the margin moves by 0.0019 — about 550× smaller than the 1.0517 the code adapter produces. A monitor thresholded to ignore ordinary variation would likely miss it. "Likely" is the honest word: this run does not measure a threshold. **This narrows the open question rather than closing it:** token suppression does land in the middle ranks, so the coverage case has one real instance; it is not yet a demonstration that a margin statistic *cannot* see such a change.

---

## 8. ARI-E: the scaffold decides more than the model

### 8.1 A power analysis, run before spending the compute

The original ARI-E plan was two agents against a self-hosted model on 25 tasks at two repeats. We simulated its power first:

| pass-rate gap | repeats | 25 cases | 100 cases | 200 cases |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 (the null) | 2 | 8% | 6% | 6% |
| 0.20 | 2 | **8%** | 11% | 37% |
| 0.20 | 5 | 23% | 74% | 96% |
| 0.40 | 5 | 99% | 100% | 100% |

That design detects a 20-point gap **8% of the time**. The published datapoint that motivated the work — 24/25 against 19/25 on the same model and endpoint under two harnesses — *is* a 20-point gap. The plan would have missed it more than nine times in ten.

Two rules follow. **Repeats buy more than cases,** because repeats sharpen the control that carries the effect. And below about 10 cases the false-positive rate reaches 18% instead of 5%, so ARI-E should not be run that small.

### 8.2 The measurement

We used `nvidia/Open-SWE-Traces`: two agent scaffolds (**SWE-agent**, **OpenHands**) crossed with two models (**Qwen3.5-122B**, **Minimax-M2.5**) over ~18,000 SWE-bench-style instances, up to three rollouts per cell. Neither scaffold was built by us — if we wrote both harnesses, the measured effect would be a property of our own design choices. An outcome is `resolved == 1`, produced by running the repository's own test suite against the agent's patch; the agent never saw that test. A benchmark whose verdict came from a judge would measure the judge. Only instances with at least two rollouts on both sides are used, because one rollout gives no self-consistency control.

The estimator is the same on both contrasts, so they sit side by side:

| contrast | cases | self-consistency | cross agreement | **ARI-E effect** | 95% CI |
| --- | ---: | ---: | ---: | ---: | :--- |
| scaffold, model = Qwen3.5-122B | 10,788 | 0.874 | 0.785 | **+0.089** | [+0.083, +0.094] |
| scaffold, model = Minimax-M2.5 | 12,195 | 0.899 | 0.849 | **+0.050** | [+0.044, +0.052] |
| model, scaffold = SWE-agent | 10,674 | 0.880 | 0.839 | **+0.041** | [+0.039, +0.046] |
| model, scaffold = OpenHands | 11,933 | 0.892 | 0.833 | **+0.059** | [+0.054, +0.062] |

**Mean scaffold effect +0.069. Mean model effect +0.050.** Every interval excludes zero.

**1. Agents disagree with themselves 10 to 13% of the time.** Two rollouts of one scaffold, one model and one instance reach different verdicts about one time in nine. Any comparison that ignores this attributes that noise to whatever it happens to be comparing.

**2. The control removes most of the apparent effect, and this is what justifies the metric.**

| contrast | apparent disagreement | agent's own noise | ARI-E effect | share removed |
| --- | ---: | ---: | ---: | ---: |
| scaffold, Qwen | 0.215 | 0.126 | +0.089 | **59%** |
| scaffold, Minimax | 0.151 | 0.101 | +0.050 | **67%** |
| model, SWE-agent | 0.161 | 0.120 | +0.041 | **74%** |
| model, OpenHands | 0.167 | 0.108 | +0.059 | **65%** |

A report that diffs two scaffolds and stops would say the scaffold changed 21.5% of outcomes on Qwen. The attributable figure is 8.9%. **The naive number is 2.4× too large**, and on the model contrasts nearly 4×.

**3. The scaffold matters more than the model, but not overwhelmingly.** 1.4:1 on the means. That supports the direction of the published claim without supporting its strongest form; on this dataset the two are the same order of magnitude.

**4. Scaffold and model interact, so a harness effect quoted without naming the model hides something.** Qwen3.5 scores 46.8% under SWE-agent and 33.4% under OpenHands — a 13.5-point spread — while Minimax scores 46.9% and 42.1%, a 4.9-point spread.

**5. An ARI-E effect is not a score gap.** Qwen differs by 13.5 points of pass rate between scaffolds but produces an 8.9-point effect, because the effect is a difference of *agreement* rates, and that compresses.

**The largest threat to these numbers:** roughly 22% of rows carry `resolved = -1` and were dropped, because an ungraded run is not a failed one. If grading failure correlates with task difficulty, the surviving set is easier than the whole.

---

## 9. Why representation drift matters downstream

The obvious objection to ARI-R is that a single retrieval rarely moves its top-K set, so who cares. We tested it in a multi-hop retrieval agent: a self-avoiding nearest-neighbour walk over NFCorpus (3,633 documents) with bge-large — each hop embeds the current query, retrieves the nearest unvisited document, and steps to it. This is an LLM-free but faithful model of an iterative RAG/ReAct loop. A per-hop query perturbation σ mimics a black-box API's per-call drift. 15-hop walks, 100 seed documents, 30 perturbed runs each. The analysis auto-selects the operating σ as the largest that still "looks reproducible" by the standard check (recall@10 ≥ 0.98), which landed at σ = 1e-3.

| depth | σ = 0 | σ = 1e-4 | **σ = 1e-3 (operating, recall 0.988)** | σ = 3e-3 (recall 0.966) |
| --- | --- | --- | --- | --- |
| 1 | 0.000 | 0.000 | 0.002 | 0.028 |
| 5 | 0.000 | 0.007 | 0.040 | 0.147 |
| 10 | 0.000 | 0.007 | 0.071 | 0.258 |
| 15 | 0.000 | 0.012 | **0.115** | 0.392 |

| σ | recall@10 (standard check) | top-1 flip rate | SEMQ Hamming |
| --- | --- | --- | --- |
| 1e-4 | 0.9986 | 0.0007 | 0.00813 |
| 1e-3 | 0.9879 | 0.0133 | 0.07776 |
| 3e-3 | 0.9661 | 0.0435 | 0.21941 |

At a per-call drift where the standard reproducibility check reads **recall@10 = 0.988** — a practitioner would call this pipeline reproducible — **11.5% of the agent's 15-hop trajectories diverge** from the deterministic reference, growing monotonically with depth. That is an **8.6× amplification** of the per-hop top-1 flip rate. The drift driving it is invisible to top-K overlap and plainly visible to the instrument (Hamming 0.078). At σ = 0 the divergence is exactly 0 at every depth, so all of it is attributable to per-call representation non-determinism.

This is the same shape as the ARI-D amplification result in §7.2 — a sub-1% per-step divergence becoming a 20–35% chance of a different complete answer — arriving one layer down by a different route. **Representation non-determinism compounds with depth into outcome-level divergence.** A bit-reproducible embedder has zero compounding. The instrument diagnoses the risk; determinism removes it.

---

## 10. What the instrument is worth at each layer

A result set that only supports the instrument is not evidence. Across the three layers, SEMQ does progressively less work, and we would rather state that than let a reviewer discover it.

| index | SEMQ's role | strength |
| --- | --- | --- |
| ARI-R | the instrument | **load-bearing** |
| ARI-D | the instrument, but a cheaper statistic matches it on the changes measured so far | **contested** |
| ARI-E | verification only, no quantizer | **absent from the metric** |

**At the representation layer it is load-bearing.** The drift the panel measures is sub-rank-gap by construction: aggregate retrieval quality cannot resolve it (§3.2), and the alternatives that can — comparing stored top-*k* lists per query — need a query set, cover only the distribution you saved, and mix real drift with tie-order churn. A corpus-side, bit-exact code needs none of that.

**At the decoding layer it is contested.** An 8-byte top-2 margin recovers 86% of the signal for 1/2000th of the storage and is equally bit-reproducible (§7.3). What SEMQ adds is coverage — it reads the whole distribution, so it sees changes below the top ranks that a margin statistic is structurally blind to — and §7.4 shows one real mechanism, token suppression, that lands there. That is one instance, not a general case, so the honest posture is both statistics rather than either: the margin as an always-on tripwire, SEMQ for the coverage the tripwire lacks and for attestation.

**At the harness layer it is absent.** ARI-E compares verdicts and sequences of tool names. Nothing in it is a vector, so a quantizer has no place there and forcing one in would be decoration.

What connects all three layers is not the quantizer but the **verification**: a signed manifest binds every report to its inputs, so changing the report fails the check and changing an input while the report stays identical also fails it. Signing uses SEMQ; **verification does not require SEMQ**, because a check that needs the publisher's code is not independent.

---

## 11. Limitations and open questions

**Scope of the measurement.**
- ARI measures a **system** property — model + BLAS + threads + hardware + libraries — not a model in isolation. A number without its environment manifest is not third-party-verifiable.
- ARI does **not** detect semantic drift: a model swap, a fine-tune, or a prompt change.
- An API's headline is a measured lower bound (§5.3).

**Instrument.**
- Only QBIN n=2 is characterised. Whether the universality of `b` holds across operators is open.
- The ~14× LSH sensitivity gap rests on **one synthetic run** at matched bitrate on anisotropic vectors and has never been replicated. It is the weakest number in this paper.
- The probe-verifiability sweep is one encoder, one corpus, one seed, and simulates reassociation by algebraic identity rather than measuring it across real architectures.
- **Cross-architecture invariance carries one unresolved conflict.** Handed a single fixed, serialised artifact and asked to reproduce rankings across CPU architectures (Graviton aarch64 producer → Intel x86_64 consumer, same lockfile, search threads pinned to 1), faiss PQ/OPQ held 1.000000 on 5 of 5 cells while `semq_qbin` flipped one ranking row in 323 (0.996904) on one corpus; `semq_canonical` held everywhere. This is measured on the **index-search path**, not the encode-only path an ARI reading uses, and the analysis attributes it to search-path score deltas an order of magnitude above faiss's rather than to anything qbin-specific. It also contradicts an earlier result reporting 1.000000 for qbin on a Mac NEON ↔ Linux AVX-512 pair. QBIN is the operator ARI makes canonical, so the encode path should be re-verified on a Graviton↔x86 pair before §4.3's structural-invariance argument is leaned on. On the question that matches a deployment — what each host *independently produces* from the same corpus — SEMQ built byte-identical files on 4 of 4 cells and faiss on 1 of 5, because k-means codebook training accumulates floats and diverges at corpus scale.
- `Context` caps `max_dim` at 65,536, so long-vocabulary ARI-D requires chunking with a shared scale.

**Panel coverage.**
- `mach` and `lib` are not measured for the self-hosted panel on CPU; the GPU run doubles as a partial `mach`.
- Mistral's `conc` was measured at burst = 16 rather than 64 (rate-limited at 64).
- Rates are input-distribution dependent: the API numbers shifted from an earlier synthetic sample to real BEIR text — Voyage moved 0.40 → 0.62 — which is precisely why the frozen real inputs matter.
- GPU scope is A10G + T4, 8 encoders plus one 7B, n = 512.

**ARI-D.**
- Every run is **greedy**. Whether ARI-D separates infrastructure drift from *sampling* noise above temperature 0 is untested, and it is the harder and more valuable claim.
- The free-running generation column rests on 48 prompts while the per-step columns rest on 2,265, hence the Wilson intervals.
- These are simulated serving changes on one machine, not observations of a provider actually changing something.
- CPU dynamic int8 via `qnnpack` is not the GPTQ/AWQ scheme a hosted provider would use.
- The tail-coverage question (§7.4) is narrowed, not settled.

**ARI-E.**
- ~22% of rows were ungraded and dropped; if grading failure tracks difficulty the surviving set is easier than the whole.
- The dataset was built to train models, not to test scaffolds, so nothing guarantees the three rollouts of an instance are independent draws.
- Outcome agreement only. The two scaffolds do not share a tool vocabulary — SWE-agent calls `bash`/`str_replace_editor`/`submit`, OpenHands calls `execute_bash`/`str_replace_editor`/`think`/`finish`/`fetch` — and `think` has no counterpart at all, so a name mapping would hide a real capability difference.
- Two scaffolds and two models, which interact; treat these as two measurements rather than a constant.

**The five open questions we would answer next.**

1. **Is real serving drift ever tail-confined?** §7.4 gives one instance (token suppression, 28.72× enrichment at ranks 3–10). Whether a margin statistic can be thresholded to catch it is unmeasured, and that decides whether SEMQ's coverage advantage at the decoding layer is commercially real or only true.
2. **Does ARI-D separate infrastructure drift from sampling noise above temperature 0?**
3. **Re-derive the 14× LSH figure** on a real corpus before it goes in front of anyone.
4. **Re-verify QBIN's encode-path cross-architecture invariance** on a Graviton↔x86 pair, and resolve the conflict with the earlier NEON↔AVX-512 result.
5. **Publish a signing key.** The `--expect-key` machinery is worth nothing without somewhere to look the key up, and all 13 leaderboard rows read `self-reported` until then.

---

## 12. Conclusion

Reproducibility is a required property of AI systems with no defined measurement. We have given one, at three layers, with the instrument's own limits measured as carefully as its capabilities.

The result we would defend hardest is the panel: five commercial embedding APIs, one frozen input set, one bit-exact metric, bootstrap intervals, and a spread from 0.169 to 1.000. Reproducibility is a provider-specific engineering property, one provider has solved it, and until now there was no number to say so. The results we would defend next are the two amplification findings — a sub-1% per-step logit divergence becoming a 20–35% chance of a different complete answer, and a per-call embedding drift that reads as "reproducible" on recall@10 producing 11.5% trajectory divergence in a 15-hop agent. Both say the same thing: small, invisible, per-step non-determinism is not a small problem in systems that iterate.

We would defend the instrument claim narrowly and no further. SEMQ is not necessary for ARI. A frozen product quantizer is a viable probe, an 8-byte margin statistic beats SEMQ on cost at the decoding layer, and the harness layer needs no quantizer at all. What SEMQ offers is a guarantee checkable in advance rather than audited per shipped artifact, a diagonal sensitivity Jacobian that lets drift be attributed to a dimension, and a one-scalar calibration state that a verifier can check by eye. That is a preference argued on verification cost, not an impossibility result — a narrower claim than the instrument's behaviour on the panel might invite, and the one the measurements support.

---

## References

1. ReproRAG: reproducibility of retrieval-augmented generation embeddings. arXiv:2509.18869.
2. EU Artificial Intelligence Act, Article 15 (accuracy, robustness and cybersecurity of high-risk AI systems).
3. NIST AI Risk Management Framework, Manage 4.1.
4. US Treasury AI Risk Management Framework, February 2026.
5. Thakur, N. et al. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.
6. Muennighoff, N. et al. MTEB: Massive Text Embedding Benchmark.
7. Zhang, Wang, Ge, Xu, Hamm, Reddy. Stop Comparing LLM Agents Without Disclosing the Harness. arXiv:2605.23950.
8. Harness-Bench: Measuring Harness Effects across Models in Realistic Agent Workflows. arXiv:2605.27922.
9. Kapoor, S. et al. Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation. arXiv:2510.11977.
10. Defeating Nondeterminism in LLM Inference — batch-invariant kernels at ~60% throughput cost.
11. openai/openai-python issue #868 — non-deterministic embeddings, community-documented since 2023.
12. QwenLM/qwen-code PR #348 — precision pinning does not route as documented on OpenRouter.
13. Dettmers, T. et al. QLoRA / NF4 quantization.
14. The SEMQ Group. Canonical Quantization for Reproducible Semantic Retrieval (draft).
15. nvidia/Open-SWE-Traces. Hugging Face dataset.
16. Muvon/octobench — 24/25 vs 19/25, same model and endpoint, two harnesses.

*Sourcing caveat, carried forward from the internal proposal that motivated §8: a widely repeated claim that a scaffold swap moved one model from 42% to 78% on CORE-Bench traces only to vendor marketing content. It could not be confirmed in HAL's published results and is deliberately not cited.*

---

## Appendix A — The `(s, b, κ)` fingerprint registry

| model | dim | s | b | κ | status |
| --- | ---: | ---: | ---: | ---: | --- |
| BAAI/bge-large-en-v1.5 | 1024 | 0.077546 | 0.9939 | 2.887 | measured *(ARI-Canonical reference)* |
| BAAI/bge-m3 | 1024 | 0.082476 | 0.9944 | 2.774 | measured |
| intfloat/multilingual-e5-large | 1024 | 0.076236 | 0.9907 | 2.893 | measured |
| openai/text-embedding-3-large | 3072 | 0.051514 | 0.9319 | 1.652 | measured |
| voyage/voyage-4-large | 1024 | 0.080435 | 0.9785 | 2.489 | measured |
| cohere/embed-v4.0 | 1536 | 0.069946 | 0.9953 | 2.657 | measured |
| mistral/mistral-embed | 1024 | 0.081177 | 0.9146 | 1.484 | measured |
| gemini/gemini-embedding-001 | 3072 | 0.050995 | 0.9901 | 2.547 | measured |
| nomic-ai/nomic-embed-text-v1.5 | 768 | 0.093399 | 0.9989 | 2.924 | measured |
| mixedbread-ai/mxbai-embed-large-v1 | 1024 | 0.078308 | 0.9774 | 2.546 | measured |
| Snowflake/snowflake-arctic-embed-l | 1024 | 0.082256 | 1.0074 | 3.058 | measured |
| sentence-transformers/all-mpnet-base-v2 | 768 | 0.096253 | — | — | `floor_limited` |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 0.132349 | — | — | `floor_limited` |
| intfloat/e5-mistral-7b-instruct | 4096 | — | 0.9600 | 2.065 | `kappa_only` (sweep, not in panel) |
| Alibaba-NLP/gte-Qwen2-7B-instruct | 3584 | — | 0.9939 | 2.641 | `kappa_only` (sweep, not in panel) |

`b = 0.979 ± 0.028` across the 13 models carrying a fit; `κ ∈ [1.48, 3.06]`. `floor_limited` models are bit-reproducible (ARI = 1.000); the floor lives only in the synthetic sensitivity sweep.

## Appendix B — Artifacts and reproduction

| artifact | location |
| --- | --- |
| Normative spec (probe, input set, condition set, report schema, fingerprints) | `spec/` |
| Frozen input set, N = 1,000, hash `e9ec8b01…` | `data/ari-bench-v0.1.jsonl` |
| Reference implementation (probe, agents, metrics, report, attestation) | `ari/` |
| Independent verifier (stdlib + `cryptography`, does not import SEMQ) | `ari/verify_report.py` |
| Leaderboard, submissions, scorer | `leaderboard/` |
| §3.2 regime discrimination | `experiments/regime-discrimination/` |
| §4.2 drift sensitivity | `experiments/drift-sensitivity/` |
| §4.3 probe verifiability | `experiments/probe-verifiability/` |
| §6 deployed-agent panel | `experiments/deployed-agent-panel/` |
| §6.3 GPU determinism | `experiments/gpu-determinism/` |
| §7.1 ARI-D | `experiments/decoding-reproducibility/RESULTS.md` |
| §7.2 ARI-D at production scale | `experiments/decoding-reproducibility/RESULTS-production-models.md` |
| §7.3 decoding baselines | `experiments/decoding-reproducibility/BASELINES.md` |
| §7.4 drift rank profile | `experiments/drift-rank-profile/` |
| §8.1 ARI-E power | `experiments/harness-power/` |
| §8.2 ARI-E harness effect | `experiments/harness-effect/` |
| §4.4 Jacobian decomposability (L1_19), §9 agentic compounding (L3_09), §11 cross-architecture (L2_05) | `semq-research`, `docs/findings.md` |

Every experiment directory carries a `RESULTS.md` with the protocol, the machine-readable JSON or CSV it was generated from, and a reproduce command. Figures in `docs/figures/` are generated by `docs/figures/make_figures.py`, which reads the experiment JSON directly — no figure recomputes a number, so no figure can disagree with the table it illustrates.

Licensing: code Apache-2.0; spec and input set CC-BY-4.0, with BEIR text retaining its original licenses.
