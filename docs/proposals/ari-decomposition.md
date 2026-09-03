# Proposal: split ARI into three indices

> **Status: adopted.** ARI-D shipped as [`spec/arid-bench-v0.1.md`](../../spec/arid-bench-v0.1.md);
> ARI-E's first measurement is in [`experiments/harness-effect/`](../../experiments/harness-effect/).
> Retained as the design record.

**Status:** proposal, not adopted · August 2026 · supersedes nothing yet

**Summary.** ARI is named for agent reproducibility and measures embedding
reproducibility. That gap is the single most attackable thing about the programme. This
proposes replacing the one index with three. **ARI-R** (representation), **ARI-D**
(decoding), **ARI-E** (environment). Publishing ARI-R as the only one that exists today,
and being explicit that it is the smallest of the three terms.

The argument is that the decomposition is *more* defensible than the single index, not
less, and that the market is already asking for the two layers we do not cover.

---

## 1. The problem

The original whitepaper (since removed) admitted the scope limit in one bullet:

> **Coverage.** ARI measures embedding-layer reproducibility. Token-level output
> reproducibility needs a separate symbolic probe.

That bullet is doing far too much work. An index called the *Agent* Reproducibility Index
does not touch:

| source of agent irreproducibility | covered today |
| --- | --- |
| embedding/representation drift | **yes** |
| sampling, tie-breaks, batch-dependent logits | no |
| tool-call ordering, results, wall-clock, network | no |
| prompt path (retrieval → different context → different everything) | no |
| multi-turn state accumulation | no |
| harness/scaffold differences | no |

A reviewer who reads the title and then the limitations section concludes that the name
oversells the measurement. That conclusion is correct, and it contaminates the parts of
the work that *are* sound. We have already had to retract two claims in §3.1 for
overreach ([experiments/probe-verifiability](../../experiments/probe-verifiability/RESULTS.md)).
A third overreach in the title would not survive.

**The naming problem is also a strategy problem.** While ARI stays one number, every
layer we do not measure is a hole. When it becomes three numbers, the layers we do not
measure become a roadmap.

---

## 2. The decomposition

### ARI-R. Representation reproducibility

*Does the same input produce the same internal representation across environments?*

This is the existing ARI, unchanged: SEMQ codes over embeddings, HER and H̄ across the
condition set (`same`, `proc`, `mach`, `prec`, `lib`, `conc`, `time`).

**Status: built, measured, 13 deployed agents on the panel.** This is the whole of the
current evidence base and it ships as-is.

### ARI-D. Decoding reproducibility

*Given the same representation, does the same token sequence come out?*

The same high-pass argument applies one layer up. A logit vector is a vector. SEMQ
quantizes it the same way. The interesting property is that logit perturbations are
observable *before* they flip an `argmax`. A precision change shifts token probabilities
long before it changes the emitted token, exactly as an embedding perturbation shifts
cosine long before it changes a ranking.

Measurement: SEMQ over the logit vector at each generation step, giving a symbolic trace
per completion, at fixed seed and greedy decoding so that any difference is infrastructure.

**Status: built and measured on CPU**
([experiments/decoding-reproducibility](../../experiments/decoding-reproducibility/RESULTS.md)).
The claim holds in its strong form: under bf16, **99.07% of decoding steps emit the
identical token while the SEMQ logit code changes**, and token flips concentrate at a **41×
smaller** top-2 margin than surviving steps. Direct evidence that probabilities move
before the argmax does. Controls (fresh process, single thread, padded batch) are clean.

Two findings change the design:

- **H̄, not HER, is the ARI-D metric.** Exact-match HER saturates at 0.0000 for both bf16
  and int8. Mean symbol disagreement separates them, 0.0343 against 0.8614.
- **`Context` caps `max_dim` at 65,536**, so a full-vocabulary logit probe works for
  TinyLlama (32,000) but not for Llama-3 (128,256), Qwen2.5 (151,936) or Gemma (256,000).
  Raising the cap, chunking, or a validated top-*k* slice is a prerequisite for shipping
  this layer.

Still open: whether ARI-D separates infrastructure drift from *sampling* noise at
temperature > 0. That is the harder and more valuable claim, and it is not yet tested.

### ARI-E. Environment and harness reproducibility

*Holding the model fixed, how much does the outcome depend on the scaffold around it?*

This is the layer where, on current public evidence, most of the variance lives. We
already have infrastructure pointed at it: `qwen-harness-trace` runs the same model behind
the same endpoint under two different harnesses, with notarized checkpoints at each
decision boundary.

Proposed measurement: outcome agreement across harnesses at matched model and endpoint,
plus checkpoint-level attribution of *where* two harnesses diverge.

**Status: infrastructure built, matrix not yet run** (blocked on the GPU provisioning
task).

---

## 3. Why now. The market is asking for the layers we do not cover

This is not a speculative roadmap. Each of the two uncovered layers has visible,
independent demand.

### ARI-D: silent serving changes are a named, unsolved problem

- **OpenRouter has shipped product surface for it.** Their routing API now exposes a
  `quantizations` field so callers can restrict which precision tiers serve their traffic,
  and `provider.ignore` to blacklist providers outright. Their own documentation describes
  the failure mode plainly: some providers serve more heavily quantized variants, your
  application still returns a response, and *nothing in your logs tells you why* the
  output changed.
- **There is no verifier.** OpenRouter's guidance for detecting this is behavioural. 
  A/B your own prompts against a pinned high-precision endpoint and watch for drift. A
  `qwen-code` PR that tried to pin precision found the field did not route as documented.
  So the industry has a control surface with no instrument to confirm it is working. That
  is precisely an ARI-D-shaped hole.
- **Practitioners are building ad-hoc versions of this already.** The recurring advice
  across the drift-detection writeups is to maintain a fixed "golden probe set," capture
  logprobs where available, run it on a schedule, and alert on distributional shift rather
  than trusting version strings. That is ARI-D, hand-rolled, without a standard.
- **Temperature-0 determinism is publicly known to be false**, and the mechanism. 
  batch-dependent kernel selection in normalization, matmul and attention. Is the same
  class of effect ARI-R already measures at the embedding layer. Thinking Machines' work
  on batch-invariant kernels showed reproducibility is recoverable at roughly a 60%
  throughput cost, which means there is a real decision to be instrumented: *are you
  paying for determinism, and did you get it?*

### ARI-E: harness variance now has its own literature

- **"Stop Comparing LLM Agents Without Disclosing the Harness"** (Zhang, Wang, Ge, Xu,
  Hamm, Reddy. ArXiv:2605.23950) argues the harness "is often a stronger determinant of
  agent performance than the model it wraps," and that harness-induced variance can exceed
  model-induced variance *including cases of model ranking reversal*.
- **"Harness-Bench"** (arXiv:2605.27922) evaluates 5,194 execution trajectories over 106
  sandboxed tasks and concludes capability "should be reported at the model-harness
  configuration level rather than attributed to the base model alone."
- **HAL** (Kapoor et al., arXiv:2510.11977) built standardized evaluation infrastructure
  across 21,730 rollouts, 9 models and 9 benchmarks, explicitly to separate models from
  scaffolds from benchmarks. At roughly $40,000 in compute.
- **The datapoint that started this thread:** `Muvon/octobench` scored 24/25 vs 19/25 on
  the *same model and same endpoint* under two harnesses.

Two preprints in one quarter naming the problem, and no standard index for it. That is
the same structural gap ARI was created to fill at the representation layer. A required
property with no defined measurement.

> **A caveat on sourcing.** A widely repeated claim that a scaffold swap moved one model
> from 42% to 78% on CORE-Bench traces only to vendor marketing content. I could not
> confirm it in HAL's published results and it is **not** cited here. The arXiv preprints
> and the octobench numbers are the load-bearing evidence.

---

## 4. Why ARI-R is still worth publishing, even as the smallest term

The obvious objection to this proposal is that it concedes the point: if E dominates, why
ship R at all?

**Because R is the only layer where a zero is meaningful.**

In ARI-D and ARI-E, stochasticity is *intended*. Temperature is a product decision. Tool
results depend on the network and the clock. A harness is supposed to make choices. So a
non-zero reading at those layers is ambiguous by construction. You cannot separate
designed randomness from infrastructure drift without a great deal of additional
scaffolding, and any threshold you pick is a judgement call.

At the representation layer, **any** variation is unintended. Nobody ships an encoder
meaning for it to return different vectors on Tuesday. So `HER = 1.000` is a clean
statement and `HER = 0.749` is unambiguously a defect. That is what makes ARI-R
certifiable in a way the other two layers are not, and it is why it is the right place to
start rather than an arbitrary subset.

This reframes the thin slice as *the controllable substrate*. And it is an argument the
public write-ups did not make anywhere at the time.

---

## 5. The honest liability

**On current public evidence, ARI-E probably dominates ARI-R.** If harness choice swings
outcomes by more than model choice does, then representation-layer reproducibility is the
smallest of the three terms we would be publishing.

We should say this in the public write-ups rather than wait for it to be raised in diligence.
The decomposition makes it sayable: "R is the layer we can measure exactly, D and E are
larger and harder, here is the roadmap" is a credible position. "ARI measures agent
reproducibility" followed by a limitations bullet that quietly excludes most of an agent
is not.

There is also a real defensive benefit. Under the single index, any finding that harness
variance dominates is a *refutation* of ARI's premise. Under the decomposition, the same
finding is a **result**. It is ARI-E doing its job, and it increases the value of the
programme rather than undermining it.

---

## 6. Migration

| step | change | cost |
| --- | --- | --- |
| 1 | Rename the current index **ARI-R** throughout; define ARI as the tuple `(ARI-R, ARI-D, ARI-E)` with D and E marked *not yet measured*. | Low. Spec, public write-ups, README, report schema. |
| 2 | Rewrite §7 to state the coverage limit as a decomposition rather than a caveat, and add §4's "zero is meaningful" argument. | Low, prose only. |
| 3 | Add a `layer` field to `spec/report-schema.json` so a report declares which index it carries. | Low, additive and backwards-compatible if defaulted to `R`. |
| 4 | ~~Build ARI-D~~ — **done on CPU.** Remaining: raise or work around `max_dim ≤ 65,536`, and test separation from sampling noise at temperature > 0. | Medium. |
| 5 | Run ARI-E on the `qwen-harness-trace` matrix. | Already scoped; blocked on GPU provisioning. |

Steps 1–3 are documentation and cost days, not weeks. They are worth doing before any
external conversation regardless of whether 4 and 5 proceed.

---

## 7. Arguments against

**"Three indices are harder to sell than one."** True for a headline number and false for
a standard. Standards that survive are usually decomposable. A single score invites the
question of what it excludes, and a decomposition answers it in advance. It also lets a
customer buy the layer they care about.

**"ARI-D may not be a separate measurement."** ~~Possible.~~ **Answered.** ARI-D produces a
signal at steps where token output is identical, so it is not reducible to comparing
outputs. It is a distinct measurement with its own metric (H̄ rather than HER) and its own
engineering prerequisite (the `max_dim` cap). See
[experiments/decoding-reproducibility](../../experiments/decoding-reproducibility/RESULTS.md).

**"Naming the dominant layer we do not measure invites the criticism."** It does. The
alternative is that the criticism arrives during diligence with the added implication that
we had not noticed. Given the §3.1 history, pre-emption is the better trade.

**"This is a rebrand of work we have not done."** The strongest objection. Mitigation: do
not publish ARI-D or ARI-E as *numbers* until they are measured. Publish the
decomposition as a **scope statement**, what the index covers and what it does not,
which is honest on day one, and let the two empty slots be a roadmap rather than a claim.

---

## Sources

- [Stop Comparing LLM Agents Without Disclosing the Harness](https://arxiv.org/abs/2605.23950). ArXiv:2605.23950
- [Harness-Bench: Measuring Harness Effects across Models in Realistic Agent Workflows](https://arxiv.org/abs/2605.27922). ArXiv:2605.27922
- [Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation](https://arxiv.org/abs/2510.11977). Kapoor et al.
- [How OpenRouter Model Routing Works: Providers, Fallbacks & Auto Router](https://openrouter.ai/blog/insights/model-routing/)
- [How to Evaluate LLM Provider Performance](https://openrouter.ai/blog/insights/evaluate-llm-provider-performance/)
- [feat(openaiContentGenerator): Avoid quantized models on OpenRouter](https://github.com/QwenLM/qwen-code/pull/348). Precision pinning does not route as documented
- [UI Option to Filter OpenRouter Quantization (FP4/Int4)](https://github.com/RooCodeInc/Roo-Code/issues/11325). User-visible symptom of silent quantization
- [Silent Quantization: Why the Model You Pay For Today Isn't the Model You Paid For Last Quarter](https://tianpan.co/blog/2026-05-02-silent-quantization-model-you-paid-for-last-quarter)
- [Defeating Nondeterminism in LLM Inference](https://www.llmwatch.com/p/eli5-defeating-nondeterminism-in). Batch-invariant kernels, ~60% throughput cost
- [Your LLM Is Lying to You Silently: 4 Statistical Signals That Catch Drift Before Users Do](https://dev.to/aiwithmohit/your-llm-is-lying-to-you-silently-4-statistical-signals-that-catch-drift-before-users-do-4cg2)
- [Muvon/octobench](https://github.com/Muvon/octobench). 24/25 vs 19/25, same model and endpoint, two harnesses
