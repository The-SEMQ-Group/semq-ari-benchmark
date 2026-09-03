# ARI-D. Does a serving change move the logits before it moves the tokens?

**Status: run on CPU.** Script: [`run_matrix.py`](run_matrix.py). Machine-readable:
[`results/decoding_matrix.json`](results/decoding_matrix.json).

**Headline: under bf16, 99.07% of decoding steps emit the identical token while SEMQ reads
a changed logit code.** Token-level output comparison is silent at exactly those steps, and
yet 25% of the completions eventually diverge. The per-step signal precedes the visible
failure.

This is the first measurement of the proposed ARI-D layer
([docs/proposals/ari-decomposition.md](../../docs/proposals/ari-decomposition.md)).

## Setup

`TinyLlama/TinyLlama-1.1B-Chat-v1.0`, 12 prompts, 48 new tokens each, **greedy**
(`do_sample=False`). Greedy decoding removes sampling noise, so every difference below is
infrastructure. The reference is fp32 / CPU / 4 threads / unbatched. 539 decoding steps
total.

Two arms:

- **Teacher-forced.** Every condition is scored on the *reference* token sequence in one
  forward pass, so position *i* is evaluated against an identical context in every
  condition. This isolates the per-step computation from compounding divergence. It is the
  arm that tests the claim.
- **Free-running.** Every condition generates on its own. This is what production does.

SEMQ uses QUANT at 8 bins over the full 32,000-dimension logit vector, calibrated once on
the reference logits and frozen.

## Results

| condition | axis | token agree | SEMQ HER | SEMQ H̄ | early warning | median margin, flipped | median margin, survived | free exact | 1st divergence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | `same` | 100.00% | 1.0000 | 0.0000 | 0.00% | — | 1.706 | 100.0% | 44.9 |
| proc | `proc` | 100.00% | 1.0000 | 0.0000 | 0.00% | — | 1.706 | 100.0% | 44.9 |
| threads1 | `proc` | 100.00% | 1.0000 | 0.0000 | 0.00% | — | 1.706 | 100.0% | 44.9 |
| batched | `batch` | 100.00% | 1.0000 | 0.0000 | 0.00% | — | 1.706 | 100.0% | 44.9 |
| **bf16** | `prec` | **99.07%** | 0.0000 | **0.0343** | **99.07%** | **0.043** | **1.757** | **75.0%** | 38.3 |
| **int8** | `prec` | **74.21%** | 0.0000 | **0.8614** | **74.21%** | **0.588** | **2.681** | **0.0%** | 3.5 |

*Early warning* = share of steps where the emitted token was identical but the SEMQ logit
code changed. *Margin* = gap between the best and second-best reference logit.

## Reading

**1. The claim holds, and in its strong form.** Under bf16 the model emits the same token
at 99.07% of steps. SEMQ's code differs at 100% of steps. So at 99.07% of steps a monitor
comparing tokens sees nothing while the instrument sees the regime change. The gap is not
marginal. It is nearly the whole trace.

**2. The mechanism is confirmed directly, not by analogy.** The proposal argued that
precision "shifts token probabilities long before it flips an argmax." The margin columns
test that: under bf16 the steps that *did* flip sit at a median top-2 margin of **0.043**,
while the steps that survived sit at **1.757**. A 41× gap. Token flips concentrate almost
entirely where the decision was nearly tied anyway, while the distribution moved
everywhere. That is exactly the embedding-layer story one level up.

**3. Controls are clean, which is what makes (1) worth anything.** A fresh process,
single-threaded BLAS, and scoring inside a padded batch of 4 all give 100% token agreement,
HER = 1.0000 and identical free-running output. SEMQ on logits is not an instrument that
fires at everything.

**4. For ARI-D, HER is the wrong metric and H̄ is the right one.** HER is exact-match over a
32,000-dimension code, so it saturates at 0.0000 for *both* precision changes and cannot
rank them. H̄ separates them cleanly: 0.0343 for bf16 versus 0.8614 for int8, a 25×
difference that HER flattens to nothing. This differs from ARI-R, where HER carried the
signal. **The ARI-D metric should be H̄, not HER.**

**5. Compounding is fast.** Free-running, bf16 changes the completion for 3 of 12 prompts,
first diverging around token 38. int8 changes **every** completion, first diverging around
token 3.5. A per-step difference of 0.65% of steps becomes a 25% chance of a different
answer within 48 tokens.

## An engineering blocker worth stating plainly

**SEMQ's `Context` caps `max_dim` at 65,536.** TinyLlama's 32,000-token vocabulary fits.
Most current models do not:

| model | vocabulary | fits? |
| --- | ---: | :--- |
| TinyLlama-1.1B | 32,000 | yes |
| Llama-3 | 128,256 | **no** |
| Qwen2.5 | 151,936 | **no** |
| Gemma | 256,000 | **no** |

Shipping ARI-D beyond toy models requires either raising the cap, chunking the vocabulary
across several contexts, or restricting the probe to a top-*k* slice. The last option
changes what is being measured and would need its own validation. This is a real
prerequisite, not a detail.

## Caveats

- **One model, 1.1B parameters, 12 prompts, 539 steps.** Nothing here shows the ratios hold
  at scale or across architectures.
- **Greedy only.** This deliberately removes sampling noise to isolate infrastructure. It
  does not show that ARI-D separates infrastructure drift from sampling noise at
  temperature > 0, which is the harder and more useful claim. That needs a run at fixed
  seed and temperature > 0.
- **Teacher forcing is an idealisation.** It scores every condition against an identical
  context. In production the context drifts too, which is what the free-running arm shows.
- **`batched` was a null result.** Scoring inside a padded batch of 4 changed nothing. This
  does not contradict the published work on batch-invariance, which concerns large-batch
  GPU serving. It means this test did not use a configuration that shows the effect. A
  GPU run at realistic batch sizes is the next step.
- **CPU dynamic int8** on Linear layers via `qnnpack` is not the GPTQ/AWQ scheme a hosted
  provider would use.
- **HER = 0.0000 is partly a calibration artifact.** Logits have a much wider dynamic range
  than embeddings, so a scale frozen from reference logits is far from the perturbed
  distribution. That is the intended monitoring setup, but it is why H̄ is the informative
  number.
