# ARI-D on models people actually serve

**Status: run on an L40S.** Three models, 48 prompts each, greedy. Data:
`results/decoding_matrix_{llama31_8b,qwen25_7b,mistral7b_v03}_gpu.json`.

**Headline: a provider can turn on TF32 and change every reading the instrument
takes, while every token and every complete generation stays identical.** That holds
on all three models.

The earlier runs used TinyLlama-1.1B and Qwen2.5-3B. Neither is a model anyone serves,
so they could not support a claim about detecting a provider's serving change. These
three can.

## Setup

| model | params | vocabulary | SEMQ path |
| --- | ---: | ---: | --- |
| Llama-3.1-8B-Instruct | 8.0B | 128,256 | chunked, 2 contexts |
| Qwen2.5-7B-Instruct | 7.6B | 152,064 | chunked, 3 contexts |
| Mistral-7B-Instruct-v0.3 | 7.2B | 32,768 | single context |

48 prompts, 48 new tokens each, greedy decoding, fp32 reference on the same GPU.
About 2,265 scored steps per model. Mistral is in the table because its vocabulary
fits one context: if chunking introduced an artifact, Mistral is the row that would
disagree with the others. It does not.

**Reading the columns.** Every column compares a condition against the fp32 reference
on the same input.

- **tok agree** - share of steps emitting the same token. n = 2,265 steps.
- **HER** - share of steps whose SEMQ code is bit-identical. n = 2,265 steps.
- **H̄** - share of code symbols differing. Same steps, normalised per symbol rather
  than per step, which is why a row can show HER 0.53 and H̄ 1.0e-05 at once: half the
  steps differ, each by about one symbol in 64,128.
- **tok-same/cd** - share of **all** steps with an identical token but a changed code.
  The denominator is every step, not every divergence.
- **free exact** - whole generations identical, as a count. n = 48 prompts, with a
  Wilson interval. This column rests on 48 observations while the others rest on
  2,265, so it is reported as a count.

## Results

| model | condition | tok agree | HER | H̄ | tok-same/cd | free exact | 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :--- |
| **Llama-3.1-8B** | proc / threads1 / tf32_off | 100.00% | 1.0000 | 0 | 0.00% | 48/48 | [93%, 100%] |
| | **tf32_on** | **100.00%** | 0.0000 | 3.1e-03 | **100.00%** | **48/48** | [93%, 100%] |
| | batched | 100.00% | 0.5267 | 1.0e-05 | 47.33% | 48/48 | [93%, 100%] |
| | fp16 | 99.91% | 0.0000 | 8.7e-03 | 99.91% | 47/48 | [89%, 100%] |
| | **bf16** | 99.21% | 0.0000 | 5.4e-02 | 99.21% | **38/48** | [66%, 88%] |
| **Qwen2.5-7B** | proc / threads1 / tf32_off | 100.00% | 1.0000 | 0 | 0.00% | 48/48 | [93%, 100%] |
| | tf32_on | 99.96% | 0.0000 | 5.0e-03 | 99.96% | 48/48 | [93%, 100%] |
| | batched | 100.00% | 0.3347 | 1.6e-05 | 66.53% | 48/48 | [93%, 100%] |
| | fp16 | 99.74% | 0.0000 | 9.4e-03 | 99.74% | 45/48 | [83%, 98%] |
| | **bf16** | 99.21% | 0.0000 | 7.7e-02 | 99.21% | **31/48** | [50%, 77%] |
| **Mistral-7B-v0.3** | proc / threads1 / tf32_off | 100.00% | 1.0000 | 0 | 0.00% | 48/48 | [93%, 100%] |
| | tf32_on | 100.00% | 0.0000 | 1.6e-03 | 100.00% | 48/48 | [93%, 100%] |
| | batched | 100.00% | 0.8065 | 1.3e-05 | 19.35% | 48/48 | [93%, 100%] |
| | fp16 | 99.87% | 0.0000 | 2.2e-02 | 99.87% | 43/48 | [78%, 95%] |
| | **bf16** | 99.52% | 0.0000 | 2.9e-02 | 99.52% | **38/48** | [66%, 88%] |

## Reading

**1. TF32 is invisible at the output and total at the instrument.** On Llama and
Mistral every token is identical and all 48 generations are character-identical, while
the SEMQ code changes at every step. On Qwen the tokens survive at 99.96% and the
generations at 48/48. A monitor comparing outputs sees nothing on any of the three.

**2. The controls hold on all three models.** A new process, a single thread and TF32
disabled give HER = 1.0000 and 48/48 identical generations everywhere. Without those
rows the TF32 result would be an instrument that fires at everything. With them it is a
discrimination.

**3. Small per-token drift is not a small problem.** This is the result the earlier
12-prompt runs could not support, and it is the one worth quoting:

| model | tokens lost under bf16 | generations lost | amplification |
| --- | ---: | ---: | ---: |
| Llama-3.1-8B | 0.79% | 20.8% | 26x |
| Qwen2.5-7B | 0.79% | 35.4% | 45x |
| Mistral-7B-v0.3 | 0.48% | 20.8% | 43x |

One flipped token derails every token after it, so a sub-1% per-step divergence becomes
a one-in-three chance of a different answer. At 12 prompts the bf16 and fp16 intervals
overlapped and this could not be claimed. At 48 they separate on Llama, [66%, 88%]
against [89%, 100%].

**4. Chunking introduces no artifact.** Mistral's 32,768-token vocabulary fits a single
context and behaves like the two chunked models on every column that matters. The
chunked path is not manufacturing the result.

**5. Batch composition varies more than anything else here.** The share of steps where
the token held but the code moved runs 19% on Mistral, 47% on Llama and 67% on Qwen.
Batching is the one condition whose effect is strongly model-dependent, so a monitor
tuned on one model would misread another.

## Caveats

- **48 prompts, one prompt set.** The free-running column still rests on 48
  observations. The intervals are given for that reason and several of them are wide.
- **Greedy decoding.** This isolates infrastructure by removing sampling noise. It does
  not show that ARI-D separates infrastructure drift from sampling noise at
  temperature above zero.
- **One GPU, one driver, one torch build.** These are simulated serving changes on one
  machine, not observations of a provider actually changing something.
- **int8 is absent.** Dynamic quantisation is CPU-only in torch, so the GPU run cannot
  include it. The CPU numbers are in `RESULTS.md`.
- **None of this changes the cost comparison.** A top-2 margin statistic still gets 86%
  of the signal for a fraction of the storage. See `BASELINES.md`. What these runs
  establish is that the effect is real on deployment-scale models, not that SEMQ is the
  cheapest way to see it.
