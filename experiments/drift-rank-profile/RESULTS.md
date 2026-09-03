# Where drift lands, by token rank

**Status: run on Qwen2.5-1.5B-Instruct, CPU, 48 prompts.** Data:
`results/drift_rank_profile.json`.

## The question

SEMQ reads the whole logit vector. A top-2 margin statistic reads two numbers.
The one advantage SEMQ keeps at the decoding layer is coverage: it can see a
change that never reaches the top ranks.

`BASELINES.md` showed that with synthetic tail noise. SEMQ moved and the margin
read zero. That shows SEMQ **can** see what the margin cannot. It does not show
that a real serving change puts drift down there.

This experiment asks the question the other way round, and the pre-registration
in `run.py` fixed both answers before the run:

- **Supports coverage.** Drift sits below the top few ranks. The margin barely
  moves.
- **Refutes coverage.** Drift concentrates in the top ranks. The margin sees it
  for 8 bytes a step, and the extra 16,000 bytes buy nothing.

## Setup

The model is Qwen2.5-1.5B-Instruct in fp32 on CPU. The run uses the same 48
prompts that ARI-D reports on, loaded from that experiment's file, and 48 new
tokens per prompt with greedy decoding. This gives 2,265 scored steps, the same
count ARI-D gets. The run compares every condition against the fp32 reference on
the same input.

| condition | what it changes |
| --- | --- |
| `logit_bias_refusal_set` | Subtracts 5.0 from 40 tokens of refusal and hedging vocabulary |
| `bf16` | Casts the weights to bfloat16. The control, expected to be isotropic |
| `adapter Code-LoRA-r16` | Merges a public rank-16 code adapter into the weights |
| `adapter de-pii-redactor` | Merges a public rank-32 redaction adapter into the weights |

A tokenizer change is absent on purpose. Two tokenizers do not share vocabulary
indices, so no common space exists in which to measure a per-token change.

**The circularity this avoids.** A bias placed on tokens chosen *by rank* would
settle the question by construction. Put the bias at rank 500, and the margin
cannot see it. That proves nothing. The run picks the 40 biased tokens by
meaning. Where they landed is a measurement: their median rank is **29,564** and
their lowest rank is **2**.

## Detection

| condition | token agreement | top-2 margin change | SEMQ HER | SEMQ H-bar | max abs d-logit |
| --- | ---: | ---: | ---: | ---: | ---: |
| logit bias, refusal set | **100.00%** | **0.0019** | 0.0000 | 5.14e-04 | 5.00 |
| bf16 | 98.90% | 0.0732 | 0.0000 | 6.65e-02 | 0.83 |
| adapter Code-LoRA-r16 | 88.87% | 1.0517 | 0.0000 | 8.94e-01 | 17.30 |
| adapter de-pii-redactor | 94.79% | 0.4610 | 0.0000 | 3.94e-01 | 12.64 |

## Rank profile

Bucket size confounds the raw share of change in a rank band, because the bands
hold very different numbers of tokens. Ranks 1001 and above hold 99.34% of a
151,936 token vocabulary. An isotropic change puts 99.3% of its mass there and
looks tail-confined. The bf16 row scores 99.288%, which is that number.

Enrichment divides the confound out. It is the share of total absolute logit
change in a band, divided by the share of the vocabulary the band holds. A value
of 1.00 is exactly chance.

| condition | ranks 1-2 | 3-10 | 11-100 | 101-1000 | 1001+ |
| --- | ---: | ---: | ---: | ---: | ---: |
| logit bias, refusal set | 5.03x | **28.72x** | **21.86x** | **9.33x** | 0.94x |
| bf16 | 1.42x | 1.29x | 1.20x | 1.07x | 1.00x |
| adapter Code-LoRA-r16 | 1.44x | 0.96x | 0.69x | 0.60x | 1.00x |
| adapter de-pii-redactor | 1.88x | 1.48x | 1.27x | 1.10x | 1.00x |

| band | tokens | share of vocabulary |
| --- | ---: | ---: |
| ranks 1-2 | 2 | 0.0013% |
| ranks 3-10 | 8 | 0.0053% |
| ranks 11-100 | 90 | 0.0592% |
| ranks 101-1000 | 900 | 0.5924% |
| ranks 1001+ | 150,936 | 99.3418% |

## What the numbers say

**1. No condition refutes coverage.** The pre-registered refutation needs drift
to concentrate in the top ranks. Nothing does that. The two adapters read 1.44x
and 1.88x at ranks 1 and 2, and the code adapter is *below* chance in the middle
bands at 0.69x and 0.60x. Adapter drift spreads across the vocabulary.

**2. Token suppression is the one condition that concentrates.** The bias arm
reads 28.72x at ranks 3 to 10 and 21.86x at ranks 11 to 100. The control reads
1.29x and 1.20x in the same bands. This is the first measurement here of a
change that lives in the middle ranks rather than everywhere.

**3. Output monitoring is blind to the bias arm, and only to that arm.** Token
agreement holds at 100.00%. Every emitted token is identical. The adapters drop
to 88.87% and 94.79%, so an output monitor catches those without any instrument.

**4. The margin is nearly blind to the bias, but not structurally blind.** This
corrects an earlier reading of a 12 prompt run, where the margin moved by exactly
zero. At 48 prompts one biased token reaches rank 2, and the margin moves by
0.0019. That is about 550 times smaller than the 1.0517 the code adapter
produces. A monitor thresholded to ignore ordinary variation would likely miss
it. "Likely" is the honest word, because this run does not measure a threshold.

**5. This does not settle the cost argument.** SEMQ registers the bias arm at
H-bar 5.14e-04, and the margin registers it at 0.0019. Both are small, and both
are non-zero. The run adds the bias to the same logits in the same process, so a
no-change control reads exactly zero by construction. Both readings are real
rather than noise. The claim this run supports is narrow: drift from token
suppression sits in the middle ranks, and adapter drift does not sit in the top
ranks. The claim it does not support is that a margin statistic cannot see the
change at all.

**6. HER separates nothing here.** Every condition reads HER 0.0000, including
the bias arm. The per-step hash changes at every step under all four conditions,
so hash equality alone would fire on everything. H-bar is the statistic that
discriminates, which agrees with what ARI-D found.

## Caveats

- **One model, and a small one.** Qwen2.5-1.5B-Instruct. A 7B run did not
  complete. Nothing here shows the profile holds at deployment scale.
- **The bias is analytic, not served.** The run subtracts 5.0 from the chosen
  logits. This is what a `logit_bias` API parameter does, so the mechanism is
  faithful. It is not an observation of a provider changing a safety layer.
- **The adapters are third-party artifacts.** Two public LoRA adapters, trained
  by other people for their own purposes. That is the point, because neither was
  built to make this run come out any particular way. It also means their
  training data and quality are unknown.
- **Greedy decoding, one prompt set.** The profile above may depend on the
  prompt distribution.
- **`robertou2/task-12-Qwen-Qwen2.5-7B-Instruct` is excluded** from the 7B
  adapter list. It publishes a zero byte `adapter_model.safetensors`.

## Reproduce

```bash
cd semq-ari-benchmark
PYTHONPATH=$PWD DRIFT_DEVICE=cpu \
  python experiments/drift-rank-profile/run.py
```

Set `DRIFT_MODEL` to change the base model, and `DRIFT_ADAPTER` to a comma
separated list to change the adapters. `DRIFT_N_PROMPTS` and `DRIFT_NEW_TOKENS`
cut the run down for a fast check.
