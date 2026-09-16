# Logit changes by token rank

The run used Qwen2.5-1.5B-Instruct on CPU, 48 prompts, and 48 new tokens with greedy decoding.
It evaluated 2,265 steps against an fp32 reference.
Data: [drift_rank_profile.json](results/drift_rank_profile.json).

## Conditions

The bias tokens were selected by meaning, not rank. Their median rank was 29,564 and minimum rank was two.

| condition | what it changes |
| --- | --- |
| `logit_bias_refusal_set` | Subtracts 5.0 from 40 tokens of refusal and hedging vocabulary |
| `bf16` | Casts the weights to bfloat16. The control, expected to be isotropic |
| `adapter Code-LoRA-r16` | Merges a public rank-16 code adapter into the weights |
| `adapter de-pii-redactor` | Merges a public rank-32 redaction adapter into the weights |

## Detection

Each statistic compares the condition with the same reference input.

| condition | token agreement | top-2 margin change | SEMQ HER | SEMQ H-bar | max abs d-logit |
| --- | ---: | ---: | ---: | ---: | ---: |
| logit bias, refusal set | **100.00%** | **0.0019** | 0.0000 | 5.14e-04 | 5.00 |
| bf16 | 98.90% | 0.0732 | 0.0000 | 6.65e-02 | 0.83 |
| adapter Code-LoRA-r16 | 88.87% | 1.0517 | 0.0000 | 8.94e-01 | 17.30 |
| adapter de-pii-redactor | 94.79% | 0.4610 | 0.0000 | 3.94e-01 | 12.64 |

## Rank enrichment

Enrichment divides a band's share of absolute logit change by its share of vocabulary. One indicates proportional change.

| condition | ranks 1-2 | 3-10 | 11-100 | 101-1000 | 1001+ |
| --- | ---: | ---: | ---: | ---: | ---: |
| logit bias, refusal set | 5.03x | **28.72x** | **21.86x** | **9.33x** | 0.94x |
| bf16 | 1.42x | 1.29x | 1.20x | 1.07x | 1.00x |
| adapter Code-LoRA-r16 | 1.44x | 0.96x | 0.69x | 0.60x | 1.00x |
| adapter de-pii-redactor | 1.88x | 1.48x | 1.27x | 1.10x | 1.00x |

## Band sizes

Raw change shares require this correction because most tokens are in the final band.

| band | tokens | share of vocabulary |
| --- | ---: | ---: |
| ranks 1-2 | 2 | 0.0013% |
| ranks 3-10 | 8 | 0.0053% |
| ranks 11-100 | 90 | 0.0592% |
| ranks 101-1000 | 900 | 0.5924% |
| ranks 1001+ | 150,936 | 99.3418% |

## Interpretation

The bias condition retained all emitted tokens and produced a top-2 margin change of 0.0019.
Its change was concentrated in middle ranks after adjustment for band size.
The margin was small but nonzero. This corrects the earlier twelve-prompt result in which it was zero.
No detection threshold was measured, so the experiment does not establish whether a margin monitor would miss the change.

The adapter changes affected output tokens and extended across the vocabulary.
All conditions had HER zero; H̄ distinguished their magnitudes.
This experiment does not settle the storage-versus-coverage choice.

## Limits

The 7B run did not complete. The reported result covers one 1.5B model and one prompt set.
The bias was applied analytically to logits, not observed as a provider change.
The two adapters are external artifacts with separate training assumptions.
Tokenizer changes are excluded because vocabulary indices would no longer identify the same tokens.
The 7B adapter `robertou2/task-12-Qwen-Qwen2.5-7B-Instruct` was excluded because its published weight file was empty.

## Reproduce

Install `.[selfhosted]`, `peft`, and the [SEMQ SDK](../../ari/README.md#install-the-canonical-probe).
From the repository root:

```bash
DRIFT_DEVICE=cpu python experiments/drift-rank-profile/run.py
```

The command writes `experiments/drift-rank-profile/results/drift_rank_profile.json`.
Use `DRIFT_MODEL` and comma-separated `DRIFT_ADAPTER` values to change the model and adapters.
Use `DRIFT_N_PROMPTS` and `DRIFT_NEW_TOKENS` for a smaller development run.
Smaller runs are not comparable with the table above.
