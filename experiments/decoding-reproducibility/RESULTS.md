# Internal decoding: initial CPU results

This historical run used TinyLlama-1.1B, 12 prompts, 48 new tokens, and greedy decoding.
It scored 539 steps against an fp32 CPU reference with four threads.
Data: [decoding_matrix.json](results/decoding_matrix.json).

Teacher forcing evaluates each condition on the reference token sequence.
Free generation lets each condition produce its own sequence.
The probe used QUANT with eight bins and one calibration over the reference logits.
These measurements are distinct from the hosted-output ARI-D specification.

## Results

Early warning is the fraction of all steps with an unchanged token and changed code. Margin is the top-2 reference-logit gap.

| condition | axis | token agree | SEMQ HER | SEMQ H̄ | early warning | median margin, flipped | median margin, survived | free exact | 1st divergence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | `same` | 100.00% | 1.0000 | 0.0000 | 0.00% | — | 1.706 | 100.0% | 44.9 |
| proc | `proc` | 100.00% | 1.0000 | 0.0000 | 0.00% | — | 1.706 | 100.0% | 44.9 |
| threads1 | `proc` | 100.00% | 1.0000 | 0.0000 | 0.00% | — | 1.706 | 100.0% | 44.9 |
| batched | `batch` | 100.00% | 1.0000 | 0.0000 | 0.00% | — | 1.706 | 100.0% | 44.9 |
| **bf16** | `prec` | **99.07%** | 0.0000 | **0.0343** | **99.07%** | **0.043** | **1.757** | **75.0%** | 38.3 |
| **int8** | `prec` | **74.21%** | 0.0000 | **0.8614** | **74.21%** | **0.588** | **2.681** | **0.0%** | 3.5 |

## Interpretation

Under bf16, token agreement was 99.07 percent, while every logit code changed.
Nine of twelve freely generated completions remained identical.
Under int8, no complete generation remained identical.
The median reference margin was 0.043 for bf16 token flips and 1.757 for unchanged tokens.
Exact-code agreement was zero for both precision changes. H̄ distinguished their magnitudes.

## Later runs

Vocabulary chunking now handles vectors longer than the SDK's 65,536-dimension context limit.
Each chunk uses the same calibration scale.
See [larger-model results](RESULTS-production-models.md) for GPU measurements and [baselines](BASELINES.md) for alternative detectors.

## Limits

The table covers one small model, one prompt set, and CPU execution.
Greedy decoding removes intentional sampling but does not identify each source of computational variation.
Dynamic int8 quantization of linear layers is not equivalent to every hosted quantization method.
The fixed reference calibration affects the magnitude of code disagreement.
The batch condition's zero difference applies only to this configuration.

## Reproduce

Install `.[selfhosted]` and the [SEMQ SDK](../../ari/README.md#install-the-canonical-probe).
From the repository root:

```bash
ARI_D_DEVICE=cpu python experiments/decoding-reproducibility/run_matrix.py
```

The current script can use a larger prompt set than this historical run.
Match the recorded commit, prompts, model, and output metadata before claiming an exact reproduction.
The command writes results under `experiments/decoding-reproducibility/results/` and can replace an existing result.
