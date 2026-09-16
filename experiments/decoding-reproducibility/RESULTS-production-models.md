# Internal decoding: larger-model GPU results

The experiment used one L40S, 48 prompts per model, 48 new tokens, and greedy decoding.
Each condition was compared with an fp32 reference on the same GPU.
Results are in `results/decoding_matrix_{llama31_8b,qwen25_7b,mistral7b_v03}_gpu.json`.

## Models

Chunked vocabularies use one calibration scale across contexts.

| model | params | vocabulary | SEMQ path |
| --- | ---: | ---: | --- |
| Llama-3.1-8B-Instruct | 8.0B | 128,256 | chunked, 2 contexts |
| Qwen2.5-7B-Instruct | 7.6B | 152,064 | chunked, 3 contexts |
| Mistral-7B-Instruct-v0.3 | 7.2B | 32,768 | single context |

## Results

Token agreement and HER use approximately 2,265 teacher-forced steps per model.
H̄ measures differing code positions under the experiment's normalization.
`tok-same/cd` uses all steps as its denominator.
`free exact` counts identical complete generations out of 48. Its interval is a Wilson interval.

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

## bf16 differences

The ratios below compare token and generation disagreement rates. They are not causal amplification estimates.

| model | tokens lost under bf16 | generations lost | amplification |
| --- | ---: | ---: | ---: |
| Llama-3.1-8B | 0.79% | 20.8% | 26x |
| Qwen2.5-7B | 0.79% | 35.4% | 45x |
| Mistral-7B-v0.3 | 0.48% | 20.8% | 43x |

## Interpretation and limits

TF32 changed every logit code in these runs while all 48 complete generations per model remained identical.
Qwen's teacher-forced token agreement was 99.96 percent; Llama and Mistral reached 100 percent.
The fresh-process, one-thread, and TF32-off controls retained code agreement.
The single-context Mistral result is a control against a simple chunking explanation.
It does not prove that chunking is correct under all inputs.

These are simulated deployment changes on one GPU, driver, and PyTorch build.
They do not establish that a provider changed its serving configuration.
The output intervals use only 48 observations per model.
CPU dynamic int8 is absent from the GPU matrix.
See [baseline costs](BASELINES.md) before selecting a monitoring statistic.

## Reproduce

Use the [GPU procedure](../../infra/README.md) with enough memory for the fp32 reference.
Install the SEMQ SDK and model dependencies. From the repository root:

```bash
ARI_D_MODEL=Qwen/Qwen2.5-7B-Instruct ARI_D_DEVICE=cuda python experiments/decoding-reproducibility/run_matrix.py
```

Repeat with the desired model identifier from the source configuration.
Record the output path printed by the script and compare its environment metadata with the historical run.
