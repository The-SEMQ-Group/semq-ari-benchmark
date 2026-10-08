# ARI-D-Logit-v0.2

Status: draft. Replaces the v0.1 logit probe for white-box decoding measurements.
Implementation: [`ari/logit_probe.py`](../ari/logit_probe.py).

This probe is for the self-hosted instrument that reads full logit vectors: the decoding-reproducibility, drift-rank-profile and matched-budget experiments.
The hosted-API bench ([ARI-D-Bench-v0.1](arid-bench-v0.1.md)) uses no SEMQ probe and is unchanged.

## Probe requirements

For each decoding step, take the logit vector over the full vocabulary as float32.

1. **Centre.** Subtract the row mean over the full vocabulary, computed in binary64, and cast the result to float32.
2. **Chunk.** Split at fixed boundaries into consecutive chunks of at most 65,536 logits. The last chunk holds the remainder. The layout depends only on the vocabulary size.
3. **Encode.** Rescale each chunk to unit L2 norm. Encode it with SEMQ QUANT from the public SDK (`semq` 1.x), with `n_bins = 8` (4 bits per logit) over the chunk's fixed range `2 / sqrt(width)`.

Compare codes chunk by chunk with `ari.code_metrics.chunked_code_diff`, because each chunk pads its own final byte.
Record the SDK version and build id with every capture.

## Changes from v0.1

| | v0.1 | v0.2 |
| --- | --- | --- |
| Range | one 99th-percentile scale, calibrated on the reference and shared by every chunk | fixed `2 / sqrt(width)` per chunk |
| Offset | kept | removed by centring |
| Normalisation | none | each chunk rescaled to unit norm |
| Chunk layout | any; codes did not depend on it | fixed by the vocabulary size; codes depend on it |

v0.1 and v0.2 codes are not comparable.

## Evidence

The evidence comes from teacher-forced logits of `Qwen/Qwen2.5-0.5B-Instruct`, with 151,936 logits in three chunks.
The runs covered 20 prompts of ARI-D-Bench-v0.1, each with 32 steps (640 rows).
The reference was fp32 on CPU with 4 threads.
The conditions were a fresh process, 1 thread, bf16 and int8 on CPU, and fp32 and bf16 on an Apple GPU.
The table compares v0.1, emulated, with two public-SDK candidates.
P1 renormalises raw logits per chunk; P2 centres first and is this probe.

| Measure | v0.1 | P1 | P2 (v0.2) |
| --- | --- | --- | --- |
| Changed rows detected, Apple GPU fp32 | 71.3% | 76.4% | 82.5% |
| Changed rows detected, bf16 and int8 | 100% | 100% | 100% |
| Unchanged rows flagged | 0 | 0 | 0 |
| Rows whose code changes when every logit gets +3 | 100% | 100% | 1.7% (float rounding of the shifted input) |
| Rows whose code changes when every logit is scaled by 1.1 | 100% | 0.9% | 2.2% |
| Sign symbols positive | 17.6% | 17.6% | 47.2% |
| Spearman(coordinates changed, centred L2 drift), bf16/int8 | 0.53–0.74 | 0.45–0.51 | 0.84–0.86 |
| Spearman(coordinates changed, raw L2 drift), bf16/int8 | 0.96 | 0.57–0.67 | 0.54–0.70 |

Neither L2 drift predicts the change in the output distribution: Spearman with KL ranges from −0.15 to 0.29.
None of the probes predicts a top-1 flip: AUROC 0.40–0.60.
The decoding baselines finding stands: to detect a change in the decision, measure the margin.

## Known limits

Normalising each chunk on its own ties every code in a chunk to that chunk's norm.
A change concentrated in a few logits can flip codes elsewhere in the same chunk.
That coupling is why v0.2 tracks raw per-logit drift less closely than v0.1 did.

A chunk whose centred logits are all zero has no direction and cannot be encoded.

The evidence covers one 0.5B model on one machine.
Re-check the detection table on the production-scale models before freezing.
