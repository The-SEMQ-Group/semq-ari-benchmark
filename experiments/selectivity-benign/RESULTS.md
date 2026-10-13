# Benign alarms at matched storage: results

Pre-registration: [PREREGISTRATION.md](PREREGISTRATION.md), frozen at `72f1247` (2026-10-12 22:01 UTC) before any capture; one amendment before capture (`4830384`, GPU regions).
Analysis: `analyze.py`, unchanged since `72f1247`, run once on 2026-10-13 on the full data.
Output: [results/results.json](results/results.json).

## Data

- 449 of 450 planned captures. Every capture's recorded settings match its frozen spec in `episodes.json`.
- **One capture failed after its retry:** `batch512_gpu-24`, Qwen3-Embedding-0.6B at batch 512 on the A10G. It ran out of GPU memory both times.
- **Qwen3-Embedding-0.6B is therefore dropped whole**, as pre-registered. The analysis covers 24 models: 336 benign and 48 material cells.
- No capture contained a non-finite vector.

## Primary result

| detector | stored per vector | benign cells alarmed | 95% Wilson | material cells caught |
| --- | --- | --- | --- | --- |
| ARI-R code | d/4 bytes | **0 / 336** | [0.000, 0.011] | 48 / 48 |
| random-projection sketch | d/4 bytes | 10 / 336 | [0.016, 0.054] | 48 / 48 |
| fp16 prefix | d/4 bytes | 20 / 336 | [0.039, 0.090] | 48 / 48 |
| full fp32 cosine (not matched) | 4d bytes | 23 / 336 | [0.046, 0.101] | 48 / 48 |
| SHA-256, block hashes | 32 bytes, d/4 bytes | 0 / 336 | [0.000, 0.011] | 0 / 48 |

Paired comparison on the same benign cells, one-sided exact sign test:

| code vs | only the other alarmed | only the code alarmed | p | Holm p |
| --- | --- | --- | --- | --- |
| sketch | 10 | 0 | 0.00098 | 0.00098 |
| fp16 prefix | 20 | 0 | 9.5e-07 | 1.9e-06 |
| full cosine (secondary) | 23 | 0 | 1.2e-07 | n/a |

**Pre-registered claim: supported for both matched-storage detectors.** At equal stored bytes, the ARI-R code raised fewer benign alarms than the sketch and the fp16 prefix. It also caught every material cell.

**Secondary results:**
- The upper bound of the code's benign interval is 1.1%, below the pre-registered 5%.
- Every material cell (TF32 and bf16 on 24 models) was caught by every continuous detector, so the thresholds fixed in the first study carried over to sixteen unseen models.

## Read with care

- **Zero margin on one cell.** The largest benign code score equals the threshold exactly: 5.208e-06 in `cpu_vs_a10g-00`, all-MiniLM-L6-v2 on CPU vs A10G. The cell does not alarm because the rule is strict (score > threshold).
  - This is the same model and condition whose score set the threshold in the first study. Both captures are deterministic and gave the same value, so it repeats that calibration cell rather than being a near miss on new data.
  - Even so, the code has no headroom on that cell. A threshold rule that is less tight would be more robust.
- **The code reacts to some benign changes, below its threshold.** 81 of the 336 benign cells changed at least one code symbol. The code's selectivity comes from its threshold, not from codes never changing.
- **The smallest material score is about 25 times the threshold** (1.28e-04).
- **Where the other detectors' alarms fell.** They concentrate on moves between devices (CPU vs GPU, A10G vs T4 or L4) and a few models: multilingual-e5-base, e5-large-v2, bge-m3 and all-mpnet-base-v2.

## Deviations

- One capture failed (above). It was handled by the pre-registered rule.
- The A10G capture ran in us-west-2 and the L4 in us-east-2, per the amendment made before capture.
