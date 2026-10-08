# Selectivity at matched storage: results

Pre-registration: [PREREGISTRATION.md](PREREGISTRATION.md), frozen at `565aa57` (2026-10-07 12:44 UTC) before any capture.
Analysis: `analyze.py`, unchanged since `e1b1dbe` (2026-10-07 12:46 UTC), run once on 2026-10-08 on the full data.
Output: [results/results.json](results/results.json).

## Data

- 600 episodes and 18 references, all captured. Every capture matches its frozen spec in `episodes.json`.
- One host per machine: A10G episodes on one g5.xlarge, T4 episodes on one g4dn.xlarge, CPU episodes and CPU references on one c7i.2xlarge.
- Stacks: transformers 4.57.6 for the eight ARI-R encoders, 5.19.0 for google/embeddinggemma-2.
- 486 of 600 episodes exactly duplicate an earlier episode of the same cell, as expected for deterministic conditions. The unit of inference is the cell: 54 benign and 36 material cells, of which 30 benign and 20 material are in the test half.

## Primary result

Test half. Thresholds are set at zero alarms on the calibration benign cells.

| detector | stored per vector | material flagged | 95% CI (cluster) | benign flagged | 95% CI (cluster) |
| --- | --- | --- | --- | --- | --- |
| ARI-R code | d/4 bytes | 1.000 | [1.000, 1.000] | 0.000 | [0.000, 0.000] |
| SHA-256 | 32 bytes | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] |
| block hashes | d/4 bytes | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] |
| fp16 prefix | d/4 bytes | 1.000 | [1.000, 1.000] | 0.071 | [0.000, 0.176] |
| random-projection sketch | d/4 bytes | 1.000 | [1.000, 1.000] | 0.035 | [0.000, 0.107] |
| full fp32 cosine (reference, not matched) | 4d bytes | 1.000 | [1.000, 1.000] | 0.106 | [0.000, 0.241] |

Episode-level Wilson intervals for the material rate: [0.977, 1.000] for the code, sketch, fp16 prefix and cosine; [0.000, 0.023] for both hashes. Every material condition (TF32, bf16, fp16, int8) is flagged at 1.000 by every detector that flags material at all.

**Pre-registered claim: supported.** The lower bound of the code's material rate (1.000) exceeds the upper bound for SHA-256 and for block hashes (0.000). Both hashes reach a threshold of 1.0 because benign cross-device conditions change every vector's bytes, so they can raise no alarm on any run without also alarming on benign ones.

**Against the continuous detectors: no claim, as pre-registered.** All catch every material episode. Their benign alarm rates (0.035–0.106) have intervals that include 0, so the code's 0.000 is not distinguishable from them at this sample size.

**Input-level unit (secondary).** At the per-input threshold, the code flags 0.529 of material inputs and 0.000 of benign inputs. The continuous detectors flag 1.000 of material inputs and 0.0001–0.0002 of benign inputs. The code is a threshold detector: it catches every material run, not every changed vector.

## Deviations (operational, no change to the analysis)

- The CPU machine ran two captures in parallel, ran out of memory at 2026-10-08 00:04 UTC and hung. It was rebooted on the same host and resumed one capture at a time.
- Two captures killed by the out-of-memory event (`int8-028`, `threads1-011`) were captured again after the reboot. The machine's `STATUS` file lists them as failed because the failure log is cumulative.

## Exploratory (not pre-registered)

The benign test alarms of the continuous detectors fall in one to three cells, each counted six times because of duplicate episodes. The code raised no alarm in any of them.

| detector | benign test cells that alarmed |
| --- | --- |
| sketch | bge-m3, CPU vs A10G |
| fp16 prefix | bge-m3, CPU vs A10G; bge-m3, A10G vs T4 |
| full cosine | bge-m3, CPU vs A10G; bge-m3, A10G vs T4; all-mpnet-base-v2, GPU batch size |
