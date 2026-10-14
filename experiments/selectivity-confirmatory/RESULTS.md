# Selectivity at matched storage: results

Pre-registration: [PREREGISTRATION.md](PREREGISTRATION.md), frozen at `565aa57` (2026-10-07 12:44 UTC) before any capture.
Analysis: `analyze.py` as committed at `e1b1dbe` (2026-10-07 12:46 UTC), run once on 2026-10-08 on the full data.
Output: [results/results.json](results/results.json).
Changed after publication (2026-10-14), with no change to any pre-registered output on the full data; see [Changes after publication](#changes-after-publication).

## Data

- 600 episodes and 18 references, all captured. Every capture matches its frozen spec in `episodes.json`.
- One host per machine: A10G episodes on one g5.xlarge, T4 episodes on one g4dn.xlarge, CPU episodes and CPU references on one c7i.2xlarge.
- Stacks: transformers 4.57.6 for the eight ARI-R encoders, 5.19.0 for google/embeddinggemma-2.
- 486 of 600 episodes exactly duplicate an earlier episode of the same cell, as expected for deterministic conditions. The unit of inference is the cell: 54 benign and 36 material cells, of which 30 benign and 20 material are in the test half. Repeated episodes add no independent cells.
- [results/capture_manifest.json](results/capture_manifest.json) lists every capture with the SHA-256 of its archived `.npy` file, the archive prefix, the host and the library versions. The vectors are in the operator's private bucket (`infra/operator.env`, not committed).
- [results/episode_scores.json](results/episode_scores.json) gives every episode's score for each detector, its cell and its half, so the thresholds and rates below can be checked without the vectors.
- Not recorded at capture time: the resolved model and tokenizer revisions and the harness source revision. Models were loaded by id without a revision pin. `capture_episode.py` records all three for later captures, and stops if the model and tokenizer revisions it resolves differ.

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

> **Note added 2026-10-12, after publication of these results (no change to the analysis).**
> The code's benign interval above, [0.000, 0.000], understates the uncertainty. A cluster
> bootstrap of 30 cells that are all zero can only return zero. For 0 alarming cells out of
> 30, the 95% Wilson upper bound is 0.114, which overlaps the continuous detectors' intervals.
> Their benign alarms come from 1 cell (sketch), 2 cells (fp16 prefix) and 3 cells (full cosine).
> In a paired comparison on the same test cells, every disagreement favours the code: 1, 2
> and 3 cells. An exact sign test gives p = 1.0, 0.5 and 0.25 (two-sided), so none is
> significant. The episode-level Wilson intervals reported in `results.json` (for example
> [0.016, 0.075] for the sketch) treat duplicate episodes of one cell as independent, so they
> are too narrow. The cell is the unit of inference. A follow-up pre-registered study with
> more benign cells tests the comparison directly.

**Cell-level intervals (not pre-registered).** The cluster bootstrap above returns a zero-width interval whenever every test cell has the same outcome, so [1.000, 1.000] and [0.000, 0.000] are not uncertainty bounds. A Wilson interval over cells counts each cell once. A material cell counts as detected only if every episode was flagged, and a benign cell counts as alarming if any episode alarmed.

| detector | material cells detected | 95% Wilson | benign cells alarming | 95% Wilson |
| --- | --- | --- | --- | --- |
| ARI-R code | 20 / 20 | [0.839, 1.000] | 0 / 30 | [0.000, 0.114] |
| SHA-256 | 0 / 20 | [0.000, 0.161] | 0 / 30 | [0.000, 0.114] |
| block hashes | 0 / 20 | [0.000, 0.161] | 0 / 30 | [0.000, 0.114] |
| fp16 prefix | 20 / 20 | [0.839, 1.000] | 2 / 30 | [0.018, 0.213] |
| random-projection sketch | 20 / 20 | [0.839, 1.000] | 1 / 30 | [0.006, 0.167] |
| full fp32 cosine | 20 / 20 | [0.839, 1.000] | 3 / 30 | [0.035, 0.256] |

With these intervals the pre-registered comparison still holds: the code's lower bound (0.839) exceeds the hashes' upper bound (0.161). Against the continuous detectors the benign intervals overlap, as the note above says.

**Input-level unit (secondary).** At the per-input threshold, the code flags 0.529 of material inputs and 0.000 of benign inputs. The continuous detectors flag 1.000 of material inputs and 0.0001–0.0002 of benign inputs. The code is a threshold detector: it catches every material run, not every changed vector.

## Deviations (operational, no change to the analysis)

- The CPU machine ran two captures in parallel, ran out of memory at 2026-10-08 00:04 UTC and hung. It was rebooted on the same host and resumed one capture at a time.
- Two captures killed by the out-of-memory event (`int8-028`, `threads1-011`) were captured again after the reboot. The machine's `STATUS` file lists them as failed because the failure log is cumulative.

## Changes after publication

Made on 2026-10-14 after review. Rerunning `analyze.py` on the full data reproduces every pre-registered field of `results.json` exactly.

- The calibration/test split is now derived from the complete plan in `episodes.json`. It used to be derived from the episodes present, so an incomplete copy of the data could move a cell between halves. A missing capture now stops the analysis unless `--allow-missing` is given, which drops the missing episodes and keeps the split. On the full data the split is the same.
- `results.json` adds `cells_counted`, `ci_cell_wilson` and `claim_supported_cell_wilson`, listed under `not_preregistered`.
- `analyze.py` writes `episode_scores.json`, and `capture_episode.py` records provenance (above).

## Exploratory (not pre-registered)

Nothing in this section is a pre-registered claim. A later pre-registered study with more benign cells (the benign-alarm study, on `ilona/v02-reruns`) has its own scope and is reported separately; it adds evidence on the comparison with the continuous detectors and does not replace this study's pre-registered result.

The benign test alarms of the continuous detectors fall in one to three cells, each counted six times because of duplicate episodes. The code raised no alarm in any of them.

| detector | benign test cells that alarmed |
| --- | --- |
| sketch | bge-m3, CPU vs A10G |
| fp16 prefix | bge-m3, CPU vs A10G; bge-m3, A10G vs T4 |
| full cosine | bge-m3, CPU vs A10G; bge-m3, A10G vs T4; all-mpnet-base-v2, GPU batch size |
