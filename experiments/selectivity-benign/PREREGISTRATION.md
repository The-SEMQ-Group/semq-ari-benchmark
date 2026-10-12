# Benign alarms at matched storage: pre-registration

Status: **frozen 2026-10-12, before any capture.** No capture for this study exists.
Any later change is a dated amendment below, made before unblinding, or it is reported as a deviation.

## Why this study

The first selectivity study (`../selectivity-confirmatory`) found that the ARI-R code and the matched-storage continuous detectors all flag every material change.
The code raised no benign alarm in 30 test cells. The sketch raised alarms in 1 cell and the fp16 prefix in 2.
That difference is not significant: an exact sign test on the paired cells gives p = 1.0 and 0.5.
With 0 alarms in 30 cells, the code's own 95% upper bound is 11%.
This study asks the open question directly, with enough benign cells to answer it.

## Question

At equal stored bytes per vector, does the ARI-R code raise fewer false alarms on benign changes than the random-projection sketch and the fp16 prefix?

## Labels

The labels and the rule are the first study's.
A change is **benign** when the same model computes at the same precision and only scheduling, shapes or device differ.
A change is **material** when the precision or quantization of the computation changes. TF32 is material.

## Design

- **Unit:** the cell, one (model, condition) pair, with one fresh-process capture of all 1,000 frozen inputs compared against a reference capture.
- **Models (25):** the nine of the first study, and sixteen more:
  - all-MiniLM-L12-v2, paraphrase-multilingual-MiniLM-L12-v2, multi-qa-mpnet-base-dot-v1;
  - bge-small-en-v1.5, bge-base-en-v1.5;
  - e5-small-v2, e5-base-v2, e5-large-v2, multilingual-e5-base;
  - gte-small, gte-base, gte-large;
  - snowflake-arctic-embed-s, snowflake-arctic-embed-m;
  - granite-embedding-small-english-r2;
  - Qwen3-Embedding-0.6B.
  
  All are loaded in fp32 through `ari.st_load`, with the dtype checked.
- **Benign conditions (14), 350 cells:**
  - CPU against the CPU reference (fp32, 4 threads, batch 32): fresh process, 1 thread, batch 8, batch 128.
  - A10G against the A10G reference (fp32, TF32 off, batch 32): fresh process, batch 1, 8, 128, 512.
  - Across devices, fp32 with TF32 off, batch 32: CPU vs A10G, CPU vs T4, CPU vs L4, A10G vs T4, A10G vs L4.
- **Material conditions (secondary), 50 cells:** TF32 on and bf16 weights, on A10G against the A10G reference.
- **The plan:** `episodes.json`, written by `plan.py` and committed with this file. Each model's CPU captures run on one CPU host. GPU captures run on one g5.xlarge (A10G), one g4dn.xlarge (T4) and one g6.xlarge (L4).

## Detectors and thresholds, fixed

The detectors are the first study's, imported unchanged from its frozen `analyze.py`.

**No threshold is fit in this study.** Each detector's threshold is the one the first study set on its calibration half (commit `76f407f`):

| detector | threshold |
| --- | --- |
| ARI-R code | 5.208333e-06 |
| fp16 prefix | 4.594074e-08 |
| random-projection sketch | 1.959521e-06 |
| full fp32 cosine | 2.074719e-12 |
| SHA-256, block hashes | 1.0 |

Every cell in this study is therefore a test cell.
Sixteen of the 25 models were not seen when the thresholds were set, so the test also covers out-of-sample models.

## Analysis, fixed (`analyze.py`)

1. **Scores and alarms.** For each cell, compute each detector's mean per-input statistic. A cell alarms when its score exceeds the threshold.
2. **Primary test**, for D in {sketch, fp16 prefix}, on the benign cells:
   - let b = the number of cells where D alarms and the code does not;
   - let c = the number of cells where the code alarms and D does not;
   - run a one-sided exact sign test, H1: the code alarms on fewer cells, with p = P(X ≥ b), X ~ Binomial(b + c, 1/2);
   - adjust the two p-values with Holm's method at α = 0.05.
3. **Claim.** The paper may state that the code raises fewer benign alarms than D at matched storage only for a D whose Holm-adjusted p < 0.05. Otherwise the comparison is reported as measured, with no claim.
4. **Reported for every detector:**
   - benign and material alarm counts and rates, with Wilson 95% intervals over cells;
   - alarms by condition;
   - the list of alarming cells.
5. **Secondary:**
   - the same paired test against full fp32 cosine (16× the storage), reported without a claim;
   - whether the upper bound of the code's benign Wilson interval falls below 5%;
   - material detection on the 50 material cells, to check that the fixed thresholds still catch real changes on new models.
6. **Missing data.** A failed capture is retried once. A model with any missing cell after retry is dropped whole, so that all detectors are compared on the same cells, and the drop is reported. No model is replaced.
7. **Size.** The size is fixed at 25 models × 16 conditions. No cells are added after seeing results.

## Expected sensitivity

With 350 benign cells, suppose the sketch's benign rate is near the 3.5% seen in the first study and the code's is near 0. Then about 12 discordant cells are expected, and the test is very likely to reach significance.

Five discordant cells all favouring the code give p = 0.031 before adjustment. Six give p = 0.016, which survives Holm.

If the continuous detectors' benign alarms were specific to bge-m3 moving across devices, there may be too few discordant cells to reach significance. That outcome will be reported as no difference shown.

## Environment

- semq 1.0.0.
- 24 models: torch 2.6.0, transformers 4.57.6, sentence-transformers 4.1.0.
- google/embeddinggemma-2: torch 2.14.1, transformers 5.19.0, sentence-transformers 6.1.0.
- CPU captures on two c7i.2xlarge hosts (models assigned in `plan.py`). GPU captures on g5.xlarge, g4dn.xlarge and g6.xlarge in us-east-1.
- Raw float32 vectors are kept for every capture.

## Amendments

- **2026-10-12, before any capture: GPU regions.** us-east-1 had no g5.xlarge or g6.xlarge capacity at launch. The A10G runs in us-west-2 and the L4 in us-east-2; the T4 and both CPU hosts are in us-east-1. Instance types, software and analysis are unchanged.
