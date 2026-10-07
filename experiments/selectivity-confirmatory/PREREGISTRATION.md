# Selectivity at matched storage: pre-registration

Status: **signed off 2026-10-07.** No episode has been captured.
Once signed off, this file is frozen by its commit. Any later change is a dated amendment below, made before unblinding, or it is reported as a deviation.

## Question

At equal stored bytes per vector, which change detector best separates *material* changes in how an embedding model computes from *benign* ones?
The ARI-R code claims to be selective: it ignores benign floating-point jitter and flags material changes.
A byte hash flags every change, so it cannot be selective.
This study tests the claim against the strongest alternatives at the same storage.

## Labels, fixed before any data

A change is **material** when the numeric precision or quantization of the model's computation changes.
A change is **benign** when the same model computes at the same precision and only scheduling or placement differs.

| condition | label | why |
| --- | --- | --- |
| fresh process | benign | same precision, new process |
| 1 thread vs 4 | benign | same precision, different reduction order |
| batch 8 / 128 vs 32 (CPU) | benign | same precision, different shapes |
| batch 1 / 8 / 128 vs 32 (GPU, TF32 off) | benign | same precision, different kernels |
| A10G vs T4 (fp32, TF32 off) | benign | same precision, different GPU |
| CPU vs A10G (fp32, TF32 off) | benign | same precision, different device |
| TF32 on vs off (A10G) | **material** | matmul inputs rounded to a 10-bit mantissa |
| bf16 weights | material | precision change |
| fp16 weights (GPU) | material | precision change |
| int8 dynamic quantization (CPU) | material | quantization |

The hard middle cases are TF32 (material by the rule) and CPU vs GPU fp32 (benign by the rule).
Both labels were confirmed by the study owner before any capture.

## Detectors and storage

Storage is matched at `B = d / 4` bytes per vector, the size of the ARI-R code (256 bytes at `d = 1024`).

| detector | stored per vector | per-input statistic |
| --- | --- | --- |
| ARI-R code (v0.2, 2 bits per coordinate) | `B` | fraction of coordinates whose symbol changed |
| SHA-256 of the fp32 bytes | 32 bytes | 1 if any byte changed |
| block hashes | `B / 32` SHA-256 digests over contiguous coordinate blocks | fraction of blocks changed |
| fp16 prefix | the first `B / 2` coordinates in fp16 | mean absolute difference |
| random-projection sketch | `B / 4` fp32 projections, fixed seed | L2 distance |
| full fp32 (reference, not matched) | `4d` bytes | cosine distance |

## Episodes

- An episode is one fresh-process capture of all 1,000 frozen inputs for one (model, condition) pair, compared with a reference capture of the same inputs.
- The reference is fp32 on CPU, 4 threads, batch 32. For GPU-only conditions it is fp32 on the same GPU with TF32 off, and for A10G vs T4 it is the A10G capture.
- Models: nine encoders, all loaded in fp32 through `ari.st_load` with the dtype verified: the eight ARI-R encoders (all-MiniLM-L6-v2, all-mpnet-base-v2, bge-large-en-v1.5, bge-m3, multilingual-e5-large, nomic-embed-text-v1.5, mxbai-embed-large-v1, snowflake-arctic-embed-l) and google/embeddinggemma-2 (768-d, 744M parameters, checkpoint shipped in bf16).
- Size: 300 benign and 300 material episodes, balanced over models and over conditions within each label. The plan is `episodes.json`, generated with a fixed seed by `plan.py` and committed with this file.
- Repeated episodes of a deterministic condition can be byte-identical, so episodes are not independent. The unit of inference is the (model, condition) cell: 9 encoders × 6 benign conditions = 54 cells, and 9 × 4 material = 36 cells. The report states how many episodes duplicated an earlier one exactly.
- The episode score is the detector's mean per-input statistic.

## Analysis, fixed

1. Split the (model, condition) cells 50/50 into calibration and test sets, stratified by condition, with a fixed seed. Every episode of a cell goes to the same half, so no cell informs both the threshold and the test.
2. For each detector, set the threshold to the largest score on a calibration benign episode, so it raises no alarm on any calibration benign episode.
3. On the test half, report:
   - **Primary:** the fraction of material episodes flagged, with a 95% interval from a cluster bootstrap over cells (10,000 resamples). The episode-level Wilson interval is reported as secondary.
   - The fraction of benign episodes flagged.
   - The material detection rate per condition, with TF32 reported separately.
   - Both alarm units, per episode and per input.
4. **The claim the paper may make:** the ARI-R code is more selective than byte hashing if the lower bound of its primary rate exceeds the upper bound for SHA-256 and for block hashes.
   The comparison with the sketch and the fp16 prefix is reported as measured, with no claim either way.
   A tie is reported as a tie.

## Environment

- semq 1.0.0 for every model.
- The eight ARI-R encoders: torch 2.6.0, transformers 4.57.6, sentence-transformers 4.1.0 (nomic's remote code does not load under transformers 5).
- google/embeddinggemma-2: torch 2.14.1, transformers 5.19.0, sentence-transformers 6.1.0, torchvision 0.29.1, Pillow (its architecture needs transformers 5.19). Every comparison is within one model, so each model's cells share one stack.
- CPU episodes run on c7i.2xlarge. GPU episodes run on g5.xlarge (A10G) and g4dn.xlarge (T4).
- Raw vectors are kept for every episode so any detector can be recomputed.

## Amendments

None.
