# Retrieval outcomes study. Pilot plan

**Status: planned. Not run.**

The pilot checks feasibility, timing and implementation before
[PROTOCOL.md](PROTOCOL.md) is frozen. It does not estimate effects and it does
not set thresholds. Pilot episodes are never pooled with confirmatory episodes.

## Purpose

1. Measure the encode time per episode on the CPU host.
2. Check the cache manifest, the reference Recall@10 and nDCG@10.
3. Exercise every scorer in PROTOCOL.md section 2.3 against `R0` and `R_idx`.
4. Verify the shared-rotation invariance checks in PROTOCOL.md section 4.4.
5. Confirm that the synthetic arms produce instances on both sides of the
   preregistered loss, so the confirmatory run can have confirmed-degraded
   instances.
6. Produce a wall-clock estimate for the confirmatory run.

## Prerequisites

- Repository root as working directory.
- The development environment from the root README, plus `torch`,
  `sentence-transformers`, `datasets` and the `semq` SDK.
- `faiss-cpu`, if the owner accepts it (PROTOCOL.md section 13, decision 2).
- Read access to the regime-discrimination cache, or time to regenerate it.

## Pilot cells

| Arm | Instances | Replicates | Episodes | Encodes corpus |
| --- | ---: | ---: | ---: | --- |
| A0 `proc` | 12 | 1 | 12 | yes |
| A1 `threads1`, `batch8`, `batch128` | 3 | 2 | 6 | yes |
| B1 `bf16-q`, `int8-q` | 2 | 2 | 4 | yes, as probe inputs through the query path; index kept |
| B2 `swap-q` | 1 | 1 | 1 | yes, as probe inputs through the query path; index kept |
| C1 `bf16`, `int8` | 2 | 2 | 4 | yes |
| C2 `swap` | 1 | 1 | 1 | yes |
| S1 `noise-both` | 5 σ × 1 seed | 1 | 5 | no |
| S2 `noise-q` | 5 σ × 1 seed | 1 | 5 | no |
| S3 `rot-shared` | 6 | 1 | 6 | no |
| S4 `rot-doc`, `rot-q` | 3 + 3 | 1 | 6 | no |

Total 50 episodes. 28 run the encoder. Of these, 23 encode the corpus and
rebuild the index. The other 5 (arms B) encode the corpus as probe inputs
through a changed query path and keep the reference index. 22 are synthetic.
12 controls bound the false alarm rate only below 22.1% (`1 − 0.05^(1/12)`),
which is why the pilot supports no alarm-rate claim.

## Compute estimate

No encode time is recorded in `regime_matrix.json`. The pilot measures it in
step 3 below. Until then the estimate uses a placeholder `t_enc`, the seconds to
encode 5,183 documents and 300 queries under the reference configuration in a
fresh process, including model load. Published throughput for
`all-MiniLM-L6-v2` on a 4-thread CPU at sequence length 256 is in the range
50 to 150 documents per second, which gives `t_enc` between 40 and 110 seconds.
Two worked values are shown.

Per-episode components:

| Stage | Estimate | Basis |
| --- | --- | --- |
| Encode corpus and queries | `t_enc` | measured in step 3 |
| Encode probe inputs through query path only (arm B) | `t_enc`, same size as corpus | probe set is the corpus |
| Synthetic transform on cached arrays | 2 s | 5,183 × 384 float64 matmul and cast |
| Fresh process and imports | 3 s | numpy, semq |
| SEMQ codes, 5,183 × 384, two probes | 1 s | drift-sensitivity: the 11-point σ sweep, each point encoding the corpus once, runs in "a few seconds per model" on a CPU, so one point is under 1 s |
| Scorers against two references | 2 s | vector operations on 5,183 × 384 |
| Exact retrieval and 10,000-resample paired bootstrap | 1 s | 300 × 5,183 matmul; 10,000 × 300 index array |
| HNSW build and search, rotation arms only | 3 s | 5,183 points, M = 32 |
| Alignment fit and evaluation, degraded instances only | 1 s | 500 × 384 Procrustes |

Pilot arithmetic:

- Real episodes: 23 corpus encodes + 5 query-path encodes = 28 × `t_enc`.
  Plus 28 × 7 s scoring = 196 s.
- Synthetic episodes: 22 × (3 + 2 + 1 + 2 + 1) s = 198 s, plus 12 × 3 s HNSW = 36 s.
- At `t_enc` = 60 s: 28 × 60 + 196 + 234 = 2,110 s, about 35 minutes.
- At `t_enc` = 120 s: 28 × 120 + 196 + 234 = 3,790 s, about 63 minutes.
- Add the first-run download of three alternative encoders, under 5 minutes.
- bf16 on a CPU without the matching kernels can take over ten minutes per
  encode (`run_matrix.py`, comment in `main`). The stop rule covers this.

Confirmatory arithmetic, from PROTOCOL.md section 9.3, CPU arms only:

- Corpus encodes: A0 360 + A1 3 + C1 2 + C2 15 = 380.
- Query-path encodes: B1 60 + B2 15 = 75. Total 455 × `t_enc`.
- Scoring on real episodes: 455 × 7 s = 3,185 s.
- Synthetic episodes: 330 × 9 s = 2,970 s, plus 210 × 3 s HNSW = 630 s.
- Remedy evaluation, PROTOCOL.md section 8: every degraded (confirmed)
  evaluation instance and a matched sample of not-degraded instances of the
  same arms. The arms expected to degrade are S4 and B2. The evaluation half
  holds 75 of the 150 S4 instances in expectation and at most 3 B2 instances.
  Every evaluation instance of these arms enters either the degraded set or the
  matched sample, so the remedy set is at most 78 instances. Cost: 78 × `t_enc`
  for the rebuild, plus 78 × 2 s for rollback and alignment. On synthetic arms
  the rebuild is a 2 s transform, so `t_enc` is an upper bound.
- At `t_enc` = 60 s: 455 × 60 + 3,185 + 3,600 + 78 × 62 = 38,921 s, about 10.8 hours.
- At `t_enc` = 120 s: 455 × 120 + 3,185 + 3,600 + 78 × 122 = 70,901 s, about 19.7 hours.

G1 needs a GPU host and is not in this estimate.

## Procedure

1. Check out the frozen commit. Confirm the cache manifest with
   `write_cache_manifest` from `run_matrix.py`. A mismatch stops the pilot.
2. Recompute the reference retrieval metrics. Success: Recall@10 = 0.7833 and
   nDCG@10 = 0.6451 to four decimals. A mismatch stops the pilot.
3. Run one A0 episode. Record `t_enc`, host, thread count and package versions.
   Repeat once. Use the mean as `t_enc` in this document.
4. Run the remaining A0, A1, B1 and C1 episodes. Record elapsed seconds per stage.
5. Run one B2 and one C2 episode with `all-MiniLM-L12-v2`. Record Hub revisions.
6. Run S3. For each episode record ‖QᵀQ − I‖_max, max |ΔS| in float64 and
   float32, exact top-10 identical rate, ANN top-10 identical rate, and
   ΔRecall@10 for both. Success: every check in PROTOCOL.md section 4.4
   step 2 to 4 passes.
7. Run S4, S1 and S2. Record the outcome label of each episode.
8. Compute every scorer against `R0` and against `R_idx` for every episode.
   Success: no scorer error, every rate carries its denominator.
9. Run the four remedies on every S4 episode. Record cost per remedy.
10. Write `results/pilot/` with one manifest per episode and one summary JSON.
11. Fill in the measured `t_enc` and the confirmatory estimate above.

## Success checks

- Steps 1 and 2 pass without change to the cache.
- `t_enc` is recorded from two A0 episodes.
- All S3 invariance checks pass.
- At least one S4 episode is labelled degraded (confirmed) and at least one S3
  episode is labelled not degraded. This confirms both outcome labels are
  reachable on the frozen inputs.
- Every scorer runs against both references on every episode.

## Stop rule

Stop the pilot and record the reason when any of the following occurs:

1. The manifest or the reference metrics do not match (steps 1 and 2).
2. An S3 episode fails a check in PROTOCOL.md section 4.4 step 2 to 4. Fix the
   implementation before any further episode.
3. One encode exceeds 600 seconds on the CPU host. Move that arm to a host with
   the matching kernels and record the host change, or drop the arm and record
   the drop.
4. No S4 episode is labelled degraded (confirmed), or no S3 episode is labelled
   not degraded. Then the 300-query set cannot separate the outcome labels on
   this corpus. Escalate PROTOCOL.md section 13 decision 3, the secondary
   corpus, before freezing.
5. The pilot passes 3 hours of wall-clock time on the CPU host. Record the
   completed cells and the measured `t_enc`. The confirmatory estimate is then
   recomputed before freezing.
