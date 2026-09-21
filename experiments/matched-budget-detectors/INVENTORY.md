# Matched-budget detector study — artifact inventory and readiness

Prepared 2026-09-16 for SEM-49 and SEM-81 at `origin/main` commit `fe803ac`.
This document records what exists, what can be re-derived, and what the
confirmatory run needs. It changes no result and no protocol.

Verdict in one paragraph. No confirmatory episode exists in this repository,
on any branch, on local disk, or in either S3 bucket. The only observations
are the 12-prompt pilot, which is development data scored under PROTOCOL v1.
The pilot can be re-derived from the local decoding cache; PROTOCOL.md
section 0.5 was corrected accordingly in this change. The collection stage is
implemented and tested. The analysis stage of PROTOCOL v2 (sections 2, 2.1,
3.1 and 8) has no code.

## 1. Surviving artifacts

Classification uses three labels. **Pilot** is a run on the 12-prompt legacy
cache. **Development** is any observation used while designing the protocol.
**Confirmatory** is an episode collected under PROTOCOL v2 after its freeze on
2026-09-11 (PROTOCOL.md status line). No development observation is relabeled
as held-out.

### 1.1 Committed results

| Path | Size (bytes) | SHA-256 | Produced by | Protocol | Class |
| --- | ---: | --- | --- | --- | --- |
| `results/pilot.json` | 16,844 | `c270b2d0d23106934ed31ca0c95a25d25b7da8ecea23a72509406541e0cde06c` | `run_pilot.py` at commit `6ae7927` (2026-09-09), on `/home/ubuntu/mbd/cache` | v1 | pilot |
| `results/pilot.scores.npz` | 19,931 | `2e240501f57924e215454a27c31668123eae2b8bdef202eab5f56b2a47a34e1c` | same run; 80 arrays of 12 prompt-level means (5 conditions x 16 methods) | v1 | pilot |

Both files have the same git blob (`597b37ed8c` and `d2f7fb57ba`) on
`origin/main` and on every pre-squash branch that carried them. They were
committed once, in `6ae7927` ("matched-budget: pilot results, and a
degenerate null"), two days before PROTOCOL v2 was frozen. Commit `6ae7927`
belongs to the pre-squash history archived by The SEMQ Group and does not
resolve on `origin`.

`pilot.json` records `python 3.11.16`, `numpy 2.4.6`, and `ari_meta` with
three keys (`bits_per_dim`, `coords_per_byte`, `scale`). The current
`run_pilot.py:124-125` writes five keys. The two missing keys identify the
file as output of the pilot-era scorer (section 2 below).

### 1.2 Pilot input: the local decoding cache

`experiments/decoding-reproducibility/results/cache/` is untracked and ignored
(`.gitignore`, entry `experiments/decoding-reproducibility/results/cache/`).
Files are dated 2026-09-03 on disk.

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `reference.npz`, `proc.npz`, `threads1.npz`, `batched.npz` (byte-identical) | 61,956,255 | `eccd0735ae4779184fa9cf36a6d727dbc456730b552af806e98f049d22296d46` |
| `bf16.npz` | 30,201,169 | `3560bea6872aebd888926b227f800b7f25dc8cd3b01f07ccb44b5e1f03417817` |
| `int8.npz` | 47,899,922 | `fb7d2d1995e08d78e23e2d7464f5db6d47a6972bd6558c5b43f362ba78bd3669` |

Each archive holds `logits (539, 32000) float32`, `lengths (12,) int64`,
`ref_tokens (539,) int64` and `free_tokens (12,) object`. Lengths are eleven
prompts of 48 steps and one of 11. The four identical archives match
`arrays_identical: true` for `proc`, `threads1` and `batched` in `pilot.json`. The producer is `experiments/decoding-reproducibility/run_matrix.py`
on `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, CPU (`results/decoding_matrix.json`).
No manifest exists (PROTOCOL.md section 0.2). Class: development.

### 1.3 Code and protocol

| Path | SHA-256 | Role |
| --- | --- | --- |
| `PROTOCOL.md` | `20ffc10d63e42fbd8bbcf78c847813262f714b09a023572ecb272a91af3cc64b` | v2, frozen 2026-09-11; section 0.5 corrected in this change |
| `PROTOCOL-v1.md` | `0ed97aa1b22e5393bfe4a61f5234af8d0b32caee368deeb01546b3356ccc327f` | v1, frozen 2026-09-09; scored the pilot |
| `run_pilot.py` | `f9da43bef2d72247d47fe3312e9e135713f5a3f3bed2d00a8be48fab50a489a9` | pilot scorer, SDK-backed |
| `scorers.py` | `21f3600cea1a2746b3909d7ebfa24e85ee107a3bc74bd02743f7c6f710d6da65` | detector scores and byte budgets |
| `collect_embeddings.py` | `902407835d4d683f3bc151b44c8f94138703c4f68cda4166f7c5d44e03e877cd` | one confirmatory episode per invocation |
| `tests/test_scorers.py` | `886b45d8567bdea88e4f4cd0583f27d090675ecf5a7b353a4aac335c72f29e31` | 23 tests |
| `tests/test_collect_embeddings.py` | `5e9886eb44a915213eedd3c835a3fc290855d5b5a0a96a2a7ee74e1dfa73e09c` | 7 tests, publication semantics |
| `tests/test_run_pilot.py` | `03308b1fe099a61b1bbeeb410940570f0411d7b5438f00338fc3d256f2d49e70` | 4 tests (8 cases), chunking; needs the SDK |

### 1.4 Development observations cited by PROTOCOL v2

PROTOCOL.md section 2.1 quotes SHA-256 false-alarm rates of about 0.45% per
input and 50% per deployment "on the cached SciFact panel". They come from
`experiments/regime-discrimination`, not from this study. They are
development observations, not held-out evidence, and must not enter a
confirmatory table.

### 1.5 Confirmatory artifacts

**None exist.** No tracked file on any of the 37 local branches matches
`episode`, `mbd_collect` or `/ep[0-9]`. No `mbd_collect*`, `mbd` or
`ep000.npz` exists under the home directory or the temp directories (depth
4). Both S3 buckets hold zero objects under every candidate prefix (section
6). `collect_embeddings.py:203-207` records that a full collection was lost
on 2026-09-09 because the driver wrote only to local disk; no trace of it
survives anywhere searched.

## 2. What in `pilot.json` can be re-derived

Before this change, PROTOCOL.md section 0.5 stated that the pilot's cache
was not the local cache, that the recorded scale matched neither scoring path,
and that nothing in `pilot.json` could be re-derived. All three were wrong.

### 2.1 Method

1. Run the current `run_pilot.py` (same blob on `origin/main` `fe803ac`) on
   the local cache with the SEMQ SDK `1.5.1.dev32+g80e4b2c5a`, numpy `2.4.6`,
   Python `3.11.12`. Output went to a scratch directory, not to `results/`.
2. Compare every value in the new `pilot.json` and `pilot.scores.npz` with
   the committed files.

### 2.2 Result

| Quantity | Match |
| --- | --- |
| `ari_byte_mismatch`, `ari_code_hamming`, `ari_symbol_mismatch` (all 5 conditions, 12 prompts each) | bit-identical |
| `sha256_raw`, `sha256_uniform_4bit`, `sha256_uniform_8bit` | bit-identical |
| `max_abs_diff`, `rel_l2`, `coord_mismatch`, `cosine_distance`, `margin_delta`, `token_flip`, `topk20_change` | bit-identical |
| `kl_fp64`, `js_fp64` | max abs difference 2.2e-18 |
| `sketch_rp64` | max abs difference 8.5e-14 (relative 3.9e-16) |
| `detection` table (all thresholds, TPR, achieved FPR, intervals) | identical except `pos_mean` and `control_mean` of the three rows above, at the same magnitudes |
| `storage_bytes`; `conditions.*.arrays_identical`, `changed_steps`, `status`, `role` | identical |
| `ari_meta.scale` | committed `9.414882678985599`, re-derived `9.41488265991211`; see below |

The KL, JS and sketch differences are summation-order effects of a different
BLAS build. They do not move any threshold or any alarm decision.

### 2.3 The scale

The pilot-era `_ari_scores` (commit `6ae7927`, `run_pilot.py:53`) passed
`float(np.percentile(np.abs(r), 99.0))`, computed in float64, as `scale_max`.
On the local cache that is exactly `9.414882678985599`, the value
`pilot.json` records. `Context.calibrate` on the same array returns
`9.41488265991211 == float(np.float32(9.414882678985599))`. The SDK stores
the scale as float32 (`ari/semq_compat.py:51-53`), so both paths encoded with
the same scale, and every ARI array is bit-identical.

### 2.4 Consequences

`/home/ubuntu/mbd/cache` was a copy of the local cache, so the pilot's inputs
are on this machine, though not in git. Every number in `pilot.json` can be
re-derived; the file remains as committed. This change corrected PROTOCOL.md
section 0.5, which now names the local decoding cache as the pilot's input and
states that re-scoring reproduces every array bit-for-bit. The pilot is still
pilot data: twelve prompts bound the false-alarm rate only below 22.1%.

## 3. What the confirmatory run requires

### 3.1 Targets fixed by PROTOCOL v2

- 300 control episodes and 300 per intervention (PROTOCOL.md section 3).
  `collect_embeddings.py:33-42` declares one control and six interventions
  (`tf32_on`, `batch_1`, `batch_256`, `fp16`, `bf16`, `threads_1`). Total:
  300 + 6 x 300 = **2,100 episodes**.
- One episode is one fresh process over the frozen input set under one
  configuration, with its own manifest (PROTOCOL.md section 2).
  `collect_embeddings.py` runs one episode per invocation and writes
  `<out>/<condition>/epNNN.npz` and `epNNN.json` (`collect_embeddings.py:45-47`,
  `168-201`).
- The caller interleaves controls and interventions
  (`collect_embeddings.py:7-9`). No caller exists in the repository.
- No detection statistic is inspected before all 2,100 episodes are collected
  (PROTOCOL.md section 8).

### 3.2 Instance and model

- Model: `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, the
  `collect_embeddings.py:104` default and the SciFact panel encoder
  (`experiments/regime-discrimination/run_matrix.py:57`); the embedding
  probe, `n_bins=2` (PROTOCOL.md section 0.3).
- Instance: `g5.xlarge` (A10G), the `launch.sh` default (`infra/launch.sh:44`).
  `tf32_on`, `fp16` and `bf16` are CUDA-side (`collect_embeddings.py:150-160`);
  on CPU `tf32_on` would be a second control.
- Software: `bootstrap.sh` installs torch and sentence-transformers
  (`infra/bootstrap.sh:34-36`). The SDK is needed only for scoring
  (`run_pilot.py:92`).
- Region: instance in `us-east-1` (`infra/launch.sh:30`), uploads to
  `us-east-2` by default (`collect_embeddings.py:110-111`). The operator names
  the bucket; there is no default (`collect_embeddings.py:108-109`).
- Inputs: `--inputs` is a JSONL file with a `text` field; `--n` (default 200)
  records are read (`collect_embeddings.py:105`, `116-123`).
  `data/ari-bench-v0.1.jsonl` has 1,000 records. PROTOCOL v2 names neither
  the file nor `n`. Declare both before collection.

### 3.3 Runbook

The collection commands, the durability behavior and the wall-clock and cost
estimate are in [README.md](README.md), "Run the confirmatory collection".

### 3.4 Fail-closed checklist

The SEM-49 issue body was not readable in this session. The criteria are
SEM-81's acceptance criteria plus the SEM-49 title ("run and durably
publish"). Each row names the code that satisfies it or the gap.

| Criterion | Code | Status |
| --- | --- | --- |
| Protocol frozen before collection | PROTOCOL.md status line | satisfied |
| Inventory and classification without relabeling | this document, section 1 | satisfied |
| One fresh process per episode, own manifest | `collect_embeddings.py:168-201` | satisfied per invocation |
| Interleaved conditions | `collect_embeddings.py:7-9` | **gap**: no driver; use the loop in README.md |
| Effective settings read back, not assumed | `collect_embeddings.py:181-190` | satisfied |
| Inputs bound by hash | `collect_embeddings.py:131`, `193` | satisfied; **gap**: input file and `n` not declared in PROTOCOL |
| Commit recorded in manifest | `collect_embeddings.py:198-199` | **gap**: `push_tree.sh:39` excludes `.git/`, so `git rev-parse HEAD` writes an empty string on an rsynced tree. Record the commit another way, or clone. |
| Durable publication, fail closed | `collect_embeddings.py:81-97`, `208-211` | satisfied; **gap**: episodes bucket unnamed, instance role `semq-benchmarks-ec2` (`infra/launch.sh:60`) not verified for `s3:PutObject` |
| Resumable without re-encoding | `collect_embeddings.py:54-78`, `135-145` | satisfied |
| Retain all episodes durably | S3 destination per episode | satisfied once the bucket exists |
| Score episodes: three ARI quantities via the SDK | `run_pilot.py:71-125` | **gap**: reads `reference.npz` logits (`run_pilot.py:196-199`) and hard-codes the decoding-cache condition names (`run_pilot.py:46-47`). No scorer reads `<condition>/epNNN.npz`. |
| Embedding-probe baselines: block hashes, float16 anchor, uniform quantization | `scorers.py:103-152`, `272-282` | implemented, not wired to any driver |
| Both alarm units (per input, per deployment) | PROTOCOL.md section 2.1 | **gap**: no code |
| Three disjoint roles split by input and episode | PROTOCOL.md section 2 | **gap**: no code |
| Achieved FPR at zero with one-sided `1 - 0.05^(1/N)` bound | PROTOCOL.md sections 3 and 3.1 | **gap**: `run_pilot.py:169-186` uses a quantile threshold and a two-sided Wilson interval |
| Paired episode-level contrasts, Holm-Bonferroni per probe | PROTOCOL.md section 8 | **gap**: no code (no `Holm` in the repository) |
| Exclusions listed, analyses with and without | PROTOCOL.md section 8 | **gap**: no code |
| Verification on the same frozen vectors, local vs instance | PROTOCOL.md section 6 | **gap**: no code |
| Paper tables and claims match retained evidence | `docs/paper/latex/ari.tex` | pending; see section 5 |

## 4. Readiness verdict

**Frozen and ready:** PROTOCOL v2 with v1 retained; the code in section 1.3
with its tests; provisioning with cost guards (`infra/launch.sh`,
`infra/bootstrap.sh`). Full suite on this branch: 68 passed, 10 skipped
without the SDK; the study directory plus `tests/test_sdk_surface.py`: 42
passed with the SDK.

**Missing:** the rows marked **gap** in section 3.4, plus `MAX_MINUTES=1440`
at launch (README.md). The scoring and analysis gaps can be closed and tested
on synthetic episodes without a GPU or AWS. Close them before the first real
episode is scored, so that no analysis choice follows the data (PROTOCOL.md
section 8).

**Exact next command**, on a machine with GPU access and AWS credentials,
after the collection gaps are closed:

```bash
cd infra && MAX_MINUTES=1440 ./launch.sh --dry-run && MAX_MINUTES=1440 ./launch.sh
```

Then the two blocks in README.md, "Run the confirmatory collection".

## 5. Paper paragraph if the run is not done before camera-ready

The paragraph with the bold lead-in "**Matched-budget comparison (pending).**"
exists only in the uncommitted working copy of `docs/paper/latex/ari.tex`, not
on `origin/main`. Do not apply these edits without also committing it.

1. Replace "Its pilot is degenerate: on a 12-prompt legacy cache every
   detector separates the declared interventions perfectly and none
   separates the controls" with: "Its pilot is degenerate: on a 12-prompt
   TinyLlama-1.1B CPU logit cache, every detector separates the `bf16` and
   `int8` interventions perfectly, no detector separates `threads1` or
   `batched` from the reference, and the control arrays are byte-identical to
   the reference." Reason: `pilot.json` shows TPR 0 for `threads1` and
   `batched`, whose arrays equal the reference.
2. Add after the sizing sentence: "A confirmatory collection on 2026-09-09
   was lost before publication; the collection code now publishes each
   episode as it is written and fails closed on upload error." Reason: SEM-49
   and `collect_embeddings.py:203-207`.
3. Add at the end: "No confirmatory episode has been collected at the time
   of writing. No comparison in this paper rests on the matched-budget
   study, and no ARI advantage at matched budget is claimed." Reason:
   PROTOCOL.md section 7 and SEM-81's last criterion.
4. Label any cited 0.45% and 50% SHA-256 false-alarm rates as development
   observations on the cached SciFact panel, not results of this study
   (section 1.4).
5. Keep the pilot out of every table. If a row is wanted, its denominator is
   12 prompts and its false-alarm upper bound is 22.1%.

## 6. S3 inventory

Searched on 2026-09-16 with read-only listing. Bucket names are given by
their `infra/operator.env` variable names, or by the operator-supplied
variable `$RESEARCH_RESULTS_BUCKET` for the research bucket that file does
not name. No account identifier is recorded here.

### 6.1 Buckets and totals

| Bucket | Objects | Bytes | Top-level prefixes |
| --- | ---: | ---: | --- |
| `$RESEARCH_RESULTS_BUCKET` | 3,959 | 98,452,622,870 | `cross-arch/`, `demos/`, `diagnostics/`, `embeddings/`, `kv-repair/`, `results/`, `run-logs/`, `scale-validation/`, `sdk-source/` |
| `BASELINES_BUCKET` | 32 | 302,213,475 | `ari-baselines/`, `arid-dry-run/`, `capture-sessions/` |

### 6.2 Prefixes probed

Listed recursively in both buckets, zero objects each: `matched-budget/`,
`matched-budget-detectors/`, `mbd/`, `mbd_collect/`, `episodes/`,
`detectors/`, `confirmatory/`, `pilot/`, `home/ubuntu/mbd/`, `cache/`,
`experiments/matched-budget-detectors/`. A full recursive listing of both
buckets was also filtered for `matched`, `mbd`, `episode`, `detector`,
`confirmator`, `pilot`, `ep[0-9]{3}`, `tinyllama`, `control`, `tf32`,
`reference.npz`, `proc.npz` and `manifest`.

### 6.3 Hits and classification

No filtered hit in either bucket is a matched-budget episode, an episode
manifest, or a copy of `/home/ubuntu/mbd/cache`; all belong to other projects.

### 6.4 The durability destination

`collect_embeddings.py` writes to `<publish-s3>/<condition>/epNNN.{npz,json}`
(`collect_embeddings.py:209-211`). `--publish-s3` has no default
(`collect_embeddings.py:108-109`), and `infra/operator.env.example` names only
`BASELINES_BUCKET` and `BATCH_RESULTS_BUCKET`. No prefix that the durability
change would write to exists in either bucket. The first confirmatory run
creates it.
