# Matched-budget detector study — artifact inventory and readiness

Prepared 2026-09-16 for SEM-49 and SEM-81 at repository commit `73139c8`.
This document records what exists, what can be re-derived, and what the
confirmatory run needs. It changes no result and no protocol.

Verdict in one paragraph. No confirmatory episode exists in this repository,
on any branch, on local disk, or in either S3 bucket. The only observations
are the 12-prompt pilot, which is development data scored under PROTOCOL v1.
The pilot can be re-derived from the local decoding cache, which corrects
PROTOCOL.md §0.5. The collection stage is implemented and tested. The
analysis stage of PROTOCOL v2 (§2, §2.1, §3.1, §8) has no code.

## 1. Surviving artifacts

Classification uses three labels. **Pilot** is a run on the 12-prompt legacy
cache. **Development** is any observation used while designing the protocol.
**Confirmatory** is an episode collected under PROTOCOL v2 after its freeze on
2026-09-11 (`PROTOCOL.md:3`). No development observation is relabeled as
held-out.

### 1.1 Committed results

| Path | Size (bytes) | SHA-256 | Produced by | Protocol | Class |
| --- | ---: | --- | --- | --- | --- |
| `results/pilot.json` | 16,844 | `c270b2d0d23106934ed31ca0c95a25d25b7da8ecea23a72509406541e0cde06c` | `run_pilot.py` at commit `6ae7927` (2026-09-09), on `/home/ubuntu/mbd/cache` | v1 | pilot |
| `results/pilot.scores.npz` | 19,931 | `2e240501f57924e215454a27c31668123eae2b8bdef202eab5f56b2a47a34e1c` | same run; 80 arrays of 12 prompt-level means (5 conditions x 16 methods) | v1 | pilot |

Both files have the same git blob on every branch that carries them
(`597b37ed8c` and `d2f7fb57ba`): `origin/main`,
`experiment/matched-budget-detectors`, `backup/sem-49-preflight`,
`ilona/sem-48-protocol-v2`, `ilona/sem-49-confirmatory-prep`,
`ilona/sem-49-episode-durability`, `ilona/sem-49-sdk-authority` and
`ilona/sdk-only-probe-and-paper`. They were committed once, in `6ae7927`
("matched-budget: pilot results, and a degenerate null"), two days before
PROTOCOL v2 was frozen.

`pilot.json` records `python 3.11.16`, `numpy 2.4.6`, and `ari_meta` with
three keys (`bits_per_dim`, `coords_per_byte`, `scale`). The current
`run_pilot.py:124-125` writes five keys. The two missing keys identify the
file as output of the pilot-era scorer (§2 below).

### 1.2 Pilot input: the local decoding cache

`experiments/decoding-reproducibility/results/cache/` is untracked and ignored
(`.gitignore`, entry `experiments/decoding-reproducibility/results/cache/`).
Files are dated 2026-09-03 on disk.

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `reference.npz` | 61,956,255 | `eccd0735ae4779184fa9cf36a6d727dbc456730b552af806e98f049d22296d46` |
| `proc.npz` | 61,956,255 | same as `reference.npz` |
| `threads1.npz` | 61,956,255 | same as `reference.npz` |
| `batched.npz` | 61,956,255 | same as `reference.npz` |
| `bf16.npz` | 30,201,169 | `3560bea6872aebd888926b227f800b7f25dc8cd3b01f07ccb44b5e1f03417817` |
| `int8.npz` | 47,899,922 | `fb7d2d1995e08d78e23e2d7464f5db6d47a6972bd6558c5b43f362ba78bd3669` |

Each archive holds `logits (539, 32000) float32`, `lengths (12,) int64`,
`ref_tokens (539,) int64` and `free_tokens (12,) object`. Lengths are eleven
prompts of 48 steps and one of 11. Four archives are byte-identical, which
matches `arrays_identical: true` for `proc`, `threads1` and `batched` in
`pilot.json`. The producer is `experiments/decoding-reproducibility/run_matrix.py`
on `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, CPU (`results/decoding_matrix.json`).
No manifest exists (`PROTOCOL.md:34-40`). Class: development.

### 1.3 Code and protocol

| Path | SHA-256 | Role |
| --- | --- | --- |
| `PROTOCOL.md` | `766a052b4bcd36c7008110520b2f508e6fd43c862874fa2791dc2f64e396b81a` | v2, frozen 2026-09-11 |
| `PROTOCOL-v1.md` | `0ed97aa1b22e5393bfe4a61f5234af8d0b32caee368deeb01546b3356ccc327f` | v1, frozen 2026-09-09; scored the pilot |
| `run_pilot.py` | `f9da43bef2d72247d47fe3312e9e135713f5a3f3bed2d00a8be48fab50a489a9` | pilot scorer, SDK-backed since PR #18 |
| `scorers.py` | `21f3600cea1a2746b3909d7ebfa24e85ee107a3bc74bd02743f7c6f710d6da65` | detector scores and byte budgets |
| `collect_embeddings.py` | `902407835d4d683f3bc151b44c8f94138703c4f68cda4166f7c5d44e03e877cd` | one confirmatory episode per invocation |
| `tests/test_scorers.py` | `886b45d8567bdea88e4f4cd0583f27d090675ecf5a7b353a4aac335c72f29e31` | 23 tests |
| `tests/test_collect_embeddings.py` | `5e9886eb44a915213eedd3c835a3fc290855d5b5a0a96a2a7ee74e1dfa73e09c` | 7 tests, publication semantics |
| `tests/test_run_pilot.py` | `03308b1fe099a61b1bbeeb410940570f0411d7b5438f00338fc3d256f2d49e70` | 4 tests (8 cases), chunking; needs the SDK |

### 1.4 Development observations cited by PROTOCOL v2

`PROTOCOL.md:146-151` quotes a per-input SHA-256 false-alarm rate of about
0.45% and a per-deployment rate of 50% "on the cached SciFact panel". Those
numbers come from `experiments/regime-discrimination`, not from this study.
They are development observations that motivated §2.1. They are not held-out
evidence and must not enter a confirmatory table.

### 1.5 Confirmatory artifacts

**None exist.** Evidence:

- Git: no tracked file on any of the 37 local branches matches `episode`,
  `mbd_collect` or `/ep[0-9]`. The six study branches listed in §1.1 carry
  only the files in §1.1 and §1.3.
- Local disk: no `mbd_collect*`, `mbd` or `ep000.npz` under the home
  directory or the temp directories (depth 4).
- S3: see §6. Zero objects under every candidate prefix in both buckets.
- `collect_embeddings.py:203-207` records that a full collection was lost on
  2026-09-09 because the driver wrote only to local disk. No trace of that
  collection survives anywhere searched.

## 2. What in `pilot.json` can be re-derived

PROTOCOL.md §0.5 (`PROTOCOL.md:94-98`) states that the pilot's cache
`/home/ubuntu/mbd/cache` "is not the cache above", that the recorded scale
`9.414882678985599` "is not what either path produces on the cache that is
here", and that "nothing in `pilot.json` can be re-derived". The check below
shows that all three statements are wrong.

### 2.1 Method

1. Run the current `run_pilot.py` (commit `73139c8`) on the local cache with
   the SEMQ SDK `1.5.1.dev32+g80e4b2c5a`, numpy `2.4.6`, Python `3.11.12`.
   Output went to a scratch directory, not to `results/`.
2. Compare every value in the new `pilot.json` and `pilot.scores.npz` with
   the committed files.

### 2.2 Result

| Quantity | Committed pilot | Re-derived | Match |
| --- | --- | --- | --- |
| `ari_byte_mismatch`, `ari_code_hamming`, `ari_symbol_mismatch` (all 5 conditions, 12 prompts each) | | | bit-identical |
| `sha256_raw`, `sha256_uniform_4bit`, `sha256_uniform_8bit` | | | bit-identical |
| `max_abs_diff`, `rel_l2`, `coord_mismatch`, `cosine_distance`, `margin_delta`, `token_flip`, `topk20_change` | | | bit-identical |
| `kl_fp64`, `js_fp64` | | | max abs difference 2.2e-18 |
| `sketch_rp64` | | | max relative difference 3e-14 |
| `detection` table (all thresholds, TPR, achieved FPR, intervals) | | | identical except the `pos_mean` and `control_mean` of the three rows above, at the same magnitudes |
| `storage_bytes` | | | identical |
| `conditions.*.arrays_identical`, `changed_steps`, `status`, `role` | | | identical |
| `ari_meta.scale` | `9.414882678985599` | `9.41488265991211` | see below |

The KL, JS and sketch differences are summation-order effects of a different
BLAS build. They do not move any threshold or any alarm decision.

### 2.3 The scale

The pilot-era `_ari_scores` (commit `6ae7927`, `run_pilot.py:53`) computed
`float(np.percentile(np.abs(r), 99.0))` in float64 and passed it as
`scale_max`. On the local cache this expression returns exactly
`9.414882678985599`, the value `pilot.json` records. `Context.calibrate` on
the same array returns `9.41488265991211`, and
`float(np.float32(9.414882678985599)) == 9.41488265991211`. The SDK stores the
scale as float32 (`ari/semq_compat.py:51-53`), so both paths encoded with the
same scale. That is why every ARI array is bit-identical.

### 2.4 Consequences

- `/home/ubuntu/mbd/cache` was a copy of the local cache. The pilot's inputs
  are on this machine, though not in git.
- Every number in `pilot.json` can be re-derived. `results/pilot.json` remains
  as committed; regenerating it would change only the `ari_meta` keys and
  float64 noise below 1e-17.
- The pilot is still pilot data. Twelve prompts bound a false-alarm rate only
  below 22.1% (`PROTOCOL.md:181-182`). Nothing in this section makes the
  pilot confirmatory.
- `PROTOCOL.md:94-103` needs a factual correction. Proposed replacement for
  the paragraph beginning "**The pilot's own inputs are not in this
  repository.**":

  > **The pilot's inputs are the local decoding cache.** `pilot.json` records
  > its cache as `/home/ubuntu/mbd/cache`, a copy of
  > `experiments/decoding-reproducibility/results/cache/`, which is ignored
  > by git. The recorded scale `9.414882678985599` is the float64
  > `numpy.percentile` of that cache; its float32 value equals the SDK's
  > `calibrate` output `9.41488265991211`. Re-scoring the cache with the
  > current path reproduces every ARI, hash and distance array bit-for-bit
  > (see INVENTORY.md §2). The cache itself is still unmanifested and is
  > still pilot data.

  The correction is editorial. It changes no scoring rule and no target.
  Record it in `CHANGELOG.md` when applied (CONTRIBUTING.md, "Change a
  specification").

## 3. What the confirmatory run requires

### 3.1 Targets fixed by PROTOCOL v2

- 300 control episodes and 300 episodes per intervention
  (`PROTOCOL.md:177-178`). `collect_embeddings.py:33-42` declares one control
  and six interventions (`tf32_on`, `batch_1`, `batch_256`, `fp16`, `bf16`,
  `threads_1`). Total: 300 + 6 x 300 = **2,100 episodes**.
- One episode is one fresh process over the frozen input set under one
  configuration, with its own manifest (`PROTOCOL.md:138-141`).
  `collect_embeddings.py` runs one episode per invocation and writes
  `<out>/<condition>/epNNN.npz` and `epNNN.json` (`collect_embeddings.py:45-47`,
  `168-201`).
- Controls and interventions are interleaved by the caller
  (`collect_embeddings.py:7-9`). No caller exists in the repository.
- No detection statistic is inspected before all 2,100 episodes are collected
  (`PROTOCOL.md:298-300`).

### 3.2 Instance and model

- Model: `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, the
  default in `collect_embeddings.py:104` and the encoder of the SciFact panel
  (`experiments/regime-discrimination/run_matrix.py:57`). This is the
  embedding probe, `n_bins=2` (`PROTOCOL.md:46`).
- Instance: `g5.xlarge` (A10G, 24 GB), the `launch.sh` default
  (`infra/launch.sh:44`). The model is small; the GPU is required because
  `tf32_on`, `fp16` and `bf16` are CUDA-side interventions
  (`collect_embeddings.py:150-151`, `157-160`). On CPU the TF32 flags are
  inert and `tf32_on` would collapse into a second control.
- Software: `bootstrap.sh` installs torch and sentence-transformers
  (`infra/bootstrap.sh:34-36`). The SEMQ SDK is not needed for collection.
  It is needed for scoring (`run_pilot.py:92`).
- Region: `launch.sh` provisions in `us-east-1` (`infra/launch.sh:30`).
  `collect_embeddings.py` uploads to a bucket in `us-east-2` by default
  (`collect_embeddings.py:110-111`). The bucket is chosen by the operator; no
  default exists (`collect_embeddings.py:108-109`).
- Inputs: `--inputs` takes a JSONL file with a `text` field and reads the
  first `--n` records, default 200 (`collect_embeddings.py:105`, `116-123`).
  `data/ari-bench-v0.1.jsonl` has 1,000 such records (`data/README.md`).
  PROTOCOL v2 does not name the input file or `n`. Declare both before
  collection.

### 3.3 Collection commands

Run from a laptop with AWS credentials. Angle brackets mark operator values.

```bash
cd infra
cp operator.env.example operator.env         # fill CA_* and bucket names
MAX_MINUTES=1440 ./launch.sh                 # default 300 min is too short, see 3.5
./push_tree.sh <ip>                          # private repo: rsync the tree
ssh -i ~/.ssh/semq-ari-gpu-us-east-1.pem ubuntu@<ip> 'tail -f /var/log/semq-bootstrap.log'
```

On the instance, after "bootstrap complete":

```bash
cd ~/semq-ari-benchmark/experiments/matched-budget-detectors
RUN=mbd/$(date -u +%Y%m%dT%H%M%SZ)
for ep in $(seq 0 299); do
  for cond in control tf32_on batch_1 batch_256 fp16 bf16 threads_1; do
    ~/venv/bin/python collect_embeddings.py --condition "$cond" --episode "$ep" \
      --inputs ../../data/ari-bench-v0.1.jsonl --n 200 \
      --out ~/mbd_collect --publish-s3 "s3://<episodes-bucket>/$RUN" \
      --publish-region us-east-2 || exit 1
  done
done
```

The inner loop cycles every condition at each episode index, which is the
interleaving `collect_embeddings.py:7-9` requires. `|| exit 1` stops the run
at the first failed publication. Rerunning the same loop resumes: complete
episodes are validated and re-uploaded, not re-encoded
(`collect_embeddings.py:54-78`, `135-145`).

### 3.4 Durability after PR #17

The durability change is merge `db4b974` (PR #17,
`ilona/sem-49-episode-durability`, commits `cc8d286`, `0b86c5d`, `aab16b1`
and "fail closed when episode publication fails"). No `CHANGELOG.md` entry
exists for it on any branch; the rationale is the comment at
`collect_embeddings.py:203-207`.

- Each episode is uploaded as it is written, both `.npz` and `.json`
  (`collect_embeddings.py:208-211`).
- `publish_files` runs `aws s3 cp` per file with a 600 s timeout and raises
  `RuntimeError` if any upload fails (`collect_embeddings.py:81-97`). Tests:
  `tests/test_collect_embeddings.py:50-74`.
- The reuse path also uploads, so a retry after a failed upload still
  publishes (`collect_embeddings.py:135-145`; test at `117-138`).
- Existing output is reused only after manifest and checksum validation;
  partial or corrupt output raises (`collect_embeddings.py:54-78`; tests at
  `77-115`).
- The destination prefix is `<publish-s3>/<condition>/epNNN.*`. No such
  prefix exists in either bucket today (§6).

### 3.5 Wall-clock and cost estimate

Assumptions, each stated so it can be replaced:

1. One episode is one fresh Python process. Torch import, CUDA init and
   loading all-MiniLM-L6-v2 from the local HF cache take 8-15 s.
2. Encoding 200 texts of at most 512 characters on an A10G takes under 1 s.
3. Two `aws s3 cp` calls of about 300 KB and 1 KB take 1-3 s.
4. Per-episode total: 15-30 s. Midpoint 20 s.
5. `g5.xlarge` on-demand is about $1.01/h (`infra/README.md:19`). A 200 GB gp3
   volume is about $0.55/day (`infra/README.md:22`); the default 1,000 MB/s
   throughput and 16,000 IOPS (`infra/launch.sh:70-71`) add a few dollars per
   day (`infra/launch.sh:67-69`).

Arithmetic:

- Episodes: 2,100.
- Wall clock: 2,100 x 20 s = 42,000 s = 11.7 h. Range 2,100 x 15 s = 8.8 h
  to 2,100 x 30 s = 17.5 h. Add about 15 min for bootstrap.
- Compute: 11.7 h x $1.01/h = $11.8. Range $8.9 to $17.7.
- Volume: one day, about $1 to $4 with the default throughput settings.
- S3: 2,100 x (307,200 B + about 1 KB) = about 650 MB. Storage cost is under
  one cent per month.
- **Expected total: about $15. Range $10 to $25.**

`launch.sh` arms `shutdown -h +MAX_MINUTES` with `MAX_MINUTES` default 300
(`infra/launch.sh:48`, `infra/bootstrap.sh:14`) and
`instance-initiated-shutdown-behavior terminate` (`infra/launch.sh:241`). Five
hours is shorter than the run. Set `MAX_MINUTES=1440`. If the timer fires
early, every published episode survives and the loop resumes on a new
instance.

### 3.6 Fail-closed checklist

The SEM-49 issue body was not readable in this session. The criteria below
are SEM-81's acceptance criteria, which name SEM-49 as the related run, plus
the SEM-49 title ("run and durably publish"). Each row names the code that
satisfies it or the gap.

| Criterion | Code | Status |
| --- | --- | --- |
| Protocol frozen before collection | `PROTOCOL.md:3` | satisfied |
| Inventory and classification without relabeling | this document, §1 | satisfied |
| One fresh process per episode, own manifest | `collect_embeddings.py:168-201` | satisfied per invocation |
| Interleaved conditions | `collect_embeddings.py:7-9` | **gap**: no driver; use the loop in §3.3 |
| Effective settings read back, not assumed | `collect_embeddings.py:181-190` | satisfied |
| Inputs bound by hash | `collect_embeddings.py:131`, `193` | satisfied; **gap**: input file and `n` not declared in PROTOCOL |
| Commit recorded in manifest | `collect_embeddings.py:198-199` | **gap**: `push_tree.sh:39` excludes `.git/`, so `git rev-parse HEAD` writes an empty string on an rsynced tree. Record the commit another way, or clone. |
| Durable publication, fail closed | `collect_embeddings.py:81-97`, `208-211` | satisfied; **gap**: episodes bucket unnamed, instance role `semq-benchmarks-ec2` (`infra/launch.sh:60`) not verified for `s3:PutObject` |
| Resumable without re-encoding | `collect_embeddings.py:54-78`, `135-145` | satisfied |
| Retain all episodes durably | S3 destination per episode | satisfied once the bucket exists |
| Score episodes: three ARI quantities via the SDK | `run_pilot.py:71-125` | **gap**: reads `reference.npz` with `logits` and `lengths` (`run_pilot.py:196-199`, `227`) and hard-codes the decoding-cache condition names (`run_pilot.py:46-47`). No scorer reads `<condition>/epNNN.npz`. |
| Embedding-probe baselines: block hashes, float16 anchor, uniform quantization | `scorers.py:103-152`, `272-282` | implemented, not wired to any driver |
| Both alarm units (per input, per deployment) | `PROTOCOL.md:143-157` | **gap**: no code |
| Three disjoint roles split by input and episode | `PROTOCOL.md:128-136` | **gap**: no code |
| Achieved FPR at zero with one-sided `1 - 0.05^(1/N)` bound | `PROTOCOL.md:164-166`, `194-200` | **gap**: `run_pilot.py:169-186` uses a quantile threshold and a two-sided Wilson interval |
| Paired episode-level contrasts, Holm-Bonferroni per probe | `PROTOCOL.md:289-296` | **gap**: no code (no `Holm` in the repository) |
| Exclusions listed, analyses with and without | `PROTOCOL.md:302-304` | **gap**: no code |
| Verification on the same frozen vectors, local vs instance | `PROTOCOL.md:271-275` | **gap**: no code |
| Paper tables and claims match retained evidence | `docs/paper/latex/ari.tex` | pending; see §5 |

## 4. Readiness verdict

**Frozen and ready**

- PROTOCOL v2, with v1 retained (`PROTOCOL.md`, `PROTOCOL-v1.md`).
- Per-episode collection with durable, fail-closed publication
  (`collect_embeddings.py`), 7 tests passing without the SDK.
- Detector scores and byte budgets (`scorers.py`), 23 tests passing.
- SDK-backed ARI scoring with chunking (`run_pilot.py:71-125`), 8 test
  cases passing with the SDK.
- Provisioning with cost guards (`infra/launch.sh`, `infra/bootstrap.sh`).
- Full suite at `73139c8`: 68 passed, 10 skipped without the SDK; the study
  directory plus `tests/test_sdk_surface.py`: 42 passed with the SDK.

**Missing before collection**

1. A named episodes bucket, referred to by an `operator.env` variable, and a
   check that the instance role can write to it.
2. A declaration in PROTOCOL v2 of the input file and `n` (§3.2).
3. `MAX_MINUTES=1440` at launch (§3.5).
4. A way to record the commit on an rsynced tree (§3.6).

**Missing before analysis**

5. An episode scorer that reads `<condition>/epNNN.npz` manifests and
   applies `scorers.py` and `run_pilot._ari_scores` at `n_bins=2`.
6. The §2, §2.1, §3.1, §6 and §8 analysis: role split, both alarm units,
   one-sided bounds, paired episode resampling, Holm-Bonferroni per probe,
   exclusion accounting, cross-machine verification.

Items 5 and 6 can be written and tested on synthetic episodes without a GPU
or AWS. They should exist before the first real episode is scored, so that
no analysis choice is made after seeing data (`PROTOCOL.md:298-300`).

**Exact next command**, on a machine with GPU access and AWS credentials,
after items 1-4:

```bash
cd infra && MAX_MINUTES=1440 ./launch.sh --dry-run && MAX_MINUTES=1440 ./launch.sh
```

Then the two blocks in §3.3.

## 5. Paper paragraph if the run is not done before camera-ready

The paragraph "**Matched-budget comparison (pending).**" exists only in the
uncommitted working copy of `docs/paper/latex/ari.tex` (line 388 there). It
is not on `origin/main`. The edits below apply to that paragraph. Do not
apply them without also committing the paragraph.

1. Replace "Its pilot is degenerate: on a 12-prompt legacy cache every
   detector separates the declared interventions perfectly and none
   separates the controls" with: "Its pilot is degenerate: on a 12-prompt
   TinyLlama-1.1B CPU logit cache, every detector separates the `bf16` and
   `int8` interventions perfectly, no detector separates `threads1` or
   `batched` from the reference, and the control arrays are byte-identical to
   the reference." Reason: `pilot.json` shows TPR 0 for `threads1` and
   `batched`, whose arrays equal the reference; "every detector separates the
   declared interventions" is false for two of four.
2. Add after the sizing sentence: "A confirmatory collection on 2026-09-09
   was lost before publication; the collection code now publishes each
   episode as it is written and fails closed on upload error." Reason: SEM-49
   and `collect_embeddings.py:203-207`.
3. Add at the end: "No confirmatory episode has been collected at the time
   of writing. No comparison in this paper rests on the matched-budget
   study, and no ARI advantage at matched budget is claimed." Reason:
   PROTOCOL §7 and SEM-81's last criterion.
4. If the paper cites 0.45% and 50% false-alarm rates for SHA-256, label
   them as development observations on the cached SciFact panel, not as
   results of this study (§1.4).
5. Keep the pilot out of every table. If a table row is wanted, its
   denominator is 12 prompts and its false-alarm upper bound is 22.1%.

## 6. S3 inventory

Searched on 2026-09-16 with read-only listing. Bucket names are given by
their `infra/operator.env` variable names or as supplied by the operator; no
account identifier is recorded here.

### 6.1 Buckets and totals

| Bucket | Objects | Bytes | Top-level prefixes |
| --- | ---: | ---: | --- |
| `semq-research` | 3,959 | 98,452,622,870 | `cross-arch/`, `demos/`, `diagnostics/`, `embeddings/`, `kv-repair/`, `results/`, `run-logs/`, `scale-validation/`, `sdk-source/` |
| `BASELINES_BUCKET` | 32 | 302,213,475 | `ari-baselines/`, `arid-dry-run/`, `capture-sessions/` |

### 6.2 Prefixes probed

Each of these prefixes was listed recursively in both buckets and returned
zero objects: `matched-budget/`, `matched-budget-detectors/`, `mbd/`,
`mbd_collect/`, `episodes/`, `detectors/`, `confirmatory/`, `pilot/`,
`home/ubuntu/mbd/`, `cache/`, `experiments/matched-budget-detectors/`.

A full recursive listing of both buckets was also filtered for the terms
`matched`, `mbd`, `episode`, `detector`, `confirmator`, `pilot`, `ep[0-9]{3}`,
`tinyllama`, `control`, `tf32`, `reference.npz`, `proc.npz` and `manifest`.

### 6.3 Hits and classification

No object in either bucket is a matched-budget episode, episode manifest, or
copy of `/home/ubuntu/mbd/cache`. The filtered hits are:

| Key (bucket) | Size | Last modified | Classification |
| --- | ---: | --- | --- |
| `arid-dry-run/transcripts/salesforce_control_probes.jsonl.gz` (`BASELINES_BUCKET`) | 2,413 | 2026-09-01 | ARI-D dry run; matched `control`; unrelated |
| `cross-arch/L4_07-20260721/notary/tinyllama/*` (`semq-research`, 3 objects) | 30,977 and smaller | 2026-07-21 | SEMQ cross-architecture notary bundle; matched `tinyllama`; unrelated |
| `cross-arch/*/manifest*.json`, `results/*/MANIFEST.sha256` (`semq-research`, about 80 objects) | 473 to 637,467 | 2026-06 to 2026-08 | SEMQ research run manifests; matched `manifest`; unrelated |
| `kv-repair/20260914_6e9da625b1e6/results/replay/kivi4/cache/*` (`semq-research`) | 787,184 each | 2026-09-14 | KV-cache repair replay; matched `cache`; unrelated |
| `embeddings/all_minilm_l6_v2/beir_{fiqa,nfcorpus,scifact}/{corpus,queries}.npy` (`semq-research`, 12 objects) | 460,928 to 88,532,096 | 2026-06-12 | Single-run BEIR embeddings of the same encoder; development data; not episodes |
| `scale-validation/20260911_7b_476eb287a5a1/*` (`semq-research`) | up to 379,333,103 | 2026-09-11 | 7B scale validation with `COMPLETE.json`; dated on the freeze day; unrelated to this study |

Classification of every hit: none is pilot, none is confirmatory. The
`embeddings/all_minilm_l6_v2/` objects are development data from a different
project and do not carry the manifest fields `collect_embeddings.py` writes.

### 6.4 The durability destination

`collect_embeddings.py` writes to `<publish-s3>/<condition>/epNNN.{npz,json}`
(`collect_embeddings.py:209-211`). `--publish-s3` has no default
(`collect_embeddings.py:108-109`) and `operator.env.example` names no episodes
bucket (`infra/operator.env.example:9-10` names `BASELINES_BUCKET` and
`BATCH_RESULTS_BUCKET`). No prefix that the durability change would write to
exists in either bucket. The first confirmatory run creates it.
