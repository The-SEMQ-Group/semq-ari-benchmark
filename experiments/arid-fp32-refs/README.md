# ARI-D fp32 references (spec rollout item 4)

The rehosted class's yardstick: the completions the **published weights**
produce in fp32 under our control, for every prompt in the frozen set. A
rehosted provider's transcripts compare against this via `reference_match`
(exact) and `reference_first_divergence` (interval) — the quantization
question answered directly, per spec §7.

## Session plan (one GPU session per model release, artifact reused)

The node is set by the fp32 reference: fp32 is 4 bytes per parameter
resident, so Llama-3.3-70B needs ~263 GiB and Qwen2.5-72B ~271 GiB. Only the
largest GPU nodes clear that.

**This session runs in a purchased Capacity Block, because on-demand capacity
does not exist.** Every zone in us-east-1 refused every node that fits
(8xL40S, 4xL40S, 8xA10G, 8xL4) across repeated attempts on 2026-09-03 -- and
that is capacity, not quota: G/VT on-demand quota there is already 192 vCPUs.
Capacity Blocks for ML are the self-service way to reserve GPU capacity, they
need no sales conversation, and the P4d block quota was already 192.

    reservation  cr-0606d40d25b16aa35
    node         p4d.24xlarge -- 8xA100 40GB, ~317 GiB usable, 96 vCPU
    window       2026-09-05 11:30 UTC -> 2026-09-06 11:30 UTC (Sat 07:30 EDT)
    fee          $283.20 upfront, credit-covered

A100s hold the fp32 70B natively, so nothing about the measurement changes to
fit the hardware. The block is also cheaper per hour than the g6e.48xlarge it
replaces ($11.80 vs $30.13) and, being a reservation, cannot be lost to
capacity at launch time.

```bash
# from the laptop, INSIDE the window -- the block reports `scheduled` until
# it opens and launch.sh refuses rather than falling back to on-demand.
# Type and zone are read from the reservation; do not pass them.
CAPACITY_RESERVATION=cr-0606d40d25b16aa35 VOLUME_GB=500 MAX_MINUTES=1200 \
    ./infra/launch.sh

# the repo is private and the box holds no credentials, so the tree goes up
# over rsync rather than being cloned
./infra/push_tree.sh <instance-ip>

# on the box. HF_TOKEN is passed in the invocation, never in user-data:
# Llama-3.3 is gated, and user-data stays readable from the metadata service.
HF_TOKEN=hf_... SMOKE_ONLY=1 bash ~/semq-ari-benchmark/infra/run_refs.sh
HF_TOKEN=hf_... bash ~/semq-ari-benchmark/infra/run_refs.sh

# subsets, when only part of the matrix needs regenerating
HF_TOKEN=hf_... MODELS=llama33 DTYPES=fp32 bash .../run_refs.sh
```

Billing starts when the window opens whether or not anything is launched, and
the ~286 GB of checkpoint comes out of the same 24 hours -- which is why the
volume is provisioned at 1000 MB/s rather than gp3's default 125, turning a
38-minute download into about five.

`run_refs.sh` checks VRAM, disk and gated-repo access before the download,
runs a 3-prompt smoke pass before committing to the full matrix, and resumes:
rerunning it never regenerates a completed prompt. `--limit 3` smoke-tests any
single pass by hand. Model revisions are pinned in `generate_refs.py`
(resolved 2026-09-01).

```bash
# back home, the reference axis for a rehosted row:
./infra/fetch_results.sh <instance-ip>
python reference_compare.py --reference results/refs_llama33_fp32.jsonl \
    --transcript ../arid-dry-run/results/transcripts/together_same.jsonl.gz
```

Only `llama33` has dry-run transcripts to compare against
(`meta-llama/Llama-3.3-70B-Instruct-Turbo` on Together). The `qwen25`
reference is generated in the same session for the panel, where it is the
second rehosted model, and has nothing to compare against yet.

## Why the details matter

- **The template is imported from the dry-run harness** — byte-identical by
  construction. At temperature 0 a one-byte template difference produces a
  different completion from identical full-precision weights and would read
  as provider infidelity (§7).
- **Each dtype is a fresh load, never a cast**: casting a loaded model
  crushes fp32-born buffers (rotary frequencies) and poisons later passes —
  a failure mode this project has measured, not a hypothetical.
- **bf16/fp16 are exploratory**: they exist to *test* whether a faithful
  half-precision serving matches a same-precision reference more closely —
  a hypothesis no experiment currently supports (§13.4). Nothing about them
  is assumed in v0.1. The spec prices them as "near zero" marginal cost
  because the model is already loaded; the fresh-load rule below means that
  is not true here, and each is a full pass. They stay in scope because the
  hypothesis is worth two models of evidence, not because they are free.
- The honest-serving ceiling applies when reading results: faithful bf16
  serving sits well below 1.000 on the exact share; the quantization
  signature is the graded companion collapsing, not a low exact share alone.
