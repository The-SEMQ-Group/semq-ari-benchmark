# ARI-Bench-v0.1 — the canonical input set

*Status: FROZEN (v0.1-preview).*

The input set `X_N` must be fixed and published so ARI reports are comparable across agents.

**ARI-Bench does not introduce new data.** ARI is a measurement of *reproducibility*, not
*capability* — the inputs are simply texts we embed and re-embed under different conditions.
They do not need to be novel; they need to be **fixed, public, and representative**.
ARI-Bench v0.1 is therefore a **fixed, versioned slice of an existing standard
embedding-benchmark corpus**, not an authored prompt set.

We invent the instrument (SEMQ) and the metric (ARI). We deliberately **reuse the
community's data.**

## Why reuse standard data (not author our own prompts)

- **Credibility / no cherry-picking.** A neutral, recognised corpus removes any "you picked
  prompts to flatter/penalise model X" objection — critical for a measurement meant to be
  cited by regulators and procurement.
- **Alignment with the leaderboards people already read.** Drawing inputs from
  **MTEB / BEIR** makes ARI the *companion axis* to the embedding leaderboard everyone
  already cites: "for each model on MTEB, here is its ARI on the same data."
- **Zero curation.** No prompt authoring, no maintenance of a bespoke set.

## v0.1 composition — FROZEN

A fixed slice of **BEIR**, domain-diverse (medical / scientific / financial). Frozen file:
[`../data/ari-bench-v0.1.jsonl`](../data/ari-bench-v0.1.jsonl).

| source | domain | items |
| --- | --- | --- |
| BEIR / NFCorpus | medical | 334 |
| BEIR / SciFact | scientific | 333 |
| BEIR / FiQA-2018 | financial | 333 |

- **Size:** **N = 1,000** (v0.1-preview). At N = 1,000 the HER standard error is ≤ 0.016;
  a later freeze can scale to N = 5,000 (SE ≤ 0.007) for tighter discrimination.
- **Content hash (the pin):** `e9ec8b01c62635dee9fbbbdf8127cde5cac094aba66368ee8de306acc3afbe1d`
  (SHA-256 over the ordered `input_id\0text\n` records). Any submitter confirms they ran the
  exact same inputs in the same order by matching this hash.
- **Seed / corpora:** `seed=20260701`, corpora `[nfcorpus, scifact, fiqa]`, produced by
  [`ari/tools/build_ari_bench.py`](../ari/tools/build_ari_bench.py).

Each item is title+body truncated to 512 characters (a typical RAG-chunk length; reproducibility is length-independent). BEIR corpora are publicly available under permissive terms; ARI-Bench v0.1 redistributes real
document text as the frozen input set (small, standard, reproducible).

## Reference model for agent-internal probes

- When the AUT **exposes embeddings natively** (e.g. an embedding API), probe those directly.
- When it **does not**, hook the probe after the final pre-pooling layer of a published
  reference model (`BAAI/bge-large-en-v1.5` for v0.1, matching ARI-Canonical).

## Optional refinement (defensive, not required for v0.1)

The inputs most likely to reveal drift are those whose embeddings sit near QBIN bin edges
(high angular concentration — the κ effect). If a sharper instrument is wanted later, we can
**select** such items *from the standard pool* — ranked by proximity to bin edges — rather
than authoring anything. This stays anchored to standard data while over-sampling the
high-signal region. It is a v0.2 refinement, not a blocker.

## How the slice is produced

Deterministically, by [`ari/tools/build_ari_bench.py`](../ari/tools/build_ari_bench.py):
for each corpus it sorts candidate ids, seeds a fixed RNG, samples without replacement, then
globally sorts by id. Same `(corpora, n, seed)` → byte-identical JSONL → identical content
hash, on any machine. Freezing v0.1 = running it once with real BEIR and recording the hash.

## Release checklist

v0.1-preview is frozen (real BEIR, N = 1,000, hash above). Remaining items for a full v1.0:

- [x] Run `build_ari_bench.py` against real BEIR (N = 1,000); content hash recorded above.
- [ ] Publish the slice (or a loader/index into the source datasets) as a Hugging Face dataset.
- [ ] Ship the reference-model pin + `s` calibration alongside (see ari-canonical-v0.1.md).
- [ ] (v0.2, optional) Scale to N = 5,000 for tighter discrimination; add the
      bin-edge-proximity selection as a high-signal variant.
