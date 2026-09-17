# Archived paper evidence

`L3_09_agentic_compounding/` is an unchanged copy of the seven-file S3 bundle:

    s3://$RESEARCH_RESULTS_BUCKET/results/L3_09_agentic_compounding/2026-07-07_113131_20260707-112517/

`RESEARCH_RESULTS_BUCKET` is set in `infra/operator.env` (see
`infra/operator.env.example`). Downloaded with authenticated read access on
2026-09-14. All six entries in `MANIFEST.sha256` were verified. On 2026-09-16 the
remote `MANIFEST.sha256` was downloaded again and is byte-identical to this copy;
`shasum -a 256 -c` passes for all six files. Figure 1 verifies these hashes
before loading `analysis.json`. The hashes establish consistency with the
downloaded manifest; the manifest is not itself a signed scientific attestation.

The aggregate curves substantiate the plotted values. Individual trajectories
are absent; `results.jsonl` stores only the sigma grid. See
[review limitations](../REVIEW.md) for the review-time statement of the
corpus-size and Git SHA gaps, resolved below as far as the evidence allows.
The companion experiment source is
`semq-research/experiments/L3_09_agentic_compounding/experiment.py`.
[`../EVIDENCE.md`](../EVIDENCE.md) indexes every paper table and figure.

The bundle's `environment.json` and `replay.md` name the container registry
URI as captured. They are kept byte-for-byte because `MANIFEST.sha256` pins
them. Quote the image tag and digest from them, not the registry host.

## Corpus size

`config.yaml` carries two corpus numbers. Under `datasets`, the dataset
metadata reads `n_corpus: 3633`. Under `runtime`, the run setting reads
`corpus_size_limit: 3000`. `analysis.json` reports
`"summary": {"n_corpus": 3633, "n_hops": 15, ...}` and `report.md` reads
`**n_corpus**: 3633`. The full NFCorpus corpus is 3,633 documents.

The walk ran over all 3,633 documents. The limit had no effect. Evidence:

1. In `experiment.py` at commit `1753810bfd47570e3c72e3e28140f16187f49fb2`
   (2026-07-07 11:21:59 UTC, the last change to the file before the run and
   the only version of it in July), `setup()` applies
   `d.truncate(self.corpus_size_limit)` to the dataset. `run()` then obtains
   the matrix with `encode_or_load(embedder, dataset.corpus_texts, embedder.id,
   dataset.id, "corpus", ...)` and reports `n_corpus` as `int(n)` from
   `n, dim = E.shape`. `n_corpus` is therefore the row count of the matrix the
   walk used, not the configured limit.
2. `encode_or_load` (`src/semq_research/core/embeddings_cache.py`, unchanged
   between that commit and the current head) first calls
   `download_embeddings(model_id, dataset_id, kind)` and returns the cached
   array on a hit. The cache key is
   `embeddings/{model_id}/{dataset_id}/{kind}.npy`. It does not include the
   limit, and the row count is not compared with `len(texts)`. The truncated
   text list is used only on a cache miss.
3. The cache object exists and predates the run.
   `s3://$RESEARCH_RESULTS_BUCKET/embeddings/bge_large_en_v15/beir_nfcorpus/corpus.meta.json`
   (read 2026-09-16) records `"shape": [3633, 1024]`, `"dtype": "float32"`,
   `"created_at_utc": "2026-06-11T17:06:02+00:00"`,
   `"encoder_image_revision": "20260611-141157"`. `corpus.npy` is 14,880,896
   bytes, which equals 3633 x 1024 x 4 plus the NumPy header. A cache miss
   would have encoded 3,000 texts and reported `n_corpus: 3000`; the reported
   3,633 can only come from the cache hit.

The run log line `[cache] HIT ... shape=(3633, 1024)` is not in the bundle,
so item 3 is the direct evidence for the hit.

The limit did not apply to another stage. The two other sample sizes in the
paper are seed and run counts: divergence is a fraction of
`n_seeds x n_runs = 100 x 30 = 3000` trajectories (every value in
`analysis.json["divergence"]` is a multiple of 1/3000; for example
`0.0023333333333333335` is 7/3000), and the single-step statistics use
`2 x n_seeds = 200` probe documents times 30 draws (every value in
`analysis.json["step"]` is a multiple of 1/6000, for example `0.0435` is
261/6000; `recall_at_k` values are multiples of 1/60000). The "3,000
trajectories" and "200-document sample" in `ari.tex` (`app:walk`) are these
counts and match the code (`rng.choice(n, min(2 * self.n_seeds, n))`).

The plotted values do not depend on the limit through their denominators.
They depend on the corpus through the nearest-neighbor walk itself, which ran
on the 3,633-row matrix. The paper's "NFCorpus (3,633 documents)" is
consistent with the archived run. An exact replay must keep the cache hit or
set `corpus_size_limit` to 3633 or null; a replay that misses the cache with
`corpus_size_limit: 3000` would walk a different corpus.

## Source revision

What the bundle records:

- `environment.json`: `"git": null`; `"packages"` includes
  `"semq": "1.3.0rc1"` and `"semq-research": "0.1.0"`; `"container"` gives
  `"image_revision": "20260707-112517"` and
  `"image_digest": "sha256:f67a3d281a01232db2a30aaa3d4353d17f4defe0fca0a3f6c08ebd357a312930"`;
  `"captured_at_utc": "2026-07-07T11:32:01+00:00"`.
- `replay.md`: `Git SHA: <unknown>`.
- `report.md`: no revision information.

What is absent: a Git SHA for `semq-research`. `semq-research` version `0.1.0`
is the static value in its `pyproject.toml` and does not identify a commit.
The SHA is absent because the image build excludes `.git/` (`.dockerignore`),
so `git rev-parse HEAD` fails inside the container and `_capture_git()` in
`src/semq_research/core/provenance.py` returns `None`; the ECR tag used was a
timestamp rather than the Makefile's default short-SHA tag.

Reconstruction from the `semq-research` history (not from the bundle): commit
`1753810bfd47570e3c72e3e28140f16187f49fb2` ("experiment(L3_09): agentic
compounding of representation drift") is dated 2026-07-07 11:21:59 UTC. The
image tag `20260707-112517` is a UTC build timestamp 3 minutes and 18 seconds
later. The next commit, `4e65c99737ccbf66d80d26874b0d31feafddbe9c` at 11:41:09
UTC, changed only `docs/findings.md` and `experiments/BUNDLES.md`. The source
in the image is therefore the working tree at or shortly after `1753810`.
Uncommitted edits cannot be excluded because the Dockerfile copies the working
tree. The Makefile writes the build's short SHA into the image label
`org.opencontainers.image.revision`; inspecting the image config at the
recorded digest would confirm the commit if that build path was used.


`embedding_panel/` contains fourteen report snapshots from `ari-leaderboard`,
pinned to the commit and SHA-256 values in `MANIFEST.json`. The paper's embedding
and normalized tables read these snapshots. They preserve historical report
bytes, including nonconformant self-hosted `time` labels; they are not new
captures or a validation of those labels. Raw vectors and signature sidecars are
not copied here. The source repository retains the original attestations.
