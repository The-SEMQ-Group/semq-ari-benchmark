# Paper evidence index

This index links every table and figure in `docs/paper/latex/ari.tex` to the
files that produce it. [`EVIDENCE_MANIFEST.json`](EVIDENCE_MANIFEST.json) holds
the same rows with a SHA-256 for each file. Run
`python docs/paper/check_evidence_manifest.py` from the repository root to
verify the hashes against the working tree. Labels follow the 2026-09-16
working draft on branch `ilona/sdk-only-probe-and-paper`.

S3 locations name buckets by the variables in `infra/operator.env`
(`RESEARCH_RESULTS_BUCKET`, `BASELINES_BUCKET`, `BATCH_RESULTS_BUCKET`). This
document names no bucket or account.

"Raw" means the vectors, packed codes, transcripts or trajectories behind the
published numbers. "Summary" means the JSON or CSV the table or figure reads.

## Tables and figures

| Label | Item | Experiment | Summary files (in repository) | Generator and command | Environment record | Raw artifacts |
| --- | --- | --- | --- | --- | --- | --- |
| `tab:panel` | Table 1, API panel | `experiments/deployed-agent-panel` | `docs/paper/evidence/embedding_panel/{gemini,openai,mistral,voyage,cohere}*.json`, `MANIFEST.json` (pinned to `ari-leaderboard` commit `4f49fe35`) | `build_tables.py` -> `tables/panel.tex`; `python docs/paper/latex/build_tables.py` | Snapshot `environment` fields read `provider-internal`; `measured_at` 2026-07-06 | Not retained. Attestation sidecars are in `ari-leaderboard/submissions/`. |
| `fig:arid` | Figure 2, hosted decoding | `experiments/arid-dry-run` | `results/analysis.json`, `time_pair*.json`, `door_compare.json`, `manifest.json` | `make_figure2.py`; `cd docs/paper/latex && python make_figure2.py`. Capture: `run_dry.py`, `analyze_dry.py`, `door_compare.py`, `time_pair.py` | `manifest.json`: endpoint, model, params, prompt hash per run. No host record | Transcripts absent from the repository; digests in `manifest.json`; `RESULTS.md` places them under `$BASELINES_BUCKET/arid-dry-run/transcripts/` (not verified here). |
| `fig:overview` left | Figure 1, SciFact gap | `experiments/regime-discrimination` | `results/regime_matrix.json` (+ `.attestation.json`, `.attestation.notary`) | `make_figure1.py`; `cd docs/paper/latex && python make_figure1.py`. Capture: `run_matrix.py`; codes: `export_codes.py` | `regime_matrix.json` `hosts` (macOS arm64 CPU; g5.xlarge A10G GPU). No torch or SDK version in the JSON | CPU vectors `results/cache/*.npz` (gitignored) and codes `results/codes/` (untracked) exist in the operator's working tree; their hashes match `codes/MANIFEST.json`. GPU-row vectors and codes: not retained, no location recorded. |
| `fig:overview` right | Figure 1, retrieval walk | `docs/paper/evidence/L3_09_agentic_compounding` | `analysis.json` after `MANIFEST.sha256` check (seven files) | same `make_figure1.py`. Capture: `semq-research` `L3_09_agentic_compounding/experiment.py` in the container named in `replay.md` | `environment.json` (container tag `20260707-112517`, c7i.2xlarge, CPython 3.11.15, semq 1.3.0rc1, semq-research 0.1.0; `"git": null`) | Bundle: `$RESEARCH_RESULTS_BUCKET/results/L3_09_agentic_compounding/2026-07-07_113131_20260707-112517/` (7 objects; remote manifest identical to the in-repo copy, verified 2026-09-16). Embedding cache: `$RESEARCH_RESULTS_BUCKET/embeddings/bge_large_en_v15/beir_nfcorpus/` (4 objects, `corpus.meta.json` shape `[3633, 1024]`). Trajectories: never written. |
| `tab:decoding` | Table 2, decoding range | `experiments/decoding-reproducibility` | `results/decoding_matrix_{llama31_8b,qwen25_7b,mistral7b_v03}_gpu.json` (+ `.attestation.json`, `.attestation.notary` each) | `build_tables.py` -> `tables/decoding.tex`. Capture: `run_matrix.py` (`ARI_D_DEVICE`) | JSON: model, `device: cuda`, 48 tokens, 8 bins. `RESULTS-production-models.md`: one L40S. No torch, CUDA or SDK version | Logits and codes not retained; no location recorded. |
| `tab:conditions` | Table 3, condition set | none | `spec/ari-canonical-v0.1.md` | Typed inline in `ari.tex` | n/a | n/a |
| `tab:arid` | Table 4, hosted rates | `experiments/arid-dry-run` | as `fig:arid` | `build_tables.py` -> `tables/arid.tex` | as `fig:arid` | as `fig:arid` |
| `fig:batch` | Figure 3, batch invariance | `experiments/batch-invariance` | `results/summary_with_control.json` (related: `summary_boundary.json`, `summary_all_encoders.json`, `summary.json`, `nonassociativity.json`) | `make_figure3.py`; `cd docs/paper/latex && python make_figure3.py`. Capture: `run_batch_sweep.py --label h100 --n 1000` | `RESULTS.md` prose only: p5.48xlarge, torch 2.5.1+cu121, semq 1.5.0. No environment block in the JSON | Code matrices "retained in S3" per `RESULTS.md`; prefix and count not recorded (`$BATCH_RESULTS_BUCKET`, 211 objects reported by the operator, not verified here). |
| `tab:scifact` | Table 5, full SciFact | `experiments/regime-discrimination` | `results/regime_matrix.json` (+ sidecars) | `build_tables.py` -> `tables/scifact.tex` | as `fig:overview` left | as `fig:overview` left; the `fp16`, `gpu_tf32_*`, `gpu_bf16` rows have no raw artifacts anywhere recorded. |
| `tab:normalized` | Table 6, normalized panel | `experiments/deployed-agent-panel`, `experiments/drift-sensitivity` | five API snapshots, `MANIFEST.json`, `spec/fingerprints-v0.1.csv` (related: `experiments/drift-sensitivity/results/*.csv`) | `build_tables.py` -> `tables/normalized.tex` | as `tab:panel`; registry has none | Not retained: API vectors and coordinate-level calibration trials. |
| `tab:replication` | Table 7, pilot replication | `experiments/deployed-agent-panel` | `artifacts/openai_proc_pilot_n1000.json` | Typed inline in `ari.tex` | none in the artifact | Not retained. |
| `tab:decfull` | Table 8, per-model decoding | `experiments/decoding-reproducibility` | as `tab:decoding` | `build_tables.py` -> `tables/decfull.tex` | as `tab:decoding` | as `tab:decoding`. The TinyLlama CPU sentence (74.21%, 9 of 12) reads `results/decoding_matrix.json`; its logits cache `results/cache/*.npz` (311 MB) is gitignored and exists in the operator's working tree. |

Text-only rows in the manifest: `text:gpu-hosts` (`experiments/gpu-determinism/results/*.csv`;
GPU code tarballs not in the repository, prefix unrecorded) and `text:selfhosted-panel`
(nine self-hosted snapshots; SFR raw vectors under `$BASELINES_BUCKET/capture-sessions/sfr-2r/`,
not verified here). Appendix `app:resid` states its raw vectors were not retained; `app:arie`
withholds its numbers.

## Raw artifacts not in the repository

Every row above except `tab:conditions` has `s3_upload_needed: true`. The
manifest's `s3_upload_list` gives the exact list. In short:

1. Local, not uploaded: SciFact CPU vectors (7 files, 48 MB) and codes (8 files,
   6.1 MB) under `experiments/regime-discrimination/results/`; TinyLlama pilot
   logits cache (6 files, 311 MB) under `experiments/decoding-reproducibility/results/cache/`.
   Planned prefixes: `$RESEARCH_RESULTS_BUCKET/results/ari-benchmark/<experiment>/2026-09-17/`
   with a `MANIFEST.sha256` beside the objects. `results/ari-benchmark/` was
   empty on 2026-09-16.
2. In S3 per the experiment notes, prefix not recorded in the repository:
   batch-invariance code matrices; A10G/T4 GPU code tarballs.
3. In the baselines bucket per the experiment notes, not verified here: hosted
   decoding transcripts; SFR raw vectors.
4. Not retained anywhere known: API panel vectors, replication pilot vectors,
   fingerprint calibration trials, SciFact GPU-row vectors, L40S decoding
   logits, retrieval-walk trajectories.

## Retrieval-walk provenance

See [Corpus size](evidence/README.md#corpus-size) and
[Source revision](evidence/README.md#source-revision) in the evidence README.

## S3 archive status, 2026-09-17

Bucket names are operator configuration (`infra/operator.env`); the prefixes
below are relative to the variables named.

| Artifact set | Location | Objects | Verified |
| --- | --- | ---: | --- |
| SciFact CPU embedding cache | `$RESEARCH_RESULTS_BUCKET/results/ari-benchmark/regime-discrimination/2026-09-17/cache/` | 7 | one file downloaded, sha256 matches the archived MANIFEST.sha256 |
| SciFact packed codes and manifest | `$RESEARCH_RESULTS_BUCKET/results/ari-benchmark/regime-discrimination/2026-09-17/codes/` | 8 | two files downloaded; HER recomputed from archived codes alone equals the table |
| TinyLlama decoding pilot cache | `$RESEARCH_RESULTS_BUCKET/results/ari-benchmark/decoding-reproducibility/2026-09-17/cache/` | 6 | one file downloaded, sha256 matches |
| Hosted decoding transcripts | `$BASELINES_BUCKET/arid-dry-run/transcripts/` | 16 | two downloaded, sha256 equal to `results/manifest.json` |
| H100 batch-invariance code matrices | `$BATCH_RESULTS_BUCKET/batch-invariance/results/` | 553 | listed; `all-encoders/` holds the 211 objects behind the figure; not yet checksum-verified |
| SFR-Embedding-2_R capture session | `$BASELINES_BUCKET/capture-sessions/sfr-2r/` | 6 | listed; not yet downloaded |
| L3_09 retrieval-walk bundle | `$RESEARCH_RESULTS_BUCKET/results/L3_09_agentic_compounding/2026-07-07_113131_20260707-112517/` | 7 | MANIFEST.sha256 byte-identical to the local copy; all six files verify |

Still in no archive: API panel raw vectors, replication pilot vectors,
fingerprint calibration trials, SciFact GPU-row vectors and codes, L40S
decoding logits and codes, and the retrieval-walk trajectories, which were
never written.
