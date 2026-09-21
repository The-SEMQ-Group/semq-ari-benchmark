# Paper evidence index

This index links every table and figure in `docs/paper/latex/ari.tex` to the
files that produce it. [`EVIDENCE_MANIFEST.json`](EVIDENCE_MANIFEST.json) is
the source: result files, attestation sidecars and archived raw artifacts carry
a SHA-256; specification files, generator scripts, generated tables and figures
carry the commit that last changed them on `main`. Run
`python docs/paper/check_evidence_manifest.py` from the repository root to
verify the hashes against the working tree. The table below is generated from
the manifest by `python docs/paper/check_evidence_manifest.py --render`; the
test suite fails when the committed table differs from the rendered one. Labels
follow the 2026-09-16 working draft on branch `ilona/sdk-only-probe-and-paper`.

S3 locations name buckets by the variables in `infra/operator.env`
(`RESEARCH_RESULTS_BUCKET`, `BASELINES_BUCKET`, `BATCH_RESULTS_BUCKET`). This
document names no bucket or account.

"Raw" means the vectors, packed codes, transcripts or trajectories behind the
published numbers. "Summary" means the JSON or CSV the table or figure reads.

## Tables and figures

| Label | Item | Experiment | Summary files (in repository) | Generator and command | Environment record | Raw artifacts |
| --- | --- | --- | --- | --- | --- | --- |
| `tab:panel` | Table 1, API panel | `experiments/deployed-agent-panel` | `docs/paper/evidence/embedding_panel/MANIFEST.json`, `docs/paper/evidence/embedding_panel/gemini_gemini-embedding-001.json`, `docs/paper/evidence/embedding_panel/openai_text-embedding-3-large.json`, `docs/paper/evidence/embedding_panel/mistral_mistral-embed.json`, `docs/paper/evidence/embedding_panel/voyage_voyage-4-large.json`, `docs/paper/evidence/embedding_panel/cohere_embed-v4.0.json` | `docs/paper/latex/build_tables.py` -> `docs/paper/latex/tables/panel.tex`; `python docs/paper/latex/build_tables.py  (repository root; --check compares)` | Each snapshot's 'environment' field: blas, hardware and precision read 'provider-internal'; 'measured_at' is 2026-07-06. No host record exists for a hosted API. | not retained |
| `fig:arid` | Figure 2, hosted decoding | `experiments/arid-dry-run` | `results/analysis.json`, `results/time_pair.json`, `results/time_pair_together.json`, `results/time_pair_salesforce.json`, `results/door_compare.json`, `results/manifest.json` | `docs/paper/latex/make_figure2.py` -> `docs/paper/latex/figure2.pdf`; `cd docs/paper/latex && python make_figure2.py` | manifest.json records per run: endpoint URL, model, params (temperature, top_p, max_tokens, seed), prompt_content_hash, calls, reconnects, wall_s. No client host record. | not in repository; archived in the baselines bucket, verified 2026-09-17; `s3://$BASELINES_BUCKET/arid-dry-run/transcripts/` (verified 2026-09-17: two transcripts downloaded; sha256 equal to results/manifest.json runs[5] and runs[8].) |
| `fig:overview` | Figure 1, SciFact gap and retrieval walk | `experiments/regime-discrimination`, `docs/paper/evidence/L3_09_agentic_compounding` | `results/regime_matrix.json`, `MANIFEST.sha256`, `analysis.json`, `config.yaml`, `environment.json`, `replay.md`, `report.md`, `results.jsonl` (+ 2 attestation sidecars) | `docs/paper/latex/make_figure1.py` -> `docs/paper/latex/figure1.pdf`; `cd docs/paper/latex && python make_figure1.py` | left: regime_matrix.json 'hosts': cpu 'macOS arm64: fp32 reference and CPU conditions', gpu 'g5.xlarge A10G, us-east-2: GPU conditions'. The attestation sidecar records only the signing host (macOS-26.5.1-arm64, CPython 3.11.12). No torch or SDK version is stored in regime_matrix.json; codes/MANIFEST.json records semq 1.5.1.dev32+g80e4b2c5a for the re-encoded codes.; right: docs/paper/evidence/L3_09_agentic_compounding/environment.json (container image tag 20260707-112517 and digest, c7i.2xlarge, Debian 12, CPython 3.11.15, numpy 2.4.6, torch 2.12.1+cpu, semq 1.3.0rc1, semq-research 0.1.0, PYTHONHASHSEED=42; 'git': null). | left: CPU conditions archived 2026-09-17; GPU conditions not retained; `s3://$RESEARCH_RESULTS_BUCKET/results/ari-benchmark/regime-discrimination/2026-09-17/` (uploaded 2026-09-17; one cache file and two code files downloaded, sha256 matches MANIFEST.sha256; HER recomputed from the archived int8 codes equals regime_matrix.json.); gpu_conditions: not retained in repository; location not recorded; right: aggregates archived; trajectories never written; `s3://$RESEARCH_RESULTS_BUCKET/results/L3_09_agentic_compounding/2026-07-07_113131_20260707-112517/` (2026-09-16: remote MANIFEST.sha256 downloaded and byte-identical to the in-repo copy; all six listed files verify with shasum -a 256 -c.); `s3://$RESEARCH_RESULTS_BUCKET/embeddings/bge_large_en_v15/beir_nfcorpus/` (2026-09-16: corpus.meta.json downloaded; it records shape [3633, 1024], dtype float32, created_at_utc 2026-06-11T17:06:02+00:00, encoder_image_revision 20260611-141157.) |
| `tab:decoding` | Table 2, decoding range | `experiments/decoding-reproducibility` | `results/decoding_matrix_llama31_8b_gpu.json`, `results/decoding_matrix_qwen25_7b_gpu.json`, `results/decoding_matrix_mistral7b_v03_gpu.json` (+ 6 attestation sidecars) | `docs/paper/latex/build_tables.py` -> `docs/paper/latex/tables/decoding.tex`; `python docs/paper/latex/build_tables.py  (repository root; --check compares)` | Each JSON records model, device 'cuda', n_new_tokens 48, quant_bins 8, and the 48 prompts. RESULTS-production-models.md names one L40S. No torch, CUDA, driver, or SDK version is recorded. Attestation sidecars record only the signing host (macOS-26.5.1-arm64). | not retained in repository; location not recorded; `s3://$RESEARCH_RESULTS_BUCKET/results/ari-benchmark/decoding-reproducibility/2026-09-17/` (uploaded 2026-09-17; cache/bf16.npz downloaded, sha256 matches MANIFEST.sha256.) |
| `tab:conditions` | Table 3, condition set | none | `spec/ari-canonical-v0.1.md` | Typed inline in ari.tex from the specification; no measurement. | n/a | not applicable |
| `tab:arid` | Table 4, hosted rates | `experiments/arid-dry-run` | `results/analysis.json`, `results/time_pair.json`, `results/time_pair_together.json`, `results/time_pair_salesforce.json`, `results/door_compare.json` | `docs/paper/latex/build_tables.py` -> `docs/paper/latex/tables/arid.tex`; `python docs/paper/latex/build_tables.py  (repository root; --check compares)` | Same as fig:arid. | not in repository; archived in the baselines bucket, verified 2026-09-17; `s3://$BASELINES_BUCKET/arid-dry-run/transcripts/` (verified 2026-09-17: two transcripts downloaded; sha256 equal to results/manifest.json runs[5] and runs[8].) |
| `fig:batch` | Figure 3, batch invariance | `experiments/batch-invariance` | `results/summary_with_control.json` (related: `results/summary_boundary.json`, `results/summary_all_encoders.json`, `results/summary.json`, `results/nonassociativity.json`) | `docs/paper/latex/make_figure3.py` -> `docs/paper/latex/figure3.pdf`; `cd docs/paper/latex && python make_figure3.py` | RESULTS.md prose only: 'Run 2026-09-08 on p5.48xlarge (8x H100 80GB, Hopper), torch 2.5.1+cu121, semq 1.5.0'. summary_with_control.json has no environment block (keys: label, n, reference_batch, batches, cells). | not in repository; in the batch-results bucket, listed 2026-09-17, not checksum-verified; `s3://$BATCH_RESULTS_BUCKET/batch-invariance/results/` (listed 2026-09-17; no object checksum compared yet.) |
| `tab:scifact` | Table 5, full SciFact | `experiments/regime-discrimination` | `results/regime_matrix.json` (+ 2 attestation sidecars) | `docs/paper/latex/build_tables.py` -> `docs/paper/latex/tables/scifact.tex`; `python docs/paper/latex/build_tables.py  (repository root; --check compares)` | Same as fig:overview left. | CPU conditions archived 2026-09-17; GPU conditions not retained; `s3://$RESEARCH_RESULTS_BUCKET/results/ari-benchmark/regime-discrimination/2026-09-17/` (uploaded 2026-09-17; one cache file and two code files downloaded, sha256 matches MANIFEST.sha256; HER recomputed from the archived int8 codes equals regime_matrix.json.); gpu_conditions: not retained in repository; location not recorded |
| `tab:normalized` | Table 6, normalized panel | `experiments/deployed-agent-panel`, `experiments/drift-sensitivity` | `docs/paper/evidence/embedding_panel/MANIFEST.json`, `docs/paper/evidence/embedding_panel/gemini_gemini-embedding-001.json`, `docs/paper/evidence/embedding_panel/openai_text-embedding-3-large.json`, `docs/paper/evidence/embedding_panel/mistral_mistral-embed.json`, `docs/paper/evidence/embedding_panel/voyage_voyage-4-large.json`, `docs/paper/evidence/embedding_panel/cohere_embed-v4.0.json`, `spec/fingerprints-v0.1.csv` (related: `results/encoder_fingerprints.csv`, `results/kappa_per_dataset.csv`, `results/sensitivity_curves.csv`) | `docs/paper/latex/build_tables.py` -> `docs/paper/latex/tables/normalized.tex`; `python docs/paper/latex/build_tables.py  (repository root; --check compares)` | Same as tab:panel for the snapshots. The fingerprint registry records no capture environment. | not retained |
| `tab:replication` | Table 7, pilot replication | `experiments/deployed-agent-panel` | `artifacts/openai_proc_pilot_n1000.json` | Typed inline in ari.tex; not generated by build_tables.py. | None. The artifact records n_inputs 1000, resamples 3, probe_backend semq, agent text-embedding-3-large. | not retained |
| `tab:decfull` | Table 8, per-model decoding | `experiments/decoding-reproducibility` | `results/decoding_matrix_llama31_8b_gpu.json`, `results/decoding_matrix_qwen25_7b_gpu.json`, `results/decoding_matrix_mistral7b_v03_gpu.json` (+ 6 attestation sidecars) | `docs/paper/latex/build_tables.py` -> `docs/paper/latex/tables/decfull.tex`; `python docs/paper/latex/build_tables.py  (repository root; --check compares)` | Each JSON records model, device 'cuda', n_new_tokens 48, quant_bins 8, and the 48 prompts. RESULTS-production-models.md names one L40S. No torch, CUDA, driver, or SDK version is recorded. Attestation sidecars record only the signing host (macOS-26.5.1-arm64). | not retained in repository; location not recorded; `s3://$RESEARCH_RESULTS_BUCKET/results/ari-benchmark/decoding-reproducibility/2026-09-17/` (uploaded 2026-09-17; cache/bf16.npz downloaded, sha256 matches MANIFEST.sha256.) |
| `text:gpu-hosts` | Text, GPU hosts | `experiments/gpu-determinism` | `results/a10g.csv`, `results/isolation_2x2.csv`, `results/mach_cross_gpu.csv` | Numbers quoted in prose from the CSVs and RESULTS.md. | RESULTS.md and README.md prose: g5.xlarge A10G (sm_86), g4dn.xlarge T4 (sm_75), n=512, PyTorch 2.3 default TF32 off, determinism cell uses use_deterministic_algorithms(warn_only=True) and CUBLAS_WORKSPACE_CONFIG=:4096:8. No environment JSON. | not in repository; tarred for S3 upload, key not recorded |
| `text:selfhosted-panel` | Text, self-hosted panel | `experiments/deployed-agent-panel` | `docs/paper/evidence/embedding_panel/BAAI_bge-large-en-v1.5.json`, `docs/paper/evidence/embedding_panel/BAAI_bge-m3.json`, `docs/paper/evidence/embedding_panel/intfloat_multilingual-e5-large.json`, `docs/paper/evidence/embedding_panel/sentence-transformers_all-mpnet-base-v2.json`, `docs/paper/evidence/embedding_panel/sentence-transformers_all-MiniLM-L6-v2.json`, `docs/paper/evidence/embedding_panel/nomic-ai_nomic-embed-text-v1.5.json`, `docs/paper/evidence/embedding_panel/mixedbread-ai_mxbai-embed-large-v1.json`, `docs/paper/evidence/embedding_panel/Snowflake_snowflake-arctic-embed-l.json`, `docs/paper/evidence/embedding_panel/Salesforce_SFR-Embedding-2_R.json`, `artifacts/selfhosted_bge_large_pinned.json`, `artifacts/selfhosted_bge_large_x86_pinned.json`, `artifacts/selfhosted_bge_large_x86_unpinned.json` | Quoted in prose. | Snapshot 'environment' fields (arm64 for the eight 2026-07-06 captures; x86_64+L40S(g6e.xlarge) for SFR, 2026-08-31). | not in repository; `s3://$BASELINES_BUCKET/capture-sessions/sfr-2r/` (listed 2026-09-17 (sfr-out.tar.gz, ari-bundle.tar.gz, two scripts, two wheels); not downloaded.) |

The `text:` rows cover numbers quoted in prose. Appendix `app:resid` states
its raw vectors were not retained; `app:arie` withholds its numbers.

## Raw artifacts not in the repository

Every row above except `tab:conditions` has `s3_upload_needed: true`. The
manifest's `s3_upload_list` gives the exact list and status. In short:

1. Uploaded 2026-09-17 and verified by download: SciFact CPU vectors (7 files,
   48 MB) and codes (8 files, 6.1 MB) from `experiments/regime-discrimination/results/`;
   TinyLlama pilot logits cache (6 files, 311 MB) from
   `experiments/decoding-reproducibility/results/cache/`. Prefixes:
   `$RESEARCH_RESULTS_BUCKET/results/ari-benchmark/<experiment>/2026-09-17/`
   with a `MANIFEST.sha256` beside the objects.
2. Already archived and verified by download 2026-09-17: hosted decoding
   transcripts (16 objects).
3. Already archived, listed 2026-09-17, no checksum compared: H100
   batch-invariance code matrices (553 objects); SFR raw vectors (6 objects).
4. In S3 per the experiment notes, prefix not recorded in the repository:
   A10G/T4 GPU code tarballs, SciFact GPU-row vectors, L40S decoding logits.
5. Not retained anywhere known: API panel vectors, replication pilot vectors,
   fingerprint calibration trials, retrieval-walk trajectories.

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
