# Claim-to-evidence index

Checked 2026-09-16 against the paper draft `docs/paper/latex/ari.tex` on branch
`ilona/sdk-only-probe-and-paper` and the result files committed under `experiments/`, `spec/` and
`data/` at `v0.1-preview` (commit `73139c8`). Derived values were recomputed from the JSON and CSV
fields named in each row. Paths are relative to the repository root.

Status values: `match` (the paper number equals the file value at the printed precision),
`mismatch` (it does not), `not found` (no committed result file under `experiments/`, `spec/` or
`data/` contains the value; the note names the nearest source). Rows marked "prose only" match a
Markdown results page whose underlying capture is not in the repository.

## Embedding panel (Section 3.1, Table 1, Appendices H and I)

| Claim in paper | Source file and field | Status |
| --- | --- | --- |
| ARI-HER range 0.169 to 1.000 over five APIs | No snapshot under `experiments/`; `docs/analysis/null-relative-effects.md` published ARI column (0.169333 to 1.000000); `experiments/deployed-agent-panel/RESULTS.md` leaderboard | not found (prose only; the report snapshots live in `docs/paper/evidence/embedding_panel/` on the working branch, pinned to leaderboard commit `4f49fe35`) |
| Table 1 per-condition HER (gemini 1.000 x4; openai 0.859/0.862/0.851/0.822; mistral 0.734/0.753/0.791/0.554; voyage 0.571/0.615/0.209/0.676; cohere 0.133/0.146/0.130/0.232) | same as above | not found (prose only; every value matches `null-relative-effects.md` and `RESULTS.md`) |
| `time` at ~76 h; `conc` 64-way burst, 16 for one provider | `experiments/deployed-agent-panel/RESULTS.md` caveats | not found (prose only) |
| Normalized table `d` and `kappa_byte/4` (3072, 0.637; 3072, 0.413; 1024, 0.371; 1024, 0.622; 1536, 0.664) and `b` (0.990, 0.932, 0.915, 0.979, 0.995) | `spec/fingerprints-v0.1.csv` columns `dim`, `kappa`/4, `b` | match |
| Normalized table `H-bar` (0.000, 1.363, 1.349, 9.566, 12.200) and `sigma-hat` bounds | panel snapshots (working branch only) | not found |
| 13 fitted architectures: `b` mean 0.979, sample SD 0.028; `kappa_byte` in [1.484, 3.058] | `spec/fingerprints-v0.1.csv`, rows with a `b` value (13); recomputed mean 0.979, SD 0.027597; min/max `kappa` 1.484/3.058 | match |
| Per-model `kappa` moves 0.03 to 0.35 across two additional corpora | `experiments/drift-sensitivity/results/kappa_per_dataset.csv`; ranges tabulated in `drift-sensitivity/RESULTS.md` | match |
| all-mpnet-base-v2 and all-MiniLM-L6-v2 `floor_limited`, about 1% of code bytes | `spec/fingerprints-v0.1.csv` `status`, `notes` | match |
| Replication pilot: `proc` 0.851, 0.856, 0.846 (sd 0.005); `same` 0.849, 0.837, 0.870 (sd 0.017); CI half-width 0.018 | `experiments/deployed-agent-panel/artifacts/openai_proc_pilot_n1000.json` `conditions.*.per_resample_HER`, `HER_ci95`; recomputed sd 0.0050, 0.0167; half-width 0.0177 | match |
| mistral-embed 0.753 at 19 h against 0.554 at ~76 h | `experiments/deployed-agent-panel/RESULTS.md` finding 3 | not found (prose only) |
| Cache control: 150 nonce inputs, gemini 1.000, control 0.807 | `docs/methodology.md` | not found (prose only) |
| Nine self-hosted models record HER 1.000 in `proc`, `conc`, `time` cells; those `time` cells are immediate repeats | `experiments/deployed-agent-panel/RESULTS.md` leaderboard (8 rows) and SFR addendum | not found (prose only; the audit's correction is recorded there) |
| Eight CPU precision diagnostics, 256 inputs, bf16 HER 0.000 to 0.094 | `experiments/deployed-agent-panel/RESULTS.md` prec table (n=256; 0.000 to 0.094) | not found (prose only) |
| SFR: HER 0.0, `H-bar` 24.4 bits at 4,096 dimensions | `experiments/deployed-agent-panel/RESULTS.md` SFR addendum | not found (prose only) |
| Float residuals: 3 of 30 texts differed; about 2,300 of 3,072 dimensions | `docs/methodology.md` | not found (prose only; raw vectors not retained) |

## SciFact and GPU determinism (Section 3.2, Appendix G)

| Claim in paper | Source file and field | Status |
| --- | --- | --- |
| 5,183 documents, 300 judged queries, all-MiniLM-L6-v2 | `experiments/regime-discrimination/results/regime_matrix.json` `n_docs`, `n_queries`, `encoder` | match |
| Precision and compute-mode changes alter codes of 47 to 100% of documents | `regime_matrix.json` rows `gpu_tf32_on.semq_her` 0.5267 (47.3% changed); `fp16`, `bf16`, `int8` `semq_her` 0.0; `gpu_bf16` 0.0002 | match |
| No delta Recall@10 interval excludes zero | `regime_matrix.json` `recall_delta_significant` false for every row | match |
| Fresh process, single thread, batch 8 and 128 leave every code intact (HER 1.000) | rows `proc`, `threads1`, `batch8`, `batch128` `semq_her` 1.0 | match |
| TF32 on: HER 0.5267, delta R@10 = 0; TF32 off restores 0.9992 | rows `gpu_tf32_on`, `gpu_tf32_off` `semq_her` 0.526722, 0.999228 | match |
| Table 3 delta R@10 and CIs (+0.0000; +0.0033 [0, +0.0100]; -0.0033 [-0.0100, 0]; -0.0061 [-0.0278, +0.0156]) | `recall_delta`, `recall_delta_ci95` per row | match |
| Table 3 mean dot product (1.000000, 1.000127, 1.001052, 1.001034, 0.938093) | `cosine_mean` per row (a dot product, as the paper states) | match |
| Table 3 top-10 same (100.00, 96.33, 92.67, 45.00, 43.00, 0.00%) | `top10_identical` per row | match |
| int8 changes every top-10 list; bf16 changes 55% | `int8.top10_identical` 0.0; `bf16.top10_identical` 0.45 | match |
| 2x2 over TF32 x deterministic: HER 1.000 off, 0.749 on; flag has no effect | `experiments/gpu-determinism/results/isolation_2x2.csv`; mean of `proc_tf32on_*` 0.7490, both determinism columns identical | match |
| A10G and T4 cross-GPU agreement mean 0.999 over eight encoders | `experiments/gpu-determinism/results/mach_cross_gpu.csv` `mach_a10g_vs_t4` mean 0.9990 | match |
| GPU fp32 TF32 on disagrees with CPU on 12 to 46% of inputs; seven of eight recover to about 1.000 with TF32 off | `experiments/gpu-determinism/results/a10g.csv` `mach_gpu_vs_cpu_tf32on` (1-HER: 12.5% to 45.9%); `mach_gpu_vs_cpu_det` seven values at or above 0.996 | match |
| mxbai-embed-large-v1: 1.000 across GPUs, 0.537 against CPU | `mach_cross_gpu.csv` row `mxbai-embed-large-v1` (1.0, 0.5371, 0.5371) | match |

## Batch invariance and non-associativity (Section 3.2, Appendix E)

| Claim in paper | Source file and field | Status |
| --- | --- | --- |
| Four of seven encoders disagree at batch 1 (TF32 off): three at 0.9990, mxbai 0.5560 | `experiments/batch-invariance/results/summary_with_control.json` cell A `comparisons[batch=1].HER`: e5 0.999, nomic 0.999, arctic 0.999, mxbai 0.556 | match |
| All 21 same-batch control cells read 1.0000 | `summary_with_control.json` `control_HER` for 21 cells, all 1.0 | match |
| Deterministic-algorithms cell bit-identical to cell A for all seven models | cells A and B identical in `summary_with_control.json` | match |
| all-mpnet-base-v2 under TF32 disagrees at every batch size, no ordering | cell C mpnet HER 0.856, 0.853, 0.882, 0.849 | match |
| MiniLM boundary sweep: 0.9470 at batch 1, 0.9990 at batch 2, 1.0000 from batch 4 | `experiments/batch-invariance/results/summary_boundary.json` cell C MiniLM | match |
| mxbai reads exactly 0.5560 at batch 1 in all three cells | `summary_with_control.json` mxbai cells A, B, C batch 1 | match |
| TF32 lowers agreement on two models, leaves one unchanged, raises one | cell A against cell C batch 1: MiniLM and nomic lower, mxbai equal, arctic higher | match |
| Seven encoders on p5.48xlarge, torch 2.5.1+cu121, 1,000 inputs, batch-32 reference | `summary_with_control.json` `n` 1000, `reference_batch` 32; instance and torch version in `batch-invariance/RESULTS.md` | match (`n`, reference); instance and torch prose only |
| 50,000 triples at scale 0.1; 16,015 differ in fp32 (32.0%), max 5.96e-8; 15,922 in fp16, max 4.88e-4 | `experiments/batch-invariance/results/nonassociativity.json` `dtypes.float32.pairwise`, `dtypes.float16.pairwise`, `coord_scale` | match |
| 1,024-term sum in 32 orders: 31 distinct fp32 results, spread 5.85e-7, float64 reference error 3.98e-7; width 384 gives 8 | `nonassociativity.json` `dtypes.float32.permutation` | match |

## Decoding (Section 3.3, Tables 2 and 4, Appendices K and M)

| Claim in paper | Source file and field | Status |
| --- | --- | --- |
| 48 prompts; 2,265 teacher-forced steps for Llama and Qwen, 2,269 for Mistral | `experiments/decoding-reproducibility/results/decoding_matrix_{llama31_8b,qwen25_7b,mistral7b_v03}_gpu.json` `rows[].steps`, `n_prompts` | match |
| Table 4 per-model values (token agreement, HER, `r_byte`, early warning, generations identical) | same files, `token_agreement`, `semq_her`, `semq_hbar`, `early_warning_steps`, `free_exact_count` | match (all 15 rows) |
| Table 2 ranges (TF32 99.96 to 100.00%; batched HER 0.3347 to 0.8065, early warning 19.35 to 66.53%; fp16 99.74 to 99.91%, 43 to 47 of 48; bf16 99.21 to 99.52%, 31 to 38 of 48) | same files, min and max over the three models | match |
| One teacher-forced argmax differs on Qwen under TF32 | qwen `tf32_on.token_agreement` 0.99956 = 2264/2265 | match |
| Batching moves 19 to 67% of logit codes while tokens are unchanged | `batched.early_warning_steps` 0.1935, 0.6653; `token_agreement` 1.0 | match |
| bf16: fewer than 1% of tokens change; 20.8 to 35.4% of generations differ; 26 to 45x ratio; per-token divergence 0.48 to 0.79% | `bf16.token_agreement` 0.99205, 0.99205, 0.99515; `free_exact_count` 38, 31, 38 of 48 (20.8%, 35.4%, 20.8%); ratios 26.2, 44.5, 42.9 | match |
| Legacy byte disagreement separates TF32 and bf16 by a factor of 17 on Llama | llama `bf16.semq_hbar` 0.05383 / `tf32_on.semq_hbar` 0.003074 = 17.5 | match |
| Batched rows: HER 0.33 to 0.81, `r_byte` about 1e-5 | `batched.semq_her`, `semq_hbar` (1.0e-5, 1.6e-5, 1.3e-5) | match |
| TinyLlama CPU int8: token agreement 74.21%, no generation identical; bf16 99.07%, 9 of 12 | `experiments/decoding-reproducibility/results/decoding_matrix.json` `int8`, `bf16` rows (`free_exact_match` 0.0, 0.75 of 12) | match |
| Controls: fresh process, single thread, TF32 off read HER 1.0000, 48/48 | `proc`, `threads1`, `tf32_off` rows in the three GPU files | match |

## Alternative statistics and rank profile (Appendix O)

| Claim in paper | Source file and field | Status |
| --- | --- | --- |
| At sigma 1e-4, KL reads -5.1e-10 and the probe 1.33e-4 | `experiments/decoding-reproducibility/results/baselines.json` `sensitivity.KL[0]`, `sensitivity["SEMQ Hbar"][0]` | match |
| KL moves 4.6% and JS 363% under regrouping; probe and top-2 margin unchanged | `baselines.json` `reassociation.*.max_rel_diff` (0.0458, 3.635; 0, 0) | match |
| Top-2 margin recovers 86% of the probe's signal; 8 bytes against 16,000 (about 1/2000) | `sensitivity["top-2 margin delta"][1]` 1.151e-3 / `["SEMQ Hbar"][1]` 1.331e-3 = 0.865; `storage_bytes` 8 and 16000 | match |
| Noise below rank 20: margin and token flip read exactly zero; probe unchanged at 0.129 | `baselines.json` `tail_only["20"]` (margin 0.0, token flip 0.0, SEMQ Hbar 0.1289; full 0.1287) | match |
| Live test: subtract 5.0 from 40 refusal tokens on Qwen2.5-1.5B; 100.00% tokens kept; margin change 0.0019; 22 to 29x enrichment in ranks 3 to 100 | `experiments/drift-rank-profile/results/drift_rank_profile.json` `bias_strength` -5.0, row `logit_bias_refusal_set` (`detail` 40 tokens, `token_agreement` 1.0, `top2_margin_delta` 0.00191, enrichment 28.72 and 21.86) | match |
| Frozen PQ on 5,000 SciFact abstracts: identity held in every configuration including 91% near-tie density; formula disagreement 0 to about 5% of rows | `experiments/probe-verifiability/results/near_tie_sweep.json` `n_docs` 5000; `near_tie_rate["1e-06"]` 0.9119 at 384 subspaces; `form_disagreement_rows` 0 to 0.0464 | match |
| Constructed near-duplicate codebook gave 39% disagreement | `docs/retractions.md` (constructed-codebook result; no result file) | not found (prose only; recorded as a retracted measurement) |
| Matched-budget pilot: 12 prompts bound the false-alarm rate below 22.1%; 200 controls give 1.49%; 299 give 1%; target 300 and 300 | `experiments/matched-budget-detectors/PROTOCOL.md` section 3; arithmetic 1-0.05^(1/N) recomputed (0.2209, 0.0149, 0.0100) | match (arithmetic); protocol prose |

## Hosted decoding panel (Figure 2, Appendix D)

| Claim in paper | Source file and field | Status |
| --- | --- | --- |
| 100 prompts, k=8, k=12 under burst | `experiments/arid-dry-run/results/analysis.json` `conditions[].calls` 800 and 1200, `k` | match |
| Table ARI-D cells (gemini 1.000/1.000/1.000; mistral 0.525/0.550/0.432; gpt-4o-mini 0.470/0.443/0.479/0.469 -> 0.464; Llama-3.3-70B 0.452/0.431/0.448/0.435 -> 0.438; platform 0.107/0.115/0.109) | `analysis.json` `exact_generation.mean` for full-scope transcripts; `time_pair.json`, `time_pair_together.json`, `time_pair_salesforce.json`; means recomputed 0.4636 and 0.4380 | match |
| One vendor holds 1.0000 over 2,800 completions including a 64-in-flight burst | gemini `calls` 800+800+1200; `achieved_in_flight_peak` 64 | match |
| Remaining measured rates 0.107 to 0.550 | min and max of non-gemini full-scope `exact_generation.mean` (0.1071, 0.5496) | match |
| Two endpoints lack `time`; the platform lacks `conc` | no gemini or mistral time pair; no salesforce conc transcript | match |
| Burst sweep {16, 64, 128}, k=12, peak equal to request, zero reconnects | together conc transcripts `k` 12, `achieved_in_flight_peak`, `reconnects` 0 | match |
| Length confounder: Spearman abs(rho) at most 0.18 in every bucket, inconsistent sign | `analysis.json` `openai_same` `length_confounder_rho_per_bucket` (max 0.178); `openai_proc` reaches 0.271 in instruction following | mismatch (holds for the `same` transcript only; see corrections) |
| `same` floor 0.041 (open-ended prose) to 0.759 (instruction following) | `openai_same` `same_floor_per_bucket` 0.0411, 0.7589 | match |
| Top-2 margin median 2.0 against 11.5 nats | `analysis.json` `top2_margins_same_rep0.openai` medians 2.0 and 11.5 | match |
| `topk_logprob_overlap` 0.961 | `openai_same` `topk_logprob_overlap.mean` 0.9611 | match |
| Platform resolves to gpt-4o-mini-2024-07-18; 6 of 12 fingerprints appear in the direct run's 45 | `experiments/arid-dry-run/results/door_compare.json` `fingerprint_overlap` (45, 12, 6); snapshot name in `arid-dry-run/RESULTS.md` | match (snapshot name prose only) |
| Platform completions mean 1,050 against 440 bytes | `door_compare.json` `mean_completion_bytes` 1050, 440 | match |
| Prefix comparison gap at every horizon; no-seed arm 0.441 against 0.470 | `door_compare.json` `prefix_pair_rate` and `full_pair_rate` (0.4414, 0.4696) | match |
| All 12 platform day-one fingerprints reappear on day two | `time_pair_salesforce.json` `fingerprints_shared` 12 | match |
| `maxTokens: 8` returned 250 completion tokens | `arid-dry-run/RESULTS.md` (control-probe transcript not in repository) | not found (prose only) |
| Power: 252 cells, 200 trials, 500 resamples; 0.10 drop detected 0.89 to 0.96 (bimodal) and 0.53 to 0.55 (unique, concentrated); 0.05 drop 0.04 to 0.26 | `experiments/arid-power-sim/results/power_grid.json` `design`, `cells` at `n_prompts` 100, `k` 8, floors 0.7 and 0.9 (0.885 to 0.965; 0.53, 0.55; 0.045 to 0.26) | match |
| False-positive rates at most 1% across the grid | `power_grid.json` `flag_rate` at `delta` 0: max 0.01 at 100 prompts, 0.02 in two cells at 50 prompts | mismatch (see corrections) |

## Iterative retrieval walk (Section 3.3, Appendix F)

| Claim in paper | Source file and field | Status |
| --- | --- | --- |
| 11.5% of 3,000 trajectories diverge by hop 15 at sigma 1e-3; clean top-10 overlap 0.9879 on a 200-document sample; NFCorpus 3,633 documents; zero divergence at sigma 0 | No file under `experiments/`. `docs/paper/evidence/L3_09_agentic_compounding/analysis.json` on the working branch (hash-verified per `docs/paper/REVIEW.md`) | not found (evidence bundle is not on `origin/main`) |

## Outcome layer and reproducibility statement

| Claim in paper | Source file and field | Status |
| --- | --- | --- |
| About 22% of ARI-E rows were ungraded and dropped | `experiments/harness-effect/results/extract_summary.json` `unknown_outcome` / (`kept` + `unknown_outcome`) = 3,435 / 16,000 = 21.5% | match |
| ARI-E numerical contrasts withheld | `experiments/harness-effect/results/harness_effect.json` exists with historical values; the paper reports none | match (nothing to check; page marked historical) |
| Frozen 1,000-item input set with content hash | `data/ari-bench-v0.1.jsonl`; `python -m ari.run --inputs ... --validate` prints `e9ec8b01c62635de...` | match |
| Historical release `v0.1-preview` | `git tag --points-at 73139c8` | match |
| Re-encoded CPU SciFact code matrices with hash manifest | `experiments/regime-discrimination/export_codes.py` and evidence on the working branch only | not found (not on `origin/main`) |
| Removed draft numbers: 449 and 547 bytes, 6.4 and 12.7 ms, 47% and 46% unbounded shares, 66 of 246,058 | `ari.tex` grep | match (absent from the draft, as the audit requires) |

## Corrections needed in `ari.tex`

The draft was not edited. Two statements do not match the committed files at the stated scope:

1. Appendix D, "Power": "False-positive rates are at most 1% across the grid." The grid in
   `power_grid.json` includes 50-prompt cells, two of which read 0.02. Write "at most 1% at 100
   prompts" or "at most 2% across the grid".
2. Appendix D, "Length confounder": "|rho| <= 0.18 with inconsistent sign" holds for the vendor
   `same` transcript. The vendor `proc` transcript reads 0.271 in the instruction-following bucket.
   Scope the sentence to the `same` transcript or raise the bound to 0.28.

Statements whose evidence is not on `origin/main` (embedding panel snapshots, retrieval-walk
bundle, re-encoded SciFact codes) match the working-branch files that `build_tables.py` and
`test_paper_evidence.py` check, but a reader of the tagged release cannot verify them. Merge the
evidence bundle before the next tag or say so in the reproducibility statement.
