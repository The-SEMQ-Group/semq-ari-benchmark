# Archived paper evidence

`L3_09_agentic_compounding/` is an unchanged copy of the seven-file S3 bundle:

    s3://semq-research/results/L3_09_agentic_compounding/2026-07-07_113131_20260707-112517/

Downloaded with authenticated read access on 2026-09-14. All six entries in
`MANIFEST.sha256` were verified. Figure 1 verifies these hashes before loading
`analysis.json`. The hashes establish consistency with the downloaded manifest;
the manifest is not itself a signed scientific attestation.

The aggregate curves substantiate the plotted values. Individual trajectories
are absent; `results.jsonl` stores only the sigma grid. The config/result
corpus-size discrepancy and missing historical Git SHA remain unresolved. The companion experiment source is
`semq-research/experiments/L3_09_agentic_compounding/experiment.py`.


`embedding_panel/` contains fourteen report snapshots from `ari-leaderboard`,
pinned to the commit and SHA-256 values in `MANIFEST.json`. The paper's embedding
and normalized tables read these snapshots. They preserve historical report
bytes, including nonconformant self-hosted `time` labels; they are not new
captures or a validation of those labels. Raw vectors and signature sidecars are
not copied here. The source repository retains the original attestations.
