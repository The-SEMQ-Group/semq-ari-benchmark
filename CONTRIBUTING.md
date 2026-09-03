# Contributing to ARI

The most valuable contribution is **submitting an ARI report** for an agent so it appears on
the leaderboard. Reports are third-party-verifiable, so anyone can submit — the CI validates
every submission against the spec.

The board, the scorer, and the submission flow live in the
[**ari-leaderboard**](https://github.com/The-SEMQ-Group/ari-leaderboard) repo, which pins this
one as a submodule so a report still binds this repo's frozen input set by digest.

## Submit an ARI report (leaderboard entry)

1. **Run the protocol** on your agent with the canonical probe and the frozen input set:
   - Probe: SEMQ QBIN n=2, 99th-pct calibration — [`spec/ari-canonical-v0.1.md`](spec/ari-canonical-v0.1.md).
   - Inputs: [`data/ari-bench-v0.1.jsonl`](data/ari-bench-v0.1.jsonl) (verify the content hash in
     [`spec/ari-bench-v0.1.md`](spec/ari-bench-v0.1.md)).
   - Conditions: [`spec/condition-set.md`](spec/condition-set.md).
   - The reference harness in [`ari/`](ari/) can produce a report for you
     (`ari/tools/run_report.py`).
2. **Produce a report** conforming to [`spec/report-schema.json`](spec/report-schema.json), with
   an `environment` manifest and per-condition audit digests.
3. **Validate and submit** following
   [`ari-leaderboard`'s `SUBMISSION_FORMAT.md`](https://github.com/The-SEMQ-Group/ari-leaderboard/blob/main/submissions/SUBMISSION_FORMAT.md):
   run `scoring/score.py` there and open a PR adding your report under its `submissions/`. CI
   re-runs the scorer on every submission; a green check means it is schema-valid and
   self-consistent.

## Other contributions

- **Spec / methodology** — issues and PRs against `spec/` and `docs/`. The spec is versioned;
  breaking changes bump the version.
- **Reference implementation** — `ari/` (the harness). The scorer lives in `ari-leaderboard`.

## Ground rules

- Reproducibility over convenience: pin the environment, publish audit digests, report CIs.
- Precision over hype: claims must be exactly what the data support (see the honest caveats in
  the results docs). A number a reviewer can refute weakens the standard.
- Be fair to named providers: use representation-level language; the drift ARI measures is not
  the same as "the model's answers change."

By contributing you agree your contribution is licensed under this repository's terms
(Apache-2.0 for code; CC-BY-4.0 for the spec and data — see the README Licensing section).
