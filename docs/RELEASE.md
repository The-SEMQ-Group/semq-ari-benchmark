# Release procedure

This procedure cuts an immutable, audited release of the benchmark repository. It takes one
engineer about two hours plus the LaTeX build. Run every command from the repository root in a clean
checkout of `main`.

## Tag name

Tags follow `v<spec-major>.<spec-minor>[-<label>]`, for example `v0.1-preview`. The spec version is
the `ari_version` in `spec/report-schema.json`. Use a label (`-preview`, `-rc1`) for any release whose
evidence bundle is incomplete. Never move or delete a published tag. A corrected release gets a new
tag and a `CHANGELOG.md` entry that names the tag it supersedes.

## Checklist

1. Confirm the branch is merged and CI is green. Run `git status --porcelain` and confirm empty output.
2. Confirm the paper cites the tag you are about to create in reference [10] of
   `docs/paper/latex/ari.tex`, or cites the previous tag with a note that the release is pending.
3. When `docs/paper/latex/build_tables.py` is present on the paper branch, regenerate the paper
   tables: `python docs/paper/latex/build_tables.py`, then `python docs/paper/latex/build_tables.py --check`.
   Expected output: `6 paper tables checked`. The script is not on `main`; if it is absent, record
   that the tables were not regenerated.
4. Regenerate the figures from `docs/paper/latex/`: `python make_figure1.py`, `make_figure2.py`,
   `make_figure3.py`. Each script reads saved JSON only.
5. Build the PDF twice from `docs/paper/latex/`: `pdflatex -interaction=nonstopmode -halt-on-error ari.tex`,
   twice. Check `grep -o "newlabel{bodyend}{{[0-9]*}" ari.aux` reads 4 or less.
6. Run the number check by hand: for every row in [CLAIM_TO_EVIDENCE.md](paper/CLAIM_TO_EVIDENCE.md),
   open the named result file and compare the field with the paper. Every numeric claim in the paper
   needs a row with status `match`. Resolve every `mismatch` row by editing the paper or the
   evidence, never by editing the index alone. When `tests/test_paper_evidence.py` is present on
   the paper branch, run it as well.
7. Run the retired-claims search described in [RETIRED_CLAIMS.md](RETIRED_CLAIMS.md) over
   `docs/`, `experiments/*/README.md`, `experiments/*/RESULTS.md` and `README.md`. Add or mark any new hit.
8. Build the decks: `python docs/deck/build_deck.py` (needs `python-pptx`). It writes
   `docs/deck/ari-overview.pptx` and `docs/deck/ari-technical.pptx`; both are ignored by git.
   Confirm slide 2 of each deck is the superseded-statements slide. Attach both files to the
   GitHub release in step 15.
9. Run the test suite: `python -m pytest -q -rs`. Expected: no failures. Record the skipped tests and
   their reasons in the release notes.
10. Verify the signed reports with the standalone verifier:
    `python ari/verify_report.py experiments/harness-effect/results/harness_effect.json`, and the
    same for every report under `experiments/*/results/` that has an `.attestation.json` sidecar.
    Expected: `ALL CHECKS PASSED`.
11. Verify the frozen inputs: `python -m ari.run --inputs data/ari-bench-v0.1.jsonl --out /tmp/ari-report.json --validate`.
    The command fails if the content hash differs from the specification.
12. Confirm `docs/REPOSITORY_AUDIT.md` is current. If any high-priority finding changed status, add a
    dated line to it before release.
13. Commit everything from steps 3 to 12 to `main`. This is the release commit. Run
    `git status --porcelain` again and confirm empty output.
14. Tag the release commit: `git tag -a <tag> -m "<tag>: <one-line summary>"`. Sign the tag with the
    release key (`git tag -s`) when the signer is available. Before tagging, write and check the
    manifest:
    1. `python docs/release_manifest.py` writes `docs/release_manifest.json`. The file is not
       committed; `.gitignore` lists it. It records the release commit, `branch`, `dirty`, the Python
       version, platform, interpreter basename and installed packages, and the SHA-256 of every file
       under `spec/`, `data/`, `experiments/*/results/` and `docs/paper/latex/*.pdf`.
    2. Confirm `git.branch` reads `main` and `git.dirty` reads `false`. If not, stop; the checkout is
       not the release commit.
    3. `python docs/release_manifest.py --check`. Expected: `OK: 0 difference(s)`.
15. Push `main` and the tag. Create the GitHub release from the tag. Attach `ari.pdf`,
    `docs/release_manifest.json`, the two decks from step 8, and the wheel from `python -m build`.
16. Archive the release: upload the tag archive (`git archive <tag>`) and the attached files to the
    operator's archive bucket. When the paper branch carries an evidence bundle under
    `docs/paper/evidence/`, upload it as well. Record the bucket path and the archive's SHA-256 in
    the release notes.
17. Update `CITATION.cff` (`version`, `date-released`) and `CHANGELOG.md` in a follow-up commit if
    they were not already updated for this tag.

## Who signs

| Role | Responsibility |
| --- | --- |
| Release engineer | Runs steps 1 to 13 and records the outputs. |
| Measurement owner | Confirms step 6 (numbers) and step 12 (audit status). |
| Maintainer with the release key | Writes the manifest, signs and pushes the tag (steps 14 and 15). |

A release is not complete until all three have recorded their approval in the release pull request.

## What the manifest does and does not establish

`docs/release_manifest.json` binds a tag to exact file bytes and to the Python environment used for
the tests. It does not establish that any capture is correct, that signatures were made by a trusted
party, or that historical result files satisfy the current specification. Those limits are recorded
in the [repository audit](REPOSITORY_AUDIT.md).

## Venue requirements

See [Venue requirements](paper/latex/README.md#venue-requirements) in the submission build notes.

## Open items at this release

1. `docs/paper/latex/neurips_2026.sty` is the official NeurIPS 2025 style with three strings changed
   locally (package name, ordinal, year). NeurIPS had not published a 2026 style file when this was
   checked (`media.neurips.cc/Conferences/NeurIPS2026/Styles.zip` returned 404). Swap in the official
   file when it appears and rebuild; the `bodyend` check in step 5 must still read 4 or less.
   Compliance with the official 2026 format is not established until then.
2. The CFP deadline had passed on the check date (2026-09-16). Confirm the submission status with the
   workshop organizers before citing the venue.
3. Self-hosted `time` captures after a gap greater than 24 hours, the ARI-E rerun, and the
   coordinate-rate calibration refit remain open. See "Work required before stronger claims" in the
   [repository audit](REPOSITORY_AUDIT.md).
4. The `v0.1-preview` tag was created before this procedure existed and has no manifest. The first
   manifest will be written at the next tag. It will record that tag's commit on `main` with
   `dirty` `false`, the Python environment of the release machine, and the SHA-256 of the files
   listed in step 14 as they stand at that commit. It will not describe the `v0.1-preview` tree.
