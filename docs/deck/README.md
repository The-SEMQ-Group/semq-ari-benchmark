# Presentation decks

`build_deck.py` generates `ari-overview.pptx` and `ari-technical.pptx` from the experiment JSON
and the figures in `docs/figures/`. The decks are historical presentation artifacts. They are not
maintained scientific evidence. Use the [paper](../paper/latex/README.md) and the
[repository audit](../REPOSITORY_AUDIT.md) for current definitions and supported measurements.

## Regenerate

From the repository root, with `python-pptx` installed:

```bash
python docs/deck/build_deck.py
```

The build reads only local files. It writes both `.pptx` files in this directory and prints the slide
count of each. The files are not committed (`.gitignore`); step 8 of [RELEASE.md](../RELEASE.md)
builds them and attaches them to the GitHub release. Regeneration on 2026-09-16 produced 10 and 13 slides.

## Superseded statements

The 2026-09-16 repository audit superseded statements that the decks made. The build now opens each
deck with a slide that lists them, labels the affected slides, and corrects wording where the correct
statement is unambiguous. The [retired-claims index](../RETIRED_CLAIMS.md) records each change.

| Deck, slide | Statement in the pre-audit export | Status |
| --- | --- | --- |
| Both, title | "Every number in this deck is read from a signed measurement." | Corrected. Numbers are read from experiment JSON; not every source is signed. |
| Overview 4 (now 5) | "Retrieval quality moves by exactly zero ... attributes the effect to that one switch." | Corrected. No Recall@10 change was detected; a nonsignificant difference is not equivalence. The control associates the effect with TF32. |
| Overview 5 (now 6) | "The instrument reads the change at every step. That is the difference between finding out now and finding out from a customer." | Corrected. Early warning is contemporaneous code disagreement, not a validated predictor. |
| Overview 6, 7, 9 (now 7, 8, 10) | ARI-E effects +0.069 and +0.050, naive 21.5% against 8.9%, "Strong. Largest effect of the three". | Marked historical and withheld. Earlier estimator; partial trajectory export; paper withholds the numbers pending a rerun. |
| Overview 8 (now 9) | "The verifier is 175 lines and does not use our software." | Corrected. The verifier uses the standard library and `cryptography`; it does not import the SDK or harness. |
| Overview 9 (now 10) | "Strong. The TF32 result is clean." | Corrected. The TF32 association replicated; kernel attribution needs profiling. |
| Technical 2 (now 3) | "Invariance follows from the operator's form ... CI asserts it on three architectures per commit." | Corrected. A fixed scale is not a proof of cross-platform invariance; each platform needs verification. |
| Technical 3, 4 (now 4, 5) | "Retrieval quality is blind"; "The CI ... is exactly [0.0000, 0.0000]. Not small. Zero." | Corrected. Recall@10 did not detect the change at 300 queries; ranked lists did change. |
| Technical 5 (now 6) | "The logits move before the tokens." | Corrected to "logit codes change while tokens do not". |
| Technical 9, 10 (now 10, 11) | ARI-E contrasts table and power table. | Marked historical and withheld. |
| Technical 12 (now 13) | Retraction table. | Extended with the self-hosted `time` cells, the two-encoder batch-invariance report, the byte-rate units, and the withheld ARI-E results. |

Statements the decks did not make, but that a reader may look for, are also superseded: the
self-hosted `time` cells and the decoding `H-bar` units. The first slide of each deck lists them.
