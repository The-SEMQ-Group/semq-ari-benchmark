# Advisor review for independent publication

Reviewed 2026-09-14 against the working-tree paper, measurement implementation,
archived experiment summaries and selected primary references. The abstract is
unchanged. Existing user edits and figures were preserved. This is a source and
evidence review, not a reproduction of GPU or paid API captures.

The strongest contribution is the measurement protocol: fixed inputs, frozen
probe calibration, explicit execution conditions and evidence-bound reporting.
The controlled examples support a gap between exact-code reproducibility and
retrieval quality. They do not establish detector superiority, universal backend
mechanisms or long-run provider rankings.

## Material corrections made

- Defined condition HER and the three-condition arithmetic mean. The score is
  not joint agreement across all conditions; `same` is a separate control.
- Corrected the SciFact similarity column to dot product. The implementation
  sums coordinate products without dividing by both vector norms. Values above
  one under bf16 are not cosine similarities.
- Distinguished nonsignificant differences from equivalence, and removed a
  universal Recall@10 resolution claim unsupported by the paired intervals.
- Removed causal conclusions from three-provider rank concordance, shared
  fingerprints, nonce/cache controls and single batch captures. Kernel selection
  is plausible but needs profiling for attribution.
- Separated teacher-forced argmax agreement from free-generation agreement;
  Qwen has one TF32 argmax disagreement despite identical free generations.
- Relabeled the decoding table's legacy changed-byte fractions. These are not
  the embedding panel's mean changed-bit count, despite historical `semq_hbar`
  JSON names. Contemporary code reports coordinate and legacy byte rates separately.
- Qualified the normalized scale's byte-to-coordinate approximation and made
  clear that its bounds are not statistical confidence intervals.
- Corrected the verifier dependency: it uses `cryptography`. Signature checking
  establishes artifact integrity and key possession, not scientific correctness.
- Narrowed claims about KL/JS and alternatives to the tested arithmetic and
  perturbations. Matched-budget superiority remains unmeasured.
- Corrected reference [6]'s authors against arXiv v2 and shortened the conclusion
  around the protocol rather than a catalogue of auxiliary experiments.

## Evidence to resolve before an archival release

1. **Iterative retrieval provenance located.** The script is
   `../semq-research/experiments/L3_09_agentic_compounding/experiment.py`.
   Its findings table matches every plotted trajectory value and gives overlap
   0.9879 at sigma 1e-3. The reviewed companion HEAD is
   `5d47230c8696bef7016f00fcdfa15602edd20648`; the historical bundle is
   `s3://semq-research/results/L3_09_agentic_compounding/2026-07-07_113131_20260707-112517/`.
   The paper now distinguishes clean top-10 overlap from relevance-based Recall@10,
   declares 3,000 seed/run comparisons and a separate 200-document single-step
   sample, and notes that cumulative divergence is monotone by construction.
   Queries use cached document embeddings, not repeated model inference. The original bundle is now retained under
   `docs/paper/evidence/L3_09_agentic_compounding/` and all six manifest hashes
   verify. Its analysis confirms divergence 0.115 and clean top-10 overlap
   0.9879166666666667. Figure 1 now loads the complete trajectory curves from
   that hash-checked analysis. Bind the historical source/configuration to the
   archival release;
   the current script defaults to 10 hops, while the reported run uses 15.
   Its `semq_hamming` is a packed-buffer disagreement fraction, not a bit count.
   The untouched abstract still calls clean-ranking overlap “Recall@10”; this
   terminology should be revisited when abstract edits are authorized.
2. **Coordinate calibration.** Historical fingerprint fits use packed-byte rates;
   dividing their prefactors by four is only approximately a coordinate-rate fit
   when changed coordinates do not collide within bytes. Refit from retained
   coordinate comparisons, propagate parameter uncertainty, and validate inversion
   within the fitted range. Until then the normalized provider ordering is
   conditional and exploratory. Zero code disagreement does not prove zero float
   perturbation: changes can remain within every quantizer region.
3. **Release pin.** Reference [10] names commit
   `bc21a20bfaaf59bdcd65accf6bb274470b1a6894`, whereas the reviewed checkout has HEAD
   `dfd1c38492896e14414af59a7b3eb0e785c442e0` plus uncommitted work. Create a final
   code/data release and pin its actual version; a moving working tree cannot
   substantiate an immutable reproducibility statement.
4. **Protocol conformance and coverage.** Original self-hosted precision rows
   appear to use 256 inputs (rates are multiples of 1/256), whereas the embedding
   API panel uses 1,000. State actual sample sizes for every experiment. Also
   distinguish missing decoding conditions from completed ARI scores. Complete
   `time` cells or retain the explicit exploratory status.
5. **Long-run interpretation.** Independently repeat condition captures across
   times and load episodes. Input bootstrap intervals describe variation across
   the frozen texts conditional on the captured environment; all-one bootstrap
   intervals are degenerate and do not certify perfect population repeatability.
6. **Independent regeneration.** Store model revisions, dependencies, numerical
   settings, input order, scales and raw vectors where feasible; identify access
   requirements for the SEMQ SDK. The mock probe has different thresholds and
   packing and cannot substantiate canonical measurements. Figure 1 now reads archived trajectory aggregates directly; its SciFact values
   remain inline and should likewise be bound to the experiment output.
7. **Scope and terminology.** The body primarily supports representation-level
   ARI-R. Hosted completion bytes and retrospective agent verdicts are distinct
   estimators, not evidence of the same mechanism. Keep those extensions clearly
   separate and use a consistent expansion of ARI across the paper and repository.
   Define “early warning” as contemporaneous code disagreement with unchanged
   argmax; no prospective prediction of failure was validated.

The independent draft retains the workshop layout and four-page body limit.
After tightening the introduction and hosted-panel discussion and moving retrieval
stress-test details to an appendix, two-pass compilation resolves `bodyend` to
page 4. It is not a replacement for the submitted workshop PDF. The local style is an adapted
2025 file, so no claim of compliance with an official 2026 style is established.

## Primary references checked

- [LLM-42, arXiv v2](https://arxiv.org/abs/2601.17768): author list corrected.
- [GPU floating-point noise structure](https://arxiv.org/abs/2511.00025): supports
  treating isotropic injected noise as a stress test rather than backend realism.
- [Open-SWE-Traces](https://arxiv.org/abs/2606.16038): bibliographic identity checked;
  the retrospective analysis still depends on the archived dataset extraction.

## Validation

The revised source builds with two `pdflatex` passes. Measurement unit tests
could not run in the existing `.venv` because `pytest` is absent; no measurement
implementation was changed. The abstract was compared byte-for-byte with the
pre-review working-tree source and is unchanged.

## Authenticated archive inspection

AWS SSO authentication enabled download and verification of the seven-file
L3_09 bundle. All six manifest entries match. The archived analysis substantiates
the reported trajectory and overlap values, but does not permit recomputing them
from individual trajectories: `results.jsonl` contains only four sigma rows.
No per-seed/per-run outcomes are retained, so uncertainty across seed documents
cannot be recovered from this bundle. The config requests a 3,000-document limit
while the analysis reports 3,633; the historical runtime behavior needs resolving
for exact replay. Replay identifies a CPU container digest and `c7i.2xlarge` host,
but records its Git SHA as unknown. These are provenance limitations, not evidence
that the saved aggregate values are wrong. A fresh rerun with retained trajectories
and source/model/input pins would provide stronger archival evidence.
