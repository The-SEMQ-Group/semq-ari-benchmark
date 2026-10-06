# Port to the public SEMQ SDK

Status: ported; results to be regenerated.
The SEMQ SDK became public on 2026-10-05 as `semq` 1.0.0 (https://github.com/The-SEMQ-Group/semq).
This document records why the harness moves to it, what the move changes, and what remains undecided.

## What changed in the SDK

The public SDK is not a later version of the private 1.2–1.5 builds; it has a new interface.

| Harness use | Private 1.5 | Public 1.0.0 |
| --- | --- | --- |
| Quantizer | `semq.Context`, `batch_encode` | `Codec.quant(dim, bins)`, `encode` |
| Range | `calibrate(ref, 0.99)` or `scale_max` | fixed `2 / sqrt(dim)` |
| Input | any finite rows | unit-norm rows |
| Comparison | `Context.compare_codes` | none; `ari.code_metrics` covers it |
| Value bounds | `Context.quant_regions` (`semq.regions`) | none |
| Notarization | `semq.notary`, `semq.Repo` | none; `Encoding.state_id` is a SHA-256 identity |
| Packed layout | `sign * bins + bin`, LSB first | unchanged |
| License | commercial | PolyForm Noncommercial 1.0.0 |

## Probe comparison

The fixed range changes the probe, so before adopting it we compared both probes against ground truth.

**Method.**
We captured raw float32 embeddings of the 1,000 frozen inputs from eight encoders.
The reference run was fp32 on CPU, with 4 threads and batch 32.
The comparison conditions were a fresh process, 1 thread, batch 8, batch 128, dynamic int8, bf16 on CPU (two encoders), and fp32 and bf16 on an Apple GPU.
Each condition ran in its own process.
A vector counts as changed when any byte differs from the reference.
The calibrated probe was a numerical emulation, because the private build is not installable.
On the committed SciFact codes it reproduced the private SDK bit for bit: 5,183 × 384 symbols, `scale_max` to the last bit.
That emulation reimplements QUANT, which this repository may not contain, so the comparison scripts stay outside it.
The fixed probe was the public 1.0.0 wheel.

**Results at `n_bins = 2`.**

| Measure | Calibrated (v0.1) | Fixed (v0.2) |
| --- | --- | --- |
| Unchanged vectors flagged as changed | 0 | 0 |
| Changed vectors detected, bf16 (10 cells) | 88.2–99.9% | 93.1–100%, at least the calibrated rate in every cell |
| Changed vectors detected, int8 | 100% | 100% |
| Changed vectors detected, batch size (14 cells, 2.6–9.2% of vectors changed) | 0% in 12 cells; 1.2% and 3.7% in 2 | 0% in all 14 |
| Spearman(true L2 drift, coordinates changed) | 0.864–0.946 | 0.875–0.947, at least the calibrated value for every encoder |
| Magnitude-symbol entropy | 0.64–0.76 bits | 0.87–0.92 bits |
| Codes changed by calibrating on FiQA rather than SciFact | 60–100% of rows | 0% by construction |

Both probes miss nearly all of the small batch-size changes, because those changes stay inside one quantization cell.
Under synthetic perturbations, the spread of flip rates across encoders is similar for the two probes.

**Reading.**
The two probes are close on detection and on tracking drift. The fixed probe is equal or slightly better in most cells, not all.
The decisive difference is calibration dependence.
A v0.1 code depends on the corpus that produced its scale, and a scale change of 0.3–3.6% rewrites most codes.
A v0.2 code depends only on the vector.

**Limits.**
The comparison used one machine (Apple M5 Max), CPU and Apple GPU conditions, and 1,000 inputs.
It did not include hosted APIs or CUDA conditions.

## Capture finding: checkpoint dtype

`mixedbread-ai/mxbai-embed-large-v1` declares `torch_dtype: float16` in its config.
transformers 5 loads a checkpoint in its declared dtype by default, so an unqualified load produces fp16 weights, even on CPU.
In fp16 its Apple GPU vs CPU HER was about 0.5. In fp32 it was 0.998–0.999, in line with the other seven encoders.
The GPU-determinism experiment reports mxbai at 0.537 against the CPU reference.
Check which transformers version and dtype that capture used before reusing the number.
Capture tools should pass the intended dtype explicitly and assert it after loading.

## Ported

- `ari/semq_compat.py`: the only SDK call site; `encode_packed`, `unpack`, `max_magnitude`.
- `ari/probe.py`: the v0.2 probe; `fixed_scale_codes` refuses a v0.1 baseline scale.
- Every capture tool through `load_probe` and `fixed_scale_codes`, unchanged.
- `experiments/regime-discrimination`: `run_matrix.py`, `export_codes.py`.
- `pyproject.toml`: `semq==1.0.0` is a dependency, checked again at import; Python 3.11 or later.
- CI: one job, no CodeArtifact role; it fails if a test skips for want of `semq`.

## Decisions

1. **Attestation: signed with the KMS key, without SEMQ.**
   `ari/attest.py` writes the `NTRY` sidecar itself. That sidecar is a PureEdDSA signature over the raw manifest digest, which `verify_report.py` already defines.
   A local key and the AWS KMS key produce the same format.
   The `semq.Repo` copy of each input is dropped, because the manifest binds every input by SHA-256.
   The 10 committed reports were signed with a demo key that is not in `spec/signers.json`.
   They are re-signed with `alias/semq-ari-attestation`, with their manifests kept byte for byte (`ari/tools/resign_attestations.py`).
2. **Change profiles and bound quality: retired.**
   They need `quant_regions`, and computing bin edges here would reimplement the operator.
   The modules and their tests are removed.
   The optional `change_profile` block stays in the report schema, marked retired, so existing reports still validate.
3. **Logit experiments: a new probe, ARI-D-Logit-v0.2** (`spec/arid-logit-probe-v0.2.md`, `ari/logit_probe.py`).
   Rows are centred over the vocabulary, split into fixed 65,536-wide chunks, and each chunk is encoded by the public SDK.
   The decoding-reproducibility, baselines, drift-rank-profile and matched-budget code uses it.
