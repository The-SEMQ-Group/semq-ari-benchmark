# Float-detector gate check: are the raw float embeddings retained anywhere?

**Result: no. The float detectors are not implemented, per this gate's stop condition.** No
`float_exact` / `float_cosine` / `topk_overlap` detector code was written. No synthetic data
was generated. Nothing was left "ready for when there's data" — see [Why not stub the
detectors](#why-not-stub-the-detectors-anyway) below for why that's a deliberate choice, not
an oversight.

## What was checked

Every place in this repo (and its referenced external artifacts) that could plausibly hold a
raw embedding vector from one of the 13 panel runs:

1. **`experiments/*/artifacts/` and `experiments/*/results/`** — inspected every file.
   `experiments/deployed-agent-panel/artifacts/*.json` hold only aggregate statistics
   (`HER`, `Hbar`, `HER_ci95`, `per_resample_HER`, boolean findings) — no vectors.
   `experiments/drift-sensitivity/results/*.csv` hold fitted scalars (`s`, `b`, `kappa`) and
   sensitivity curves, not raw embeddings. `experiments/gpu-determinism/results/*.csv` hold
   Hamming/HER summaries per config cell.
2. **`data/`** — `data/ari-bench-v0.1.jsonl` is the frozen *input* text set (BEIR-derived
   queries), not embeddings. No other file in `data/`.
3. **the board's `submissions/*.json`** (in `ari-leaderboard`) — `audit_hashes` is a SHA-256 digest of each
   condition's **code matrix** (`ari/report.py::condition_digest`, post-quantization,
   bit-packed uint8), not of the raw float matrix, and a hash cannot be inverted back to the
   vectors that produced it even if it were.
4. **The capture pipeline itself** (`ari/tools/run_report.py`, `ari/tools/_encode_worker.py`,
   `ari/tools/selfhosted_pilot.py`, `ari/tools/gpu_capture.py`, `ari/agents.py`) —
   read in full. Every path that touches a raw float vector (`base_vecs`, `enc(texts)`,
   `v.npy`) writes it, if at all, inside a `tempfile.TemporaryDirectory()` that is deleted
   the moment the `with` block exits (e.g. `run_report.py:68-72`,
   `selfhosted_pilot.py:63-64`, `gpu_capture.py:120-121`). The float matrix exists only long
   enough to be handed to `probe.encode(...)`, which returns the quantized code; nothing
   downstream (`ari/metrics.py::aggregate`, `ari/report.py::build_report`) ever sees or
   stores the float matrix.
5. **The one thing that *is* persisted long-term** — `ari/tools/capture_time_baseline.py` and
   `ari/tools/refresh_leaderboard.py` write `codes.npy` (the **quantized SEMQ code**, not
   floats) to `~/ari_time_baseline/<slug>/` (outside the repo, on whatever machine ran the
   capture) or to a local `_ari_baselines/` directory that CI syncs to an S3 bucket
   (`.github/workflows/refresh-leaderboard.yml`, `ARI_BASELINE_S3`). Even granting access to
   that bucket, it holds post-quantization codes, by design (`refresh_leaderboard.py:132`:
   `np.save(d / "codes.npy", codes)`) — not the vectors a `float_exact`/`float_cosine`
   detector would need.
6. **The one place raw floats were ever inspected at all**: the "pilot finding" in
   `spec/condition-set.md` §"`same`: a control..." describes OpenAI's per-call
   non-determinism "verified at the float level (~2,300/3,072 dimensions differ, magnitude
   ~1e-4)". `experiments/deployed-agent-panel/RESULTS.md` §7 confirms this was a one-off,
   ad-hoc check ("It is representation-level... which no cosine-based check surfaces") — the
   comparison that produced that sentence is not saved anywhere as data; only the prose
   conclusion survived.

## Conclusion

For all 13 panel submissions, the raw embedding vectors were computed, immediately quantized
by the SEMQ probe, and discarded. What survives is: the input text (`data/`), the quantized
codes' SHA-256 digests (`audit_hashes`), and aggregate HER/H̄ statistics. There is no float
matrix anywhere in this repo, in a referenced external artifact this task has access to, or
reconstructable from what's stored. `float_exact`, `float_cosine`, and `topk_overlap` all
require the original reference and candidate float vectors (or, for `topk_overlap`, the
ranked lists derived from them) — none of which exist for any of the 13 rows.

## What re-capturing would take

This is scoping information, not a plan to execute here — re-capture is explicitly a
separate task with its own budget:

- **Self-hosted (8 models).** Deterministic and locally reproducible: re-run
  `ari/tools/run_report.py --agent bge --model <id>` (or the equivalent selfhosted path) over
  `data/ari-bench-v0.1.jsonl`, but with the raw float matrix written to durable storage
  *before* quantization (the current code intentionally never does this). CPU/GPU time only,
  no external cost. Because these agents are bit-deterministic, a fresh capture today would
  reproduce the same floats as the original run.
- **API (5 providers: OpenAI, Voyage, Cohere, Mistral, Gemini).** Re-running today would
  produce *a* set of vectors, but not necessarily *the original* ones: several of these APIs
  are themselves non-deterministic per call (that's the whole point of the `same` diagnostic
  — see [`docs/analysis/null-relative-effects.md`](null-relative-effects.md)), so the specific
  float-level drift instance behind each published `HER` is gone for good, not just
  unpersisted. A new capture would measure *a* float-level drift for these providers, not
  *the* one already scored. It would also re-incur the original billed cost across 1,000
  inputs × the conditions each provider supports (`same`, `proc`, `conc`, `time`), times 5
  providers.
- Either way, the capture tooling itself needs a small, deliberate change (persist the
  pre-quantization float matrix somewhere durable, e.g. alongside `codes.npy` in the S3-synced
  baseline dir) before a future run could feed `float_exact`/`float_cosine`/`topk_overlap`.
  That change is not made in this PR, for the same reason the detectors aren't: writing the
  plumbing for data that doesn't exist yet is exactly what the stop condition rules out.

## Why not stub the detectors anyway

It would be easy to write `float_exact`/`float_cosine`/`topk_overlap` against a `Detector`
interface and have them all report `"untested"` today, "ready" for when floats exist. That
was deliberately not done: a detector that has never run against a real vector is an
unverified guess at an interface, not a working detector — the first real capture would be
the first time its I/O contract, its bootstrap-CI code, and its calibration-null logic ever
executed, which is exactly the kind of code this project's own instrument (SEMQ) exists to
avoid trusting on faith. Untested code sitting next to tested code also reads, to a future
skimmer of `ari/metrics.py`, as if it carries the same evidentiary weight as the shipped
`semq` detector. The Matrix view instead shows `float_exact` / `float_cosine` /
`topk_overlap` columns as `untested` for every row — an honest statement that these axes are
measured for nobody yet, not a placeholder for code that silently doesn't run.

## Scope note

The multi-detector schema already supports nesting arbitrary named detectors under a
condition (`spec/report-schema.json`, `$defs/conditionResult`, `ari/rpc.py`). When a future,
separately-budgeted capture task retains raw floats, `float_exact` / `float_cosine` /
`topk_overlap` slot into that same shape — no schema change needed. Nothing in this commit
depends on that happening; it is only noted so the next person doesn't have to re-derive it.
