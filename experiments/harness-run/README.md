# ARI-E-Bench v0.1 — harness runner

Produces the self-hosted ARI-E rows the spec describes (`../../spec/ari-e-bench-v0.1.md`):
SWE-agent vs OpenHands on one model, over the frozen 200-task set, graded by held-out
tests, `ARI = Δ = D_cross − D_self`.

## Pieces

- `../../ari/tools/build_ari_e_bench.py` — freezes the task set to
  `../../data/ari-e-bench-v0.1.jsonl` (deterministic, hash-pinned).
- `run_harness.py` — drives one harness over the task set, `k` rollouts per task against
  one model endpoint, grades every patch, writes `trajectories_<harness>.jsonl`.
- `grade.py` — grades a patch with the instance's held-out tests (the `swebench` harness).
- `emit_report.py` — combines both harnesses' trajectories into the shared ARI report
  envelope (reuses the tested `ari.harness` estimator; `ARI == same − harness`).

## Setup

The two harnesses have conflicting dependencies, so each is pinned and installed in its own
virtualenv; the grader and orchestration live in a third:

| venv | contents |
| --- | --- |
| `.venv` | `datasets`, `swebench` (grader), and the orchestration; `run_harness.py` runs here |
| `.venv-sweagent` | SWE-agent **v1.1.0** (`3ea751c087f32b16e039a2233dd6eefecef325d5`), `pip install -e` from a clone |
| `.venv-openhands-v0` | OpenHands **v0.62.0** (`7fbb48c40679afd674970966b96185657d92a487`) — the last V0 tag carrying the SWE-bench eval harness; V1's `openhands-ai` SDK dropped it — plus its eval deps (`datasets pandas swebench`) |

Both harnesses are cloned under `../../../harnesses/` (a sibling of the repo, referenced by
`run_harness.py`); adjust the paths there if you clone elsewhere.

Grading pulls each instance's prebuilt eval image and runs its tests, so **Docker is
required**. The OpenHands runtime starts an agent server *inside* the container, which needs
**native x86_64** — it does not come up under arm64 emulation, so run on a Linux x86 host.

## Run

Serve any OpenAI-compatible endpoint (a hosted API, or `vllm serve <model>
--tensor-parallel-size N` for a self-hosted model), then:

```bash
export OPENAI_API_KEY=...        # any value for a local vLLM; litellm requires it set
BASE=http://localhost:8000/v1    # omit --api-base for a hosted provider

# smoke one instance per harness first (confirms the endpoint + the OpenHands runtime)
python run_harness.py --harness sweagent  --model openai/<model> --api-base $BASE \
  --instance-ids psf__requests-2317 --k 1
python run_harness.py --harness openhands --model openai/<model> --api-base $BASE \
  --instance-ids psf__requests-2317 --k 1

# then the full row: each harness over the 200 tasks, k rollouts each
python run_harness.py --harness sweagent  --model openai/<model> --api-base $BASE --n 200 --k 5 --workers 16
python run_harness.py --harness openhands --model openai/<model> --api-base $BASE --n 200 --k 5 --workers 16

# combine both harnesses' trajectories and emit the report
cat runs/*/trajectories_sweagent.jsonl runs/*/trajectories_openhands.jsonl > trajectories.jsonl
python emit_report.py --trajectories trajectories.jsonl \
  --agent-id <model> --endpoint <endpoint-id> \
  --sweagent-sha 3ea751c087f32b16e039a2233dd6eefecef325d5 \
  --openhands-sha 7fbb48c40679afd674970966b96185657d92a487 \
  --input-content-hash d44f4975279798ce7271ee8971acafa9920199010d77811a9da4fbfdfaa91b26 \
  --k 5 --measured-at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --out report.json
```

Sign the report with `ARI_SIGNING_KEY` the same way the other layers do.

## Sizing and what a row resolves

Task count is set by power, not taste (see `../harness-power/`): 200 tasks × k=5 detects a
20-point pass-rate gap 96.5% of the time, 100 tasks 74%; below 100 a headline gap is not
resolvable. Rollouts run at temperature > 0 on purpose — self-consistency needs
within-harness variation, or `D_self` is ~0 by construction and the control is empty.

The cross-harness gap this pair produces is small (a few points of agreement), so a
controlled row resolves a *large* gap cleanly but its interval may overlap zero for a small
true effect. That is an honest null, not a failure: the high-n, small-effect estimate is the
reference reading cited in the spec, not something a controlled row is sized to reproduce.
The row's contribution is provenance — independent draws, a grader we run, an attested
report — over the same estimator.
