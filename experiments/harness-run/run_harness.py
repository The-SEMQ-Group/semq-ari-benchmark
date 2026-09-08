"""Run one harness over the frozen ARI-E task set, k times, and emit trajectories.

Drives a coding harness (SWE-agent or OpenHands) against one model endpoint, k
independent rollouts per task, then grades every patch with the held-out tests and
writes the JSON Lines `ari.harness` consumes:

    {"case_id": <instance_id>, "harness": "sweagent"|"openhands", "run": <k>,
     "outcome": <resolved>, "tool_calls": [...]}

The endpoint is the only thing that changes between a hosted smoke and the block: pass
--api-base for a self-hosted vLLM server. Everything else -- the task set, k, grading,
the emitted format -- is identical, and identical across the two harnesses.

Self-consistency needs within-harness variation, so rollouts run at temperature > 0;
at temperature 0 every run would be near-identical and D_self would be ~0 by
construction, which is exactly the noise the control exists to measure.

    # SWE-agent, hosted smoke
    python run_harness.py --harness sweagent --model gpt-4o \
        --instance-ids psf__requests-2317 --k 2 --temperature 0.7

    # OpenHands, self-hosted (block)
    python run_harness.py --harness openhands --model openai/qwen3-coder-30b \
        --api-base http://localhost:8000/v1 --n 200 --k 5 --workers 16

Grading reuses grade.py (proven against a gold patch). Run once per harness; the report
(emit_report.py) combines both harnesses' trajectories.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HARNESSES = REPO.parent / "harnesses"
FROZEN = REPO / "data" / "ari-e-bench-v0.1.jsonl"
GRADER_DATASET = "SWE-bench/SWE-bench_Verified"

# Pinned harness revisions (spec §5: a row records the exact harness SHA).
PINS = {
    "sweagent": "3ea751c087f32b16e039a2233dd6eefecef325d5",   # v1.1.0
    "openhands": "7fbb48c40679afd674970966b96185657d92a487",  # v0.62.0 (last V0 with the eval harness)
}
SWEAGENT_BIN = HERE / ".venv-sweagent" / "bin" / "sweagent"
SWEAGENT_CONFIG = HARNESSES / "SWE-agent" / "config" / "default.yaml"
OPENHANDS_DIR = HARNESSES / "OpenHands-v0"
OPENHANDS_PY = HERE / ".venv-openhands-v0" / "bin" / "python"


def frozen_instance_ids(n: int | None) -> list[str]:
    ids = [json.loads(l)["instance_id"] for l in FROZEN.read_text().splitlines() if l.strip()]
    return ids[:n] if n else ids


# ---------------------------------------------------------------------------
# SWE-agent
# ---------------------------------------------------------------------------

def run_sweagent(instance_ids, *, model, api_base, temperature, out_dir, workers,
                 cost_limit, **_) -> dict[str, dict]:
    """Return {instance_id: {"patch": diff, "tool_calls": [...]}}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(SWEAGENT_BIN), "run-batch",
        "--instances.type", "swe_bench", "--instances.subset", "verified",
        "--instances.split", "test",
        "--instances.filter", "|".join(f"^{i}$" for i in instance_ids),
        "--instances.evaluate", "False",
        "--config", str(SWEAGENT_CONFIG),
        "--agent.model.name", model,
        "--agent.model.temperature", str(temperature),
        "--agent.model.per_instance_cost_limit", str(cost_limit),
        "--output_dir", str(out_dir), "--num_workers", str(workers),
    ]
    if api_base:
        cmd += ["--agent.model.api_base", api_base]
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    preds = json.loads((out_dir / "preds.json").read_text())
    result = {}
    for iid, p in preds.items():
        result[iid] = {"patch": p.get("model_patch") or "",
                       "tool_calls": _sweagent_tool_calls(out_dir, iid)}
    return result


def _sweagent_tool_calls(out_dir: Path, instance_id: str) -> list[str]:
    traj = out_dir / instance_id / f"{instance_id}.traj"
    if not traj.exists():
        return []
    try:
        steps = json.loads(traj.read_text()).get("trajectory", [])
    except Exception:
        return []
    return [str(s.get("action", "")).strip().split("\n")[0][:80]
            for s in steps if isinstance(s, dict) and s.get("action")]


# ---------------------------------------------------------------------------
# OpenHands (V0 eval harness)
# ---------------------------------------------------------------------------

def run_openhands(instance_ids, *, model, api_base, temperature, out_dir, workers,
                  max_iter=100, **_) -> dict[str, dict]:
    """Return {instance_id: {"patch": diff, "tool_calls": [...]}}.

    OpenHands V0 reads the LLM from `[llm.<name>]` in the repo-root config.toml
    (--llm-config), takes the instance list via --eval-ids, and writes
    test_result.git_patch per instance to output.jsonl.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # LLM config (repo-root config.toml, read by --llm-config qwen).
    api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    llm_lines = [f'[llm.qwen]', f'model = "{model}"',
                 f'api_key = "{api_key}"', f'temperature = {temperature}']
    if api_base:
        llm_lines.insert(2, f'base_url = "{api_base}"')
    (OPENHANDS_DIR / "config.toml").write_text("\n".join(llm_lines) + "\n")

    # Instance selection: the swe_bench eval dir's config.toml `selected_ids` list is
    # the path run_infer.filter_dataset actually honors. (--eval-ids / --eval-n-limit
    # are unreliable here: --eval-n-limit triggers a *random* sample of that many from
    # the full set, which silently runs the wrong instances.)
    sel = OPENHANDS_DIR / "evaluation" / "benchmarks" / "swe_bench" / "config.toml"
    sel.write_text("selected_ids = [" + ", ".join(f'"{i}"' for i in instance_ids) + "]\n")

    cmd = [
        str(OPENHANDS_PY), "-m", "evaluation.benchmarks.swe_bench.run_infer",
        "--agent-cls", "CodeActAgent", "--llm-config", "qwen",
        "--dataset", GRADER_DATASET, "--split", "test",
        "--eval-num-workers", str(workers), "--max-iterations", str(max_iter),
        "--eval-output-dir", str(out_dir),
    ]
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(OPENHANDS_DIR), env={**os.environ})

    result = {}
    for of in out_dir.rglob("output.jsonl"):
        for line in of.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            iid = d["instance_id"]
            result[iid] = {
                "patch": (d.get("test_result") or {}).get("git_patch") or "",
                "tool_calls": _openhands_tool_calls(d.get("history") or []),
            }
    return result


def _openhands_tool_calls(history) -> list[str]:
    calls = []
    for ev in history:
        if isinstance(ev, dict):
            act = ev.get("action") or ev.get("tool")
            if act:
                calls.append(str(act)[:80])
    return calls


RUNNERS = {"sweagent": run_sweagent, "openhands": run_openhands}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a harness over ARI-E-Bench, k times.")
    ap.add_argument("--harness", choices=list(RUNNERS), required=True)
    ap.add_argument("--model", required=True, help="litellm model name (gpt-4o, "
                    "openai/qwen3-coder-30b, together_ai/Qwen/Qwen2.5-Coder-32B-Instruct)")
    ap.add_argument("--api-base", default=None, help="OpenAI-compatible base URL (vLLM on the block)")
    ap.add_argument("--n", type=int, default=None, help="first N tasks (prefix); default all 200")
    ap.add_argument("--instance-ids", default=None,
                    help="comma-separated instance_ids to run instead of the prefix (smoke)")
    ap.add_argument("--k", type=int, required=True, help="rollouts per task")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cost-limit", type=float, default=1.0, help="per-instance $ cap (SWE-agent API safety)")
    ap.add_argument("--max-iter", type=int, default=100, help="OpenHands max agent iterations")
    ap.add_argument("--timeout", type=int, default=1800, help="grading timeout per instance")
    ap.add_argument("--out-dir", type=Path, default=HERE / "runs" / "live")
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    from grade import run_swebench  # proven grading wrapper

    if args.instance_ids:
        want = {i.strip() for i in args.instance_ids.split(",")}
        ids = [i for i in frozen_instance_ids(None) if i in want]
        missing = want - set(ids)
        if missing:
            raise SystemExit(f"instance_ids not in the frozen set: {sorted(missing)}")
    else:
        ids = frozen_instance_ids(args.n)
    runner = RUNNERS[args.harness]
    print(f"harness={args.harness} (pin {PINS[args.harness]})  model={args.model}  "
          f"tasks={len(ids)}  k={args.k}")

    trajectories = []
    for run in range(args.k):
        run_dir = args.out_dir / args.harness / f"run{run}"
        print(f"\n=== {args.harness} run {run+1}/{args.k} -> {run_dir} ===", flush=True)
        t0 = time.time()
        outputs = runner(ids, model=args.model, api_base=args.api_base,
                         temperature=args.temperature, out_dir=run_dir,
                         workers=args.workers, cost_limit=args.cost_limit,
                         max_iter=args.max_iter)
        print(f"  {len(outputs)} patches in {time.time()-t0:.0f}s", flush=True)

        for iid in ids:
            patch = (outputs.get(iid) or {}).get("patch", "")
            rid = f"{args.harness}_run{run}_{iid}"
            r = run_swebench([{"instance_id": iid, "model_name_or_path": rid,
                               "model_patch": patch}], rid, workers=1,
                             timeout=args.timeout, report_dir=args.out_dir / "grading")
            trajectories.append({
                "case_id": iid, "harness": args.harness, "run": run,
                "outcome": bool(r.get(iid, False)),
                "tool_calls": (outputs.get(iid) or {}).get("tool_calls", []),
            })

    out = args.out_dir / f"trajectories_{args.harness}.jsonl"
    out.write_text("\n".join(json.dumps(t) for t in trajectories) + "\n")
    n_res = sum(t["outcome"] for t in trajectories)
    print(f"\nwrote {len(trajectories)} rollouts -> {out}  ({n_res} resolved)")
    print(f"harness pin: {args.harness}@{PINS[args.harness]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
