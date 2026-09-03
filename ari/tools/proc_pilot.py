"""`proc` pilot — resolve the three sub-decisions for the API `proc` condition.

For a black-box API we cannot impose "new process"; we approximate it with **fresh-context
resampling** (fresh client + connection). This pilot measures, for one provider:

  same  — K reencodes on the SAME persistent client (within-session determinism control)
  proc  — K reencodes on a FRESH client/connection each time (cross-context drift)

and reports HER / H̄ per resample, so we can decide: does fresh-context actually induce
drift? how many resamples K stabilise the estimate? (The window is a wall-clock knob for the
real run.) Run it on AWS where OPENAI_API_KEY is set:

    python proc_pilot.py --agent openai --n 200 --resamples 8 --out proc_pilot.json

Offline logic check (no key, no network), simulating a provider that drifts across contexts:

    python proc_pilot.py --agent mock --n 200 --resamples 8 --mock-drift 3e-3 --probe-backend mock
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ari import metrics                         # noqa: E402
from ari.agents import (                          # noqa: E402
    MockAgent, OpenAIAgent, VoyageAgent, CohereAgent)
from ari.inputs import sample_inputs, load_ari_bench  # noqa: E402
from ari.probe import load_probe                 # noqa: E402


def build_api_agent(name: str, model: str | None, dimensions: int | None, max_workers: int):
    if name == "openai":
        return OpenAIAgent(model_id=model or "text-embedding-3-large",
                           dimensions=dimensions, max_workers=max_workers)
    if name == "voyage":
        return VoyageAgent(model_id=model or "voyage-4-large", max_workers=max_workers)
    if name == "cohere":
        return CohereAgent(model_id=model or "embed-v4.0", max_workers=max_workers)
    raise ValueError(f"unknown API agent: {name}")


class APISource:
    """Generic black-box API source: `same` reuses the persistent client, `proc` uses a
    fresh client/connection. Works for OpenAI / Voyage / Cohere alike."""

    def __init__(self, agent):
        self.agent = agent

    def baseline(self, texts):
        return self.agent.encode(texts)                  # persistent client

    def resample(self, kind, texts, i):
        return self.agent.encode(texts) if kind == "same" else self.agent.encode_fresh(texts)


class MockSource:
    """Deterministic within-session; injects per-resample noise on `proc` to emulate a
    provider whose routing sends fresh contexts to slightly different backends."""

    def __init__(self, drift: float, dim: int = 256):
        self.agent = MockAgent(dim=dim)
        self.drift = drift

    def baseline(self, texts):
        return self.agent.encode(texts)

    def resample(self, kind, texts, i):
        X = self.agent.encode(texts)
        if kind == "same" or self.drift == 0.0:
            return X
        rng = np.random.default_rng(1000 + i)            # different per fresh-context resample
        return X + rng.normal(0.0, self.drift, size=X.shape)


def _bootstrap_her_ci(he_mat: np.ndarray, n_boot: int = 1000, seed: int = 0):
    """95% CI on HER by BLOCK bootstrap over inputs (resample whole input rows, so the K
    resamples of an input stay together — accounts for input-level correlation). he_mat is
    (n_inputs, K) boolean."""
    rng = np.random.default_rng(seed)
    n = he_mat.shape[0]
    stats = np.empty(n_boot)
    for b in range(n_boot):
        rows = rng.integers(0, n, n)
        stats[b] = he_mat[rows].mean()
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return [round(float(lo), 6), round(float(hi), 6)]


def run_pilot(source, inputs, k: int, probe_backend: str) -> dict:
    base_vecs = source.baseline(inputs.texts)
    probe = load_probe(base_vecs, backend=probe_backend)
    base_codes = probe.encode(base_vecs)

    n = len(inputs)
    out = {"n_inputs": n, "resamples": k, "n_observations": n * k,
           "probe_backend": probe.backend, "conditions": {}}
    for kind in ("same", "proc"):
        he_cols, hbars = [], []
        for i in range(k):
            he, hamming = metrics.per_input(base_codes, probe.encode(source.resample(kind, inputs.texts, i)))
            he_cols.append(he); hbars.append(float(hamming.mean()))
        he_mat = np.stack(he_cols, axis=1)                       # (n, K)
        out["conditions"][kind] = {
            "HER": round(float(he_mat.mean()), 6),
            "HER_ci95": _bootstrap_her_ci(he_mat),               # block bootstrap over inputs
            "Hbar_mean": round(st.mean(hbars), 4),
            "per_resample_HER": [round(float(c.mean()), 6) for c in he_cols],
        }
    same, proc = out["conditions"]["same"], out["conditions"]["proc"]
    # proc vs same "significant" only if their 95% CIs do not overlap
    out["findings"] = {
        "within_session_deterministic": same["HER"] >= 1.0 - 1e-9,
        "proc_HER": proc["HER"],
        "proc_vs_same_ci_overlap": not (proc["HER_ci95"][0] > same["HER_ci95"][1]
                                        or same["HER_ci95"][0] > proc["HER_ci95"][1]),
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="API `proc` fresh-context resampling pilot.")
    ap.add_argument("--agent", choices=["openai", "voyage", "cohere", "mock"], default="mock")
    ap.add_argument("--model", default=None, help="provider default if omitted")
    ap.add_argument("--dimensions", type=int, default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--resamples", type=int, default=8)
    ap.add_argument("--inputs", type=Path, help="ARI-Bench JSONL; omit for the sample set")
    ap.add_argument("--probe-backend", choices=["auto", "semq", "mock"], default="auto")
    ap.add_argument("--max-workers", type=int, default=16, help="concurrent API requests")
    ap.add_argument("--mock-drift", type=float, default=3e-3, help="mock only: proc σ")
    ap.add_argument("--out", type=Path, default=Path("proc_pilot.json"))
    args = ap.parse_args(argv)

    inputs = load_ari_bench(args.inputs) if args.inputs else sample_inputs(args.n)
    source = (MockSource(args.mock_drift) if args.agent == "mock"
              else APISource(build_api_agent(args.agent, args.model, args.dimensions, args.max_workers)))
    result = run_pilot(source, inputs, args.resamples, args.probe_backend)
    result["agent"] = "mock" if args.agent == "mock" else source.agent.agent_id
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    f, c = result["findings"], result["conditions"]
    print(f"agent={result['agent']}  probe={result['probe_backend']}  "
          f"n={result['n_inputs']}  K={result['resamples']}  (N={result['n_observations']} obs)")
    for kind in ("same", "proc"):
        ci = c[kind]["HER_ci95"]
        print(f"  {kind} HER = {c[kind]['HER']:.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]  "
              f"H̄={c[kind]['Hbar_mean']:.2f}")
    print(f"  within-session deterministic: {f['within_session_deterministic']}  |  "
          f"proc vs same CIs overlap (proc>same is noise): {f['proc_vs_same_ci_overlap']}")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
