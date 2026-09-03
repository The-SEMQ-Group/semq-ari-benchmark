"""Weekly re-measurement of the commercial-API rows, in one shot.

Runs the full comparable core — `same`, `proc`, `conc`, `time` — for each black-box
embedding API, assembles ONE schema-valid signed report per provider, and *upserts* the
row into `leaderboard.json` (replacing the prior row for that agent, not appending a
duplicate — which `scoring/score.py --append` would do). Self-hosted rows are deterministic
and are never re-measured.

It reuses the tested measurement primitives from the `ari` package (the same code that
produced the current board): `agents`, `probe.fixed_scale_codes`, `metrics.aggregate`,
`report.build_report`. The only new logic here is orchestration, the rolling `time`
baseline, and the upsert.

    # dry run — no network, no keys; exercises the whole pipeline with a fake agent
    python refresh_leaderboard.py --mock --limit 64 --agents openai,cohere

    # real run (keys in env: OPENAI_API_KEY, VOYAGE_API_KEY, CO_API_KEY, MISTRAL_API_KEY,
    # GEMINI_API_KEY). --baseline-dir persists the `time` baseline between runs (S3-synced
    # by CI). First run per agent has no baseline -> `time` is skipped (bootstrap) and a
    # baseline is captured for next week.
    python refresh_leaderboard.py --inputs data/ari-bench-v0.1.jsonl \\
        --baseline-dir ./_ari_baselines --agents openai,voyage,cohere,mistral,gemini

The `time` axis is a rolling comparison: this run's re-encode vs the baseline captured
>=1 run ago. Order matters — we compare against the OLD baseline, then recapture a fresh
one. So the leaderboard's `time` reflects the gap since the last run (weekly cadence -> ~7d
gap, comfortably past the canonical 76h).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
sys.path.insert(0, str(REPO))

from ari import metrics, report                                   # noqa: E402
from ari.agents import DEFAULT_API_MODELS                         # noqa: E402
from ari.inputs import InputSet, load_ari_bench, sample_inputs    # noqa: E402
from ari.probe import load_probe, fixed_scale_codes              # noqa: E402

LEADERBOARD = REPO / "leaderboard" / "leaderboard.json"
SUBMISSIONS = REPO / "leaderboard" / "submissions"
SCORER = REPO / "leaderboard" / "scoring" / "score.py"
API = DEFAULT_API_MODELS
CALM_WORKERS, BURST_WORKERS = 2, 48
# mistral rate-limits hard; keep its burst gentle and cap every mistral encode's concurrency
# so a 1000-item batch doesn't 429 (the SDK already retries 5x on top of this).
BURST_OVERRIDE = {"mistral": 8}
WORKERS_CAP = {"mistral": 4}


def _semq_version(mock: bool) -> str:
    """The pinned SEMQ SDK version that produced these codes — recorded in the report so a
    calibration bump (and any encoding change) is auditable."""
    if mock:
        return "mock"
    try:
        import semq
        return str(getattr(semq, "__version__", "unknown"))
    except Exception:
        return "unavailable"


# --------------------------------------------------------------------------- mock
class _MockAgent:
    """A deterministic-per-seed fake for --mock: lets us exercise orchestration, report
    assembly, and the upsert end-to-end with no network. `drift` fakes provider
    non-determinism so the produced HER is < 1 (a realistic-looking row)."""

    def __init__(self, provider, model_id, drift=2e-3, dim=256, max_workers=8):
        self.provider, self._model, self.dim, self.drift = provider, model_id, dim, drift
        self.snapshot = "mock"
        self._rng = np.random.default_rng(abs(hash(provider)) % (2**32))
        self._base = self._rng.standard_normal((1, dim)).astype(np.float32)  # replaced per call size

    @property
    def agent_id(self):
        return self._model

    def _emit(self, texts):
        n = len(texts)
        base = np.random.default_rng(abs(hash((self.provider, n))) % (2**32)).standard_normal(
            (n, self.dim)).astype(np.float32)
        return base + self._rng.standard_normal((n, self.dim)).astype(np.float32) * self.drift

    def encode(self, texts):
        return self._emit(texts)

    def encode_fresh(self, texts):
        return self._emit(texts)

    def encode_batched(self, texts, batch_size=128):
        return self._emit(texts)   # mock: batch composition changes nothing → batch HER = 1.0


def make_agent(provider, model, workers, mock):
    if mock:
        return _MockAgent(provider, model or API[provider][1], max_workers=workers)
    cls, default = API[provider]
    return cls(model_id=model or default, max_workers=workers)


# --------------------------------------------------------------------------- baseline
def _slug(provider, agent):
    return f"{provider}_{agent.agent_id.replace('/', '_')}"


def load_baseline(base_dir, slug, content_hash):
    """Return (codes, scale_s, dim, unix_ts) for a usable prior baseline, else None."""
    d = base_dir / slug
    cp, mp = d / "codes.npy", d / "meta.json"
    if not (cp.exists() and mp.exists()):
        return None
    meta = json.loads(mp.read_text())
    if meta.get("content_hash") != content_hash:
        print(f"  [{slug}] baseline inputs differ (hash) — ignoring, will recapture")
        return None
    return np.load(cp), meta["scale_s"], meta["dim"], meta["unix_ts"]


def save_baseline(base_dir, slug, codes, scale_s, dim, content_hash, n, now):
    d = base_dir / slug
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "codes.npy", codes)
    (d / "meta.json").write_text(json.dumps(
        {"unix_ts": now, "scale_s": float(scale_s), "dim": int(dim),
         "content_hash": content_hash, "n": int(n)}, indent=2))


# --------------------------------------------------------------------------- measure
def measure_agent(provider, model, inputs, base_dir, workers, mock, now):
    """Measure the full core for one API and return its report — or None on a bootstrap run
    (no prior baseline yet), which only seeds the baseline and leaves the row untouched."""
    workers = min(workers, WORKERS_CAP.get(provider, workers))   # rate-limit-friendly per provider
    agent = make_agent(provider, model, workers, mock)
    slug = _slug(provider, agent)
    ch = inputs.content_hash
    prior = load_baseline(base_dir, slug, ch)
    # fixed-scale quantiser: the real SEMQ probe, or a self-contained numpy stand-in for --mock
    # (so --mock needs no SDK/keys/network — its whole purpose).
    q = ((lambda v, s, dim: np.clip(np.round(v / (s or 0.07)).astype(np.int32) + 128, 0, 255).astype(np.uint8))
         if mock else fixed_scale_codes)

    if prior is None:
        # bootstrap: capture the baseline (one encode + calibration) and stop — no row change,
        # no PR. Next run measures a real `time` against this and publishes the full row.
        v0 = agent.encode(inputs.texts)
        probe = load_probe(v0, backend="mock" if mock else "semq")
        s, dim = float(probe.s), int(v0.shape[1])
        # save via the same path the comparison uses next run, so the baseline and future `time`
        # codes are guaranteed shape/scale-consistent.
        save_baseline(base_dir, slug, q(v0, s, dim), s, dim, ch, len(inputs), now)
        print(f"  [{slug}] bootstrap — baseline seeded, row unchanged (time next run)")
        return None, slug

    # reuse the prior baseline's calibration scale so every condition (incl. time) is
    # comparable to it.
    prior_codes, s, dim, prior_ts = prior
    fs = lambda vecs: q(vecs, s, dim)
    base = fs(agent.encode(inputs.texts))          # E0 (this run)
    same = fs(agent.encode(inputs.texts))          # within-session re-encode
    proc = fs(agent.encode_fresh(inputs.texts))    # fresh client/connection
    calm = fs(make_agent(provider, model, CALM_WORKERS, mock).encode(inputs.texts))
    bw = BURST_OVERRIDE.get(provider, BURST_WORKERS)
    burst = fs(make_agent(provider, model, bw, mock).encode(inputs.texts))
    time_codes = fs(agent.encode(inputs.texts))    # vs the baseline captured a run ago

    metrics_by = {"same": metrics.aggregate(base, same),
                  "proc": metrics.aggregate(base, proc),
                  "conc": metrics.aggregate(calm, burst),
                  "time": metrics.aggregate(prior_codes, time_codes)}
    codes_by = {"same": same, "proc": proc, "conc": burst, "time": time_codes}
    print(f"  [{slug}] time gap since last baseline: {round((now - prior_ts) / 3600, 1)}h")

    # batch (diagnostic, not folded into ARI): same texts sent in batches vs one-per-request.
    try:
        batched = fs(agent.encode_batched(inputs.texts))
        metrics_by["batch"] = metrics.aggregate(base, batched)
        codes_by["batch"] = batched
    except NotImplementedError:
        pass                              # provider has no batched path (e.g. Gemini) — skip
    except Exception as e:                # a batch failure must not sink the row
        print(f"  [{slug}] batch axis skipped: {e}")

    environment = {"blas": "provider-internal", "threads": 0, "hardware": "provider-internal",
                   "precision": "provider-internal",
                   "library_versions": {"probe_backend": "mock" if mock else "semq",
                                        "semq": _semq_version(mock),
                                        "snapshot": str(agent.snapshot)}}
    # The signed report stays schema-clean (spec/report-schema.json is additionalProperties:
    # false). measured_at is leaderboard *display* metadata, attached to the row at upsert,
    # not baked into the cryptographic report.
    rep = report.build_report(
        agent_id=f"{provider}/{agent.agent_id}", input_set=inputs.name,
        input_content_hash=ch, environment=environment,
        metrics_by_condition=metrics_by, codes_by_condition=codes_by, agent_class="api",
        fingerprint={"dim": dim, "s": round(float(s), 6)})

    # recapture a fresh baseline for next run (AFTER the time comparison above)
    save_baseline(base_dir, slug, base, s, dim, ch, len(inputs), now)
    return rep, slug


# --------------------------------------------------------------------------- upsert
def validate(report_path, scorer=SCORER):
    """Gate through the canonical scorer (schema + invariants). Non-zero exit => reject."""
    return subprocess.call([sys.executable, str(scorer), str(report_path)]) == 0


def upsert_row(lb, report, measured_at, scorer=SCORER):
    """Replace the row for report['agent_id'] (or append if new), stamped measured_at."""
    from importlib import util
    spec = util.spec_from_file_location("score", scorer)
    score = util.module_from_spec(spec); spec.loader.exec_module(score)
    row = score.to_leaderboard_row(report)
    row["measured_at"] = measured_at
    entries = lb.setdefault("entries", [])
    for i, e in enumerate(entries):
        if e["agent_id"] == row["agent_id"]:
            entries[i] = row
            return "updated"
    entries.append(row)
    return "added"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Weekly re-measure of the commercial-API rows.")
    ap.add_argument("--agents", default="openai,voyage,cohere,mistral,gemini")
    ap.add_argument("--inputs", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--baseline-dir", type=Path, default=Path.home() / "ari_time_baseline")
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--mock", action="store_true", help="fake agents; no network/keys")
    ap.add_argument("--leaderboard", type=Path, default=LEADERBOARD)
    ap.add_argument("--submissions-dir", type=Path, default=None,
                    help="where measured reports are written (default: "
                         "leaderboard/submissions, or leaderboard/_mock_submissions "
                         "under --mock)")
    ap.add_argument("--scorer", type=Path, default=SCORER,
                    help="score.py used as the row gate (default: leaderboard/scoring/)")
    args = ap.parse_args(argv)

    inputs = load_ari_bench(args.inputs) if args.inputs else sample_inputs(args.limit or 64)
    if args.limit and len(inputs.ids) > args.limit:
        inputs = InputSet(inputs.name, inputs.ids[:args.limit], inputs.texts[:args.limit])

    lb = json.loads(args.leaderboard.read_text())
    now = time.time()
    measured_at = time.strftime("%Y-%m-%d", time.gmtime(now))
    sub_dir = args.submissions_dir or (
        (REPO / "leaderboard" / "_mock_submissions") if args.mock else SUBMISSIONS)
    changed, seeded, failures = [], [], []
    for provider in [a.strip() for a in args.agents.split(",") if a.strip()]:
        if provider not in API:
            print(f"skip unknown agent '{provider}'"); continue
        print(f"[{provider}] measuring same/proc/conc/time ...")
        try:
            rep, slug = measure_agent(provider, None, inputs, args.baseline_dir,
                                      args.max_workers, args.mock, now)
        except Exception as e:                       # one provider failing must not sink the run
            print(f"  [{provider}] FAILED: {e}"); failures.append(provider); continue

        if rep is None:                              # bootstrap: baseline seeded, no row change
            seeded.append(provider); continue

        sub_dir.mkdir(parents=True, exist_ok=True)
        rep_path = sub_dir / f"{rep['agent_id'].replace('/', '_')}.json"
        rep_path.write_text(json.dumps(rep, indent=2) + "\n")
        if not validate(rep_path, args.scorer):
            print(f"  [{provider}] report failed the scorer — not upserting"); failures.append(provider); continue
        action = upsert_row(lb, rep, measured_at, args.scorer)
        print(f"  [{provider}] {action}: ARI={rep['ARI']:.4f}")
        changed.append(provider)

    if seeded:
        print(f"\nseeded {len(seeded)} baseline(s): {', '.join(seeded)} (no row change)")
    if changed:
        lb["last_refreshed"] = measured_at
        args.leaderboard.write_text(json.dumps(lb, indent=2) + "\n")
        print(f"\nupdated {len(changed)} row(s): {', '.join(changed)} -> {args.leaderboard}")
        # append one time-series point per updated row (board's Trends view reads this)
        history = args.leaderboard.parent / "history.jsonl"
        with history.open("a") as fh:
            for p in changed:
                e = next((x for x in lb["entries"]
                          if x["agent_id"].split("/", 1)[0] == p and x["agent_class"] == "api"), None)
                if not e:
                    continue
                fh.write(json.dumps({"date": measured_at, "agent_id": e["agent_id"],
                                     "agent_class": e["agent_class"], "ARI": e["ARI"],
                                     "proc_HER": e.get("proc_HER"), "conc_HER": e.get("conc_HER"),
                                     "time_HER": e.get("time_HER")}) + "\n")
        print(f"appended {len(changed)} point(s) -> {history}")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
    # fail the job only if EVERYTHING failed — partial success must still persist baselines
    # (the S3-push step runs after this) and open a PR for the rows that did update.
    return 1 if (failures and not (changed or seeded)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
