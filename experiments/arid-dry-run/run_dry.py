"""ARI-D dry run, rollout item 3: run one condition for one subject.

  python run_dry.py --subject together --condition same
  python run_dry.py --subject together --condition conc --burst 64
  python run_dry.py --subject together --condition sweep        # conc at 16/64/128
  python run_dry.py --subject openai --condition same --limit 10  # smoke test

Writes one transcript per (subject, condition[, burst]) under
``results/transcripts/`` as gzipped JSONL — one record per measured call,
raw response included (§5: transcripts retained and bound) — and updates
``results/manifest.json`` with the transcript digests, the prompt-set hash,
and the request parameters. The transcripts are the evidence; ``analyze_dry.py``
is pure over them.

The prompt-set hash is verified against the frozen pin before any call is
made; ``--limit N`` runs the first N prompts and records ``prefix:N``
(§4: valid, marked not comparable).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

from providers import SUBJECTS, Caller, SalesforceAuth, StaticToken, MAX_TOKENS, SEED

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROMPTS = REPO / "data" / "arid-bench-v0.1.jsonl"
FROZEN_HASH = "af5b1a2bb35b9fa5f8d4e8f05b85a98f495a3edeb93e95da9c9ced6d783e0d6f"

K = {"same": 8, "proc": 8, "conc": 12}  # §6


def load_prompts(limit: int | None) -> tuple[list[dict], str]:
    items, h = [], hashlib.sha256()
    for line in PROMPTS.open():
        it = json.loads(line)
        h.update(it["input_id"].encode() + b"\x00" + it["text"].encode() + b"\n")
        items.append(it)
    if h.hexdigest() != FROZEN_HASH:
        sys.exit(f"prompt set does not match the frozen pin ({h.hexdigest()})")
    return (items[:limit] if limit else items), (f"prefix:{limit}" if limit else "full")


def record(prompt: dict, rep: int, result: dict, extra: dict) -> dict:
    return {
        "input_id": prompt["input_id"],
        "bucket": prompt["bucket"],
        "rep": rep,
        **extra,
        **result,
    }


def build_auth(subject):
    if subject.endpoint_kind == "sfgen":
        need = ("SF_MY_DOMAIN", "SF_CONSUMER_KEY", "SF_CONSUMER_SECRET")
        missing = [v for v in need if not os.environ.get(v)]
        if missing:
            sys.exit(f"missing env vars: {', '.join(missing)}")
        return SalesforceAuth(os.environ["SF_MY_DOMAIN"],
                              os.environ["SF_CONSUMER_KEY"],
                              os.environ["SF_CONSUMER_SECRET"])
    key = os.environ.get(subject.key_env)
    if not key:
        sys.exit(f"{subject.key_env} not set")
    return StaticToken(key)


def run_same(caller: Caller, prompts: list[dict], k: int, out, pace: float) -> None:
    """k back-to-back calls per prompt over ONE connection — the provider's
    own floor. Repeats are adjacent by design here; that is what `same` means."""
    for p in prompts:
        for rep in range(k):
            out(record(p, rep, caller.call(p["text"]), {"condition": "same"}))
            time.sleep(pace)


def run_proc(subject, auth, prompts: list[dict], k: int, out, pace: float) -> None:
    """Fresh connection (new TLS session) per call, sequential."""
    for p in prompts:
        for rep in range(k):
            c = Caller(subject, auth)
            try:
                out(record(p, rep, c.call(p["text"]), {"condition": "proc"}))
            finally:
                c.close()
            time.sleep(pace)


def run_conc(subject, auth, prompts: list[dict], k: int, burst: int, out) -> None:
    """All k x n calls dispatched through a pool that keeps `burst` in flight
    (§6: waves, no filler traffic). Work is ordered (rep, prompt) so one
    prompt's repeats are spread across the run, not adjacent. The achieved
    in-flight peak and every rate-limit retry are recorded, not hidden."""
    gauge_lock = threading.Lock()
    in_flight = 0
    peak = 0
    local = threading.local()

    def one(task):
        nonlocal in_flight, peak
        rep, p = task
        if not hasattr(local, "caller"):
            local.caller = Caller(subject, auth)
        with gauge_lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            r = local.caller.call(p["text"])
        finally:
            with gauge_lock:
                in_flight -= 1
        return record(p, rep, r, {"condition": "conc", "burst": burst})

    tasks = [(rep, p) for rep in range(k) for p in prompts]
    with concurrent.futures.ThreadPoolExecutor(max_workers=burst) as pool:
        for rec in pool.map(one, tasks):
            out(rec)
    out.meta["achieved_in_flight_peak"] = peak


class Writer:
    def __init__(self, path: Path):
        self.path = path
        self.n = 0
        self.meta: dict = {}
        self._f = gzip.open(path, "wt")

    def __call__(self, rec: dict) -> None:
        self._f.write(json.dumps(rec) + "\n")
        self.n += 1
        if self.n % 50 == 0:
            print(f"  {self.n} calls", flush=True)

    def close(self) -> str:
        self._f.close()
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def run_one(subject, auth, prompts, scope, condition: str, burst: int | None,
            pace: float = 0.0, label: str | None = None) -> dict:
    tag = f"{subject.name}_{condition}" + (f"_b{burst}" if condition == "conc" else "")
    if subject.endpoint_kind == "chat" and not subject.supports_seed:
        tag += "_noseed"
    if label:
        tag += f"_{label}"
    if scope != "full":
        # a prefix smoke must not overwrite (and orphan the digest of) a full
        # run's transcript — the manifest binds files by content
        tag += "_" + scope.replace(":", "")
    tdir = HERE / "results" / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    w = Writer(tdir / f"{tag}.jsonl.gz")
    print(f"== {tag}: {len(prompts)} prompts x k={K[condition]}", flush=True)
    t0 = time.monotonic()
    caller = Caller(subject, auth)
    try:
        if condition == "same":
            run_same(caller, prompts, K["same"], w, pace)
        elif condition == "proc":
            run_proc(subject, auth, prompts, K["proc"], w, pace)
        else:
            run_conc(subject, auth, prompts, K["conc"], burst, w)
    finally:
        caller.close()
    digest = w.close()
    entry = {
        "transcript": w.path.name,
        "sha256": digest,
        "calls": w.n,
        "scope": scope,
        "subject": subject.name,
        "class": subject.klass,
        "model": subject.model,
        "endpoint": f"https://{subject.host}{subject.path}",
        "condition": condition,
        "k": K[condition],
        "burst": burst if condition == "conc" else None,
        "params": {"temperature": 0, "top_p": 1, "max_tokens": MAX_TOKENS,
                   **({"seed": SEED} if subject.supports_seed else {}),
                   **({"pace_s": pace} if pace else {})},
        "prompt_content_hash": FROZEN_HASH,
        "reconnects": caller.reconnects,
        "wall_s": round(time.monotonic() - t0, 1),
        **w.meta,
    }
    print(f"   done: {w.n} calls in {entry['wall_s']}s, sha256={digest[:16]}...", flush=True)
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ARI-D dry run: one subject, one condition.")
    ap.add_argument("--subject", required=True, choices=list(SUBJECTS))
    ap.add_argument("--condition", required=True, choices=["same", "proc", "conc", "sweep"])
    ap.add_argument("--burst", type=int, default=64)
    ap.add_argument("--model", default=None, help="override the subject's default model")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts (prefix:N, smoke tests)")
    ap.add_argument("--pace", type=float, default=0.0,
                    help="seconds between sequential calls (rate-capped orgs; same/proc only)")
    ap.add_argument("--no-seed", action="store_true",
                    help="omit the seed (confounder arm: quantifies the seed's contribution "
                         "when comparing against a door that cannot send one)")
    ap.add_argument("--label", default=None,
                    help="suffix for the transcript tag (e.g. t2 for the time re-run, so a "
                         "second pass never overwrites the first pass's digest-bound evidence)")
    args = ap.parse_args(argv)

    subject = SUBJECTS[args.subject]
    if args.model:
        subject.model = args.model
    if args.no_seed:
        subject.supports_seed = False
    auth = build_auth(subject)
    prompts, scope = load_prompts(args.limit)

    entries = []
    if args.condition == "sweep":
        if subject.klass != "rehosted":
            sys.exit("the burst sweep runs on the open-model rehosted row (§6)")
        for burst in (16, 64, 128):
            entries.append(run_one(subject, auth, prompts, scope, "conc", burst))
    else:
        entries.append(run_one(subject, auth, prompts, scope, args.condition,
                               args.burst if args.condition == "conc" else None,
                               pace=args.pace, label=args.label))

    mpath = HERE / "results" / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"runs": []}
    manifest["runs"].extend(entries)
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {len(manifest['runs'])} runs recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
