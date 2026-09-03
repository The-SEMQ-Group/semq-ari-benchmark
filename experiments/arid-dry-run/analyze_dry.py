"""ARI-D dry-run analysis: detectors + the three calibrations, pure over transcripts.

  python analyze_dry.py            # reads results/, writes results/analysis.json

Computes, per (subject, condition):

- ``exact_generation``  — mean over prompts of the byte-identical pair share,
                          bootstrap CI over prompts (§7, §8)
- ``first_divergence``  — d/max(len_a, len_b) in BYTES, prefix rule,
                          identical pairs read 1.0 (§7)
- ``topk_logprob_overlap`` — mean Jaccard of top-k token sets per step, over
                          steps up to and including the first divergent chosen
                          token; identical sequences use every step (§7)

and the three dry-run calibrations (§13, item 3):

1. burst sweep table  — conc at each measured burst vs the same floor
2. length confounder  — Spearman rho between pair max byte length and the
                        pair's first_divergence value, per bucket, over
                        non-identical pairs (if length dominates, redesign)
3. set discrimination — same floor per bucket + top-2 logprob margin
                        distribution per bucket (a saturated, high-margin set
                        separates nothing; triggers the §4 amendment path)

Δ-vs-same uses the Matrix's conservative CI-overlap classification.
"""
from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
B = 2000  # bootstrap resamples over prompts


# ---------------------------------------------------------------- detectors

def first_divergence(a: bytes, b: bytes) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return d / max(len(a), len(b))


def topk_overlap_pair(lp_a: list[dict], lp_b: list[dict]) -> float | None:
    if not lp_a or not lp_b:
        return None
    n = min(len(lp_a), len(lp_b))
    vals = []
    for i in range(n):
        sa = {t for t, _ in lp_a[i]["top"]}
        sb = {t for t, _ in lp_b[i]["top"]}
        if not sa or not sb:
            break
        vals.append(len(sa & sb) / len(sa | sb))
        if lp_a[i]["token"] != lp_b[i]["token"]:
            break  # include the divergent step, stop after it
    return sum(vals) / len(vals) if vals else None


def boot_ci(per_prompt: list[float], seed_tag: str) -> tuple[float, float, float]:
    mean = sum(per_prompt) / len(per_prompt)
    rng = random.Random(int(hashlib.sha256(seed_tag.encode()).hexdigest()[:12], 16))
    n = len(per_prompt)
    means = sorted(
        sum(per_prompt[rng.randrange(n)] for _ in range(n)) / n for _ in range(B)
    )
    return mean, means[int(0.025 * B)], means[int(0.975 * B)]


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 8 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = rank(xs), rank(ys)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((a - ma) * (b - mb) for a, b in zip(ra, rb))
    va = sum((a - ma) ** 2 for a in ra) ** 0.5
    vb = sum((b - mb) ** 2 for b in rb) ** 0.5
    return cov / (va * vb) if va and vb else None


# ---------------------------------------------------------------- aggregation

def load_run(entry: dict) -> dict[str, list[dict]]:
    by_prompt = defaultdict(list)
    with gzip.open(RESULTS / "transcripts" / entry["transcript"], "rt") as f:
        for line in f:
            rec = json.loads(line)
            by_prompt[rec["input_id"]].append(rec)
    return by_prompt


def condition_stats(entry: dict, by_prompt: dict) -> dict:
    tag = f"{entry['subject']}:{entry['condition']}:{entry.get('burst')}"
    per_prompt_exact, per_prompt_fd, per_prompt_topk = [], [], []
    bucket_exact = defaultdict(list)
    conf_len, conf_fd = defaultdict(list), defaultdict(list)
    fingerprints = set()

    for pid, recs in sorted(by_prompt.items()):
        recs = sorted(recs, key=lambda r: r["rep"])
        bucket = recs[0]["bucket"]
        for r in recs:
            fp = r["parsed"].get("system_fingerprint")
            if fp:
                fingerprints.add(fp)
        texts = [r["parsed"]["text"].encode() for r in recs]
        lps = [r["parsed"]["logprobs"] for r in recs]
        ex, fd, tk = [], [], []
        for i, j in itertools.combinations(range(len(texts)), 2):
            ex.append(1.0 if texts[i] == texts[j] else 0.0)
            v = first_divergence(texts[i], texts[j])
            fd.append(v)
            if texts[i] != texts[j]:
                conf_len[bucket].append(float(max(len(texts[i]), len(texts[j]))))
                conf_fd[bucket].append(v)
            o = topk_overlap_pair(lps[i], lps[j])
            if o is not None:
                tk.append(o)
        per_prompt_exact.append(sum(ex) / len(ex))
        per_prompt_fd.append(sum(fd) / len(fd))
        bucket_exact[bucket].append(sum(ex) / len(ex))
        if tk:
            per_prompt_topk.append(sum(tk) / len(tk))

    e_mean, e_lo, e_hi = boot_ci(per_prompt_exact, tag + ":exact")
    f_mean, f_lo, f_hi = boot_ci(per_prompt_fd, tag + ":fd")
    out = {
        "transcript": entry["transcript"],
        **{k: entry[k] for k in ("subject", "class", "model", "condition", "k",
                                 "burst", "scope", "calls", "reconnects")},
        "exact_generation": {"mean": round(e_mean, 4), "ci": [round(e_lo, 4), round(e_hi, 4)]},
        "first_divergence": {"mean": round(f_mean, 4), "ci": [round(f_lo, 4), round(f_hi, 4)]},
        "same_floor_per_bucket": {
            b: round(sum(v) / len(v), 4) for b, v in sorted(bucket_exact.items())
        },
        "length_confounder_rho_per_bucket": {
            b: (None if (r := spearman(conf_len[b], conf_fd[b])) is None else round(r, 3))
            for b in sorted(conf_len)
        },
        "system_fingerprints": sorted(fingerprints),
    }
    if per_prompt_topk:
        t_mean, t_lo, t_hi = boot_ci(per_prompt_topk, tag + ":topk")
        out["topk_logprob_overlap"] = {"mean": round(t_mean, 4), "ci": [round(t_lo, 4), round(t_hi, 4)]}
    if entry["condition"] == "conc":
        out["achieved_in_flight_peak"] = entry.get("achieved_in_flight_peak")
    return out


def margin_distribution(by_prompt: dict) -> dict:
    """Top-2 logprob margin per step, rep 0 of `same` (§13's discrimination
    check): a set whose margins are uniformly wide cannot show drift."""
    per_bucket = defaultdict(list)
    for pid, recs in by_prompt.items():
        r0 = min(recs, key=lambda r: r["rep"])
        for step in r0["parsed"]["logprobs"]:
            if len(step["top"]) >= 2:
                per_bucket[r0["bucket"]].append(step["top"][0][1] - step["top"][1][1])
    out = {}
    for b, ms in sorted(per_bucket.items()):
        ms.sort()
        n = len(ms)
        out[b] = {
            "steps": n,
            "median": round(ms[n // 2], 3),
            "p10": round(ms[n // 10], 3),
            "share_below_0.1_nats": round(sum(1 for m in ms if m < 0.1) / n, 4),
        }
    return out


def classify(cond: dict, same: dict) -> str:
    lo_c, hi_c = cond["exact_generation"]["ci"]
    lo_s, hi_s = same["exact_generation"]["ci"]
    if hi_c < lo_s:
        return "below same (drift)"
    if lo_c > hi_s:
        return "above same (noise-floor artifact, never an improvement)"
    return "≈ same"


def main() -> int:
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    conditions = []
    margins = {}
    same_by_subject = {}
    for entry in manifest["runs"]:
        if entry["condition"] not in ("same", "proc", "conc"):
            continue  # control-surface probes carry no per-prompt records
        by_prompt = load_run(entry)
        stats = condition_stats(entry, by_prompt)
        conditions.append(stats)
        if entry["condition"] == "same":
            same_by_subject[entry["subject"]] = stats
            margins[entry["subject"]] = margin_distribution(by_prompt)

    for c in conditions:
        s = same_by_subject.get(c["subject"])
        if s is not None and c["condition"] != "same":
            c["vs_same"] = classify(c, s)

    analysis = {"conditions": conditions, "top2_margins_same_rep0": margins}
    (RESULTS / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")

    for c in conditions:
        burst = f" b={c['burst']}" if c["burst"] else ""
        extra = f"  vs_same: {c['vs_same']}" if "vs_same" in c else ""
        print(f"{c['subject']:<9} {c['condition']}{burst:<7} "
              f"exact={c['exact_generation']['mean']:.4f} {c['exact_generation']['ci']}  "
              f"fd={c['first_divergence']['mean']:.4f}{extra}")
    for subj, m in margins.items():
        print(f"-- margins {subj}: " + ", ".join(
            f"{b}: med={v['median']} p10={v['p10']}" for b, v in m.items()))
    print("wrote results/analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
