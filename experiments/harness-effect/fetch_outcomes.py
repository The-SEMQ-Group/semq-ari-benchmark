# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""Fetch the ARI-E outcome columns of nvidia/Open-SWE-Traces at a pinned revision.

The metric reads three columns: ``instance_id``, ``trajectory_id`` and
``resolved``. The transcripts hold almost all of the bytes and the metric never
reads them. Parquet stores columns separately, so this reads the file footers,
adds up the byte size of the three columns, checks that sum against a budget,
and then requests only those byte ranges. No full data file is downloaded.

The output is one gzip-compressed CSV with one row per rollout plus a manifest
that records the revision, the source files with their sizes and LFS digests, the bytes read,
and the SHA-256 of the CSV. run.py reads the CSV and checks it against the
manifest, so the analysis runs offline and against a known input.

Usage, from the repository root:

    python experiments/harness-effect/fetch_outcomes.py --revision <commit>

``--revision`` must be a commit hash. A branch name is refused because it does
not identify the data.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE.parents[1]))

DATASET = "nvidia/Open-SWE-Traces"
COLUMNS = ("instance_id", "trajectory_id", "resolved")
HARNESSES = ("sweagent", "openhands")
MODELS = ("qwen35_122b", "minimax_m25")

# The repository has used two layouts. Until 2026-08-21 a cell was one
# directory, ``data/<model>_<harness>_trajectories/``, and the Qwen model was
# written ``qwen35``. Since then a cell is ``data/<harness>/<model>/<source>/``.
# Both name the harness and the model as whole path tokens, so one pattern
# covers both. ``minisweagent`` does not match ``sweagent`` because the token
# must start after a separator.
_HARNESS = re.compile(r"(?:^|[/_])(openhands|sweagent)(?=[/_.]|$)")
_MODEL = re.compile(r"(?:^|[/_])(qwen35_122b|qwen35|minimax_m25)(?=[/_.]|$)")


def cell_of(path: str) -> tuple[str, str] | None:
    """(harness, model) for a parquet path, or None when it is another cell."""
    if not path.endswith(".parquet") or not path.startswith("data/"):
        return None
    h = _HARNESS.search(path)
    m = _MODEL.search(path)
    if not h or not m:
        return None
    model = {"qwen35": "qwen35_122b"}.get(m.group(1), m.group(1))
    return h.group(1), model


class _Counting:
    """File wrapper that counts the bytes pyarrow actually requests."""

    def __init__(self, f):
        self._f = f
        self.bytes_read = 0

    def read(self, n=-1):
        b = self._f.read(n)
        self.bytes_read += len(b)
        return b

    def __getattr__(self, name):
        return getattr(self._f, name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", required=True,
                    help="dataset commit hash to read")
    ap.add_argument("--budget-mb", type=float, default=2048.0,
                    help="abort when the projected column bytes exceed this")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    args = ap.parse_args()

    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        sys.exit("--revision must be a 40-character commit hash")

    import pyarrow
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    api = HfApi()
    info = api.dataset_info(DATASET, revision=args.revision, files_metadata=True)
    if info.sha != args.revision:
        sys.exit(f"resolved {info.sha}, expected {args.revision}")

    files = []
    for s in info.siblings:
        cell = cell_of(s.rfilename)
        if cell is None:
            continue
        files.append({
            "path": s.rfilename, "harness": cell[0], "model": cell[1],
            "size_bytes": int(s.size or 0),
            "lfs_sha256": (s.lfs or {}).get("sha256") if isinstance(s.lfs, dict)
            else getattr(s.lfs, "sha256", None),
        })
    files.sort(key=lambda f: f["path"])
    if not files:
        sys.exit("no parquet files matched the four cells")
    repo_bytes = sum(int(s.size or 0) for s in info.siblings)
    cell_bytes = sum(f["size_bytes"] for f in files)
    print(f"revision {args.revision}")
    print(f"repository {repo_bytes / 1e9:.2f} GB, four cells "
          f"{cell_bytes / 1e9:.2f} GB in {len(files)} parquet files")

    fs = HfFileSystem()

    def open_file(path):
        # cache_type="none" makes every read an exact byte range, so the byte
        # count below is the transfer and not a read-ahead buffer.
        return fs.open(f"datasets/{DATASET}@{args.revision}/{path}", "rb",
                       cache_type="none")

    # Phase 1: footers only. Project the bytes the three columns need.
    projected = 0
    footer_bytes = 0
    for f in files:
        with open_file(f["path"]) as raw:
            cf = _Counting(raw)
            md = pq.ParquetFile(cf).metadata
            footer_bytes += cf.bytes_read
        f["rows"] = md.num_rows
        f["row_groups"] = md.num_row_groups
        need = 0
        for rg in range(md.num_row_groups):
            g = md.row_group(rg)
            for ci in range(g.num_columns):
                c = g.column(ci)
                if c.path_in_schema in COLUMNS:
                    need += c.total_compressed_size
        f["column_bytes"] = need
        projected += need
    print(f"footers read {footer_bytes / 1e6:.1f} MB; the three columns need "
          f"{projected / 1e6:.1f} MB")
    if projected > args.budget_mb * 1e6:
        sys.exit(f"projected {projected / 1e6:.1f} MB exceeds the budget of "
                 f"{args.budget_mb:.0f} MB; nothing read")

    # Phase 2: the three columns.
    rows = []
    data_bytes = 0
    for f in files:
        with open_file(f["path"]) as raw:
            cf = _Counting(raw)
            table = pq.ParquetFile(cf, pre_buffer=True).read(columns=list(COLUMNS))
            data_bytes += cf.bytes_read
        d = table.to_pydict()
        for iid, tid, res in zip(d["instance_id"], d["trajectory_id"], d["resolved"]):
            rows.append((f["harness"], f["model"], str(iid), str(tid), int(res),
                         f["path"]))
        print(f"  {f['path']}  {len(d['resolved'])} rows", flush=True)
    rows.sort()

    tag = args.revision[:12]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"outcomes.{tag}.csv.gz"
    # mtime=0 keeps the archive byte-identical across runs, so its digest
    # depends on the rows alone.
    with gzip.GzipFile(out, "wb", compresslevel=9, mtime=0) as gz:
        text = ["harness,model,instance_id,trajectory_id,resolved,source_file"]
        text += [",".join(map(str, r)) for r in rows]
        gz.write(("\n".join(text) + "\n").encode())

    per_cell = Counter((r[0], r[1]) for r in rows)
    ungraded = Counter((r[0], r[1]) for r in rows if r[4] < 0)
    keys = Counter((r[0], r[1], r[3]) for r in rows)
    duplicate_ids = sum(1 for v in keys.values() if v > 1)

    manifest = {
        "dataset": DATASET,
        "revision": args.revision,
        "revision_last_modified": info.last_modified.isoformat()
        if info.last_modified else None,
        "columns": list(COLUMNS),
        "repository_bytes": repo_bytes,
        "cell_bytes": cell_bytes,
        "projected_column_bytes": projected,
        "bytes_read": {"footers": footer_bytes, "columns": data_bytes,
                       "total": footer_bytes + data_bytes},
        "files": files,
        "output": {"name": out.name, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                   "size_bytes": out.stat().st_size, "rows": len(rows)},
        "rows_per_cell": {f"{h}/{m}": per_cell[(h, m)] for h in HARNESSES for m in MODELS},
        "ungraded_per_cell": {f"{h}/{m}": ungraded[(h, m)] for h in HARNESSES for m in MODELS},
        "duplicate_trajectory_ids": duplicate_ids,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tools": {"pyarrow": pyarrow.__version__,
                  "huggingface_hub": __import__("huggingface_hub").__version__},
    }
    mpath = args.out_dir / f"outcomes.{tag}.manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {out} ({len(rows)} rows, sha256 {manifest['output']['sha256'][:12]}...)")
    print(f"wrote {mpath}")
    print(f"bytes read {manifest['bytes_read']['total'] / 1e6:.1f} MB")
    for k in manifest["rows_per_cell"]:
        print(f"  {k:<24} rows {manifest['rows_per_cell'][k]:>7}"
              f"  ungraded {manifest['ungraded_per_cell'][k]:>6}")
    if duplicate_ids:
        print(f"warning: {duplicate_ids} trajectory ids appear more than once in a cell")


if __name__ == "__main__":
    main()
