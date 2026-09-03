"""Control-surface probes for the platform subject, with retained evidence.

The RESULTS claims about the Salesforce Models API's control surface (the
`parameters` block silently ignored on both endpoints) were first observed
interactively; this script re-runs them as bound evidence — every request and
raw response lands in ``results/transcripts/salesforce_control_probes.jsonl.gz``
and its digest in the manifest, so the claims are checkable, not anecdotal.

  python probe_control_surface.py

Probes (both endpoints where applicable):
  1. maxTokens 8 against a long-essay prompt — is the cap applied?
  2. temperature as a non-numeric string — is the field validated?
  3. maxTokens as a non-numeric string — is the field validated?
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from providers import SUBJECTS, Caller, SalesforceAuth

HERE = Path(__file__).resolve().parent

PROBES = [
    {"name": "maxTokens_cap_generations", "path": "/einstein/platform/v1/models/{model}/generations",
     "body": {"prompt": "Write a 200-word essay about rivers.",
              "parameters": {"temperature": 0, "maxTokens": 8}}},
    {"name": "maxTokens_cap_chat", "path": "/einstein/platform/v1/models/{model}/chat-generations",
     "body": {"messages": [{"role": "user", "content": "Write a 200-word essay about rivers."}],
              "parameters": {"temperature": 0, "maxTokens": 8}}},
    {"name": "temperature_garbage", "path": "/einstein/platform/v1/models/{model}/generations",
     "body": {"prompt": "hi", "parameters": {"temperature": "BANANA", "maxTokens": 8}}},
    {"name": "maxTokens_garbage", "path": "/einstein/platform/v1/models/{model}/generations",
     "body": {"prompt": "hi", "parameters": {"maxTokens": "BANANA"}}},
    {"name": "toplevel_temperature_garbage", "path": "/einstein/platform/v1/models/{model}/generations",
     "body": {"prompt": "hi", "temperature": "BANANA"}},
]


def main() -> int:
    subject = SUBJECTS["salesforce"]
    auth = SalesforceAuth(os.environ["SF_MY_DOMAIN"],
                          os.environ["SF_CONSUMER_KEY"],
                          os.environ["SF_CONSUMER_SECRET"])
    out = HERE / "results" / "transcripts" / "salesforce_control_probes.jsonl.gz"
    records = []
    for probe in PROBES:
        caller = Caller(subject, auth)
        # bypass build_body: the probe IS the request body, verbatim
        payload = json.dumps(probe["body"]).encode()
        path = probe["path"].format(model=subject.model)
        import http.client
        conn = http.client.HTTPSConnection(subject.host, timeout=120)
        conn.request("POST", path, body=payload, headers={
            "Authorization": f"Bearer {auth.token()}",
            "Content-Type": "application/json",
            **subject.extra_headers,
        })
        resp = conn.getresponse()
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        body = json.loads(raw) if resp.status == 200 else {"raw": raw.decode(errors="replace")}
        usage = (body.get("parameters") or {}).get("usage") or {}
        gen = body.get("generation") or {}
        gdet = body.get("generationDetails") or {}
        rec = {
            "probe": probe["name"],
            "endpoint": path,
            "request_body": probe["body"],
            "status": resp.status,
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": (gen.get("parameters") or {}).get("finish_reason"),
            "response": body,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        records.append(rec)
        print(f"{probe['name']}: HTTP {rec['status']}  completion_tokens={rec['completion_tokens']}")
        conn.close()

    with gzip.open(out, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    mpath = HERE / "results" / "manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["runs"] = [r for r in manifest["runs"]
                        if r.get("transcript") != out.name]
    manifest["runs"].append({
        "transcript": out.name, "sha256": digest, "calls": len(records),
        "scope": "control-surface probes", "subject": "salesforce",
        "class": "platform", "model": subject.model,
        "endpoint": f"https://{subject.host}/einstein/platform/v1/...",
        "condition": "probe", "k": None, "burst": None, "params": {},
        "prompt_content_hash": None, "reconnects": 0, "wall_s": None,
    })
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"evidencia: {out.name} sha256={digest[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
