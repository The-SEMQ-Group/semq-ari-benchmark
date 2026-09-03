# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.

"""Resolve the immutable Hub commit an experiment actually loaded.

A name is not an identity: `all-MiniLM-L6-v2` or `BeIR/scifact` can both change
under the same string, so a report that only records the name cannot be
re-derived. This resolves the commit the local snapshot is pinned to -- what
`from_pretrained` / `load_dataset` will really use, and it works offline -- and
falls back to the Hub API. It returns the sentinel ``UNKNOWN`` rather than
guessing, so an unattributable run is visibly unattributable instead of wearing
a made-up revision.

This is the single resolver the attestation references block uses. The cache
fingerprint in the experiment drivers grew the same logic independently; folding
those onto this function is a follow-up (they must keep resolving offline, which
is why the duplication is tolerated for now, not designed in).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# Recorded in a reference when the commit cannot be resolved. Not a pin:
# ari.attest.validate_references rejects it at sign time, because "whatever was
# latest" is not a target a verifier can re-fetch.
UNKNOWN = "unknown"

# Strings that name a moving target rather than an immutable commit. A reference
# pinned to one of these is not reproducible, so signing must refuse it.
NON_PINS = frozenset({"", UNKNOWN, "main", "master", "latest", "head", "HEAD"})


@lru_cache(maxsize=None)
def hub_revision(repo_id: str, repo_type: str = "model") -> str:
    """The resolved commit for a Hub repo, local snapshot first so it works offline.

    ``repo_type`` is ``"model"`` or ``"dataset"``. Any failure -- offline, gated,
    or a local path with no Hub identity -- resolves to ``UNKNOWN`` rather than
    raising, so a driver can always build a references block; the pin is enforced
    later, at sign time.
    """
    try:
        from huggingface_hub import HfApi, constants

        prefix = "datasets--" if repo_type == "dataset" else "models--"
        ref = (Path(constants.HF_HUB_CACHE) /
               f"{prefix}{repo_id.replace('/', '--')}" / "refs" / "main")
        if ref.is_file():
            return ref.read_text().strip()
        return HfApi().repo_info(repo_id, repo_type=repo_type).sha or UNKNOWN
    except Exception:
        return UNKNOWN


def is_pinned(revision) -> bool:
    """True when a revision names an immutable commit, not a moving ref."""
    return bool(revision) and str(revision).strip() not in NON_PINS
