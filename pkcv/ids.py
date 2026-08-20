"""Stable, globally unique penalty-kick identifiers and deduplication keys.

``pk_id`` is intentionally human-readable and deterministic:

    <source_slug>:<normalised source identifier>

Determinism matters more than prettiness: re-ingesting a source must produce
byte-identical ids so downstream artifacts stay joinable and the pipeline stays
idempotent. ``pk_uid`` is the 16-hex digest of ``pk_id`` for use as a filename
or a partition key.
"""

from __future__ import annotations

import hashlib
import re

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    return _SLUG_RE.sub("-", str(value).strip().lower()).strip("-")


def normalise_identifier(value: str) -> str:
    """Normalise a source-native id so equivalent ids collapse.

    Zero-padded and non-padded numeric components are made equivalent
    (``07-03`` and ``7-3`` are the same kick in the Mendeley/figshare pair),
    which is what makes cross-source deduplication possible at all.
    """
    raw = str(value).strip()
    parts = re.split(r"[-_./ ]+", raw)
    out = []
    for part in parts:
        if not part:
            continue
        out.append(str(int(part)) if part.isdigit() else part.lower())
    return "-".join(out) if out else raw.lower()


def make_pk_id(source_slug: str, source_identifier: str) -> str:
    return f"{slugify(source_slug)}:{normalise_identifier(source_identifier)}"


def make_pk_uid(pk_id: str) -> str:
    return hashlib.sha1(pk_id.encode("utf-8")).hexdigest()[:16]


def make_dedup_key(
    *,
    deposit_family: str,
    source_identifier: str,
) -> str:
    """Key under which two records are considered *the same physical kick*.

    ``deposit_family`` groups every publication route for one underlying data
    collection (e.g. the Mendeley record and its figshare mirror both declare
    ``womens-collegiate-pifer-li``). Within a family, the normalised source
    identifier is the kick.

    Records from different families are never merged automatically: we have no
    evidence that a kick in one corpus is the same physical event as a kick in
    another, and a wrong merge is worse than a duplicate.
    """
    return f"{slugify(deposit_family)}/{normalise_identifier(source_identifier)}"
