"""Source adapter contract.

A source adapter answers three questions, in order:

1. ``inventory()``  -- what exists, under what licence, and can we legally get it?
2. ``fetch()``      -- download the raw bytes into ``data/raw/<slug>/``.
3. ``ingest()``     -- map those bytes onto the canonical schema.

``inventory()`` must never download the payload and must never fail hard: a
blocked source returns a report saying so, and the run continues.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# Access states, ordered from most to least usable.
ACCESS_OPEN = "open"  # downloadable now, licence permits research use
ACCESS_GATED = "gated"  # legitimate access exists but needs a credential we do not hold
ACCESS_BLOCKED = "blocked"  # no lawful route to the media
ACCESS_ERROR = "error"  # probe failed; state unknown

MEDIA_VIDEO = "video"
MEDIA_POSE_TABLE = "pose_table"
MEDIA_RENDER_ONLY = "render_only"
MEDIA_NONE = "none"


@dataclass
class SourceReport:
    """Everything the inventory step learned about one candidate source."""

    source: str
    title: str
    url: str
    access: str
    media_kind: str
    doi: str | None = None
    license: str | None = None
    license_url: str | None = None
    attribution: str | None = None
    access_note: str = ""
    n_pk_discovered: int | None = None
    n_pk_accessible: int | None = None
    labels_available: list[str] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)
    redistribute_derived: bool = False
    redistribute_video: bool = False
    redistribution_note: str = ""
    deposit_family: str | None = None
    notes: list[str] = field(default_factory=list)
    checked_at_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class IngestResult:
    """Canonical frames produced by an adapter. Any may be empty."""

    metadata: pd.DataFrame
    tracks: pd.DataFrame | None = None
    poses: pd.DataFrame | None = None
    ball: pd.DataFrame | None = None
    geometry: pd.DataFrame | None = None
    events: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)

    def frames(self) -> dict[str, pd.DataFrame]:
        out = {"metadata": self.metadata}
        for name in ("tracks", "poses", "ball", "geometry", "events"):
            df = getattr(self, name)
            if df is not None and len(df):
                out[name] = df
        return out


class SourceAdapter(ABC):
    #: short, stable, filesystem-safe identifier -- becomes the pk_id prefix
    slug: str = "unnamed"
    #: human title
    title: str = ""
    #: groups every publication route for one underlying collection, so that a
    #: mirror of an existing deposit deduplicates against it
    deposit_family: str = ""

    def __init__(self, cfg, source_cfg: dict[str, Any] | None = None):
        self.cfg = cfg
        self.source_cfg = source_cfg or {}

    @property
    def raw_dir(self) -> Path:
        p = Path(self.cfg.paths.raw) / self.slug
        p.mkdir(parents=True, exist_ok=True)
        return p

    @abstractmethod
    def inventory(self) -> SourceReport: ...

    @abstractmethod
    def fetch(self, limit: int | None = None) -> SourceReport: ...

    @abstractmethod
    def ingest(self, limit: int | None = None) -> IngestResult: ...
