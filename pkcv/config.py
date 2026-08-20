"""Configuration and on-disk layout."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


@dataclass
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def qc(self) -> Path:
        return self.root / "qc"

    @property
    def metadata_parquet(self) -> Path:
        return self.processed / "metadata.parquet"

    def artifact(self, name: str) -> Path:
        """Path of a derived artifact parquet under processed/."""
        sub = {
            "metadata": "metadata.parquet",
            "tracks": "tracks/tracks.parquet",
            "poses": "poses/poses.parquet",
            "ball": "ball/ball.parquet",
            "geometry": "geometry/geometry.parquet",
            "events": "events/events.parquet",
            "temporal_frames": "temporal/temporal_frames.parquet",
            "temporal_snapshots": "temporal/temporal_snapshots.parquet",
            "qc": "qc/qc.parquet",
        }[name]
        return self.processed / sub

    def ensure(self) -> None:
        for p in (self.raw, self.manifests, self.processed, self.qc):
            p.mkdir(parents=True, exist_ok=True)
        for sub in ("tracks", "poses", "ball", "geometry", "events", "temporal", "qc"):
            (self.processed / sub).mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    paths: Paths
    sources: dict[str, Any] = field(default_factory=dict)
    vision: dict[str, Any] = field(default_factory=dict)
    temporal: dict[str, Any] = field(default_factory=dict)
    qc: dict[str, Any] = field(default_factory=dict)
    hf: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        data_root = os.environ.get("PKCV_DATA_ROOT") or data.get("data_root", "data")
        root = Path(data_root)
        if not root.is_absolute():
            root = REPO_ROOT / root
        return cls(
            paths=Paths(root),
            sources=data.get("sources", {}),
            vision=data.get("vision", {}),
            temporal=data.get("temporal", {}),
            qc=data.get("qc", {}),
            hf=data.get("hf", {}),
            raw=data,
        )

    def source_config(self, name: str) -> dict[str, Any]:
        return dict(self.sources.get(name, {}) or {})


def hf_token() -> str | None:
    """Token from the environment only. Never read from, or written to, the repo."""
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(var)
        if val:
            return val.strip()
    return None
