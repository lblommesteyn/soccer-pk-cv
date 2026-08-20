"""Publish derived artifacts to a Hugging Face dataset repository.

What goes up is decided by licence, per record, not per run:

* every derived table (metadata, tracks, poses, ball, geometry, events,
  temporal, QC) for records whose source permits redistributing derivatives;
* original video **only** where the source licence explicitly permits
  redistribution *and* the record is flagged ``redistribute_video``;
* for everything else, the source reference travels instead of the media.

Rows whose source forbids redistributing derivatives are withheld too, and the
withholding is reported in the upload manifest rather than passing silently.

The token is read from ``HF_TOKEN`` in the environment. It is never written to
disk, never logged, and never committed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from pkcv.config import Config, hf_token
from pkcv.io import read_parquet, utc_now, write_json, write_parquet
from pkcv.schemas import PROCESSING_VERSION

log = logging.getLogger("pkcv.hf")

ARTIFACTS = (
    "metadata",
    "tracks",
    "poses",
    "ball",
    "geometry",
    "events",
    "temporal_frames",
    "temporal_snapshots",
    "qc",
)

REMOTE_PATHS = {
    "metadata": "data/metadata.parquet",
    "tracks": "data/tracks.parquet",
    "poses": "data/poses.parquet",
    "ball": "data/ball.parquet",
    "geometry": "data/geometry.parquet",
    "events": "data/events.parquet",
    "temporal_frames": "data/temporal_frames.parquet",
    "temporal_snapshots": "data/temporal_snapshots.parquet",
    "qc": "data/qc.parquet",
}


class HFTokenMissing(RuntimeError):
    pass


@dataclass
class PublishPlan:
    repo_id: str
    files: list[tuple[Path, str]] = field(default_factory=list)
    withheld: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def _publishable_pk_ids(md: pd.DataFrame) -> tuple[set[str], list[dict[str, Any]]]:
    allow = md["redistribute_derived"].astype(bool)
    withheld = [
        {
            "pk_id": r["pk_id"],
            "source": r["source"],
            "reason": "source does not permit redistributing derived artifacts",
            "license": r["license"],
        }
        for _, r in md[~allow].iterrows()
    ]
    return set(md.loc[allow, "pk_id"]), withheld


def build_plan(cfg: Config, repo_id: str, staging: Path) -> PublishPlan:
    """Filter every artifact by licence and stage what may lawfully be uploaded."""
    staging.mkdir(parents=True, exist_ok=True)
    md = read_parquet(cfg.paths.artifact("metadata"), "metadata")
    if not len(md):
        raise RuntimeError("no metadata to publish; run `pkcv ingest` first")

    allowed, withheld = _publishable_pk_ids(md)
    plan = PublishPlan(repo_id=repo_id, withheld=withheld)

    for artifact in ARTIFACTS:
        df = read_parquet(cfg.paths.artifact(artifact), artifact)
        if not len(df):
            continue
        before = len(df)
        df = df[df["pk_id"].isin(allowed)]
        out = staging / Path(REMOTE_PATHS[artifact]).name
        write_parquet(df, out, artifact)
        plan.files.append((out, REMOTE_PATHS[artifact]))
        plan.stats[artifact] = {"rows": int(len(df)), "rows_withheld": int(before - len(df))}

    # Video, only where the licence says so explicitly.
    video_rows = md[md["redistribute_video"].astype(bool) & md["video_relpath"].notna()]
    for _, r in video_rows.iterrows():
        src = Path(cfg.paths.root) / str(r["video_relpath"])
        if src.exists():
            plan.files.append((src, f"video/{r['source']}/{Path(src).name}"))
    plan.stats["video_files"] = int(len(video_rows))
    plan.stats["video_withheld"] = int(
        (md["has_video"].astype(bool) & ~md["redistribute_video"].astype(bool)).sum()
    )
    return plan


def write_manifest(cfg: Config, plan: PublishPlan, staging: Path) -> Path:
    manifest = {
        "repo_id": plan.repo_id,
        "processing_version": PROCESSING_VERSION,
        "published_at_utc": utc_now(),
        "files": [remote for _, remote in plan.files],
        "stats": plan.stats,
        "withheld_records": plan.withheld,
    }
    path = staging / "manifest.json"
    write_json(manifest, path)
    plan.files.append((path, "manifest.json"))
    return path


def publish(
    cfg: Config,
    repo_id: str | None = None,
    dry_run: bool = False,
    private: bool = True,
) -> dict[str, Any]:
    repo_id = repo_id or cfg.hf.get("repo_id")
    if not repo_id:
        raise RuntimeError("no HF repo_id configured")
    staging = Path(cfg.paths.root) / "hf_staging"
    plan = build_plan(cfg, repo_id, staging)

    from pkcv.hf.card import write_card

    card_path = write_card(cfg, staging / "README.md", repo_id)
    plan.files.append((card_path, "README.md"))
    write_manifest(cfg, plan, staging)

    result = {
        "repo_id": repo_id,
        "dry_run": dry_run,
        "files": [remote for _, remote in plan.files],
        "stats": plan.stats,
        "withheld": len(plan.withheld),
        "staging": str(staging),
    }
    if dry_run:
        return result

    token = hf_token()
    if not token:
        raise HFTokenMissing(
            "HF_TOKEN is not set. Export a write-scoped token and re-run; "
            "the pipeline never stores tokens on disk."
        )
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    for local, remote in plan.files:
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"pkcv {PROCESSING_VERSION}: {remote}",
        )
        log.info("uploaded %s", remote)
    result["uploaded"] = len(plan.files)
    return result
