"""Pipeline stages: inventory, ingest, process, qc.

Resumability and idempotence come from two rules, applied everywhere:

* raw downloads are skipped when the file already exists and its checksum matches;
* every parquet write is an upsert keyed on ``pk_id`` (plus role/frame where
  relevant), so reprocessing one penalty rewrites exactly its own rows.

Running any stage twice therefore produces the same tables, and running it after
an interruption resumes rather than restarts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pkcv import dedup
from pkcv.config import Config
from pkcv.io import read_parquet, upsert_parquet, utc_now, write_json, write_parquet
from pkcv.qc import checks as qc_checks
from pkcv.qc import overlays
from pkcv.sources import registry
from pkcv.sources.base import ACCESS_GATED, ACCESS_OPEN
from pkcv.temporal import build as temporal

log = logging.getLogger("pkcv")

_UPSERT_KEYS = {
    "metadata": ["pk_id"],
    "tracks": ["pk_id"],
    "poses": ["pk_id"],
    "ball": ["pk_id"],
    "geometry": ["pk_id"],
    "events": ["pk_id"],
    "temporal_frames": ["pk_id"],
    "temporal_snapshots": ["pk_id"],
    "qc": ["pk_id"],
}


@dataclass
class StageResult:
    stage: str
    counts: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "counts": self.counts, "notes": self.notes, "at": utc_now()}


# ----------------------------------------------------------------- inventory


def inventory(cfg: Config, sources: list[str] | None = None) -> StageResult:
    cfg.paths.ensure()
    slugs = registry.resolve(sources)
    reports = []
    for slug in slugs:
        adapter = registry.build(slug, cfg)
        log.info("inventory: %s", slug)
        try:
            report = adapter.inventory()
        except Exception as exc:  # a broken source must not stop the others
            log.exception("inventory failed for %s", slug)
            from pkcv.sources.base import ACCESS_ERROR, MEDIA_NONE, SourceReport

            report = SourceReport(
                source=slug, title=adapter.title, url="", access=ACCESS_ERROR,
                media_kind=MEDIA_NONE, access_note=f"{type(exc).__name__}: {exc}",
                checked_at_utc=utc_now(),
            )
        reports.append(report.to_dict())
    write_json(reports, cfg.paths.manifests / "source_inventory.json")
    counts = {
        r["source"]: {
            "access": r["access"],
            "media_kind": r["media_kind"],
            "discovered": r["n_pk_discovered"],
            "accessible": r["n_pk_accessible"],
        }
        for r in reports
    }
    return StageResult("inventory", counts, [n for r in reports for n in r["notes"]])


# -------------------------------------------------------------------- ingest


def ingest(
    cfg: Config,
    sources: list[str] | None = None,
    limit: int | None = None,
    include_gated: bool = False,
) -> StageResult:
    cfg.paths.ensure()
    slugs = registry.resolve(sources)
    counts: dict[str, Any] = {}
    notes: list[str] = []

    for slug in slugs:
        adapter = registry.build(slug, cfg)
        report = adapter.inventory()
        if report.access != ACCESS_OPEN and not (include_gated and report.access == ACCESS_GATED):
            counts[slug] = {"skipped": report.access, "note": report.access_note}
            notes.append(f"{slug}: skipped ({report.access}) -- {report.access_note}")
            continue
        log.info("fetch: %s", slug)
        fetch_report = adapter.fetch(limit=limit)
        notes.extend(f"{slug}: {n}" for n in fetch_report.notes)
        log.info("ingest: %s", slug)
        result = adapter.ingest(limit=limit)
        notes.extend(f"{slug}: {n}" for n in result.notes)
        written = {}
        for artifact, df in result.frames().items():
            path = cfg.paths.artifact(artifact)
            upsert_parquet(df, path, artifact, _UPSERT_KEYS[artifact])
            written[artifact] = int(len(df))
        counts[slug] = written

    # Deduplicate across everything ingested so far, not just this run.
    md = read_parquet(cfg.paths.artifact("metadata"), "metadata")
    if len(md):
        md, dupes = dedup.resolve_duplicates(md)
        write_parquet(md, cfg.paths.artifact("metadata"), "metadata")
        write_json(dupes.to_dict("records"), cfg.paths.manifests / "duplicates.json")
        counts["_dedup"] = dedup.summarise(md)
    return StageResult("ingest", counts, notes)


# ------------------------------------------------------------------- process


def _select(md: pd.DataFrame, limit: int | None, all_: bool, pk_ids: list[str] | None) -> pd.DataFrame:
    sel = md[md["is_primary"].astype(bool)]
    if pk_ids:
        return sel[sel["pk_id"].isin(pk_ids)]
    sel = sel[sel["media_kind"].isin(["video", "pose_table"])]
    if all_:
        return sel
    return sel.head(limit or 5)


def process(
    cfg: Config,
    limit: int | None = None,
    all_: bool = False,
    pk_ids: list[str] | None = None,
    force: bool = False,
) -> StageResult:
    cfg.paths.ensure()
    md = read_parquet(cfg.paths.artifact("metadata"), "metadata")
    if not len(md):
        return StageResult("process", {}, ["no metadata; run `pkcv ingest` first"])

    targets = _select(md, limit, all_, pk_ids)
    tracks_all = read_parquet(cfg.paths.artifact("tracks"), "tracks")
    poses_all = read_parquet(cfg.paths.artifact("poses"), "poses")
    ball_all = read_parquet(cfg.paths.artifact("ball"), "ball")
    done = set(read_parquet(cfg.paths.artifact("temporal_snapshots"), "temporal_snapshots").get("pk_id", []))

    frames_out, snaps_out, notes = [], [], []
    processor = None
    n_vision = 0

    for _, row in targets.iterrows():
        pk_id = row["pk_id"]
        if pk_id in done and not force:
            continue

        tracks = tracks_all[tracks_all["pk_id"] == pk_id]
        poses = poses_all[poses_all["pk_id"] == pk_id]
        ball = ball_all[ball_all["pk_id"] == pk_id]

        if row["media_kind"] == "video" and (not len(tracks) or force):
            from pkcv.vision import ClipProcessor, VisionConfig, VisionUnavailable

            try:
                if processor is None:
                    processor = ClipProcessor(VisionConfig.from_dict(cfg.vision))
                video = Path(cfg.paths.root) / str(row["video_relpath"])
                res = processor.process(pk_id, row["source"], video, float(row["fps"] or 25.0))
                for artifact, df in (
                    ("tracks", res.tracks), ("poses", res.poses), ("ball", res.ball),
                    ("geometry", res.geometry), ("events", res.events),
                ):
                    if len(df):
                        upsert_parquet(df, cfg.paths.artifact(artifact), artifact, _UPSERT_KEYS[artifact])
                tracks, poses, ball = res.tracks, res.poses, res.ball
                notes.extend(f"{pk_id}: {f}" for f in res.failures)
                n_vision += 1
            except VisionUnavailable as exc:
                notes.append(f"{pk_id}: vision skipped -- {exc}")
                continue

        if not len(tracks):
            notes.append(f"{pk_id}: no tracks; nothing to anchor")
            continue

        f = temporal.build_frames(row, tracks, poses, ball)
        s = temporal.build_snapshots(f, row)
        if len(f):
            frames_out.append(f)
        if len(s):
            snaps_out.append(s)

    counts: dict[str, Any] = {
        "selected": int(len(targets)),
        "vision_clips": n_vision,
        "temporal_frames_kicks": len(frames_out),
        "temporal_snapshot_kicks": len(snaps_out),
    }
    if frames_out:
        df = pd.concat(frames_out, ignore_index=True)
        upsert_parquet(df, cfg.paths.artifact("temporal_frames"), "temporal_frames", ["pk_id"])
        counts["temporal_frames_rows"] = int(len(df))
    if snaps_out:
        df = pd.concat(snaps_out, ignore_index=True)
        upsert_parquet(df, cfg.paths.artifact("temporal_snapshots"), "temporal_snapshots", ["pk_id"])
        counts["temporal_snapshot_rows"] = int(len(df))
        counts["snapshot_availability"] = float(df["snapshot_available"].astype(bool).mean())
    return StageResult("process", counts, notes)


# ------------------------------------------------------------------------ qc


def run_qc(cfg: Config, overlay_limit: int = 5, pk_ids: list[str] | None = None) -> StageResult:
    cfg.paths.ensure()
    md = read_parquet(cfg.paths.artifact("metadata"), "metadata")
    if not len(md):
        return StageResult("qc", {}, ["no metadata; run `pkcv ingest` first"])

    tracks = read_parquet(cfg.paths.artifact("tracks"), "tracks")
    poses = read_parquet(cfg.paths.artifact("poses"), "poses")
    ball = read_parquet(cfg.paths.artifact("ball"), "ball")
    events = read_parquet(cfg.paths.artifact("events"), "events")
    frames = read_parquet(cfg.paths.artifact("temporal_frames"), "temporal_frames")
    snaps = read_parquet(cfg.paths.artifact("temporal_snapshots"), "temporal_snapshots")

    graded = md[md["is_primary"].astype(bool) & md["media_kind"].isin(["video", "pose_table"])]
    if pk_ids:
        graded = graded[graded["pk_id"].isin(pk_ids)]

    rows = []
    for _, row in graded.iterrows():
        pk = row["pk_id"]
        rows.extend(
            qc_checks.check_kick(
                row,
                tracks[tracks["pk_id"] == pk],
                poses[poses["pk_id"] == pk],
                ball[ball["pk_id"] == pk],
                events[events["pk_id"] == pk],
                snaps[snaps["pk_id"] == pk],
                cfg.qc,
            )
        )
    qc_df = pd.DataFrame(rows)
    if len(qc_df):
        upsert_parquet(qc_df, cfg.paths.artifact("qc"), "qc", ["pk_id"])
        roll = qc_checks.rollup(qc_df).set_index("pk_id")
        md = md.set_index("pk_id")
        md.loc[roll.index, "qc_status"] = roll["qc_status"]
        md.loc[roll.index, "qc_reasons"] = roll["qc_reasons"]
        write_parquet(md.reset_index(), cfg.paths.artifact("metadata"), "metadata")
        md = md.reset_index()

    summary = qc_checks.corpus_summary(qc_df, md)
    write_json(summary, cfg.paths.qc / "qc_summary.json")

    # Visual QC on a sample, biased toward the worst kicks -- a contact sheet of
    # clean kicks proves nothing.
    made = []
    if overlay_limit:
        order = graded.copy()
        rank = {"fail": 0, "warn": 1, "pending": 2, "pass": 3}
        order["_r"] = order["qc_status"].map(rank).fillna(2)
        for _, row in order.sort_values("_r").head(overlay_limit).iterrows():
            pk = row["pk_id"]
            p = poses[poses["pk_id"] == pk]
            f = frames[frames["pk_id"] == pk]
            s = snaps[snaps["pk_id"] == pk]
            if not len(p) or not len(f):
                continue
            safe = pk.replace(":", "__").replace("/", "_")
            try:
                sheet = overlays.snapshot_sheet(row, p, f, s, cfg.paths.qc / "overlays" / f"{safe}_snapshots.png")
                bg = (
                    Path(cfg.paths.root) / str(row["video_relpath"])
                    if pd.notna(row.get("video_relpath"))
                    else None
                )
                vid = overlays.overlay_video(
                    row, p, f, ball[ball["pk_id"] == pk],
                    cfg.paths.qc / "overlays" / f"{safe}_overlay.mp4",
                    background_video=bg if bg and bg.exists() else None,
                )
                made.extend(str(x) for x in (sheet, vid) if x)
            except Exception as exc:
                log.exception("overlay failed for %s", pk)
                rows.append({"pk_id": pk, "overlay_error": str(exc)})

    counts = {
        "kicks_graded": int(len(graded)),
        "qc_rows": int(len(qc_df)),
        "status_counts": md["qc_status"].value_counts().to_dict() if len(md) else {},
        "overlays_written": len(made),
    }
    return StageResult("qc", counts, [f"overlay: {m}" for m in made])
