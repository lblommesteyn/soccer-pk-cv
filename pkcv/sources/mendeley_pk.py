"""Mendeley Data record ``brx9bsxnpx`` -- both published versions.

    v1  "EPL Penalty Kick Data Recorded by a Computer Vision Model"
        88 penalties, 2023-24 Premier League, broadcast footage.
    v2  "Women's Soccer Penalty Kick Data Captured by Computer Vision Models"
        132 penalties, US collegiate women, training + game footage.

Both are CC BY 4.0 and both are *pose tables*, not video: each ships a single
frame-level CSV of the kicker's smoothed bounding box and 12 COCO joints. The
"Steps to reproduce" text on both versions refers to a ``Videos.zip`` in an
"Instructions and Code" folder, but the public file listing of each version
contains only the CSV -- the clips were never deposited. That is recorded in the
inventory report rather than worked around, because it is the single fact that
determines what this pipeline can and cannot derive from these sources.

Consequence for the canonical schema: keeper, ball and goal geometry are
recorded as missing with reason ``source_provides_kicker_pose_only``. They are
not estimated, because there is nothing to estimate them from.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pkcv.ids import make_dedup_key, make_pk_id, make_pk_uid
from pkcv.io import sha256_file, utc_now
from pkcv.schemas import KEYPOINT_INDICES, KEYPOINT_NAMES, PROCESSING_VERSION
from pkcv.sources.base import (
    ACCESS_ERROR,
    ACCESS_OPEN,
    MEDIA_POSE_TABLE,
    IngestResult,
    SourceAdapter,
    SourceReport,
)

DATASET_ID = "brx9bsxnpx"
FILES_API = "https://data.mendeley.com/public-api/datasets/{ds}/files?folder_id=root&version={v}"
DATASET_API = "https://data.mendeley.com/public-api/datasets/{ds}?version={v}"
UA = {"User-Agent": "pkcv/0.1 (research dataset pipeline)", "Accept": "application/json"}

LICENSE = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
ATTRIBUTION = "Li, Feiting; Pifer, Nathan David (Florida State University). Mendeley Data, doi:10.17632/brx9bsxnpx"

#: The upstream pipeline is documented as YOLOv8-pose but the deposit does not
#: pin a weight file, so the version is recorded as unspecified rather than
#: guessed.
UPSTREAM_MODEL = "YOLOv8-pose (upstream, weights unspecified)"


def _get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


class MendeleyPKAdapter(SourceAdapter):
    """Base for the two versions; subclasses set the version-specific bits."""

    version: str = "1"
    csv_name: str = ""
    expected_kicks: int = 0

    # descriptive context that the deposit states in prose but not in columns
    competition: str | None = None
    gender: str | None = None
    level: str | None = None
    season: str | None = None
    default_fps: float | None = None
    fps_provenance: str = ""

    # ---------------------------------------------------------------- inventory

    def _remote_files(self) -> list[dict[str, Any]]:
        entries = _get_json(FILES_API.format(ds=DATASET_ID, v=self.version))
        out = []
        for e in entries:
            if e.get("filename") is None:
                continue
            cd = e.get("content_details") or {}
            out.append(
                {
                    "filename": e["filename"],
                    "size_bytes": cd.get("size"),
                    "content_type": cd.get("content_type"),
                    "sha256_upstream": cd.get("sha256_hash"),
                    "download_url": cd.get("download_url"),
                }
            )
        return out

    def inventory(self) -> SourceReport:
        report = SourceReport(
            source=self.slug,
            title=self.title,
            url=f"https://data.mendeley.com/datasets/{DATASET_ID}/{self.version}",
            doi=f"10.17632/{DATASET_ID}.{self.version}",
            access=ACCESS_OPEN,
            media_kind=MEDIA_POSE_TABLE,
            license=LICENSE,
            license_url=LICENSE_URL,
            attribution=ATTRIBUTION,
            deposit_family=self.deposit_family,
            redistribute_derived=True,
            redistribute_video=False,
            redistribution_note=(
                "CC BY 4.0 permits redistribution of derived tables with attribution. "
                "No video is present in this deposit, so no video redistribution question arises."
            ),
            labels_available=[
                "kick_direction",
                "goal",
                "footedness",
                "camera_direction",
            ],
            checked_at_utc=utc_now(),
        )
        try:
            meta = _get_json(DATASET_API.format(ds=DATASET_ID, v=self.version))
            report.title = meta.get("name") or report.title
            files = self._remote_files()
        except Exception as exc:  # network / API shape change
            report.access = ACCESS_ERROR
            report.access_note = f"probe failed: {type(exc).__name__}: {exc}"
            return report

        report.files = files
        names = {f["filename"] for f in files}
        report.n_pk_discovered = self.expected_kicks
        report.n_pk_accessible = self.expected_kicks if self.csv_name in names else 0
        report.access_note = f"public download, no credential required; {len(files)} file(s) deposited"
        if not any(f["filename"].lower().endswith((".mp4", ".zip", ".mov")) for f in files):
            report.notes.append(
                "No video in the public file listing. The deposit's 'Steps to reproduce' text "
                "references a Videos.zip inside an 'Instructions and Code' folder, but that folder "
                "is not published under this version -- the clips are not obtainable from Mendeley. "
                "Keeper, ball and goal geometry are therefore underivable from this source."
            )
        return report

    # ------------------------------------------------------------------- fetch

    @property
    def csv_path(self) -> Path:
        return self.raw_dir / self.csv_name

    def fetch(self, limit: int | None = None) -> SourceReport:
        report = self.inventory()
        if report.access != ACCESS_OPEN:
            return report
        target = next((f for f in report.files if f["filename"] == self.csv_name), None)
        if target is None:
            report.access = ACCESS_ERROR
            report.access_note = f"expected file {self.csv_name!r} not present upstream"
            return report
        if not self.csv_path.exists():
            req = urllib.request.Request(target["download_url"], headers={"User-Agent": UA["User-Agent"]})
            with urllib.request.urlopen(req, timeout=300) as resp, open(self.csv_path, "wb") as fh:
                fh.write(resp.read())
        digest = sha256_file(self.csv_path)
        report.notes.append(f"local sha256={digest}")
        if target.get("sha256_upstream") and target["sha256_upstream"] != digest:
            report.access = ACCESS_ERROR
            report.access_note = (
                f"checksum mismatch: upstream {target['sha256_upstream']} != local {digest}"
            )
        return report

    # ------------------------------------------------------ version-specific IO

    def _read_raw(self) -> pd.DataFrame:
        raise NotImplementedError

    def _kick_context(self, kick: pd.DataFrame) -> dict[str, Any]:
        return {}

    # ------------------------------------------------------------------ ingest

    def ingest(self, limit: int | None = None) -> IngestResult:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"{self.csv_path} not fetched; run `pkcv ingest` first")
        df = self._read_raw()
        notes: list[str] = []

        n_bad_frame = int(df["frame_num"].isna().sum())
        if n_bad_frame:
            notes.append(
                f"{n_bad_frame} row(s) dropped: non-numeric frame index in the source CSV"
            )
            df = df[df["frame_num"].notna()].copy()
        df["frame_num"] = df["frame_num"].astype(int)
        df["last_touch"] = pd.to_numeric(df["last_touch"], errors="coerce").fillna(0).astype(int)

        meta_rows, track_rows, pose_rows, event_rows, ball_rows, geom_rows = [], [], [], [], [], []
        kick_ids = list(dict.fromkeys(df["kick_id_raw"]))
        if limit is not None:
            kick_ids = kick_ids[:limit]

        for raw_id in kick_ids:
            kick = df[df["kick_id_raw"] == raw_id].sort_values("frame_num").reset_index(drop=True)
            pk_id = make_pk_id(self.slug, str(raw_id))
            ctx = self._kick_context(kick)
            fps = ctx.get("fps") or self.default_fps
            anchor = self._resolve_contact(kick)

            f0 = int(kick["frame_num"].min())
            kick["t_s"] = (kick["frame_num"] - f0) / fps
            if anchor["frame_idx"] is None:
                kick["t_ms"] = np.nan
            else:
                kick["t_ms"] = (kick["frame_num"] - anchor["frame_idx"]) / fps * 1000.0

            meta_rows.append(self._metadata_row(pk_id, raw_id, kick, ctx, fps, anchor))
            track_rows.extend(self._track_rows(pk_id, kick, fps))
            pose_rows.extend(self._pose_rows(pk_id, kick, ctx))
            event_rows.extend(self._event_rows(pk_id, kick, fps, anchor, f0))
            ball_rows.append(self._absent_row(pk_id, "ball"))
            geom_rows.append(self._absent_row(pk_id, "geometry"))

        return IngestResult(
            metadata=pd.DataFrame(meta_rows),
            tracks=pd.DataFrame(track_rows),
            poses=pd.DataFrame(pose_rows),
            ball=pd.DataFrame(ball_rows),
            geometry=pd.DataFrame(geom_rows),
            events=pd.DataFrame(event_rows),
            notes=notes,
        )

    # ------------------------------------------------------------ contact frame

    def _resolve_contact(self, kick: pd.DataFrame) -> dict[str, Any]:
        """Locate ball contact from the source's ``last_touch`` marker.

        Three real cases occur in these deposits and each is reported honestly:

        * exactly one marker -> contact known, confidence 1.0
        * several markers    -> the kick's rows span more than one upstream
          track (the upstream tracker lost identity and each fragment carries
          its own terminal marker). The clip is documented as ending at contact,
          so the latest marker is taken, confidence 0.5, and the kick is flagged
          fragmented. The alternative fragments are kept in ``tracks``/``poses``.
        * no marker          -> contact is unknown. Nothing is invented.
        """
        marked = kick[kick["last_touch"] == 1]
        n_tracks = int(kick["track_id"].nunique())
        if len(marked) == 1:
            return {
                "frame_idx": int(marked["frame_num"].iloc[0]),
                "confidence": 1.0,
                "method": "source_last_touch",
                "missing_reason": None,
                "n_markers": 1,
                "n_tracks": n_tracks,
                "anchor_track_id": str(marked["track_id"].iloc[0]),
            }
        if len(marked) > 1:
            last = marked.sort_values("frame_num").iloc[-1]
            return {
                "frame_idx": int(last["frame_num"]),
                "confidence": 0.5,
                "method": "source_last_touch_latest_of_multiple",
                "missing_reason": None,
                "n_markers": int(len(marked)),
                "n_tracks": n_tracks,
                "anchor_track_id": str(last["track_id"]),
            }
        return {
            "frame_idx": None,
            "confidence": None,
            "method": "source_last_touch",
            "missing_reason": "source_has_no_last_touch_marker",
            "n_markers": 0,
            "n_tracks": n_tracks,
            "anchor_track_id": None,
        }

    # ------------------------------------------------------------------- rows

    def _prov(self, derivation: str, provenance: str) -> dict[str, Any]:
        return {
            "processing_version": PROCESSING_VERSION,
            "derivation": derivation,
            "model_name": UPSTREAM_MODEL,
            "model_version": None,
            "provenance": provenance,
        }

    def _metadata_row(self, pk_id, raw_id, kick, ctx, fps, anchor) -> dict[str, Any]:
        first = kick.iloc[0]
        reasons = []
        if anchor["frame_idx"] is None:
            reasons.append("no_contact_frame")
        if anchor["n_markers"] > 1:
            reasons.append(f"fragmented_track({anchor['n_tracks']} tracks, {anchor['n_markers']} markers)")
        span_ms = len(kick) / fps * 1000.0
        return {
            "pk_id": pk_id,
            "pk_uid": make_pk_uid(pk_id),
            "source": self.slug,
            "source_identifier": str(raw_id),
            "source_dataset_doi": f"10.17632/{DATASET_ID}.{self.version}",
            "source_version": self.version,
            "dedup_key": make_dedup_key(deposit_family=self.deposit_family, source_identifier=str(raw_id)),
            "is_primary": True,
            "duplicate_of_pk_id": None,
            "duplicate_evidence": None,
            "media_kind": MEDIA_POSE_TABLE,
            "has_video": False,
            "video_relpath": None,
            "video_sha256": None,
            "fps": float(fps),
            "n_frames": int(len(kick)),
            "frame_width": None,
            "frame_height": None,
            "clip_duration_s": span_ms / 1000.0,
            "competition": self.competition,
            "gender": self.gender,
            "level": self.level,
            "season": self.season,
            "match_context": ctx.get("match_context"),
            "kicker_ref": ctx.get("kicker_ref"),
            "label_kick_direction": _norm_lcr(first.get("kick_direction")),
            "label_keeper_direction": None,
            "label_outcome": None,
            "label_goal": _norm_goal(first.get("goal")),
            "label_footedness": _norm_lcr(first.get("r_or_l")),
            "label_camera_direction": _norm_lcr(first.get("camera_direction")),
            "label_provenance": f"source_provided:{self.slug}",
            "license": LICENSE,
            "license_url": LICENSE_URL,
            "attribution": ATTRIBUTION,
            "redistribute_derived": True,
            "redistribute_video": False,
            "redistribution_note": "CC BY 4.0; no video in deposit",
            "ingested_at_utc": utc_now(),
            "qc_status": "pending",
            "qc_reasons": ";".join(reasons) or None,
            **self._prov("source_provided", f"mendeley:{DATASET_ID}/v{self.version}/{self.csv_name}"),
        }

    def _track_rows(self, pk_id, kick, fps) -> list[dict[str, Any]]:
        rows = []
        prov = self._prov("source_provided", f"mendeley:{DATASET_ID}/v{self.version}")
        cx = kick["bbox_cx"].to_numpy(dtype=float)
        cy = kick["bbox_cy"].to_numpy(dtype=float)
        t = kick["t_s"].to_numpy(dtype=float)
        vx, vy = _finite_diff(cx, cy, t)
        for i, r in kick.iterrows():
            rows.append(
                {
                    "pk_id": pk_id,
                    "source": self.slug,
                    "source_identifier": str(r["kick_id_raw"]),
                    "role": "kicker",
                    "track_id": str(r["track_id"]),
                    "frame_idx": int(r["frame_num"]),
                    "t_s": float(r["t_s"]),
                    "t_ms_rel_contact": _nn(r["t_ms"]),
                    "bbox_cx": _nn(r["bbox_cx"]),
                    "bbox_cy": _nn(r["bbox_cy"]),
                    "bbox_w": _nn(r["bbox_w"]),
                    "bbox_h": _nn(r["bbox_h"]),
                    "vx_px_s": _nn(vx[i]),
                    "vy_px_s": _nn(vy[i]),
                    "confidence": None,  # not published upstream
                    "is_missing": bool(pd.isna(r["bbox_cx"])),
                    "missing_reason": None if pd.notna(r["bbox_cx"]) else "source_bbox_absent",
                    **prov,
                }
            )
        # Keeper is not present in these deposits: one explicit absence row per
        # kick, so a consumer can distinguish "no keeper" from "keeper not looked for".
        rows.append(
            {
                "pk_id": pk_id,
                "source": self.slug,
                "source_identifier": str(kick["kick_id_raw"].iloc[0]),
                "role": "keeper",
                "track_id": None,
                "frame_idx": -1,
                "t_s": None,
                "t_ms_rel_contact": None,
                "bbox_cx": None,
                "bbox_cy": None,
                "bbox_w": None,
                "bbox_h": None,
                "vx_px_s": None,
                "vy_px_s": None,
                "confidence": None,
                "is_missing": True,
                "missing_reason": "source_provides_kicker_pose_only",
                **prov,
            }
        )
        return rows

    def _pose_rows(self, pk_id, kick, ctx) -> list[dict[str, Any]]:
        rows = []
        prov = self._prov("source_provided", f"mendeley:{DATASET_ID}/v{self.version}")
        # Camera direction flips the sign of the goal-ward axis. Normalising by
        # box height and by camera side makes L/R kicks comparable across clips.
        cam = _norm_lcr(kick.iloc[0].get("camera_direction"))
        flip = -1.0 if cam == "L" else 1.0
        for _, r in kick.iterrows():
            h = float(r["bbox_h"]) if pd.notna(r["bbox_h"]) and float(r["bbox_h"]) > 0 else np.nan
            for k in KEYPOINT_INDICES:
                x, y = r.get(f"kp_{k}_x"), r.get(f"kp_{k}_y")
                xc, yc = r.get(f"kp_{k}_xc"), r.get(f"kp_{k}_yc")
                missing = pd.isna(x) or pd.isna(y)
                rows.append(
                    {
                        "pk_id": pk_id,
                        "source": self.slug,
                        "role": "kicker",
                        "frame_idx": int(r["frame_num"]),
                        "t_s": float(r["t_s"]),
                        "t_ms_rel_contact": _nn(r["t_ms"]),
                        "kp_index": int(k),
                        "kp_name": KEYPOINT_NAMES[k],
                        "x": _nn(x),
                        "y": _nn(y),
                        "x_c": _nn(xc),
                        "y_c": _nn(yc),
                        "x_n": None if (missing or np.isnan(h)) else float(xc) / h * flip,
                        "y_n": None if (missing or np.isnan(h)) else float(yc) / h,
                        "confidence": None,  # upstream publishes coordinates only
                        "is_missing": bool(missing),
                        "missing_reason": "source_keypoint_absent" if missing else None,
                        **prov,
                    }
                )
        return rows

    def _event_rows(self, pk_id, kick, fps, anchor, f0) -> list[dict[str, Any]]:
        prov_src = self._prov("source_provided", f"mendeley:{DATASET_ID}/v{self.version}")
        prov_none = self._prov("pkcv_derived", "pkcv:events")
        rows = [
            {
                "pk_id": pk_id,
                "source": self.slug,
                "event_name": "ball_contact",
                "frame_idx": anchor["frame_idx"],
                "t_s": None if anchor["frame_idx"] is None else (anchor["frame_idx"] - f0) / fps,
                "confidence": anchor["confidence"],
                "method": anchor["method"],
                "is_missing": anchor["frame_idx"] is None,
                "missing_reason": anchor["missing_reason"],
                **prov_src,
            }
        ]
        for name, reason in (
            ("keeper_commit", "no_keeper_observations_in_source"),
            ("plant_foot", "requires_ball_position_or_video_absent_from_source"),
        ):
            rows.append(
                {
                    "pk_id": pk_id,
                    "source": self.slug,
                    "event_name": name,
                    "frame_idx": None,
                    "t_s": None,
                    "confidence": None,
                    "method": "not_attempted",
                    "is_missing": True,
                    "missing_reason": reason,
                    **prov_none,
                }
            )
        return rows

    def _absent_row(self, pk_id: str, kind: str) -> dict[str, Any]:
        base = {
            "pk_id": pk_id,
            "source": self.slug,
            "is_missing": True,
            "missing_reason": "source_provides_kicker_pose_only",
            **self._prov("pkcv_derived", "pkcv:absence-record"),
        }
        if kind == "ball":
            base["frame_idx"] = -1
        return base


# ---------------------------------------------------------------- helpers


def _nn(v):
    """None for NaN, float otherwise -- keeps parquet nulls honest."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return None
    return float(v)


def _norm_lcr(v):
    if v is None or pd.isna(v):
        return None
    s = str(v).strip().upper()
    return s if s in {"L", "C", "R"} else None


def _norm_goal(v):
    if v is None or pd.isna(v):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _finite_diff(x: np.ndarray, y: np.ndarray, t: np.ndarray):
    """Central-difference velocity; NaN wherever either neighbour is missing."""
    n = len(x)
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    if n < 2:
        return vx, vy
    for i in range(n):
        lo, hi = max(0, i - 1), min(n - 1, i + 1)
        dt = t[hi] - t[lo]
        if dt <= 0 or np.isnan(x[hi]) or np.isnan(x[lo]):
            continue
        vx[i] = (x[hi] - x[lo]) / dt
        vy[i] = (y[hi] - y[lo]) / dt
    return vx, vy


# ------------------------------------------------------------------ concrete


class MendeleyEPLv1(MendeleyPKAdapter):
    slug = "mendeley-epl-v1"
    title = "EPL Penalty Kick Data Recorded by a Computer Vision Model (Mendeley brx9bsxnpx v1)"
    deposit_family = "mendeley-brx9bsxnpx-epl"
    version = "1"
    csv_name = "kicker_pose_keypoints.csv"
    expected_kicks = 88
    competition = "English Premier League"
    gender = "men"
    level = "professional"
    season = "2023-24"
    default_fps = 25.0
    fps_provenance = "stated in deposit description (25 fps); not published per-kick"

    def _read_raw(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df = df.rename(columns={"kick_id": "kick_id_raw"})
        df["frame_num"] = pd.to_numeric(df["frame"], errors="coerce")
        return df

    def _kick_context(self, kick: pd.DataFrame) -> dict[str, Any]:
        return {"fps": self.default_fps, "match_context": "game", "kicker_ref": None}


class MendeleyWomenV2(MendeleyPKAdapter):
    slug = "mendeley-women-v2"
    title = "Women's Soccer Penalty Kick Data Captured by Computer Vision Models (Mendeley brx9bsxnpx v2)"
    deposit_family = "womens-collegiate-li-pifer"
    version = "2"
    csv_name = "penalty_pose_keypoints.csv"
    expected_kicks = 133  # deposit description states 133; the CSV contains 132
    competition = "US collegiate women's soccer"
    gender = "women"
    level = "collegiate"
    season = None
    default_fps = 25.0
    fps_provenance = "published per kick in the CSV `fps` column"

    def _read_raw(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df = df.rename(columns={"kick_id": "kick_id_raw", "kicker_track_id_main": "track_id"})
        df["frame_num"] = pd.to_numeric(df["frame"], errors="coerce")
        return df

    def _kick_context(self, kick: pd.DataFrame) -> dict[str, Any]:
        fps = float(kick["fps"].iloc[0]) if "fps" in kick and pd.notna(kick["fps"].iloc[0]) else self.default_fps
        ctxt = str(kick["game_or_training"].iloc[0]).strip().lower() if "game_or_training" in kick else None
        return {
            "fps": fps,
            "match_context": ctxt,
            "kicker_ref": f"{self.slug}#{kick['kicker'].iloc[0]}" if "kicker" in kick else None,
        }
