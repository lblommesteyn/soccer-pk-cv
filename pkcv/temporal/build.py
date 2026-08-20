"""Contact-anchored temporal representation.

Two tables come out of this module:

``temporal_frames``
    Every observed frame of every kick, with time expressed relative to
    estimated ball contact (``t_ms_rel_contact``; contact is 0, the run-up is
    negative). This is the lossless view.

``temporal_snapshots``
    The experiment table: exactly one row per (kick, offset) for the canonical
    offsets -2000 .. 0 ms, whether or not the observation exists. A missing
    snapshot is a present row with ``snapshot_available = False`` and a reason,
    never an absent row -- otherwise "how early can direction be predicted?"
    silently becomes "how early, among the clips that happened to be long
    enough?", which is a different and much easier question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pkcv.schemas import PROCESSING_VERSION, SNAPSHOT_OFFSETS_MS
from pkcv.temporal import features as feat

#: Columns carried through interpolation.
_NUMERIC = [
    "kicker_cx",
    "kicker_cy",
    "kicker_h",
    "kicker_vx",
    "kicker_vy",
    "kicker_speed",
    "hip_shoulder_angle_deg",
    "pelvis_orientation_deg",
    "ankle_separation_n",
    "plant_ankle_x_n",
    "kick_ankle_x_n",
    "kick_ankle_vx_n",
    "torso_lean_deg",
    "keeper_cx",
    "keeper_cy",
    "keeper_vx",
    "keeper_vy",
    "ball_x",
    "ball_y",
    "ball_vx",
    "ball_vy",
    "kicker_ball_dist_n",
    "keeper_goal_offset_n",
]


def _prov(derivation: str, provenance: str) -> dict:
    return {
        "processing_version": PROCESSING_VERSION,
        "derivation": derivation,
        "model_name": None,
        "model_version": None,
        "provenance": provenance,
    }


def build_frames(
    meta_row: pd.Series,
    tracks: pd.DataFrame,
    poses: pd.DataFrame,
    ball: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fuse one kick's tracks, poses and ball into a per-frame observation table."""
    pk_id = meta_row["pk_id"]
    fps = float(meta_row["fps"]) if pd.notna(meta_row.get("fps")) else np.nan

    if "role" not in getattr(tracks, "columns", []):
        return pd.DataFrame()
    kicker = tracks[(tracks["role"] == "kicker") & (tracks["frame_idx"] >= 0)].copy()
    if not len(kicker):
        return pd.DataFrame()

    # A kick whose rows span several upstream tracks is several *people*, not one
    # person seen twice: the upstream tracker lost identity mid-clip. Keep only
    # the track that contains the contact anchor, so the timeline describes a
    # single body. The discarded fragments stay in tracks.parquet.
    if kicker["track_id"].nunique() > 1:
        at_contact = kicker.loc[kicker["t_ms_rel_contact"].abs() < 1e-6, "track_id"]
        chosen = (
            at_contact.iloc[0]
            if len(at_contact)
            else kicker["track_id"].value_counts().idxmax()
        )
        kicker = kicker[kicker["track_id"] == chosen]

    kicker = kicker.sort_values("frame_idx").drop_duplicates("frame_idx", keep="last")
    kicker = kicker.set_index("frame_idx")

    df = pd.DataFrame(index=kicker.index)
    df["pk_id"] = pk_id
    df["source"] = meta_row["source"]
    df["t_s"] = kicker["t_s"]
    df["t_ms_rel_contact"] = kicker["t_ms_rel_contact"]
    df["kicker_cx"] = kicker["bbox_cx"]
    df["kicker_cy"] = kicker["bbox_cy"]
    df["kicker_h"] = kicker["bbox_h"]
    df["kicker_vx"] = kicker["vx_px_s"]
    df["kicker_vy"] = kicker["vy_px_s"]
    df["kicker_speed"] = np.hypot(kicker["vx_px_s"], kicker["vy_px_s"])
    df["kicker_available"] = ~kicker["is_missing"].astype(bool)

    # A clip can yield tracks but no pose at all (nobody matched a pose box).
    # That is an empty frame with no columns, not a frame with zero rows.
    pose_kick = poses[poses["role"] == "kicker"] if ("role" in getattr(poses, "columns", [])) else poses.iloc[0:0]
    if len(pose_kick):
        pf = feat.compute(pose_kick, meta_row.get("label_footedness"), fps)
        df = df.join(pf, how="left")
    for col in (
        "hip_shoulder_angle_deg",
        "pelvis_orientation_deg",
        "ankle_separation_n",
        "plant_ankle_x_n",
        "kick_ankle_x_n",
        "kick_ankle_vx_n",
        "torso_lean_deg",
    ):
        if col not in df:
            df[col] = np.nan
    if "pose_n_visible_kp" not in df:
        df["pose_n_visible_kp"] = 0
    if "pose_available" not in df:
        df["pose_available"] = False
    df["pose_available"] = df["pose_available"].astype("boolean").fillna(False).astype(bool)
    df["pose_n_visible_kp"] = pd.to_numeric(df["pose_n_visible_kp"], errors="coerce").fillna(0).astype(int)

    keeper = (
        tracks[(tracks["role"] == "keeper") & (tracks["frame_idx"] >= 0)]
        if "role" in getattr(tracks, "columns", [])
        else tracks.iloc[0:0]
    )
    if len(keeper):
        k = keeper.sort_values("frame_idx").drop_duplicates("frame_idx", keep="last").set_index("frame_idx")
        df["keeper_cx"] = k["bbox_cx"]
        df["keeper_cy"] = k["bbox_cy"]
        df["keeper_vx"] = k["vx_px_s"]
        df["keeper_vy"] = k["vy_px_s"]
        df["keeper_available"] = ~k["is_missing"].astype(bool)
    else:
        df["keeper_cx"] = df["keeper_cy"] = df["keeper_vx"] = df["keeper_vy"] = np.nan
        df["keeper_available"] = False
    df["keeper_available"] = df["keeper_available"].astype("boolean").fillna(False).astype(bool)

    if (
        ball is not None
        and len(ball)
        and "frame_idx" in getattr(ball, "columns", [])
        and (ball["frame_idx"] >= 0).any()
    ):
        b = (
            ball[ball["frame_idx"] >= 0]
            .sort_values("frame_idx")
            .drop_duplicates("frame_idx", keep="last")
            .set_index("frame_idx")
        )
        df["ball_x"] = b["x"]
        df["ball_y"] = b["y"]
        df["ball_vx"] = b["vx_px_s"]
        df["ball_vy"] = b["vy_px_s"]
        df["ball_available"] = ~b["is_missing"].astype(bool)
    else:
        df["ball_x"] = df["ball_y"] = df["ball_vx"] = df["ball_vy"] = np.nan
        df["ball_available"] = False
    df["ball_available"] = df["ball_available"].astype("boolean").fillna(False).astype(bool)

    df["kicker_ball_dist_n"] = np.hypot(df["ball_x"] - df["kicker_cx"], df["ball_y"] - df["kicker_cy"]) / df[
        "kicker_h"
    ]
    df["keeper_goal_offset_n"] = np.nan  # needs goal geometry; absent from pose-table sources
    df["is_observed_frame"] = True
    df = df.reset_index().rename(columns={"index": "frame_idx"})
    for k, v in _prov("pkcv_derived", f"pkcv:temporal/build@{PROCESSING_VERSION}").items():
        df[k] = v
    # A kick with no contact anchor has no meaningful timeline; it is kept out of
    # the temporal tables rather than anchored on a guess.
    return df[df["t_ms_rel_contact"].notna()]


def build_snapshots(
    frames: pd.DataFrame,
    meta_row: pd.Series,
    offsets: tuple[int, ...] = SNAPSHOT_OFFSETS_MS,
    max_interp_gap_ms: float = 120.0,
) -> pd.DataFrame:
    """Sample ``frames`` at each canonical offset.

    ``max_interp_gap_ms`` bounds how far apart the two bracketing observations
    may be before interpolation is refused. The default of 120 ms is three
    frames at 25 fps: beyond that the kicker's limb positions are no longer
    linear in time and an interpolated pose would be fiction.
    """
    pk_id = meta_row["pk_id"]
    rows = []
    base_cols = [c for c in frames.columns if c not in ("pk_id", "source")]

    if not len(frames):
        for off in offsets:
            rows.append(_unavailable(pk_id, meta_row, off, "no_anchored_frames"))
        return pd.DataFrame(rows)

    f = frames.sort_values("t_ms_rel_contact").reset_index(drop=True)
    t = f["t_ms_rel_contact"].to_numpy(dtype=float)
    frame_step_ms = 1000.0 / float(meta_row["fps"]) if pd.notna(meta_row.get("fps")) else 40.0
    exact_tol = frame_step_ms / 2.0

    for off in offsets:
        if off < t.min() - exact_tol:
            rows.append(_unavailable(pk_id, meta_row, off, "before_clip_start"))
            continue
        if off > t.max() + exact_tol:
            rows.append(_unavailable(pk_id, meta_row, off, "after_clip_end"))
            continue

        i = int(np.argmin(np.abs(t - off)))
        gap = float(abs(t[i] - off))
        if gap <= exact_tol:
            row = f.loc[i, base_cols].to_dict()
            method, avail, reason = "exact", True, None
        else:
            hi = int(np.searchsorted(t, off))
            lo = hi - 1
            if lo < 0 or hi >= len(t) or (t[hi] - t[lo]) > max_interp_gap_ms:
                rows.append(_unavailable(pk_id, meta_row, off, "gap_exceeds_interpolation_limit", gap))
                continue
            w = (off - t[lo]) / (t[hi] - t[lo])
            row = f.loc[lo, base_cols].to_dict()
            for c in _NUMERIC:
                if c in f:
                    a, b = f.at[lo, c], f.at[hi, c]
                    row[c] = np.nan if (pd.isna(a) or pd.isna(b)) else float(a) * (1 - w) + float(b) * w
            for c in ("kicker_available", "pose_available", "keeper_available", "ball_available"):
                if c in f:
                    row[c] = bool(f.at[lo, c]) and bool(f.at[hi, c])
            row["frame_idx"] = int(f.at[lo, "frame_idx"])
            row["t_s"] = float(f.at[lo, "t_s"]) + w * (float(f.at[hi, "t_s"]) - float(f.at[lo, "t_s"]))
            method, avail, reason = "interp", True, None

        row.update(
            pk_id=pk_id,
            source=meta_row["source"],
            t_ms_rel_contact=float(off),
            offset_ms=int(off),
            snapshot_available=avail,
            snapshot_method=method,
            gap_ms=gap,
            unavailable_reason=reason,
            is_observed_frame=(method == "exact"),
        )
        row.update(
            _prov(
                "pkcv_derived" if method == "exact" else "pkcv_interpolated",
                f"pkcv:temporal/snapshots@{PROCESSING_VERSION}",
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _unavailable(pk_id, meta_row, offset, reason, gap=None) -> dict:
    row = {
        "pk_id": pk_id,
        "source": meta_row["source"],
        "frame_idx": -1,
        "t_s": None,
        "t_ms_rel_contact": float(offset),
        "offset_ms": int(offset),
        "snapshot_available": False,
        "snapshot_method": "unavailable",
        "gap_ms": gap,
        "unavailable_reason": reason,
        "kicker_available": False,
        "pose_available": False,
        "keeper_available": False,
        "ball_available": False,
        "is_observed_frame": False,
        "pose_n_visible_kp": 0,
    }
    row.update(_prov("pkcv_derived", f"pkcv:temporal/snapshots@{PROCESSING_VERSION}"))
    return row
