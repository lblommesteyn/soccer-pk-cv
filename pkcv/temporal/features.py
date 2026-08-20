"""Per-frame scalar features derived from the kicker's 2D pose.

Everything here is expressed in the *normalised* frame ``(x_n, y_n)``: box-centred
pixels divided by the bounding-box height, with the sign of the horizontal axis
flipped for left-side camera angles. That makes a kick filmed from the left
directly comparable with one filmed from the right, and removes the scale
difference between a tight broadcast crop and a wide training camera. Without
it, "kicker leans right" would mean opposite things in different clips.

The feature set targets the research question -- how early the kick direction is
readable -- so it is built from the cues the biomechanics literature associates
with kick direction: pelvis orientation, hip-shoulder separation, torso lean,
and the plant/kicking foot geometry.

Every feature returns ``NaN`` when the keypoints it needs are missing. There is
no imputation at this layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16
L_KNEE, R_KNEE = 13, 14


def _pivot(pose_kick: pd.DataFrame) -> pd.DataFrame:
    """Long pose rows -> one row per frame, columns ``kp{idx}_xn`` / ``kp{idx}_yn``."""
    wide = pose_kick.pivot_table(
        index="frame_idx", columns="kp_index", values=["x_n", "y_n"], aggfunc="first"
    )
    wide.columns = [f"kp{int(k)}_{'xn' if v == 'x_n' else 'yn'}" for v, k in wide.columns]
    return wide.sort_index()


def _angle_deg(dx: pd.Series, dy: pd.Series) -> pd.Series:
    """Direction of a vector in degrees, measured from the +x axis.

    Image ``y`` grows downward, so the sign is flipped to give a conventional
    orientation where positive is counter-clockwise on screen.
    """
    return np.degrees(np.arctan2(-dy, dx))


def compute(pose_kick: pd.DataFrame, footedness: str | None, fps: float) -> pd.DataFrame:
    """Return a per-frame feature frame indexed by ``frame_idx``."""
    if not len(pose_kick):
        return pd.DataFrame()
    w = _pivot(pose_kick)

    def kp(i, ax):
        col = f"kp{i}_{ax}"
        return w[col] if col in w else pd.Series(np.nan, index=w.index)

    out = pd.DataFrame(index=w.index)

    shoulder_dx = kp(R_SHOULDER, "xn") - kp(L_SHOULDER, "xn")
    shoulder_dy = kp(R_SHOULDER, "yn") - kp(L_SHOULDER, "yn")
    hip_dx = kp(R_HIP, "xn") - kp(L_HIP, "xn")
    hip_dy = kp(R_HIP, "yn") - kp(L_HIP, "yn")

    shoulder_ang = _angle_deg(shoulder_dx, shoulder_dy)
    out["pelvis_orientation_deg"] = _angle_deg(hip_dx, hip_dy)
    # Hip-shoulder separation, wrapped to (-180, 180] so a twist through the
    # 180-degree boundary does not read as a 350-degree jump.
    sep = shoulder_ang - out["pelvis_orientation_deg"]
    out["hip_shoulder_angle_deg"] = (sep + 180.0) % 360.0 - 180.0

    shoulder_mid_x = (kp(R_SHOULDER, "xn") + kp(L_SHOULDER, "xn")) / 2
    shoulder_mid_y = (kp(R_SHOULDER, "yn") + kp(L_SHOULDER, "yn")) / 2
    hip_mid_x = (kp(R_HIP, "xn") + kp(L_HIP, "xn")) / 2
    hip_mid_y = (kp(R_HIP, "yn") + kp(L_HIP, "yn")) / 2
    # Lean of the torso away from vertical; positive means leaning toward +x.
    out["torso_lean_deg"] = np.degrees(
        np.arctan2(shoulder_mid_x - hip_mid_x, -(shoulder_mid_y - hip_mid_y))
    )

    out["ankle_separation_n"] = np.hypot(
        kp(R_ANKLE, "xn") - kp(L_ANKLE, "xn"), kp(R_ANKLE, "yn") - kp(L_ANKLE, "yn")
    )

    # A right-footed kicker kicks with the right ankle and plants the left.
    # Unknown footedness leaves both columns NaN rather than assuming right.
    foot = (footedness or "").upper()
    if foot == "R":
        kick_i, plant_i = R_ANKLE, L_ANKLE
    elif foot == "L":
        kick_i, plant_i = L_ANKLE, R_ANKLE
    else:
        kick_i = plant_i = None

    if kick_i is not None:
        out["kick_ankle_x_n"] = kp(kick_i, "xn")
        out["plant_ankle_x_n"] = kp(plant_i, "xn")
        dt = 1.0 / fps if fps else np.nan
        out["kick_ankle_vx_n"] = out["kick_ankle_x_n"].diff() / dt
    else:
        out["kick_ankle_x_n"] = np.nan
        out["plant_ankle_x_n"] = np.nan
        out["kick_ankle_vx_n"] = np.nan

    visible = pose_kick[~pose_kick["is_missing"].astype(bool)].groupby("frame_idx").size()
    out["pose_n_visible_kp"] = visible.reindex(out.index).fillna(0).astype(int)
    out["pose_available"] = out["pose_n_visible_kp"] > 0
    return out
