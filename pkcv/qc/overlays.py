"""Visual QC artifacts.

Two products per kick:

``<pk_id>_snapshots.png``
    A contact sheet: the kicker's skeleton drawn at each canonical offset
    (-2000 .. 0 ms) above the feature timelines, with the contact anchor and
    every snapshot marked. This is the artifact to eyeball when asking "is the
    anchor in the right place and does the pose look like a penalty run-up?"

``<pk_id>_overlay.mp4``
    The reconstructed skeleton animated from the published parquet, drawn over
    the source video when there is one and over a black canvas otherwise. For
    the women's corpus, whose upstream deposit ships its own skeleton render,
    this doubles as an independent check that our ingest reproduces the
    depositor's geometry -- the two skeletons must land on top of each other.

Drawing from the *parquet*, never from an in-memory intermediate, is the point:
the overlay verifies what was actually published.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pkcv.schemas import SNAPSHOT_OFFSETS_MS

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
LEFT_KP = {5, 7, 9, 11, 13, 15}


def _frame_pose(poses: pd.DataFrame, frame_idx: int) -> dict[int, tuple[float, float]]:
    sub = poses[(poses["frame_idx"] == frame_idx) & (~poses["is_missing"].astype(bool))]
    return {
        int(r["kp_index"]): (float(r["x"]), float(r["y"]))
        for _, r in sub.iterrows()
        if pd.notna(r["x"]) and pd.notna(r["y"])
    }


def draw_skeleton(img, pts: dict[int, tuple[float, float]], scale=1.0, offset=(0, 0), thickness=2):
    import cv2

    ox, oy = offset
    for a, b in SKELETON:
        if a in pts and b in pts:
            pa = (int(pts[a][0] * scale + ox), int(pts[a][1] * scale + oy))
            pb = (int(pts[b][0] * scale + ox), int(pts[b][1] * scale + oy))
            colour = (255, 170, 60) if a in LEFT_KP and b in LEFT_KP else (60, 200, 255)
            cv2.line(img, pa, pb, colour, thickness, cv2.LINE_AA)
    for k, (x, y) in pts.items():
        p = (int(x * scale + ox), int(y * scale + oy))
        cv2.circle(img, p, max(2, thickness), (255, 255, 255) if k in LEFT_KP else (200, 200, 200), -1)
    return img


def snapshot_sheet(
    meta_row: pd.Series,
    poses: pd.DataFrame,
    frames: pd.DataFrame,
    snapshots: pd.DataFrame,
    out_path: str | Path,
) -> Path | None:
    """Contact sheet of skeletons at the canonical offsets plus feature timelines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kp = poses[poses["role"] == "kicker"]
    if not len(kp):
        return None

    offsets = list(SNAPSHOT_OFFSETS_MS)
    fig = plt.figure(figsize=(3.0 * len(offsets), 8.5), facecolor="#0d0f13")
    gs = fig.add_gridspec(3, len(offsets), height_ratios=[2.2, 1, 1], hspace=0.32, wspace=0.12)

    for i, off in enumerate(offsets):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#0d0f13")
        snap = snapshots[snapshots["offset_ms"] == off]
        available = len(snap) and bool(snap["snapshot_available"].iloc[0])
        if available:
            # Draw the nearest *observed* frame: an interpolated skeleton is not
            # a thing that was seen, and drawing it would misrepresent the data.
            t = frames["t_ms_rel_contact"].to_numpy(float)
            j = int(np.argmin(np.abs(t - off)))
            fidx = int(frames["frame_idx"].iloc[j])
            pts = _frame_pose(kp, fidx)
            if pts:
                xs = [p[0] for p in pts.values()]
                ys = [p[1] for p in pts.values()]
                for a, b in SKELETON:
                    if a in pts and b in pts:
                        ax.plot(
                            [pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                            color="#ffaa3c" if (a in LEFT_KP and b in LEFT_KP) else "#3cc8ff",
                            lw=2.0,
                        )
                ax.scatter(xs, ys, s=9, c="white", zorder=3)
                pad = 0.15 * max(max(xs) - min(xs), max(ys) - min(ys), 1)
                ax.set_xlim(min(xs) - pad, max(xs) + pad)
                ax.set_ylim(max(ys) + pad, min(ys) - pad)  # image y grows downward
            method = str(snap["snapshot_method"].iloc[0])
            title, colour = f"{off:+d} ms\n{method}", "#9fe8a0"
        else:
            reason = str(snap["unavailable_reason"].iloc[0]) if len(snap) else "no_snapshot_row"
            ax.text(0.5, 0.5, reason.replace("_", "\n"), ha="center", va="center",
                    color="#ff7b7b", fontsize=9, transform=ax.transAxes)
            title, colour = f"{off:+d} ms\nunavailable", "#ff7b7b"
        ax.set_title(title, color=colour, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#2a2f38")

    for row, (col, label) in enumerate(
        [
            ("hip_shoulder_angle_deg", "hip-shoulder separation (deg)"),
            ("pelvis_orientation_deg", "pelvis orientation (deg)"),
        ],
        start=1,
    ):
        ax = fig.add_subplot(gs[row, :])
        ax.set_facecolor("#0d0f13")
        if col in frames:
            ax.plot(frames["t_ms_rel_contact"], frames[col], color="#3cc8ff", lw=1.6)
        for off in offsets:
            snap = snapshots[snapshots["offset_ms"] == off]
            ok = len(snap) and bool(snap["snapshot_available"].iloc[0])
            ax.axvline(off, color="#9fe8a0" if ok else "#ff7b7b", ls="--", lw=0.9, alpha=0.8)
        ax.axvline(0, color="#ffffff", lw=1.6, alpha=0.9)
        ax.set_ylabel(label, color="#c9d1d9", fontsize=9)
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#2a2f38")
        if row == 2:
            ax.set_xlabel("time relative to ball contact (ms)", color="#c9d1d9", fontsize=9)

    fig.suptitle(
        f"{meta_row['pk_id']}   |   direction={meta_row.get('label_kick_direction')}   "
        f"goal={meta_row.get('label_goal')}   foot={meta_row.get('label_footedness')}   "
        f"cam={meta_row.get('label_camera_direction')}   fps={meta_row.get('fps')}",
        color="#e6edf3",
        fontsize=12,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out_path


def overlay_video(
    meta_row: pd.Series,
    poses: pd.DataFrame,
    frames: pd.DataFrame,
    ball: pd.DataFrame,
    out_path: str | Path,
    background_video: str | Path | None = None,
    canvas: tuple[int, int] = (1280, 720),
) -> Path | None:
    """Animate the published skeleton, ball track and temporal anchors."""
    import cv2

    kp = poses[poses["role"] == "kicker"]
    if not len(kp):
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fps = float(meta_row.get("fps") or 25.0)
    cap = cv2.VideoCapture(str(background_video)) if background_video else None
    if cap is not None and cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        cap = None
        pts_all = kp[~kp["is_missing"].astype(bool)]
        w = int(min(canvas[0], max(pts_all["x"].max() * 1.1, 640))) if len(pts_all) else canvas[0]
        h = int(min(canvas[1], max(pts_all["y"].max() * 1.1, 360))) if len(pts_all) else canvas[1]

    scale = min(1.0, 1280.0 / max(w, 1))
    ow, oh = int(w * scale), int(h * scale)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    if not writer.isOpened():
        return None

    ball_by_frame = {}
    if len(ball):
        for _, r in ball[ball["frame_idx"] >= 0].iterrows():
            if pd.notna(r.get("x")):
                ball_by_frame[int(r["frame_idx"])] = (float(r["x"]), float(r["y"]))
    ball_trail: list[tuple[int, int]] = []

    tmap = dict(zip(frames["frame_idx"].astype(int), frames["t_ms_rel_contact"].astype(float), strict=False))
    snap_frames = {}
    for off in SNAPSHOT_OFFSETS_MS:
        if tmap:
            nearest = min(tmap, key=lambda f: abs(tmap[f] - off))
            if abs(tmap[nearest] - off) <= 1000.0 / fps:
                snap_frames[nearest] = off

    for fidx in sorted(kp["frame_idx"].unique()):
        fidx = int(fidx)
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, img = cap.read()
            if not ok:
                img = np.zeros((h, w, 3), np.uint8)
        else:
            img = np.zeros((h, w, 3), np.uint8)
        img = cv2.resize(img, (ow, oh))

        draw_skeleton(img, _frame_pose(kp, fidx), scale=scale, thickness=2)

        if fidx in ball_by_frame:
            bx, by = ball_by_frame[fidx]
            ball_trail.append((int(bx * scale), int(by * scale)))
        for i in range(1, len(ball_trail)):
            cv2.line(img, ball_trail[i - 1], ball_trail[i], (80, 255, 120), 2, cv2.LINE_AA)
        if ball_trail:
            cv2.circle(img, ball_trail[-1], 6, (80, 255, 120), -1)

        t_ms = tmap.get(fidx)
        label = f"{meta_row['pk_id']}  frame {fidx}"
        if t_ms is not None:
            label += f"  t={t_ms:+.0f} ms"
        cv2.putText(img, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        if fidx in snap_frames:
            cv2.putText(img, f"SNAPSHOT {snap_frames[fidx]:+d} ms", (12, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 160), 2, cv2.LINE_AA)
            cv2.rectangle(img, (2, 2), (ow - 3, oh - 3), (150, 255, 160), 3)
        if t_ms is not None and abs(t_ms) < 1e-6:
            cv2.putText(img, "BALL CONTACT", (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (90, 200, 255), 2, cv2.LINE_AA)
            cv2.rectangle(img, (2, 2), (ow - 3, oh - 3), (90, 200, 255), 5)
        writer.write(img)

    writer.release()
    if cap is not None:
        cap.release()
    return out_path
