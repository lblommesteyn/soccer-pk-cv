"""Full-fidelity QC overlay drawn on the source footage.

Draws, per frame: the goal quad, the kicker and keeper boxes with pose
skeletons, the ball with its trajectory trail, and a timeline strip showing
where the frame sits relative to ball contact and the canonical snapshot
offsets.

This is deliberately drawn from the produced tables, not from in-memory
intermediates, so what you see is what was written.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pkcv.schemas import SNAPSHOT_OFFSETS_MS

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

COL = {
    "kicker": (60, 170, 255),   # orange
    "keeper": (120, 255, 120),  # green
    "other": (150, 150, 150),
    "ball": (60, 240, 255),     # yellow
    "goal": (255, 90, 235),     # magenta
    "text": (240, 240, 240),
}


def _quad(row) -> np.ndarray | None:
    if row is None or bool(row.get("is_missing", True)):
        return None
    try:
        return np.array(
            [
                [row["quad_tl_x"], row["quad_tl_y"]],
                [row["quad_tr_x"], row["quad_tr_y"]],
                [row["quad_br_x"], row["quad_br_y"]],
                [row["quad_bl_x"], row["quad_bl_y"]],
            ],
            dtype=float,
        )
    except (KeyError, TypeError):
        return None


def render(
    meta_row: pd.Series,
    video_path: str | Path,
    tracks: pd.DataFrame,
    poses: pd.DataFrame,
    ball: pd.DataFrame,
    geometry: pd.DataFrame,
    events: pd.DataFrame,
    out_path: str | Path,
    max_width: int = 1280,
) -> Path | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(meta_row.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 25.0)
    scale = min(1.0, max_width / max(w, 1))
    ow, oh = int(w * scale), int(h * scale)
    strip_h = 64

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh + strip_h)
    )
    if not writer.isOpened():
        cap.release()
        return None

    contact = events[events["event_name"] == "ball_contact"] if len(events) else events
    contact_frame = (
        int(contact["frame_idx"].iloc[0])
        if len(contact) and pd.notna(contact["frame_idx"].iloc[0])
        else None
    )
    commit = events[events["event_name"] == "keeper_commit"] if len(events) else events
    commit_frame = (
        int(commit["frame_idx"].iloc[0])
        if len(commit) and pd.notna(commit["frame_idx"].iloc[0])
        else None
    )

    geo_by = {int(r["frame_idx"]): r for _, r in geometry.iterrows() if pd.notna(r["frame_idx"])}
    trk_by: dict[int, list] = {}
    for _, r in tracks[tracks["frame_idx"] >= 0].iterrows():
        trk_by.setdefault(int(r["frame_idx"]), []).append(r)
    pose_by: dict[tuple[int, str], dict] = {}
    for _, r in poses[~poses["is_missing"].astype(bool)].iterrows():
        pose_by.setdefault((int(r["frame_idx"]), r["role"]), {})[int(r["kp_index"])] = (
            float(r["x"]), float(r["y"])
        )
    ball_by = {
        int(r["frame_idx"]): (float(r["x"]), float(r["y"]))
        for _, r in ball[(ball["frame_idx"] >= 0) & ball["x"].notna()].iterrows()
    }

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    snap_frames = {}
    if contact_frame is not None:
        for off in SNAPSHOT_OFFSETS_MS:
            f = int(round(contact_frame + off / 1000.0 * fps))
            if 0 <= f < n:
                snap_frames[f] = off

    trail: list[tuple[int, int]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img = cv2.resize(frame, (ow, oh))

        q = _quad(geo_by.get(idx))
        if q is not None:
            cv2.polylines(img, [(q * scale).astype(int)], True, COL["goal"], 3, cv2.LINE_AA)
            cv2.putText(img, "goal", tuple((q[0] * scale).astype(int) + np.array([4, -8])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL["goal"], 2, cv2.LINE_AA)

        for r in trk_by.get(idx, []):
            role = r["role"]
            if pd.isna(r.get("bbox_cx")):
                continue
            c = COL.get(role, COL["other"])
            x1 = int((r["bbox_cx"] - r["bbox_w"] / 2) * scale)
            y1 = int((r["bbox_cy"] - r["bbox_h"] / 2) * scale)
            x2 = int((r["bbox_cx"] + r["bbox_w"] / 2) * scale)
            y2 = int((r["bbox_cy"] + r["bbox_h"] / 2) * scale)
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 3)
            cv2.putText(img, role, (x1, max(14, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, c, 2, cv2.LINE_AA)
            pts = pose_by.get((idx, role))
            if pts:
                for a, b in SKELETON:
                    if a in pts and b in pts:
                        cv2.line(img,
                                 (int(pts[a][0] * scale), int(pts[a][1] * scale)),
                                 (int(pts[b][0] * scale), int(pts[b][1] * scale)),
                                 c, 2, cv2.LINE_AA)
                for x, y in pts.values():
                    cv2.circle(img, (int(x * scale), int(y * scale)), 3, (255, 255, 255), -1)

        if idx in ball_by:
            bx, by = ball_by[idx]
            trail.append((int(bx * scale), int(by * scale)))
        for i in range(1, len(trail)):
            cv2.line(img, trail[i - 1], trail[i], COL["ball"], 2, cv2.LINE_AA)
        if idx in ball_by and trail:
            cv2.circle(img, trail[-1], 10, COL["ball"], 2, cv2.LINE_AA)
            cv2.putText(img, "ball", (trail[-1][0] + 13, trail[-1][1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL["ball"], 1, cv2.LINE_AA)

        t_ms = None if contact_frame is None else (idx - contact_frame) / fps * 1000.0
        header = f"{meta_row['pk_id']}   frame {idx}"
        if t_ms is not None:
            header += f"   t={t_ms:+.0f} ms"
        cv2.putText(img, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL["text"], 2, cv2.LINE_AA)
        if contact_frame is not None and idx == contact_frame:
            cv2.putText(img, "BALL CONTACT", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (90, 200, 255), 3, cv2.LINE_AA)
            cv2.rectangle(img, (3, 3), (ow - 4, oh - 4), (90, 200, 255), 6)
        if commit_frame is not None and idx == commit_frame:
            cv2.putText(img, "KEEPER COMMITS", (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        COL["keeper"], 3, cv2.LINE_AA)
        if idx in snap_frames:
            cv2.putText(img, f"SNAPSHOT {snap_frames[idx]:+d} ms", (ow - 330, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 255, 160), 2, cv2.LINE_AA)

        # timeline strip
        strip = np.full((strip_h, ow, 3), 22, np.uint8)
        y0 = strip_h // 2
        cv2.line(strip, (20, y0), (ow - 20, y0), (70, 70, 70), 2)

        def sx(f):
            return int(20 + (ow - 40) * f / max(n - 1, 1))

        for f, off in snap_frames.items():
            cv2.line(strip, (sx(f), y0 - 12), (sx(f), y0 + 12), (150, 255, 160), 2)
            cv2.putText(strip, f"{off // 1000 if off % 1000 == 0 else off / 1000:g}s",
                        (sx(f) - 12, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 160), 1)
        if contact_frame is not None:
            cv2.line(strip, (sx(contact_frame), 6), (sx(contact_frame), strip_h - 6),
                     (90, 200, 255), 3)
        if commit_frame is not None:
            cv2.line(strip, (sx(commit_frame), 12), (sx(commit_frame), strip_h - 12),
                     COL["keeper"], 2)
        cv2.circle(strip, (sx(idx), y0), 7, (255, 255, 255), -1)
        writer.write(np.vstack([img, strip]))
        idx += 1

    cap.release()
    writer.release()
    return out_path
