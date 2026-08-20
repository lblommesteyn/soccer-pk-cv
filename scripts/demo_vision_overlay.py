"""Render everything the vision stage sees on a clip, for eyeballing.

Unlike the QC overlay, which draws only what was published, this draws the raw
detector output too: every person box, the ball candidates, the role assignment
and the goal-geometry attempt. It is the tool for asking "why did the pipeline
say it could not find X?" on a clip that has X in it.

    python scripts/demo_vision_overlay.py --video path/to/clip.mp4 --fps 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pkcv.vision.runner import ClipProcessor, VisionConfig  # noqa: E402

WHITE = (200, 200, 200)
ORANGE = (60, 170, 255)
GREEN = (120, 255, 120)
YELLOW = (60, 240, 255)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--strip", default=None, help="also write an N-frame contact strip here")
    ap.add_argument("--strip-n", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    video = Path(args.video)
    cap = cv2.VideoCapture(str(video))
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    proc = ClipProcessor(VisionConfig(imgsz=args.imgsz, device=args.device))
    persons, balls, failures = proc._detect_and_track(str(video))
    geom = proc._geometry_rows("demo", "demo", str(video))
    roles = proc._assign_roles(persons, geom)
    role_of = dict(zip(roles["track_id"], roles["role"], strict=False))
    print(f"persons: {len(persons)} rows / {persons['track_id'].nunique()} tracks")
    print(f"balls:   {len(balls)} rows / {balls['track_id'].nunique() if len(balls) else 0} tracks")
    print("roles:", roles["role"].value_counts().to_dict() if len(roles) else "none")
    print("failures:", failures)

    g = geom.iloc[0]
    print("geometry:", {k: g[k] for k in ("frame_idx", "goal_width_px", "confidence", "missing_reason") if k in g})
    if len(roles):
        print(roles[roles.role != "other"].to_string(index=False))

    best_ball = balls.groupby("track_id").size().idxmax() if len(balls) else None
    trail: list[tuple[int, int]] = []

    out_path = Path(args.out or video.with_name(video.stem + "_vision.mp4"))
    scale = min(1.0, 1280 / w)
    ow, oh = int(w * scale), int(h * scale)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))

    cap = cv2.VideoCapture(str(video))
    idx = 0
    kept = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img = cv2.resize(frame, (ow, oh))

        for _, r in persons[persons["frame_idx"] == idx].iterrows():
            role = role_of.get(int(r["track_id"]), "other")
            colour = {"kicker": ORANGE, "keeper": GREEN}.get(role, WHITE)
            thick = 3 if role in ("kicker", "keeper") else 1
            x1 = int((r["cx"] - r["w"] / 2) * scale)
            y1 = int((r["cy"] - r["h"] / 2) * scale)
            x2 = int((r["cx"] + r["w"] / 2) * scale)
            y2 = int((r["cy"] + r["h"] / 2) * scale)
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick)
            if role in ("kicker", "keeper"):
                cv2.putText(img, role, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, colour, 2, cv2.LINE_AA)

        b = balls[(balls["frame_idx"] == idx) & (balls["track_id"] == best_ball)] if len(balls) else []
        if len(b):
            r = b.iloc[0]
            p = (int(r["cx"] * scale), int(r["cy"] * scale))
            trail.append(p)
            cv2.circle(img, p, 9, YELLOW, 2)
            cv2.putText(img, f"ball {r['conf']:.2f}", (p[0] + 12, p[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)
        for i in range(1, len(trail)):
            cv2.line(img, trail[i - 1], trail[i], YELLOW, 2, cv2.LINE_AA)

        if not bool(g["is_missing"]):
            lx, rx = int(g["post_left_x"] * scale), int(g["post_right_x"] * scale)
            cy_ = int(g["crossbar_y"] * scale)
            by = int(max(g["post_left_y"], g["post_right_y"]) * scale)
            cv2.rectangle(img, (lx, cy_), (rx, by), (255, 120, 255), 2)
            cv2.putText(img, "goal", (lx, cy_ - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 120, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, f"goal: {g['missing_reason']}", (12, oh - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 255), 1, cv2.LINE_AA)

        cv2.putText(img, f"frame {idx}  t={idx / fps:.2f}s  persons={len(persons[persons['frame_idx'] == idx])}",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        writer.write(img)
        kept.append(img)
        idx += 1
    cap.release()
    writer.release()
    print("wrote", out_path)

    if args.strip and kept:
        sel = np.linspace(0, len(kept) - 1, args.strip_n).astype(int)
        tiles = [cv2.resize(kept[i], (480, 270)) for i in sel]
        half = len(tiles) // 2
        cv2.imwrite(args.strip, np.vstack([np.hstack(tiles[:half]), np.hstack(tiles[half:])]))
        print("wrote", args.strip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
