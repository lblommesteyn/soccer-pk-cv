"""Dump the ball track and the contact decision for one clip.

Contact is the anchor the whole dataset hangs on, so when it fails this shows
exactly why: which ball track was chosen, how long it survives, and what its
speed profile looks like against the threshold.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pkcv.vision.runner import ClipProcessor, VisionConfig, _central_diff  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    video = ROOT / args.video if not Path(args.video).is_absolute() else Path(args.video)
    cache = Path(args.cache) if args.cache else video.with_suffix(".det.pkl")

    proc = ClipProcessor(VisionConfig(imgsz=args.imgsz))
    if cache.exists():
        persons, balls, geometry = pickle.load(open(cache, "rb"))
        print(f"loaded detections from {cache.name}")
    else:
        persons, balls, _ = proc._detect_and_track(str(video))
        geometry = proc._geometry_rows("dbg", "dbg", str(video))
        pickle.dump((persons, balls, geometry), open(cache, "wb"))

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"frames={n} fps={fps:.2f}")
    print(f"ball rows={len(balls)} tracks={balls['track_id'].nunique() if len(balls) else 0}")
    if len(balls):
        print("\nball tracks by length:")
        for tid, g in sorted(balls.groupby("track_id"), key=lambda kv: -len(kv[1]))[:8]:
            print(f"  track {tid}: {len(g)} frames, "
                  f"f{int(g.frame_idx.min())}-{int(g.frame_idx.max())}, "
                  f"conf {g.conf.mean():.2f}, "
                  f"x {g.cx.min():.0f}-{g.cx.max():.0f} y {g.cy.min():.0f}-{g.cy.max():.0f}")

    gw = float(np.nanmedian(geometry.loc[~geometry.is_missing.astype(bool), "goal_width_px"]))
    print(f"\nmedian goal width = {gw:.0f}px")

    if len(balls):
        tid = balls.groupby("track_id").size().idxmax()
        b = balls[balls.track_id == tid].sort_values("frame_idx")
        t = b.frame_idx.to_numpy(float) / fps
        vx, vy = _central_diff(b.cx.to_numpy(float), b.cy.to_numpy(float), t)
        speed = np.hypot(vx, vy) / gw
        base = float(np.nanpercentile(speed, 25))
        thresh = max(base * 6.0, proc.cfg.contact_min_speed_gw_s)
        print(f"chosen track {tid}: baseline(p25)={base:.4f} gw/s  threshold={thresh:.3f} gw/s")
        print(f"speed max={np.nanmax(speed):.3f}  p90={np.nanpercentile(speed, 90):.3f}")
        print("\nframe  x      y      speed(gw/s)")
        for i in range(len(b)):
            f = int(b.frame_idx.iloc[i])
            if i % max(1, len(b) // 40) == 0 or speed[i] > thresh:
                flag = "  <== over threshold" if speed[i] > thresh else ""
                print(f"{f:5d} {b.cx.iloc[i]:7.0f} {b.cy.iloc[i]:7.0f} {speed[i]:10.4f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
