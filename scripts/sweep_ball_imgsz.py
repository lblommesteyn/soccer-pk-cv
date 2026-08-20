"""Measure ball-detection recall against inference resolution.

The ball is the smallest object the pipeline has to find, and contact anchoring
depends entirely on it, so this is the parameter that decides how many penalties
survive. Prints detections per clip at each resolution and confidence floor.

    python scripts/sweep_ball_imgsz.py clipA.mp4 clipB.mp4 --imgsz 960 1280 1920
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pkcv.vision.runner import COCO_SPORTS_BALL, ClipProcessor, VisionConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[960, 1280, 1920])
    ap.add_argument("--ball-conf", type=float, nargs="+", default=[0.15, 0.05])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print(f"{'clip':<44} {'imgsz':>6} {'conf':>6} {'ball_frames':>12} {'longest_track':>14}")
    for v in args.videos:
        path = ROOT / v if not Path(v).is_absolute() else Path(v)
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        for size in args.imgsz:
            for conf in args.ball_conf:
                proc = ClipProcessor(
                    VisionConfig(imgsz=size, device=args.device, ball_conf=conf)
                )
                _, balls, _ = proc._detect_and_track(str(path))
                longest = int(balls.groupby("track_id").size().max()) if len(balls) else 0
                print(f"{path.stem[:44]:<44} {size:>6} {conf:>6.2f} "
                      f"{len(balls):>7}/{n:<4} {longest:>14}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
