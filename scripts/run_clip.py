"""Run the full vision pipeline on one clip and render the QC overlay.

    python scripts/run_clip.py data/raw/commons/clips/gradel.mp4 --pk-id commons:gradel

Standalone on purpose: it is how a single clip gets inspected end to end without
touching the corpus tables.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pkcv.qc import video_overlay  # noqa: E402
from pkcv.vision import ClipProcessor, VisionConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--pk-id", default=None)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_absolute():
        video = ROOT / video
    pk_id = args.pk_id or f"clip:{video.stem}"

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"{video.name}: {n} frames @ {fps:.2f} fps", flush=True)

    proc = ClipProcessor(VisionConfig(imgsz=args.imgsz, device=args.device))
    t0 = time.time()
    res = proc.process(pk_id, "clip", video, fps)
    print(f"processed in {time.time() - t0:.0f}s", flush=True)

    print("failures:", res.failures)
    if len(res.roles):
        print(res.roles.to_string(index=False))
    if len(res.events):
        print(res.events[["event_name", "frame_idx", "confidence", "method", "missing_reason"]]
              .to_string(index=False))
    g = res.geometry
    if len(g):
        found = int((~g["is_missing"].astype(bool)).sum())
        print(f"goal found on {found}/{len(g)} frames")
    print(f"tracks={len(res.tracks)} poses={len(res.poses)} ball={len(res.ball)}")

    out = Path(args.out) if args.out else ROOT / "data/qc/overlays" / f"{video.stem}_full.mp4"
    meta = pd.Series({"pk_id": pk_id, "fps": fps})
    written = video_overlay.render(
        meta, video, res.tracks, res.poses, res.ball, res.geometry, res.events, out
    )
    print("wrote", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
