"""Cut a long recording into one clip per penalty.

    python scripts/split_shootout.py <video> --out data/raw/commons/clips/split

Writes the clips plus a JSON sidecar recording, for each one, the contact frame
in the source recording and in the cut clip. Downstream stages then treat each
clip as an ordinary single penalty.

The episodes it finds are candidates, not verified penalties. Review the
printed table before processing them: a ball resting during a stoppage looks
stationary too, and that case is reported with lower confidence rather than
silently dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pkcv.vision.runner import ClipProcessor, VisionConfig  # noqa: E402
from pkcv.vision.shootout import EpisodeConfig, find_episodes, split_video  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--min-stationary-s", type=float, default=0.8)
    ap.add_argument("--min-gap-s", type=float, default=4.0)
    ap.add_argument("--dry-run", action="store_true", help="find episodes, write nothing")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_absolute():
        video = ROOT / video
    out_dir = Path(args.out) if args.out else video.parent / f"{video.stem}_split"

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"{video.name}: {n} frames @ {fps:.2f} fps ({n / fps / 60:.1f} min)", flush=True)

    proc = ClipProcessor(VisionConfig(imgsz=args.imgsz, device=args.device))
    print("detecting...", flush=True)
    _, balls, _ = proc._detect_and_track(str(video))
    print(f"ball rows {len(balls)}", flush=True)

    geom = proc._geometry_rows("split", "split", str(video))
    good = geom[~geom["is_missing"].astype(bool)]
    if not len(good):
        print("no goal found anywhere in the recording; cannot scale the spot tolerance")
        return 1
    gw = float(np.nanmedian(good["goal_width_px"]))
    print(f"goal found on {len(good)}/{len(geom)} frames, median width {gw:.0f}px", flush=True)

    cfg = EpisodeConfig(min_stationary_s=args.min_stationary_s, min_gap_s=args.min_gap_s)
    eps = find_episodes(balls, gw, fps, cfg)
    print(f"\n{len(eps)} candidate penalt{'y' if len(eps) == 1 else 'ies'}:")
    print(f"{'#':>3} {'contact_f':>10} {'t':>9} {'still_s':>8} {'departed':>9} {'conf':>5}")
    for k, e in enumerate(eps, 1):
        still_s = (e.still_to - e.still_from) / fps
        print(f"{k:>3} {e.contact_frame:>10} {e.contact_frame / fps:>8.1f}s "
              f"{still_s:>7.1f}s {str(e.departed):>9} {e.confidence:>5.2f}")

    if args.dry_run or not eps:
        return 0

    written = split_video(str(video), str(out_dir), eps, fps, n, cfg, prefix=video.stem[:24])
    sidecar = out_dir / "episodes.json"
    sidecar.write_text(json.dumps({
        "source_video": str(video),
        "fps": fps,
        "n_frames": n,
        "goal_width_px": gw,
        "clips": written,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {len(written)} clip(s) to {out_dir}")
    print(f"sidecar: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
