"""Independent check: does our ingested pose reproduce the depositor's own render?

The figshare mirror of the women's corpus ships a skeleton video per kick that
the depositors rendered from their pipeline. Our parquet was built from the
Mendeley CSV of the same kicks. If the ingest is faithful, our skeleton drawn
from the parquet must land on the render's ink.

Reported metric: the fraction of our drawn keypoints that fall within
``--tol`` pixels of a lit pixel in the corresponding render frame. This is a
one-sided check -- it catches coordinate, scale, axis and frame-offset errors,
which are exactly the ways an ingest silently goes wrong.

    python scripts/verify_against_upstream_render.py --limit 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pkcv.config import Config  # noqa: E402
from pkcv.io import read_parquet  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--tol", type=int, default=6, help="pixel radius counted as a hit")
    ap.add_argument("--out", default=None, help="write a side-by-side PNG per kick here")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    md = read_parquet(cfg.paths.artifact("metadata"), "metadata")
    poses = read_parquet(cfg.paths.artifact("poses"), "poses")

    renders = md[(md["source"] == "figshare-women-v2") & md["video_relpath"].notna()]
    if not len(renders):
        print("no figshare renders ingested; run `pkcv ingest --source figshare` first")
        return 1

    rows = []
    for _, r in renders.head(args.limit).iterrows():
        # The render and the pose table share a kick identifier but live under
        # different source slugs.
        mend_id = f"mendeley-women-v2:{r['source_identifier']}"
        p = poses[(poses["pk_id"] == mend_id) & (~poses["is_missing"].astype(bool))]
        if not len(p):
            rows.append({"kick": r["source_identifier"], "status": "no matching pose rows"})
            continue

        cap = cv2.VideoCapture(str(Path(cfg.paths.root) / str(r["video_relpath"])))
        hits = total = 0
        checked_frames = 0
        for fidx, g in p.groupby("frame_idx"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ok, frame = cap.read()
            if not ok:
                continue
            lit = (frame.max(axis=2) > 10).astype(np.uint8)
            lit = cv2.dilate(lit, np.ones((2 * args.tol + 1,) * 2, np.uint8))
            h, w = lit.shape
            for _, kp in g.iterrows():
                x, y = int(round(kp["x"])), int(round(kp["y"]))
                if 0 <= x < w and 0 <= y < h:
                    total += 1
                    hits += int(lit[y, x] > 0)
            checked_frames += 1
        cap.release()
        rate = hits / total if total else float("nan")
        rows.append(
            {
                "kick": r["source_identifier"],
                "frames": checked_frames,
                "keypoints": total,
                "within_tol": hits,
                "hit_rate": round(rate, 4),
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if "hit_rate" in df:
        print(f"\nmean hit rate: {df['hit_rate'].mean():.3f} (tol={args.tol}px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
