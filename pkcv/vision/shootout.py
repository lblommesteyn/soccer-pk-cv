"""Split a long recording into individual penalties.

A shootout recording holds a dozen kicks under one camera, which makes it the
most valuable kind of file available: consistent framing, consistent lighting,
many kicks. But every downstream stage here is defined per penalty, so the file
has to be cut first.

The cut points come from the same signal contact detection uses. A penalty has a
distinctive ball signature: the ball is placed on the spot, sits still for
seconds, then leaves abruptly. So an *episode* is a run of stationary ball
observations near one location followed by departure, and a shootout is a
sequence of such episodes at roughly the same spot.

Two guards matter, because both failure modes produce plausible-looking rubbish:

* a ball merely resting during a stoppage also sits still, so an episode must
  end in departure or in the track being lost -- a ball that is still picked up,
  stationary, at the end of its track is not a kick;
* the same physical kick can be split across two tracker ids, so episodes closer
  together than ``min_gap_s`` are merged rather than counted twice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EpisodeConfig:
    #: Drift, in goal-widths, still counted as sitting on the spot.
    spot_tol_goal_widths: float = 0.05
    #: A ball must hold still at least this long to be a placed penalty.
    min_stationary_s: float = 0.8
    #: Two contacts closer than this are the same kick seen twice.
    min_gap_s: float = 4.0
    #: How much footage to keep either side of contact when cutting.
    pre_roll_s: float = 4.0
    post_roll_s: float = 2.5


@dataclass
class Episode:
    contact_frame: int
    still_from: int
    still_to: int
    spot_x: float
    spot_y: float
    departed: bool
    track_id: str

    @property
    def confidence(self) -> float:
        # An observed departure is direct evidence; a track that simply ends is
        # consistent with a kick but also with the tracker giving up.
        return 0.8 if self.departed else 0.45

    def window(self, fps: float, n_frames: int, cfg: EpisodeConfig) -> tuple[int, int]:
        lo = max(0, int(self.contact_frame - cfg.pre_roll_s * fps))
        hi = min(n_frames - 1, int(self.contact_frame + cfg.post_roll_s * fps))
        return lo, hi


def find_episodes(
    balls: pd.DataFrame,
    goal_width_px: float,
    fps: float,
    cfg: EpisodeConfig | None = None,
) -> list[Episode]:
    """Return every stationary-then-departing ball episode, in time order."""
    cfg = cfg or EpisodeConfig()
    if balls is None or not len(balls) or not goal_width_px or goal_width_px <= 0:
        return []

    tol = cfg.spot_tol_goal_widths * goal_width_px
    min_still = max(int(cfg.min_stationary_s * fps), 5)
    episodes: list[Episode] = []

    for tid, g in balls.groupby("track_id"):
        g = g.sort_values("frame_idx").reset_index(drop=True)
        if len(g) < min_still:
            continue
        x = g["cx"].to_numpy(float)
        y = g["cy"].to_numpy(float)
        frames = g["frame_idx"].to_numpy(int)

        i = 0
        while i < len(g):
            # Grow a stationary run from i, anchored on its own opening median.
            j = i
            anchor_x, anchor_y = x[i], y[i]
            while j < len(g) and np.hypot(x[j] - anchor_x, y[j] - anchor_y) <= tol:
                if j - i >= 4:  # re-centre once there is enough support
                    anchor_x = float(np.median(x[i : j + 1]))
                    anchor_y = float(np.median(y[i : j + 1]))
                j += 1
            run = j - i
            if run >= min_still:
                departed = j < len(g)
                episodes.append(
                    Episode(
                        contact_frame=int(frames[j]) if departed else int(frames[j - 1]) + 1,
                        still_from=int(frames[i]),
                        still_to=int(frames[j - 1]),
                        spot_x=float(np.median(x[i:j])),
                        spot_y=float(np.median(y[i:j])),
                        departed=departed,
                        track_id=str(tid),
                    )
                )
                i = j
            else:
                i += 1

    episodes.sort(key=lambda e: e.contact_frame)

    # Merge near-duplicates: the same kick can appear under two tracker ids.
    merged: list[Episode] = []
    for e in episodes:
        if merged and (e.contact_frame - merged[-1].contact_frame) < cfg.min_gap_s * fps:
            # Keep whichever has the stronger evidence.
            if e.confidence > merged[-1].confidence:
                merged[-1] = e
            continue
        merged.append(e)
    return merged


def split_video(
    video_path: str,
    out_dir: str,
    episodes: list[Episode],
    fps: float,
    n_frames: int,
    cfg: EpisodeConfig | None = None,
    prefix: str = "pk",
) -> list[dict]:
    """Cut one clip per episode with ffmpeg. Returns the written clip records."""
    import subprocess
    from pathlib import Path

    cfg = cfg or EpisodeConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for k, ep in enumerate(episodes, start=1):
        lo, hi = ep.window(fps, n_frames, cfg)
        dst = out / f"{prefix}_{k:02d}.mp4"
        if not dst.exists():
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{lo / fps:.3f}", "-i", str(video_path),
                "-t", f"{(hi - lo) / fps:.3f}",
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an",
                str(dst),
            ]
            try:
                subprocess.run(cmd, check=True, timeout=1800)
            except (subprocess.SubprocessError, FileNotFoundError, OSError):
                continue
        written.append({
            "index": k,
            "path": str(dst),
            "source_contact_frame": ep.contact_frame,
            # Contact in the *cut* clip's own frame numbering, which is what the
            # per-clip pipeline will see.
            "clip_contact_frame": ep.contact_frame - lo,
            "start_frame": lo,
            "end_frame": hi,
            "departed": ep.departed,
            "confidence": ep.confidence,
            "spot_x": ep.spot_x,
            "spot_y": ep.spot_y,
        })
    return written
