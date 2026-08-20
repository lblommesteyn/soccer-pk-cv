"""Goal-frame recovery.

A goal filmed from the stand is a trapezoid, not an axis-aligned rectangle, so
this returns the four corners. The earlier bounding-box version was the thing
that made overlays look wrong even when detection was right.

Method, in order:

1. **White mask.** The goal frame is bright and desaturated; grass, kit and
   crowd are not. This is far more selective than a plain edge map, which fires
   on every line on the pitch and every seat in the stand.
2. **Segments**, split into near-vertical (post candidates) and near-horizontal
   (crossbar candidates).
3. **Post clustering** on x. A real post produces many overlapping collinear
   segments, so clustering by x and weighting by total length picks the posts
   out of the noise far more reliably than taking the longest single line.
4. **Pair selection**: the two heaviest clusters far enough apart to be a goal.
5. **Crossbar**: the longest horizontal segment near the post tops that spans
   between them. Its slope is used, so a tilted handheld shot stays tight.
6. **Net gate**: the mouth must be edge-dense. Broadcast graphics, wall panels
   and stand railings form the same posts-and-bar shape but are smooth;
   measured on real footage the mesh scores 0.07-0.12 against 0.01-0.03.

Everything is a heuristic and reports a confidence that says so. When any stage
fails the frame yields nothing, and the caller records a missing row.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GoalConfig:
    white_v_min: int = 150
    white_s_max: int = 70
    #: A goal must span at least this fraction of frame width to be worth
    #: believing; below it we are almost certainly looking at railings.
    min_width_frac: float = 0.10
    #: Posts lean in perspective; this bounds how far from vertical they may be.
    max_lean_deg: float = 25.0
    #: Plausible projected width/height for a 7.32 x 2.44 m goal. Head-on is
    #: about 3.0; a high stand view compresses it, an oblique one stretches it.
    min_aspect: float = 1.1
    max_aspect: float = 9.0
    #: Mesh texture inside the mouth. See the module docstring for measurements.
    min_net_density: float = 0.045
    #: Line detection runs on a downscaled copy. The goal is a large structure,
    #: so this costs almost no accuracy and is roughly 4x faster, which matters
    #: because geometry runs on every frame.
    work_width: int = 960


def white_mask(frame: np.ndarray, cfg: GoalConfig) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return (
        (hsv[:, :, 2] > cfg.white_v_min) & (hsv[:, :, 1] < cfg.white_s_max)
    ).astype(np.uint8) * 255


def _segments(mask: np.ndarray, min_len: int) -> np.ndarray:
    s = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=40, minLineLength=min_len, maxLineGap=12)
    if s is None:
        return np.zeros((0, 4))
    # OpenCV 4 returns (N, 1, 4); OpenCV 5 returns (N, 4).
    return np.asarray(s, dtype=float).reshape(-1, 4)


def _cluster(values: np.ndarray, weights: np.ndarray, tol: float) -> list[dict]:
    """Greedy weighted 1-D clustering, heaviest seed first."""
    clusters: list[dict] = []
    for i in np.argsort(-weights):
        v, w = float(values[i]), float(weights[i])
        for c in clusters:
            if abs(v - c["centre"]) <= tol:
                c["members"].append(int(i))
                c["weight"] += w
                c["centre"] = float(
                    np.average(values[c["members"]], weights=weights[c["members"]])
                )
                break
        else:
            clusters.append({"centre": v, "weight": w, "members": [int(i)]})
    return sorted(clusters, key=lambda c: -c["weight"])


def find_goal(frame: np.ndarray, cfg: GoalConfig | None = None) -> dict | None:
    """Return the goal quad for one frame, or None if it is not confidently found.

    Coordinates are always returned in the *original* frame's pixels, whatever
    resolution the search ran at.
    """
    cfg = cfg or GoalConfig()
    full_w = frame.shape[1]
    if cfg.work_width and full_w > cfg.work_width:
        k = cfg.work_width / full_w
        frame = cv2.resize(frame, (cfg.work_width, int(frame.shape[0] * k)))
    else:
        k = 1.0
    h, w = frame.shape[:2]
    segs = _segments(white_mask(frame, cfg), min_len=max(30, h // 30))
    if len(segs) < 4:
        return None

    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    length = np.hypot(dx, dy)
    ang = np.abs(np.degrees(np.arctan2(dy, dx)))

    verts = segs[(ang > 90 - cfg.max_lean_deg) & (length > h * 0.05)]
    horis = segs[(ang < cfg.max_lean_deg) & (length > w * 0.05)]
    if len(verts) < 2 or len(horis) < 1:
        return None

    vx = (verts[:, 0] + verts[:, 2]) / 2
    vlen = np.hypot(verts[:, 2] - verts[:, 0], verts[:, 3] - verts[:, 1])
    posts = _cluster(vx, vlen, tol=max(8.0, w * 0.01))
    if len(posts) < 2:
        return None

    best = None
    for i in range(len(posts)):
        for j in range(i + 1, len(posts)):
            a, b = posts[i], posts[j]
            if abs(a["centre"] - b["centre"]) < cfg.min_width_frac * w:
                continue
            score = a["weight"] + b["weight"]
            if best is None or score > best[0]:
                best = (score, a, b)
    if best is None:
        return None
    _, pa_, pb_ = best
    left, right = sorted((pa_, pb_), key=lambda c: c["centre"])

    def extent(cluster: dict) -> tuple[float, float]:
        ys = np.concatenate([verts[cluster["members"], 1], verts[cluster["members"], 3]])
        return float(np.percentile(ys, 5)), float(np.percentile(ys, 95))

    lt, lb = extent(left)
    rt, rb = extent(right)

    lo, hi = left["centre"], right["centre"]
    span = hi - lo
    top_band = min(lt, rt) + 0.35 * (max(lb, rb) - min(lt, rt))
    hy = (horis[:, 1] + horis[:, 3]) / 2
    hx1 = np.minimum(horis[:, 0], horis[:, 2])
    hx2 = np.maximum(horis[:, 0], horis[:, 2])
    overlap = np.minimum(hx2, hi) - np.maximum(hx1, lo)
    cand = (hy < top_band) & (overlap > 0.25 * span)
    if not cand.any():
        return None
    hl = np.hypot(horis[:, 2] - horis[:, 0], horis[:, 3] - horis[:, 1])
    bar = horis[int(np.argmax(np.where(cand, hl, -1.0)))]
    bx1, by1, bx2, by2 = bar if bar[0] <= bar[2] else bar[[2, 3, 0, 1]]
    slope = (by2 - by1) / max(bx2 - bx1, 1e-6)

    quad = np.array(
        [
            [lo, by1 + slope * (lo - bx1)],
            [hi, by1 + slope * (hi - bx1)],
            [hi, rb],
            [lo, lb],
        ],
        dtype=float,
    )
    height = float(np.mean([quad[3, 1] - quad[0, 1], quad[2, 1] - quad[1, 1]]))
    if height <= 0:
        return None
    aspect = span / height
    if not (cfg.min_aspect <= aspect <= cfg.max_aspect):
        return None

    x0, x1 = int(max(quad[:, 0].min(), 0)), int(min(quad[:, 0].max(), w))
    y0, y1 = int(max(quad[:, 1].min(), 0)), int(min(quad[:, 1].max(), h))
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    net = float(cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 50, 150).mean() / 255.0)
    if net < cfg.min_net_density:
        return None

    inv = 1.0 / k
    return {
        "quad": quad * inv,
        "post_left_x": float(lo * inv),
        "post_left_y": float(lb * inv),
        "post_right_x": float(hi * inv),
        "post_right_y": float(rb * inv),
        "crossbar_y": float(min(quad[0, 1], quad[1, 1]) * inv),
        "goal_width_px": float(span * inv),
        "goal_height_px": height * inv,
        # A regulation goal is 7.32 m wide. This fixes scale along the goal line
        # only; it is not a full calibration and must not be used across the pitch.
        "px_per_m": float(span * inv / 7.32),
        "net_edge_density": net,
        "confidence": float(np.clip(net / 0.12, 0.2, 0.85)),
    }
