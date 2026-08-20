"""Video CV stage: detection, tracking, pose, ball and event estimation.

This stage runs only on sources whose ``media_kind`` is ``video``. It is
deliberately refused for ``pose_table`` sources (the pose already exists and
re-deriving it would destroy provenance) and for ``render_only`` sources
(a skeleton render on black has no ball, keeper or goal to find -- running a
detector on it would return numbers that mean nothing).

Model choices are sized for a 10 GB RTX 3080: YOLO11m for pose and detection at
960 px runs comfortably in well under 4 GB, so a second process can share the
card. Nothing here needs a large GPU.

Failure discipline: every stage returns rows with ``is_missing`` and a reason.
A frame where the ball was not detected produces a null-coordinate row, not an
interpolated guess. Interpolation happens once, visibly, in the temporal layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pkcv.schemas import KEYPOINT_INDICES, KEYPOINT_NAMES, PROCESSING_VERSION

log = logging.getLogger(__name__)

COCO_PERSON = 0
COCO_SPORTS_BALL = 32


class VisionUnavailable(RuntimeError):
    """Raised when the vision extras are not installed."""


@dataclass
class VisionConfig:
    det_weights: str = "yolo11m.pt"
    pose_weights: str = "yolo11m-pose.pt"
    imgsz: int = 960
    device: str = "cuda:0"
    half: bool = True
    person_conf: float = 0.35
    ball_conf: float = 0.15  # the ball is small and low-contrast; a low bar plus
    # a trajectory filter beats a high bar plus dropouts
    tracker: str = "botsort.yaml"
    max_det: int = 40
    #: A goalkeeper stands near the goal line; a kicker approaches from the
    #: penalty spot. Roles are assigned geometrically rather than by class.
    keeper_depth_quantile: float = 0.25

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> VisionConfig:
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ClipResult:
    tracks: pd.DataFrame = field(default_factory=pd.DataFrame)
    poses: pd.DataFrame = field(default_factory=pd.DataFrame)
    ball: pd.DataFrame = field(default_factory=pd.DataFrame)
    geometry: pd.DataFrame = field(default_factory=pd.DataFrame)
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    failures: list[str] = field(default_factory=list)


def _require_ultralytics():
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise VisionUnavailable(
            "vision extras not installed; `pip install -e .[vision]`"
        ) from exc
    from ultralytics import YOLO

    return YOLO


def _prov(model_name: str | None, model_version: str | None, provenance: str) -> dict:
    return {
        "processing_version": PROCESSING_VERSION,
        "derivation": "pkcv_derived",
        "model_name": model_name,
        "model_version": model_version,
        "provenance": provenance,
    }


class ClipProcessor:
    """Runs the full per-clip vision pipeline."""

    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg
        self._det = None
        self._pose = None

    # ------------------------------------------------------------ lazy models

    @property
    def det(self):
        if self._det is None:
            YOLO = _require_ultralytics()
            self._det = YOLO(self.cfg.det_weights)
        return self._det

    @property
    def pose(self):
        if self._pose is None:
            YOLO = _require_ultralytics()
            self._pose = YOLO(self.cfg.pose_weights)
        return self._pose

    # ------------------------------------------------------------------- main

    def process(self, pk_id: str, source: str, video_path: str | Path, fps: float) -> ClipResult:
        video_path = str(video_path)
        result = ClipResult()

        persons, balls, failures = self._detect_and_track(video_path)
        result.failures.extend(failures)
        if not len(persons):
            result.failures.append("no_person_detections")
            return result

        roles = self._assign_roles(persons)
        persons = persons.merge(roles, on="track_id", how="left")
        persons["role"] = persons["role"].fillna("other")

        kicker_id = _first_track_for(roles, "kicker")
        keeper_id = _first_track_for(roles, "keeper")

        result.tracks = self._track_rows(pk_id, source, persons, fps, video_path)
        result.ball = self._ball_rows(pk_id, source, balls, fps, video_path)
        result.poses = self._pose_rows(pk_id, source, video_path, persons, kicker_id, keeper_id, fps)
        result.geometry = self._geometry_rows(pk_id, source, video_path)

        contact = self._contact_frame(persons, balls, kicker_id, fps)
        result.events = self._event_rows(pk_id, source, persons, balls, kicker_id, keeper_id, contact, fps)

        # Anchor every table on the contact frame once it is known.
        if contact["frame_idx"] is not None:
            for df in (result.tracks, result.poses, result.ball):
                if len(df):
                    df["t_ms_rel_contact"] = (df["frame_idx"] - contact["frame_idx"]) / fps * 1000.0
        return result

    # ------------------------------------------------------- detect and track

    def _detect_and_track(self, video_path: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        failures: list[str] = []
        person_rows, ball_rows = [], []
        stream = self.det.track(
            source=video_path,
            stream=True,
            persist=True,
            tracker=self.cfg.tracker,
            classes=[COCO_PERSON, COCO_SPORTS_BALL],
            conf=min(self.cfg.person_conf, self.cfg.ball_conf),
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            half=self.cfg.half,
            max_det=self.cfg.max_det,
            verbose=False,
        )
        for frame_idx, res in enumerate(stream):
            boxes = getattr(res, "boxes", None)
            if boxes is None or boxes.id is None:
                continue
            xywh = boxes.xywh.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            for (cx, cy, w, h), c, cf, tid in zip(xywh, cls, conf, ids, strict=False):
                row = {
                    "frame_idx": frame_idx,
                    "track_id": int(tid),
                    "cx": float(cx),
                    "cy": float(cy),
                    "w": float(w),
                    "h": float(h),
                    "conf": float(cf),
                }
                if c == COCO_PERSON and cf >= self.cfg.person_conf:
                    person_rows.append(row)
                elif c == COCO_SPORTS_BALL and cf >= self.cfg.ball_conf:
                    ball_rows.append(row)
        if not ball_rows:
            failures.append("no_ball_detections")
        return pd.DataFrame(person_rows), pd.DataFrame(ball_rows), failures

    # ------------------------------------------------------------------ roles

    def _assign_roles(self, persons: pd.DataFrame) -> pd.DataFrame:
        """Assign kicker / keeper by track geometry.

        The keeper is the person who stays highest in frame (nearest the goal,
        hence smallest ``cy``) across the clip; the kicker is the person whose
        box travels the furthest, because they run up. Both are heuristics, and
        both are recorded with a confidence so a downstream consumer can weigh
        them -- there is no keeper/kicker class in COCO to appeal to.
        """
        if not len(persons):
            return pd.DataFrame(columns=["track_id", "role", "role_confidence"])
        agg = persons.groupby("track_id").agg(
            n=("frame_idx", "size"),
            cy_med=("cy", "median"),
            cx_span=("cx", lambda s: float(s.max() - s.min())),
            cy_span=("cy", lambda s: float(s.max() - s.min())),
            h_med=("h", "median"),
        )
        # Ignore blink-length tracks: they are detector noise, not people.
        agg = agg[agg["n"] >= max(3, int(0.1 * persons["frame_idx"].nunique()))]
        if not len(agg):
            return pd.DataFrame(columns=["track_id", "role", "role_confidence"])

        agg["travel"] = np.hypot(agg["cx_span"], agg["cy_span"]) / agg["h_med"].clip(lower=1)
        keeper = agg["cy_med"].idxmin()
        remaining = agg.drop(index=keeper)
        kicker = remaining["travel"].idxmax() if len(remaining) else None

        rows = []
        for tid in agg.index:
            if tid == keeper:
                role, conf = "keeper", _separation_conf(agg["cy_med"], keeper, smaller_is_better=True)
            elif tid == kicker:
                role, conf = "kicker", _separation_conf(remaining["travel"], kicker, smaller_is_better=False)
            else:
                role, conf = "other", None
            rows.append({"track_id": int(tid), "role": role, "role_confidence": conf})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------- rows

    def _track_rows(self, pk_id, source, persons, fps, video_path) -> pd.DataFrame:
        prov = _prov(self.cfg.det_weights, _weights_version(self.cfg.det_weights), f"pkcv:vision/track:{Path(video_path).name}")
        out = []
        for role in ("kicker", "keeper"):
            sub = persons[persons["role"] == role].sort_values("frame_idx")
            if not len(sub):
                out.append(
                    {
                        "pk_id": pk_id,
                        "source": source,
                        "source_identifier": pk_id,
                        "role": role,
                        "track_id": None,
                        "frame_idx": -1,
                        "is_missing": True,
                        "missing_reason": f"no_track_assigned_role_{role}",
                        **prov,
                    }
                )
                continue
            t = sub["frame_idx"].to_numpy(dtype=float) / fps
            vx, vy = _central_diff(sub["cx"].to_numpy(float), sub["cy"].to_numpy(float), t)
            for i, (_, r) in enumerate(sub.iterrows()):
                out.append(
                    {
                        "pk_id": pk_id,
                        "source": source,
                        "source_identifier": pk_id,
                        "role": role,
                        "track_id": str(int(r["track_id"])),
                        "frame_idx": int(r["frame_idx"]),
                        "t_s": float(r["frame_idx"]) / fps,
                        "t_ms_rel_contact": None,
                        "bbox_cx": float(r["cx"]),
                        "bbox_cy": float(r["cy"]),
                        "bbox_w": float(r["w"]),
                        "bbox_h": float(r["h"]),
                        "vx_px_s": _nn(vx[i]),
                        "vy_px_s": _nn(vy[i]),
                        "confidence": float(r["conf"]),
                        "is_missing": False,
                        "missing_reason": None,
                        **prov,
                    }
                )
        return pd.DataFrame(out)

    def _ball_rows(self, pk_id, source, balls, fps, video_path) -> pd.DataFrame:
        prov = _prov(self.cfg.det_weights, _weights_version(self.cfg.det_weights), f"pkcv:vision/ball:{Path(video_path).name}")
        if not len(balls):
            return pd.DataFrame(
                [
                    {
                        "pk_id": pk_id,
                        "source": source,
                        "frame_idx": -1,
                        "is_missing": True,
                        "missing_reason": "no_ball_detections",
                        **prov,
                    }
                ]
            )
        # Keep the single most persistent ball track; stray detections on
        # advertising boards and the centre circle are common.
        best = balls.groupby("track_id").size().idxmax()
        b = balls[balls["track_id"] == best].sort_values("frame_idx")
        t = b["frame_idx"].to_numpy(float) / fps
        vx, vy = _central_diff(b["cx"].to_numpy(float), b["cy"].to_numpy(float), t)
        rows = []
        for i, (_, r) in enumerate(b.iterrows()):
            rows.append(
                {
                    "pk_id": pk_id,
                    "source": source,
                    "frame_idx": int(r["frame_idx"]),
                    "t_s": float(r["frame_idx"]) / fps,
                    "t_ms_rel_contact": None,
                    "x": float(r["cx"]),
                    "y": float(r["cy"]),
                    "bbox_w": float(r["w"]),
                    "bbox_h": float(r["h"]),
                    "vx_px_s": _nn(vx[i]),
                    "vy_px_s": _nn(vy[i]),
                    "confidence": float(r["conf"]),
                    "is_missing": False,
                    "missing_reason": None,
                    **prov,
                }
            )
        return pd.DataFrame(rows)

    def _pose_rows(self, pk_id, source, video_path, persons, kicker_id, keeper_id, fps) -> pd.DataFrame:
        """Second pass: pose on the whole frame, matched to role tracks by IoU."""
        prov = _prov(self.cfg.pose_weights, _weights_version(self.cfg.pose_weights), f"pkcv:vision/pose:{Path(video_path).name}")
        wanted = {kicker_id: "kicker", keeper_id: "keeper"}
        wanted = {k: v for k, v in wanted.items() if k is not None}
        if not wanted:
            return pd.DataFrame()
        boxes_by_frame: dict[int, list[tuple[str, np.ndarray]]] = {}
        for _, r in persons[persons["track_id"].isin(wanted)].iterrows():
            role = wanted[int(r["track_id"])]
            box = np.array(
                [r["cx"] - r["w"] / 2, r["cy"] - r["h"] / 2, r["cx"] + r["w"] / 2, r["cy"] + r["h"] / 2]
            )
            boxes_by_frame.setdefault(int(r["frame_idx"]), []).append((role, box))

        rows = []
        stream = self.pose.predict(
            source=video_path,
            stream=True,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            half=self.cfg.half,
            conf=self.cfg.person_conf,
            verbose=False,
        )
        for frame_idx, res in enumerate(stream):
            targets = boxes_by_frame.get(frame_idx)
            if not targets or res.keypoints is None or res.boxes is None or not len(res.boxes):
                continue
            det = res.boxes.xyxy.cpu().numpy()
            kps = res.keypoints.data.cpu().numpy()  # (n, 17, 3)
            for role, box in targets:
                j = int(np.argmax([_iou(box, d) for d in det]))
                if _iou(box, det[j]) < 0.3:
                    continue
                bx = det[j]
                cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
                h = max(bx[3] - bx[1], 1.0)
                for k in KEYPOINT_INDICES:
                    x, y, c = kps[j][k]
                    missing = bool(c < 0.3)
                    rows.append(
                        {
                            "pk_id": pk_id,
                            "source": source,
                            "role": role,
                            "frame_idx": frame_idx,
                            "t_s": frame_idx / fps,
                            "t_ms_rel_contact": None,
                            "kp_index": int(k),
                            "kp_name": KEYPOINT_NAMES[k],
                            "x": None if missing else float(x),
                            "y": None if missing else float(y),
                            "x_c": None if missing else float(x - cx),
                            "y_c": None if missing else float(y - cy),
                            "x_n": None if missing else float((x - cx) / h),
                            "y_n": None if missing else float((y - cy) / h),
                            "confidence": float(c),
                            "is_missing": missing,
                            "missing_reason": "low_keypoint_confidence" if missing else None,
                            **prov,
                        }
                    )
        return pd.DataFrame(rows)

    # --------------------------------------------------------------- geometry

    def _geometry_rows(self, pk_id, source, video_path) -> pd.DataFrame:
        """Locate the goal frame from the two strongest near-vertical white lines.

        Returns a missing row when the goal is not confidently found, which is
        the common case on a tight kicker-side camera where the posts are out of
        shot. Nothing is extrapolated.
        """
        import cv2

        prov = _prov("opencv-lsd", cv2.__version__, f"pkcv:vision/geometry:{Path(video_path).name}")
        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        miss = {
            "pk_id": pk_id,
            "source": source,
            "frame_idx": 0,
            "is_missing": True,
            "missing_reason": "goal_frame_not_detected",
            **prov,
        }
        if not ok:
            miss["missing_reason"] = "video_unreadable"
            return pd.DataFrame([miss])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=frame.shape[0] // 4, maxLineGap=12)
        if lines is None:
            return pd.DataFrame([miss])
        verticals = [
            ln[0]
            for ln in lines
            if abs(ln[0][2] - ln[0][0]) < 0.15 * abs(ln[0][3] - ln[0][1] or 1)
        ]
        if len(verticals) < 2:
            return pd.DataFrame([miss])
        verticals.sort(key=lambda ln: (ln[0] + ln[2]) / 2)
        left, right = verticals[0], verticals[-1]
        lx, rx = (left[0] + left[2]) / 2, (right[0] + right[2]) / 2
        if rx - lx < 0.15 * frame.shape[1]:
            miss["missing_reason"] = "candidate_posts_too_close_to_be_a_goal"
            return pd.DataFrame([miss])
        crossbar_y = float(min(left[1], left[3], right[1], right[3]))
        goal_w = float(rx - lx)
        return pd.DataFrame(
            [
                {
                    "pk_id": pk_id,
                    "source": source,
                    "frame_idx": 0,
                    "post_left_x": float(lx),
                    "post_left_y": float(max(left[1], left[3])),
                    "post_right_x": float(rx),
                    "post_right_y": float(max(right[1], right[3])),
                    "crossbar_y": crossbar_y,
                    "goal_width_px": goal_w,
                    "goal_height_px": float(max(left[1], left[3]) - crossbar_y),
                    # A regulation goal is 7.32 m wide; that fixes the scale
                    # along the goal line only, not across the pitch.
                    "px_per_m": goal_w / 7.32,
                    "confidence": 0.4,  # two-line heuristic, not a calibration
                    "is_missing": False,
                    "missing_reason": None,
                    **prov,
                }
            ]
        )

    # ----------------------------------------------------------------- events

    def _contact_frame(self, persons, balls, kicker_id, fps) -> dict:
        """Ball contact = the frame where the ball's speed first jumps while it
        is close to the kicker's feet.

        A penalty ball is stationary until struck, so the first large positive
        step in ball speed is the strike. Requiring proximity to the kicker
        rejects a speed jump caused by a redetection elsewhere in the frame.
        """
        if kicker_id is None or not len(balls):
            return {"frame_idx": None, "confidence": None, "method": "ball_speed_onset", "reason": "no_ball_or_kicker_track"}
        best = balls.groupby("track_id").size().idxmax()
        b = balls[balls["track_id"] == best].sort_values("frame_idx")
        if len(b) < 4:
            return {"frame_idx": None, "confidence": None, "method": "ball_speed_onset", "reason": "ball_track_too_short"}
        k = persons[persons["track_id"] == kicker_id].set_index("frame_idx")
        t = b["frame_idx"].to_numpy(float) / fps
        vx, vy = _central_diff(b["cx"].to_numpy(float), b["cy"].to_numpy(float), t)
        speed = np.hypot(vx, vy)
        scale = float(k["h"].median()) if len(k) else 1.0
        speed_n = speed / max(scale, 1.0)  # body-heights per second
        near = []
        for i, (_, r) in enumerate(b.iterrows()):
            f = int(r["frame_idx"])
            if f not in k.index:
                near.append(np.inf)
                continue
            kr = k.loc[f]
            kr = kr.iloc[0] if isinstance(kr, pd.DataFrame) else kr
            near.append(float(np.hypot(r["cx"] - kr["cx"], r["cy"] - kr["cy"]) / max(kr["h"], 1.0)))
        near = np.asarray(near)
        cand = np.where((speed_n > 3.0) & (near < 1.5))[0]
        if not len(cand):
            return {"frame_idx": None, "confidence": None, "method": "ball_speed_onset", "reason": "no_speed_onset_near_kicker"}
        i = int(cand[0])
        margin = float(speed_n[i] / max(np.nanmedian(speed_n[:i]) if i else 1e-6, 1e-6))
        return {
            "frame_idx": int(b["frame_idx"].iloc[i]),
            "confidence": float(np.clip(margin / 10.0, 0.1, 0.95)),
            "method": "ball_speed_onset",
            "reason": None,
        }

    def _event_rows(self, pk_id, source, persons, balls, kicker_id, keeper_id, contact, fps) -> pd.DataFrame:
        prov = _prov(self.cfg.det_weights, _weights_version(self.cfg.det_weights), "pkcv:vision/events")
        rows = [
            {
                "pk_id": pk_id,
                "source": source,
                "event_name": "ball_contact",
                "frame_idx": contact["frame_idx"],
                "t_s": None if contact["frame_idx"] is None else contact["frame_idx"] / fps,
                "confidence": contact["confidence"],
                "method": contact["method"],
                "is_missing": contact["frame_idx"] is None,
                "missing_reason": contact.get("reason"),
                **prov,
            }
        ]

        # Keeper commit: first frame at which lateral speed exceeds a quarter of
        # a body height per second and keeps rising -- a real dive, not a shuffle.
        commit = {"frame_idx": None, "confidence": None, "reason": "no_keeper_track"}
        if keeper_id is not None:
            kp = persons[persons["track_id"] == keeper_id].sort_values("frame_idx")
            if len(kp) >= 4:
                t = kp["frame_idx"].to_numpy(float) / fps
                vx, _ = _central_diff(kp["cx"].to_numpy(float), kp["cy"].to_numpy(float), t)
                scale = max(float(kp["h"].median()), 1.0)
                lat = np.abs(vx) / scale
                idx = np.where(lat > 0.25)[0]
                if len(idx):
                    i = int(idx[0])
                    commit = {
                        "frame_idx": int(kp["frame_idx"].iloc[i]),
                        "confidence": float(np.clip(lat[i], 0.1, 0.95)),
                        "reason": None,
                    }
                else:
                    commit["reason"] = "keeper_lateral_speed_never_exceeded_threshold"
            else:
                commit["reason"] = "keeper_track_too_short"
        rows.append(
            {
                "pk_id": pk_id,
                "source": source,
                "event_name": "keeper_commit",
                "frame_idx": commit["frame_idx"],
                "t_s": None if commit["frame_idx"] is None else commit["frame_idx"] / fps,
                "confidence": commit["confidence"],
                "method": "keeper_lateral_velocity_threshold",
                "is_missing": commit["frame_idx"] is None,
                "missing_reason": commit["reason"],
                **prov,
            }
        )

        # Plant foot: the plant lands just before contact and is momentarily
        # stationary, so it is the pose-velocity minimum in the 400 ms before
        # contact. Needs both a contact anchor and pose, so it is emitted as
        # missing here and filled by the pose-aware pass in temporal/events.
        rows.append(
            {
                "pk_id": pk_id,
                "source": source,
                "event_name": "plant_foot",
                "frame_idx": None,
                "t_s": None,
                "confidence": None,
                "method": "ankle_velocity_minimum_pre_contact",
                "is_missing": True,
                "missing_reason": (
                    "requires_contact_anchor_and_pose"
                    if contact["frame_idx"] is None
                    else "not_yet_estimated"
                ),
                **prov,
            }
        )
        return pd.DataFrame(rows)


# ------------------------------------------------------------------ helpers


def _first_track_for(roles: pd.DataFrame, role: str) -> int | None:
    sub = roles[roles["role"] == role]
    return int(sub["track_id"].iloc[0]) if len(sub) else None


def _separation_conf(series: pd.Series, winner, smaller_is_better: bool) -> float:
    """How clearly the winner beats the runner-up, mapped to (0.1, 0.95)."""
    if len(series) < 2:
        return 0.3
    ordered = series.sort_values(ascending=smaller_is_better)
    best, second = float(ordered.iloc[0]), float(ordered.iloc[1])
    denom = max(abs(second), 1e-6)
    return float(np.clip(abs(second - best) / denom, 0.1, 0.95))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _central_diff(x: np.ndarray, y: np.ndarray, t: np.ndarray):
    n = len(x)
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - 1), min(n - 1, i + 1)
        dt = t[hi] - t[lo]
        if dt > 0:
            vx[i] = (x[hi] - x[lo]) / dt
            vy[i] = (y[hi] - y[lo]) / dt
    return vx, vy


def _nn(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def _weights_version(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    from pkcv.io import sha256_file

    return sha256_file(p)[:16]
