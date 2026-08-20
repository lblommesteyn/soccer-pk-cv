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
    #: BoT-SORT's global motion compensation raises inside OpenCV 5's sparse
    #: optical flow and silently falls back to the identity transform, so its
    #: main advantage on handheld footage is unavailable anyway -- and it emits
    #: one warning per frame. ByteTrack does the same job here without either.
    tracker: str = "bytetrack.yaml"
    max_det: int = 40
    #: A goalkeeper stands near the goal line; a kicker approaches from the
    #: penalty spot. Roles are assigned geometrically rather than by class.
    keeper_depth_quantile: float = 0.25
    #: A broadcast goal is filmed at an angle, so posts are not vertical in the
    #: image. This bounds how far from vertical a line may lean and still be a
    #: post candidate.
    post_max_lean_deg: float = 30.0
    #: The goal is not visible in every frame of a clip, so probe several.
    geometry_probe_frames: int = 12
    #: A goal must span at least this fraction of the frame width, and its
    #: width/height ratio must be plausible for a 7.32 x 2.44 m goal seen from
    #: any angle (head-on is ~3.0; a very oblique view compresses it).
    goal_min_width_frac: float = 0.15
    goal_min_aspect: float = 1.2
    goal_max_aspect: float = 8.0
    #: The goal mouth is a mesh, so it is edge-dense. Broadcast graphics and
    #: flat wall panels form the same two-posts-and-a-bar shape but are smooth;
    #: measured on real footage the gap is 0.07-0.10 against 0.01-0.03.
    net_min_edge_density: float = 0.045
    #: Ball speed, in goal-widths per second, that counts as struck, and how
    #: many consecutive frames must exceed it.
    #: How far, in goal-widths, the ball may drift and still count as sitting
    #: on the spot. Camera pan moves it a few pixels even when it is still.
    spot_tol_goal_widths: float = 0.05
    #: A ball must hold still this long to be believed to be on the spot.
    min_stationary_s: float = 0.4
    #: Window around contact in which a keeper's dive counts as a commitment.
    commit_window_before_s: float = 1.2
    commit_window_after_s: float = 0.6
    #: Lateral speed, in keeper body-heights per second, that counts as a dive.
    commit_lateral_speed: float = 0.35

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
    roles: pd.DataFrame = field(default_factory=pd.DataFrame)
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

        # Order matters: geometry gives the scale that contact needs, and
        # contact gives the instant that role assignment needs.
        result.geometry = self._geometry_rows(pk_id, source, video_path)
        contact = self._contact_frame(balls, result.geometry, fps)
        if contact["frame_idx"] is None and contact.get("reason"):
            result.failures.append(f"contact:{contact['reason']}")

        roles = self._assign_roles(persons, result.geometry, balls, contact["frame_idx"])
        persons = persons.merge(roles, on="track_id", how="left")
        persons["role"] = persons["role"].fillna("other")
        result.roles = roles

        kicker_id = _first_track_for(roles, "kicker")
        keeper_id = _first_track_for(roles, "keeper")

        result.tracks = self._track_rows(pk_id, source, persons, fps, video_path)
        result.ball = self._ball_rows(pk_id, source, balls, fps, video_path)
        result.poses = self._pose_rows(pk_id, source, video_path, persons, kicker_id, keeper_id, fps)
        result.events = self._event_rows(
            pk_id, source, persons, balls, kicker_id, keeper_id, contact, fps
        )

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

    def _assign_roles(
        self,
        persons: pd.DataFrame,
        geometry: pd.DataFrame | None = None,
        balls: pd.DataFrame | None = None,
        contact_frame: int | None = None,
    ) -> pd.DataFrame:
        """Assign kicker and keeper.

        COCO has no goalkeeper class, so both roles are inferred from the
        penalty's own geometry rather than from appearance:

        * **keeper** -- the track that spends the run-up closest to the goal
          mouth, in goal-widths. Measured before contact, because after contact
          the keeper dives and outfield players pour into the box.
        * **kicker** -- the track closest to the ball at the moment of contact.
          This is what actually defines the kicker, and it is far more reliable
          than "the person who moves most", which on any real clip picks a
          running defender or a panning-induced track.

        When contact or the ball is unavailable the kicker falls back to
        greatest travel during the run-up, at a reduced confidence, and
        ``role_basis`` records which rule was used.
        """
        empty = pd.DataFrame(columns=["track_id", "role", "role_confidence", "role_basis"])
        if not len(persons):
            return empty

        n_frames = persons["frame_idx"].nunique()
        agg = persons.groupby("track_id").agg(
            n=("frame_idx", "size"),
            cx_med=("cx", "median"),
            cy_med=("cy", "median"),
            cx_span=("cx", lambda s: float(s.max() - s.min())),
            cy_span=("cy", lambda s: float(s.max() - s.min())),
            h_med=("h", "median"),
        )
        agg = agg[agg["n"] >= max(3, int(0.08 * n_frames))]
        if not len(agg):
            return empty
        agg["travel"] = np.hypot(agg["cx_span"], agg["cy_span"]) / agg["h_med"].clip(lower=1)

        goal = _goal_centre(geometry, upto_frame=contact_frame)

        # ---- keeper -------------------------------------------------------
        if goal is not None:
            gx, gy, gw = goal
            pre = persons if contact_frame is None else persons[persons["frame_idx"] <= contact_frame]
            pre = pre if len(pre) else persons
            dist = (
                pre.assign(d=np.hypot(pre["cx"] - gx, pre["cy"] - gy) / gw)
                .groupby("track_id")["d"]
                .median()
                .reindex(agg.index)
            )
            keeper = dist.idxmin()
            keeper_conf = _separation_conf(dist.dropna(), keeper, smaller_is_better=True)
            keeper_basis = "nearest_goal_mouth_pre_contact"
        else:
            keeper = agg["cy_med"].idxmin()
            keeper_conf = 0.5 * _separation_conf(agg["cy_med"], keeper, smaller_is_better=True)
            keeper_basis = "highest_in_frame_no_goal_geometry"

        # ---- kicker -------------------------------------------------------
        remaining = agg.drop(index=keeper, errors="ignore")
        kicker, kicker_conf, kicker_basis = None, None, None
        if len(remaining):
            # The ball is usually *not* detected on the contact frame itself --
            # the tracker loses it the moment it is struck. The kicker is the
            # player standing over the spot, so the last observed ball position
            # at or before contact is the right anchor, and it is the spot.
            ball_at_contact = None
            if balls is not None and len(balls) and contact_frame is not None:
                upto = balls[balls["frame_idx"] <= contact_frame]
                if len(upto):
                    last = upto.sort_values("frame_idx").iloc[-1]
                    ball_at_contact = (float(last["cx"]), float(last["cy"]))
            if ball_at_contact is not None:
                # Look in a short window around contact: on the contact frame
                # alone the kicker's box can be missed by one detection.
                w = max(2, int(0.08 * n_frames / 10) + 2)
                near = persons[
                    (persons["frame_idx"] >= contact_frame - w)
                    & (persons["frame_idx"] <= contact_frame + w)
                ]
                near = near[near["track_id"].isin(remaining.index)]
                if len(near):
                    bx, by = ball_at_contact
                    d = np.hypot(near["cx"] - bx, near["cy"] - by) / near["h"].clip(lower=1)
                    near = near.assign(d=d).groupby("track_id")["d"].min()
                    kicker = near.idxmin()
                    kicker_conf = _separation_conf(near, kicker, smaller_is_better=True)
                    kicker_basis = "nearest_ball_spot_at_contact"
            if kicker is None:
                kicker = remaining["travel"].idxmax()
                kicker_conf = 0.5 * _separation_conf(
                    remaining["travel"], kicker, smaller_is_better=False
                )
                kicker_basis = "greatest_travel_no_contact_anchor"

        rows = []
        for tid in agg.index:
            if tid == keeper:
                role, conf, basis = "keeper", keeper_conf, keeper_basis
            elif kicker is not None and tid == kicker:
                role, conf, basis = "kicker", kicker_conf, kicker_basis
            else:
                role, conf, basis = "other", None, None
            rows.append(
                {"track_id": int(tid), "role": role, "role_confidence": conf, "role_basis": basis}
            )
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
        """Recover the goal quad on every frame.

        Per-frame rather than once per clip: this footage is handheld and pans,
        so a single quad drifts off the goal within a second. Frames where the
        goal is not confidently found produce a missing row, so a consumer can
        see exactly when the goal was in shot.
        """
        import cv2

        from pkcv.vision.geometry import GoalConfig, find_goal

        prov = _prov(
            "pkcv-goal-quad", cv2.__version__, f"pkcv:vision/geometry:{Path(video_path).name}"
        )
        gcfg = GoalConfig(
            min_width_frac=self.cfg.goal_min_width_frac,
            max_lean_deg=self.cfg.post_max_lean_deg,
            min_aspect=self.cfg.goal_min_aspect,
            max_aspect=self.cfg.goal_max_aspect,
            min_net_density=self.cfg.net_min_edge_density,
        )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return pd.DataFrame(
                [{"pk_id": pk_id, "source": source, "frame_idx": None, "is_missing": True,
                  "missing_reason": "video_unreadable", **prov}]
            )

        rows = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            g = find_goal(frame, gcfg)
            if g is None:
                rows.append(
                    {"pk_id": pk_id, "source": source, "frame_idx": idx, "is_missing": True,
                     "missing_reason": "no_goal_like_structure_in_frame", **prov}
                )
            else:
                q = g.pop("quad")
                rows.append(
                    {
                        "pk_id": pk_id, "source": source, "frame_idx": idx,
                        "quad_tl_x": float(q[0][0]), "quad_tl_y": float(q[0][1]),
                        "quad_tr_x": float(q[1][0]), "quad_tr_y": float(q[1][1]),
                        "quad_br_x": float(q[2][0]), "quad_br_y": float(q[2][1]),
                        "quad_bl_x": float(q[3][0]), "quad_bl_y": float(q[3][1]),
                        **g, "is_missing": False, "missing_reason": None, **prov,
                    }
                )
            idx += 1
        cap.release()
        return pd.DataFrame(rows)

    # ----------------------------------------------------------------- events

    def _contact_frame(self, balls, geometry, fps) -> dict:
        """Ball contact = the frame the ball leaves the penalty spot.

        The obvious rule -- wait for ball speed to rise -- does not work, and the
        reason is worth recording. A penalty ball is stationary on the spot,
        which a detector tracks very well; the instant it is struck it crosses
        tens of pixels per frame and the tracker *loses* it. On real footage the
        stationary track simply ends at contact and speed never rises at all.

        So contact is detected as departure rather than acceleration:

        1. find the ball track that holds still longest -- that is the ball on
           the spot, not a ball in the crowd or a second ball on the touchline;
        2. contact is the first frame it moves further than ``spot_tol`` from
           that spot, or, if the track simply ends, the frame after its last
           stationary observation.

        Both branches are reported through ``method`` so a consumer can tell a
        seen departure from an inferred one.
        """
        miss = {"frame_idx": None, "confidence": None, "method": "stationary_ball_departure"}
        if balls is None or not len(balls):
            return {**miss, "reason": "no_ball_detections"}

        scale = _median_goal_width(geometry)
        if scale is None:
            return {**miss, "reason": "no_goal_geometry_for_scale"}
        tol = self.cfg.spot_tol_goal_widths * scale
        min_still = max(int(self.cfg.min_stationary_s * fps), 5)

        # Score each track by how long it stays inside a tol-sized box.
        best = None
        for tid, g in balls.groupby("track_id"):
            g = g.sort_values("frame_idx")
            if len(g) < min_still:
                continue
            spot_x = float(g["cx"].iloc[: max(len(g) // 2, 3)].median())
            spot_y = float(g["cy"].iloc[: max(len(g) // 2, 3)].median())
            near = (np.hypot(g["cx"] - spot_x, g["cy"] - spot_y) <= tol).to_numpy()
            run = 0
            for v in near:
                if not v:
                    break
                run += 1
            if run >= min_still and (best is None or run > best["run"]):
                best = {"tid": tid, "g": g, "spot": (spot_x, spot_y), "near": near, "run": run}

        if best is None:
            return {**miss, "reason": "no_stationary_ball_track_found"}

        g, near, run = best["g"], best["near"], best["run"]
        frames = g["frame_idx"].to_numpy(int)

        if run < len(near):
            # The ball was still seen moving away: departure is observed.
            return {
                "frame_idx": int(frames[run]),
                "confidence": float(np.clip(run / (self.cfg.min_stationary_s * fps * 2), 0.3, 0.9)),
                "method": "stationary_ball_departure_observed",
                "reason": None,
            }

        # The track ends while still on the spot: the tracker lost the struck
        # ball. Contact is the next frame, and confidence is lower because the
        # departure itself was never seen.
        return {
            "frame_idx": int(frames[-1]) + 1,
            "confidence": 0.45,
            "method": "stationary_ball_track_ends_on_spot",
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

        # Keeper commit: the first decisive lateral movement, searched only in a
        # window around contact. Searched over the whole clip it fires on the
        # keeper shuffling along his line a full four seconds early, which is
        # not a commitment to a side.
        commit = {"frame_idx": None, "confidence": None, "reason": "no_keeper_track"}
        if keeper_id is not None:
            kp = persons[persons["track_id"] == keeper_id].sort_values("frame_idx")
            if contact["frame_idx"] is None:
                commit["reason"] = "no_contact_anchor_to_window_around"
            elif len(kp) < 4:
                commit["reason"] = "keeper_track_too_short"
            else:
                lo = contact["frame_idx"] - self.cfg.commit_window_before_s * fps
                hi = contact["frame_idx"] + self.cfg.commit_window_after_s * fps
                w = kp[(kp["frame_idx"] >= lo) & (kp["frame_idx"] <= hi)]
                if len(w) < 4:
                    commit["reason"] = "no_keeper_observations_near_contact"
                else:
                    t = w["frame_idx"].to_numpy(float) / fps
                    vx, _ = _central_diff(w["cx"].to_numpy(float), w["cy"].to_numpy(float), t)
                    scale = max(float(w["h"].median()), 1.0)
                    lat = np.abs(vx) / scale
                    idx = np.where(lat > self.cfg.commit_lateral_speed)[0]
                    if len(idx):
                        i = int(idx[0])
                        commit = {
                            "frame_idx": int(w["frame_idx"].iloc[i]),
                            "confidence": float(np.clip(lat[i] / self.cfg.commit_lateral_speed / 4, 0.15, 0.9)),
                            "reason": None,
                        }
                    else:
                        commit["reason"] = "keeper_lateral_speed_never_exceeded_threshold"
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


def _goal_centre(geometry: pd.DataFrame | None, upto_frame: int | None = None):
    """Median goal-mouth centre and width over the frames where it was found."""
    if geometry is None or not len(geometry):
        return None
    g = geometry[~geometry["is_missing"].astype(bool)]
    if upto_frame is not None:
        pre = g[g["frame_idx"] <= upto_frame]
        g = pre if len(pre) else g
    if not len(g):
        return None
    cx = float(np.nanmedian((g["quad_tl_x"] + g["quad_tr_x"]) / 2))
    cy = float(np.nanmedian((g["quad_tl_y"] + g["quad_bl_y"]) / 2))
    gw = float(np.nanmedian(g["goal_width_px"]))
    if not np.isfinite(cx) or not np.isfinite(gw) or gw <= 0:
        return None
    return cx, cy, gw


def _median_goal_width(geometry: pd.DataFrame | None) -> float | None:
    if geometry is None or not len(geometry):
        return None
    g = geometry[~geometry["is_missing"].astype(bool)]
    if not len(g):
        return None
    gw = float(np.nanmedian(g["goal_width_px"]))
    return gw if np.isfinite(gw) and gw > 0 else None


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
