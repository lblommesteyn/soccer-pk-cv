"""Canonical PKCV schemas.

Every derived artifact in this project carries the same provenance spine:

    pk_id, source, source_identifier, processing_version,
    model_name, model_version, provenance, confidence / qc_status

The rule this schema enforces structurally: **a missing estimate is a row with
``is_missing = True`` and a ``missing_reason``, never an invented number.**
Nullable float columns are used everywhere an estimate can fail.
"""

from __future__ import annotations

import pyarrow as pa

# --------------------------------------------------------------------------
# versions
# --------------------------------------------------------------------------

#: Bump when any derived artifact's *meaning* changes. Written into every row.
PROCESSING_VERSION = "0.1.0"

#: Canonical snapshot offsets relative to estimated ball contact (t=0).
SNAPSHOT_OFFSETS_MS: tuple[int, ...] = (-2000, -1500, -1000, -750, -500, -250, 0)

#: COCO-17 keypoint indices retained by the upstream Mendeley datasets and by
#: our own pose stage. Indices 0-4 (face) are dropped: they carry no
#: biomechanical signal for kick direction and are the least reliable on
#: broadcast-resolution crops.
KEYPOINT_INDICES: tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

KEYPOINT_NAMES: dict[int, str] = {
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

ROLES: tuple[str, ...] = ("kicker", "keeper")

EVENT_NAMES: tuple[str, ...] = ("ball_contact", "keeper_commit", "plant_foot")

#: How an artifact came to exist. Distinguishing these two is the whole point:
#: we must never present an upstream depositor's numbers as our own inference,
#: nor our inference as ground truth.
DERIVATIONS: tuple[str, ...] = (
    "source_provided",  # shipped by the upstream dataset, copied verbatim
    "pkcv_derived",  # computed by this pipeline
    "pkcv_interpolated",  # computed by this pipeline by interpolating between real frames
)

# --------------------------------------------------------------------------
# shared provenance columns
# --------------------------------------------------------------------------

_PROVENANCE_FIELDS = [
    pa.field("processing_version", pa.string(), nullable=False),
    pa.field("derivation", pa.string(), nullable=False),
    pa.field("model_name", pa.string(), nullable=True),
    pa.field("model_version", pa.string(), nullable=True),
    pa.field("provenance", pa.string(), nullable=False),
]


def _with_provenance(fields: list[pa.Field]) -> pa.Schema:
    return pa.schema(fields + _PROVENANCE_FIELDS)


# --------------------------------------------------------------------------
# metadata.parquet -- one row per penalty kick
# --------------------------------------------------------------------------

METADATA_SCHEMA = _with_provenance(
    [
        # identity
        pa.field("pk_id", pa.string(), nullable=False),
        pa.field("pk_uid", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_identifier", pa.string(), nullable=False),
        pa.field("source_dataset_doi", pa.string(), nullable=True),
        pa.field("source_version", pa.string(), nullable=True),
        # deduplication
        pa.field("dedup_key", pa.string(), nullable=False),
        pa.field("is_primary", pa.bool_(), nullable=False),
        pa.field("duplicate_of_pk_id", pa.string(), nullable=True),
        pa.field("duplicate_evidence", pa.string(), nullable=True),
        # media
        pa.field("media_kind", pa.string(), nullable=False),  # video | pose_table | render_only | none
        pa.field("has_video", pa.bool_(), nullable=False),
        pa.field("video_relpath", pa.string(), nullable=True),
        pa.field("video_sha256", pa.string(), nullable=True),
        pa.field("fps", pa.float64(), nullable=True),
        pa.field("n_frames", pa.int32(), nullable=True),
        pa.field("frame_width", pa.int32(), nullable=True),
        pa.field("frame_height", pa.int32(), nullable=True),
        pa.field("clip_duration_s", pa.float64(), nullable=True),
        # context
        pa.field("competition", pa.string(), nullable=True),
        pa.field("gender", pa.string(), nullable=True),
        pa.field("level", pa.string(), nullable=True),
        pa.field("season", pa.string(), nullable=True),
        pa.field("match_context", pa.string(), nullable=True),  # game | training | shootout
        pa.field("kicker_ref", pa.string(), nullable=True),
        # ground-truth labels (copied verbatim from source; never inferred)
        pa.field("label_kick_direction", pa.string(), nullable=True),  # L | C | R
        pa.field("label_keeper_direction", pa.string(), nullable=True),
        pa.field("label_outcome", pa.string(), nullable=True),  # goal | save | miss | post
        pa.field("label_goal", pa.int8(), nullable=True),
        pa.field("label_footedness", pa.string(), nullable=True),  # L | R
        pa.field("label_camera_direction", pa.string(), nullable=True),  # L | C | R
        pa.field("label_provenance", pa.string(), nullable=False),
        # licensing
        pa.field("license", pa.string(), nullable=False),
        pa.field("license_url", pa.string(), nullable=True),
        pa.field("attribution", pa.string(), nullable=False),
        pa.field("redistribute_derived", pa.bool_(), nullable=False),
        pa.field("redistribute_video", pa.bool_(), nullable=False),
        pa.field("redistribution_note", pa.string(), nullable=True),
        # pipeline state
        pa.field("ingested_at_utc", pa.string(), nullable=False),
        pa.field("qc_status", pa.string(), nullable=False),  # pass | warn | fail | pending
        pa.field("qc_reasons", pa.string(), nullable=True),
    ]
)

# --------------------------------------------------------------------------
# tracks.parquet -- frame-level box tracks per role
# --------------------------------------------------------------------------

TRACKS_SCHEMA = _with_provenance(
    [
        pa.field("pk_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_identifier", pa.string(), nullable=False),
        pa.field("role", pa.string(), nullable=False),
        pa.field("track_id", pa.string(), nullable=True),
        pa.field("frame_idx", pa.int32(), nullable=False),
        pa.field("t_s", pa.float64(), nullable=True),
        pa.field("t_ms_rel_contact", pa.float64(), nullable=True),
        pa.field("bbox_cx", pa.float64(), nullable=True),
        pa.field("bbox_cy", pa.float64(), nullable=True),
        pa.field("bbox_w", pa.float64(), nullable=True),
        pa.field("bbox_h", pa.float64(), nullable=True),
        pa.field("vx_px_s", pa.float64(), nullable=True),
        pa.field("vy_px_s", pa.float64(), nullable=True),
        pa.field("confidence", pa.float64(), nullable=True),
        pa.field("is_missing", pa.bool_(), nullable=False),
        pa.field("missing_reason", pa.string(), nullable=True),
    ]
)

# --------------------------------------------------------------------------
# poses.parquet -- long format, one row per (pk, role, frame, keypoint)
# --------------------------------------------------------------------------

POSES_SCHEMA = _with_provenance(
    [
        pa.field("pk_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("role", pa.string(), nullable=False),
        pa.field("frame_idx", pa.int32(), nullable=False),
        pa.field("t_s", pa.float64(), nullable=True),
        pa.field("t_ms_rel_contact", pa.float64(), nullable=True),
        pa.field("kp_index", pa.int8(), nullable=False),
        pa.field("kp_name", pa.string(), nullable=False),
        pa.field("x", pa.float64(), nullable=True),  # image pixels, origin top-left
        pa.field("y", pa.float64(), nullable=True),
        pa.field("x_c", pa.float64(), nullable=True),  # box-centred pixels
        pa.field("y_c", pa.float64(), nullable=True),
        pa.field("x_n", pa.float64(), nullable=True),  # box-centred, /bbox_h, camera-flipped
        pa.field("y_n", pa.float64(), nullable=True),
        pa.field("confidence", pa.float64(), nullable=True),
        pa.field("is_missing", pa.bool_(), nullable=False),
        pa.field("missing_reason", pa.string(), nullable=True),
    ]
)

# --------------------------------------------------------------------------
# ball.parquet
# --------------------------------------------------------------------------

BALL_SCHEMA = _with_provenance(
    [
        pa.field("pk_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("frame_idx", pa.int32(), nullable=False),
        pa.field("t_s", pa.float64(), nullable=True),
        pa.field("t_ms_rel_contact", pa.float64(), nullable=True),
        pa.field("x", pa.float64(), nullable=True),
        pa.field("y", pa.float64(), nullable=True),
        pa.field("bbox_w", pa.float64(), nullable=True),
        pa.field("bbox_h", pa.float64(), nullable=True),
        pa.field("vx_px_s", pa.float64(), nullable=True),
        pa.field("vy_px_s", pa.float64(), nullable=True),
        pa.field("confidence", pa.float64(), nullable=True),
        pa.field("is_missing", pa.bool_(), nullable=False),
        pa.field("missing_reason", pa.string(), nullable=True),
    ]
)

# --------------------------------------------------------------------------
# geometry.parquet -- goal-frame geometry, one row per pk (may be all-null)
# --------------------------------------------------------------------------

GEOMETRY_SCHEMA = _with_provenance(
    [
        pa.field("pk_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("frame_idx", pa.int32(), nullable=True),
        pa.field("post_left_x", pa.float64(), nullable=True),
        pa.field("post_left_y", pa.float64(), nullable=True),
        pa.field("post_right_x", pa.float64(), nullable=True),
        pa.field("post_right_y", pa.float64(), nullable=True),
        pa.field("crossbar_y", pa.float64(), nullable=True),
        # Full quadrilateral, in image pixels, ordered
        # top-left, top-right, bottom-right, bottom-left. A goal filmed from the
        # stand is a trapezoid, not an axis-aligned box, so the corners are kept
        # rather than a bounding rectangle.
        pa.field("quad_tl_x", pa.float64(), nullable=True),
        pa.field("quad_tl_y", pa.float64(), nullable=True),
        pa.field("quad_tr_x", pa.float64(), nullable=True),
        pa.field("quad_tr_y", pa.float64(), nullable=True),
        pa.field("quad_br_x", pa.float64(), nullable=True),
        pa.field("quad_br_y", pa.float64(), nullable=True),
        pa.field("quad_bl_x", pa.float64(), nullable=True),
        pa.field("quad_bl_y", pa.float64(), nullable=True),
        pa.field("net_edge_density", pa.float64(), nullable=True),
        pa.field("goal_width_px", pa.float64(), nullable=True),
        pa.field("goal_height_px", pa.float64(), nullable=True),
        pa.field("px_per_m", pa.float64(), nullable=True),
        pa.field("confidence", pa.float64(), nullable=True),
        pa.field("is_missing", pa.bool_(), nullable=False),
        pa.field("missing_reason", pa.string(), nullable=True),
    ]
)

# --------------------------------------------------------------------------
# events.parquet -- one row per (pk, event)
# --------------------------------------------------------------------------

EVENTS_SCHEMA = _with_provenance(
    [
        pa.field("pk_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("event_name", pa.string(), nullable=False),
        pa.field("frame_idx", pa.int32(), nullable=True),
        pa.field("t_s", pa.float64(), nullable=True),
        pa.field("confidence", pa.float64(), nullable=True),
        pa.field("method", pa.string(), nullable=False),
        pa.field("is_missing", pa.bool_(), nullable=False),
        pa.field("missing_reason", pa.string(), nullable=True),
    ]
)

# --------------------------------------------------------------------------
# temporal_frames.parquet -- fused per-frame observation
# --------------------------------------------------------------------------

TEMPORAL_FRAMES_SCHEMA = _with_provenance(
    [
        pa.field("pk_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("frame_idx", pa.int32(), nullable=False),
        pa.field("t_s", pa.float64(), nullable=False),
        pa.field("t_ms_rel_contact", pa.float64(), nullable=False),
        # kicker
        pa.field("kicker_cx", pa.float64(), nullable=True),
        pa.field("kicker_cy", pa.float64(), nullable=True),
        pa.field("kicker_h", pa.float64(), nullable=True),
        pa.field("kicker_vx", pa.float64(), nullable=True),
        pa.field("kicker_vy", pa.float64(), nullable=True),
        pa.field("kicker_speed", pa.float64(), nullable=True),
        pa.field("kicker_available", pa.bool_(), nullable=False),
        # kicker pose-derived scalars (see temporal/features.py for definitions)
        pa.field("hip_shoulder_angle_deg", pa.float64(), nullable=True),
        pa.field("pelvis_orientation_deg", pa.float64(), nullable=True),
        pa.field("ankle_separation_n", pa.float64(), nullable=True),
        pa.field("plant_ankle_x_n", pa.float64(), nullable=True),
        pa.field("kick_ankle_x_n", pa.float64(), nullable=True),
        pa.field("kick_ankle_vx_n", pa.float64(), nullable=True),
        pa.field("torso_lean_deg", pa.float64(), nullable=True),
        pa.field("pose_n_visible_kp", pa.int8(), nullable=True),
        pa.field("pose_available", pa.bool_(), nullable=False),
        # keeper
        pa.field("keeper_cx", pa.float64(), nullable=True),
        pa.field("keeper_cy", pa.float64(), nullable=True),
        pa.field("keeper_vx", pa.float64(), nullable=True),
        pa.field("keeper_vy", pa.float64(), nullable=True),
        pa.field("keeper_available", pa.bool_(), nullable=False),
        # ball
        pa.field("ball_x", pa.float64(), nullable=True),
        pa.field("ball_y", pa.float64(), nullable=True),
        pa.field("ball_vx", pa.float64(), nullable=True),
        pa.field("ball_vy", pa.float64(), nullable=True),
        pa.field("ball_available", pa.bool_(), nullable=False),
        # interaction
        pa.field("kicker_ball_dist_n", pa.float64(), nullable=True),
        pa.field("keeper_goal_offset_n", pa.float64(), nullable=True),
        pa.field("is_observed_frame", pa.bool_(), nullable=False),
    ]
)

# --------------------------------------------------------------------------
# temporal_snapshots.parquet -- the canonical experiment table
# --------------------------------------------------------------------------

def _relaxed(schema: pa.Schema, keep_required: set[str]) -> list[pa.Field]:
    """Same fields, but only ``keep_required`` stay non-nullable.

    A snapshot row exists for every (kick, offset) even when the observation
    does not, so columns that are always present in an *observed* frame must be
    allowed to be null in an *unavailable* snapshot.
    """
    return [
        f if (f.name in keep_required or f.nullable) else pa.field(f.name, f.type, nullable=True)
        for f in schema
    ]


TEMPORAL_SNAPSHOTS_SCHEMA = pa.schema(
    _relaxed(
        TEMPORAL_FRAMES_SCHEMA,
        keep_required={
            "pk_id",
            "source",
            "t_ms_rel_contact",
            "processing_version",
            "derivation",
            "provenance",
        },
    )
    + [
        pa.field("offset_ms", pa.int32(), nullable=False),
        pa.field("snapshot_available", pa.bool_(), nullable=False),
        pa.field("snapshot_method", pa.string(), nullable=False),  # exact | interp | nearest | unavailable
        pa.field("gap_ms", pa.float64(), nullable=True),
        pa.field("unavailable_reason", pa.string(), nullable=True),
    ]
)

# --------------------------------------------------------------------------
# qc.parquet
# --------------------------------------------------------------------------

QC_SCHEMA = pa.schema(
    [
        pa.field("pk_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("check", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),  # pass | warn | fail | na
        pa.field("value", pa.float64(), nullable=True),
        pa.field("threshold", pa.float64(), nullable=True),
        pa.field("message", pa.string(), nullable=True),
        pa.field("processing_version", pa.string(), nullable=False),
    ]
)


ARTIFACT_SCHEMAS: dict[str, pa.Schema] = {
    "metadata": METADATA_SCHEMA,
    "tracks": TRACKS_SCHEMA,
    "poses": POSES_SCHEMA,
    "ball": BALL_SCHEMA,
    "geometry": GEOMETRY_SCHEMA,
    "events": EVENTS_SCHEMA,
    "temporal_frames": TEMPORAL_FRAMES_SCHEMA,
    "temporal_snapshots": TEMPORAL_SNAPSHOTS_SCHEMA,
    "qc": QC_SCHEMA,
}
