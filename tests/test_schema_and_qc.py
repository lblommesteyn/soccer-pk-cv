"""Schema contracts and QC grading.

The schema tests exist because the whole "never fabricate a failed estimate"
guarantee is enforced structurally: if a column that can fail is ever declared
non-nullable, the pipeline is forced to invent a value to satisfy the writer.
"""

import pandas as pd
import pyarrow as pa
import pytest

from pkcv.io import coerce, upsert_parquet, write_parquet
from pkcv.qc import checks
from pkcv.schemas import ARTIFACT_SCHEMAS, METADATA_SCHEMA, PROCESSING_VERSION

#: Columns that describe an estimate. Every one must accept null.
ESTIMATE_COLUMNS = {
    "x", "y", "x_c", "y_c", "x_n", "y_n", "confidence",
    "bbox_cx", "bbox_cy", "bbox_w", "bbox_h",
    "vx_px_s", "vy_px_s", "frame_idx_estimate",
    "goal_width_px", "goal_height_px", "px_per_m",
}


@pytest.mark.parametrize("artifact", sorted(ARTIFACT_SCHEMAS))
def test_estimate_columns_are_nullable(artifact):
    for field in ARTIFACT_SCHEMAS[artifact]:
        if field.name in ESTIMATE_COLUMNS:
            assert field.nullable, f"{artifact}.{field.name} must be nullable"


@pytest.mark.parametrize("artifact", sorted(ARTIFACT_SCHEMAS))
def test_every_artifact_carries_pk_id_and_provenance(artifact):
    names = {f.name for f in ARTIFACT_SCHEMAS[artifact]}
    assert "pk_id" in names
    assert "processing_version" in names
    if artifact != "qc":
        assert {"derivation", "provenance"} <= names


@pytest.mark.parametrize("artifact", sorted(ARTIFACT_SCHEMAS))
def test_missingness_columns_pair_up(artifact):
    names = {f.name for f in ARTIFACT_SCHEMAS[artifact]}
    if "is_missing" in names:
        assert "missing_reason" in names, f"{artifact} can flag missing but cannot explain it"


def test_coerce_rejects_a_missing_required_column():
    with pytest.raises(ValueError, match="pk_id"):
        coerce(pd.DataFrame({"source": ["x"]}), METADATA_SCHEMA)


def test_coerce_fills_absent_nullable_columns_with_null():
    row = {f.name: None for f in METADATA_SCHEMA if not f.nullable}
    row.update(
        pk_id="a:1", pk_uid="0" * 16, source="a", source_identifier="1", dedup_key="f/1",
        is_primary=True, media_kind="pose_table", has_video=False, label_provenance="src",
        license="CC BY 4.0", attribution="x", redistribute_derived=True, redistribute_video=False,
        ingested_at_utc="now", qc_status="pending", processing_version=PROCESSING_VERSION,
        derivation="source_provided", provenance="p",
    )
    table = coerce(pd.DataFrame([row]), METADATA_SCHEMA)
    assert table.num_rows == 1
    assert pa.compute.is_null(table["fps"])[0].as_py() is True


def test_upsert_is_idempotent_and_scoped_to_its_keys(tmp_path):
    path = tmp_path / "qc.parquet"
    a = pd.DataFrame(
        [{"pk_id": "a:1", "source": "s", "check": "c", "status": "pass",
          "processing_version": PROCESSING_VERSION}]
    )
    b = pd.DataFrame(
        [{"pk_id": "b:1", "source": "s", "check": "c", "status": "fail",
          "processing_version": PROCESSING_VERSION}]
    )
    write_parquet(a, path, "qc")
    upsert_parquet(b, path, "qc", ["pk_id"])
    upsert_parquet(b, path, "qc", ["pk_id"])  # replayed: must not duplicate
    out = pd.read_parquet(path)
    assert sorted(out["pk_id"]) == ["a:1", "b:1"]

    a2 = a.copy()
    a2["status"] = "warn"
    upsert_parquet(a2, path, "qc", ["pk_id"])
    out = pd.read_parquet(path)
    assert len(out) == 2
    assert out.set_index("pk_id").loc["a:1", "status"] == "warn"
    assert out.set_index("pk_id").loc["b:1", "status"] == "fail"


# ------------------------------------------------------------------ QC


def _meta(media="pose_table", **kw):
    base = {
        "pk_id": "t:1", "source": "t", "media_kind": media, "n_frames": 20, "fps": 25.0,
        "label_kick_direction": "L", "label_goal": 1, "label_footedness": "R",
    }
    base.update(kw)
    return pd.Series(base)


def _frames_df(n=20, missing=False):
    return pd.DataFrame(
        [{"pk_id": "t:1", "role": "kicker", "frame_idx": i, "track_id": "a",
          "is_missing": missing} for i in range(n)]
    )


def _poses_df(n=20, missing=False):
    return pd.DataFrame(
        [{"pk_id": "t:1", "role": "kicker", "frame_idx": i, "kp_index": k, "is_missing": missing}
         for i in range(n) for k in range(12)]
    )


def _events_df(missing=False, method="source_last_touch"):
    return pd.DataFrame(
        [{"pk_id": "t:1", "event_name": "ball_contact", "is_missing": missing,
          "confidence": None if missing else 1.0, "method": method,
          "missing_reason": "source_has_no_last_touch_marker" if missing else None}]
    )


def _snaps_df(available=True):
    return pd.DataFrame(
        [{"pk_id": "t:1", "offset_ms": o, "snapshot_available": available,
          "unavailable_reason": None if available else "before_clip_start"}
         for o in (-2000, -1500, -1000, -750, -500, -250, 0)]
    )


def _status(rows, check):
    return next(r["status"] for r in rows if r["check"] == check)


def test_pose_table_sources_mark_ball_and_keeper_not_applicable():
    rows = checks.check_kick(
        _meta(), _frames_df(), _poses_df(), pd.DataFrame(), _events_df(), _snaps_df()
    )
    # A licensing fact must not be graded as a pipeline failure.
    assert _status(rows, "ball_track") == checks.NA
    assert _status(rows, "keeper_track") == checks.NA
    assert _status(rows, "contact_anchor") == checks.PASS


def test_video_sources_fail_when_the_ball_is_never_found():
    rows = checks.check_kick(
        _meta(media="video"), _frames_df(), _poses_df(), pd.DataFrame(), _events_df(), _snaps_df()
    )
    assert _status(rows, "ball_track") == checks.FAIL


def test_a_missing_contact_anchor_fails():
    rows = checks.check_kick(
        _meta(), _frames_df(), _poses_df(), pd.DataFrame(), _events_df(missing=True), _snaps_df()
    )
    assert _status(rows, "contact_anchor") == checks.FAIL


def test_an_ambiguous_contact_anchor_only_warns():
    rows = checks.check_kick(
        _meta(), _frames_df(), _poses_df(), pd.DataFrame(),
        _events_df(method="source_last_touch_latest_of_multiple"), _snaps_df()
    )
    assert _status(rows, "contact_anchor") == checks.WARN


def test_rollup_takes_the_worst_status():
    rows = checks.check_kick(
        _meta(), _frames_df(), _poses_df(missing=True), pd.DataFrame(),
        _events_df(missing=True), _snaps_df(available=False)
    )
    roll = checks.rollup(pd.DataFrame(rows))
    assert roll["qc_status"].iloc[0] == checks.FAIL
    assert "contact_anchor:fail" in roll["qc_reasons"].iloc[0]
