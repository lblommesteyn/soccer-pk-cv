"""The snapshot layer is where an honest dataset can quietly become a dishonest
one, so these tests pin the three behaviours that matter: a snapshot outside the
clip is reported unavailable rather than clamped, interpolation is refused
across a large gap, and an exact hit is never relabelled as interpolated.
"""

import numpy as np
import pandas as pd
import pytest

from pkcv.schemas import SNAPSHOT_OFFSETS_MS
from pkcv.temporal import build


def _meta(fps=25.0, foot="R"):
    return pd.Series(
        {
            "pk_id": "test:1",
            "source": "test",
            "fps": fps,
            "label_footedness": foot,
            "n_frames": 0,
        }
    )


def _tracks(frames, contact_frame, fps=25.0, track_id="t1"):
    rows = []
    for f in frames:
        rows.append(
            {
                "pk_id": "test:1",
                "source": "test",
                "role": "kicker",
                "track_id": track_id,
                "frame_idx": f,
                "t_s": f / fps,
                "t_ms_rel_contact": (f - contact_frame) / fps * 1000.0,
                "bbox_cx": 100.0 + f,
                "bbox_cy": 200.0,
                "bbox_w": 40.0,
                "bbox_h": 100.0,
                "vx_px_s": 1.0,
                "vy_px_s": 0.0,
                "confidence": None,
                "is_missing": False,
                "missing_reason": None,
            }
        )
    return pd.DataFrame(rows)


def _empty_poses():
    return pd.DataFrame(columns=["pk_id", "role", "frame_idx", "kp_index", "x_n", "y_n", "is_missing"])


def test_offsets_before_the_clip_are_unavailable_not_clamped():
    # 1.0 s of footage ending at contact: -2000 and -1500 ms cannot exist.
    frames = list(range(0, 26))
    f = build.build_frames(_meta(), _tracks(frames, contact_frame=25), _empty_poses())
    snaps = build.build_snapshots(f, _meta())

    assert len(snaps) == len(SNAPSHOT_OFFSETS_MS), "one row per offset, always"
    by_off = snaps.set_index("offset_ms")
    for off in (-2000, -1500):
        assert not by_off.loc[off, "snapshot_available"]
        assert by_off.loc[off, "unavailable_reason"] == "before_clip_start"
    for off in (-1000, -500, 0):
        assert by_off.loc[off, "snapshot_available"]
    # An unavailable snapshot must not smuggle in a value.
    assert pd.isna(by_off.loc[-2000, "kicker_cx"]) or by_off.loc[-2000, "kicker_cx"] is None


def test_exact_hits_are_labelled_exact():
    frames = list(range(0, 76))  # 3 s at 25 fps; every canonical offset lands on a frame
    meta = _meta()
    f = build.build_frames(meta, _tracks(frames, contact_frame=75), _empty_poses())
    snaps = build.build_snapshots(f, meta).set_index("offset_ms")
    for off in SNAPSHOT_OFFSETS_MS:
        assert snaps.loc[off, "snapshot_available"]
        assert snaps.loc[off, "snapshot_method"] == "exact"
        assert snaps.loc[off, "derivation"] == "pkcv_derived"


def test_interpolation_is_refused_across_a_large_gap():
    # A hole from -1200 ms to -300 ms: -750 ms sits inside it.
    frames = [f for f in range(0, 76) if not (45 <= f <= 67)]
    meta = _meta()
    f = build.build_frames(meta, _tracks(frames, contact_frame=75), _empty_poses())
    snaps = build.build_snapshots(f, meta, max_interp_gap_ms=120.0).set_index("offset_ms")
    assert not snaps.loc[-750, "snapshot_available"]
    assert snaps.loc[-750, "unavailable_reason"] == "gap_exceeds_interpolation_limit"
    assert snaps.loc[0, "snapshot_available"]


def test_interpolated_snapshots_are_marked_as_such():
    meta = _meta(fps=30.0)  # 33.3 ms frames: canonical offsets rarely land exactly
    frames = list(range(0, 91))
    f = build.build_frames(meta, _tracks(frames, contact_frame=90, fps=30.0), _empty_poses())
    snaps = build.build_snapshots(f, meta)
    interp = snaps[snaps["snapshot_method"] == "interp"]
    assert len(interp), "at least one offset should need interpolation at 30 fps"
    assert (interp["derivation"] == "pkcv_interpolated").all()


def test_fragmented_tracks_collapse_to_the_track_holding_contact():
    a = _tracks(list(range(0, 30)), contact_frame=75, track_id="fragment_a")
    b = _tracks(list(range(40, 76)), contact_frame=75, track_id="fragment_b")
    f = build.build_frames(_meta(), pd.concat([a, b], ignore_index=True), _empty_poses())
    # Only the fragment containing t=0 survives into the timeline.
    assert f["frame_idx"].min() >= 40
    assert len(f) == 36


def test_kicks_without_a_contact_anchor_produce_no_frames():
    t = _tracks(list(range(0, 20)), contact_frame=10)
    t["t_ms_rel_contact"] = np.nan
    f = build.build_frames(_meta(), t, _empty_poses())
    assert len(f) == 0
    snaps = build.build_snapshots(f, _meta())
    assert len(snaps) == len(SNAPSHOT_OFFSETS_MS)
    assert (~snaps["snapshot_available"]).all()
    assert (snaps["unavailable_reason"] == "no_anchored_frames").all()


@pytest.mark.parametrize("foot", ["R", "L"])
def test_features_pick_the_right_kicking_ankle(foot):
    from pkcv.temporal import features

    rows = []
    for f in range(3):
        for kp, x in [(15, -0.2), (16, 0.3), (5, -0.1), (6, 0.1), (11, -0.1), (12, 0.1)]:
            rows.append(
                {
                    "pk_id": "test:1",
                    "role": "kicker",
                    "frame_idx": f,
                    "kp_index": kp,
                    "x_n": x,
                    "y_n": 0.5,
                    "is_missing": False,
                }
            )
    out = features.compute(pd.DataFrame(rows), foot, fps=25.0)
    expected = 0.3 if foot == "R" else -0.2
    assert out["kick_ankle_x_n"].iloc[0] == pytest.approx(expected)


def test_features_are_null_when_footedness_is_unknown():
    from pkcv.temporal import features

    rows = [
        {"pk_id": "t", "role": "kicker", "frame_idx": 0, "kp_index": kp, "x_n": 0.1, "y_n": 0.2, "is_missing": False}
        for kp in (5, 6, 11, 12, 15, 16)
    ]
    out = features.compute(pd.DataFrame(rows), None, fps=25.0)
    assert out["kick_ankle_x_n"].isna().all()
