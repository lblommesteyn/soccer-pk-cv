"""Contact detection and goal geometry.

These pin the two behaviours that were wrong on real footage and would silently
come back: a penalty ball's track *ends* at contact rather than accelerating,
and a goal is a quadrilateral rather than a bounding box.
"""

import numpy as np
import pandas as pd
import pytest

from pkcv.vision.runner import ClipProcessor, VisionConfig


def _geometry(width=600.0, n=10):
    return pd.DataFrame(
        [{"frame_idx": i, "goal_width_px": width, "is_missing": False,
          "quad_tl_x": 100.0, "quad_tr_x": 700.0, "quad_tl_y": 50.0, "quad_bl_y": 300.0}
         for i in range(n)]
    )


def _balls(frames, xs, ys, track_id=1):
    return pd.DataFrame(
        [{"frame_idx": f, "track_id": track_id, "cx": x, "cy": y, "w": 20.0, "h": 20.0,
          "conf": 0.9} for f, x, y in zip(frames, xs, ys, strict=False)]
    )


@pytest.fixture
def proc():
    return ClipProcessor(VisionConfig())


def test_contact_when_the_track_simply_ends_on_the_spot(proc):
    """The real-footage case: the tracker loses the struck ball entirely."""
    frames = list(range(0, 60))
    xs = [500.0 + 0.02 * f for f in frames]  # sub-pixel camera drift only
    ys = [600.0] * len(frames)
    out = proc._contact_frame(_balls(frames, xs, ys), _geometry(), fps=50.0)
    assert out["frame_idx"] == 60, "contact is the frame after the last stationary observation"
    assert out["method"] == "stationary_ball_track_ends_on_spot"
    assert 0 < out["confidence"] < 1


def test_contact_when_departure_is_actually_observed(proc):
    frames = list(range(0, 70))
    xs, ys = [], []
    for f in frames:
        if f < 50:
            xs.append(500.0)
            ys.append(600.0)
        else:
            xs.append(500.0 + 80.0 * (f - 49))  # struck: leaves fast
            ys.append(600.0 - 20.0 * (f - 49))
    out = proc._contact_frame(_balls(frames, xs, ys), _geometry(), fps=50.0)
    assert out["frame_idx"] == 50
    assert out["method"] == "stationary_ball_departure_observed"


def test_a_ball_that_never_settles_yields_no_contact(proc):
    frames = list(range(0, 60))
    xs = [500.0 + 40.0 * f for f in frames]
    ys = [600.0] * len(frames)
    out = proc._contact_frame(_balls(frames, xs, ys), _geometry(), fps=50.0)
    assert out["frame_idx"] is None
    assert out["reason"] == "no_stationary_ball_track_found"


def test_contact_needs_goal_geometry_for_scale(proc):
    frames = list(range(0, 60))
    out = proc._contact_frame(
        _balls(frames, [500.0] * 60, [600.0] * 60),
        pd.DataFrame([{"frame_idx": 0, "goal_width_px": None, "is_missing": True}]),
        fps=50.0,
    )
    assert out["frame_idx"] is None
    assert out["reason"] == "no_goal_geometry_for_scale"


def test_no_ball_detections_is_reported_not_guessed(proc):
    out = proc._contact_frame(pd.DataFrame(), _geometry(), fps=50.0)
    assert out["frame_idx"] is None
    assert out["reason"] == "no_ball_detections"


def test_the_stationary_ball_wins_over_a_longer_moving_track(proc):
    """A ball in the crowd can be tracked longer than the one on the spot."""
    still = _balls(list(range(0, 40)), [500.0] * 40, [600.0] * 40, track_id=1)
    moving = _balls(
        list(range(0, 80)), [100.0 + 30 * f for f in range(80)], [200.0] * 80, track_id=2
    )
    out = proc._contact_frame(pd.concat([still, moving], ignore_index=True), _geometry(), fps=50.0)
    assert out["frame_idx"] == 40, "must anchor on the ball that was on the spot"


# ------------------------------------------------------------------ geometry


def _goal_image(w=960, h=540, left=300, right=700, top=150, base=380):
    """Synthesise a goal: two white posts, a white crossbar, and a net mesh."""
    img = np.full((h, w, 3), 40, np.uint8)
    img[:, :, 1] = 90  # greenish pitch
    white = (250, 250, 250)
    for x in (left, right):
        img[top:base, x - 4:x + 4] = white
    img[top - 4:top + 4, left:right] = white
    # net: a mesh is what distinguishes a goal from any other rectangle
    for x in range(left, right, 8):
        img[top:base, x:x + 1] = (200, 200, 200)
    for y in range(top, base, 8):
        img[y:y + 1, left:right] = (200, 200, 200)
    return img


def test_goal_is_returned_as_four_corners():
    from pkcv.vision.geometry import find_goal

    g = find_goal(_goal_image())
    assert g is not None, "a synthetic goal with posts, bar and net must be found"
    quad = g["quad"]
    assert quad.shape == (4, 2)
    assert g["post_left_x"] == pytest.approx(300, abs=15)
    assert g["post_right_x"] == pytest.approx(700, abs=15)
    assert g["goal_width_px"] == pytest.approx(400, abs=30)
    # 7.32 m across the mouth is the only real-world scale a single view gives.
    assert g["px_per_m"] == pytest.approx(400 / 7.32, rel=0.15)


def test_a_smooth_rectangle_is_rejected_as_not_a_net():
    """Broadcast graphics and wall panels form the same shape without a mesh."""
    from pkcv.vision.geometry import find_goal

    img = np.full((540, 960, 3), 40, np.uint8)
    white = (250, 250, 250)
    for x in (300, 700):
        img[150:380, x - 4:x + 4] = white
    img[146:154, 300:700] = white
    assert find_goal(img) is None


def test_goal_coordinates_are_in_original_pixels_after_downscaling():
    from pkcv.vision.geometry import GoalConfig, find_goal

    big = _goal_image(w=1920, h=1080, left=600, right=1400, top=300, base=760)
    g = find_goal(big, GoalConfig(work_width=960))
    assert g is not None
    assert g["post_left_x"] == pytest.approx(600, abs=30)
    assert g["post_right_x"] == pytest.approx(1400, abs=30)
