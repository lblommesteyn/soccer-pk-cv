"""Shootout splitting.

The splitter turns one long recording into many penalties, so a false episode
becomes a fabricated penalty in the corpus. These tests pin the guards that
prevent that.
"""

import pandas as pd

from pkcv.vision.shootout import EpisodeConfig, find_episodes

GOAL_W = 600.0
FPS = 50.0


def _ball(rows):
    return pd.DataFrame(rows)


def _still(track, f0, n, x=500.0, y=600.0):
    return [{"frame_idx": f0 + i, "track_id": track, "cx": x, "cy": y} for i in range(n)]


def _fly(track, f0, n, x=500.0, y=600.0, dx=90.0):
    return [
        {"frame_idx": f0 + i, "track_id": track, "cx": x + dx * (i + 1), "cy": y - 20.0 * (i + 1)}
        for i in range(n)
    ]


def test_finds_each_kick_in_a_sequence():
    rows = []
    for k in range(4):
        base = k * 600  # 12 s apart
        rows += _still(f"t{k}", base, 100)
        rows += _fly(f"t{k}", base + 100, 8)
    eps = find_episodes(_ball(rows), GOAL_W, FPS)
    assert len(eps) == 4
    assert [e.contact_frame for e in eps] == [100, 700, 1300, 1900]
    assert all(e.departed for e in eps)
    assert all(e.confidence > 0.5 for e in eps)


def test_a_ball_that_never_departs_is_not_a_kick():
    """A ball resting during a stoppage sits still too."""
    rows = _still("t0", 0, 400)
    eps = find_episodes(_ball(rows), GOAL_W, FPS)
    # The run ends with the track ending, which is consistent with a kick the
    # tracker lost -- so it is reported, but at reduced confidence.
    assert len(eps) == 1
    assert eps[0].departed is False
    assert eps[0].confidence < 0.5


def test_one_kick_split_across_two_track_ids_is_counted_once():
    rows = _still("a", 0, 100) + _still("b", 100, 60) + _fly("b", 160, 8)
    eps = find_episodes(_ball(rows), GOAL_W, FPS, EpisodeConfig(min_gap_s=4.0))
    assert len(eps) == 1, "the same kick must not be counted twice"
    assert eps[0].departed is True


def test_a_briefly_still_ball_is_ignored():
    rows = _still("t0", 0, 10) + _fly("t0", 10, 20)  # 0.2 s still: not a placed penalty
    assert find_episodes(_ball(rows), GOAL_W, FPS) == []


def test_drift_within_tolerance_still_counts_as_stationary():
    rows = [
        {"frame_idx": i, "track_id": "t0", "cx": 500.0 + 0.05 * i, "cy": 600.0}
        for i in range(120)
    ] + _fly("t0", 120, 8, x=506.0)
    eps = find_episodes(_ball(rows), GOAL_W, FPS)
    assert len(eps) == 1 and eps[0].departed


def test_no_geometry_scale_means_no_episodes():
    rows = _still("t0", 0, 200) + _fly("t0", 200, 8)
    assert find_episodes(_ball(rows), 0.0, FPS) == []
    assert find_episodes(_ball(rows), None, FPS) == []


def test_window_is_clipped_to_the_recording():
    rows = _still("t0", 0, 100) + _fly("t0", 100, 8)
    ep = find_episodes(_ball(rows), GOAL_W, FPS)[0]
    lo, hi = ep.window(FPS, n_frames=150, cfg=EpisodeConfig(pre_roll_s=4.0, post_roll_s=2.5))
    assert lo == 0, "cannot start before the recording"
    assert hi == 149, "cannot run past the end"
