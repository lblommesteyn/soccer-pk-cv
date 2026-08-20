# Dataset schema

Canonical definitions live in [`pkcv/schemas/core.py`](../pkcv/schemas/core.py);
this document explains them. All tables are Parquet (zstd) and join on `pk_id`.

## Provenance spine

Every derived artifact carries the same five columns:

| column | meaning |
| --- | --- |
| `processing_version` | pipeline version that produced the row (`0.1.0`) |
| `derivation` | `source_provided`, `pkcv_derived`, or `pkcv_interpolated` |
| `model_name`, `model_version` | model that produced the estimate, if any |
| `provenance` | the exact origin, e.g. `mendeley:brx9bsxnpx/v1/kicker_pose_keypoints.csv` |

`derivation` is the load-bearing one. It keeps an upstream depositor's numbers
from being read as our inference, and our inference from being read as ground
truth.

## Missingness contract

Any column that holds an estimate is nullable, and every table that can report a
failure has both `is_missing` and `missing_reason`. **A value is never invented
to fill a hole.** `tests/test_schema_and_qc.py` enforces both properties, so the
guarantee cannot decay by accident.

Reasons currently emitted:

| reason | meaning |
| --- | --- |
| `source_provides_kicker_pose_only` | the deposit contains no keeper/ball/goal at all |
| `source_keypoint_absent` | upstream published no coordinate for this joint |
| `source_bbox_absent` | upstream published no box for this frame |
| `source_has_no_last_touch_marker` | contact frame unknown |
| `low_keypoint_confidence` | our pose model scored the joint below threshold |
| `no_ball_detections`, `no_track_assigned_role_keeper` | our vision stage found nothing |
| `goal_frame_not_detected`, `candidate_posts_too_close_to_be_a_goal` | goal geometry not confidently recovered |
| `video_gated_no_nda_password`, `video_gated_agreement_form_required` | licensing, not failure |

## Identifiers

`pk_id` is `<source_slug>:<normalised source identifier>` — readable and
deterministic, so re-ingesting produces byte-identical ids and downstream tables
stay joinable. `pk_uid` is its 16-hex digest, for filenames and partition keys.

`dedup_key` is `<deposit_family>/<normalised identifier>`. Records sharing one
are the same physical kick; the richest is elected `is_primary` and the rest
point at it via `duplicate_of_pk_id`. Nothing is deleted.

---

## `metadata.parquet` — one row per penalty

Identity (`pk_id`, `source`, `source_identifier`, `source_dataset_doi`),
deduplication (`dedup_key`, `is_primary`, `duplicate_of_pk_id`,
`duplicate_evidence`), media (`media_kind` ∈ `video | pose_table | render_only |
none`, `has_video`, `video_relpath`, `video_sha256`, `fps`, `n_frames`,
`frame_width`, `frame_height`, `clip_duration_s`), context (`competition`,
`gender`, `level`, `season`, `match_context`, `kicker_ref`), labels, licensing
and pipeline state (`ingested_at_utc`, `qc_status`, `qc_reasons`).

### Labels

`label_kick_direction` (L/C/R), `label_keeper_direction`, `label_outcome`,
`label_goal`, `label_footedness`, `label_camera_direction`. **All copied
verbatim from the source; none inferred.** `label_provenance` names the deposit.

### Licensing

`license`, `license_url`, `attribution`, and two independent flags:
`redistribute_derived` and `redistribute_video`. `pkcv publish` filters on them
per record and reports every withheld row in the upload manifest.

---

## `tracks.parquet` — one row per (kick, role, frame)

`role` ∈ `kicker | keeper`, plus `track_id`, `frame_idx`, `t_s`,
`t_ms_rel_contact`, `bbox_cx/cy/w/h`, `vx_px_s`, `vy_px_s`, `confidence`.

A role with no observations gets one row with `frame_idx = -1` and
`is_missing = True`, so "no keeper in this source" is distinguishable from
"keeper never looked for".

## `poses.parquet` — one row per (kick, role, frame, keypoint)

12 COCO joints (indices 5-16: shoulders through ankles). Face keypoints are
dropped: no biomechanical signal for kick direction and least reliable at
broadcast resolution.

Three coordinate systems: `x, y` (image pixels), `x_c, y_c` (box-centred), and
`x_n, y_n` — box-centred, divided by box height, with the horizontal axis
flipped for left-side cameras. Only `x_n, y_n` are comparable across clips;
without the flip, "leans right" means opposite things at opposite camera angles.

## `ball.parquet`

Ball position, box size and velocity per frame, with `is_missing` where the ball
was not detected. All-missing for the Mendeley pose-table records, populated for
video sources.

## `geometry.parquet` — one row **per frame**, not per kick

The goal is stored as a real quadrilateral: `quad_tl_*`, `quad_tr_*`,
`quad_br_*`, `quad_bl_*`, ordered top-left, top-right, bottom-right,
bottom-left, plus derived `post_left_x`, `post_right_x`, `crossbar_y`,
`goal_width_px`, `goal_height_px` and `net_edge_density`.

Per frame rather than per kick because this footage is handheld and pans: a
single quad drifts off the goal within a second. Frames where the goal was not
confidently found carry `is_missing` with a reason, so a consumer can see
exactly when the goal was in shot.

`px_per_m` comes from the 7.32 m goal width. **It fixes scale along the goal
line only.** It is not a homography and must not be used to measure distances
across the pitch.

Detection rate varies enormously with framing, and that variation is real data
about the clip rather than noise to be smoothed away. On a behind-the-goal
handheld clip the goal is located on 100% of frames; on a wide oblique stadium
view with the goal at the frame edge it drops below 5%. The `goal_geometry` QC
check records the per-clip rate so the usable subset can be selected on it.

## `events.parquet` — one row per (kick, event)

`ball_contact`, `keeper_commit`, `plant_foot`, each with `frame_idx`, `t_s`,
`confidence` and `method`. Methods in use:

| method | used when |
| --- | --- |
| `source_last_touch` | the deposit marks exactly one contact frame (confidence 1.0) |
| `source_last_touch_latest_of_multiple` | several markers from a fragmented track; latest taken (confidence 0.5) |
| `stationary_ball_departure_observed` | derived from video: the ball sat on the spot, then was seen leaving it |
| `stationary_ball_track_ends_on_spot` | the stationary track ended without a departure being seen -- the tracker lost the struck ball. Same event, weaker evidence, confidence 0.45 |
| `keeper_lateral_velocity_threshold` | first frame the keeper exceeds 0.35 body-heights/s laterally, searched only in a window around contact |
| `not_attempted` | the input needed does not exist in this source |

### Why contact is departure, not acceleration

The intuitive rule -- wait for ball speed to rise -- cannot work on real
penalty footage, and it fails silently. A penalty ball is stationary on the
spot, which a detector tracks very well; the instant it is struck it crosses
tens of pixels per frame and the tracker **loses** it. On the first real clip
tested, the ball track sat at x≈507, y≈595 for 240 consecutive frames and then
simply ended. Ball speed never rose at all, so a speed-threshold detector
reports "no contact" on a clip containing an obvious penalty.

Contact is therefore detected as departure from the spot. The two branches --
departure seen, versus track ending while still on the spot -- are reported
through different `method` values and different confidences, because they are
different strengths of evidence for the same event.

This also fixes role assignment downstream: the kicker is whoever stands over
the ball at contact, which is far more reliable than "the track that moves
most" (that picks officials and running defenders).

## `temporal_frames.parquet` — one row per observed frame

The lossless anchored view: kicker kinematics, pose-derived scalars
(`hip_shoulder_angle_deg`, `pelvis_orientation_deg`, `torso_lean_deg`,
`ankle_separation_n`, `plant_ankle_x_n`, `kick_ankle_x_n`, `kick_ankle_vx_n`),
keeper, ball, and interaction terms — each paired with an `*_available` flag.

Kicks whose rows span multiple upstream tracks are collapsed to the track
containing contact, so the timeline describes one body.

## `temporal_snapshots.parquet` — the experiment table

Exactly one row per (kick, offset) for **-2000, -1500, -1000, -750, -500, -250,
0 ms**, whether or not the observation exists.

| column | meaning |
| --- | --- |
| `snapshot_available` | whether an observation exists at this offset |
| `snapshot_method` | `exact`, `interp`, or `unavailable` |
| `gap_ms` | distance to the nearest observed frame |
| `unavailable_reason` | `before_clip_start`, `after_clip_end`, `gap_exceeds_interpolation_limit`, `no_anchored_frames` |

`exact` means within half a frame interval. `interp` is linear between
bracketing observations and is refused when they are more than 120 ms apart —
three frames at 25 fps, beyond which limb positions are not linear in time and an
interpolated pose would be fiction. Interpolated rows carry
`derivation = pkcv_interpolated`.

**Condition on `snapshot_available`.** Early offsets are far better covered in
the women's corpus (longer clips) than the EPL one, so an uncontrolled
comparison across offsets partly measures which corpus survived to that offset.

## `qc.parquet` — one row per (kick, check)

`status` ∈ `pass | warn | fail | na`, with `value`, `threshold` and `message`.
`na` means the check cannot apply to that source — grading a licensing fact as
a pipeline failure would misattribute it. Checks: `contact_anchor`,
`kicker_track`, `kicker_track_single_identity`, `pose_completeness`,
`keeper_track`, `ball_track`, `fps_plausible`, `min_frames`,
`snapshot_coverage`, `snapshot_at_contact`, `labels_present`.

A kick's `qc_status` in `metadata.parquet` is the worst status among applicable
checks. Nothing is dropped on failure; failures are recorded so consumers filter.

---

## Suggested use for the research question

```python
import pandas as pd

snaps = pd.read_parquet("data/processed/temporal/temporal_snapshots.parquet")
meta  = pd.read_parquet("data/processed/metadata.parquet")

df = snaps.merge(meta[["pk_id", "label_kick_direction", "source", "qc_status"]], on="pk_id")
df = df[df.snapshot_available & df.qc_status.ne("fail")]

# Accuracy as a function of how early you look, per corpus.
for offset, g in df.groupby("offset_ms"):
    ...  # fit on hip_shoulder_angle_deg, pelvis_orientation_deg, kick_ankle_x_n, ...
```

Hold out by `kicker_ref`, not by kick: the women's corpus has ~8 penalties per
kicker, and a random split would leak a kicker's idiosyncratic run-up across the
fold boundary and inflate accuracy.
