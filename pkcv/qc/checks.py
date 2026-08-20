"""Per-kick quality control.

Each check returns ``pass`` / ``warn`` / ``fail`` / ``na``. ``na`` is used when a
check cannot apply to a source at all -- there is no ball track to grade in a
pose-table deposit, and grading it ``fail`` would make a licensing fact look
like a pipeline defect.

A kick's overall ``qc_status`` is the worst status among checks that apply.
Nothing is dropped on a failure; failures are recorded so a consumer can filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pkcv.schemas import PROCESSING_VERSION, SNAPSHOT_OFFSETS_MS

PASS, WARN, FAIL, NA = "pass", "warn", "fail", "na"
_ORDER = {NA: 0, PASS: 1, WARN: 2, FAIL: 3}

DEFAULTS = {
    "min_pose_completeness": 0.70,
    "warn_pose_completeness": 0.90,
    "min_kicker_coverage": 0.80,
    "min_snapshot_coverage": 0.50,
    "fps_min": 10.0,
    "fps_max": 120.0,
    "min_frames": 10,
    "min_goal_frames": 0.60,
    # Apparent player size in pixels. Measured across the Commons corpus:
    # clips whose contact anchored had a median role-box height around 121 px,
    # those that failed around 61 px. Ball size scales with player size, and the
    # ball is the smallest thing the pipeline must find, so this is the binding
    # constraint on whether a clip is processable at all.
    "warn_scene_scale_px": 85.0,
    "fail_scene_scale_px": 50.0,
    # How far the kicker's box centre may sit from the ball at contact, in
    # kicker body-heights. A player striking a ball is on top of it; correctly
    # assigned clips measure 0.3-0.9 here. This catches the kicker role landing
    # on a bystander, which no other check sees.
    "warn_kicker_ball_dist": 1.5,
    "fail_kicker_ball_dist": 3.0,
}


def _row(pk_id, source, check, status, value=None, threshold=None, message=None) -> dict:
    return {
        "pk_id": pk_id,
        "source": source,
        "check": check,
        "status": status,
        "value": None if value is None else float(value),
        "threshold": None if threshold is None else float(threshold),
        "message": message,
        "processing_version": PROCESSING_VERSION,
    }


def check_kick(
    meta_row: pd.Series,
    tracks: pd.DataFrame,
    poses: pd.DataFrame,
    ball: pd.DataFrame,
    events: pd.DataFrame,
    snapshots: pd.DataFrame,
    thresholds: dict | None = None,
    geometry: pd.DataFrame | None = None,
) -> list[dict]:
    th = {**DEFAULTS, **(thresholds or {})}
    pk_id, source = meta_row["pk_id"], meta_row["source"]
    media = str(meta_row.get("media_kind"))
    out: list[dict] = []

    # ---- contact anchor -------------------------------------------------
    contact = events[events["event_name"] == "ball_contact"]
    if len(contact) and not bool(contact["is_missing"].iloc[0]):
        conf = contact["confidence"].iloc[0]
        method = str(contact["method"].iloc[0])
        if "latest_of_multiple" in method:
            out.append(
                _row(pk_id, source, "contact_anchor", WARN, conf,
                     message=f"contact taken from the latest of several markers ({method})")
            )
        else:
            out.append(_row(pk_id, source, "contact_anchor", PASS, conf, message=method))
    else:
        reason = contact["missing_reason"].iloc[0] if len(contact) else "no_contact_event_row"
        out.append(_row(pk_id, source, "contact_anchor", FAIL, message=str(reason)))

    # ---- track integrity ------------------------------------------------
    kicker = tracks[(tracks["role"] == "kicker") & (tracks["frame_idx"] >= 0)]
    n_frames = int(meta_row.get("n_frames") or len(kicker))
    if not len(kicker):
        out.append(_row(pk_id, source, "kicker_track", FAIL, message="no kicker track rows"))
    else:
        # Coverage is measured over the run-up window, not the whole clip. A
        # video penalty is often embedded in footage that starts long before the
        # kicker walks up and continues through the celebration, so whole-clip
        # coverage measures how much unrelated footage the clip contains rather
        # than how well the kicker was tracked.
        seen = kicker[~kicker["is_missing"].astype(bool)]
        window = None
        contact_row = events[events["event_name"] == "ball_contact"] if len(events) else events
        if len(contact_row) and pd.notna(contact_row["frame_idx"].iloc[0]) and pd.notna(meta_row.get("fps")):
            cf = int(contact_row["frame_idx"].iloc[0])
            fps_v = float(meta_row["fps"])
            lo = cf - int(abs(SNAPSHOT_OFFSETS_MS[0]) / 1000.0 * fps_v)
            window = (max(lo, 0), cf)
        if window is not None:
            span = max(window[1] - window[0] + 1, 1)
            got = int(((seen["frame_idx"] >= window[0]) & (seen["frame_idx"] <= window[1])).sum())
            denom, label = span, f"{got}/{span} run-up frames"
        else:
            got, denom = len(seen), max(n_frames, 1)
            label = f"{got}/{denom} clip frames (no contact anchor to window on)"
        coverage = got / denom
        out.append(
            _row(pk_id, source, "kicker_track", PASS if coverage >= th["min_kicker_coverage"] else WARN,
                 coverage, th["min_kicker_coverage"], message=label)
        )
        n_tracks = int(kicker["track_id"].nunique())
        out.append(
            _row(pk_id, source, "kicker_track_single_identity", PASS if n_tracks <= 1 else WARN, n_tracks, 1,
                 message=f"{n_tracks} distinct track id(s) merged into this kick"
                 if n_tracks > 1 else "single continuous track")
        )

    # ---- pose completeness ----------------------------------------------
    kp = poses[poses["role"] == "kicker"]
    if not len(kp):
        out.append(_row(pk_id, source, "pose_completeness", FAIL, message="no kicker pose rows"))
    else:
        completeness = float((~kp["is_missing"].astype(bool)).mean())
        status = (
            PASS if completeness >= th["warn_pose_completeness"]
            else WARN if completeness >= th["min_pose_completeness"]
            else FAIL
        )
        out.append(
            _row(pk_id, source, "pose_completeness", status, completeness, th["min_pose_completeness"],
                 message=f"{completeness:.1%} of keypoint slots populated")
        )

    # ---- keeper / ball / geometry: applicability depends on the source ----
    keeper = tracks[(tracks["role"] == "keeper") & (tracks["frame_idx"] >= 0)]
    ball_obs = ball[ball["frame_idx"] >= 0] if len(ball) else ball
    pose_table_like = media in {"pose_table", "render_only", "none"}
    for name, present, why in (
        ("keeper_track", len(keeper) > 0, "keeper"),
        ("ball_track", len(ball_obs) > 0, "ball"),
    ):
        if pose_table_like:
            out.append(
                _row(pk_id, source, name, NA,
                     message=f"source is {media}; {why} is not observable and was not estimated")
            )
        else:
            out.append(
                _row(pk_id, source, name, PASS if present else FAIL,
                     message=None if present else f"no {why} observations recovered from video")
            )

    # ---- timing ----------------------------------------------------------
    fps = meta_row.get("fps")
    if fps is None or pd.isna(fps):
        out.append(_row(pk_id, source, "fps_plausible", FAIL, message="fps unknown"))
    else:
        ok = th["fps_min"] <= float(fps) <= th["fps_max"]
        out.append(_row(pk_id, source, "fps_plausible", PASS if ok else FAIL, fps,
                        message=f"fps={float(fps):.2f}"))
    out.append(
        _row(pk_id, source, "min_frames", PASS if n_frames >= th["min_frames"] else FAIL,
             n_frames, th["min_frames"], message=f"{n_frames} frames")
    )

    # ---- snapshot coverage ------------------------------------------------
    if len(snapshots):
        avail = snapshots["snapshot_available"].astype(bool)
        frac = float(avail.mean())
        missing = snapshots.loc[~avail, "offset_ms"].astype(int).tolist()
        out.append(
            _row(pk_id, source, "snapshot_coverage",
                 PASS if frac >= th["min_snapshot_coverage"] else WARN, frac, th["min_snapshot_coverage"],
                 message=f"{int(avail.sum())}/{len(SNAPSHOT_OFFSETS_MS)} offsets available"
                 + (f"; missing {missing}" if missing else ""))
        )
        at_contact = snapshots[snapshots["offset_ms"] == 0]
        out.append(
            _row(pk_id, source, "snapshot_at_contact",
                 PASS if len(at_contact) and bool(at_contact["snapshot_available"].iloc[0]) else FAIL,
                 message=None if len(at_contact) and bool(at_contact["snapshot_available"].iloc[0])
                 else "no observation at t=0")
        )
    else:
        out.append(_row(pk_id, source, "snapshot_coverage", FAIL, message="no snapshots built"))

    # ---- kicker really is at the ball --------------------------------------
    # An independent consistency check: it uses the ball, not the reasoning that
    # picked the kicker, so it catches role assignment latching onto the wrong
    # person even when contact, tracking and scale all look fine.
    if pose_table_like:
        out.append(
            _row(pk_id, source, "kicker_at_ball", NA,
                 message=f"source is {media}; no ball to check the kicker against")
        )
    else:
        cr = events[events["event_name"] == "ball_contact"] if len(events) else events
        cf = int(cr["frame_idx"].iloc[0]) if (
            len(cr) and pd.notna(cr["frame_idx"].iloc[0])) else None
        bb = ball[(ball["frame_idx"] <= cf) & ball["x"].notna()] if (
            cf is not None and len(ball) and "x" in ball) else None
        kk = tracks[(tracks["role"] == "kicker") & tracks["frame_idx"].between(cf - 4, cf + 4)] if (
            cf is not None and "role" in getattr(tracks, "columns", [])) else None
        if cf is None:
            out.append(_row(pk_id, source, "kicker_at_ball", NA,
                            message="no contact anchor to check against"))
        elif bb is None or not len(bb) or kk is None or not len(kk):
            out.append(_row(pk_id, source, "kicker_at_ball", FAIL,
                            message="no ball or kicker observation near contact"))
        else:
            bl = bb.sort_values("frame_idx").iloc[-1]
            k = kk.iloc[len(kk) // 2]
            d = float(np.hypot(k["bbox_cx"] - bl["x"], k["bbox_cy"] - bl["y"])
                      / max(float(k["bbox_h"] or 1), 1.0))
            status = (
                PASS if d <= th["warn_kicker_ball_dist"]
                else WARN if d <= th["fail_kicker_ball_dist"]
                else FAIL
            )
            out.append(
                _row(pk_id, source, "kicker_at_ball", status, d, th["warn_kicker_ball_dist"],
                     message=f"kicker sits {d:.2f} body-heights from the ball at contact")
            )

    # ---- labels -----------------------------------------------------------
    have = [c for c in ("label_kick_direction", "label_goal", "label_footedness")
            if pd.notna(meta_row.get(c))]
    provenance = str(meta_row.get("label_provenance") or "")
    if "label_kick_direction" in have:
        status, msg = PASS, f"labels: {', '.join(have)}"
    elif provenance.startswith("none"):
        # The source publishes footage but no annotations. That is a property of
        # the corpus, not a processing failure, so it warns rather than fails --
        # but it still warns, because a kick with no direction label cannot be
        # used to answer the question this dataset exists for.
        status, msg = WARN, "source supplies no outcome labels; annotate before supervised use"
    else:
        status, msg = FAIL, "source supplies labels but this record has no kick direction"
    out.append(_row(pk_id, source, "labels_present", status, len(have), 1, message=msg))

    # ---- scene scale (video sources only) ----------------------------------
    if pose_table_like:
        out.append(
            _row(pk_id, source, "scene_scale", NA,
                 message=f"source is {media}; no footage to measure apparent size in")
        )
    else:
        cols = getattr(tracks, "columns", [])
        obs = (
            tracks[(tracks["frame_idx"] >= 0) & tracks["bbox_h"].notna()]
            if {"bbox_h", "frame_idx", "role"} <= set(cols)
            else tracks.iloc[0:0]
        )
        if not len(obs):
            out.append(_row(pk_id, source, "scene_scale", FAIL, message="no boxes to measure"))
        else:
            scale = float(obs.groupby("role")["bbox_h"].median().max())
            status = (
                PASS if scale >= th["warn_scene_scale_px"]
                else WARN if scale >= th["fail_scene_scale_px"]
                else FAIL
            )
            out.append(
                _row(pk_id, source, "scene_scale", status, scale, th["warn_scene_scale_px"],
                     message=f"median role-box height {scale:.0f}px; "
                             "below ~85px the ball is too small to track reliably")
            )

    # ---- goal geometry (video sources only) --------------------------------
    if pose_table_like:
        out.append(
            _row(pk_id, source, "goal_geometry", NA,
                 message=f"source is {media}; no footage to recover the goal from")
        )
    elif geometry is None or not len(geometry):
        out.append(_row(pk_id, source, "goal_geometry", FAIL, message="no geometry rows"))
    else:
        found = float((~geometry["is_missing"].astype(bool)).mean())
        status = (
            PASS if found >= th["min_goal_frames"]
            else WARN if found > 0
            else FAIL
        )
        out.append(
            _row(pk_id, source, "goal_geometry", status, found, th["min_goal_frames"],
                 message=f"goal located on {found:.0%} of frames")
        )
    return out


def rollup(qc_rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-check rows to one ``qc_status`` + reason string per kick."""
    if not len(qc_rows):
        return pd.DataFrame(columns=["pk_id", "qc_status", "qc_reasons"])
    out = []
    for pk_id, g in qc_rows.groupby("pk_id"):
        worst = max(g["status"], key=lambda s: _ORDER.get(s, 0))
        bad = g[g["status"].isin([WARN, FAIL])]
        reasons = "; ".join(f"{r['check']}:{r['status']}" for _, r in bad.iterrows())
        out.append({"pk_id": pk_id, "qc_status": worst, "qc_reasons": reasons or None})
    return pd.DataFrame(out)


def corpus_summary(qc_rows: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    """Aggregate success rates -- the numbers the completion report quotes."""
    if not len(qc_rows):
        return {}
    piv = qc_rows.pivot_table(index="pk_id", columns="check", values="status", aggfunc="first")
    md = metadata.set_index("pk_id")
    summary: dict = {"n_kicks": int(len(piv)), "by_check": {}}
    for check in piv.columns:
        counts = piv[check].value_counts().to_dict()
        applicable = int(sum(v for k, v in counts.items() if k != NA))
        ok = int(counts.get(PASS, 0))
        summary["by_check"][check] = {
            "pass": ok,
            "warn": int(counts.get(WARN, 0)),
            "fail": int(counts.get(FAIL, 0)),
            "na": int(counts.get(NA, 0)),
            "success_rate_of_applicable": (ok / applicable) if applicable else None,
        }
    joined = piv.join(md[["source"]], how="left")
    summary["by_source"] = {
        src: {"n_kicks": int(len(g))}
        for src, g in joined.groupby("source")
    }
    return summary
