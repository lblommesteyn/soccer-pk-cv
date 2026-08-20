"""End-to-end corpus report -- the numbers the project is judged on.

Everything here is computed from the parquet tables, so the report cannot
disagree with what was published.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pkcv.config import Config
from pkcv.io import read_json, read_parquet
from pkcv.schemas import SNAPSHOT_OFFSETS_MS


def corpus_report(cfg: Config) -> dict[str, Any]:
    md = read_parquet(cfg.paths.artifact("metadata"), "metadata")
    qc = read_parquet(cfg.paths.artifact("qc"), "qc")
    snaps = read_parquet(cfg.paths.artifact("temporal_snapshots"), "temporal_snapshots")
    events = read_parquet(cfg.paths.artifact("events"), "events")
    inventory = read_json(cfg.paths.manifests / "source_inventory.json", default=[]) or []
    dupes = read_json(cfg.paths.manifests / "duplicates.json", default=[]) or []

    rep: dict[str, Any] = {"sources": {}, "corpus": {}, "success_rates": {}, "blockers": [], "snapshots": {}}

    for item in inventory:
        rep["sources"][item["source"]] = {
            "title": item.get("title"),
            "access": item.get("access"),
            "media_kind": item.get("media_kind"),
            "license": item.get("license"),
            "discovered": item.get("n_pk_discovered"),
            "accessible": item.get("n_pk_accessible"),
            "note": item.get("access_note"),
        }
        if item.get("access") in {"gated", "blocked", "error"}:
            rep["blockers"].append(
                {"source": item["source"], "access": item["access"], "reason": item.get("access_note")}
            )

    if len(md):
        primary = md[md["is_primary"].astype(bool)]
        usable = primary[primary["media_kind"].isin(["video", "pose_table"])]
        rep["corpus"] = {
            "records_ingested": int(len(md)),
            "unique_after_dedup": int(len(primary)),
            "duplicates_resolved": int(len(dupes)),
            "with_usable_media": int(len(usable)),
            "by_source": md.groupby("source").size().to_dict(),
            "unique_by_source": primary.groupby("source").size().to_dict(),
        }
        if len(usable):
            rep["corpus"]["label_coverage"] = {
                c: float(usable[c].notna().mean())
                for c in ("label_kick_direction", "label_goal", "label_footedness", "label_camera_direction")
            }

    if len(qc):
        piv = qc.pivot_table(index="pk_id", columns="check", values="status", aggfunc="first")
        for check in piv.columns:
            counts = piv[check].value_counts()
            applicable = int(counts.drop(labels=["na"], errors="ignore").sum())
            passed = int(counts.get("pass", 0))
            rep["success_rates"][check] = {
                "pass": passed,
                "warn": int(counts.get("warn", 0)),
                "fail": int(counts.get("fail", 0)),
                "not_applicable": int(counts.get("na", 0)),
                "rate": (passed / applicable) if applicable else None,
            }

    if len(events):
        contact = events[events["event_name"] == "ball_contact"]
        if len(contact):
            rep["success_rates"]["contact_frame_resolved"] = {
                "resolved": int((~contact["is_missing"].astype(bool)).sum()),
                "of": int(len(contact)),
                "rate": float((~contact["is_missing"].astype(bool)).mean()),
            }

    if len(snaps):
        for off in SNAPSHOT_OFFSETS_MS:
            sub = snaps[snaps["offset_ms"] == off]
            if len(sub):
                rep["snapshots"][str(off)] = {
                    "available": int(sub["snapshot_available"].astype(bool).sum()),
                    "of": int(len(sub)),
                    "rate": float(sub["snapshot_available"].astype(bool).mean()),
                }
        reasons = snaps.loc[~snaps["snapshot_available"].astype(bool), "unavailable_reason"]
        rep["snapshots"]["unavailable_reasons"] = reasons.value_counts().to_dict()

    manifest = read_json(cfg.paths.root / "hf_staging" / "manifest.json", default=None)
    if manifest:
        rep["huggingface"] = {
            "repo_id": manifest.get("repo_id"),
            "published_at": manifest.get("published_at_utc"),
            "files": manifest.get("files"),
            "stats": manifest.get("stats"),
        }
    return rep


def render_text(rep: dict[str, Any]) -> str:
    L: list[str] = ["", "=" * 78, "  soccer-pk-cv corpus report", "=" * 78, ""]

    L.append("SOURCES")
    for src, s in rep.get("sources", {}).items():
        L.append(f"  {src:<20} access={s['access']:<8} media={s['media_kind']:<12} "
                 f"discovered={s['discovered']} accessible={s['accessible']}")
        L.append(f"  {'':<20} {s['note']}")
    L.append("")

    c = rep.get("corpus", {})
    if c:
        L.append("CORPUS")
        L.append(f"  records ingested        : {c.get('records_ingested')}")
        L.append(f"  unique after dedup      : {c.get('unique_after_dedup')} "
                 f"({c.get('duplicates_resolved')} duplicates resolved)")
        L.append(f"  with usable media       : {c.get('with_usable_media')}")
        for src, n in (c.get("unique_by_source") or {}).items():
            L.append(f"      {src:<24} {n}")
        L.append("")

    if rep.get("success_rates"):
        L.append("SUCCESS RATES")
        for check, s in rep["success_rates"].items():
            if "rate" in s and s["rate"] is not None:
                extra = f" (n/a: {s['not_applicable']})" if s.get("not_applicable") else ""
                L.append(f"  {check:<32} {s['rate']:6.1%}{extra}")
            else:
                L.append(f"  {check:<32} n/a for every kick")
        L.append("")

    if rep.get("snapshots"):
        L.append("SNAPSHOT AVAILABILITY")
        for off in SNAPSHOT_OFFSETS_MS:
            s = rep["snapshots"].get(str(off))
            if s:
                L.append(f"  {off:>6} ms   {s['available']:>5} / {s['of']:<5} {s['rate']:6.1%}")
        for reason, n in (rep["snapshots"].get("unavailable_reasons") or {}).items():
            L.append(f"      unavailable: {reason} x{n}")
        L.append("")

    if rep.get("blockers"):
        L.append("BLOCKERS")
        for b in rep["blockers"]:
            L.append(f"  {b['source']:<20} {b['access']}: {b['reason']}")
        L.append("")

    if rep.get("huggingface"):
        h = rep["huggingface"]
        L.append("HUGGING FACE")
        L.append(f"  repo   : {h['repo_id']}")
        L.append(f"  at     : {h['published_at']}")
        L.append(f"  files  : {len(h.get('files') or [])}")
        L.append("")
    return "\n".join(L)
