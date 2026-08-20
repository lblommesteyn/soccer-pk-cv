"""Render the full QC overlay for every clip that has usable output.

Draws from the published parquet, so the video shows what was actually written.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pkcv.config import Config  # noqa: E402
from pkcv.io import read_parquet  # noqa: E402
from pkcv.qc import video_overlay  # noqa: E402


def main() -> int:
    cfg = Config.load()
    md = read_parquet(cfg.paths.artifact("metadata"), "metadata")
    qc = read_parquet(cfg.paths.artifact("qc"), "qc")
    piv = qc.pivot_table(index="pk_id", columns="check", values="status", aggfunc="first")
    keep = piv[(piv.get("contact_anchor") == "pass")].index

    tables = {n: read_parquet(cfg.paths.artifact(n), n)
              for n in ("tracks", "poses", "ball", "geometry", "events")}
    out_dir = cfg.paths.qc / "overlays"
    made = 0
    for _, row in md[md["pk_id"].isin(keep) & (md["media_kind"] == "video")].iterrows():
        pk = row["pk_id"]
        video = Path(cfg.paths.root) / str(row["video_relpath"])
        if not video.exists():
            continue
        safe = pk.replace(":", "__").replace("/", "_")
        try:
            p = video_overlay.render(
                row, video,
                *(t[t["pk_id"] == pk] for t in
                  (tables["tracks"], tables["poses"], tables["ball"],
                   tables["geometry"], tables["events"])),
                out_dir / f"{safe}_full.mp4",
            )
            print("wrote", p, flush=True)
            made += 1
        except Exception as exc:
            print(f"failed {pk}: {type(exc).__name__}: {exc}", flush=True)
    print(f"{made} overlay(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
