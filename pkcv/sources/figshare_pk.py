"""figshare article 31464526 -- the mirror of Mendeley ``brx9bsxnpx`` v2.

Same authors, same 132 women's collegiate penalties, same CC BY 4.0 licence.
It is therefore *not* a new corpus and every kick it contains deduplicates
against ``mendeley-women-v2`` on the shared deposit family. What it adds that
Mendeley does not publish is ``Skeleton Videos.zip``: one 1920x1080 MP4 per
kick.

Those MP4s are skeleton renders on a black background -- the underlying footage
is absent. Verified empirically at ingest: the mean non-black pixel fraction of
the first frame is well under 1%. They are consequently registered as
``render_only`` media. Running detection on them would produce numbers, and
those numbers would be meaningless, so the vision stage refuses them.

Their real value is metrological: they carry the authoritative per-kick frame
count, resolution and container fps, which lets us cross-check the pose table's
timeline independently of the CSV.
"""

from __future__ import annotations

import json
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pkcv.io import sha256_file, utc_now
from pkcv.sources.base import (
    ACCESS_ERROR,
    ACCESS_OPEN,
    MEDIA_RENDER_ONLY,
    IngestResult,
    SourceAdapter,
    SourceReport,
)

ARTICLE_ID = 31464526
API = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
UA = {"User-Agent": "pkcv/0.1 (research dataset pipeline)"}
VIDEO_ZIP = "Skeleton Videos.zip"
CLIP_RE = re.compile(r"(\d+)-(\d+)_skeleton_only\.mp4$", re.IGNORECASE)

#: A render is "black background only" if fewer than this fraction of pixels in
#: the probed frame are non-black. Skeleton strokes on 1920x1080 come in around
#: 0.2-0.5%; any real footage is orders of magnitude above.
BLACK_FRAME_MAX_INK = 0.05


def _api() -> dict[str, Any]:
    req = urllib.request.Request(API, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


class FigshareWomenMirror(SourceAdapter):
    slug = "figshare-women-v2"
    title = "Frame-Level Women's Soccer Penalty-Kick Data Captured by Computer Vision Models (figshare 31464526)"
    #: identical to the Mendeley v2 family -- this is what drives deduplication
    deposit_family = "womens-collegiate-li-pifer"

    # ---------------------------------------------------------------- inventory

    def inventory(self) -> SourceReport:
        report = SourceReport(
            source=self.slug,
            title=self.title,
            url=f"https://figshare.com/articles/dataset/_/{ARTICLE_ID}",
            access=ACCESS_OPEN,
            media_kind=MEDIA_RENDER_ONLY,
            deposit_family=self.deposit_family,
            redistribute_derived=True,
            redistribute_video=True,
            redistribution_note=(
                "CC BY 4.0 and the renders contain no third-party footage (skeleton strokes on "
                "black), so they are redistributable with attribution."
            ),
            checked_at_utc=utc_now(),
        )
        try:
            meta = _api()
        except Exception as exc:
            report.access = ACCESS_ERROR
            report.access_note = f"probe failed: {type(exc).__name__}: {exc}"
            return report

        report.title = meta.get("title", report.title)
        report.doi = meta.get("doi")
        lic = meta.get("license") or {}
        report.license = lic.get("name")
        report.license_url = lic.get("url")
        report.attribution = "; ".join(a["full_name"] for a in meta.get("authors", []))
        report.files = [
            {
                "filename": f["name"],
                "size_bytes": f.get("size"),
                "download_url": f.get("download_url"),
                "md5_upstream": f.get("computed_md5") or f.get("supplied_md5"),
            }
            for f in meta.get("files", [])
        ]
        zip_entry = next((f for f in report.files if f["filename"] == VIDEO_ZIP), None)
        report.n_pk_discovered = 132
        report.n_pk_accessible = 132 if zip_entry else 0
        report.access_note = "public download, no credential required"
        report.labels_available = ["(labels live in the Mendeley v2 CSV mirror of this deposit)"]
        report.notes.append(
            "Duplicate of Mendeley brx9bsxnpx v2 (same authors, same kicks). Ingested as a media "
            "attachment to the Mendeley records, not as new penalties."
        )
        report.notes.append(
            "Skeleton renders only -- no real footage, so keeper/ball/goal geometry remain underivable."
        )
        return report

    # ------------------------------------------------------------------- fetch

    @property
    def zip_path(self) -> Path:
        return self.raw_dir / "skeleton_videos.zip"

    @property
    def clips_dir(self) -> Path:
        p = self.raw_dir / "clips"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def fetch(self, limit: int | None = None) -> SourceReport:
        report = self.inventory()
        if report.access != ACCESS_OPEN:
            return report
        entry = next((f for f in report.files if f["filename"] == VIDEO_ZIP), None)
        if entry is None:
            report.access = ACCESS_ERROR
            report.access_note = f"{VIDEO_ZIP!r} not present upstream"
            return report
        if not self.zip_path.exists():
            req = urllib.request.Request(entry["download_url"], headers=UA)
            with urllib.request.urlopen(req, timeout=900) as resp, open(self.zip_path, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
        report.notes.append(f"zip sha256={sha256_file(self.zip_path)}")

        with zipfile.ZipFile(self.zip_path) as zf:
            names = [n for n in zf.namelist() if CLIP_RE.search(n) and not n.startswith("__MACOSX")]
            names.sort()
            if limit is not None:
                names = names[:limit]
            for name in names:
                out = self.clips_dir / Path(name).name
                if out.exists():
                    continue
                with zf.open(name) as src, open(out, "wb") as dst:
                    dst.write(src.read())
        report.notes.append(f"{len(list(self.clips_dir.glob('*.mp4')))} clip(s) extracted")
        return report

    # ------------------------------------------------------------------ ingest

    def ingest(self, limit: int | None = None) -> IngestResult:
        """Emit one metadata row per render, marked as a duplicate.

        The rows exist so that provenance, checksums and measured clip geometry
        are recorded; ``is_primary=False`` keeps them out of every analysis
        that selects the canonical corpus.
        """
        import cv2  # local import: ingest of other sources must not need opencv

        clips = sorted(self.clips_dir.glob("*.mp4"))
        if limit is not None:
            clips = clips[:limit]
        from pkcv.ids import make_dedup_key, make_pk_id, make_pk_uid
        from pkcv.schemas import PROCESSING_VERSION

        rows = []
        for clip in clips:
            m = CLIP_RE.search(clip.name)
            if not m:
                continue
            raw_id = f"{int(m.group(1))}-{int(m.group(2))}"
            pk_id = make_pk_id(self.slug, raw_id)
            cap = cv2.VideoCapture(str(clip))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or None
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
            ok, frame = cap.read()
            cap.release()
            ink = float((frame.max(axis=2) > 10).mean()) if ok else float("nan")
            render_only = bool(ok and ink < BLACK_FRAME_MAX_INK)
            rows.append(
                {
                    "pk_id": pk_id,
                    "pk_uid": make_pk_uid(pk_id),
                    "source": self.slug,
                    "source_identifier": raw_id,
                    "source_dataset_doi": f"10.6084/m9.figshare.{ARTICLE_ID}",
                    "source_version": "2",
                    "dedup_key": make_dedup_key(
                        deposit_family=self.deposit_family, source_identifier=raw_id
                    ),
                    "is_primary": False,
                    "duplicate_of_pk_id": None,  # resolved centrally in dedup
                    "duplicate_evidence": "same deposit family as mendeley-women-v2 (same authors, same kicks)",
                    "media_kind": MEDIA_RENDER_ONLY if render_only else "video",
                    "has_video": True,
                    "video_relpath": str(clip.relative_to(self.cfg.paths.root)).replace("\\", "/"),
                    "video_sha256": sha256_file(clip),
                    "fps": fps,
                    "n_frames": n,
                    "frame_width": w,
                    "frame_height": h,
                    "clip_duration_s": (n / fps) if (n and fps) else None,
                    "competition": "US collegiate women's soccer",
                    "gender": "women",
                    "level": "collegiate",
                    "season": None,
                    "match_context": None,
                    "kicker_ref": f"mendeley-women-v2#{int(m.group(1))}",
                    "label_kick_direction": None,
                    "label_keeper_direction": None,
                    "label_outcome": None,
                    "label_goal": None,
                    "label_footedness": None,
                    "label_camera_direction": None,
                    "label_provenance": "none (labels held by the Mendeley v2 mirror)",
                    "license": "CC BY 4.0",
                    "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "attribution": "Li, Feiting; Pifer, N. David. figshare, doi:10.6084/m9.figshare.31464526",
                    "redistribute_derived": True,
                    "redistribute_video": True,
                    "redistribution_note": "skeleton renders contain no third-party footage",
                    "ingested_at_utc": utc_now(),
                    "qc_status": "pending",
                    "qc_reasons": None if render_only else "render_ink_fraction_above_threshold",
                    "processing_version": PROCESSING_VERSION,
                    "derivation": "source_provided",
                    "model_name": None,
                    "model_version": None,
                    "provenance": f"figshare:{ARTICLE_ID}/{VIDEO_ZIP}/{clip.name}",
                }
            )
        notes = []
        if rows:
            inks = [r for r in rows if r["media_kind"] != MEDIA_RENDER_ONLY]
            notes.append(
                f"{len(rows)} render(s) registered; {len(inks)} exceeded the black-background "
                "threshold and were not classified render_only"
            )
        return IngestResult(metadata=pd.DataFrame(rows), notes=notes)
