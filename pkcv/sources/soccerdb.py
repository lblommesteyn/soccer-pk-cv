"""SoccerDB (Xinhua Zhiyun / MMSPORTS 2020) -- ``Penalty Kick`` segments.

The annotation tables live in a public GitHub repository and are fetched
directly. Class 8 is ``Penalty Kick`` and, unlike SoccerNet's award
annotation, each segment carries ``event_start_time``/``event_end_time`` --
tight bounds around the kick itself, which is exactly what a clip cutter needs.

Video is gated: SoccerDB-only matches require a signed agreement form emailed
to the depositor, and the remainder are SoccerNet matches under the SoccerNet
NDA. This adapter never attempts either. It publishes the segment index and the
SoccerDB-to-SoccerNet crosswalk so that a credentialed user can cut the clips
themselves, and reports the source as ``gated``.

One caveat worth stating plainly: the GitHub repository carries no licence
file, so the annotation tables are "publicly posted, licence unstated". They
are used here to *count and locate* penalties; their contents are not
republished.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from pkcv.ids import make_dedup_key, make_pk_id, make_pk_uid
from pkcv.io import utc_now
from pkcv.schemas import PROCESSING_VERSION
from pkcv.sources.base import (
    ACCESS_ERROR,
    ACCESS_GATED,
    MEDIA_NONE,
    IngestResult,
    SourceAdapter,
    SourceReport,
)

RAW = "https://raw.githubusercontent.com/newsdata/SoccerDB/master/dataset/video_dataset/{name}"
SEG_INFO = "seg_info.csv"
CROSSWALK = "SoccerDB2SoccerNet.csv"
PENALTY_CLS = "8"
UA = {"User-Agent": "pkcv/0.1 (research dataset pipeline)"}


def _download(name: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    req = urllib.request.Request(RAW.format(name=name), headers=UA)
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    return dest


def _to_seconds(hhmmss: str) -> float | None:
    try:
        h, m, s = str(hhmmss).split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except (ValueError, AttributeError):
        return None


class SoccerDBPenalties(SourceAdapter):
    slug = "soccerdb"
    title = "SoccerDB -- Penalty Kick segments (class 8)"
    deposit_family = "soccerdb"

    @property
    def index_dir(self) -> Path:
        return self.raw_dir / "index"

    def _tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        seg = pd.read_csv(_download(SEG_INFO, self.index_dir / SEG_INFO))
        walk = pd.read_csv(_download(CROSSWALK, self.index_dir / CROSSWALK))
        walk.columns = ["soccerdb_name", "soccernet_name"]
        return seg, walk

    @staticmethod
    def _penalties(seg: pd.DataFrame) -> pd.DataFrame:
        mask = seg["cls_id"].astype(str).str.split().apply(lambda parts: PENALTY_CLS in parts)
        return seg[mask].copy()

    # -------------------------------------------------------------- inventory

    def inventory(self) -> SourceReport:
        report = SourceReport(
            source=self.slug,
            title=self.title,
            url="https://github.com/newsdata/SoccerDB",
            doi="10.1145/3422844.3423051",
            access=ACCESS_GATED,
            media_kind=MEDIA_NONE,
            license="Annotations: publicly posted on GitHub, licence unstated. Video: signed agreement form / SoccerNet NDA.",
            license_url="https://github.com/newsdata/SoccerDB/tree/master/dataset",
            attribution="Jiang et al., SoccerDB (ACM MMSPORTS 2020)",
            deposit_family=self.deposit_family,
            redistribute_derived=False,
            redistribute_video=False,
            redistribution_note=(
                "No licence is declared for the annotation tables, so neither they nor anything "
                "derived from them is republished. Counts and locations only."
            ),
            labels_available=["penalty_segment_bounds", "highlight_class"],
            checked_at_utc=utc_now(),
        )
        try:
            seg, walk = self._tables()
        except Exception as exc:
            report.access = ACCESS_ERROR
            report.access_note = f"index probe failed: {type(exc).__name__}: {exc}"
            return report

        pens = self._penalties(seg)
        vids = set(pens["video_name"].str.replace(".mkv", "", regex=False))
        mapped = walk[
            walk["soccerdb_name"].str.replace(".mkv", "", regex=False).isin(vids)
            & walk["soccernet_name"].notna()
        ]
        report.n_pk_discovered = int(len(pens))
        report.n_pk_accessible = 0
        report.files = [{"filename": SEG_INFO}, {"filename": CROSSWALK}]
        report.access_note = (
            "Video requires either the SoccerDB agreement form (emailed to Xinhua Zhiyun) or the "
            "SoccerNet NDA. Neither is held, so no penalties are processable from this source."
        )
        report.notes.append(
            f"{len(pens)} penalty-kick segments across {pens['video_name'].nunique()} half-videos"
        )
        report.notes.append(
            f"{len(mapped)} of those half-videos map to SoccerNet matches, so a SoccerNet NDA "
            "holder could cut those segments directly using the bounds published here."
        )
        report.notes.append(
            "Segment bounds are event-tight (event_start_time/event_end_time), unlike SoccerNet's "
            "award-instant annotation."
        )
        return report

    # ------------------------------------------------------------------ fetch

    def fetch(self, limit: int | None = None) -> SourceReport:
        """Fetch the public index only. The media is gated and is not attempted."""
        return self.inventory()

    # ----------------------------------------------------------------- ingest

    def ingest(self, limit: int | None = None) -> IngestResult:
        seg, walk = self._tables()
        pens = self._penalties(seg)
        if limit is not None:
            pens = pens.head(limit)
        sn = dict(
            zip(
                walk["soccerdb_name"].str.replace(".mkv", "", regex=False),
                walk["soccernet_name"],
                strict=False,
            )
        )
        rows: list[dict[str, Any]] = []
        for _, r in pens.iterrows():
            raw_id = str(r["seg_id"])
            pk_id = make_pk_id(self.slug, raw_id)
            base = str(r["video_name"]).replace(".mkv", "")
            start, end = _to_seconds(r["event_start_time"]), _to_seconds(r["event_end_time"])
            rows.append(
                {
                    "pk_id": pk_id,
                    "pk_uid": make_pk_uid(pk_id),
                    "source": self.slug,
                    "source_identifier": raw_id,
                    "source_dataset_doi": "10.1145/3422844.3423051",
                    "source_version": "2020",
                    "dedup_key": make_dedup_key(deposit_family=self.deposit_family, source_identifier=raw_id),
                    "is_primary": True,
                    "duplicate_of_pk_id": None,
                    "duplicate_evidence": None,
                    "media_kind": MEDIA_NONE,
                    "has_video": False,
                    "video_relpath": None,
                    "video_sha256": None,
                    "fps": None,
                    "n_frames": None,
                    "frame_width": None,
                    "frame_height": None,
                    "clip_duration_s": (end - start) if (start is not None and end is not None) else None,
                    "competition": None,
                    "gender": "men",
                    "level": "professional",
                    "season": None,
                    "match_context": "game",
                    "kicker_ref": None,
                    "label_kick_direction": None,
                    "label_keeper_direction": None,
                    "label_outcome": None,
                    "label_goal": None,
                    "label_footedness": None,
                    "label_camera_direction": None,
                    "label_provenance": "source_provided:soccerdb (segment bounds only)",
                    "license": "annotations licence unstated; video gated",
                    "license_url": "https://github.com/newsdata/SoccerDB/tree/master/dataset",
                    "attribution": "Jiang et al., SoccerDB (ACM MMSPORTS 2020)",
                    "redistribute_derived": False,
                    "redistribute_video": False,
                    "redistribution_note": "no declared licence; index retained locally only",
                    "ingested_at_utc": utc_now(),
                    "qc_status": "fail",
                    "qc_reasons": "video_gated_agreement_form_required",
                    "processing_version": PROCESSING_VERSION,
                    "derivation": "source_provided",
                    "model_name": None,
                    "model_version": None,
                    "provenance": (
                        f"soccerdb:{SEG_INFO}#{raw_id}"
                        + (f" -> soccernet:{sn[base]}" if base in sn and pd.notna(sn.get(base)) else "")
                    ),
                }
            )
        return IngestResult(
            metadata=pd.DataFrame(rows),
            notes=[
                f"{len(rows)} penalty segment(s) inventoried; none processable without video access."
            ],
        )
