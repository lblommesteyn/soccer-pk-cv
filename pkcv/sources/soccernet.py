"""SoccerNet-v2 action spotting -- "Penalty" annotations.

The label files (``Labels-v2.json``) are openly downloadable and give an
authoritative count of penalty annotations across 500 broadcast matches. The
*video* is a different matter: SoccerNet distributes it under a signed
non-disclosure agreement, and the download requires a password issued to the
signatory. This pipeline never attempts to work around that. Without
``SOCCERNET_PASSWORD`` in the environment the source is reported ``gated`` and
contributes labels-only inventory rows.

Two caveats that a consumer of this corpus must know, recorded here rather than
buried:

* The ``Penalty`` class marks the *award* of a penalty, not the instant of the
  kick. The kick follows by a variable delay, typically tens of seconds. A clip
  window anchored on the label is a search region, not a kick.
* Annotation timestamps are at one-second resolution, which is 25-50 frames --
  far coarser than the contact anchor this dataset is built around. Contact
  must be re-derived from video; the label cannot serve as ``t=0``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from pkcv.ids import make_dedup_key, make_pk_id, make_pk_uid
from pkcv.io import utc_now
from pkcv.schemas import PROCESSING_VERSION
from pkcv.sources.base import (
    ACCESS_ERROR,
    ACCESS_GATED,
    ACCESS_OPEN,
    MEDIA_NONE,
    MEDIA_VIDEO,
    IngestResult,
    SourceAdapter,
    SourceReport,
)

PENALTY_LABEL = "Penalty"
SPLITS = ("train", "valid", "test")


class SoccerNetPenalties(SourceAdapter):
    slug = "soccernet-v2"
    title = "SoccerNet-v2 action spotting -- Penalty annotations"
    deposit_family = "soccernet-v2"

    @property
    def labels_dir(self) -> Path:
        return self.raw_dir / "labels"

    # ---------------------------------------------------------------- helpers

    def _password(self) -> str | None:
        return os.environ.get("SOCCERNET_PASSWORD") or self.source_cfg.get("password")

    def _download_labels(self) -> None:
        from SoccerNet.Downloader import SoccerNetDownloader

        self.labels_dir.mkdir(parents=True, exist_ok=True)
        dl = SoccerNetDownloader(LocalDirectory=str(self.labels_dir))
        dl.downloadGames(files=["Labels-v2.json"], split=list(SPLITS))

    def _scan_labels(self) -> list[dict[str, Any]]:
        rows = []
        for path in self.labels_dir.rglob("Labels-v2.json"):
            try:
                with open(path, encoding="utf-8") as fh:
                    doc = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            game = str(path.parent.relative_to(self.labels_dir)).replace("\\", "/")
            for i, ann in enumerate(doc.get("annotations", [])):
                if ann.get("label") != PENALTY_LABEL:
                    continue
                rows.append(
                    {
                        "game": game,
                        "url_local": doc.get("UrlLocal"),
                        "index": i,
                        "half": int(ann.get("gameTime", "1 - 00:00").split("-")[0].strip() or 1),
                        "game_time": ann.get("gameTime"),
                        "position_ms": int(ann.get("position", 0)),
                        "team": ann.get("team"),
                        "visibility": ann.get("visibility"),
                    }
                )
        return rows

    # -------------------------------------------------------------- inventory

    def inventory(self) -> SourceReport:
        report = SourceReport(
            source=self.slug,
            title=self.title,
            url="https://www.soccer-net.org/",
            doi=None,
            access=ACCESS_GATED,
            media_kind=MEDIA_VIDEO,
            license="Labels: CC BY 4.0. Video: SoccerNet NDA, redistribution prohibited.",
            license_url="https://www.soccer-net.org/data",
            attribution="Deliege et al., SoccerNet-v2 (CVPR-W 2021)",
            deposit_family=self.deposit_family,
            redistribute_derived=True,
            redistribute_video=False,
            redistribution_note=(
                "Video is NDA-restricted and must never be republished. Derived numeric "
                "artifacts are publishable; frames and clips are not."
            ),
            labels_available=["penalty_award_timestamp", "team", "visibility"],
            checked_at_utc=utc_now(),
        )
        try:
            if not any(self.labels_dir.rglob("Labels-v2.json")):
                self._download_labels()
            events = self._scan_labels()
        except Exception as exc:
            report.access = ACCESS_ERROR
            report.access_note = f"label probe failed: {type(exc).__name__}: {exc}"
            return report

        report.n_pk_discovered = len(events)
        report.files = [{"filename": "Labels-v2.json", "count": len(list(self.labels_dir.rglob('Labels-v2.json')))}]
        if self._password():
            report.access = ACCESS_OPEN
            report.n_pk_accessible = len(events)
            report.access_note = "SOCCERNET_PASSWORD present; video download available"
        else:
            report.n_pk_accessible = 0
            report.access_note = (
                "Video requires a SoccerNet NDA password. Set SOCCERNET_PASSWORD to enable. "
                "Labels-only inventory recorded; no penalties processed from this source."
            )
        report.notes.append(
            f"{len(events)} 'Penalty' annotations across "
            f"{len({e['game'] for e in events})} matches"
        )
        report.notes.append(
            "'Penalty' marks the award, not the kick; timestamps are 1 s resolution. "
            "Contact must be re-derived from video -- the label cannot be used as t=0."
        )
        return report

    # ------------------------------------------------------------------ fetch

    def fetch(self, limit: int | None = None) -> SourceReport:
        report = self.inventory()
        if report.access != ACCESS_OPEN:
            return report
        from SoccerNet.Downloader import SoccerNetDownloader

        events = self._scan_labels()
        games = list(dict.fromkeys(e["game"] for e in events))
        if limit is not None:
            games = games[:limit]
        dl = SoccerNetDownloader(LocalDirectory=str(self.raw_dir / "video"))
        dl.password = self._password()
        quality = self.source_cfg.get("video_files", ["1_224p.mkv", "2_224p.mkv"])
        dl.downloadGames(files=list(quality), split=["train", "valid", "test"])
        report.notes.append(f"requested video for {len(games)} match(es) at {quality}")
        return report

    # ----------------------------------------------------------------- ingest

    def ingest(self, limit: int | None = None) -> IngestResult:
        events = self._scan_labels()
        if limit is not None:
            events = events[:limit]
        gated = self._password() is None
        rows = []
        for e in events:
            raw_id = f"{e['game']}#{e['half']}#{e['position_ms']}"
            pk_id = make_pk_id(self.slug, raw_id.replace("/", "_"))
            rows.append(
                {
                    "pk_id": pk_id,
                    "pk_uid": make_pk_uid(pk_id),
                    "source": self.slug,
                    "source_identifier": raw_id,
                    "source_dataset_doi": None,
                    "source_version": "v2",
                    "dedup_key": make_dedup_key(
                        deposit_family=self.deposit_family, source_identifier=raw_id.replace("/", "-")
                    ),
                    "is_primary": True,
                    "duplicate_of_pk_id": None,
                    "duplicate_evidence": None,
                    "media_kind": MEDIA_NONE if gated else MEDIA_VIDEO,
                    "has_video": not gated,
                    "video_relpath": None,
                    "video_sha256": None,
                    "fps": None,
                    "n_frames": None,
                    "frame_width": None,
                    "frame_height": None,
                    "clip_duration_s": None,
                    "competition": e["game"].split("/")[0] if "/" in e["game"] else None,
                    "gender": "men",
                    "level": "professional",
                    "season": e["game"].split("/")[1] if e["game"].count("/") >= 1 else None,
                    "match_context": "game",
                    "kicker_ref": None,
                    "label_kick_direction": None,
                    "label_keeper_direction": None,
                    "label_outcome": None,
                    "label_goal": None,
                    "label_footedness": None,
                    "label_camera_direction": None,
                    "label_provenance": "source_provided:soccernet-v2 (penalty award timestamp only)",
                    "license": "Labels CC BY 4.0; video NDA-restricted",
                    "license_url": "https://www.soccer-net.org/data",
                    "attribution": "Deliege et al., SoccerNet-v2",
                    "redistribute_derived": True,
                    "redistribute_video": False,
                    "redistribution_note": "video must not be republished",
                    "ingested_at_utc": utc_now(),
                    "qc_status": "fail" if gated else "pending",
                    "qc_reasons": "video_gated_no_nda_password" if gated else None,
                    "processing_version": PROCESSING_VERSION,
                    "derivation": "source_provided",
                    "model_name": None,
                    "model_version": None,
                    "provenance": f"soccernet-v2:{e['game']}/Labels-v2.json#{e['index']}",
                }
            )
        notes = []
        if gated:
            notes.append(
                f"{len(rows)} penalty annotation(s) inventoried but not processable: "
                "video is behind the SoccerNet NDA and no password is configured."
            )
        return IngestResult(metadata=pd.DataFrame(rows), notes=notes)
