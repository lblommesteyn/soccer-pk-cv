"""Wikimedia Commons -- freely licensed penalty-kick video.

This is the source that makes the project's original goal reachable. Unlike the
Mendeley deposits (pose tables, no footage) and unlike SoccerNet/SoccerDB
(footage behind an NDA or an agreement form), Commons hosts real penalty video
under CC0, CC BY, CC BY-SA and public-domain terms: freely downloadable, freely
redistributable with attribution, and containing the whole scene -- shooter,
goalkeeper, ball, goal and net.

Two things this adapter has to be careful about.

**"Penalty" is not unique to association football.** A plain text search returns
ice-hockey penalty shots, rugby penalty goals and even a wrestling move by the
same name. Titles are filtered against a sport blocklist, and anything that
survives is still only a *candidate* until the vision stage finds a goal with a
net in it -- which is the real test.

**Clip quality varies enormously**, because these are filmed by spectators. Some
are 4K from behind the goal; others are a wide stadium view where the kicker is
forty pixels tall. Frame size and duration are recorded at ingest, and the QC
stage grades what was actually recoverable rather than assuming.

Every file's licence and author are read from the Commons API per file and
carried into the metadata, so attribution is per record, never assumed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from pkcv.ids import make_dedup_key, make_pk_id, make_pk_uid
from pkcv.io import sha256_file, utc_now
from pkcv.schemas import PROCESSING_VERSION
from pkcv.sources.base import (
    ACCESS_ERROR,
    ACCESS_OPEN,
    MEDIA_VIDEO,
    IngestResult,
    SourceAdapter,
    SourceReport,
)

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "pkcv/0.1 (research; https://github.com/lblommesteyn/soccer-pk-cv)"}

SEARCH_TERMS = (
    "penalty kick football",
    "penalty shootout football",
    "penalti futbol",
    "penaltı",
    "Elfmeter",
    "penalty soccer goalkeeper",
    "rzut karny",
    "tir au but penalty",
)

#: Other sports use the word "penalty" for something that is not a penalty kick.
OTHER_SPORT = re.compile(
    r"\b(hockey|kometa|zlin|jesenice|triglav|nhl|khl|puck|"
    r"rugby|maroochydore|caloundra|ulster|line ?out|"
    r"wrestling|sabre|"
    r"handball|futsal|beach ?soccer|"
    r"basketball|netball|water ?polo)\b",
    re.IGNORECASE,
)

#: Licences that permit redistribution at all. Anything not matching is
#: downloaded for local analysis only and its video is never republished.
REDISTRIBUTABLE = re.compile(r"^(cc0|cc by|public domain|pd-|no restrictions)", re.IGNORECASE)

#: Share-alike files may be redistributed, but only under the same licence.
#: Publishing them alongside permissively licensed files without saying so would
#: misrepresent the terms, so the obligation is recorded per record.
SHARE_ALIKE = re.compile(r"(?:^|[-\s])sa(?:[-\s]|\d|$)", re.IGNORECASE)


def _api(params: dict[str, Any]) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _strip_html(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"<[^>]+>", "", value).strip() or None


class WikimediaCommonsPenalties(SourceAdapter):
    slug = "commons"
    title = "Wikimedia Commons -- freely licensed penalty-kick video"
    deposit_family = "wikimedia-commons"

    # ------------------------------------------------------------- discovery

    @property
    def clips_dir(self) -> Path:
        p = self.raw_dir / "clips"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def index_path(self) -> Path:
        return self.raw_dir / "commons_index.json"

    def _search(self) -> dict[str, str]:
        titles: dict[str, str] = {}
        for term in self.source_cfg.get("search_terms", SEARCH_TERMS):
            d = _api({
                "action": "query", "list": "search",
                "srsearch": f"{term} filetype:video",
                "srnamespace": "6", "srlimit": "50",
            })
            for r in d.get("query", {}).get("search", []):
                titles.setdefault(r["title"], term)
            time.sleep(0.3)
        return titles

    def _file_info(self, titles: list[str]) -> list[dict[str, Any]]:
        out = []
        for i in range(0, len(titles), 20):
            d = _api({
                "action": "query", "titles": "|".join(titles[i:i + 20]),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|mediatype|extmetadata",
            })
            for page in d.get("query", {}).get("pages", {}).values():
                ii = (page.get("imageinfo") or [{}])[0]
                if ii.get("mediatype") != "VIDEO":
                    continue
                em = ii.get("extmetadata", {})
                lic = (em.get("LicenseShortName") or {}).get("value")
                out.append({
                    "title": page["title"],
                    "url": ii.get("url"),
                    "descriptionurl": ii.get("descriptionurl"),
                    "size_bytes": ii.get("size"),
                    "width": ii.get("width"),
                    "height": ii.get("height"),
                    "duration_s": round(float(ii.get("duration") or 0), 2),
                    "license": lic,
                    "license_url": (em.get("LicenseUrl") or {}).get("value"),
                    "artist": _strip_html((em.get("Artist") or {}).get("value")),
                    "credit": _strip_html((em.get("Credit") or {}).get("value")),
                    "redistributable": bool(lic and REDISTRIBUTABLE.match(lic)),
                    "share_alike": bool(lic and SHARE_ALIKE.search(lic)),
                })
            time.sleep(0.3)
        return out

    def _candidates(self) -> list[dict[str, Any]]:
        titles = self._search()
        infos = self._file_info(list(titles))
        min_w = int(self.source_cfg.get("min_width", 640))
        max_dur = float(self.source_cfg.get("max_duration_s", 240))
        kept = []
        for f in infos:
            name = f["title"]
            if OTHER_SPORT.search(name):
                f["excluded"] = "title names a different sport"
            elif (f["width"] or 0) < min_w:
                f["excluded"] = f"width {f['width']} below {min_w}"
            elif f["duration_s"] > max_dur:
                f["excluded"] = f"duration {f['duration_s']}s above {max_dur}s"
            elif not f["license"]:
                f["excluded"] = "no licence recorded"
            else:
                f["excluded"] = None
                kept.append(f)
        self._all_infos = infos
        return kept

    # -------------------------------------------------------------- inventory

    def inventory(self) -> SourceReport:
        report = SourceReport(
            source=self.slug,
            title=self.title,
            url="https://commons.wikimedia.org/",
            access=ACCESS_OPEN,
            media_kind=MEDIA_VIDEO,
            license="per file: CC0 / CC BY / CC BY-SA / public domain",
            license_url="https://commons.wikimedia.org/wiki/Commons:Licensing",
            attribution="per file; recorded in metadata.attribution",
            deposit_family=self.deposit_family,
            redistribute_derived=True,
            redistribute_video=True,
            redistribution_note=(
                "Redistribution is decided per file from its own licence. Share-alike files are "
                "kept for local analysis and not republished, because republishing them would "
                "impose CC BY-SA on the aggregate."
            ),
            labels_available=[],
            checked_at_utc=utc_now(),
        )
        try:
            kept = self._candidates()
        except Exception as exc:
            report.access = ACCESS_ERROR
            report.access_note = f"Commons search failed: {type(exc).__name__}: {exc}"
            return report

        report.files = kept
        report.n_pk_discovered = len(self._all_infos)
        report.n_pk_accessible = len(kept)
        report.access_note = (
            f"public download, no credential required; {len(kept)} of "
            f"{len(self._all_infos)} candidate videos pass the sport/quality filter"
        )
        report.notes.append(
            "'Penalty' also names events in hockey, rugby and wrestling; titles matching other "
            "sports are excluded and the vision stage is the real test (it must find a goal "
            "with a net)."
        )
        n_share = sum(1 for f in kept if f.get("share_alike"))
        report.notes.append(
            f"{sum(1 for f in kept if f['redistributable'])} of {len(kept)} permit video "
            f"redistribution with attribution; {n_share} of those are share-alike, so any "
            "republished copy must carry the same licence."
        )
        report.notes.append(
            "Spectator-filmed, so framing varies from behind-the-goal 4K to a wide stand view "
            "where the kicker is a few dozen pixels tall. QC grades what was recoverable."
        )
        return report

    # ------------------------------------------------------------------ fetch

    @staticmethod
    def _slug(title: str) -> str:
        base = title.replace("File:", "")
        base = os.path.splitext(base)[0]
        base = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_").lower()
        return base[:60]

    def _download(self, info: dict[str, Any]) -> Path | None:
        ext = os.path.splitext(info["title"])[1] or ".webm"
        raw = self.clips_dir / f"{self._slug(info['title'])}{ext}"
        if raw.exists() and raw.stat().st_size > 0:
            return raw
        for attempt in range(4):
            try:
                req = urllib.request.Request(info["url"], headers=UA)
                with urllib.request.urlopen(req, timeout=900) as r, open(raw, "wb") as fh:
                    while chunk := r.read(1 << 20):
                        fh.write(chunk)
                return raw
            except Exception:
                if raw.exists():
                    raw.unlink()
                time.sleep(6 * (attempt + 1))
        return None

    @staticmethod
    def _transcode(src: Path) -> Path | None:
        """Normalise to H.264 mp4 so OpenCV seeks reliably.

        Commons serves WebM/VP9 and Ogg/Theora. OpenCV can often decode them but
        frame-accurate seeking is unreliable, and every stage here indexes by
        frame number, so a normalise pass is cheaper than the bugs.
        """
        dst = src.with_suffix(".mp4")
        if dst.exists() and dst.stat().st_size > 0:
            return dst
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src), "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", "-an", str(dst),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=1800)
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return None
        return dst if dst.exists() else None

    def fetch(self, limit: int | None = None) -> SourceReport:
        report = self.inventory()
        if report.access != ACCESS_OPEN:
            return report
        picks = report.files[: limit] if limit is not None else report.files
        fetched = []
        for info in picks:
            raw = self._download(info)
            if raw is None:
                report.notes.append(f"download failed: {info['title']}")
                continue
            mp4 = self._transcode(raw)
            if mp4 is None:
                report.notes.append(f"transcode failed (ffmpeg missing?): {info['title']}")
                continue
            info["local_raw"] = str(raw)
            info["local_mp4"] = str(mp4)
            fetched.append(info)
        with open(self.index_path, "w", encoding="utf-8") as fh:
            json.dump(fetched, fh, indent=1)
        report.notes.append(f"{len(fetched)} clip(s) downloaded and normalised to mp4")
        return report

    # ----------------------------------------------------------------- ingest

    def ingest(self, limit: int | None = None) -> IngestResult:
        import cv2

        if not self.index_path.exists():
            raise FileNotFoundError(f"{self.index_path} missing; run `pkcv ingest` first")
        with open(self.index_path, encoding="utf-8") as fh:
            entries = json.load(fh)
        if limit is not None:
            entries = entries[:limit]

        rows = []
        for e in entries:
            mp4 = Path(e["local_mp4"])
            if not mp4.exists():
                continue
            cap = cv2.VideoCapture(str(mp4))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or None
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
            cap.release()

            ident = self._slug(e["title"])
            pk_id = make_pk_id(self.slug, ident)
            attribution = "; ".join(
                x for x in (e.get("artist"), e.get("credit"), e.get("descriptionurl")) if x
            )
            rows.append({
                "pk_id": pk_id,
                "pk_uid": make_pk_uid(pk_id),
                "source": self.slug,
                "source_identifier": e["title"],
                "source_dataset_doi": None,
                "source_version": None,
                "dedup_key": make_dedup_key(
                    deposit_family=self.deposit_family, source_identifier=ident
                ),
                "is_primary": True,
                "duplicate_of_pk_id": None,
                "duplicate_evidence": None,
                "media_kind": MEDIA_VIDEO,
                "has_video": True,
                "video_relpath": str(mp4.relative_to(self.cfg.paths.root)).replace("\\", "/"),
                "video_sha256": sha256_file(mp4),
                "fps": fps,
                "n_frames": n,
                "frame_width": w,
                "frame_height": h,
                "clip_duration_s": (n / fps) if (n and fps) else None,
                "competition": None,
                "gender": None,
                "level": None,
                "season": None,
                "match_context": "game",
                "kicker_ref": None,
                # Commons supplies no penalty labels. Direction and outcome are
                # deliberately left null rather than inferred: an inferred label
                # would be a model output masquerading as ground truth.
                "label_kick_direction": None,
                "label_keeper_direction": None,
                "label_outcome": None,
                "label_goal": None,
                "label_footedness": None,
                "label_camera_direction": None,
                "label_provenance": "none (Commons supplies no penalty outcome labels)",
                "license": e.get("license") or "unknown",
                "license_url": e.get("license_url"),
                "attribution": attribution or e["title"],
                "redistribute_derived": True,
                "redistribute_video": bool(e.get("redistributable")),
                "redistribution_note": (
                    (
                        "share-alike: redistributable with attribution, but any copy must "
                        "carry the same licence"
                        if e.get("share_alike")
                        else "licence permits redistribution with attribution"
                    )
                    if e.get("redistributable")
                    else "licence unclear or non-free; video kept local, derived data only"
                ),
                "ingested_at_utc": utc_now(),
                "qc_status": "pending",
                "qc_reasons": None,
                "processing_version": PROCESSING_VERSION,
                "derivation": "source_provided",
                "model_name": None,
                "model_version": None,
                "provenance": f"wikimedia-commons:{e['title']}",
            })

        return IngestResult(
            metadata=pd.DataFrame(rows),
            notes=[
                f"{len(rows)} penalty clip(s) registered as video",
                "No outcome labels exist for this source; kick direction must be annotated "
                "separately or derived and clearly marked as derived.",
            ],
        )
