"""Dataset card generation.

The card is generated from the tables themselves rather than hand-written, so
the counts it advertises cannot drift away from the data that ships beside it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pkcv.config import Config
from pkcv.io import read_json, read_parquet, utc_now
from pkcv.schemas import PROCESSING_VERSION, SNAPSHOT_OFFSETS_MS


def _fmt_counts(series: pd.Series) -> str:
    return ", ".join(f"`{k}`: {v}" for k, v in series.value_counts().items()) or "none"


def write_card(cfg: Config, out_path: str | Path, repo_id: str) -> Path:
    md = read_parquet(cfg.paths.artifact("metadata"), "metadata")
    snaps = read_parquet(cfg.paths.artifact("temporal_snapshots"), "temporal_snapshots")
    qc = read_parquet(cfg.paths.artifact("qc"), "qc")
    inventory = read_json(cfg.paths.manifests / "source_inventory.json", default=[]) or []

    primary = md[md["is_primary"].astype(bool)] if len(md) else md
    processed = primary[primary["media_kind"].isin(["video", "pose_table"])] if len(primary) else primary

    lines: list[str] = []
    lines.append("---")
    lines.append("license: cc-by-4.0")
    lines.append("task_categories:\n- video-classification")
    lines.append("tags:\n- soccer\n- football\n- penalty-kick\n- pose-estimation\n- sports-analytics")
    lines.append("---\n")
    lines.append("# soccer-pk-cv\n")
    lines.append(
        "Contact-anchored computer-vision representations of soccer penalty kicks, assembled from "
        "openly licensed research deposits.\n"
    )
    lines.append(
        "The dataset is built around one question: **how early before ball contact is the kick "
        "direction readable from the kicker, goalkeeper, ball and their interaction?** Every kick is "
        "therefore anchored on estimated ball contact and sampled at a fixed ladder of offsets "
        f"({', '.join(f'{o} ms' for o in SNAPSHOT_OFFSETS_MS)}).\n"
    )
    lines.append(f"Pipeline version `{PROCESSING_VERSION}`, generated {utc_now()}.\n")

    lines.append("## Contents\n")
    lines.append("| file | rows | one row per |")
    lines.append("| --- | --- | --- |")
    for artifact, unit in [
        ("metadata", "penalty kick"),
        ("tracks", "kick x role x frame"),
        ("poses", "kick x role x frame x keypoint"),
        ("ball", "kick x frame"),
        ("geometry", "kick"),
        ("events", "kick x event"),
        ("temporal_frames", "kick x observed frame"),
        ("temporal_snapshots", "kick x canonical offset"),
        ("qc", "kick x check"),
    ]:
        df = read_parquet(cfg.paths.artifact(artifact), artifact)
        lines.append(f"| `data/{artifact}.parquet` | {len(df):,} | {unit} |")
    lines.append("")

    lines.append("## Corpus\n")
    if len(md):
        lines.append(f"- records ingested: **{len(md):,}**")
        lines.append(f"- unique penalties after deduplication: **{int(md['is_primary'].sum()):,}**")
        lines.append(f"- penalties with usable media (video or pose table): **{len(processed):,}**\n")
        lines.append("| source | records | unique | media | licence |")
        lines.append("| --- | --- | --- | --- | --- |")
        for src, g in md.groupby("source"):
            lines.append(
                f"| `{src}` | {len(g)} | {int(g['is_primary'].sum())} | "
                f"{g['media_kind'].mode().iloc[0]} | {g['license'].iloc[0]} |"
            )
        lines.append("")

    lines.append("## Labels\n")
    if len(processed):
        lines.append(f"- kick direction: {_fmt_counts(processed['label_kick_direction'])}")
        lines.append(f"- outcome (goal=1): {_fmt_counts(processed['label_goal'])}")
        lines.append(f"- footedness: {_fmt_counts(processed['label_footedness'])}")
        lines.append(f"- camera side: {_fmt_counts(processed['label_camera_direction'])}")
        lines.append(
            "\nAll labels are copied verbatim from the upstream deposits; none are inferred by this "
            "pipeline. `label_provenance` records which deposit supplied each one.\n"
        )

    lines.append("## Snapshot availability\n")
    if len(snaps):
        lines.append(
            "How often each canonical offset actually exists. A kick whose clip starts 1.3 s before "
            "contact has no -2000 ms observation, and that row is present-and-unavailable rather "
            "than silently absent -- so availability can be conditioned on.\n"
        )
        lines.append("| offset (ms) | available | share |")
        lines.append("| --- | --- | --- |")
        for off in SNAPSHOT_OFFSETS_MS:
            sub = snaps[snaps["offset_ms"] == off]
            if not len(sub):
                continue
            n = int(sub["snapshot_available"].astype(bool).sum())
            lines.append(f"| {off} | {n} / {len(sub)} | {n / len(sub):.0%} |")
        lines.append("")

    lines.append("## Known limitations\n")
    lines.append(
        "- **Keeper, ball and goal geometry are absent for every currently ingested penalty.** The "
        "two openly licensed penalty deposits publish the kicker's pose only; their clips were never "
        "deposited. These columns are present in the schema and populated with explicit "
        "`missing_reason` values rather than estimates.\n"
        "- **Contact anchoring inherits the upstream marker.** Where a deposit marks several contact "
        "frames (its tracker lost the kicker's identity mid-clip), the latest marker is used, "
        "confidence is halved, and the kick is flagged in `qc.parquet`.\n"
        "- **Clip lengths differ sharply between corpora**, so early offsets are far better covered "
        "in some sources than others. Condition on `snapshot_available` before comparing across "
        "sources, or an apparent effect of time will really be an effect of corpus.\n"
        "- **Outcome labels are imbalanced** in the EPL corpus, which is dominated by scored "
        "penalties; it is not a random sample of penalties taken.\n"
    )

    lines.append("## Sources and licensing\n")
    for rep in inventory:
        lines.append(
            f"- **{rep.get('source')}** -- {rep.get('title')}\n"
            f"  - access: `{rep.get('access')}`, media: `{rep.get('media_kind')}`\n"
            f"  - licence: {rep.get('license')}\n"
            f"  - discovered: {rep.get('n_pk_discovered')}, accessible: {rep.get('n_pk_accessible')}\n"
            f"  - {rep.get('access_note')}"
        )
    lines.append(
        "\nNo original broadcast footage is redistributed here. Video is published only where a "
        "source licence explicitly permits it; otherwise this repository carries the derived "
        "numeric artifacts and a reference to the source.\n"
    )

    lines.append("## Citation\n")
    lines.append("Cite the underlying deposits, not just this aggregation:\n")
    for rep in inventory:
        if rep.get("doi"):
            lines.append(f"- {rep.get('attribution')} -- https://doi.org/{rep['doi']}")
    lines.append(f"\nPipeline: `{repo_id}`, code at https://github.com/lblommesteyn/soccer-pk-cv\n")

    if len(qc):
        lines.append("## QC\n")
        lines.append("| check | pass | warn | fail | n/a |")
        lines.append("| --- | --- | --- | --- | --- |")
        for check, g in qc.groupby("check"):
            c = g["status"].value_counts()
            lines.append(
                f"| `{check}` | {c.get('pass', 0)} | {c.get('warn', 0)} | "
                f"{c.get('fail', 0)} | {c.get('na', 0)} |"
            )
        lines.append("")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
