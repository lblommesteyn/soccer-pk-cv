"""Cross-source deduplication.

Two records describe the same physical penalty when they share a
``dedup_key`` -- that is, when they come from the same underlying data
collection (``deposit_family``) and carry the same normalised kick identifier.
This is deliberately conservative: it merges a deposit with its own mirror, and
merges nothing else. Two kicks from unrelated corpora are never fused, because
there is no evidence they are the same event and a wrong merge silently
corrupts every label attached to it.

Within a duplicate group one record is elected *primary*; the others keep their
rows (nothing is deleted) but are marked ``is_primary = False`` and carry a
pointer to the primary. Analyses select ``is_primary``; provenance stays whole.
"""

from __future__ import annotations

import pandas as pd

#: Higher wins. Richness of what the record can support, not its prestige.
MEDIA_RANK = {
    "video": 3,
    "pose_table": 2,
    "render_only": 1,
    "none": 0,
}


def _score(row: pd.Series) -> tuple:
    labelled = int(pd.notna(row.get("label_kick_direction")))
    media = MEDIA_RANK.get(str(row.get("media_kind")), 0)
    frames = int(row.get("n_frames") or 0)
    return (labelled, media, frames, str(row.get("source")))


def resolve_duplicates(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(metadata_with_flags, duplicate_report)``.

    ``duplicate_report`` has one row per non-primary record, naming what it
    duplicates and why -- so the count of "unique PKs after deduplication" is
    always auditable rather than asserted.
    """
    if not len(metadata):
        return metadata, pd.DataFrame(
            columns=["dedup_key", "duplicate_pk_id", "primary_pk_id", "evidence"]
        )

    md = metadata.copy()
    md["is_primary"] = True
    md["duplicate_of_pk_id"] = None

    report_rows = []
    for key, group in md.groupby("dedup_key", sort=False):
        if len(group) == 1:
            continue
        ranked = sorted(group.index, key=lambda i: _score(md.loc[i]), reverse=True)
        primary = ranked[0]
        for idx in ranked[1:]:
            md.at[idx, "is_primary"] = False
            md.at[idx, "duplicate_of_pk_id"] = md.at[primary, "pk_id"]
            evidence = (
                f"same deposit family and kick identifier as {md.at[primary, 'pk_id']} "
                f"(dedup_key={key})"
            )
            md.at[idx, "duplicate_evidence"] = evidence
            report_rows.append(
                {
                    "dedup_key": key,
                    "duplicate_pk_id": md.at[idx, "pk_id"],
                    "duplicate_source": md.at[idx, "source"],
                    "primary_pk_id": md.at[primary, "pk_id"],
                    "primary_source": md.at[primary, "source"],
                    "evidence": evidence,
                }
            )
    return md, pd.DataFrame(report_rows)


def summarise(metadata: pd.DataFrame) -> dict:
    if not len(metadata):
        return {"records": 0, "unique_pks": 0, "duplicates": 0, "by_source": {}}
    return {
        "records": int(len(metadata)),
        "unique_pks": int(metadata["is_primary"].sum()),
        "duplicates": int((~metadata["is_primary"].astype(bool)).sum()),
        "by_source": metadata.groupby("source")["is_primary"].agg(["size", "sum"]).to_dict("index"),
    }
