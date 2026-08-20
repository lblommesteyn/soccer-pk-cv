"""Parquet IO with schema enforcement, atomic writes and idempotent upserts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pkcv.schemas import ARTIFACT_SCHEMAS


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def coerce(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    """Project ``df`` onto ``schema``: add missing nullable columns, drop extras.

    Raises if a non-nullable column is absent, which is the point -- a silent
    all-null ``pk_id`` would poison every join downstream.
    """
    df = df.copy()
    for field in schema:
        if field.name not in df.columns:
            if not field.nullable:
                raise ValueError(f"required column {field.name!r} missing from frame")
            df[field.name] = None
    df = df[[f.name for f in schema]]
    for field in schema:
        if pa.types.is_boolean(field.type):
            df[field.name] = df[field.name].fillna(False).astype(bool)
        elif pa.types.is_string(field.type):
            df[field.name] = df[field.name].astype("object").where(df[field.name].notna(), None)
            df[field.name] = df[field.name].map(lambda v: None if v is None else str(v))
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False)


def write_parquet(df: pd.DataFrame, path: str | Path, artifact: str) -> Path:
    """Atomically write ``df`` as ``artifact``. Never leaves a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = coerce(df, ARTIFACT_SCHEMAS[artifact])
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def read_parquet(path: str | Path, artifact: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        if artifact:
            return pd.DataFrame({f.name: pd.Series(dtype="object") for f in ARTIFACT_SCHEMAS[artifact]})
        return pd.DataFrame()
    return pq.read_table(path).to_pandas()


def upsert_parquet(
    df: pd.DataFrame,
    path: str | Path,
    artifact: str,
    key: list[str],
) -> Path:
    """Replace every existing row group matching ``df``'s key values, then append.

    This is what makes ``process`` resumable *and* idempotent: reprocessing a
    penalty replaces exactly its own rows and touches nothing else.
    """
    existing = read_parquet(path, artifact)
    if len(existing) and len(df):
        incoming_keys = set(map(tuple, df[key].astype(str).itertuples(index=False, name=None)))
        mask = existing[key].astype(str).apply(tuple, axis=1).isin(incoming_keys)
        existing = existing[~mask]
    combined = pd.concat([existing, df], ignore_index=True) if len(existing) else df
    return write_parquet(combined, path, artifact)


def write_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)
    return path


def read_json(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
