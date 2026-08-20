# soccer-pk-cv

An end-to-end, provenance-correct pipeline that turns openly licensed soccer
penalty-kick research data into a contact-anchored dataset for one question:

> **How early before ball contact can kick direction be predicted from the
> kicker, goalkeeper, ball and their interaction?**

Every kick is anchored on estimated ball contact and sampled at a fixed ladder
of offsets — **-2000, -1500, -1000, -750, -500, -250, 0 ms** — so "how early" is
a column, not an analysis choice made later.

```
authorized sources → manifests → CV / pose ingest → canonical per-PK tables
                                → contact anchoring → QC → Hugging Face
```

## Status

| | |
| --- | --- |
| Penalties discovered across all candidate sources | **554** |
| Openly accessible and ingested | **352 records → 221 unique** |
| Processed end to end | **220** |
| Contact frame resolved | **96.4%** |
| Snapshot available at -1000 ms / 0 ms | **90.9% / 96.4%** |
| Keeper, ball, goal geometry recovered | **0 — see [the honest limitation](#the-central-limitation)** |

Run `python -m pkcv report` for the live version of this table.

## The central limitation

Read this before using the data.

The two openly licensed penalty deposits (Mendeley `brx9bsxnpx` v1 and v2)
publish **the kicker's 2D pose and nothing else**. Both describe a `Videos.zip`
in their "Steps to reproduce" text, but neither version actually deposits it —
verified against the Mendeley file API, not inferred from the landing page. The
figshare mirror ships MP4s, but they are skeleton renders on a black background
(measured: under 0.5% non-black pixels), not footage.

So for every penalty currently in this dataset there is **no goalkeeper, no
ball, and no goal geometry** — and none is estimated. Those columns exist in the
schema and are filled with explicit `missing_reason` values. A pipeline that
quietly produced keeper positions from a source containing no keeper would be
worse than one that produces none.

The video-capable half of the pipeline (`pkcv/vision/`) is implemented, tested
and ready; it activates the moment a source with real footage becomes
accessible. Two such sources are inventoried and blocked only on credentials:

- **SoccerNet-v2** — 173 penalty annotations across 148 matches. Video is under
  an NDA; set `SOCCERNET_PASSWORD` to unlock.
- **SoccerDB** — 128 penalty segments with event-tight bounds, 37 of whose
  half-videos map onto SoccerNet matches. Needs a signed agreement form.

## Install

```bash
git clone https://github.com/lblommesteyn/soccer-pk-cv
cd soccer-pk-cv
pip install -e .            # ingest / temporal / qc / publish
pip install -e .[vision]    # adds torch + ultralytics for the video path
```

The vision stage is sized for a 10 GB RTX 3080: YOLO11m at 960 px, which leaves
room for a second process on the same card. No large GPU is required.

## Use

```bash
python -m pkcv inventory                     # what exists, licence, accessibility
python -m pkcv ingest --source mendeley      # download + map onto the schema
python -m pkcv process --limit 5             # validate on five kicks first
python -m pkcv process --all                 # then the whole accessible corpus
python -m pkcv qc --overlays 8               # grade + render visual QC
python -m pkcv publish --dry-run             # stage, licence-filter, write the card
python -m pkcv publish                       # upload (needs HF_TOKEN)
python -m pkcv report                        # the numbers above
```

Every stage is **resumable and idempotent**: downloads are checksum-verified and
skipped, and every parquet write is an upsert keyed on `pk_id`, so reprocessing
one penalty rewrites exactly its own rows.

## Layout

```
pkcv/
  sources/     one adapter per corpus: inventory() → fetch() → ingest()
  vision/      detection, tracking, pose, ball, goal geometry, events
  temporal/    contact anchoring, pose features, canonical snapshots
  schemas/     canonical pyarrow schemas + provenance spine
  qc/          per-kick checks and visual QC overlays
  hf/          licence-filtered publishing and dataset card
data/
  raw/         downloads (gitignored)
  manifests/   source inventory, duplicate resolution
  processed/   metadata.parquet, tracks/, poses/, ball/, temporal/, qc/
  qc/          overlays and summaries
```

See [`docs/SCHEMA.md`](docs/SCHEMA.md) for the tables and
[`docs/SOURCES.md`](docs/SOURCES.md) for what each source actually contains,
including the discrepancies found while verifying them.

## Design rules

**Never fabricate a failed estimate.** A missing value is a row with
`is_missing = True` and a `missing_reason`. This is enforced structurally: the
schema declares every estimate column nullable, and `tests/test_schema_and_qc.py`
fails the build if that ever changes.

**Never infer a label the source already supplies.** `label_*` columns are
copied verbatim; `label_provenance` names the deposit each came from.

**A missing snapshot is a present row.** If a clip starts 1.3 s before contact,
its -2000 ms row exists with `snapshot_available = False` and
`unavailable_reason = "before_clip_start"`. Dropping it would silently turn
"how early can direction be predicted?" into "how early, among clips that
happened to be long enough?" — an easier and different question.

**Deduplicate conservatively.** Records merge only within a `deposit_family`
(one collection, however many publication routes) on a normalised kick id.
Kicks from unrelated corpora are never fused: a wrong merge corrupts every
label attached to it, silently.

**Licence decisions are per record.** Derived tables ship where the source
permits derivatives; video ships only where a licence explicitly allows
redistribution. Nothing is withheld silently — the upload manifest lists it.

## Verification

`scripts/verify_against_upstream_render.py` checks the ingest against evidence
the pipeline did not produce: the depositors' own skeleton renders. Our
coordinates, drawn from the published parquet, must land on their ink.

```
kick  frames  keypoints  within_tol  hit_rate
 1-1     144       1715        1715    1.0000
 1-2     104       1240        1228    0.9903
 1-3     136       1628        1616    0.9926
 1-4      91       1092        1080    0.9890
 1-5      74        887         875    0.9865
mean hit rate: 0.992 (tol=6px)
```

That the frame indices align too rules out an off-by-one in the timeline, which
would otherwise shift every anchor.

## Licensing

The ingested corpora are CC BY 4.0. This repository redistributes no broadcast
footage. Cite the underlying deposits:

- Li, F. & Pifer, N. D. *EPL Penalty Kick Data Recorded by a Computer Vision
  Model.* Mendeley Data, [10.17632/brx9bsxnpx.1](https://doi.org/10.17632/brx9bsxnpx.1)
- Li, F. & Pifer, N. D. *Women's Soccer Penalty Kick Data Captured by Computer
  Vision Models.* Mendeley Data, [10.17632/brx9bsxnpx.2](https://doi.org/10.17632/brx9bsxnpx.2)
  and figshare [10.6084/m9.figshare.31464526](https://doi.org/10.6084/m9.figshare.31464526)
- Deliège et al. *SoccerNet-v2.* CVPR-W 2021 (annotations only)
- Jiang et al. *SoccerDB.* ACM MMSPORTS 2020 (annotations only)

Code is MIT.
