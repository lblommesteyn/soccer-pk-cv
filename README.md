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

**Sources**

| | |
| --- | --- |
| Penalties discovered across all candidate sources | **603** |
| Real penalty **video**, freely licensed (Wikimedia Commons) | 62 found, **49 usable**, 20 ingested |
| Pose-table penalties, labelled but no footage (Mendeley) | **220** |
| Gated behind an NDA or agreement form (SoccerNet, SoccerDB) | **301** |

**What the vision stage actually recovers**, on the 20 ingested video penalties:

| check | pass | note |
| --- | --- | --- |
| goalkeeper located | 20/20 | |
| ball tracked | 18/20 | |
| **ball contact anchored** | **14/20** | |
| kicker tracked through the run-up | 12/20 | |
| scene scale adequate | 14/20 | below ~85px player height the ball is untrackable |
| goal located on >=60% of frames | 9/20 | the bottleneck |
| **all four core checks pass** | **3/20** | the honest "fully parsed" count |

Read the last row, not the third. A contact anchor alone does not mean the
penalty was parsed correctly -- on one clip the anchor was right while the goal
quad had latched onto a fence and the keeper was a player in midfield. The
corpus is small and its quality is legible per clip in `qc.parquet` rather than
averaged away.

Run `python -m pkcv report` for the live version.

## What it produces

For each penalty clip, anchored on estimated ball contact:

- **goal frame** as a real quadrilateral, recovered per frame so it tracks a
  panning handheld camera;
- **goalkeeper** and **kicker**, identified from the penalty's own geometry
  rather than from appearance, each with a 2D pose;
- **ball** position and trajectory;
- **ball contact**, **keeper commit**, and the canonical snapshot ladder;
- a QC overlay drawn on the source footage showing all of it.

```bash
python scripts/run_clip.py data/raw/commons/clips/<clip>.mp4
```

## Sources, and what each can actually give you

**Wikimedia Commons** is the only open source with real footage. It hosts
spectator- and press-filmed penalty video under CC0, CC BY, CC BY-SA and
public-domain terms, containing the whole scene: shooter, goalkeeper, ball, goal
and net. 62 candidate videos, 49 usable after filtering. It supplies no outcome
labels, so kick direction and result are left null rather than inferred.

**Mendeley `brx9bsxnpx`** (v1 EPL, v2 women's collegiate) supplies 220 labelled
penalties but **only the kicker's 2D pose**. Both versions describe a
`Videos.zip` in their "Steps to reproduce" text; neither actually deposits it,
verified against the Mendeley file API. For those records there is no
goalkeeper, no ball and no goal geometry, and none is estimated -- the columns
exist and carry explicit `missing_reason` values.

**SoccerNet-v2** (173 penalties) and **SoccerDB** (128 segments) are inventoried
but gated behind an NDA and an agreement form. Their adapters are written and
ready; set `SOCCERNET_PASSWORD` to unlock the former.

See [`docs/SOURCES.md`](docs/SOURCES.md) for the full audit, including the
upstream defects found while verifying each one.

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
