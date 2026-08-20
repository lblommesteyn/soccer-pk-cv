# Source inventory

What each candidate corpus actually contains, verified against its API rather
than its landing page. Regenerate the machine-readable version with
`python -m pkcv inventory` (writes `data/manifests/source_inventory.json`).

| source | penalties | access | media | licence |
| --- | --- | --- | --- | --- |
| `mendeley-epl-v1` | 88 | open | pose table | CC BY 4.0 |
| `mendeley-women-v2` | 132 in file (133 claimed) | open | pose table | CC BY 4.0 |
| `figshare-women-v2` | 132 | open | skeleton renders | CC BY 4.0 |
| `soccernet-v2` | 173 | **gated** (NDA) | broadcast video | labels CC BY 4.0, video NDA |
| `soccerdb` | 128 | **gated** (agreement form) | broadcast video | annotations unstated, video gated |

Total discovered: **554**. Openly accessible: **352 records → 221 unique
penalties** after deduplication.

---

## Mendeley Data `brx9bsxnpx`

The single most important structural fact: **the two "datasets" are two
versions of one Mendeley record**, not two records.

- **v1** — *EPL Penalty Kick Data Recorded by a Computer Vision Model*.
  88 penalties, 4,396 frames, 2023-24 Premier League broadcast footage.
  File: `kicker_pose_keypoints.csv` (2.6 MB).
- **v2** — *Women's Soccer Penalty Kick Data Captured by Computer Vision
  Models*. 132 penalties, 11,140 frames, US collegiate women, training and game.
  File: `penalty_pose_keypoints.csv` (7.3 MB).

Because Mendeley surfaces only the newest version by default, v1 is easy to
miss entirely. Both are fetched here by explicit version number.

**They do not overlap.** v2 replaces v1's population rather than extending it:
different competition, different gender, different identifier scheme
(`1`, `4`, `6`… vs `1-1`, `1-2`, `7-3`…). Cross-version deduplication is
therefore a no-op, and the pipeline demonstrates that rather than asserting it —
they carry different `deposit_family` values, and the resolved duplicate count
between them is 0.

### No video is deposited

Both versions' "Steps to reproduce" text instructs the reader to "run the model
on the clips contained in the `Videos.zip` file in the Instructions and Code
folder". **That folder is not published under either version.** The public file
API returns exactly one CSV per version.

This single fact determines the shape of the whole dataset: keeper, ball and
goal geometry cannot be derived, because there is nothing to derive them from.

### Schema

Both CSVs share a core: `kick_id`, `frame`, `last_touch`, `track_id`,
`camera_direction`, `r_or_l` (footedness), `kick_direction` (L/C/R), `goal`,
smoothed `bbox_*`, and 12 COCO joints (indices 5-16) in both absolute pixel and
box-centred coordinates. v2 adds `kicker`, `fps` and `game_or_training`.

### Defects found while ingesting

| finding | where | handling |
| --- | --- | --- |
| 13 EPL kicks carry **two** `last_touch=1` markers, each at the end of a different `track_id` — the upstream tracker lost identity mid-clip, so the kick's rows are two people | v1 | contact taken from the latest marker at halved confidence; the timeline keeps only the fragment containing contact; both fragments stay in `tracks.parquet`; QC flags `kicker_track_single_identity` |
| 4 EPL and 4 women's kicks carry **no** `last_touch` marker | v1, v2 | contact recorded as missing; the kick produces no temporal frames rather than a guessed anchor |
| 1 row has a blank `frame` index | v1 | dropped, reported in ingest notes |
| 26 rows have null `last_touch` (kicks 15, 73) | v1 | treated as 0 |
| `r_or_l` is `R` for **all 132** women's kicks | v2 | footedness is constant and carries no information in that corpus |
| `goal` is 1 for 82 of 88 EPL kicks | v1 | the EPL corpus is not a random sample of penalties taken; noted on the dataset card |
| deposit describes **133** penalties; the CSV contains **132** | v2 | see below |

### The 132 / 133 discrepancy, resolved

The v2 description claims 133 penalties; its CSV holds 132; the figshare mirror
ships 132 renders. Neither is a subset of the other:

- kick `15-3` has a render but **no rows** in the pose CSV
- kick `5-4` has rows in the pose CSV but **no render**

The union is 133 — which is where the description's number comes from. Both
publication routes are each missing a different kick. The pipeline surfaces this
automatically: `15-3` survives deduplication as a render-only primary record
with no labels.

---

## figshare 31464526 — mirror of Mendeley v2

Same authors, same 132 kicks, same CC BY 4.0. It is **not** a new corpus, and
every one of its records deduplicates against `mendeley-women-v2`.

What it adds is `Skeleton Videos.zip`: one 1920×1080 MP4 per kick. These are
skeleton strokes on a black background — measured at ingest, the non-black pixel
fraction of the first frame is 0.2–0.5%, against a 5% classification threshold.
All 132 classify as `render_only`. There is no footage, so running a detector on
them would return numbers about nothing.

Their value is twofold: they carry authoritative per-kick frame counts and
container fps, and they provide **independent evidence for verifying the
ingest** — see `scripts/verify_against_upstream_render.py`, which confirms 99.2%
of our published keypoints land within 6 px of the depositors' own ink.

---

## SoccerNet-v2 — gated

173 `Penalty` annotations across 148 of the 500 matches. Labels download without
credentials and are counted here; video requires a signed NDA and an issued
password. The pipeline never attempts to work around that: without
`SOCCERNET_PASSWORD` the source reports `gated` and contributes zero penalties.

Two caveats that would matter even with access:

- `Penalty` marks the **award** of a penalty, not the kick. The kick follows by
  a variable delay of tens of seconds. A window around the label is a search
  region, not a clip.
- Annotation timestamps are at 1 s resolution — 25 to 50 frames. Far too coarse
  to serve as the contact anchor; contact would have to be re-derived from video
  by `pkcv/vision/`.

---

## SoccerDB — gated

128 penalty-kick segments (class 8) across 63 half-videos, from the public
`seg_info.csv` in the project's GitHub repository. Unlike SoccerNet, each
segment carries `event_start_time` and `event_end_time` — **event-tight bounds
around the kick itself**, which is what a clip cutter needs.

37 of those half-videos map to SoccerNet matches through the repository's own
`SoccerDB2SoccerNet.csv` crosswalk, so a SoccerNet NDA holder could cut that
subset directly using SoccerDB's tighter bounds. The remainder need a separate
agreement form emailed to the depositor.

The repository carries no licence file, so its annotation tables are "publicly
posted, licence unstated". They are used here to count and locate penalties;
neither they nor anything derived from them is republished.

---

## Ruled out

- **arXiv 2507.12617** (1,010 → 640 web-scraped penalty clips): no data
  availability statement, no DOI, no release. The clips were scraped from
  copyrighted broadcasts and are not redistributable in any case.
- **Kaggle penalty datasets** (EPL/Serie A/Liga Portugal outcomes, World Cup
  shootouts): outcome tables only, no video and no frame-level data.
- **Soccer event *image* datasets** (e.g. 81 penalty stills): single frames, so
  there is no run-up and no timeline to anchor.
