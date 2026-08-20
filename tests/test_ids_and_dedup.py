import pandas as pd

from pkcv import dedup
from pkcv.ids import make_dedup_key, make_pk_id, make_pk_uid, normalise_identifier


def test_identifier_normalisation_collapses_zero_padding():
    # This is what lets the figshare render "07-03" match the Mendeley kick "7-3".
    assert normalise_identifier("07-03") == normalise_identifier("7-3") == "7-3"
    assert normalise_identifier("1") == "1"
    # Separators collapse to '-', so the same kick written with '/', '_' or '.'
    # resolves to one identifier.
    assert normalise_identifier("england_epl/2015#1#900") == "england-epl-2015#1#900"
    assert normalise_identifier("a.b_c") == normalise_identifier("a/b c") == "a-b-c"


def test_pk_id_is_deterministic_and_namespaced():
    a = make_pk_id("mendeley-women-v2", "07-03")
    b = make_pk_id("mendeley-women-v2", "7-3")
    assert a == b == "mendeley-women-v2:7-3"
    assert make_pk_id("mendeley-epl-v1", "7-3") != a
    assert make_pk_uid(a) == make_pk_uid(a)
    assert len(make_pk_uid(a)) == 16


def test_dedup_key_separates_deposit_families():
    same = make_dedup_key(deposit_family="womens-collegiate-li-pifer", source_identifier="07-03")
    mirror = make_dedup_key(deposit_family="womens-collegiate-li-pifer", source_identifier="7-3")
    other = make_dedup_key(deposit_family="soccerdb", source_identifier="7-3")
    assert same == mirror
    assert same != other


def _record(pk_id, source, dedup_key, media, labelled, n_frames=10):
    return {
        "pk_id": pk_id,
        "source": source,
        "dedup_key": dedup_key,
        "media_kind": media,
        "label_kick_direction": "L" if labelled else None,
        "n_frames": n_frames,
        "duplicate_evidence": None,
    }


def test_labelled_pose_table_wins_over_render_mirror():
    md = pd.DataFrame(
        [
            _record("figshare-women-v2:7-3", "figshare-women-v2", "fam/7-3", "render_only", False, 160),
            _record("mendeley-women-v2:7-3", "mendeley-women-v2", "fam/7-3", "pose_table", True, 80),
        ]
    )
    out, report = dedup.resolve_duplicates(md)
    primary = out[out["is_primary"]]
    assert len(primary) == 1
    assert primary["pk_id"].iloc[0] == "mendeley-women-v2:7-3"
    assert out.loc[out["pk_id"] == "figshare-women-v2:7-3", "duplicate_of_pk_id"].iloc[0] == (
        "mendeley-women-v2:7-3"
    )
    assert len(report) == 1


def test_records_from_different_families_are_never_merged():
    md = pd.DataFrame(
        [
            _record("a:1", "a", "famA/1", "pose_table", True),
            _record("b:1", "b", "famB/1", "pose_table", True),
        ]
    )
    out, report = dedup.resolve_duplicates(md)
    assert out["is_primary"].all()
    assert len(report) == 0
