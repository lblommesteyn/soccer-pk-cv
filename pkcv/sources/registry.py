"""Source registry: the single place a new corpus is wired in."""

from __future__ import annotations

from pkcv.sources.base import SourceAdapter
from pkcv.sources.commons import WikimediaCommonsPenalties
from pkcv.sources.figshare_pk import FigshareWomenMirror
from pkcv.sources.mendeley_pk import MendeleyEPLv1, MendeleyWomenV2
from pkcv.sources.soccerdb import SoccerDBPenalties
from pkcv.sources.soccernet import SoccerNetPenalties

ADAPTERS: dict[str, type[SourceAdapter]] = {
    cls.slug: cls
    for cls in (
        WikimediaCommonsPenalties,
        MendeleyEPLv1,
        MendeleyWomenV2,
        FigshareWomenMirror,
        SoccerNetPenalties,
        SoccerDBPenalties,
    )
}

#: Convenience aliases accepted by `--source`.
ALIASES = {
    "commons": ["commons"],
    "video": ["commons"],
    "mendeley": ["mendeley-epl-v1", "mendeley-women-v2"],
    "figshare": ["figshare-women-v2"],
    "soccernet": ["soccernet-v2"],
    "soccerdb": ["soccerdb"],
    "all": list(ADAPTERS),
}


def resolve(names: list[str] | None) -> list[str]:
    if not names:
        return list(ADAPTERS)
    out: list[str] = []
    for name in names:
        for slug in ALIASES.get(name, [name]):
            if slug not in ADAPTERS:
                raise KeyError(f"unknown source {slug!r}; known: {sorted(ADAPTERS)}")
            if slug not in out:
                out.append(slug)
    return out


def build(slug: str, cfg) -> SourceAdapter:
    return ADAPTERS[slug](cfg, cfg.source_config(slug))
