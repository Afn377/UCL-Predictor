from pathlib import Path

import pytest

from src.data.download_ucl import parse_score
from src.data.team_names import club_id, load_team_name_mapping, normalize_team_name


@pytest.mark.parametrize(
    "text,normal,extra,pen",
    [
        ("2-1 (1-0)", (2, 1), (0, 0), (None, None)),
        ("3-4 pen. 1-1 a.e.t. (1-1, 0-1)", (1, 1), (0, 0), (3, 4)),
        ("4-3 a.e.t. (3-3, 2-0)", (3, 3), (1, 0), (None, None)),
        ("2-4 pen. 1-0 a.e.t. (1-0, 1-0)", (1, 0), (0, 0), (2, 4)),
    ],
)
def test_regulation_score(text, normal, extra, pen):
    result = parse_score(text)
    assert (result["home_goals"], result["away_goals"]) == normal
    assert (result["home_extra_time_goals"], result["away_extra_time_goals"]) == extra
    assert (result["home_penalties"], result["away_penalties"]) == pen


def test_extra_time_requires_normal_score():
    with pytest.raises(ValueError, match="regulation"):
        parse_score("3-2 a.e.t.")


def test_aliases_share_id():
    mapping = load_team_name_mapping(Path("src/data/team_name_mapping.csv"))
    assert club_id(normalize_team_name("SL Benfica (POR)", mapping)) == club_id(
        normalize_team_name("Sport Lisboa e Benfica (POR)", mapping)
    )


