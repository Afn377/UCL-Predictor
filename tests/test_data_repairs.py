from pathlib import Path

import pandas as pd
import pytest

from src.data.download_ucl import parse_score
from src.data.team_names import club_id, load_team_name_mapping, normalize_team_name
from src.features.form import add_rest_and_congestion_features
from src.features.strength import (
    STRUCTURAL_FEATURES,
    add_strength_features,
    attach_clubelo,
)


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


def test_rest_is_unknown_after_missing_seasons():
    data = pd.DataFrame(
        {
            "team": ["A"] * 3,
            "date": pd.to_datetime(["2015-01-01", "2020-01-01", "2020-01-04"]),
        }
    )
    result = add_rest_and_congestion_features(data)
    assert pd.isna(result.iloc[1].days_since_match_before)
    assert result.iloc[2].days_since_match_before == 3


def fixtures():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-04-01", "2024-04-08", "2024-04-15"]),
            "competition": ["Champions League"] * 3,
            "season": ["2023_24"] * 3,
            "stage": ["quarterfinal", "quarterfinal", "semifinal"],
            "home_team": ["A", "B", "A"],
            "away_team": ["B", "A", "B"],
            "home_goals": [2, 0, 5],
            "away_goals": [0, 1, 0],
        }
    )


def test_strength_and_ties_ignore_current_and_future_scores():
    data = fixtures()
    actual = add_strength_features(data)
    changed = data.copy()
    changed.loc[1:, ["home_goals", "away_goals"]] = 10
    other = add_strength_features(changed)
    pd.testing.assert_frame_equal(
        actual.iloc[:2][STRUCTURAL_FEATURES], other.iloc[:2][STRUCTURAL_FEATURES]
    )
    assert actual.iloc[1].second_leg == 1
    assert actual.iloc[1].aggregate_lead == -2
    assert actual.iloc[2].second_leg == 0


def test_neutral_final_and_missing_xg():
    data = fixtures().iloc[:1].assign(stage="final")
    result = add_strength_features(data)
    assert result.iloc[0].neutral_venue == 1
    assert result.iloc[0].home_xg_power_attack == 0


def test_clubelo_never_reads_same_day_or_future_rating():
    data = fixtures()
    history = pd.DataFrame(
        {
            "team": ["A", "A", "B"],
            "date": ["2024-03-30", "2024-04-01", "2024-03-30"],
            "elo": [1600, 2500, 1500],
        }
    )
    result = attach_clubelo(data, history)
    assert result.iloc[0].clubelo_diff == 100
