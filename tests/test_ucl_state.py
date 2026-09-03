import pandas as pd
import pytest

from src.simulation.ucl_state import (
    build_league_phase_table,
    league_phase_matches,
    load_ucl_matches,
)


def make_matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-09-16"),
                "season": "2025_26",
                "competition": "Champions League",
                "stage": "league_phase",
                "home_team": "Arsenal",
                "away_team": "PSV",
                "home_goals": 2,
                "away_goals": 0,
            },
            {
                "date": pd.Timestamp("2025-09-17"),
                "season": "2025_26",
                "competition": "Champions League",
                "stage": "league_phase",
                "home_team": "PSV",
                "away_team": "Dortmund",
                "home_goals": 1,
                "away_goals": 1,
            },
            {
                "date": pd.Timestamp("2025-09-18"),
                "season": "2025_26",
                "competition": "Champions League",
                "stage": "round_of_16",
                "home_team": "Arsenal",
                "away_team": "Dortmund",
                "home_goals": 1,
                "away_goals": 2,
            },
            {
                "date": pd.Timestamp("2024-09-16"),
                "season": "2024_25",
                "competition": "Champions League",
                "stage": "league_phase",
                "home_team": "Arsenal",
                "away_team": "Dortmund",
                "home_goals": 0,
                "away_goals": 3,
            },
        ]
    )


def test_league_phase_matches_filters_season_stage_and_cutoff() -> None:
    matches = league_phase_matches(make_matches(), season="2025_26", cutoff_date="2025-09-16")

    assert len(matches) == 1
    assert matches.iloc[0]["home_team"] == "Arsenal"


def test_build_league_phase_table_aggregates_home_and_away_results() -> None:
    table = build_league_phase_table(make_matches(), season="2025_26")

    arsenal = table.loc[table["team"] == "Arsenal"].iloc[0]
    psv = table.loc[table["team"] == "PSV"].iloc[0]

    assert arsenal["played"] == 1
    assert arsenal["won"] == 1
    assert arsenal["goals_for"] == 2
    assert arsenal["goal_difference"] == 2
    assert arsenal["points"] == 3

    assert psv["played"] == 2
    assert psv["drawn"] == 1
    assert psv["lost"] == 1
    assert psv["points"] == 1


def test_build_league_phase_table_sorts_by_points_then_goal_difference() -> None:
    table = build_league_phase_table(make_matches(), season="2025_26")

    assert list(table["team"]) == ["Arsenal", "Dortmund", "PSV"]
    assert list(table["rank"]) == [1, 2, 3]


def test_load_ucl_matches_reads_matching_season_and_validates_columns(tmp_path) -> None:
    raw_dir = tmp_path / "ucl"
    raw_dir.mkdir()
    make_matches().to_csv(raw_dir / "2025_26.csv", index=False)
    pd.DataFrame({"date": ["2024-09-16"]}).to_csv(raw_dir / "2024_25.csv", index=False)

    matches = load_ucl_matches(raw_dir=raw_dir, season="2025_26")

    assert matches["season"].unique().tolist() == ["2025_26"]
    assert pd.api.types.is_datetime64_any_dtype(matches["date"])


def test_load_ucl_matches_rejects_missing_columns(tmp_path) -> None:
    raw_dir = tmp_path / "ucl"
    raw_dir.mkdir()
    pd.DataFrame({"date": ["2025-09-16"]}).to_csv(raw_dir / "2025_26.csv", index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_ucl_matches(raw_dir=raw_dir, season="2025_26")
