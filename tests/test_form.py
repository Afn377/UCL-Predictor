import pandas as pd

from src.features.form import add_rolling_form_features, build_team_match_history, points_from_result


def make_matches(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "season": "2020_21",
        "competition": "Premier League",
        "stage": None,
        "home_team_raw": None,
        "away_team_raw": None,
        "home_elo_before": 1500.0,
        "away_elo_before": 1500.0,
        "elo_diff": 0.0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_points_from_result_for_home_and_away_perspectives() -> None:
    assert points_from_result(2, is_home=True) == 3
    assert points_from_result(2, is_home=False) == 0
    assert points_from_result(1, is_home=True) == 1
    assert points_from_result(1, is_home=False) == 1
    assert points_from_result(0, is_home=True) == 0
    assert points_from_result(0, is_home=False) == 3


def test_build_team_match_history_creates_two_rows_per_match() -> None:
    matches = make_matches(
        [
            {
                "date": "2020-01-01",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_goals": 2,
                "away_goals": 1,
                "result": 2,
            }
        ]
    )

    history = build_team_match_history(matches)

    assert len(history) == 2
    assert set(history["team"]) == {"Arsenal", "Chelsea"}
    arsenal = history[history["team"] == "Arsenal"].iloc[0]
    chelsea = history[history["team"] == "Chelsea"].iloc[0]
    assert arsenal["goals_for"] == 2
    assert arsenal["goals_against"] == 1
    assert arsenal["points"] == 3
    assert chelsea["goals_for"] == 1
    assert chelsea["goals_against"] == 2
    assert chelsea["points"] == 0


def test_rolling_form_uses_only_previous_matches() -> None:
    matches = make_matches(
        [
            {
                "date": "2020-01-01",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_goals": 2,
                "away_goals": 0,
                "result": 2,
            },
            {
                "date": "2020-01-08",
                "home_team": "Arsenal",
                "away_team": "Tottenham",
                "home_goals": 1,
                "away_goals": 1,
                "result": 1,
            },
            {
                "date": "2020-01-15",
                "home_team": "Arsenal",
                "away_team": "Liverpool",
                "home_goals": 0,
                "away_goals": 3,
                "result": 0,
            },
        ]
    )

    featured = add_rolling_form_features(matches, windows=(5,))

    assert pd.isna(featured.loc[0, "home_ppg_5_before"])
    assert featured.loc[1, "home_ppg_5_before"] == 3
    assert featured.loc[1, "home_goals_for_5_before"] == 2
    assert featured.loc[1, "home_goals_against_5_before"] == 0
    assert featured.loc[1, "home_goal_difference_5_before"] == 2
    assert featured.loc[2, "home_ppg_5_before"] == 2
    assert featured.loc[2, "home_goal_difference_5_before"] == 1


def test_rolling_window_limits_history_length() -> None:
    matches = make_matches(
        [
            {
                "date": "2020-01-01",
                "home_team": "Arsenal",
                "away_team": "Team A",
                "home_goals": 1,
                "away_goals": 0,
                "result": 2,
            },
            {
                "date": "2020-01-02",
                "home_team": "Arsenal",
                "away_team": "Team B",
                "home_goals": 1,
                "away_goals": 1,
                "result": 1,
            },
            {
                "date": "2020-01-03",
                "home_team": "Arsenal",
                "away_team": "Team C",
                "home_goals": 0,
                "away_goals": 2,
                "result": 0,
            },
        ]
    )

    featured = add_rolling_form_features(matches, windows=(2,))

    assert featured.loc[2, "home_ppg_2_before"] == 2
    assert featured.loc[2, "home_goal_difference_2_before"] == 0.5


def test_diff_features_compare_home_and_away_prior_form() -> None:
    matches = make_matches(
        [
            {
                "date": "2020-01-01",
                "home_team": "Arsenal",
                "away_team": "Team A",
                "home_goals": 2,
                "away_goals": 0,
                "result": 2,
            },
            {
                "date": "2020-01-01",
                "home_team": "Chelsea",
                "away_team": "Team B",
                "home_goals": 0,
                "away_goals": 1,
                "result": 0,
            },
            {
                "date": "2020-01-08",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_goals": 1,
                "away_goals": 1,
                "result": 1,
            },
        ]
    )

    featured = add_rolling_form_features(matches, windows=(5,))
    target = featured[(featured["home_team"] == "Arsenal") & (featured["away_team"] == "Chelsea")].iloc[0]

    assert target["home_ppg_5_before"] == 3
    assert target["away_ppg_5_before"] == 0
    assert target["ppg_5_diff"] == 3
    assert target["goal_difference_5_diff"] == 3
