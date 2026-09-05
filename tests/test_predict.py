import numpy as np
import pandas as pd

from src.models.predict import (
    build_prediction_features,
    final_elo_ratings,
    latest_form_by_team,
    predict_match_probabilities,
)


def make_history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "season": "2020_21",
                "competition": "Premier League",
                "stage": None,
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_goals": 2,
                "away_goals": 0,
                "result": 2,
            },
            {
                "date": "2020-01-08",
                "season": "2020_21",
                "competition": "Premier League",
                "stage": None,
                "home_team": "Chelsea",
                "away_team": "Arsenal",
                "home_goals": 1,
                "away_goals": 1,
                "result": 1,
            },
            {
                "date": "2020-01-15",
                "season": "2020_21",
                "competition": "Premier League",
                "stage": None,
                "home_team": "Arsenal",
                "away_team": "Tottenham",
                "home_goals": 3,
                "away_goals": 1,
                "result": 2,
            },
        ]
    )


def make_model_dataset() -> pd.DataFrame:
    rows = []
    teams = ["Arsenal", "Chelsea", "Tottenham", "Liverpool"]
    for i in range(60):
        rows.append(
            {
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                "season": "2020_21",
                "competition": "Premier League",
                "stage": None,
                "home_team": teams[i % len(teams)],
                "away_team": teams[(i + 1) % len(teams)],
                "elo_diff": i - 30,
                "ppg_5_diff": (i % 5) - 2,
                "goal_difference_5_diff": (i % 7) - 3,
                "result": i % 3,
            }
        )
    return pd.DataFrame(rows)


def test_final_elo_ratings_updates_after_matches() -> None:
    ratings = final_elo_ratings(make_history())

    assert ratings["Arsenal"] > ratings["Chelsea"]


def test_latest_form_by_team_uses_recent_completed_matches() -> None:
    form = latest_form_by_team(make_history())

    assert form.loc["Arsenal", "ppg_5"] == 7 / 3
    assert form.loc["Arsenal", "goal_difference_5"] == 4 / 3


def test_build_prediction_features_returns_model_columns() -> None:
    features = build_prediction_features("Arsenal", "Chelsea", make_history())

    assert list(features.columns) == ["elo_diff", "ppg_5_diff", "goal_difference_5_diff"]
    assert features.loc[0, "elo_diff"] > 0


def test_predict_match_probabilities_sum_to_one() -> None:
    probabilities = predict_match_probabilities(
        "Arsenal",
        "Chelsea",
        make_model_dataset(),
        make_history(),
    )

    assert set(probabilities) == {
        "away_win_probability",
        "draw_probability",
        "home_win_probability",
    }
    assert np.isclose(sum(probabilities.values()), 1)
