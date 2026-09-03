import numpy as np
import pandas as pd

from src.simulation.league_phase_simulator import (
    qualification_summary,
    simulate_league_phase,
    split_league_phase_schedule,
)


def make_schedule() -> pd.DataFrame:
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
                "date": pd.Timestamp("2025-10-01"),
                "season": "2025_26",
                "competition": "Champions League",
                "stage": "league_phase",
                "home_team": "Dortmund",
                "away_team": "Arsenal",
                "home_goals": 1,
                "away_goals": 1,
            },
        ]
    )


def test_split_league_phase_schedule_separates_completed_and_remaining() -> None:
    completed, remaining = split_league_phase_schedule(make_schedule(), season="2025_26", cutoff_date="2025-09-16")

    assert completed["home_team"].tolist() == ["Arsenal"]
    assert remaining["home_team"].tolist() == ["Dortmund"]
    assert np.isnan(remaining.iloc[0]["home_goals"])


def test_qualification_summary_averages_rank_and_outcome_flags() -> None:
    tables = [
        pd.DataFrame(
            {
                "rank": [1, 9, 25],
                "team": ["Arsenal", "PSV", "Dortmund"],
                "points": [18, 11, 4],
            }
        ),
        pd.DataFrame(
            {
                "rank": [2, 7, 26],
                "team": ["Arsenal", "PSV", "Dortmund"],
                "points": [17, 13, 3],
            }
        ),
    ]

    summary = qualification_summary(tables)
    arsenal = summary.loc[summary["team"] == "Arsenal"].iloc[0]
    psv = summary.loc[summary["team"] == "PSV"].iloc[0]
    dortmund = summary.loc[summary["team"] == "Dortmund"].iloc[0]

    assert arsenal["average_rank"] == 1.5
    assert arsenal["top_8_probability"] == 1.0
    assert psv["top_8_probability"] == 0.5
    assert dortmund["elimination_probability"] == 1.0


def test_simulate_league_phase_returns_probabilities_for_all_teams() -> None:
    completed, _ = split_league_phase_schedule(make_schedule(), season="2025_26", cutoff_date="2025-09-16")
    goal_lambdas = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-10-01"),
                "season": "2025_26",
                "competition": "Champions League",
                "stage": "league_phase",
                "home_team": "Dortmund",
                "away_team": "Arsenal",
                "lambda_home": 0.2,
                "lambda_away": 2.2,
            }
        ]
    )

    summary = simulate_league_phase(completed, goal_lambdas, season="2025_26", n_simulations=20, random_seed=1)

    assert set(summary["team"]) == {"Arsenal", "PSV", "Dortmund"}
    assert summary["top_24_probability"].between(0, 1).all()
