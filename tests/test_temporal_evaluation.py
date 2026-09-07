import numpy as np
import pandas as pd
from src.evaluation.temporal_evaluation import (
    brier_score,
    evaluate_models_for_split,
    evaluate_probabilities_with_brier,
    make_temporal_split,
)

def make_dataset():
    rows = []

    for i in range(90):
        rows.append(
            {
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                "season": "2020_21",
                "competition": "Premier League",
                "stage": None,
                "home_team": f"Home {i}",
                "away_team": f"Away {i}",
                "elo_diff": i - 45,
                "ppg_5_diff": (i % 5) - 2,
                "goal_difference_5_diff": (i % 7) - 3,
                "home_goals": i % 4,
                "away_goals": (i + 1) % 3,
                "result": i % 3,
            }
        )

    for i in range(30):
        rows.append(
            {
                "date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=i),
                "season": "2021_22",
                "competition": "Premier League",
                "stage": None,
                "home_team": f"Test Home {i}",
                "away_team": f"Test Away {i}",
                "elo_diff": i - 15,
                "ppg_5_diff": (i % 5) - 2,
                "goal_difference_5_diff": (i % 7) - 3,
                "home_goals": i % 4,
                "away_goals": (i + 1) % 3,
                "result": i % 3,
            }
        )

    return pd.DataFrame(rows)


def test_make_temporal_split_keeps_test_after_train():
    df = make_dataset()
    train, test = make_temporal_split(
        df,
        train_start="2020-01-01",
        test_start="2021-01-01",
        test_end="2022-01-01",
    )

    assert train["date"].max() < test["date"].min()
    assert test["date"].min() >= pd.Timestamp("2021-01-01")
    assert test["date"].max() <= pd.Timestamp("2022-01-01")


def test_brier_score_is_zero_for_perfect_predictions():
    y_true = pd.Series([0,1,2])
    probabilities = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    assert brier_score(y_true, probabilities) == 0


def test_evaluate_probabilities_with_brier_returns_three_metrics():
    y_true = pd.Series([0,1,2])
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.1, 0.7],
        ]
    )

    result = evaluate_probabilities_with_brier(y_true, probabilities)
    assert set(result) == {"log_loss", "accuracy", "brier_score"}


def test_evaluate_models_for_split_returns_four_models():
    df = make_dataset()
    train, test = make_temporal_split(
        df,
        train_start="2020-01-01",
        test_start="2021-01-01",
        test_end="2022-01-01",
    )

    rows = evaluate_models_for_split(train, test, split_name="2021_22")

    assert len(rows) == 4
    assert {row["model"] for row in rows} == {
        "naive_base_rate",
        "elo_logistic",
        "feature_logistic",
        "poisson_score_model",
    }
    assert all(row["split"] == "2021_22" for row in rows)
