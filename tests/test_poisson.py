import numpy as np
import pandas as pd

from src.models.poisson import (
    evaluate_poisson_predictions,
    poisson_probability,
    predict_goal_lambdas,
    prepare_score_model_dataset,
    scoreline_probabilities,
    scoreline_result_probabilities,
    temporal_train_test_split,
    train_poisson_models,
)


def make_score_data() -> pd.DataFrame:
    rows = []
    for i in range(60):
        rows.append(
            {
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                "season": "2020_21",
                "competition": "Premier League",
                "stage": None,
                "home_team": f"Home {i}",
                "away_team": f"Away {i}",
                "elo_diff": i - 30,
                "ppg_5_diff": (i % 5) - 2,
                "goal_difference_5_diff": (i % 7) - 3,
                "home_goals": i % 4,
                "away_goals": (i + 1) % 3,
            }
        )
    return pd.DataFrame(rows)


def test_prepare_score_model_dataset_drops_rows_missing_features() -> None:
    data = make_score_data()
    data.loc[0, "ppg_5_diff"] = np.nan

    prepared = prepare_score_model_dataset(data)

    assert len(prepared) == len(data) - 1
    assert prepared["date"].is_monotonic_increasing


def test_train_poisson_models_predict_positive_lambdas() -> None:
    data = prepare_score_model_dataset(make_score_data())
    home_model, away_model = train_poisson_models(data)
    predictions = predict_goal_lambdas(home_model, away_model, data.head())

    assert (predictions["lambda_home"] > 0).all()
    assert (predictions["lambda_away"] > 0).all()


def test_poisson_probability_for_zero_goals() -> None:
    assert np.isclose(poisson_probability(0, 1.5), np.exp(-1.5))


def test_scoreline_probabilities_have_expected_grid() -> None:
    scorelines = scoreline_probabilities(1.2, 0.8, max_goals=2)

    assert len(scorelines) == 9
    assert {"home_goals", "away_goals", "probability"}.issubset(scorelines.columns)


def test_scoreline_result_probabilities_sum_to_one_after_truncation() -> None:
    probabilities = scoreline_result_probabilities(1.5, 1.0, max_goals=8)

    assert set(probabilities) == {
        "away_win_probability",
        "draw_probability",
        "home_win_probability",
    }
    assert np.isclose(sum(probabilities.values()), 1)


def test_evaluate_poisson_predictions_returns_deviance_metrics() -> None:
    data = prepare_score_model_dataset(make_score_data())
    train, test = temporal_train_test_split(data, split_date="2020-02-15")
    home_model, away_model = train_poisson_models(train)
    predictions = predict_goal_lambdas(home_model, away_model, test)
    metrics = evaluate_poisson_predictions(test, predictions)

    assert set(metrics) == {"home_goal_deviance", "away_goal_deviance", "mean_goal_deviance"}
    assert metrics["mean_goal_deviance"] >= 0
