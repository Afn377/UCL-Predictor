import numpy as np
import pandas as pd

from src.evaluation.calibration import (
    PROBABILITY_COLUMNS,
    build_calibration_curve,
    predictions_for_model,
    probability_frame,
)


def make_dataset() -> pd.DataFrame:
    rows = []
    for i in range(45):
        rows.append(
            {
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                "season": "2020_21",
                "competition": "Premier League",
                "home_team": f"Home {i}",
                "away_team": f"Away {i}",
                "stage": None,
                "elo_diff": i - 20,
                "ppg_5_diff": (i % 5) - 2,
                "goal_difference_5_diff": (i % 7) - 3,
                "home_goals": i % 4,
                "away_goals": (i + 1) % 3,
                "result": i % 3,
            }
        )
    return pd.DataFrame(rows)


def test_probability_frame_uses_class_probability_columns() -> None:
    frame = probability_frame(np.array([[0.2, 0.3, 0.5]]))

    assert list(frame.columns) == PROBABILITY_COLUMNS


def test_predictions_for_naive_model_outputs_metadata_and_probabilities() -> None:
    data = make_dataset()
    train = data.iloc[:30]
    test = data.iloc[30:]

    predictions = predictions_for_model(train, test, "naive_base_rate")

    assert len(predictions) == len(test)
    assert set(PROBABILITY_COLUMNS).issubset(predictions.columns)
    assert set(predictions["model"]) == {"naive_base_rate"}
    assert np.allclose(predictions[PROBABILITY_COLUMNS].sum(axis=1), 1)


def test_predictions_for_poisson_model_outputs_result_probabilities() -> None:
    data = make_dataset()
    train = data.iloc[:30]
    test = data.iloc[30:]

    predictions = predictions_for_model(
        train,
        test,
        "poisson_score_model",
        ["elo_diff", "ppg_5_diff", "goal_difference_5_diff"],
    )

    assert len(predictions) == len(test)
    assert set(predictions["model"]) == {"poisson_score_model"}
    assert np.allclose(predictions[PROBABILITY_COLUMNS].sum(axis=1), 1)


def test_predictions_for_xgboost_model_outputs_result_probabilities() -> None:
    data = make_dataset()
    train = data.iloc[:30]
    test = data.iloc[30:]

    predictions = predictions_for_model(
        train,
        test,
        "xgboost_classifier",
        ["elo_diff", "ppg_5_diff", "goal_difference_5_diff"],
    )

    assert len(predictions) == len(test)
    assert set(predictions["model"]) == {"xgboost_classifier"}
    assert np.allclose(predictions[PROBABILITY_COLUMNS].sum(axis=1), 1)


def test_build_calibration_curve_returns_observed_frequency_by_bin() -> None:
    predictions = pd.DataFrame(
        {
            "model": ["m", "m", "m"],
            "result": [0, 1, 2],
            "prob_0": [0.8, 0.7, 0.1],
            "prob_1": [0.1, 0.2, 0.8],
            "prob_2": [0.1, 0.1, 0.1],
        }
    )

    curve = build_calibration_curve(predictions, bins=2)

    assert {"model", "class_label", "mean_predicted_probability", "observed_frequency", "count"}.issubset(
        curve.columns
    )
    assert curve["count"].sum() == 9
