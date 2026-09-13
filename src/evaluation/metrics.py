import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.data.market import MATCH_KEYS

SEASON_SPLITS = [
    ("2019_20", "2019-07-01", "2020-07-01"),
    ("2020_21", "2020-07-01", "2021-07-01"),
    ("2021_22", "2021-07-01", "2022-07-01"),
    ("2022_23", "2022-07-01", "2023-07-01"),
    ("2023_24", "2023-07-01", "2024-07-01"),
    ("2024_25", "2024-07-01", "2025-07-01"),
    ("2025_26", "2025-07-01", "2026-07-01"),
]


def ranked_probability_score(y_true, probabilities):
    """Measure probability error across away win, draw, then home win."""
    outcomes = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        probabilities.shape != (len(outcomes), 3)
        or not len(outcomes)
        or not np.isin(outcomes, [0, 1, 2]).all()
    ):
        raise ValueError(
            "RPS requires three ordered probabilities and labels in {0, 1, 2}"
        )
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise ValueError("Invalid probability distribution")
    actual = np.eye(3)[outcomes.astype(int)]
    # third cumulative probability is always 1, so drop it
    cumulative_forecast = np.cumsum(probabilities, axis=1)[:, :2]
    cumulative_outcome = np.cumsum(actual, axis=1)[:, :2]
    return float(((cumulative_forecast - cumulative_outcome) ** 2).mean())


def evaluate_probabilities_with_brier(y_true, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    rps = ranked_probability_score(y_true, probabilities)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    actual = np.eye(3)[np.asarray(y_true, dtype=int)]
    return {
        "accuracy": float(accuracy_score(y_true, probabilities.argmax(axis=1))),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "brier_score": float(((actual - probabilities) ** 2).mean()),
        "rps": rps,
    }


def prediction_frame(test, probabilities, model, split, train_end):
    frame = test[MATCH_KEYS + ["season", "stage", "result"]].reset_index(drop=True)
    probabilities = np.asarray(probabilities, dtype=float)
    # reject bad predictions before writing an evaluation file
    if (
        probabilities.shape != (len(test), 3)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise ValueError(f"Invalid predictions for {model}")
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    return pd.concat(
        [frame, pd.DataFrame(probabilities, columns=[f"prob_{i}" for i in range(3)])],
        axis=1,
    ).assign(
        model=model,
        split=split,
        train_end=str(pd.Timestamp(train_end).date()),
    )
