from __future__ import annotations

import argparse
import os
from math import exp, factorial
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

from src.models.baselines import FULL_FEATURES


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "src" / "data" / "processed" / "matches_with_features.csv"
DEFAULT_OUTPUT = ROOT / "src" / "data" / "processed" / "poisson_predictions.csv"

TARGET_COLUMNS = ["home_goals", "away_goals"]
REQUIRED_COLUMNS = ["date", "home_team", "away_team", *FULL_FEATURES, *TARGET_COLUMNS]


def prepare_score_model_dataset(matches: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in matches.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"matches is missing required columns: {missing}")

    data = matches.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values(
        ["date", "competition", "home_team", "away_team"],
        kind="mergesort",
    ).reset_index(drop=True)
    data = data.dropna(subset=FULL_FEATURES)
    return data.reset_index(drop=True)


def train_poisson_models(
    train: pd.DataFrame,
    features: list[str] | None = None,
) -> tuple[PoissonRegressor, PoissonRegressor]:
    feature_columns = features or FULL_FEATURES
    home_model = PoissonRegressor(alpha=1e-4, max_iter=1000)
    away_model = PoissonRegressor(alpha=1e-4, max_iter=1000)

    home_model.fit(train[feature_columns], train["home_goals"])
    away_model.fit(train[feature_columns], train["away_goals"])
    return home_model, away_model


def predict_goal_lambdas(
    home_model: PoissonRegressor,
    away_model: PoissonRegressor,
    matches: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    feature_columns = features or FULL_FEATURES
    predictions = matches[["date", "season", "competition", "stage", "home_team", "away_team"]].copy()
    predictions["lambda_home"] = home_model.predict(matches[feature_columns])
    predictions["lambda_away"] = away_model.predict(matches[feature_columns])
    return predictions


def poisson_probability(goal_count: int, goal_lambda: float) -> float:
    return exp(-goal_lambda) * goal_lambda**goal_count / factorial(goal_count)


def scoreline_probabilities(lambda_home: float, lambda_away: float, max_goals: int = 6) -> pd.DataFrame:
    rows = []
    for home_goals in range(max_goals + 1):
        home_probability = poisson_probability(home_goals, lambda_home)
        for away_goals in range(max_goals + 1):
            rows.append(
                {
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "probability": home_probability * poisson_probability(away_goals, lambda_away),
                }
            )

    return pd.DataFrame(rows)


def scoreline_result_probabilities(lambda_home: float, lambda_away: float, max_goals: int = 10) -> dict[str, float]:
    scorelines = scoreline_probabilities(lambda_home, lambda_away, max_goals=max_goals)
    home_goals = scorelines["home_goals"]
    away_goals = scorelines["away_goals"]

    away_win = scorelines.loc[home_goals < away_goals, "probability"].sum()
    draw = scorelines.loc[home_goals == away_goals, "probability"].sum()
    home_win = scorelines.loc[home_goals > away_goals, "probability"].sum()
    total = away_win + draw + home_win

    return {
        "away_win_probability": float(away_win / total),
        "draw_probability": float(draw / total),
        "home_win_probability": float(home_win / total),
    }


def result_probability_array(predictions: pd.DataFrame, max_goals: int = 10) -> pd.DataFrame:
    rows = []
    for row in predictions.itertuples(index=False):
        probabilities = scoreline_result_probabilities(
            row.lambda_home,
            row.lambda_away,
            max_goals=max_goals,
        )
        rows.append(
            {
                0: probabilities["away_win_probability"],
                1: probabilities["draw_probability"],
                2: probabilities["home_win_probability"],
            }
        )

    return pd.DataFrame(rows)


def evaluate_poisson_predictions(test: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    return {
        "home_goal_deviance": mean_poisson_deviance(test["home_goals"], predictions["lambda_home"]),
        "away_goal_deviance": mean_poisson_deviance(test["away_goals"], predictions["lambda_away"]),
        "mean_goal_deviance": (
            mean_poisson_deviance(test["home_goals"], predictions["lambda_home"])
            + mean_poisson_deviance(test["away_goals"], predictions["lambda_away"])
        )
        / 2,
    }


def temporal_train_test_split(matches: pd.DataFrame, split_date: str = "2024-07-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    train = matches[matches["date"] < split_date].copy()
    test = matches[matches["date"] >= split_date].copy()
    return train, test


def run_poisson_model(input_path: Path = DEFAULT_INPUT) -> tuple[pd.DataFrame, dict[str, float]]:
    matches = pd.read_csv(input_path)
    data = prepare_score_model_dataset(matches)
    train, test = temporal_train_test_split(data)
    home_model, away_model = train_poisson_models(train)
    predictions = predict_goal_lambdas(home_model, away_model, test)
    metrics = evaluate_poisson_predictions(test, predictions)
    return predictions, metrics


def write_poisson_predictions(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
) -> tuple[pd.DataFrame, dict[str, float]]:
    predictions, metrics = run_poisson_model(input_path=input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return predictions, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple Poisson score model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    predictions, metrics = write_poisson_predictions(input_path=args.input, output_path=args.output)
    print(f"Wrote {len(predictions)} Poisson score predictions to {args.output}.")
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
