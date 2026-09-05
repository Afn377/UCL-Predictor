from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.elo import DEFAULT_INITIAL_RATING, DEFAULT_K_FACTOR, expected_score, score_from_result
from src.features.form import build_team_match_history
from src.models.baselines import CLASSES, FULL_FEATURES, train_logistic_model


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DATASET = ROOT / "src" / "data" / "processed" / "model_dataset.csv"
DEFAULT_MATCH_HISTORY = ROOT / "src" / "data" / "processed" / "matches_with_elo.csv"

PROBABILITY_LABELS = {
    0: "away_win_probability",
    1: "draw_probability",
    2: "home_win_probability",
}


def final_elo_ratings(
    matches: pd.DataFrame,
    initial_rating: float = DEFAULT_INITIAL_RATING,
    k_factor: float = DEFAULT_K_FACTOR,
) -> dict[str, float]:
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="raise")
    matches = matches.sort_values(
        ["date", "competition", "home_team", "away_team"],
        kind="mergesort",
    ).reset_index(drop=True)

    ratings: dict[str, float] = {}
    for row in matches.itertuples(index=False):
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        home_rating = ratings.get(home_team, initial_rating)
        away_rating = ratings.get(away_team, initial_rating)
        expected_home = expected_score(home_rating, away_rating)
        expected_away = 1 - expected_home
        actual_home, actual_away = score_from_result(int(row.result))

        ratings[home_team] = home_rating + k_factor * (actual_home - expected_home)
        ratings[away_team] = away_rating + k_factor * (actual_away - expected_away)

    return ratings


def latest_form_by_team(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    history = build_team_match_history(matches)
    form = (
        history.groupby("team", as_index=False)
        .tail(window)
        .groupby("team", as_index=False)
        .agg(
            ppg_5=("points", "mean"),
            goal_difference_5=("goal_difference", "mean"),
        )
    )
    return form.set_index("team")


def build_prediction_features(
    home_team: str,
    away_team: str,
    match_history: pd.DataFrame,
) -> pd.DataFrame:
    ratings = final_elo_ratings(match_history)
    form = latest_form_by_team(match_history)

    missing_teams = [team for team in [home_team, away_team] if team not in ratings or team not in form.index]
    if missing_teams:
        raise ValueError(f"No match history found for team(s): {', '.join(missing_teams)}")

    return pd.DataFrame(
        [
            {
                "elo_diff": ratings[home_team] - ratings[away_team],
                "ppg_5_diff": form.loc[home_team, "ppg_5"] - form.loc[away_team, "ppg_5"],
                "goal_difference_5_diff": (
                    form.loc[home_team, "goal_difference_5"] - form.loc[away_team, "goal_difference_5"]
                ),
            }
        ]
    )


def train_current_match_model(model_dataset: pd.DataFrame):
    return train_logistic_model(model_dataset, FULL_FEATURES)


def predict_match_probabilities(
    home_team: str,
    away_team: str,
    model_dataset: pd.DataFrame,
    match_history: pd.DataFrame,
) -> dict[str, float]:
    model = train_current_match_model(model_dataset)
    features = build_prediction_features(home_team, away_team, match_history)
    probabilities = model.predict_proba(features[FULL_FEATURES])[0]

    return {PROBABILITY_LABELS[label]: float(probabilities[index]) for index, label in enumerate(CLASSES)}


def predict_match(
    home_team: str,
    away_team: str,
    model_dataset_path: Path = DEFAULT_MODEL_DATASET,
    match_history_path: Path = DEFAULT_MATCH_HISTORY,
) -> dict[str, float]:
    model_dataset = pd.read_csv(model_dataset_path, parse_dates=["date"])
    match_history = pd.read_csv(match_history_path, parse_dates=["date"])
    return predict_match_probabilities(home_team, away_team, model_dataset, match_history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a match result probability from current features.")
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    args = parser.parse_args()

    probabilities = predict_match(args.home_team, args.away_team)
    print(f"{args.home_team} win: {probabilities['home_win_probability']:.1%}")
    print(f"Draw: {probabilities['draw_probability']:.1%}")
    print(f"{args.away_team} win: {probabilities['away_win_probability']:.1%}")


if __name__ == "__main__":
    main()
