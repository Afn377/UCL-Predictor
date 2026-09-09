from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.elo import DEFAULT_INITIAL_RATING, DEFAULT_K_FACTOR, expected_score, score_from_result
from src.features.form import STRENGTH_WINDOW, build_team_match_history, is_new_ucl_format
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


def latest_form_by_team(
    matches: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10),
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if as_of_date is None:
        as_of = pd.to_datetime(matches["date"]).max()
    else:
        as_of = pd.Timestamp(as_of_date)

    history = build_team_match_history(matches)
    history["date"] = pd.to_datetime(history["date"], errors="raise")
    history = history.loc[history["date"] <= as_of].copy()

    rows = []
    for team, team_history in history.groupby("team", sort=False):
        team_history = team_history.sort_values(["date", "match_id"], kind="mergesort")
        row: dict[str, float | str] = {"team": team}
        for window in windows:
            recent = team_history.tail(window)
            home_recent = team_history[team_history["is_home"]].tail(window)
            away_recent = team_history[~team_history["is_home"]].tail(window)
            row[f"ppg_{window}"] = recent["points"].mean()
            row[f"goals_for_{window}"] = recent["goals_for"].mean()
            row[f"goals_against_{window}"] = recent["goals_against"].mean()
            row[f"goal_difference_{window}"] = recent["goal_difference"].mean()
            row[f"home_ppg_{window}"] = home_recent["points"].mean()
            row[f"home_goals_for_{window}"] = home_recent["goals_for"].mean()
            row[f"home_goals_against_{window}"] = home_recent["goals_against"].mean()
            row[f"home_goal_difference_{window}"] = home_recent["goal_difference"].mean()
            row[f"away_ppg_{window}"] = away_recent["points"].mean()
            row[f"away_goals_for_{window}"] = away_recent["goals_for"].mean()
            row[f"away_goals_against_{window}"] = away_recent["goals_against"].mean()
            row[f"away_goal_difference_{window}"] = away_recent["goal_difference"].mean()

        strength_recent = team_history.tail(STRENGTH_WINDOW)
        top_opponent_recent = strength_recent[strength_recent["is_top_opponent"]]
        row[f"opponent_adjusted_goal_difference_{STRENGTH_WINDOW}"] = strength_recent[
            "opponent_adjusted_goal_difference"
        ].mean()
        row[f"goals_against_top_opponents_{STRENGTH_WINDOW}"] = top_opponent_recent["goals_against"].mean()
        row[f"ppg_top_opponents_{STRENGTH_WINDOW}"] = top_opponent_recent["points"].mean()
        row["ucl_matches"] = int((team_history["competition"] == "Champions League").sum())

        last_match_date = team_history["date"].max()
        row["days_since_match"] = (as_of - last_match_date).days
        for congestion_window in (7, 14, 30):
            window_start = as_of - pd.Timedelta(days=congestion_window)
            row[f"matches_last_{congestion_window}"] = int(
                ((team_history["date"] >= window_start) & (team_history["date"] < as_of)).sum()
            )
        rows.append(row)

    form = pd.DataFrame(rows).reset_index(drop=True)
    for window in windows:
        for venue in ["home", "away"]:
            form[f"{venue}_ppg_{window}"] = form[f"{venue}_ppg_{window}"].fillna(form[f"ppg_{window}"])
            form[f"{venue}_goals_for_{window}"] = form[f"{venue}_goals_for_{window}"].fillna(
                form[f"goals_for_{window}"]
            )
            form[f"{venue}_goals_against_{window}"] = form[f"{venue}_goals_against_{window}"].fillna(
                form[f"goals_against_{window}"]
            )
            form[f"{venue}_goal_difference_{window}"] = form[f"{venue}_goal_difference_{window}"].fillna(
                form[f"goal_difference_{window}"]
            )
    return form.set_index("team")


def build_prediction_features(
    home_team: str,
    away_team: str,
    match_history: pd.DataFrame,
    match_date: str | pd.Timestamp | None = None,
    competition: str = "Champions League",
) -> pd.DataFrame:
    rating_history = match_history.copy()
    if match_date is not None:
        rating_history["date"] = pd.to_datetime(rating_history["date"], errors="raise")
        rating_history = rating_history.loc[rating_history["date"] <= pd.Timestamp(match_date)].copy()

    ratings = final_elo_ratings(rating_history)
    form = latest_form_by_team(match_history, as_of_date=match_date)

    missing_teams = [team for team in [home_team, away_team] if team not in ratings or team not in form.index]
    if missing_teams:
        raise ValueError(f"No match history found for team(s): {', '.join(missing_teams)}")

    format_year = ""
    if match_date is not None:
        format_year = str(pd.Timestamp(match_date).year)

    home_venue_prefix = "home"
    away_venue_prefix = "away"
    return pd.DataFrame(
        [
            {
                "elo_diff": ratings[home_team] - ratings[away_team],
                "ppg_5_diff": form.loc[home_team, "ppg_5"] - form.loc[away_team, "ppg_5"],
                "goals_for_5_diff": form.loc[home_team, "goals_for_5"] - form.loc[away_team, "goals_for_5"],
                "goals_against_5_diff": (
                    form.loc[home_team, "goals_against_5"] - form.loc[away_team, "goals_against_5"]
                ),
                "goal_difference_5_diff": (
                    form.loc[home_team, "goal_difference_5"] - form.loc[away_team, "goal_difference_5"]
                ),
                "venue_ppg_5_diff": (
                    form.loc[home_team, f"{home_venue_prefix}_ppg_5"]
                    - form.loc[away_team, f"{away_venue_prefix}_ppg_5"]
                ),
                "venue_goals_for_5_diff": (
                    form.loc[home_team, f"{home_venue_prefix}_goals_for_5"]
                    - form.loc[away_team, f"{away_venue_prefix}_goals_for_5"]
                ),
                "venue_goals_against_5_diff": (
                    form.loc[home_team, f"{home_venue_prefix}_goals_against_5"]
                    - form.loc[away_team, f"{away_venue_prefix}_goals_against_5"]
                ),
                "venue_goal_difference_5_diff": (
                    form.loc[home_team, f"{home_venue_prefix}_goal_difference_5"]
                    - form.loc[away_team, f"{away_venue_prefix}_goal_difference_5"]
                ),
                "ppg_10_diff": form.loc[home_team, "ppg_10"] - form.loc[away_team, "ppg_10"],
                "goals_for_10_diff": form.loc[home_team, "goals_for_10"] - form.loc[away_team, "goals_for_10"],
                "goals_against_10_diff": (
                    form.loc[home_team, "goals_against_10"] - form.loc[away_team, "goals_against_10"]
                ),
                "goal_difference_10_diff": (
                    form.loc[home_team, "goal_difference_10"] - form.loc[away_team, "goal_difference_10"]
                ),
                "venue_ppg_10_diff": (
                    form.loc[home_team, f"{home_venue_prefix}_ppg_10"]
                    - form.loc[away_team, f"{away_venue_prefix}_ppg_10"]
                ),
                "venue_goals_for_10_diff": (
                    form.loc[home_team, f"{home_venue_prefix}_goals_for_10"]
                    - form.loc[away_team, f"{away_venue_prefix}_goals_for_10"]
                ),
                "venue_goals_against_10_diff": (
                    form.loc[home_team, f"{home_venue_prefix}_goals_against_10"]
                    - form.loc[away_team, f"{away_venue_prefix}_goals_against_10"]
                ),
                "venue_goal_difference_10_diff": (
                    form.loc[home_team, f"{home_venue_prefix}_goal_difference_10"]
                    - form.loc[away_team, f"{away_venue_prefix}_goal_difference_10"]
                ),
                "rest_days_diff": form.loc[home_team, "days_since_match"] - form.loc[away_team, "days_since_match"],
                "matches_last_7_diff": (
                    form.loc[home_team, "matches_last_7"] - form.loc[away_team, "matches_last_7"]
                ),
                "matches_last_14_diff": (
                    form.loc[home_team, "matches_last_14"] - form.loc[away_team, "matches_last_14"]
                ),
                "matches_last_30_diff": (
                    form.loc[home_team, "matches_last_30"] - form.loc[away_team, "matches_last_30"]
                ),
                "is_champions_league": int(competition == "Champions League"),
                "is_new_ucl_format": is_new_ucl_format(format_year),
                f"opponent_adjusted_goal_difference_{STRENGTH_WINDOW}_diff": (
                    form.loc[home_team, f"opponent_adjusted_goal_difference_{STRENGTH_WINDOW}"]
                    - form.loc[away_team, f"opponent_adjusted_goal_difference_{STRENGTH_WINDOW}"]
                ),
                f"goals_against_top_opponents_{STRENGTH_WINDOW}_diff": (
                    form.loc[home_team, f"goals_against_top_opponents_{STRENGTH_WINDOW}"]
                    - form.loc[away_team, f"goals_against_top_opponents_{STRENGTH_WINDOW}"]
                ),
                f"ppg_top_opponents_{STRENGTH_WINDOW}_diff": (
                    form.loc[home_team, f"ppg_top_opponents_{STRENGTH_WINDOW}"]
                    - form.loc[away_team, f"ppg_top_opponents_{STRENGTH_WINDOW}"]
                ),
                "ucl_experience_matches_diff": form.loc[home_team, "ucl_matches"] - form.loc[away_team, "ucl_matches"],
                "home_away_goal_difference_balance_diff": (
                    (
                        form.loc[home_team, "home_goal_difference_10"]
                        - form.loc[away_team, "away_goal_difference_10"]
                    )
                    - (form.loc[home_team, "goal_difference_10"] - form.loc[away_team, "goal_difference_10"])
                ),
            }
        ]
    ).fillna(0.0)


def train_current_match_model(model_dataset: pd.DataFrame):
    return train_logistic_model(model_dataset, FULL_FEATURES)


def predict_match_probabilities(
    home_team: str,
    away_team: str,
    model_dataset: pd.DataFrame,
    match_history: pd.DataFrame,
    match_date: str | pd.Timestamp | None = None,
    competition: str = "Champions League",
) -> dict[str, float]:
    model = train_current_match_model(model_dataset)
    features = build_prediction_features(
        home_team,
        away_team,
        match_history,
        match_date=match_date,
        competition=competition,
    )
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
