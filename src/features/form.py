from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "src" / "data" / "processed" / "matches_with_elo.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "src" / "data" / "processed" / "matches_with_features.csv"

DEFAULT_WINDOWS = (5, 10)
STRENGTH_WINDOW = 38
CONGESTION_WINDOWS = (7, 14, 30)
TOP_OPPONENT_ELO_CUTOFF = 1550.0
DEFAULT_ELO_BASELINE = 1500.0

REQUIRED_COLUMNS = [
    "date",
    "competition",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
]


def points_from_result(result: int, is_home: bool) -> int:
    if result == 1:
        return 1
    if result == 2:
        return 3 if is_home else 0
    if result == 0:
        return 0 if is_home else 3

    raise ValueError(f"Unexpected result value: {result}")


def build_team_match_history(matches: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in matches.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"matches is missing required columns: {missing}")

    sorted_matches = matches.copy()
    sorted_matches["date"] = pd.to_datetime(sorted_matches["date"], errors="raise")
    sorted_matches = sorted_matches.sort_values(
        ["date", "competition", "home_team", "away_team"],
        kind="mergesort",
    ).reset_index(drop=True)
    if "season" not in sorted_matches.columns:
        sorted_matches["season"] = pd.NA
    if "home_elo_before" not in sorted_matches.columns:
        sorted_matches["home_elo_before"] = DEFAULT_ELO_BASELINE
    if "away_elo_before" not in sorted_matches.columns:
        sorted_matches["away_elo_before"] = DEFAULT_ELO_BASELINE
    sorted_matches["match_id"] = sorted_matches.index

    home_rows = pd.DataFrame(
        {
            "match_id": sorted_matches["match_id"],
            "date": sorted_matches["date"],
            "team": sorted_matches["home_team"],
            "opponent": sorted_matches["away_team"],
            "is_home": True,
            "competition": sorted_matches["competition"],
            "season": sorted_matches["season"],
            "goals_for": sorted_matches["home_goals"],
            "goals_against": sorted_matches["away_goals"],
            "team_elo_before": sorted_matches["home_elo_before"],
            "opponent_elo_before": sorted_matches["away_elo_before"],
            "points": [points_from_result(result, True) for result in sorted_matches["result"]],
        }
    )
    away_rows = pd.DataFrame(
        {
            "match_id": sorted_matches["match_id"],
            "date": sorted_matches["date"],
            "team": sorted_matches["away_team"],
            "opponent": sorted_matches["home_team"],
            "is_home": False,
            "competition": sorted_matches["competition"],
            "season": sorted_matches["season"],
            "goals_for": sorted_matches["away_goals"],
            "goals_against": sorted_matches["home_goals"],
            "team_elo_before": sorted_matches["away_elo_before"],
            "opponent_elo_before": sorted_matches["home_elo_before"],
            "points": [points_from_result(result, False) for result in sorted_matches["result"]],
        }
    )

    history = pd.concat([home_rows, away_rows], ignore_index=True)
    history["goal_difference"] = history["goals_for"] - history["goals_against"]
    history["opponent_adjusted_goal_difference"] = history["goal_difference"] + (
        (history["opponent_elo_before"] - DEFAULT_ELO_BASELINE) / 400
    )
    history["is_top_opponent"] = history["opponent_elo_before"] >= TOP_OPPONENT_ELO_CUTOFF
    return history.sort_values(["team", "date", "match_id"], kind="mergesort").reset_index(drop=True)


def add_rest_and_congestion_features(
    history: pd.DataFrame,
    congestion_windows: tuple[int, ...] = CONGESTION_WINDOWS,
) -> pd.DataFrame:
    featured = history.copy()
    featured["days_since_match_before"] = pd.NA
    for window in congestion_windows:
        featured[f"matches_last_{window}_before"] = 0

    for _, team_rows in featured.groupby("team", sort=False):
        previous_dates: list[pd.Timestamp] = []
        for index, row in team_rows.iterrows():
            current_date = pd.Timestamp(row["date"])
            if previous_dates:
                featured.loc[index, "days_since_match_before"] = (current_date - previous_dates[-1]).days

            for window in congestion_windows:
                window_start = current_date - pd.Timedelta(days=window)
                featured.loc[index, f"matches_last_{window}_before"] = sum(
                    window_start <= previous_date < current_date for previous_date in previous_dates
                )

            previous_dates.append(current_date)

    return featured


def is_new_ucl_format(season: object) -> int:
    if pd.isna(season):
        return 0

    try:
        start_year = int(str(season).split("_")[0])
    except ValueError:
        return 0

    return int(start_year >= 2024)


def add_rolling_form_features(
    matches: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    balance_window = max(windows)
    featured = matches.copy()
    featured["date"] = pd.to_datetime(featured["date"], errors="raise")
    featured = featured.sort_values(
        ["date", "competition", "home_team", "away_team"],
        kind="mergesort",
    ).reset_index(drop=True)
    featured["match_id"] = featured.index

    history = add_rest_and_congestion_features(build_team_match_history(featured))
    metrics = ["points", "goals_for", "goals_against", "goal_difference"]

    for window in windows:
        for metric in metrics:
            feature_name = f"{metric}_{window}_before"
            history[feature_name] = history.groupby("team", sort=False)[metric].transform(
                lambda values: values.shift(1).rolling(window=window, min_periods=1).mean()
            )
            venue_feature_name = f"venue_{metric}_{window}_before"
            history[venue_feature_name] = history.groupby(["team", "is_home"], sort=False)[metric].transform(
                lambda values: values.shift(1).rolling(window=window, min_periods=1).mean()
            )

        history[f"ppg_{window}_before"] = history[f"points_{window}_before"]
        history[f"venue_ppg_{window}_before"] = history[f"venue_points_{window}_before"]

    history[f"opponent_adjusted_goal_difference_{STRENGTH_WINDOW}_before"] = history.groupby(
        "team", sort=False
    )["opponent_adjusted_goal_difference"].transform(
        lambda values: values.shift(1).rolling(window=STRENGTH_WINDOW, min_periods=1).mean()
    )
    history["ucl_matches_before"] = history.groupby("team", sort=False)["competition"].transform(
        lambda values: values.eq("Champions League").shift(1, fill_value=False).cumsum()
    )
    top_opponent_history = history.copy()
    top_opponent_history["top_opponent_goals_against"] = top_opponent_history["goals_against"].where(
        top_opponent_history["is_top_opponent"]
    )
    top_opponent_history["top_opponent_points"] = top_opponent_history["points"].where(
        top_opponent_history["is_top_opponent"]
    )
    history[f"goals_against_top_opponents_{STRENGTH_WINDOW}_before"] = top_opponent_history.groupby(
        "team", sort=False
    )["top_opponent_goals_against"].transform(
        lambda values: values.shift(1).rolling(window=STRENGTH_WINDOW, min_periods=1).mean()
    )
    history[f"ppg_top_opponents_{STRENGTH_WINDOW}_before"] = top_opponent_history.groupby("team", sort=False)[
        "top_opponent_points"
    ].transform(lambda values: values.shift(1).rolling(window=STRENGTH_WINDOW, min_periods=1).mean())

    feature_columns: list[str] = []
    for window in windows:
        feature_columns.extend(
            [
                f"ppg_{window}_before",
                f"goals_for_{window}_before",
                f"goals_against_{window}_before",
                f"goal_difference_{window}_before",
                f"venue_ppg_{window}_before",
                f"venue_goals_for_{window}_before",
                f"venue_goals_against_{window}_before",
                f"venue_goal_difference_{window}_before",
            ]
        )
    feature_columns.append("days_since_match_before")
    for congestion_window in CONGESTION_WINDOWS:
        feature_columns.append(f"matches_last_{congestion_window}_before")
    feature_columns.extend(
        [
            f"opponent_adjusted_goal_difference_{STRENGTH_WINDOW}_before",
            f"goals_against_top_opponents_{STRENGTH_WINDOW}_before",
            f"ppg_top_opponents_{STRENGTH_WINDOW}_before",
            "ucl_matches_before",
        ]
    )

    home_features = history[history["is_home"]][["match_id", *feature_columns]].rename(
        columns={column: f"home_{column}" for column in feature_columns}
    )
    away_features = history[~history["is_home"]][["match_id", *feature_columns]].rename(
        columns={column: f"away_{column}" for column in feature_columns}
    )

    featured = featured.merge(home_features, on="match_id", how="left")
    featured = featured.merge(away_features, on="match_id", how="left")
    featured["is_champions_league"] = (featured["competition"] == "Champions League").astype(int)
    featured["is_new_ucl_format"] = featured["season"].map(is_new_ucl_format)

    for window in windows:
        for metric in ["ppg", "goals_for", "goals_against", "goal_difference"]:
            home_venue_column = f"home_venue_{metric}_{window}_before"
            away_venue_column = f"away_venue_{metric}_{window}_before"
            featured[home_venue_column] = featured[home_venue_column].fillna(
                featured[f"home_{metric}_{window}_before"]
            )
            featured[away_venue_column] = featured[away_venue_column].fillna(
                featured[f"away_{metric}_{window}_before"]
            )

        featured[f"ppg_{window}_diff"] = (
            featured[f"home_ppg_{window}_before"] - featured[f"away_ppg_{window}_before"]
        )
        featured[f"goals_for_{window}_diff"] = (
            featured[f"home_goals_for_{window}_before"] - featured[f"away_goals_for_{window}_before"]
        )
        featured[f"goals_against_{window}_diff"] = (
            featured[f"home_goals_against_{window}_before"] - featured[f"away_goals_against_{window}_before"]
        )
        featured[f"goal_difference_{window}_diff"] = (
            featured[f"home_goal_difference_{window}_before"]
            - featured[f"away_goal_difference_{window}_before"]
        )
        featured[f"venue_ppg_{window}_diff"] = (
            featured[f"home_venue_ppg_{window}_before"] - featured[f"away_venue_ppg_{window}_before"]
        )
        featured[f"venue_goals_for_{window}_diff"] = (
            featured[f"home_venue_goals_for_{window}_before"]
            - featured[f"away_venue_goals_for_{window}_before"]
        )
        featured[f"venue_goals_against_{window}_diff"] = (
            featured[f"home_venue_goals_against_{window}_before"]
            - featured[f"away_venue_goals_against_{window}_before"]
        )
        featured[f"venue_goal_difference_{window}_diff"] = (
            featured[f"home_venue_goal_difference_{window}_before"]
            - featured[f"away_venue_goal_difference_{window}_before"]
        )
    featured["rest_days_diff"] = featured["home_days_since_match_before"] - featured["away_days_since_match_before"]
    for congestion_window in CONGESTION_WINDOWS:
        featured[f"matches_last_{congestion_window}_diff"] = (
            featured[f"home_matches_last_{congestion_window}_before"]
            - featured[f"away_matches_last_{congestion_window}_before"]
        )
    featured[f"opponent_adjusted_goal_difference_{STRENGTH_WINDOW}_diff"] = (
        featured[f"home_opponent_adjusted_goal_difference_{STRENGTH_WINDOW}_before"]
        - featured[f"away_opponent_adjusted_goal_difference_{STRENGTH_WINDOW}_before"]
    )
    featured[f"goals_against_top_opponents_{STRENGTH_WINDOW}_diff"] = (
        featured[f"home_goals_against_top_opponents_{STRENGTH_WINDOW}_before"]
        - featured[f"away_goals_against_top_opponents_{STRENGTH_WINDOW}_before"]
    )
    featured[f"ppg_top_opponents_{STRENGTH_WINDOW}_diff"] = (
        featured[f"home_ppg_top_opponents_{STRENGTH_WINDOW}_before"]
        - featured[f"away_ppg_top_opponents_{STRENGTH_WINDOW}_before"]
    )
    featured["ucl_experience_matches_diff"] = (
        featured["home_ucl_matches_before"] - featured["away_ucl_matches_before"]
    )
    featured["home_away_goal_difference_balance_diff"] = (
        (
            featured[f"home_venue_goal_difference_{balance_window}_before"]
            - featured[f"away_venue_goal_difference_{balance_window}_before"]
        )
        - featured[f"goal_difference_{balance_window}_diff"]
    )

    return featured.drop(columns=["match_id"])


def write_matches_with_form_features(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    matches = pd.read_csv(input_path)
    featured = add_rolling_form_features(matches, windows=windows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(output_path, index=False)
    return featured


def main() -> None:
    parser = argparse.ArgumentParser(description="Add leakage-safe rolling form features.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    featured = write_matches_with_form_features(input_path=args.input, output_path=args.output)
    print(f"Wrote {len(featured)} matches with rolling form features to {args.output}.")


if __name__ == "__main__":
    main()
