from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "src" / "data" / "processed" / "matches_with_elo.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "src" / "data" / "processed" / "matches_with_features.csv"

DEFAULT_WINDOWS = (5, 10)

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
    sorted_matches["match_id"] = sorted_matches.index

    home_rows = pd.DataFrame(
        {
            "match_id": sorted_matches["match_id"],
            "date": sorted_matches["date"],
            "team": sorted_matches["home_team"],
            "opponent": sorted_matches["away_team"],
            "is_home": True,
            "goals_for": sorted_matches["home_goals"],
            "goals_against": sorted_matches["away_goals"],
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
            "goals_for": sorted_matches["away_goals"],
            "goals_against": sorted_matches["home_goals"],
            "points": [points_from_result(result, False) for result in sorted_matches["result"]],
        }
    )

    history = pd.concat([home_rows, away_rows], ignore_index=True)
    history["goal_difference"] = history["goals_for"] - history["goals_against"]
    return history.sort_values(["team", "date", "match_id"], kind="mergesort").reset_index(drop=True)


def add_rolling_form_features(
    matches: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    featured = matches.copy()
    featured["date"] = pd.to_datetime(featured["date"], errors="raise")
    featured = featured.sort_values(
        ["date", "competition", "home_team", "away_team"],
        kind="mergesort",
    ).reset_index(drop=True)
    featured["match_id"] = featured.index

    history = build_team_match_history(featured)
    metrics = ["points", "goals_for", "goals_against", "goal_difference"]

    for window in windows:
        for metric in metrics:
            feature_name = f"{metric}_{window}_before"
            history[feature_name] = history.groupby("team", sort=False)[metric].transform(
                lambda values: values.shift(1).rolling(window=window, min_periods=1).mean()
            )

        history[f"ppg_{window}_before"] = history[f"points_{window}_before"]

    feature_columns: list[str] = []
    for window in windows:
        feature_columns.extend(
            [
                f"ppg_{window}_before",
                f"goals_for_{window}_before",
                f"goals_against_{window}_before",
                f"goal_difference_{window}_before",
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

    for window in windows:
        featured[f"ppg_{window}_diff"] = (
            featured[f"home_ppg_{window}_before"] - featured[f"away_ppg_{window}_before"]
        )
        featured[f"goal_difference_{window}_diff"] = (
            featured[f"home_goal_difference_{window}_before"]
            - featured[f"away_goal_difference_{window}_before"]
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
