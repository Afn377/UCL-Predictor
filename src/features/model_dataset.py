from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "src" / "data" / "processed" / "matches_with_features.csv"
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "src" / "data" / "processed" / "model_dataset.csv"

ID_COLUMNS = [
    "date",
    "season",
    "competition",
    "stage",
    "home_team",
    "away_team",
]
FEATURE_COLUMNS = [
    "elo_diff",
    "ppg_5_diff",
    "goals_for_5_diff",
    "goals_against_5_diff",
    "goal_difference_5_diff",
    "venue_ppg_5_diff",
    "venue_goals_for_5_diff",
    "venue_goals_against_5_diff",
    "venue_goal_difference_5_diff",
    "shot_proxy_xg_for_5_diff",
    "shot_proxy_xg_against_5_diff",
    "shot_proxy_xg_difference_5_diff",
    "venue_shot_proxy_xg_for_5_diff",
    "venue_shot_proxy_xg_against_5_diff",
    "venue_shot_proxy_xg_difference_5_diff",
    "ppg_10_diff",
    "goals_for_10_diff",
    "goals_against_10_diff",
    "goal_difference_10_diff",
    "venue_ppg_10_diff",
    "venue_goals_for_10_diff",
    "venue_goals_against_10_diff",
    "venue_goal_difference_10_diff",
    "shot_proxy_xg_for_10_diff",
    "shot_proxy_xg_against_10_diff",
    "shot_proxy_xg_difference_10_diff",
    "venue_shot_proxy_xg_for_10_diff",
    "venue_shot_proxy_xg_against_10_diff",
    "venue_shot_proxy_xg_difference_10_diff",
    "rest_days_diff",
    "matches_last_7_diff",
    "matches_last_14_diff",
    "matches_last_30_diff",
    "is_champions_league",
    "is_new_ucl_format",
    "opponent_adjusted_goal_difference_38_diff",
    "goals_against_top_opponents_38_diff",
    "ppg_top_opponents_38_diff",
    "ucl_experience_matches_diff",
    "home_away_goal_difference_balance_diff",
]
SHOT_PROXY_FEATURE_COLUMNS = [
    column for column in FEATURE_COLUMNS if "shot_proxy_xg" in column
]
TARGET_COLUMN = "result"
MODEL_COLUMNS = [*ID_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN]


def fill_missing_feature_values(
    matches: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    filled = matches.copy()
    shot_proxy_columns = [
        column for column in feature_columns if column in SHOT_PROXY_FEATURE_COLUMNS
    ]
    if shot_proxy_columns:
        # zero is a neutral diff, not zero shots taken
        filled[shot_proxy_columns] = filled[shot_proxy_columns].fillna(0.0)
    # unknown rest days -> 0 means no measured advantage
    if "rest_days_diff" in feature_columns:
        filled["rest_days_diff"] = filled["rest_days_diff"].fillna(0.0)
    return filled


def build_model_dataset(
    matches: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    selected_features = feature_columns or FEATURE_COLUMNS
    required_columns = [*ID_COLUMNS, *selected_features, TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in matches.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"matches is missing required columns: {missing}")

    # sort before selecting so temporal splits stay deterministic
    model_data = fill_missing_feature_values(matches, selected_features)
    model_data["date"] = pd.to_datetime(model_data["date"], errors="raise")
    model_data = model_data.sort_values(
        ["date", "competition", "home_team", "away_team"],
        kind="mergesort",
    ).reset_index(drop=True)
    model_data = model_data[required_columns]
    model_data = model_data.dropna(subset=selected_features)
    model_data[TARGET_COLUMN] = model_data[TARGET_COLUMN].astype("int64")

    return model_data.reset_index(drop=True)


def write_model_dataset(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    matches = pd.read_csv(input_path)
    model_data = build_model_dataset(matches, feature_columns=feature_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_data.to_csv(output_path, index=False)
    return model_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the model-ready match dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    model_data = write_model_dataset(input_path=args.input, output_path=args.output)
    print(f"Wrote {len(model_data)} model-ready rows to {args.output}.")


if __name__ == "__main__":
    main()
