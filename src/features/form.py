from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "src" / "data" / "processed" / "matches_with_elo.csv"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "src" / "data" / "processed" / "matches_with_features.csv"
)

DEFAULT_WINDOWS = (5, 10)
STRENGTH_WINDOW = 38
CONGESTION_WINDOWS = (7, 14, 30)
MAX_REST_DAYS = 30
TOP_OPPONENT_ELO_CUTOFF = 1550.0
DEFAULT_ELO_BASELINE = 1500.0
SHOT_XG_WEIGHTS = {
    "shots": 0.04,
    "shots_on_target": 0.24,
    "corners": 0.025,
}

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
    # results are home-team view, so away points flip win/loss
    if result == 1:
        return 1
    if result == 2:
        return 3 if is_home else 0
    if result == 0:
        return 0 if is_home else 3

    raise ValueError(f"Unexpected result value: {result}")


def build_team_match_history(matches: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in matches.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"matches is missing required columns: {missing}")

    # chronological order drives every later rolling feature
    sorted_matches = matches.copy()
    sorted_matches["date"] = pd.to_datetime(sorted_matches["date"], errors="raise")
    sorted_matches = sorted_matches.sort_values(
        ["date", "competition", "home_team", "away_team"],
        kind="mergesort",
    ).reset_index(drop=True)
    # older fixtures may lack season/elo/shot columns
    if "season" not in sorted_matches.columns:
        sorted_matches["season"] = pd.NA
    if "home_elo_before" not in sorted_matches.columns:
        sorted_matches["home_elo_before"] = DEFAULT_ELO_BASELINE
    if "away_elo_before" not in sorted_matches.columns:
        sorted_matches["away_elo_before"] = DEFAULT_ELO_BASELINE
    for column in [
        "home_shots",
        "away_shots",
        "home_shots_on_target",
        "away_shots_on_target",
        "home_corners",
        "away_corners",
    ]:
        if column not in sorted_matches.columns:
            sorted_matches[column] = pd.NA
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
            "shots_for": sorted_matches["home_shots"],
            "shots_against": sorted_matches["away_shots"],
            "shots_on_target_for": sorted_matches["home_shots_on_target"],
            "shots_on_target_against": sorted_matches["away_shots_on_target"],
            "corners_for": sorted_matches["home_corners"],
            "corners_against": sorted_matches["away_corners"],
            "team_elo_before": sorted_matches["home_elo_before"],
            "opponent_elo_before": sorted_matches["away_elo_before"],
            "points": [
                points_from_result(result, True) for result in sorted_matches["result"]
            ],
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
            "shots_for": sorted_matches["away_shots"],
            "shots_against": sorted_matches["home_shots"],
            "shots_on_target_for": sorted_matches["away_shots_on_target"],
            "shots_on_target_against": sorted_matches["home_shots_on_target"],
            "corners_for": sorted_matches["away_corners"],
            "corners_against": sorted_matches["home_corners"],
            "team_elo_before": sorted_matches["away_elo_before"],
            "opponent_elo_before": sorted_matches["home_elo_before"],
            "points": [
                points_from_result(result, False) for result in sorted_matches["result"]
            ],
        }
    )

    # keep missing shot stats as na, not errors
    history = pd.concat([home_rows, away_rows], ignore_index=True)
    shot_columns = [
        "shots_for",
        "shots_against",
        "shots_on_target_for",
        "shots_on_target_against",
        "corners_for",
        "corners_against",
    ]
    for column in shot_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history["goal_difference"] = history["goals_for"] - history["goals_against"]
    # rough xg proxy; on-target shots weigh more
    history["shot_proxy_xg_for"] = (
        SHOT_XG_WEIGHTS["shots"] * history["shots_for"]
        + SHOT_XG_WEIGHTS["shots_on_target"] * history["shots_on_target_for"]
        + SHOT_XG_WEIGHTS["corners"] * history["corners_for"]
    )
    history["shot_proxy_xg_against"] = (
        SHOT_XG_WEIGHTS["shots"] * history["shots_against"]
        + SHOT_XG_WEIGHTS["shots_on_target"] * history["shots_on_target_against"]
        + SHOT_XG_WEIGHTS["corners"] * history["corners_against"]
    )
    opponent_adjustment = (history["opponent_elo_before"] - DEFAULT_ELO_BASELINE) / 800
    team_adjustment = (history["team_elo_before"] - DEFAULT_ELO_BASELINE) / 800
    history["opponent_adjusted_shot_proxy_xg_for"] = (
        history["shot_proxy_xg_for"] + opponent_adjustment
    )
    history["opponent_adjusted_shot_proxy_xg_against"] = (
        history["shot_proxy_xg_against"] - team_adjustment
    )
    history["shot_proxy_xg_difference"] = (
        history["opponent_adjusted_shot_proxy_xg_for"]
        - history["opponent_adjusted_shot_proxy_xg_against"]
    )
    history["opponent_adjusted_goal_difference"] = history["goal_difference"] + (
        (history["opponent_elo_before"] - DEFAULT_ELO_BASELINE) / 400
    )
    history["is_top_opponent"] = (
        history["opponent_elo_before"] >= TOP_OPPONENT_ELO_CUTOFF
    )
    return history.sort_values(
        ["team", "date", "match_id"], kind="mergesort"
    ).reset_index(drop=True)


def add_rest_and_congestion_features(
    history: pd.DataFrame,
    congestion_windows: tuple[int, ...] = CONGESTION_WINDOWS,
) -> pd.DataFrame:
    featured = history.copy()
    featured["days_since_match_before"] = pd.NA
    for window in congestion_windows:
        featured[f"matches_last_{window}_before"] = 0

    for _, team_rows in featured.groupby("team", sort=False):
        # prior dates only; exclude the current match
        previous_dates: list[pd.Timestamp] = []
        for index, row in team_rows.iterrows():
            current_date = pd.Timestamp(row["date"])
            if previous_dates:
                # long gaps mean missing seasons, not rest days
                gap = (current_date - previous_dates[-1]).days
                if 0 < gap <= MAX_REST_DAYS:
                    featured.loc[index, "days_since_match_before"] = gap

            for window in congestion_windows:
                window_start = current_date - pd.Timedelta(days=window)
                featured.loc[index, f"matches_last_{window}_before"] = sum(
                    window_start <= previous_date < current_date
                    for previous_date in previous_dates
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


def rolling_prior_observed_mean(values: pd.Series, window: int) -> pd.Series:
    """Average the last observed values, leaving the current match out."""
    observed_values: list[float] = []
    means: list[float | pd.NA] = []

    for value in pd.to_numeric(values, errors="coerce"):
        recent_values = observed_values[-window:]
        if recent_values:
            means.append(sum(recent_values) / len(recent_values))
        else:
            means.append(pd.NA)

        # missing shot shouldn't evict a real observation
        if pd.notna(value):
            observed_values.append(float(value))

    return pd.Series(means, index=values.index, dtype="Float64")


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
    # reset form spell after a long gap
    history["spell"] = history.groupby("team", sort=False).date.transform(
        lambda dates: dates.diff().dt.days.gt(180).cumsum()
    )
    standard_metrics = [
        "points",
        "goals_for",
        "goals_against",
        "goal_difference",
    ]
    shot_proxy_metrics = [
        "opponent_adjusted_shot_proxy_xg_for",
        "opponent_adjusted_shot_proxy_xg_against",
        "shot_proxy_xg_difference",
    ]

    for window in windows:
        for metric in standard_metrics:
            feature_name = f"{metric}_{window}_before"
            history[feature_name] = history.groupby(["team", "spell"], sort=False)[
                metric
            ].transform(
                lambda values: values.shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )
            venue_feature_name = f"venue_{metric}_{window}_before"
            history[venue_feature_name] = history.groupby(
                ["team", "spell", "is_home"], sort=False
            )[metric].transform(
                lambda values: values.shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )
        # shot gaps; window counts observations, not rows
        for metric in shot_proxy_metrics:
            feature_name = f"{metric}_{window}_before"
            history[feature_name] = history.groupby(["team", "spell"], sort=False)[
                metric
            ].transform(lambda values: rolling_prior_observed_mean(values, window))
            venue_feature_name = f"venue_{metric}_{window}_before"
            history[venue_feature_name] = history.groupby(
                ["team", "spell", "is_home"], sort=False
            )[metric].transform(
                lambda values: rolling_prior_observed_mean(values, window)
            )

        history[f"ppg_{window}_before"] = history[f"points_{window}_before"]
        history[f"venue_ppg_{window}_before"] = history[f"venue_points_{window}_before"]

    history[f"opponent_adjusted_goal_difference_{STRENGTH_WINDOW}_before"] = (
        history.groupby(
            ["team", "spell"], sort=False
        )[
            "opponent_adjusted_goal_difference"
        ].transform(
            lambda values: values.shift(1)
            .rolling(window=STRENGTH_WINDOW, min_periods=1)
            .mean()
        )
    )
    history["ucl_matches_before"] = history.groupby("team", sort=False)[
        "competition"
    ].transform(
        lambda values: values.eq("Champions League").shift(1, fill_value=False).cumsum()
    )
    top_opponent_history = history.copy()
    top_opponent_history["top_opponent_goals_against"] = top_opponent_history[
        "goals_against"
    ].where(top_opponent_history["is_top_opponent"])
    top_opponent_history["top_opponent_points"] = top_opponent_history["points"].where(
        top_opponent_history["is_top_opponent"]
    )
    history[f"goals_against_top_opponents_{STRENGTH_WINDOW}_before"] = (
        top_opponent_history.groupby(
            ["team", "spell"], sort=False
        )[
            "top_opponent_goals_against"
        ].transform(
            lambda values: values.shift(1)
            .rolling(window=STRENGTH_WINDOW, min_periods=1)
            .mean()
        )
    )
    history[f"ppg_top_opponents_{STRENGTH_WINDOW}_before"] = (
        top_opponent_history.groupby(
            ["team", "spell"], sort=False
        )[
            "top_opponent_points"
        ].transform(
            lambda values: values.shift(1)
            .rolling(window=STRENGTH_WINDOW, min_periods=1)
            .mean()
        )
    )

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
                f"opponent_adjusted_shot_proxy_xg_for_{window}_before",
                f"opponent_adjusted_shot_proxy_xg_against_{window}_before",
                f"shot_proxy_xg_difference_{window}_before",
                f"venue_opponent_adjusted_shot_proxy_xg_for_{window}_before",
                f"venue_opponent_adjusted_shot_proxy_xg_against_{window}_before",
                f"venue_shot_proxy_xg_difference_{window}_before",
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
    featured["is_champions_league"] = (
        featured["competition"] == "Champions League"
    ).astype(int)
    featured["is_new_ucl_format"] = featured["season"].map(is_new_ucl_format)

    # sparse venue rows fall back to overall form
    venue_updates: dict[str, pd.Series] = {}
    for window in windows:
        for metric in ["ppg", "goals_for", "goals_against", "goal_difference"]:
            home_venue_column = f"home_venue_{metric}_{window}_before"
            away_venue_column = f"away_venue_{metric}_{window}_before"
            venue_updates[home_venue_column] = featured[home_venue_column].fillna(
                featured[f"home_{metric}_{window}_before"]
            )
            venue_updates[away_venue_column] = featured[away_venue_column].fillna(
                featured[f"away_{metric}_{window}_before"]
            )
        for metric in [
            "opponent_adjusted_shot_proxy_xg_for",
            "opponent_adjusted_shot_proxy_xg_against",
            "shot_proxy_xg_difference",
        ]:
            home_venue_column = f"home_venue_{metric}_{window}_before"
            away_venue_column = f"away_venue_{metric}_{window}_before"
            venue_updates[home_venue_column] = featured[home_venue_column].fillna(
                featured[f"home_{metric}_{window}_before"]
            )
            venue_updates[away_venue_column] = featured[away_venue_column].fillna(
                featured[f"away_{metric}_{window}_before"]
            )

    if venue_updates:
        featured = pd.concat(
            [
                featured.drop(columns=list(venue_updates)),
                pd.DataFrame(venue_updates, index=featured.index),
            ],
            axis=1,
        )

    derived_columns: dict[str, pd.Series] = {}
    # collect first; per-column adds fragment the frame
    for window in windows:
        derived_columns[f"ppg_{window}_diff"] = (
            featured[f"home_ppg_{window}_before"]
            - featured[f"away_ppg_{window}_before"]
        )
        derived_columns[f"goals_for_{window}_diff"] = (
            featured[f"home_goals_for_{window}_before"]
            - featured[f"away_goals_for_{window}_before"]
        )
        derived_columns[f"goals_against_{window}_diff"] = (
            featured[f"home_goals_against_{window}_before"]
            - featured[f"away_goals_against_{window}_before"]
        )
        derived_columns[f"goal_difference_{window}_diff"] = (
            featured[f"home_goal_difference_{window}_before"]
            - featured[f"away_goal_difference_{window}_before"]
        )
        derived_columns[f"venue_ppg_{window}_diff"] = (
            featured[f"home_venue_ppg_{window}_before"]
            - featured[f"away_venue_ppg_{window}_before"]
        )
        derived_columns[f"venue_goals_for_{window}_diff"] = (
            featured[f"home_venue_goals_for_{window}_before"]
            - featured[f"away_venue_goals_for_{window}_before"]
        )
        derived_columns[f"venue_goals_against_{window}_diff"] = (
            featured[f"home_venue_goals_against_{window}_before"]
            - featured[f"away_venue_goals_against_{window}_before"]
        )
        derived_columns[f"venue_goal_difference_{window}_diff"] = (
            featured[f"home_venue_goal_difference_{window}_before"]
            - featured[f"away_venue_goal_difference_{window}_before"]
        )
        derived_columns[f"shot_proxy_xg_for_{window}_diff"] = (
            featured[f"home_opponent_adjusted_shot_proxy_xg_for_{window}_before"]
            - featured[f"away_opponent_adjusted_shot_proxy_xg_for_{window}_before"]
        )
        derived_columns[f"shot_proxy_xg_against_{window}_diff"] = (
            featured[f"home_opponent_adjusted_shot_proxy_xg_against_{window}_before"]
            - featured[f"away_opponent_adjusted_shot_proxy_xg_against_{window}_before"]
        )
        derived_columns[f"shot_proxy_xg_difference_{window}_diff"] = (
            featured[f"home_shot_proxy_xg_difference_{window}_before"]
            - featured[f"away_shot_proxy_xg_difference_{window}_before"]
        )
        derived_columns[f"venue_shot_proxy_xg_for_{window}_diff"] = (
            featured[f"home_venue_opponent_adjusted_shot_proxy_xg_for_{window}_before"]
            - featured[
                f"away_venue_opponent_adjusted_shot_proxy_xg_for_{window}_before"
            ]
        )
        derived_columns[f"venue_shot_proxy_xg_against_{window}_diff"] = (
            featured[
                f"home_venue_opponent_adjusted_shot_proxy_xg_against_{window}_before"
            ]
            - featured[
                f"away_venue_opponent_adjusted_shot_proxy_xg_against_{window}_before"
            ]
        )
        derived_columns[f"venue_shot_proxy_xg_difference_{window}_diff"] = (
            featured[f"home_venue_shot_proxy_xg_difference_{window}_before"]
            - featured[f"away_venue_shot_proxy_xg_difference_{window}_before"]
        )
    derived_columns["rest_days_diff"] = (
        featured["home_days_since_match_before"]
        - featured["away_days_since_match_before"]
    )
    for congestion_window in CONGESTION_WINDOWS:
        derived_columns[f"matches_last_{congestion_window}_diff"] = (
            featured[f"home_matches_last_{congestion_window}_before"]
            - featured[f"away_matches_last_{congestion_window}_before"]
        )
    derived_columns[f"opponent_adjusted_goal_difference_{STRENGTH_WINDOW}_diff"] = (
        featured[f"home_opponent_adjusted_goal_difference_{STRENGTH_WINDOW}_before"]
        - featured[f"away_opponent_adjusted_goal_difference_{STRENGTH_WINDOW}_before"]
    )
    derived_columns[f"goals_against_top_opponents_{STRENGTH_WINDOW}_diff"] = (
        featured[f"home_goals_against_top_opponents_{STRENGTH_WINDOW}_before"]
        - featured[f"away_goals_against_top_opponents_{STRENGTH_WINDOW}_before"]
    )
    derived_columns[f"ppg_top_opponents_{STRENGTH_WINDOW}_diff"] = (
        featured[f"home_ppg_top_opponents_{STRENGTH_WINDOW}_before"]
        - featured[f"away_ppg_top_opponents_{STRENGTH_WINDOW}_before"]
    )
    derived_columns["ucl_experience_matches_diff"] = (
        featured["home_ucl_matches_before"] - featured["away_ucl_matches_before"]
    )
    derived_columns["home_away_goal_difference_balance_diff"] = (
        featured[f"home_venue_goal_difference_{balance_window}_before"]
        - featured[f"away_venue_goal_difference_{balance_window}_before"]
    ) - derived_columns[f"goal_difference_{balance_window}_diff"]
    # concat once to avoid fragmenting the frame
    featured = pd.concat(
        [featured, pd.DataFrame(derived_columns, index=featured.index)], axis=1
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
    parser = argparse.ArgumentParser(
        description="Add leakage-safe rolling form features."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    featured = write_matches_with_form_features(
        input_path=args.input, output_path=args.output
    )
    print(f"Wrote {len(featured)} matches with rolling form features to {args.output}.")


if __name__ == "__main__":
    main()
