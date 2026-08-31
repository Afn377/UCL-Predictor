from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "src" / "data" / "processed" / "matches.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "src" / "data" / "processed" / "matches_with_elo.csv"

DEFAULT_INITIAL_RATING = 1500.0
DEFAULT_K_FACTOR = 20.0

REQUIRED_COLUMNS = ["date", "home_team", "away_team", "result"]


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def score_from_result(result: int) -> tuple[float, float]:
    if result == 2:
        return 1.0, 0.0
    if result == 1:
        return 0.5, 0.5
    if result == 0:
        return 0.0, 1.0

    raise ValueError(f"Unexpected result value: {result}")


def add_elo_features(
    matches: pd.DataFrame,
    initial_rating: float = DEFAULT_INITIAL_RATING,
    k_factor: float = DEFAULT_K_FACTOR,
) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in matches.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"matches is missing required columns: {missing}")

    featured = matches.copy()
    featured["date"] = pd.to_datetime(featured["date"], errors="raise")
    featured = featured.sort_values(
        ["date", "competition", "home_team", "away_team"],
        kind="mergesort",
    ).reset_index(drop=True)

    ratings: dict[str, float] = {}
    home_elo_before: list[float] = []
    away_elo_before: list[float] = []

    for row in featured.itertuples(index=False):
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        home_rating = ratings.get(home_team, initial_rating)
        away_rating = ratings.get(away_team, initial_rating)

        home_elo_before.append(home_rating)
        away_elo_before.append(away_rating)

        expected_home = expected_score(home_rating, away_rating)
        expected_away = 1 - expected_home
        actual_home, actual_away = score_from_result(int(row.result))

        ratings[home_team] = home_rating + k_factor * (actual_home - expected_home)
        ratings[away_team] = away_rating + k_factor * (actual_away - expected_away)

    featured["home_elo_before"] = home_elo_before
    featured["away_elo_before"] = away_elo_before
    featured["elo_diff"] = featured["home_elo_before"] - featured["away_elo_before"]

    return featured


def write_matches_with_elo(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    initial_rating: float = DEFAULT_INITIAL_RATING,
    k_factor: float = DEFAULT_K_FACTOR,
) -> pd.DataFrame:
    matches = pd.read_csv(input_path)
    featured = add_elo_features(
        matches,
        initial_rating=initial_rating,
        k_factor=k_factor,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(output_path, index=False)
    return featured


def main() -> None:
    parser = argparse.ArgumentParser(description="Add leakage-safe pre-match Elo features.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--initial-rating", type=float, default=DEFAULT_INITIAL_RATING)
    parser.add_argument("--k-factor", type=float, default=DEFAULT_K_FACTOR)
    args = parser.parse_args()

    featured = write_matches_with_elo(
        input_path=args.input,
        output_path=args.output,
        initial_rating=args.initial_rating,
        k_factor=args.k_factor,
    )
    print(f"Wrote {len(featured)} matches with Elo features to {args.output}.")


if __name__ == "__main__":
    main()
