from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UCL_RAW_DIR = ROOT / "src" / "data" / "raw" / "champions_league"
DEFAULT_OUTPUT_DIR = ROOT / "src" / "data" / "processed"

REQUIRED_COLUMNS = [
    "date",
    "season",
    "competition",
    "stage",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
]


def load_ucl_matches(raw_dir: Path = DEFAULT_UCL_RAW_DIR, season: str | None = None) -> pd.DataFrame:
    paths = sorted(raw_dir.glob("*.csv"))
    if season is not None:
        paths = [path for path in paths if path.stem == season]
    if not paths:
        raise FileNotFoundError(f"No Champions League CSV files found in {raw_dir}")

    matches = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in matches.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"UCL matches are missing required columns: {missing}")

    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches["home_goals"] = pd.to_numeric(matches["home_goals"], errors="coerce")
    matches["away_goals"] = pd.to_numeric(matches["away_goals"], errors="coerce")
    if season is not None:
        matches = matches.loc[matches["season"] == season].copy()

    return matches.sort_values(["date", "stage", "home_team", "away_team"]).reset_index(drop=True)


def league_phase_matches(
    matches: pd.DataFrame,
    season: str,
    cutoff_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    mask = (matches["season"] == season) & (matches["stage"] == "league_phase")
    if cutoff_date is not None:
        mask &= matches["date"] <= pd.Timestamp(cutoff_date)

    return matches.loc[mask].dropna(subset=["home_goals", "away_goals"]).copy()


def _table_rows(matches: pd.DataFrame) -> pd.DataFrame:
    home_rows = pd.DataFrame(
        {
            "team": matches["home_team"],
            "opponent": matches["away_team"],
            "goals_for": matches["home_goals"],
            "goals_against": matches["away_goals"],
        }
    )
    away_rows = pd.DataFrame(
        {
            "team": matches["away_team"],
            "opponent": matches["home_team"],
            "goals_for": matches["away_goals"],
            "goals_against": matches["home_goals"],
        }
    )

    rows = pd.concat([home_rows, away_rows], ignore_index=True)
    rows["played"] = 1
    rows["won"] = (rows["goals_for"] > rows["goals_against"]).astype(int)
    rows["drawn"] = (rows["goals_for"] == rows["goals_against"]).astype(int)
    rows["lost"] = (rows["goals_for"] < rows["goals_against"]).astype(int)
    rows["goal_difference"] = rows["goals_for"] - rows["goals_against"]
    rows["points"] = (rows["won"] * 3) + rows["drawn"]
    return rows


def build_league_phase_table(
    matches: pd.DataFrame,
    season: str,
    cutoff_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    league_matches = league_phase_matches(matches, season=season, cutoff_date=cutoff_date)
    if league_matches.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "team",
                "played",
                "won",
                "drawn",
                "lost",
                "goals_for",
                "goals_against",
                "goal_difference",
                "points",
            ]
        )

    table = (
        _table_rows(league_matches)
        .groupby("team", as_index=False)
        .agg(
            played=("played", "sum"),
            won=("won", "sum"),
            drawn=("drawn", "sum"),
            lost=("lost", "sum"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            goal_difference=("goal_difference", "sum"),
            points=("points", "sum"),
        )
        .sort_values(
            ["points", "goal_difference", "goals_for", "won", "team"],
            ascending=[False, False, False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    table.insert(0, "rank", range(1, len(table) + 1))
    return table


def write_league_phase_table(
    season: str = "2025_26",
    raw_dir: Path = DEFAULT_UCL_RAW_DIR,
    output_path: Path | None = None,
    cutoff_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / f"ucl_league_phase_table_{season}.csv"

    matches = load_ucl_matches(raw_dir=raw_dir, season=season)
    table = build_league_phase_table(matches, season=season, cutoff_date=cutoff_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Champions League league-phase table.")
    parser.add_argument("season", nargs="?", default="2025_26")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_UCL_RAW_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cutoff-date", default=None)
    args = parser.parse_args()

    table = write_league_phase_table(
        season=args.season,
        raw_dir=args.raw_dir,
        output_path=args.output,
        cutoff_date=args.cutoff_date,
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
