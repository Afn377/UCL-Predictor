from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.build_master_matches import DEFAULT_TEAM_MAPPING_PATH
from src.data.team_names import load_team_name_mapping, normalize_team_names
from src.models.baselines import FULL_FEATURES
from src.models.poisson import prepare_score_model_dataset, train_poisson_models
from src.models.predict import build_prediction_features
from src.simulation.ucl_state import build_league_phase_table, load_ucl_matches


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_INPUT = ROOT / "src" / "data" / "processed" / "matches_with_features.csv"
DEFAULT_MATCH_HISTORY = ROOT / "src" / "data" / "processed" / "matches_with_elo.csv"
DEFAULT_OUTPUT_DIR = ROOT / "src" / "data" / "processed"


def normalize_ucl_fixture_names(matches: pd.DataFrame, mapping_path: Path = DEFAULT_TEAM_MAPPING_PATH) -> pd.DataFrame:
    mapping = load_team_name_mapping(mapping_path)
    normalized = matches.copy()
    normalized["home_team_raw"] = normalized["home_team"]
    normalized["away_team_raw"] = normalized["away_team"]
    normalized["home_team"] = normalize_team_names(normalized["home_team"], mapping)
    normalized["away_team"] = normalize_team_names(normalized["away_team"], mapping)
    return normalized


def split_league_phase_schedule(
    matches: pd.DataFrame,
    season: str,
    cutoff_date: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(cutoff_date)
    league_matches = matches.loc[(matches["season"] == season) & (matches["stage"] == "league_phase")].copy()
    league_matches["date"] = pd.to_datetime(league_matches["date"], errors="raise")

    completed = league_matches.loc[
        (league_matches["date"] <= cutoff) & league_matches["home_goals"].notna() & league_matches["away_goals"].notna()
    ].copy()
    remaining = league_matches.loc[league_matches["date"] > cutoff].copy()
    remaining[["home_goals", "away_goals"]] = np.nan

    return completed.reset_index(drop=True), remaining.reset_index(drop=True)


def train_score_models_until(model_input: pd.DataFrame, cutoff_date: str | pd.Timestamp):
    cutoff = pd.Timestamp(cutoff_date)
    score_data = prepare_score_model_dataset(model_input)
    train = score_data.loc[score_data["date"] <= cutoff].copy()
    if train.empty:
        raise ValueError("No score-model training rows are available before the cutoff date.")
    return train_poisson_models(train)


def _fixture_features(
    home_team: str,
    away_team: str,
    match_history: pd.DataFrame,
    match_date: str | pd.Timestamp,
    competition: str,
) -> pd.DataFrame:
    try:
        return build_prediction_features(
            home_team,
            away_team,
            match_history,
            match_date=match_date,
            competition=competition,
        )
    except ValueError:
        return pd.DataFrame([{feature: 0.0 for feature in FULL_FEATURES}])


def predict_remaining_goal_lambdas(
    remaining: pd.DataFrame,
    model_input: pd.DataFrame,
    match_history: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
) -> pd.DataFrame:
    home_model, away_model = train_score_models_until(model_input, cutoff_date)
    history = match_history.loc[pd.to_datetime(match_history["date"]) <= pd.Timestamp(cutoff_date)].copy()

    rows = []
    for fixture in remaining.itertuples(index=False):
        features = _fixture_features(
            str(fixture.home_team),
            str(fixture.away_team),
            history,
            fixture.date,
            fixture.competition,
        )
        rows.append(
            {
                "date": fixture.date,
                "season": fixture.season,
                "competition": fixture.competition,
                "stage": fixture.stage,
                "home_team": fixture.home_team,
                "away_team": fixture.away_team,
                "lambda_home": float(home_model.predict(features[FULL_FEATURES])[0]),
                "lambda_away": float(away_model.predict(features[FULL_FEATURES])[0]),
            }
        )

    return pd.DataFrame(rows)


def _simulated_remaining_matches(goal_lambdas: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    simulated = goal_lambdas[["date", "season", "competition", "stage", "home_team", "away_team"]].copy()
    simulated["home_goals"] = rng.poisson(goal_lambdas["lambda_home"].clip(lower=0.05).to_numpy())
    simulated["away_goals"] = rng.poisson(goal_lambdas["lambda_away"].clip(lower=0.05).to_numpy())
    return simulated


def qualification_summary(simulated_tables: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for table in simulated_tables:
        for team in table.itertuples(index=False):
            rows.append(
                {
                    "team": team.team,
                    "rank": team.rank,
                    "points": team.points,
                    "top_8": int(team.rank <= 8),
                    "top_24": int(team.rank <= 24),
                    "eliminated": int(team.rank > 24),
                }
            )

    records = pd.DataFrame(rows)
    summary = (
        records.groupby("team", as_index=False)
        .agg(
            average_rank=("rank", "mean"),
            average_points=("points", "mean"),
            top_8_probability=("top_8", "mean"),
            top_24_probability=("top_24", "mean"),
            elimination_probability=("eliminated", "mean"),
        )
        .sort_values(
            ["top_8_probability", "top_24_probability", "average_points", "average_rank", "team"],
            ascending=[False, False, False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    summary.insert(0, "projected_rank", range(1, len(summary) + 1))
    return summary


def simulate_league_phase(
    completed: pd.DataFrame,
    goal_lambdas: pd.DataFrame,
    season: str,
    n_simulations: int = 1000,
    random_seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    simulated_tables = []
    completed_base = completed[
        ["date", "season", "competition", "stage", "home_team", "away_team", "home_goals", "away_goals"]
    ].copy()

    for _ in range(n_simulations):
        simulated_matches = _simulated_remaining_matches(goal_lambdas, rng)
        all_matches = pd.concat([completed_base, simulated_matches], ignore_index=True)
        simulated_tables.append(build_league_phase_table(all_matches, season=season))

    return qualification_summary(simulated_tables)


def run_league_phase_simulation(
    season: str,
    cutoff_date: str | pd.Timestamp,
    n_simulations: int = 1000,
    random_seed: int = 7,
    raw_dir: Path | None = None,
    model_input_path: Path = DEFAULT_MODEL_INPUT,
    match_history_path: Path = DEFAULT_MATCH_HISTORY,
) -> pd.DataFrame:
    raw_matches = load_ucl_matches(raw_dir=raw_dir or ROOT / "src" / "data" / "raw" / "champions_league", season=season)
    matches = normalize_ucl_fixture_names(raw_matches)
    completed, remaining = split_league_phase_schedule(matches, season=season, cutoff_date=cutoff_date)

    if remaining.empty:
        return qualification_summary([build_league_phase_table(completed, season=season)])

    model_input = pd.read_csv(model_input_path, parse_dates=["date"])
    match_history = pd.read_csv(match_history_path, parse_dates=["date"])
    goal_lambdas = predict_remaining_goal_lambdas(remaining, model_input, match_history, cutoff_date)
    return simulate_league_phase(completed, goal_lambdas, season, n_simulations, random_seed)


def write_league_phase_simulation(
    season: str = "2025_26",
    cutoff_date: str | pd.Timestamp = "2025-11-06",
    n_simulations: int = 1000,
    random_seed: int = 7,
    output_path: Path | None = None,
) -> pd.DataFrame:
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / f"ucl_league_phase_simulation_{season}.csv"

    summary = run_league_phase_simulation(
        season=season,
        cutoff_date=cutoff_date,
        n_simulations=n_simulations,
        random_seed=random_seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Champions League league-phase qualification odds.")
    parser.add_argument("season", nargs="?", default="2025_26")
    parser.add_argument("--cutoff-date", default="2025-11-06")
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = write_league_phase_simulation(
        season=args.season,
        cutoff_date=args.cutoff_date,
        n_simulations=args.simulations,
        random_seed=args.seed,
        output_path=args.output,
    )
    print(summary.head(36).to_string(index=False))


if __name__ == "__main__":
    main()
