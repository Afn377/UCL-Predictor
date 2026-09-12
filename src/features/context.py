"""Prior-match xG and separate attacking/defending levels for forecasting."""

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.market import (
    MATCH_KEYS,
    attach_footiqo_closing_odds,
    attach_market_odds,
    load_market_odds,
)
from src.data.team_names import load_team_name_mapping, normalize_team_names
from src.features.lineups import attach_lineups
from src.features.model_dataset import FEATURE_COLUMNS
from src.features.strength import add_strength_features, attach_clubelo

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "src/data/processed/matches_with_features.csv"
DEFAULT_OUTPUT = ROOT / "src/data/processed/matches_with_context.csv"
DEFAULT_XG = ROOT / "src/data/raw/understat/matches.csv"
DEFAULT_LINEUPS = ROOT / "src/data/external/lineups.csv"
XG_METRICS = ["xg_for", "xg_against", "npxg_for", "npxg_against", "adjusted_npxgd"]
XG_FEATURES = [
    f"{side}_{metric}_{window}"
    for side in ("home", "away")
    for window in (5, 10, 38)
    for metric in XG_METRICS
]
XG_FEATURES += [
    f"{side}_{metric}"
    for side in ("home", "away")
    for metric in ("away_npxgd_10", "xg_count_38", "xg_age_days")
]
LEVEL_FEATURES = [
    f"{side}_{metric}_10"
    for side in ("home", "away")
    for metric in ("attack", "defense", "venue_attack", "venue_defense")
]
LEVEL_FEATURES += ["competition_home_goals", "competition_away_goals"]
RICH_FEATURES = list(dict.fromkeys(FEATURE_COLUMNS + LEVEL_FEATURES + XG_FEATURES))


def _mean(history, window, position):
    recent = list(history)[-window:]
    return float(np.mean([row[position] for row in recent])) if recent else np.nan


def attach_xg(
    matches: pd.DataFrame, xg: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # understat and result files spell clubs differently; normalize both
    aliases = load_team_name_mapping(ROOT / "src/data/understat_team_mapping.csv")
    data, external = matches.copy(), xg.copy()
    for frame in (data, external):
        frame["date"] = pd.to_datetime(frame.date).dt.normalize()
        for side in ("home", "away"):
            frame[f"{side}_key"] = normalize_team_names(frame[f"{side}_team"], aliases)
    keys = ["date", "competition", "home_key", "away_key"]
    if external.duplicated(keys).any():
        raise ValueError("Duplicate xG fixture keys")
    values = ["home_xg", "away_xg", "home_npxg", "away_npxg"]
    # rename source scores so the merge can't clobber result columns
    external = external.rename(
        columns={
            "home_goals": "xg_source_home_goals",
            "away_goals": "xg_source_away_goals",
        }
    )
    source_columns = values + ["xg_source_home_goals", "xg_source_away_goals"]
    joined = data.merge(
        external[keys + source_columns], on=keys, how="left", validate="one_to_one"
    )
    joined["xg_match_status"] = np.where(joined.home_xg.notna(), "exact", "unmatched")
    # sources differ by a day; accept only a unique nearby match
    fixture_keys = ["competition", "season", "home_key", "away_key"]
    missing = joined.loc[joined.home_xg.isna(), fixture_keys + ["date"]].reset_index(
        names="row_id"
    )
    candidates = missing.merge(
        external[fixture_keys + ["date"] + source_columns],
        on=fixture_keys,
        suffixes=("", "_source"),
    )
    candidates = candidates[
        (candidates.date - candidates.date_source).abs() <= pd.Timedelta(days=1)
    ]
    candidates = candidates[~candidates.row_id.duplicated(keep=False)].set_index(
        "row_id"
    )
    joined.loc[candidates.index, source_columns] = candidates[source_columns]
    joined.loc[candidates.index, "xg_match_status"] = "date_offset"
    mismatch = pd.Series(False, index=joined.index)
    for side in ("home", "away"):
        observed = (
            joined[[f"{side}_goals", f"xg_source_{side}_goals"]].notna().all(axis=1)
        )
        mismatch |= observed & (
            joined[f"{side}_goals"] != joined[f"xg_source_{side}_goals"]
        )
    joined.loc[mismatch, values] = np.nan
    joined.loc[mismatch, "xg_match_status"] = "score_mismatch"
    # earlier xG rows warm up history only, not the dataset
    warmup = external[external.date < data.date.min()].rename(
        columns={
            "xg_source_home_goals": "home_goals",
            "xg_source_away_goals": "away_goals",
        }
    )
    return joined, warmup


def add_context_features(matches: pd.DataFrame, xg: pd.DataFrame) -> pd.DataFrame:
    data, warmup = attach_xg(matches, xg)
    data["context_row_id"] = np.arange(len(data))
    warmup = warmup.assign(context_row_id=-1)
    history = pd.concat([warmup, data], ignore_index=True).sort_values(
        "date", kind="stable"
    )
    team_goals = defaultdict(lambda: deque(maxlen=38))
    venue_goals = defaultdict(lambda: deque(maxlen=38))
    team_xg = defaultdict(lambda: deque(maxlen=38))
    away_xg = defaultdict(lambda: deque(maxlen=10))
    league_goals = defaultdict(lambda: deque(maxlen=380))
    league_xg = defaultdict(lambda: deque(maxlen=380))
    last_xg = {}
    feature_rows = {}
    for date, day in history.groupby("date", sort=True):
        # build features before adding today's results to history
        pending = []
        for match in day.itertuples(index=False):
            values = {}
            league = match.competition
            values["competition_home_goals"] = _mean(league_goals[league], 380, 0)
            values["competition_away_goals"] = _mean(league_goals[league], 380, 1)
            for side, other, venue in [("home", "away", "h"), ("away", "home", "a")]:
                team, opponent = (
                    getattr(match, f"{side}_key"),
                    getattr(match, f"{other}_key"),
                )
                if team in last_xg and (date - last_xg[team]).days > 180:
                    # six-month gap means old xG is no longer recent form
                    team_xg[team].clear()
                    away_xg[team].clear()
                for metric, position in [("attack", 0), ("defense", 1)]:
                    values[f"{side}_{metric}_10"] = _mean(
                        team_goals[team], 10, position
                    )
                    values[f"{side}_venue_{metric}_10"] = _mean(
                        venue_goals[team, venue], 10, position
                    )
                for window in (5, 10, 38):
                    for position, metric in enumerate(XG_METRICS):
                        values[f"{side}_{metric}_{window}"] = _mean(
                            team_xg[team], window, position
                        )
                values[f"{side}_away_npxgd_10"] = _mean(away_xg[team], 10, 0)
                values[f"{side}_xg_count_38"] = len(team_xg[team])
                values[f"{side}_xg_age_days"] = (
                    (date - last_xg[team]).days if team in last_xg else np.nan
                )
                goals_for, goals_against = (
                    getattr(match, f"{side}_goals"),
                    getattr(match, f"{other}_goals"),
                )
                xg_for, xg_against = (
                    getattr(match, f"{side}_xg"),
                    getattr(match, f"{other}_xg"),
                )
                non_penalty_xg_for, non_penalty_xg_against = (
                    getattr(match, f"{side}_npxg"),
                    getattr(match, f"{other}_npxg"),
                )
                league_mean = _mean(league_xg[league], 380, 0)
                league_mean = league_mean if np.isfinite(league_mean) else 1.3
                opponent_defense = _mean(team_xg[opponent], 38, 3)
                opponent_attack = _mean(team_xg[opponent], 38, 2)
                opponent_defense = (
                    opponent_defense if np.isfinite(opponent_defense) else league_mean
                )
                opponent_attack = (
                    opponent_attack if np.isfinite(opponent_attack) else league_mean
                )
                adjusted_xg_difference = non_penalty_xg_for * league_mean / max(
                    0.5, opponent_defense
                ) - non_penalty_xg_against * league_mean / max(0.5, opponent_attack)
                pending.append(
                    (
                        team,
                        venue,
                        goals_for,
                        goals_against,
                        xg_for,
                        xg_against,
                        non_penalty_xg_for,
                        non_penalty_xg_against,
                        adjusted_xg_difference,
                    )
                )
            row_id = match.context_row_id
            if row_id >= 0:
                feature_rows[int(row_id)] = values
        for (
            team,
            venue,
            goals_for,
            goals_against,
            xg_for,
            xg_against,
            non_penalty_xg_for,
            non_penalty_xg_against,
            adjusted_xg_difference,
        ) in pending:
            if pd.notna(goals_for) and pd.notna(goals_against):
                team_goals[team].append((goals_for, goals_against))
                venue_goals[team, venue].append((goals_for, goals_against))
            if all(
                pd.notna(value)
                for value in (
                    xg_for,
                    xg_against,
                    non_penalty_xg_for,
                    non_penalty_xg_against,
                )
            ):
                team_xg[team].append(
                    (
                        xg_for,
                        xg_against,
                        non_penalty_xg_for,
                        non_penalty_xg_against,
                        adjusted_xg_difference,
                    )
                )
                last_xg[team] = date
                if venue == "a":
                    away_xg[team].append((non_penalty_xg_for - non_penalty_xg_against,))
        for match in day.itertuples(index=False):
            if pd.notna(match.home_goals) and pd.notna(match.away_goals):
                league_goals[match.competition].append(
                    (match.home_goals, match.away_goals)
                )
            if pd.notna(match.home_npxg) and pd.notna(match.away_npxg):
                league_xg[match.competition].append(
                    ((match.home_npxg + match.away_npxg) / 2,)
                )
    features = pd.DataFrame.from_dict(feature_rows, orient="index").reindex(data.index)
    result = pd.concat(
        [data.drop(columns=["context_row_id", "home_key", "away_key"]), features],
        axis=1,
    )
    result = add_strength_features(result)
    clubelo_path = ROOT / "src/data/raw/clubelo/history.csv"
    if clubelo_path.exists():
        result = attach_clubelo(result, pd.read_csv(clubelo_path))
    return result


def write_context_features(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    xg_path: Path = DEFAULT_XG,
    lineups_path: Path = DEFAULT_LINEUPS,
) -> pd.DataFrame:
    # real xG is required; without it the model silently changes
    if not xg_path.exists():
        raise FileNotFoundError(
            "Run python -m src.data.download_understat to populate real xG first"
        )
    data = add_context_features(pd.read_csv(input_path), pd.read_csv(xg_path))
    data = attach_market_odds(data, load_market_odds())
    data = attach_footiqo_closing_odds(data)
    data = attach_lineups(
        data, pd.read_csv(lineups_path) if lineups_path.exists() else None
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    data.loc[data.xg_match_status.ne("exact"), MATCH_KEYS + ["xg_match_status"]].to_csv(
        output_path.parent / "xg_match_audit.csv",
        index=False,
    )
    coverage = data.groupby("competition").agg(
        rows=("date", "size"),
        matched_xg=("home_xg", "count"),
        home_xg_history=("home_npxg_for_10", "count"),
        away_xg_history=("away_npxg_for_10", "count"),
        market_quotes=("market_prob_0", "count"),
        home_lineups=("home_expected_xi_quality", "count"),
        away_lineups=("away_expected_xi_quality", "count"),
    )
    coverage.to_csv(output_path.parent / "context_coverage.csv")
    print(coverage.to_string(), flush=True)
    return data


if __name__ == "__main__":
    write_context_features()
