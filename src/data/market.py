"""Historical market benchmarks; archived quotes do not imply a known forecast time."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.build_master_matches import (
    COMPETITION_NAMES,
    EXTRA_COMPETITIONS,
    RAW_DIR,
    parse_match_dates,
)
from src.data.team_names import load_team_name_mapping, normalize_team_names

MATCH_KEYS = ["date", "competition", "home_team", "away_team"]
MARKET_FEATURES = [f"market_prob_{label}" for label in range(3)]
CLOSING_FEATURES = [f"closing_prob_{label}" for label in range(3)]


def implied_probabilities(odds: pd.DataFrame) -> pd.DataFrame:
    values = odds.apply(pd.to_numeric, errors="coerce")
    valid = values.gt(1).all(axis=1) & np.isfinite(values).all(axis=1)
    # normalize by row total to strip the bookmaker margin
    inverse = 1 / values.where(valid)
    return inverse.div(inverse.sum(axis=1), axis=0)


def load_market_odds(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    frames = []
    for directory, competition in (COMPETITION_NAMES | EXTRA_COMPETITIONS).items():
        for path in sorted((raw_dir / directory).glob("*.csv")):
            raw = pd.read_csv(path)
            frame = pd.DataFrame(
                {
                    "date": parse_match_dates(raw.Date),
                    "competition": competition,
                    "home_team": raw.HomeTeam.str.strip(),
                    "away_team": raw.AwayTeam.str.strip(),
                }
            )
            for prefix, sources in [
                (
                    "market",
                    [
                        ("Avg", "average_preclosing"),
                        ("BbAv", "average_preclosing"),
                        ("B365", "bet365_preclosing"),
                    ],
                ),
                ("closing", [("AvgC", "average_closing")]),
            ]:
                probabilities = pd.DataFrame(np.nan, index=raw.index, columns=range(3))
                provider = pd.Series(pd.NA, index=raw.index, dtype="string")
                for source, name in sources:
                    # map provider H/D/A to project A/D/H
                    columns = [source + suffix for suffix in ("A", "D", "H")]
                    candidate = implied_probabilities(raw.reindex(columns=columns))
                    candidate.columns = range(3)
                    use_source = probabilities.isna().all(
                        axis=1
                    ) & candidate.notna().all(axis=1)
                    probabilities.loc[use_source] = candidate.loc[use_source]
                    provider.loc[use_source] = name
                for label in range(3):
                    frame[f"{prefix}_prob_{label}"] = probabilities[label]
                frame[f"{prefix}_source"] = provider
            frames.append(frame.dropna(subset=MATCH_KEYS))
    if not frames:
        return pd.DataFrame(columns=MATCH_KEYS + MARKET_FEATURES)
    market = pd.concat(frames, ignore_index=True)
    aliases = load_team_name_mapping(Path(__file__).with_name("team_name_mapping.csv"))
    for side in ("home", "away"):
        market[f"{side}_team"] = normalize_team_names(market[f"{side}_team"], aliases)
    if market.duplicated(MATCH_KEYS).any():
        raise ValueError("Duplicate fixtures in market data")
    return market


def attach_market_odds(matches: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    # one_to_one guards against duplicate fixtures
    return matches.merge(market, on=MATCH_KEYS, how="left", validate="one_to_one")


def attach_market_snapshots(
    fixtures: pd.DataFrame, quotes: pd.DataFrame
) -> pd.DataFrame:
    # live quotes need publish times; closing archives lack them
    required = MATCH_KEYS + [
        "snapshot_at",
        "source",
        "odds_away",
        "odds_draw",
        "odds_home",
    ]
    if set(required) - set(quotes) or "forecast_at" not in fixtures:
        raise ValueError(
            "Market snapshots require fixture keys, snapshot_at, source, odds and forecast_at"
        )
    data = quotes.copy()
    data["date"] = pd.to_datetime(data.date).dt.normalize()
    data["snapshot_at"] = pd.to_datetime(data.snapshot_at, utc=True, errors="raise")
    if (
        data[required].isna().any().any()
        or data.duplicated(MATCH_KEYS + ["snapshot_at"]).any()
    ):
        raise ValueError("Incomplete or ambiguous market snapshots")
    # group once instead of rescanning quotes per fixture
    groups = {key: group for key, group in data.groupby(MATCH_KEYS, sort=False)}
    values = pd.DataFrame(np.nan, index=fixtures.index, columns=MARKET_FEATURES)
    for index, fixture in fixtures.iterrows():
        # forecast_at is the hard info cutoff
        cutoff = pd.to_datetime(fixture.forecast_at, utc=True)
        kickoff = pd.to_datetime(fixture.get("kickoff_at", fixture.date), utc=True)
        if cutoff > kickoff or ("kickoff_at" in fixtures and cutoff == kickoff):
            raise ValueError("Market forecast must precede kickoff")
        key = tuple(fixture[column] for column in MATCH_KEYS)
        available = groups.get(key)
        if available is None:
            continue
        available = available[available.snapshot_at <= cutoff].sort_values(
            "snapshot_at"
        )
        if not available.empty:
            odds = available.tail(1)[["odds_away", "odds_draw", "odds_home"]]
            values.loc[index] = implied_probabilities(odds).iloc[0].to_numpy()
    return pd.concat([fixtures, values], axis=1)


def load_footiqo_closing_odds() -> pd.DataFrame:
    """Read UCL closing prices in the same column layout as domestic odds."""
    try:
        from src.data.footiqo_integration import load_footiqo_odds

        odds = load_footiqo_odds()

        odds["competition"] = "Champions League"

        return odds[
            [
                "date",
                "competition",
                "home_team",
                "away_team",
                "closing_prob_0",
                "closing_prob_1",
                "closing_prob_2",
                "closing_source",
            ]
        ].copy()
    except (ImportError, FileNotFoundError):
        return pd.DataFrame(
            columns=[
                "date",
                "competition",
                "home_team",
                "away_team",
                "closing_prob_0",
                "closing_prob_1",
                "closing_prob_2",
                "closing_source",
            ]
        )


def attach_footiqo_closing_odds(matches: pd.DataFrame) -> pd.DataFrame:
    """Fill UCL closing prices while keeping the other competitions' odds."""
    footiqo = load_footiqo_closing_odds()
    if footiqo.empty:
        for label in range(3):
            if f"closing_prob_{label}" not in matches.columns:
                matches[f"closing_prob_{label}"] = np.nan
        if "closing_source" not in matches.columns:
            matches["closing_source"] = pd.NA
        return matches

    merged = matches.copy()

    # join by date; sources share no reliable kickoff time
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.date
    footiqo_for_merge = footiqo.copy()
    footiqo_for_merge["date"] = pd.to_datetime(
        footiqo_for_merge["date"], errors="coerce"
    ).dt.date

    footiqo_cols_to_merge = [
        col for col in footiqo_for_merge.columns if col not in MATCH_KEYS
    ]
    merged_odds = merged.merge(
        footiqo_for_merge[MATCH_KEYS + footiqo_cols_to_merge],
        on=MATCH_KEYS,
        how="left",
        suffixes=("", "_footiqo"),
    )

    for label in range(3):
        probability_column = f"closing_prob_{label}"
        footiqo_column = f"{probability_column}_footiqo"
        if footiqo_column in merged_odds.columns:
            use_source = merged_odds[footiqo_column].notna()
            merged_odds.loc[use_source, probability_column] = merged_odds.loc[
                use_source, footiqo_column
            ]
            merged_odds.drop(columns=[footiqo_column], inplace=True)

    if "closing_source_footiqo" in merged_odds.columns:
        use_source = merged_odds["closing_source_footiqo"].notna()
        merged_odds.loc[use_source, "closing_source"] = merged_odds.loc[
            use_source, "closing_source_footiqo"
        ]
        merged_odds.drop(columns=["closing_source_footiqo"], inplace=True)

    if "date" in matches.columns:
        merged_odds["date"] = pd.to_datetime(merged_odds["date"])

    for label in range(3):
        if f"closing_prob_{label}" not in merged_odds.columns:
            merged_odds[f"closing_prob_{label}"] = np.nan
    if "closing_source" not in merged_odds.columns:
        merged_odds["closing_source"] = pd.NA

    return merged_odds
