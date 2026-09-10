"""Aggregate pre-match squad snapshots without using retrospective starting XIs."""

import numpy as np
import pandas as pd

LINEUP_METRICS = ["expected_xi_quality", "missing_player_quality", "rotation"]
LINEUP_FEATURES = [
    f"{side}_{metric}" for side in ("home", "away") for metric in LINEUP_METRICS
]
SNAPSHOT_KEYS = ["date", "competition", "home_team", "away_team", "team", "snapshot_at"]
PLAYER_COLUMNS = [
    "player_id",
    "quality",
    "rating_as_of",
    "expected_start",
    "usual_start",
    "available",
]


def aggregate_lineups(players: pd.DataFrame) -> pd.DataFrame:
    missing = set(SNAPSHOT_KEYS + PLAYER_COLUMNS) - set(players)
    if missing:
        raise ValueError(f"Missing lineup columns: {sorted(missing)}")
    data = players.copy()
    for column in ("date", "snapshot_at", "rating_as_of"):
        # utc aligns ratings, snapshots, forecasts, kickoff
        data[column] = pd.to_datetime(data[column], utc=True, errors="raise")
    if data[SNAPSHOT_KEYS + PLAYER_COLUMNS].isna().any().any():
        raise ValueError(
            "Lineup snapshots must have complete timestamps, players and ratings"
        )
    if (data.rating_as_of > data.snapshot_at).any():
        raise ValueError("Player ratings cannot come from after the snapshot")
    if data.duplicated(SNAPSHOT_KEYS + ["player_id"]).any():
        raise ValueError("Duplicate player in lineup snapshot")
    for col in ["quality", "expected_start", "usual_start", "available"]:
        # flags/probs are 0..1, quality is 0..100
        data[col] = pd.to_numeric(data[col], errors="raise")
        ceiling = 100 if col == "quality" else 1
        if not data[col].between(0, ceiling).all():
            raise ValueError(f"Invalid {col} range")
    if (~data.available.isin([0, 1])).any() or (
        (data.available == 0) & (data.expected_start > 0)
    ).any():
        raise ValueError("Unavailable players cannot have positive expected_start")
    rows = []
    for key, squad in data.groupby(SNAPSHOT_KEYS, dropna=False, sort=False):
        if key[4] not in key[2:4]:
            raise ValueError("Lineup team is not part of the fixture")
        if not np.isclose(squad.expected_start.sum(), 11) or not np.isclose(
            squad.usual_start.sum(), 11
        ):
            raise ValueError(
                "Expected and usual starting probabilities must each sum to 11"
            )
        rows.append(
            {
                **dict(zip(SNAPSHOT_KEYS, key)),
                "expected_xi_quality": (squad.quality * squad.expected_start).sum()
                / 11,
                "missing_player_quality": (
                    squad.quality * squad.usual_start * (1 - squad.available)
                ).sum()
                / 11,
                "rotation": np.abs(squad.expected_start - squad.usual_start).sum() / 2,
            }
        )
    return pd.DataFrame(rows, columns=SNAPSHOT_KEYS + LINEUP_METRICS)


def attach_lineups(
    matches: pd.DataFrame, players: pd.DataFrame | None = None
) -> pd.DataFrame:
    features = pd.DataFrame(np.nan, index=matches.index, columns=LINEUP_FEATURES)
    if players is not None and not players.empty:
        snapshots = aggregate_lineups(players)
        groups = {
            key: group
            for key, group in snapshots.groupby(SNAPSHOT_KEYS[:5], sort=False)
        }
        for index, match in matches.iterrows():
            # forecast_at is the info cutoff when present
            date = (
                pd.Timestamp(match.date).tz_localize("UTC")
                if pd.Timestamp(match.date).tzinfo is None
                else pd.Timestamp(match.date).tz_convert("UTC")
            )
            # date-only rows use midnight, so same-day news is excluded
            cutoff = pd.Timestamp(match.get("forecast_at", date))
            cutoff = (
                cutoff.tz_localize("UTC")
                if cutoff.tzinfo is None
                else cutoff.tz_convert("UTC")
            )
            if "kickoff_at" in matches and pd.notna(match.kickoff_at):
                if cutoff >= pd.to_datetime(match.kickoff_at, utc=True):
                    raise ValueError("forecast_at must precede kickoff_at")
            elif cutoff > date:
                raise ValueError("Same-day lineup use requires an explicit kickoff_at")
            for side in ("home", "away"):
                key = (
                    date,
                    match.competition,
                    match.home_team,
                    match.away_team,
                    match[f"{side}_team"],
                )
                available = groups.get(key)
                if available is None:
                    continue
                available = available[available.snapshot_at <= cutoff].sort_values(
                    "snapshot_at"
                )
                if not available.empty:
                    for metric in LINEUP_METRICS:
                        features.loc[index, f"{side}_{metric}"] = available.iloc[-1][
                            metric
                        ]
    return pd.concat([matches, features], axis=1)
