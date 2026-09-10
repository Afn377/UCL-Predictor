"""Online team strengths and tournament state, snapshotted before each match day."""

from collections import defaultdict

import numpy as np
import pandas as pd

STRUCTURAL_FEATURES = [
    "dynamic_elo_diff",
    "rating_uncertainty_sum",
    "league_rating_diff",
    "home_power_attack",
    "away_power_attack",
    "home_power_defense",
    "away_power_defense",
    "home_xg_power_attack",
    "away_xg_power_attack",
    "home_xg_power_defense",
    "away_xg_power_defense",
    "home_rest_known",
    "away_rest_known",
    "neutral_venue",
    "second_leg",
    "aggregate_lead",
    "knockout",
    "away_goals_rule",
    "new_format",
    "home_recent_matches",
    "away_recent_matches",
    "clubelo_diff",
]
COMPACT_FEATURES = STRUCTURAL_FEATURES + [
    "elo_diff",
    "rest_days_diff",
    "is_champions_league",
    "home_npxg_for_10",
    "away_npxg_for_10",
    "home_npxg_against_10",
    "away_npxg_against_10",
]


def add_strength_features(matches):
    # chronological order matters; every rating is a pre-match state
    data = matches.copy()
    data["date"] = pd.to_datetime(data.date)
    ratings, match_counts, leagues, last_match_dates = {}, defaultdict(int), {}, {}
    team_powers = defaultdict(lambda: np.zeros(4))
    league_ratings = defaultdict(float)
    recent = defaultdict(list)
    ties = {}
    output = {}
    for date, day in data.sort_values("date", kind="stable").groupby("date", sort=True):
        pending = []
        for index, row in day.iterrows():
            home_team, away_team = row.home_team, row.away_team
            competition = row.competition
            is_ucl = competition == "Champions League"
            stage = str(row.get("stage", ""))
            knockout = is_ucl and stage not in ("league_phase", "qualifying")
            # finals and 2020 single-leg rounds had no home advantage
            neutral = (
                bool(row.get("neutral_venue", False))
                if pd.notna(row.get("neutral_venue"))
                else False
            )
            neutral |= is_ucl and (
                stage == "final"
                or (stage in ("quarterfinal", "semifinal") and date.year == 2020)
            )
            tie_key = (
                str(row.season),
                competition,
                stage,
                tuple(sorted((home_team, away_team))),
            )
            first_leg = ties.get(tie_key) if knockout and not neutral else None
            is_second_leg = first_leg is not None and first_leg[0] < date
            aggregate = (
                (first_leg[2] if first_leg[1] == home_team else -first_leg[2])
                if is_second_leg
                else 0
            )
            # ratings decay toward 1500 during inactive spells
            decays = [
                np.exp(-max(0, (date - last_match_dates[team]).days - 30) / 365)
                if team in last_match_dates
                else 1
                for team in (home_team, away_team)
            ]
            strength = [
                (ratings.get(team, 1500) - 1500) * decay + 1500
                for team, decay in zip((home_team, away_team), decays)
            ]
            power = [
                team_powers[team] * decay
                for team, decay in zip((home_team, away_team), decays)
            ]
            home_league, away_league = leagues.get(home_team), leagues.get(away_team)
            league_diff = (
                league_ratings[home_league] - league_ratings[away_league]
                if home_league and away_league
                else 0
            )
            values = dict(zip(STRUCTURAL_FEATURES, [np.nan] * len(STRUCTURAL_FEATURES)))
            values.update(
                dynamic_elo_diff=strength[0] - strength[1],
                league_rating_diff=league_diff,
                rating_uncertainty_sum=sum(
                    1 / np.sqrt(match_counts[team] + 1)
                    for team in (home_team, away_team)
                ),
                neutral_venue=int(neutral),
                knockout=int(knockout),
                second_leg=int(is_second_leg),
                aggregate_lead=aggregate,
                away_goals_rule=int(knockout and str(row.season)[:4] < "2021"),
                new_format=int(is_ucl and str(row.season)[:4] >= "2024"),
            )
            for side, team, state in zip(
                ("home", "away"), (home_team, away_team), power
            ):
                for metric, value in zip(
                    (
                        "power_attack",
                        "power_defense",
                        "xg_power_attack",
                        "xg_power_defense",
                    ),
                    state,
                ):
                    values[f"{side}_{metric}"] = value
                gap = (
                    (date - last_match_dates[team]).days
                    if team in last_match_dates
                    else None
                )
                values[f"{side}_rest_known"] = int(gap is not None and 0 < gap <= 30)
                values[f"{side}_recent_matches"] = sum(
                    date - pd.Timedelta(days=30) <= match_date < date
                    for match_date in recent[team]
                )
            # snapshot now; this result hasn't touched any rating yet
            output[index] = values
            if pd.notna(row.home_goals) and pd.notna(row.away_goals):
                pending.append(
                    (
                        row,
                        strength,
                        power,
                        home_league,
                        away_league,
                        neutral,
                        tie_key,
                        is_second_leg,
                    )
                )
        for (
            row,
            strength,
            power,
            home_league,
            away_league,
            neutral,
            tie_key,
            is_second_leg,
        ) in pending:
            home_team, away_team = row.home_team, row.away_team
            home_goals, away_goals = float(row.home_goals), float(row.away_goals)
            expected = 1 / (
                1 + 10 ** ((strength[1] - strength[0] - (0 if neutral else 65)) / 400)
            )
            score = (
                1.0
                if home_goals > away_goals
                else 0.0
                if home_goals < away_goals
                else 0.5
            )
            # bigger wins move Elo more, with diminishing returns
            margin = 1 + np.log1p(max(0, abs(home_goals - away_goals) - 1))
            importance = 1.15 if row.competition == "Champions League" else 1
            delta = 20 * margin * importance * (score - expected)
            ratings[home_team], ratings[away_team] = (
                strength[0] + delta,
                strength[1] - delta,
            )
            if home_league and away_league and home_league != away_league:
                league_ratings[home_league] += 0.05 * delta
                league_ratings[away_league] -= 0.05 * delta
            home_advantage = 0 if neutral else np.log(1.25)
            for team_index, (team, opponent, goals, xg) in enumerate(
                [
                    (home_team, away_team, home_goals, row.get("home_npxg")),
                    (away_team, home_team, away_goals, row.get("away_npxg")),
                ]
            ):
                opponent_index = 1 - team_index
                offset = home_advantage if team_index == 0 else 0
                for attack, defense, observed in [(0, 1, goals), (2, 3, xg)]:
                    if pd.isna(observed):
                        continue
                    predicted = np.exp(
                        np.clip(
                            np.log(1.25)
                            + offset
                            + power[team_index][attack]
                            - power[opponent_index][defense],
                            -2,
                            2,
                        )
                    )
                    residual = np.clip(float(observed) - predicted, -3, 3)
                    team_powers[team][attack] = np.clip(
                        power[team_index][attack] + 0.04 * residual, -1.5, 1.5
                    )
                    team_powers[opponent][defense] = np.clip(
                        power[opponent_index][defense] - 0.04 * residual, -1.5, 1.5
                    )
                match_counts[team] += 1
                last_match_dates[team] = date
                recent[team] = [
                    match_date
                    for match_date in recent[team]
                    if match_date >= date - pd.Timedelta(days=30)
                ] + [date]
                if row.competition not in (
                    "Champions League",
                    "Europa League",
                    "Conference League",
                ) and pd.isna(row.get("stage")):
                    leagues[team] = row.competition
            # store first-leg margin so the return leg has the aggregate
            if (
                not is_second_leg
                and str(row.get("stage", ""))
                in ("round_of_16", "quarterfinal", "semifinal", "playoff")
                and not neutral
            ):
                ties[tie_key] = (date, home_team, home_goals - away_goals)
    return pd.concat(
        [data, pd.DataFrame.from_dict(output, orient="index").reindex(data.index)],
        axis=1,
    )


def attach_clubelo(data, history):
    """Only use ratings published strictly before the fixture date."""
    result = data.copy()
    history = history.copy()
    history["date"] = pd.to_datetime(history.date)
    if history.duplicated(["team", "date"]).any():
        raise ValueError("Duplicate ClubElo team/date")
    values = []
    groups = {
        name: group.sort_values("date") for name, group in history.groupby("team")
    }
    for row in result.itertuples():
        pair = []
        for team in (row.home_team, row.away_team):
            observed = groups.get(team)
            prior = (
                observed[observed.date < row.date]
                if observed is not None
                else pd.DataFrame()
            )
            # ratings older than 90 days are stale
            pair.append(
                float(prior.iloc[-1].elo)
                if not prior.empty and (row.date - prior.iloc[-1].date).days <= 90
                else np.nan
            )
        values.append(pair[0] - pair[1])
    result["clubelo_diff"] = values
    return result
