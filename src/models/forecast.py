"""Forecast fixtures with the five retained models and historical confidence policies."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from src.data.market import CLOSING_FEATURES, MARKET_FEATURES, attach_market_snapshots
from src.evaluation.confidence import apply_confidence_policy, fit_confidence_policy
from src.evaluation.metrics import prediction_frame
from src.features.context import DEFAULT_OUTPUT, DEFAULT_XG, ROOT, add_context_features
from src.models.closing import (
    FAMILIES,
    eligible_data,
    fit_candidate,
    select_candidates,
    training_window,
)
from src.models.predict import build_prediction_features


def prepare_forecast_features(
    fixtures,
    as_of,
    history_path=ROOT / "src/data/processed/matches_with_elo.csv",
    xg_path=DEFAULT_XG,
):
    # exact timestamp for quote filtering and leakage checks
    cutoff = pd.to_datetime(as_of, utc=True)
    day_cutoff = cutoff.tz_localize(None).normalize()
    columns = [
        "date",
        "season",
        "competition",
        "stage",
        "home_team",
        "away_team",
        "kickoff_at",
    ]
    fixtures = fixtures[[column for column in columns if column in fixtures]].copy()
    if fixtures.empty:
        raise ValueError("At least one fixture is required")
    fixtures["date"] = pd.to_datetime(fixtures.date).dt.normalize()
    if (fixtures.date < day_cutoff).any():
        raise ValueError("Fixtures must be on or after the forecast date")
    # same-day forecasts need kickoff_at to prove pre-match timing
    if "kickoff_at" in fixtures:
        if (pd.to_datetime(fixtures.kickoff_at, utc=True) <= cutoff).any():
            raise ValueError("Cannot issue a pre-match forecast after kickoff")
    elif cutoff.tz_localize(None) > fixtures.date.min():
        raise ValueError(
            "Same-day forecasts require kickoff_at; date-only forecasts use midnight"
        )
    # history stops before forecast day, so no same-day form
    fixtures["forecast_at"] = cutoff
    history = pd.read_csv(history_path, parse_dates=["date"])
    history = history[history.date < day_cutoff].copy()
    rows = []
    for fixture in fixtures.to_dict("records"):
        base = build_prediction_features(
            fixture["home_team"],
            fixture["away_team"],
            history,
            match_date=fixture["date"],
            competition=fixture["competition"],
        )
        rows.append(
            {
                **fixture,
                **base.iloc[0].to_dict(),
                "home_goals": float("nan"),
                "away_goals": float("nan"),
                "result": float("nan"),
            }
        )
    future = pd.DataFrame(rows)
    combined = pd.concat([history, future], ignore_index=True)
    xg = pd.read_csv(xg_path, parse_dates=["date"])
    features = (
        add_context_features(combined, xg[xg.date < day_cutoff])
        .tail(len(future))
        .reset_index(drop=True)
    )
    return features


def forecast_fixtures(
    fixtures,
    as_of,
    model_name="football_xg",
    context_path=DEFAULT_OUTPUT,
    history_path=ROOT / "src/data/processed/matches_with_elo.csv",
    xg_path=DEFAULT_XG,
    predictions_path=ROOT / "src/data/processed/closing_experiment/predictions.csv",
    quotes=None,
):
    cutoff = pd.to_datetime(as_of, utc=True)
    day_cutoff = cutoff.tz_localize(None).normalize()
    context = pd.read_csv(context_path, parse_dates=["date"], low_memory=False)
    train = training_window(eligible_data(context), day_cutoff)
    # only fitted examples are limited; Elo/form use all history
    if model_name not in FAMILIES:
        raise ValueError(f"Model must be one of {FAMILIES}")
    if train.empty:
        raise ValueError("No eligible training matches in the configured window")
    features = prepare_forecast_features(fixtures, as_of, history_path, xg_path)
    with threadpool_limits(limits=1):
        if model_name in ("closing_odds", "closing_blend", "closing_logistic"):
            if quotes is None:
                raise ValueError("Market-assisted forecasts require timestamped quotes")
            features = attach_market_snapshots(features, quotes)
            if features[MARKET_FEATURES].isna().any().any():
                raise ValueError(
                    "Every market-assisted fixture needs a valid quote available at forecast time"
                )
            # training uses closing_* names, live snapshots arrive as market_*
            features[CLOSING_FEATURES] = features[MARKET_FEATURES].to_numpy()
        if model_name in ("closing_blend", "closing_logistic"):
            specifications, _ = select_candidates(train)
            model = fit_candidate(train, specifications[model_name])
        else:
            model = fit_candidate(train, {"family": model_name})
        probabilities = model.predict_proba(features)
    output = prediction_frame(
        features, probabilities, model_name, "forecast", train.date.max()
    )
    # calibrate on historical predictions, not this fixture
    previous = pd.read_csv(predictions_path, parse_dates=["date"], low_memory=False)
    previous = previous[previous.date < day_cutoff]
    frames = []
    for competition, group in output.groupby("competition", sort=False):
        policy = fit_confidence_policy(previous, model_name, competition)
        frames.append(apply_confidence_policy(group, policy))
    result = pd.concat(frames, ignore_index=True).drop(columns="result")
    result["forecast_at"] = cutoff.isoformat()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fixtures",
        type=Path,
        help="CSV with date, season, competition, stage, home_team, away_team",
    )
    parser.add_argument(
        "--as-of", required=True, help="Forecast timestamp, e.g. 2026-09-07T00:00:00Z"
    )
    parser.add_argument("--model", choices=FAMILIES, default="football_xg")
    parser.add_argument("--quotes", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "src/data/processed/fixture_forecasts.csv"
    )
    args = parser.parse_args()
    result = forecast_fixtures(
        pd.read_csv(args.fixtures),
        args.as_of,
        args.model,
        quotes=pd.read_csv(args.quotes) if args.quotes else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
