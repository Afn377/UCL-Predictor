"""UCL closing-market experiments selected using earlier matches only."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.market import CLOSING_FEATURES
from src.evaluation.metrics import evaluate_probabilities_with_brier
from src.features.context import RICH_FEATURES
from src.features.model_dataset import FEATURE_COLUMNS, fill_missing_feature_values
from src.features.strength import COMPACT_FEATURES
from src.models.logistic import train_feature_model

ODDS_LOG_FEATURES = [f"log_odds_prob_{i}" for i in range(3)]
COMBINED_FEATURES = ODDS_LOG_FEATURES + COMPACT_FEATURES
FAMILIES = (
    "closing_odds",
    "closing_blend",
    "football_xg",
    "closing_logistic",
    "naive_base_rate",
)
TRAINING_YEARS = 2


def training_window(data, cutoff):
    """Use completed matches in the configured years immediately before cutoff."""
    end = pd.to_datetime(cutoff, utc=True).tz_localize(None).normalize()
    dates = pd.to_datetime(data.date, utc=True).dt.tz_localize(None)
    # lower bound inclusive, predicted match day excluded
    return data[
        (dates >= end - pd.DateOffset(years=TRAINING_YEARS)) & (dates < end)
    ].copy()


def market_probabilities(data):
    # closing odds already in the project's away/draw/home order
    probabilities = data[CLOSING_FEATURES].to_numpy(dtype=float)
    # partial rows would silently corrupt metrics
    if (
        not np.isfinite(probabilities).all()
        or (probabilities <= 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise ValueError(
            "Closing probabilities must be complete, positive and sum to one"
        )
    return probabilities


def market_design(data):
    # logs let a linear model combine market evidence additively
    result = data.copy()
    result[ODDS_LOG_FEATURES] = np.log(market_probabilities(data))
    return result


def eligible_data(data):
    result = (
        fill_missing_feature_values(data, FEATURE_COLUMNS)
        .dropna(subset=FEATURE_COLUMNS + ["result"])
        .copy()
    )
    result["date"] = pd.to_datetime(result.date)
    # pipeline relies on this exact class order
    if not result.result.isin([0, 1, 2]).all():
        raise ValueError("Results must be regulation-time away/draw/home labels")
    return result.sort_values(
        ["date", "competition", "home_team", "away_team"], kind="stable"
    )


def ucl_market_rows(data):
    # compare odds models on the same covered UCL fixtures
    result = data[data.competition.eq("Champions League")].dropna(
        subset=CLOSING_FEATURES
    )
    market_probabilities(result)
    return result


@dataclass
class ClosingModel:
    family: str
    estimator: object = None
    football_weight: float = 0.0
    base_rate: np.ndarray | None = None

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"Unknown model: {self.family}")

    def predict_proba(self, data):
        if self.family == "naive_base_rate":
            return np.tile(self.base_rate, (len(data), 1))
        if self.family == "football_xg":
            return self.estimator.predict_proba(data)
        market = market_probabilities(data)
        if self.family == "closing_odds":
            return market
        if self.family == "closing_blend":
            football_probabilities = self.estimator.predict_proba(data)
            market_weight = 1 - self.football_weight
            return (
                market_weight * market + self.football_weight * football_probabilities
            )
        return self.estimator.predict_proba(market_design(data))


def fit_candidate(train, specification, football=None):
    family = specification["family"]
    if family == "closing_odds":
        return ClosingModel(family)
    if family == "naive_base_rate":
        # minlength keeps all three classes in a short window
        labels = train["result"].to_numpy()
        outcome_counts = np.bincount(labels, minlength=3)
        return ClosingModel(family, base_rate=outcome_counts / len(labels))
    if family in ("football_xg", "closing_blend"):
        football = football or train_feature_model(train, RICH_FEATURES)
        return ClosingModel(family, football, specification.get("football_weight", 1.0))
    if family != "closing_logistic":
        raise ValueError(f"Unknown closing model: {family}")
    columns = COMBINED_FEATURES
    market_train = ucl_market_rows(train)
    if len(market_train) < 100 or market_train.result.nunique() != 3:
        raise ValueError(
            "Combined model needs 100 covered UCL training matches and all three outcomes"
        )
    estimator = train_feature_model(
        market_design(market_train),
        columns,
        regularization=specification["regularization"],
    )
    return ClosingModel(family, estimator)


def select_candidates(train):
    """Pick each model's settings using the later part of the training window."""
    market = ucl_market_rows(train)
    dates = np.sort(market.date.unique())
    if len(dates) < 20:
        raise ValueError("Selection requires at least 20 historical UCL match dates")
    # keep match days whole so a day can't predict itself
    boundary = dates[int(len(dates) * 0.75)]
    earlier = train[train.date < boundary]
    validation = market[market.date >= boundary]
    football = train_feature_model(earlier, RICH_FEATURES)
    candidate_settings = [
        {"family": "closing_odds"},
        {"family": "naive_base_rate"},
        {"family": "football_xg"},
    ]
    candidate_settings += [
        {"family": "closing_blend", "football_weight": football_weight}
        for football_weight in (0.1, 0.25, 0.5, 0.75)
    ]
    candidate_settings += [
        {"family": "closing_logistic", "regularization": regularization}
        for regularization in (0.01, 0.1, 1.0)
    ]
    scored = []
    for settings in candidate_settings:
        fitted = fit_candidate(earlier, settings, football)
        scored.append(
            {
                "specification": settings,
                **evaluate_probabilities_with_brier(
                    validation.result, fitted.predict_proba(validation)
                ),
            }
        )

    def rank(candidate):
        return candidate["accuracy"], -candidate["log_loss"]

    winners = {}
    for family in FAMILIES:
        family_candidates = [
            candidate
            for candidate in scored
            if candidate["specification"]["family"] == family
        ]
        best_candidate = max(family_candidates, key=rank)
        winners[family] = best_candidate["specification"]
    audit = {
        "training_end": str(earlier.date.max()),
        "validation_start": str(validation.date.min()),
        "validation_end": str(validation.date.max()),
        "validation_rows": len(validation),
        "selection_objective": "accuracy, then lower log loss",
        "candidates": scored,
        "selected": max(scored, key=rank)["specification"],
    }
    return winners, audit
