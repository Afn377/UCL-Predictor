"""Calibrate probabilities and freeze selection rules before the test period."""

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax

PROBABILITY_COLUMNS = [f"prob_{label}" for label in range(3)]


def wilson_interval(correct: int, count: int, z: float = 1.96) -> tuple[float, float]:
    if count == 0:
        return np.nan, np.nan
    # wilson beats the normal approximation on small samples
    accuracy = correct / count
    denominator = 1 + z * z / count
    centre = (accuracy + z * z / (2 * count)) / denominator
    half_width = (
        z
        * sqrt(accuracy * (1 - accuracy) / count + z * z / (4 * count * count))
        / denominator
    )
    return centre - half_width, centre + half_width


def temperature_scale(probabilities, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    return softmax(np.log(np.clip(probabilities, 1e-12, 1)) / temperature, axis=1)


def fit_temperature(probabilities, result):
    # short histories are too noisy to recalibrate
    probabilities, result = np.asarray(probabilities), np.asarray(result, dtype=int)
    if len(result) < 100:
        return 1.0

    def calibration_loss(temperature):
        calibrated = temperature_scale(probabilities, temperature)
        observed_probabilities = calibrated[np.arange(len(result)), result]
        return -np.log(observed_probabilities).mean()

    fit = minimize_scalar(
        calibration_loss,
        bounds=(0.5, 3.0),
        method="bounded",
    )
    if not fit.success:
        raise RuntimeError("Probability calibration failed")
    return float(fit.x)


def choose_threshold(probabilities, result, target_accuracy=0.70, min_picks=50):
    probabilities, result = np.asarray(probabilities), np.asarray(result)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == result
    # lucky picks aren't enough; the lower bound must clear the target
    for threshold in np.arange(0.50, 0.901, 0.025):
        selected = confidence >= threshold
        count = int(selected.sum())
        if count < min_picks:
            continue
        lower, _ = wilson_interval(int(correct[selected].sum()), count)
        if lower >= target_accuracy:
            return float(threshold), count, float(correct[selected].mean()), lower
    return None, 0, None, None


@dataclass(frozen=True)
class ConfidencePolicy:
    model: str
    competition: str
    trained_through: str | None
    calibration_end: str | None
    validation_start: str | None
    temperature: float = 1.0
    threshold: float | None = None
    validation_picks: int = 0
    validation_accuracy: float | None = None
    validation_lower_95: float | None = None
    target_accuracy: float = 0.70
    status: str = "insufficient_history"

    def to_dict(self):
        # manifests need plain JSON, not a dataclass
        return asdict(self)


def fit_confidence_policy(
    history, model, competition="Champions League", target_accuracy=0.70, min_picks=50
):
    data = history[
        (history.model == model) & (history.competition == competition)
    ].copy()
    data["date"] = pd.to_datetime(data.date)
    if data.empty:
        return ConfidencePolicy(
            model, competition, None, None, None, target_accuracy=target_accuracy
        )
    # need at least 3 splits: one to calibrate, two to validate
    splits = data.groupby("split").date.min().sort_values().index.tolist()
    if len(splits) < 3:
        return ConfidencePolicy(
            model,
            competition,
            str(data.date.max().date()),
            None,
            None,
            target_accuracy=target_accuracy,
        )
    validation = data[data.split.isin(splits[-2:])]
    calibration = data[data.date < validation.date.min()]
    temperature = fit_temperature(
        calibration[PROBABILITY_COLUMNS].to_numpy(), calibration.result.to_numpy()
    )
    calibrated_probabilities = temperature_scale(
        validation[PROBABILITY_COLUMNS].to_numpy(), temperature
    )
    threshold, picks, accuracy, lower = choose_threshold(
        calibrated_probabilities, validation.result, target_accuracy, min_picks
    )
    return ConfidencePolicy(
        model=model,
        competition=competition,
        trained_through=str(data.date.max().date()),
        calibration_end=str(calibration.date.max().date()),
        validation_start=str(validation.date.min().date()),
        temperature=temperature,
        threshold=threshold,
        validation_picks=picks,
        validation_accuracy=accuracy,
        validation_lower_95=lower,
        target_accuracy=target_accuracy,
        status="ready" if threshold is not None else "no_qualifying_threshold",
    )


def apply_confidence_policy(predictions, policy):
    # copy before calibrating; callers reuse the table
    data = predictions.copy()
    if (
        not data.model.eq(policy.model).all()
        or not data.competition.eq(policy.competition).all()
    ):
        raise ValueError("Policy model and competition must match the predictions")
    if (
        policy.trained_through
        and (pd.to_datetime(data.date) <= pd.Timestamp(policy.trained_through)).any()
    ):
        raise ValueError("Policy cannot be applied to dates used to fit it")
    probabilities = temperature_scale(
        data[PROBABILITY_COLUMNS].to_numpy(), policy.temperature
    )
    data[PROBABILITY_COLUMNS] = probabilities
    data["predicted_result"] = probabilities.argmax(axis=1)
    data["confidence"] = probabilities.max(axis=1)
    data["selected"] = (
        False if policy.threshold is None else data.confidence >= policy.threshold
    )
    data["policy_threshold"] = policy.threshold
    data["policy_status"] = policy.status
    data["policy_trained_through"] = policy.trained_through
    return data


def evaluate_confidence_policies(predictions):
    # each split gets a policy built only from its past
    selected_frames, policies, results = [], [], []
    ucl = predictions[predictions.competition.eq("Champions League")]
    for model, model_data in ucl.groupby("model", sort=False):
        for split, test in model_data.groupby("split", sort=True):
            history = model_data[
                pd.to_datetime(model_data.date) < pd.to_datetime(test.date).min()
            ]
            policy = fit_confidence_policy(history, model)
            output = apply_confidence_policy(test, policy)
            selected = output[output.selected]
            correct = int((selected.predicted_result == selected.result).sum())
            low, high = wilson_interval(correct, len(selected))
            selected_frames.append(output)
            policies.append({"test_split": split, **policy.to_dict()})
            results.append(
                {
                    "model": model,
                    "split": split,
                    "eligible_matches": len(test),
                    "picks": len(selected),
                    "correct": correct,
                    "coverage": len(selected) / len(test),
                    "accuracy": correct / len(selected) if len(selected) else np.nan,
                    "lower_95": low,
                    "upper_95": high,
                    "status": policy.status,
                }
            )
    return (
        pd.concat(selected_frames, ignore_index=True),
        pd.DataFrame(results),
        policies,
    )


def summarize_selective(results):
    rows = []
    for model, group in results.groupby("model"):
        picks, correct = int(group.picks.sum()), int(group.correct.sum())
        eligible = int(group.eligible_matches.sum())
        low, high = wilson_interval(correct, picks)
        ready = group[group.status.eq("ready")]
        rows.append(
            {
                "model": model,
                "eligible_matches": eligible,
                "policy_ready_matches": int(ready.eligible_matches.sum()),
                "picks": picks,
                "correct": correct,
                "coverage": picks / eligible,
                "accuracy": correct / picks if picks else np.nan,
                "lower_95": low,
                "upper_95": high,
                "test_splits": len(group),
            }
        )
    return pd.DataFrame(rows)
    # never apply a policy before its last fitting date
