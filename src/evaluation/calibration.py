from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.temporal_evaluation import SEASON_SPLITS, make_temporal_split
from src.models.baselines import (
    CLASSES,
    ELO_FEATURES,
    FULL_FEATURES,
    naive_base_rate_probabilities,
    predict_logistic_probabilities,
    train_logistic_model,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "src" / "data" / "processed" / "model_dataset.csv"
DEFAULT_PREDICTIONS_OUTPUT = ROOT / "src" / "data" / "processed" / "temporal_predictions.csv"
DEFAULT_CALIBRATION_OUTPUT = ROOT / "src" / "data" / "processed" / "calibration_curve.csv"

PROBABILITY_COLUMNS = [f"prob_{label}" for label in CLASSES]


def probability_frame(probabilities: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(probabilities, columns=PROBABILITY_COLUMNS)


def predictions_for_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
    features: list[str] | None = None,
) -> pd.DataFrame:
    if model_name == "naive_base_rate":
        probabilities = naive_base_rate_probabilities(train, len(test)).to_numpy()
    else:
        if features is None:
            raise ValueError("features must be provided for logistic models")
        model = train_logistic_model(train, features)
        probabilities = predict_logistic_probabilities(model, test, features)

    metadata = test[["date", "season", "competition", "home_team", "away_team", "result"]].reset_index(drop=True)
    return pd.concat([metadata, probability_frame(probabilities)], axis=1).assign(model=model_name)


def collect_temporal_predictions(input_path: Path = DEFAULT_INPUT) -> pd.DataFrame:
    df = pd.read_csv(input_path, parse_dates=["date"])
    prediction_frames: list[pd.DataFrame] = []

    for split_name, test_start, test_end in SEASON_SPLITS:
        train, test = make_temporal_split(
            df,
            train_start="2015-01-01",
            test_start=test_start,
            test_end=test_end,
        )

        if train.empty or test.empty:
            continue

        prediction_frames.extend(
            [
                predictions_for_model(train, test, "naive_base_rate").assign(split=split_name),
                predictions_for_model(train, test, "elo_logistic", ELO_FEATURES).assign(split=split_name),
                predictions_for_model(train, test, "feature_logistic", FULL_FEATURES).assign(split=split_name),
            ]
        )

    if not prediction_frames:
        return pd.DataFrame()

    return pd.concat(prediction_frames, ignore_index=True)


def build_calibration_curve(
    predictions: pd.DataFrame,
    bins: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    bin_edges = np.linspace(0, 1, bins + 1)

    for model_name, model_predictions in predictions.groupby("model"):
        for label, probability_column in zip(CLASSES, PROBABILITY_COLUMNS, strict=True):
            probabilities = model_predictions[probability_column].to_numpy()
            actual = (model_predictions["result"].to_numpy() == label).astype(float)
            bin_ids = np.digitize(probabilities, bin_edges[1:-1], right=True)

            for bin_id in range(bins):
                mask = bin_ids == bin_id
                if not mask.any():
                    continue

                rows.append(
                    {
                        "model": model_name,
                        "class_label": label,
                        "bin_lower": bin_edges[bin_id],
                        "bin_upper": bin_edges[bin_id + 1],
                        "count": int(mask.sum()),
                        "mean_predicted_probability": float(probabilities[mask].mean()),
                        "observed_frequency": float(actual[mask].mean()),
                    }
                )

    return pd.DataFrame(rows)


def write_calibration_outputs(
    input_path: Path = DEFAULT_INPUT,
    predictions_output: Path = DEFAULT_PREDICTIONS_OUTPUT,
    calibration_output: Path = DEFAULT_CALIBRATION_OUTPUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = collect_temporal_predictions(input_path=input_path)
    calibration = build_calibration_curve(predictions)

    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_output, index=False)
    calibration.to_csv(calibration_output, index=False)
    return predictions, calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Build temporal model predictions and calibration curves.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS_OUTPUT)
    parser.add_argument("--calibration-output", type=Path, default=DEFAULT_CALIBRATION_OUTPUT)
    args = parser.parse_args()

    predictions, calibration = write_calibration_outputs(
        input_path=args.input,
        predictions_output=args.predictions_output,
        calibration_output=args.calibration_output,
    )
    print(f"Wrote {len(predictions)} temporal predictions.")
    print(f"Wrote {len(calibration)} calibration bins.")


if __name__ == "__main__":
    main()
