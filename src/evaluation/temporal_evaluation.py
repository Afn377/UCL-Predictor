from pathlib import Path 
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import label_binarize

from src.models.baselines import (
    CLASSES,
    ELO_FEATURES,
    FULL_FEATURES,
    evaluate_probabilities,
    naive_base_rate_probabilities,
    train_logistic_model,
    predict_logistic_probabilities,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "src" / "data" / "processed" / "model_dataset.csv"
DEFAULT_OUTPUT = ROOT / "src" / "data" / "processed" / "temporal_evaluation.csv"

SEASON_SPLITS = [
    ("2019_20", "2019-07-01", "2020-07-01"),
    ("2020_21", "2020-07-01", "2021-07-01"),
    ("2021_22", "2021-07-01", "2022-07-01"),
    ("2022_23", "2022-07-01", "2023-07-01"),
    ("2023_24", "2023-07-01", "2024-07-01"),
    ("2024_25", "2024-07-01", "2025-07-01"),
    ("2025_26", "2025-07-01", "2026-07-01"),
]


def brier_score(y_true, probs):
    # simply find the mean squared error through one hot encoding the true values and comparing to the 
    # predicted probabilities

    # this helps to check whether the model's predicted probabilities are actually good and not just whether
    # its top pick was correct
    y_true_one_hot = label_binarize(y_true, classes=CLASSES)
    return ((y_true_one_hot - probs) ** 2).mean()


def evaluate_probabilities_with_brier(y_true, probabilities):
    result = evaluate_probabilities(y_true, probabilities)
    result["brier_score"] = brier_score(y_true, probabilities)
    return result


def make_temporal_split(df, train_start, test_start, test_end):
    train = df[(df["date"] >= train_start) & (df["date"] < test_start)].copy()
    test = df[(df["date"] >= test_start) & (df["date"] < test_end)].copy()
    return train, test


def evaluate_models_for_split(train, test, split_name):
    rows = []

    naive_probs = naive_base_rate_probabilities(train, len(test)).to_numpy()
    rows.append(
        {
            "split": split_name,
            "model": "naive_base_rate",
            "train_rows": len(train),
            "test_rows": len(test),
            **evaluate_probabilities_with_brier(test["result"], naive_probs),
        }
    )

    elo_model = train_logistic_model(train, ELO_FEATURES)
    elo_probs = predict_logistic_probabilities(elo_model, test, ELO_FEATURES)
    rows.append(
        {
            "split": split_name,
            "model": "elo_logistic",
            "train_rows": len(train),
            "test_rows": len(test),
            **evaluate_probabilities_with_brier(test["result"], elo_probs),
        }
    )

    feature_model = train_logistic_model(train, FULL_FEATURES)
    feature_probs = predict_logistic_probabilities(feature_model, test, FULL_FEATURES)
    rows.append(
        {
            "split": split_name,
            "model": "feature_logistic",
            "train_rows": len(train),
            "test_rows": len(test),
            **evaluate_probabilities_with_brier(test["result"], feature_probs),
        }
    )

    return rows


def run_temporal_evaluation(input_path=DEFAULT_INPUT):
    df = pd.read_csv(input_path, parse_dates=["date"])

    all_rows = []

    for split_name, test_start, test_end in SEASON_SPLITS:
        train_start = "2015-01-01"
        train, test = make_temporal_split(df, train_start, test_start, test_end)

        if train.empty or test.empty:
            continue

        all_rows.extend(evaluate_models_for_split(train, test, split_name))

    return pd.DataFrame(all_rows)

def write_temporal_evaluation(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT):
    results = run_temporal_evaluation(input_path=input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


def main():
    results = write_temporal_evaluation()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()