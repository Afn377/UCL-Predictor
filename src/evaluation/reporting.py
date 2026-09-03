from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "src" / "data" / "processed" / "temporal_evaluation.csv"
DEFAULT_OUTPUT = ROOT / "src" / "data" / "processed" / "temporal_evaluation_summary.csv"


def summarize_temporal_evaluation(results: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["model", "log_loss", "accuracy", "brier_score"]
    missing_columns = [column for column in required_columns if column not in results.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"results is missing required columns: {missing}")

    summary = (
        results.groupby("model", as_index=False)
        .agg(
            mean_log_loss=("log_loss", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_brier_score=("brier_score", "mean"),
            splits=("split", "nunique"),
            total_test_rows=("test_rows", "sum"),
        )
        .sort_values(["mean_log_loss", "mean_brier_score"], kind="mergesort")
        .reset_index(drop=True)
    )
    summary["rank_by_log_loss"] = range(1, len(summary) + 1)
    return summary


def write_temporal_evaluation_summary(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    results = pd.read_csv(input_path)
    summary = summarize_temporal_evaluation(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize walk-forward temporal evaluation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary = write_temporal_evaluation_summary(input_path=args.input, output_path=args.output)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
