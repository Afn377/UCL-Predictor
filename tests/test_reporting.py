import pandas as pd

from src.evaluation.reporting import summarize_temporal_evaluation


def test_summarize_temporal_evaluation_ranks_by_log_loss() -> None:
    results = pd.DataFrame(
        [
            {"split": "2020_21", "model": "b", "test_rows": 10, "log_loss": 0.9, "accuracy": 0.5, "brier_score": 0.2},
            {"split": "2021_22", "model": "b", "test_rows": 20, "log_loss": 1.1, "accuracy": 0.4, "brier_score": 0.3},
            {"split": "2020_21", "model": "a", "test_rows": 10, "log_loss": 0.8, "accuracy": 0.6, "brier_score": 0.1},
        ]
    )

    summary = summarize_temporal_evaluation(results)

    assert summary.loc[0, "model"] == "a"
    assert summary.loc[0, "rank_by_log_loss"] == 1
    assert summary.loc[summary["model"] == "b", "total_test_rows"].iloc[0] == 30


def test_summarize_temporal_evaluation_requires_metrics() -> None:
    try:
        summarize_temporal_evaluation(pd.DataFrame([{"model": "a"}]))
    except ValueError as exc:
        assert "missing required columns" in str(exc)
    else:
        raise AssertionError("Expected missing-column validation to fail")
