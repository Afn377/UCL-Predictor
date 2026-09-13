import pandas as pd

from src.models.closing import training_window


def test_training_window_includes_start_excludes_cutoff_and_future():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2022-06-30", "2022-07-01", "2025-06-30", "2025-07-01", "2026-01-01"]
            )
        }
    )
    result = training_window(data, "2025-07-01")
    assert result.index.tolist() == [1, 2]
    assert len(data) == 5
