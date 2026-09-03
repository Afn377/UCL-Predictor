import pandas as pd
import pytest

from src.simulation.forecast_report import build_forecast_report, validate_forecast


def make_forecast() -> pd.DataFrame:
    rows = []
    for rank in range(1, 37):
        top_8_probability = max(0, min(1, (10 - rank) / 8))
        top_24_probability = max(0, min(1, (28 - rank) / 8))
        rows.append(
            {
                "projected_rank": rank,
                "team": f"Team {rank}",
                "average_rank": float(rank),
                "average_points": 30.0 - rank / 2,
                "top_8_probability": top_8_probability,
                "top_24_probability": top_24_probability,
                "elimination_probability": 1 - top_24_probability,
            }
        )
    return pd.DataFrame(rows)


def test_build_forecast_report_contains_main_sections() -> None:
    report = build_forecast_report(
        make_forecast(),
        season="2025_26",
        cutoff_date="2025-11-06",
        simulations=500,
    )

    assert "# Champions League Forecast: 2025_26" in report
    assert "Cutoff date: 2025-11-06" in report
    assert "Simulations: 500" in report
    assert "## Projected Direct Qualifiers" in report
    assert "## Projected Playoff Places" in report
    assert "## Top-8 Bubble" in report
    assert "## Top-24 Bubble" in report


def test_build_forecast_report_formats_probabilities() -> None:
    report = build_forecast_report(make_forecast())

    assert "100.0%" in report
    assert "| Team 1 |" in report


def test_validate_forecast_requires_expected_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate_forecast(pd.DataFrame([{"team": "Arsenal"}]))
