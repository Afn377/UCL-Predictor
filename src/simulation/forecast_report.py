from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "src" / "data" / "processed" / "ucl_league_phase_simulation_2025_26.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "ucl_forecast_2025_26.md"

REQUIRED_COLUMNS = [
    "projected_rank",
    "team",
    "average_rank",
    "average_points",
    "top_8_probability",
    "top_24_probability",
    "elimination_probability",
]


def _format_probability(value: float) -> str:
    return f"{value * 100:.1f}%"


def _markdown_table(rows: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in rows.iterrows()]
    return "\n".join([header, divider, *body])


def _display_rows(forecast: pd.DataFrame) -> pd.DataFrame:
    rows = forecast.copy()
    rows["average_rank"] = rows["average_rank"].map(lambda value: f"{value:.1f}")
    rows["average_points"] = rows["average_points"].map(lambda value: f"{value:.1f}")
    rows["top_8_probability"] = rows["top_8_probability"].map(_format_probability)
    rows["top_24_probability"] = rows["top_24_probability"].map(_format_probability)
    rows["elimination_probability"] = rows["elimination_probability"].map(_format_probability)
    return rows


def validate_forecast(forecast: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in forecast.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"forecast is missing required columns: {missing}")


def build_forecast_report(
    forecast: pd.DataFrame,
    season: str = "2025_26",
    cutoff_date: str | None = None,
    simulations: int | None = None,
) -> str:
    validate_forecast(forecast)
    forecast = forecast.sort_values("projected_rank").reset_index(drop=True)
    display = _display_rows(forecast)

    title = f"# Champions League Forecast: {season}"
    metadata = []
    if cutoff_date is not None:
        metadata.append(f"- Cutoff date: {cutoff_date}")
    if simulations is not None:
        metadata.append(f"- Simulations: {simulations:,}")
    metadata.append("- Qualification format: ranks 1-8 advance directly, ranks 9-24 enter the playoff round.")

    top_8 = display.head(8)
    playoff = display.iloc[8:24]
    elimination_risk = display[display["elimination_probability"] != "0.0%"].tail(10)
    top_8_bubble = display.iloc[(forecast["top_8_probability"] - 0.5).abs().sort_values().index].head(8)
    top_24_bubble = display.iloc[(forecast["top_24_probability"] - 0.5).abs().sort_values().index].head(8)

    columns = [
        "projected_rank",
        "team",
        "average_points",
        "top_8_probability",
        "top_24_probability",
        "elimination_probability",
    ]

    sections = [
        title,
        "\n".join(metadata),
        "## Projected Direct Qualifiers",
        _markdown_table(top_8[columns], columns),
        "## Projected Playoff Places",
        _markdown_table(playoff[columns], columns),
        "## Top-8 Bubble",
        _markdown_table(top_8_bubble[columns], columns),
        "## Top-24 Bubble",
        _markdown_table(top_24_bubble[columns], columns),
    ]

    if not elimination_risk.empty:
        sections.extend(
            [
                "## Elimination Risk",
                _markdown_table(elimination_risk[columns], columns),
            ]
        )

    return "\n\n".join(sections) + "\n"


def write_forecast_report(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    season: str = "2025_26",
    cutoff_date: str | None = None,
    simulations: int | None = None,
) -> str:
    forecast = pd.read_csv(input_path)
    report = build_forecast_report(
        forecast,
        season=season,
        cutoff_date=cutoff_date,
        simulations=simulations,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a readable Champions League forecast report.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--season", default="2025_26")
    parser.add_argument("--cutoff-date", default=None)
    parser.add_argument("--simulations", type=int, default=None)
    args = parser.parse_args()

    report = write_forecast_report(
        input_path=args.input,
        output_path=args.output,
        season=args.season,
        cutoff_date=args.cutoff_date,
        simulations=args.simulations,
    )
    print(report)


if __name__ == "__main__":
    main()
