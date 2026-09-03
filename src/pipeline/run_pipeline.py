from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.data.build_master_matches import write_master_matches
from src.evaluation.calibration import write_calibration_outputs
from src.evaluation.reporting import write_temporal_evaluation_summary
from src.evaluation.temporal_evaluation import write_temporal_evaluation
from src.features.elo import write_matches_with_elo
from src.features.form import write_matches_with_form_features
from src.features.model_dataset import write_model_dataset
from src.models.poisson import write_poisson_predictions
from src.simulation.forecast_report import write_forecast_report
from src.simulation.league_phase_simulator import write_league_phase_simulation
from src.simulation.ucl_state import write_league_phase_table


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = ROOT / "src" / "data" / "processed"
DEFAULT_REPORTS_DIR = ROOT / "reports"


@dataclass(frozen=True)
class PipelineConfig:
    season: str = "2025_26"
    cutoff_date: str = "2025-11-06"
    simulations: int = 500
    random_seed: int = 7
    download_ucl: bool = False


@dataclass(frozen=True)
class PipelineStep:
    name: str
    run: Callable[[], object]


def build_pipeline_steps(config: PipelineConfig) -> list[PipelineStep]:
    simulation_output = DEFAULT_PROCESSED_DIR / f"ucl_league_phase_simulation_{config.season}.csv"
    report_output = DEFAULT_REPORTS_DIR / f"ucl_forecast_{config.season}.md"

    steps: list[PipelineStep] = []
    if config.download_ucl:
        from src.data.download_ucl import main as download_ucl_data

        steps.append(PipelineStep("download UCL data", download_ucl_data))

    steps.extend(
        [
            PipelineStep("build master matches", write_master_matches),
            PipelineStep("add Elo features", write_matches_with_elo),
            PipelineStep("add rolling form features", write_matches_with_form_features),
            PipelineStep("build model dataset", write_model_dataset),
            PipelineStep("run temporal evaluation", write_temporal_evaluation),
            PipelineStep("summarize temporal evaluation", write_temporal_evaluation_summary),
            PipelineStep("write calibration outputs", write_calibration_outputs),
            PipelineStep("write Poisson predictions", write_poisson_predictions),
            PipelineStep(
                "write UCL league-phase table",
                lambda: write_league_phase_table(season=config.season),
            ),
            PipelineStep(
                "simulate UCL league phase",
                lambda: write_league_phase_simulation(
                    season=config.season,
                    cutoff_date=config.cutoff_date,
                    n_simulations=config.simulations,
                    random_seed=config.random_seed,
                    output_path=simulation_output,
                ),
            ),
            PipelineStep(
                "write forecast report",
                lambda: write_forecast_report(
                    input_path=simulation_output,
                    output_path=report_output,
                    season=config.season,
                    cutoff_date=config.cutoff_date,
                    simulations=config.simulations,
                ),
            ),
        ]
    )
    return steps


def run_pipeline(config: PipelineConfig) -> list[str]:
    completed_steps = []
    for step in build_pipeline_steps(config):
        print(f"Running: {step.name}", flush=True)
        step.run()
        completed_steps.append(step.name)
    return completed_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the UCL forecasting pipeline end to end.")
    parser.add_argument("--season", default="2025_26")
    parser.add_argument("--cutoff-date", default="2025-11-06")
    parser.add_argument("--simulations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--download-ucl", action="store_true")
    args = parser.parse_args()

    config = PipelineConfig(
        season=args.season,
        cutoff_date=args.cutoff_date,
        simulations=args.simulations,
        random_seed=args.seed,
        download_ucl=args.download_ucl,
    )
    completed_steps = run_pipeline(config)
    print(f"Finished {len(completed_steps)} pipeline steps.", flush=True)


if __name__ == "__main__":
    main()
