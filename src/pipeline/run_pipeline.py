"""Run the reproducible five-model UCL forecasting pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

from src.data.build_master_matches import write_master_matches
from src.features.context import write_context_features
from src.features.elo import write_matches_with_elo
from src.features.form import write_matches_with_form_features
from src.features.model_dataset import write_model_dataset


@dataclass(frozen=True)
class PipelineConfig:
    download_ucl: bool = False
    download_xg: bool = False
    download_domestic: bool = False
    download_europe: bool = False
    closing_experiment: bool = True


@dataclass(frozen=True)
class PipelineStep:
    name: str
    run: Callable[[], object]


def build_pipeline_steps(config: PipelineConfig) -> list[PipelineStep]:
    # downloads are opt-in; they need network and are slow
    steps = []
    if config.download_domestic:
        from src.data.download_domestic import download_domestic

        steps.append(PipelineStep("download domestic data", download_domestic))
    if config.download_europe:
        from src.data.download_europe import download_europe

        steps.append(PipelineStep("download European data", download_europe))
    if config.download_ucl:
        from src.data.download_ucl import main as download_ucl

        steps.append(PipelineStep("download UCL data", download_ucl))
    if config.download_xg:
        from src.data.download_understat import download_understat

        steps.append(PipelineStep("download Understat xG", download_understat))
    steps.extend(
        [
            PipelineStep("build master matches", write_master_matches),
            PipelineStep("add Elo features", write_matches_with_elo),
            PipelineStep("add rolling form features", write_matches_with_form_features),
            PipelineStep("build model dataset", write_model_dataset),
            PipelineStep("add context and market features", write_context_features),
        ]
    )
    if config.closing_experiment:
        from src.evaluation.closing import evaluate_closing

        steps.append(
            PipelineStep("evaluate the five retained models", evaluate_closing)
        )
    return steps


def run_pipeline(config: PipelineConfig) -> list[str]:
    completed = []
    for step in build_pipeline_steps(config):
        # print first so long stages show up immediately
        print(f"Running: {step.name}", flush=True)
        step.run()
        completed.append(step.name)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("ucl", "xg", "domestic", "europe"):
        parser.add_argument(f"--download-{flag}", action="store_true")
    parser.add_argument("--skip-closing-experiment", action="store_true")
    args = parser.parse_args()
    completed = run_pipeline(
        PipelineConfig(
            download_ucl=args.download_ucl,
            download_xg=args.download_xg,
            download_domestic=args.download_domestic,
            download_europe=args.download_europe,
            closing_experiment=not args.skip_closing_experiment,
        )
    )
    print(f"Finished {len(completed)} pipeline steps.")


if __name__ == "__main__":
    main()
