from src.pipeline.run_pipeline import PipelineConfig, build_pipeline_steps


def test_build_pipeline_steps_excludes_download_by_default() -> None:
    steps = build_pipeline_steps(PipelineConfig())

    names = [step.name for step in steps]

    assert names[0] == "build master matches"
    assert "download UCL data" not in names
    assert names[-1] == "write forecast report"


def test_build_pipeline_steps_includes_download_when_requested() -> None:
    steps = build_pipeline_steps(PipelineConfig(download_ucl=True))

    assert steps[0].name == "download UCL data"


def test_build_pipeline_steps_keeps_forecast_steps_at_end() -> None:
    steps = build_pipeline_steps(PipelineConfig(season="2024_25", cutoff_date="2024-11-01"))

    names = [step.name for step in steps]

    assert names[-3:] == [
        "write UCL league-phase table",
        "simulate UCL league phase",
        "write forecast report",
    ]
