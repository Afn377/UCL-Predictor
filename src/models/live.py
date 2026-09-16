"""Freeze a UCL model, record forecasts before kickoff, and score them later."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from src.evaluation.closing import OUTPUT_DIR
from src.evaluation.confidence import fit_confidence_policy
from src.features.context import DEFAULT_OUTPUT, ROOT
from src.models.closing import (
    TRAINING_YEARS,
    eligible_data,
    fit_candidate,
    select_candidates,
    training_window,
)


REGISTRY = ROOT / "artifacts/closing_models"
LEDGER = ROOT / "src/data/processed/live_forecasts.sqlite"
HISTORY = ROOT / "src/data/processed/matches_with_elo.csv"
FROZEN_SOURCES = [
    "src/models/closing.py",
    "src/models/logistic.py",
    "src/models/forecast.py",
    "src/models/predict.py",
    "src/features/context.py",
    "src/features/strength.py",
    "src/features/form.py",
    "src/evaluation/confidence.py",
    "src/data/market.py",
    "src/data/team_names.py",
    "src/data/team_name_mapping.csv",
    "src/data/understat_team_mapping.csv",
    "src/features/model_dataset.py",
    "src/models/live.py",
]


def sha256(path):
    # hashes tie the frozen model to the exact data and source
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_model(
    context_path=DEFAULT_OUTPUT, experiment_dir=OUTPUT_DIR, registry=REGISTRY
):
    context_path, experiment_dir, registry = (
        Path(context_path),
        Path(experiment_dir),
        Path(registry),
    )
    experiment = json.loads((experiment_dir / "manifest.json").read_text())
    if experiment.get("training_years") != TRAINING_YEARS:
        raise ValueError("Training window changed; rerun the closing experiment first")
    if sha256(context_path) != experiment["context_sha256"]:
        raise ValueError(
            "Context changed since evaluation; rerun the closing experiment first"
        )
    now = pd.Timestamp.now(tz="UTC")
    train = eligible_data(pd.read_csv(context_path, low_memory=False))
    current_date = now.tz_localize(None).normalize()
    train = training_window(train, current_date)
    # confidence comes from out-of-period predictions, not training scores
    previous = pd.read_csv(experiment_dir / "predictions.csv", parse_dates=["date"])
    previous = previous[previous.date <= train.date.max()]
    with threadpool_limits(limits=1):
        _, audit = select_candidates(train)
        fitted = fit_candidate(train, audit["selected"])
        policy = fit_confidence_policy(previous, audit["selected"]["family"])
    # timestamp + random suffix keeps dirs unique across reruns
    model_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    directory = registry / model_id
    directory.mkdir(parents=True, exist_ok=False)
    bundle = directory / "model.joblib"
    # keep estimator and policy together so they can't drift
    joblib.dump({"model": fitted, "policy": policy}, bundle)
    manifest = {
        "model_id": model_id,
        "created_at": now.isoformat(),
        "train_end": str(train.date.max().date()),
        "train_rows": len(train),
        "train_start": str(train.date.min().date()),
        "training_years": TRAINING_YEARS,
        "context_sha256": sha256(context_path),
        "bundle_sha256": sha256(bundle),
        "source_sha256": {path: sha256(ROOT / path) for path in FROZEN_SOURCES},
        "packages": {
            name: version(name)
            for name in ("numpy", "pandas", "scikit-learn", "joblib")
        },
        "experiment_manifest_sha256": sha256(experiment_dir / "manifest.json"),
        "selection": audit,
        "policy": policy.to_dict(),
        "forecast_window_minutes": 60,
        "maximum_quote_age_minutes": 15,
        "timing": "Near kickoff live quotes; historical closing prices are only a proxy for this timing",
        "evaluation_status": "Frozen for prospective evaluation; historical periods were previously inspected",
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False)
    )
    print(f"Frozen {audit['selected']} at {directory}")
    return directory


def load_frozen(directory):
    # joblib unpickles: check the hash first
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if sha256(directory / "model.joblib") != manifest["bundle_sha256"]:
        raise ValueError("Frozen model checksum mismatch")
    if any(
        sha256(ROOT / path) != expected
        for path, expected in manifest["source_sha256"].items()
    ):
        raise ValueError(
            "Feature or prediction code changed after freezing; evaluate and freeze a new version"
        )
    if any(
        version(name) != expected for name, expected in manifest["packages"].items()
    ):
        raise ValueError(
            "Runtime packages changed after freezing; evaluate and freeze a new version"
        )
    return joblib.load(directory / "model.joblib"), manifest
