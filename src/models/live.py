"""Freeze a UCL model, record forecasts before kickoff, and score them later."""

import hashlib
import json
import sqlite3
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data.market import MATCH_KEYS
from src.data.team_names import load_team_name_mapping, normalize_team_names
from src.evaluation.closing import OUTPUT_DIR
from src.evaluation.confidence import PROBABILITY_COLUMNS, fit_confidence_policy
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


def normalize_names(frame):
    result = frame.copy()
    aliases = load_team_name_mapping(ROOT / "src/data/team_name_mapping.csv")
    for col in ("home_team", "away_team", "team"):
        # quotes have a generic team column, fixtures only home/away
        if col in result:
            result[col] = normalize_team_names(result[col], aliases)
    result["date"] = pd.to_datetime(result.date).dt.normalize()
    return result


def live_inputs(fixtures, quotes, now, manifest):
    # normalize before key checks so aliases can't make duplicates
    fixtures, quotes = normalize_names(fixtures), normalize_names(quotes)
    if fixtures.empty or not fixtures.competition.eq("Champions League").all():
        raise ValueError("Provide at least one UCL fixture")
    if fixtures.duplicated(MATCH_KEYS).any() or fixtures[MATCH_KEYS].isna().any().any():
        raise ValueError("Duplicate or incomplete fixtures")
    if "kickoff_at" not in fixtures:
        raise ValueError("Live forecasts require explicit kickoff_at timestamps")
    kickoff = pd.to_datetime(fixtures.kickoff_at, utc=True, errors="raise")
    minutes = (kickoff - now).dt.total_seconds() / 60
    if (
        kickoff.isna().any()
        or not ((minutes > 0) & (minutes <= manifest["forecast_window_minutes"])).all()
    ):
        raise ValueError(
            "Issue forecasts strictly before kickoff and within the frozen forecast window"
        )
    if not (fixtures.date == kickoff.dt.tz_localize(None).dt.normalize()).all():
        raise ValueError("Live fixture date must match the UTC kickoff date")
    if (
        pd.Timestamp(manifest["created_at"]) > now
        or (fixtures.date <= pd.Timestamp(manifest["train_end"])).any()
    ):
        raise ValueError("Forecast must follow model creation and training history")
    quotes["snapshot_at"] = pd.to_datetime(quotes.snapshot_at, utc=True, errors="raise")
    required = MATCH_KEYS + [
        "snapshot_at",
        "source",
        "odds_away",
        "odds_draw",
        "odds_home",
    ]
    if (
        quotes[required].isna().any().any()
        or quotes.duplicated(MATCH_KEYS + ["snapshot_at"]).any()
    ):
        raise ValueError("Incomplete or ambiguous quote snapshots")
    recent = quotes[
        (quotes.snapshot_at <= now)
        & (
            quotes.snapshot_at
            >= now - pd.Timedelta(minutes=manifest["maximum_quote_age_minutes"])
        )
    ]
    recent = recent.sort_values("snapshot_at").drop_duplicates(MATCH_KEYS, keep="last")
    joined = fixtures[MATCH_KEYS].merge(
        recent, on=MATCH_KEYS, how="left", validate="one_to_one"
    )
    if joined.snapshot_at.isna().any():
        raise ValueError(
            "Each fixture needs a recent quote published before this forecast"
        )
    return fixtures, joined


def record_predictions(predictions, manifest, provenance, ledger=LEDGER):
    """Save each forecast once so we can't rewrite a pick after seeing the result."""
    recorded_at = pd.Timestamp.now(tz="UTC")
    kickoff = pd.to_datetime(predictions.kickoff_at, utc=True, errors="raise")
    if predictions.empty or kickoff.isna().any() or (kickoff <= recorded_at).any():
        raise ValueError("All forecasts must be recorded before kickoff")
    probabilities = predictions[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise ValueError("Cannot record invalid probabilities")
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(ledger) as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS predictions (
            model_id TEXT NOT NULL, date TEXT NOT NULL, competition TEXT NOT NULL,
            home_team TEXT NOT NULL, away_team TEXT NOT NULL, recorded_at TEXT NOT NULL,
            payload TEXT NOT NULL, PRIMARY KEY (model_id, date, competition, home_team, away_team))""")
        # triggers keep the table append-only for other scripts
        for operation in ("UPDATE", "DELETE"):
            connection.execute(f"""CREATE TRIGGER IF NOT EXISTS forbid_{operation.lower()}
                BEFORE {operation} ON predictions BEGIN
                SELECT RAISE(ABORT, 'Forecast records are append-only'); END""")
        for row in json.loads(predictions.to_json(orient="records", date_format="iso")):
            payload = {"forecast": row, "model": manifest, "provenance": provenance}
            connection.execute(
                "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    manifest["model_id"],
                    row["date"],
                    row["competition"],
                    row["home_team"],
                    row["away_team"],
                    recorded_at.isoformat(),
                    json.dumps(payload, allow_nan=False),
                ),
            )
