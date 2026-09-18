"""Freeze a UCL model, record forecasts before kickoff, and score them later."""

import argparse
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

from src.data.market import (
    CLOSING_FEATURES,
    MARKET_FEATURES,
    MATCH_KEYS,
    attach_market_snapshots,
)
from src.data.team_names import load_team_name_mapping, normalize_team_names
from src.evaluation.closing import OUTPUT_DIR
from src.evaluation.confidence import (
    PROBABILITY_COLUMNS,
    apply_confidence_policy,
    fit_confidence_policy,
    wilson_interval,
)
from src.evaluation.metrics import evaluate_probabilities_with_brier
from src.features.context import DEFAULT_OUTPUT, DEFAULT_XG, ROOT
from src.models.closing import (
    TRAINING_YEARS,
    eligible_data,
    fit_candidate,
    select_candidates,
    training_window,
)
from src.models.forecast import prepare_forecast_features

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


def forecast_live(fixtures_path, quotes_path, model_dir, output, ledger=LEDGER):
    now = pd.Timestamp.now(tz="UTC")
    bundle, manifest = load_frozen(model_dir)
    fixtures, quotes = live_inputs(
        pd.read_csv(fixtures_path), pd.read_csv(quotes_path), now, manifest
    )
    features = prepare_forecast_features(fixtures, now)
    features = attach_market_snapshots(features, quotes)
    features[CLOSING_FEATURES] = features[MARKET_FEATURES].to_numpy()
    with threadpool_limits(limits=1):
        probabilities = bundle["model"].predict_proba(features)
    result = fixtures.copy().reset_index(drop=True)
    result[PROBABILITY_COLUMNS] = probabilities
    result["model"] = bundle["model"].family
    result = apply_confidence_policy(result, bundle["policy"])
    result["forecast_at"] = now.isoformat()
    result["model_id"] = manifest["model_id"]
    result["quote_snapshot_at"] = quotes.snapshot_at.astype(str).to_numpy()
    result["quote_source"] = quotes.source.to_numpy()
    finished = pd.Timestamp.now(tz="UTC")
    if (
        (
            finished - pd.to_datetime(result.quote_snapshot_at, utc=True)
        ).dt.total_seconds()
        > manifest["maximum_quote_age_minutes"] * 60
    ).any():
        raise ValueError(
            "Quotes expired during feature preparation; refresh quotes and retry"
        )
    provenance = {
        "fixtures_sha256": sha256(fixtures_path),
        "quotes_sha256": sha256(quotes_path),
        "history_sha256": sha256(HISTORY),
        "xg_sha256": sha256(DEFAULT_XG),
        "features_sha256": hashlib.sha256(
            features.to_csv(index=False).encode()
        ).hexdigest(),
        "lineups_sha256": None,
        "lineups_used_by_model": False,
    }
    record_predictions(result, manifest, provenance, ledger)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    result[result.selected].to_csv(
        output.with_name(output.stem + "_selected.csv"), index=False
    )
    display_forecasts(result)
    return result


def display_forecasts(result):
    view = result[
        ["home_team", "away_team", "prob_2", "prob_1", "prob_0", "selected"]
    ].rename(
        columns={
            "prob_2": "Home",
            "prob_1": "Draw",
            "prob_0": "Away",
            "selected": "High confidence",
        }
    )
    for col in ("Home", "Draw", "Away"):
        view[col] = view[col].map(lambda p: f"{p:.1%}")
    print(view.to_string(index=False))
    print(
        f"\nHigh-confidence picks: {int(result.selected.sum())}/{len(result)} "
        f"({result.selected.mean():.1%} coverage). Full probabilities are retained for every fixture."
    )


def score_ledger(
    results_path, ledger=LEDGER, output_dir=ROOT / "src/data/processed/prospective"
):
    # read-only so scoring can't alter the records
    ledger = Path(ledger)
    if not ledger.exists():
        raise ValueError("No prospective forecasts recorded yet")
    with sqlite3.connect(f"file:{ledger}?mode=ro", uri=True) as connection:
        records = connection.execute(
            "SELECT payload, recorded_at FROM predictions ORDER BY recorded_at"
        ).fetchall()
    rows = []
    for payload, recorded_at in records:
        row = json.loads(payload)["forecast"]
        rows.append({**row, "recorded_at": recorded_at})
    if not rows:
        raise ValueError("No prospective forecasts recorded yet")
    forecasts = normalize_names(pd.DataFrame(rows))
    if (
        pd.to_datetime(forecasts.recorded_at, utc=True)
        >= pd.to_datetime(forecasts.kickoff_at, utc=True)
    ).any():
        raise ValueError("Ledger contains forecasts recorded after kickoff")
    results = normalize_names(pd.read_csv(results_path))
    if results.duplicated(MATCH_KEYS).any():
        raise ValueError("Duplicate result fixtures")
    for side in ("home", "away"):
        column = f"{side}_goals"
        results[column] = pd.to_numeric(results[column], errors="raise")
        values = results[column].dropna()
        if not (
            np.isfinite(values).all()
            and (values >= 0).all()
            and (values % 1 == 0).all()
        ):
            raise ValueError("Regulation-time goals must be nonnegative integers")
    joined = forecasts.merge(
        results[MATCH_KEYS + ["home_goals", "away_goals"]],
        on=MATCH_KEYS,
        how="left",
        validate="many_to_one",
    )
    joined["settled"] = joined[["home_goals", "away_goals"]].notna().all(axis=1)
    # result file may be wrong; leave matches pending until kickoff
    joined["settled"] &= pd.to_datetime(joined.kickoff_at, utc=True) < pd.Timestamp.now(
        tz="UTC"
    )
    joined["result"] = np.where(
        joined.settled, np.sign(joined.home_goals - joined.away_goals) + 1, np.nan
    )
    summaries = []
    for model_id, group in joined.groupby("model_id"):
        settled = group[group.settled]
        for scope, subset in [
            ("all", settled),
            ("selected", settled[settled.selected]),
        ]:
            metrics = (
                evaluate_probabilities_with_brier(
                    subset.result, subset[PROBABILITY_COLUMNS].to_numpy()
                )
                if len(subset)
                else {}
            )
            correct = int((subset.predicted_result == subset.result).sum())
            lower, upper = wilson_interval(correct, len(subset))
            summaries.append(
                {
                    "model_id": model_id,
                    "scope": scope,
                    "recorded": len(group),
                    "settled": len(settled),
                    "scored": len(subset),
                    "correct": correct,
                    "coverage": len(subset) / len(settled) if len(settled) else np.nan,
                    "pending": len(group) - len(settled),
                    "lower_95": lower,
                    "upper_95": upper,
                    **metrics,
                }
            )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_dir / "scored_forecasts.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--context", type=Path, default=DEFAULT_OUTPUT)
    freeze.add_argument("--experiment-dir", type=Path, default=OUTPUT_DIR)
    freeze.add_argument("--registry", type=Path, default=REGISTRY)
    forecast = commands.add_parser("forecast")
    forecast.add_argument("fixtures", type=Path)
    forecast.add_argument("--quotes", type=Path, required=True)
    forecast.add_argument("--model-dir", type=Path, required=True)
    forecast.add_argument("--ledger", type=Path, default=LEDGER)
    forecast.add_argument(
        "--output", type=Path, default=ROOT / "src/data/processed/live_forecasts.csv"
    )
    score = commands.add_parser("score")
    score.add_argument(
        "results",
        type=Path,
        help="Final regulation-time results with fixture keys and goals",
    )
    score.add_argument("--ledger", type=Path, default=LEDGER)
    score.add_argument(
        "--output-dir", type=Path, default=ROOT / "src/data/processed/prospective"
    )
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_model(args.context, args.experiment_dir, args.registry)
    elif args.command == "forecast":
        forecast_live(
            args.fixtures, args.quotes, args.model_dir, args.output, args.ledger
        )
    elif args.command == "score":
        score_ledger(args.results, args.ledger, args.output_dir)


if __name__ == "__main__":
    main()
