"""Compare UCL odds and football signals on identical chronological test rows."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from src.data.market import MATCH_KEYS
from src.evaluation.confidence import (
    PROBABILITY_COLUMNS,
    evaluate_confidence_policies,
    summarize_selective,
)
from src.evaluation.metrics import (
    SEASON_SPLITS,
    evaluate_probabilities_with_brier,
    prediction_frame,
)
from src.features.context import DEFAULT_OUTPUT, RICH_FEATURES, ROOT
from src.models.closing import (
    TRAINING_YEARS,
    eligible_data,
    fit_candidate,
    select_candidates,
    training_window,
    ucl_market_rows,
)
from src.models.logistic import train_feature_model

OUTPUT_DIR = ROOT / "src/data/processed/closing_experiment"


def disagreement_report(predictions):
    keys = MATCH_KEYS + ["split"]
    odds = predictions[predictions.model.eq("closing_odds")].copy()
    football = predictions[predictions.model.eq("football_xg")].copy()
    joined = odds.merge(
        football, on=keys, suffixes=("_odds", "_football"), validate="one_to_one"
    )
    if len(joined) != len(odds) or len(joined) != len(football):
        raise ValueError("Disagreement comparison requires identical fixtures")
    if not joined.result_odds.equals(joined.result_football):
        raise ValueError("Disagreement labels differ")
    result = joined[keys].copy()
    result["result"] = joined.result_odds
    for source in ("odds", "football"):
        result[f"{source}_pick"] = (
            joined[[f"{column}_{source}" for column in PROBABILITY_COLUMNS]]
            .to_numpy()
            .argmax(axis=1)
        )
        result[f"{source}_correct"] = result[f"{source}_pick"] == result.result
    result["disagree"] = result.odds_pick != result.football_pick
    summaries = []
    for split, group in [("all", result), *list(result.groupby("split"))]:
        disagreements = group[group.disagree]
        summaries.append(
            {
                "split": split,
                "matches": len(group),
                "disagreements": len(disagreements),
                "football_only_correct": int(
                    (group.football_correct & ~group.odds_correct).sum()
                ),
                "odds_only_correct": int(
                    (group.odds_correct & ~group.football_correct).sum()
                ),
                "both_correct": int(
                    (group.odds_correct & group.football_correct).sum()
                ),
                "neither_correct": int(
                    (~group.odds_correct & ~group.football_correct).sum()
                ),
            }
        )
    return result, pd.DataFrame(summaries)


def evaluate_closing(input_path=DEFAULT_OUTPUT, output_dir=OUTPUT_DIR):
    input_path, output_dir = Path(input_path), Path(output_dir)
    data = eligible_data(pd.read_csv(input_path, low_memory=False))
    if data.duplicated(MATCH_KEYS).any():
        raise ValueError("Duplicate context fixtures")
    frames, audits, coverage = [], [], []
    # single-thread native pools keep small logistic fits deterministic
    with threadpool_limits(limits=1):
        for split, start, end in SEASON_SPLITS:
            train = training_window(data, start)
            all_test = data[
                (data.date >= start)
                & (data.date < end)
                & data.competition.eq("Champions League")
            ]
            # market rows are the common test set for every model
            test = ucl_market_rows(all_test)
            if test.empty:
                continue
            print(
                f"Closing experiment: {split}, {len(test)} common fixtures", flush=True
            )
            # tune inside the training window, then refit on all training rows
            model_settings, audit = select_candidates(train)
            audits.append(
                {
                    "split": split,
                    "outer_train_start": str(train.date.min()),
                    "outer_train_end": str(train.date.max()),
                    "outer_test_start": str(test.date.min()),
                    **audit,
                }
            )
            football = train_feature_model(train, RICH_FEATURES)
            # share the football fit across pure and blend candidates
            models = {
                name: fit_candidate(train, settings, football)
                for name, settings in model_settings.items()
            }
            for name, model in models.items():
                frames.append(
                    prediction_frame(
                        test, model.predict_proba(test), name, split, train.date.max()
                    )
                )
            coverage.append(
                {
                    "split": split,
                    "eligible_ucl": len(all_test),
                    "odds_covered": len(test),
                    "coverage": len(test) / len(all_test),
                }
            )
    if not frames:
        raise ValueError("No historical UCL odds overlap with evaluation dates")
    predictions = pd.concat(frames, ignore_index=True)
    summaries, seasons, confusion = [], [], []
    for model, group in predictions.groupby("model"):
        probabilities = group[PROBABILITY_COLUMNS].to_numpy()
        summaries.append(
            {
                "model": model,
                "matches": len(group),
                "correct": int((probabilities.argmax(axis=1) == group.result).sum()),
                **evaluate_probabilities_with_brier(group.result, probabilities),
            }
        )
        for split, subset in group.groupby("split"):
            seasons.append(
                {
                    "model": model,
                    "split": split,
                    "matches": len(subset),
                    **evaluate_probabilities_with_brier(
                        subset.result, subset[PROBABILITY_COLUMNS].to_numpy()
                    ),
                }
            )
        for actual in range(3):
            for predicted in range(3):
                confusion.append(
                    {
                        "model": model,
                        "actual": actual,
                        "predicted": predicted,
                        "matches": int(
                            (
                                (group.result == actual)
                                & (probabilities.argmax(axis=1) == predicted)
                            ).sum()
                        ),
                    }
                )
    summary = pd.DataFrame(summaries).sort_values(
        ["accuracy", "log_loss"], ascending=[False, True]
    )
    disagreements, disagreement_summary = disagreement_report(predictions)
    selective, selective_results, policies = evaluate_confidence_policies(predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    # separate files so downstream analysis loads only what it needs
    outputs = {
        "predictions": predictions,
        "summary": summary,
        "seasons": pd.DataFrame(seasons),
        "confusion": pd.DataFrame(confusion),
        "coverage": pd.DataFrame(coverage),
        "disagreements": disagreements,
        "disagreement_summary": disagreement_summary,
        "selective_predictions": selective,
        "selective_results": selective_results,
        "selective_summary": summarize_selective(selective_results),
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    # hash the input so results can't be tied to a different dataset
    manifest = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "context_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "training_years": TRAINING_YEARS,
        "forecast_timing": "closing-odds retrospective benchmark; exact quote times unavailable",
        "holdout_status": "Previously inspected seasons; development evaluation only",
        "model_selection": audits,
        "confidence_policies": policies,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False)
    )
    report = [
        "# UCL Closing Odds and Football Features",
        "",
        manifest["holdout_status"],
        "",
        "All models below use the same odds-covered UCL fixtures. Closing prices are not day-ahead inputs.",
        "",
        "| Model | Correct / matches | Accuracy | Log loss | RPS |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples():
        report.append(
            f"| {row.model} | {row.correct}/{row.matches} | {row.accuracy:.2%} | {row.log_loss:.4f} | {row.rps:.4f} |"
        )
    overall = disagreement_summary.iloc[0]
    report += [
        "",
        f"Odds and football disagree on {overall.disagreements} fixtures. Football alone is correct on "
        f"{overall.football_only_correct}; odds alone on {overall.odds_only_correct}.",
        "",
        "The highest row in this retrospective table is not an unbiased estimate of choosing that model in advance.",
        "",
        "Selective accuracy and coverage are saved separately in selective_summary.csv.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(summary.to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    evaluate_closing(args.input, args.output_dir)
    # guard against missing or mismatched rows skewing the comparison
