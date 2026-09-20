# UCL Match Forecasting

Predict regulation-time away win, draw, and home win probabilities for Champions
League matches. The project builds historical features, compares five models on
identical UCL fixtures, and records forecasts for later prospective scoring.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Raw datasets and processed outputs are not tracked in git (`src/data/raw/` and
`src/data/processed/` are ignored), so a fresh clone must ingest its own data. Frozen
models under `artifacts/` are ignored too; create one with
`python -m src.models.live freeze` before live forecasting. On
first run, fetch the Footiqo odds and the optional ClubElo ratings, then let the
pipeline download the rest and build everything:

```sh
python -m src.data.download_footiqo
python -m src.data.download_clubelo
python -m src.pipeline.run_pipeline --download-domestic --download-europe \
  --download-ucl --download-xg
```

Later runs can use the local data and simply run
`python -m src.pipeline.run_pipeline`. Add `--skip-closing-experiment` to stop before
model evaluation. Each downloader also has `--help`. Network access and upstream
availability are required for downloads, and upstream changes can shift results
slightly from the numbers reported below.

## Data and Features

Ingestion writes raw data under `src/data/raw`: domestic match results/statistics,
European competition results, Understat xG, ClubElo snapshots, and Footiqo historical
odds. Download modules document source URLs; team mapping files reconcile names.
Availability and coverage vary by season and competition.

The pipeline then writes processed stages under `src/data/processed`:
`matches.csv`, `matches_with_elo.csv`, `matches_with_features.csv`,
`model_dataset.csv`, and `matches_with_context.csv`.
Features include Elo, prior form and goals, shot proxies, rest/congestion,
venue performance, xG/non-penalty xG and opposition-adjusted strength.
Coverage and match-audit files describe missing external data.

## Models

| Model | Inputs and fitting |
| --- | --- |
| naive_base_rate | Outcome frequencies across training matches; constant probabilities |
| closing_odds | Normalized bookmaker implied probabilities; no fitting |
| football_xg | Standardized logistic regression on rich football features, including xG |
| closing_blend | Weighted combination of football_xg and closing odds |
| closing_logistic | Logistic regression on log market probabilities and compact football features |

All fitted models use `TRAINING_YEARS` in `src/models/closing.py`, currently two
years. Each historical test season uses only earlier training rows in that
window. Blend weights and combined-model regularization are selected on a later
validation portion of the training window. Features may summarize older history;
Elo is not reset at the window boundary.

## Results

Development-only evaluation on 966 odds-covered UCL fixtures (previously inspected
seasons, not an untouched holdout — see [Limits](#limits)). Classes are ordered
0=away win, 1=draw, 2=home win.

### Full-coverage leaderboard

| Model | Correct / matches | Accuracy | Log loss | RPS |
| --- | ---: | ---: | ---: | ---: |
| closing_odds | 601/966 | 62.22% | 0.8781 | 0.1821 |
| closing_blend | 597/966 | 61.80% | 0.8793 | 0.1816 |
| football_xg | 578/966 | 59.83% | 0.9024 | 0.1875 |
| closing_logistic | 567/966 | 58.70% | 0.9216 | 0.1910 |
| naive_base_rate | 458/966 | 47.41% | 1.0450 | 0.2384 |

Odds and football disagree on 140 fixtures: football alone is correct on 43,
odds alone on 66. The top row here is not an unbiased estimate of choosing that
model in advance.

### Selective (high-confidence) accuracy

Frozen confidence policies (`src/evaluation/confidence.py`) abstain unless a
pick's Wilson lower-95% bound clears 70% on validation data. Coverage is the
share of eligible matches picked.

| Model | Eligible | Picks | Correct | Coverage | Accuracy | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| closing_odds | 966 | 176 | 139 | 18.2% | 78.98% | [72.4%, 84.3%] |
| closing_blend | 966 | 231 | 180 | 23.9% | 77.92% | [72.1%, 82.8%] |
| football_xg | 966 | 162 | 120 | 16.8% | 74.07% | [66.8%, 80.2%] |
| closing_logistic | 966 | 165 | 119 | 17.1% | 72.12% | [64.8%, 78.4%] |
| naive_base_rate | 966 | 0 | 0 | 0.0% | — | — |

## Evaluation and Use

```sh
python -m src.evaluation.closing
python -m src.models.forecast --help
python -m src.models.live freeze
```

Running `src.evaluation.closing` regenerates the tables above plus per-season
metrics, probabilities and confusion counts under
`src/data/processed/closing_experiment/`. Metrics include accuracy, log loss,
RPS and Brier score.

See [Live forecasting](docs/live_forecasting.md) for fixture/quote schemas and the
freeze, forecast, and score workflow. Ad hoc forecasts do not enter the live
ledger. The live command validates timing and records predictions before kickoff.

## Limits

The current historical comparison covers 966 odds-covered UCL fixtures. These
seasons have been inspected during development, so results are not an untouched
holdout. Closing odds are near-kickoff information. High-confidence accuracy
applies only to the reported selected subset, not all fixtures. No 70% all-match
accuracy is established. This version forecasts individual matches; tournament
simulation was removed. Live data collection and prospective tracking remain
operational tasks.

## Project history

Early iterations included a tournament simulation, Poisson and XGBoost score models,
temporal evaluation and calibration reports. They were removed when the project
narrowed to match-level closing-odds forecasting with a stricter evaluation and a
prospective forecasting workflow. The earlier code is still in the git history.
