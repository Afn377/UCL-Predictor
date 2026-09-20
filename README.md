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
python -m src.pipeline.run_pipeline
```

The pipeline uses local raw datasets. Optional download flags are
`--download-ucl`, `--download-xg`, `--download-domestic`, and `--download-europe`.
Run `python -m src.data.download_footiqo --help` for the separate odds downloader.
Network access and upstream availability are required for downloads.

## Data and Features

Raw data under `src/data/raw` contains domestic match results/statistics, European
competition results, Understat xG, ClubElo snapshots, and Footiqo historical odds.
Download modules document source URLs; team mapping files reconcile names.
Availability and coverage vary by season and competition.

Processed stages are `matches.csv`, `matches_with_elo.csv`,
`matches_with_features.csv`, `model_dataset.csv`, and `matches_with_context.csv`.
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
