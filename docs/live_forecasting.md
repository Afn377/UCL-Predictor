# Live Forecasting

Run commands from the repository root with the virtual environment active.

1. Refresh local data and features with `python -m src.pipeline.run_pipeline`.
2. Run `python -m src.models.live freeze`. It prints a new model directory.
3. Supply fixtures and timestamped decimal odds using the schemas below.
4. Run `python -m src.models.live forecast fixtures.csv --quotes quotes.csv --model-dir artifacts/closing_models/MODEL_ID`.
5. After matches finish, run `python -m src.models.live score results.csv`.

Replace MODEL_ID with the directory printed by freeze. Older frozen bundles are
historical artifacts; changed source code invalidates them. Only load bundles
created locally.

## Inputs

Fixtures: `date,season,competition,stage,home_team,away_team,kickoff_at`.
Use `Champions League` as competition, `YYYY_MM` season labels such as `2026_27`,
and explicit UTC kickoff timestamps. Date must match the UTC kickoff date.

Quotes: `date,competition,home_team,away_team,snapshot_at,source,odds_away,odds_draw,odds_home`.
Use decimal odds and real publication timestamps. Each fixture needs a quote
no older than 15 minutes, published before the forecast. Forecasts must be issued
within 60 minutes before kickoff. Quote collection is currently manual.

Results: `date,competition,home_team,away_team,home_goals,away_goals`.
Use regulation-time goals, excluding extra time and penalties.

## Outputs and Validation

The forecast command writes all probabilities and a separate selected-picks CSV.
It records forecasts in an append-only SQLite ledger before kickoff. Scoring
joins results without replacing original forecasts and reports pending matches.
Selected accuracy must always be reported with coverage.

Training uses the configured rolling window (currently two years) before the
freeze date. Historical evaluation anchors each window at the test season start.
Precomputed Elo and rolling features can incorporate older pre-match history.
The window limits training examples, not all information used to build features.

Closing-price backtests are development results, not evidence of day-ahead or
prospective accuracy. A fresh frozen model starts a new prospective evaluation.
