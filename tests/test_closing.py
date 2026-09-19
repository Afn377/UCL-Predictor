import json
import sqlite3

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.market import CLOSING_FEATURES, MATCH_KEYS
from src.evaluation.closing import disagreement_report
from src.evaluation.confidence import PROBABILITY_COLUMNS
from src.features.context import RICH_FEATURES
from src.features.strength import COMPACT_FEATURES
from src.models.closing import ClosingModel, fit_candidate, select_candidates
from src.models.live import live_inputs, record_predictions, score_ledger


def training_rows():
    rng = np.random.default_rng(9)
    data = pd.DataFrame(
        rng.normal(size=(240, len(set(RICH_FEATURES + COMPACT_FEATURES)))),
        columns=sorted(set(RICH_FEATURES + COMPACT_FEATURES)),
    )
    data["date"] = pd.date_range("2020-01-01", periods=len(data))
    data["competition"] = "Champions League"
    data["result"] = np.tile([0, 1, 2], len(data) // 3)
    data[CLOSING_FEATURES] = [0.2, 0.3, 0.5]
    return data


@pytest.mark.parametrize(
    "family",
    [
        "closing_odds",
        "closing_blend",
        "football_xg",
        "closing_logistic",
        "naive_base_rate",
    ],
)
def test_ad_hoc_forecast_routes_models_and_limits_training(
    tmp_path, monkeypatch, family
):
    from src.models import forecast

    data = training_rows().assign(
        home_team="A", away_team="B", season="2020_21", stage="league_phase"
    )
    old = data.iloc[:1].copy().assign(date=pd.Timestamp("2017-01-01"))
    future = data.iloc[:1].copy().assign(date=pd.Timestamp("2022-01-01"))
    pd.concat([old, data, future]).to_csv(tmp_path / "context.csv", index=False)
    pd.DataFrame(columns=["date", "model", "competition"]).to_csv(
        tmp_path / "predictions.csv", index=False
    )
    fixture = (
        data.tail(1)
        .copy()
        .assign(
            date=pd.Timestamp("2021-01-01"),
            result=np.nan,
            forecast_at=pd.Timestamp("2021-01-01", tz="UTC"),
        )
    )
    monkeypatch.setattr(
        forecast, "prepare_forecast_features", lambda *args: fixture.copy()
    )
    original_fit = forecast.fit_candidate
    observed = []

    def checked_fit(train, spec):
        assert train.date.min() >= pd.Timestamp("2019-01-01")
        assert train.date.max() < pd.Timestamp("2021-01-01")
        observed.append(spec["family"])
        return original_fit(train, spec)

    monkeypatch.setattr(forecast, "fit_candidate", checked_fit)
    quotes = fixture[MATCH_KEYS].assign(
        snapshot_at="2020-12-31T23:59:00Z",
        source="test",
        odds_away=5.0,
        odds_draw=10 / 3,
        odds_home=2.0,
    )
    result = forecast.forecast_fixtures(
        fixture,
        "2021-01-01",
        family,
        context_path=tmp_path / "context.csv",
        predictions_path=tmp_path / "predictions.csv",
        quotes=quotes,
    )
    assert observed == [family]
    assert result.model.eq(family).all()
    np.testing.assert_allclose(result[PROBABILITY_COLUMNS].sum(axis=1), 1.0)


def test_closing_selection_preserves_dates_and_probability_order():
    data = training_rows()
    specs, audit = select_candidates(data)
    assert pd.Timestamp(audit["training_end"]) < pd.Timestamp(audit["validation_start"])
    assert audit["validation_rows"] == 60
    assert len(audit["candidates"]) == 10
    test = data.tail(4).copy()
    test["result"] = np.nan
    for spec in specs.values():
        probabilities = fit_candidate(data, spec).predict_proba(test)
        assert probabilities.shape == (4, 3)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_allclose(
        ClosingModel("closing_odds").predict_proba(test), [[0.2, 0.3, 0.5]] * 4
    )
    test.loc[test.index[0], "closing_prob_0"] = np.nan
    with pytest.raises(ValueError, match="complete"):
        ClosingModel("closing_odds").predict_proba(test)


def test_disagreements_reject_changed_labels_and_count_unique_corrections():
    base = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3),
            "competition": "Champions League",
            "home_team": "A",
            "away_team": "B",
            "split": "test",
            "result": [0, 1, 2],
        }
    )
    odds = base.assign(model="closing_odds")
    football = base.assign(model="football_xg")
    odds[PROBABILITY_COLUMNS] = [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]
    football[PROBABILITY_COLUMNS] = [[0.1, 0.1, 0.8], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    _, report = disagreement_report(pd.concat([odds, football]))
    overall = report.iloc[0]
    assert overall.disagreements == 2
    assert overall.football_only_correct == overall.odds_only_correct == 1
    football.loc[0, "result"] = 2
    with pytest.raises(ValueError, match="labels"):
        disagreement_report(pd.concat([odds, football]))


def live_frames():
    now = pd.Timestamp.now(tz="UTC")
    kickoff = now + pd.Timedelta(minutes=30)
    fixtures = pd.DataFrame(
        {
            "date": [kickoff.tz_localize(None).normalize()],
            "competition": ["Champions League"],
            "home_team": ["Arsenal"],
            "away_team": ["Liverpool"],
            "kickoff_at": [kickoff.isoformat()],
        }
    )
    quotes = fixtures[MATCH_KEYS].assign(
        snapshot_at=now.isoformat(),
        source="test",
        odds_away=4.0,
        odds_draw=4.0,
        odds_home=2.0,
    )
    manifest = {
        "model_id": "test",
        "train_end": "2020-01-01",
        "created_at": "2020-01-02T00:00:00Z",
        "forecast_window_minutes": 60,
        "maximum_quote_age_minutes": 15,
    }
    return now, fixtures, quotes, manifest


def test_live_requires_fresh_quotes_and_pre_kickoff_forecast():
    now, fixtures, quotes, manifest = live_frames()
    _, matched = live_inputs(fixtures, quotes, now, manifest)
    assert len(matched) == 1
    for age in (-1, 16):
        changed = quotes.assign(
            snapshot_at=(now - pd.Timedelta(minutes=age)).isoformat()
        )
        with pytest.raises(ValueError, match="recent quote"):
            live_inputs(fixtures, changed, now, manifest)
    with pytest.raises(ValueError, match="before kickoff"):
        live_inputs(fixtures.assign(kickoff_at=now.isoformat()), quotes, now, manifest)
    with pytest.raises(ValueError, match="ambiguous"):
        live_inputs(fixtures, pd.concat([quotes, quotes]), now, manifest)


def test_ledger_is_atomic_and_rejects_replacement_and_late_predictions(tmp_path):
    now, fixtures, _, manifest = live_frames()
    predictions = fixtures.assign(
        model_id="test",
        model="closing_odds",
        selected=False,
        predicted_result=2,
        forecast_at=now.isoformat(),
    )
    predictions[PROBABILITY_COLUMNS] = [0.25, 0.25, 0.5]
    ledger = tmp_path / "ledger.sqlite"
    record_predictions(predictions, manifest, {}, ledger)
    with pytest.raises(sqlite3.IntegrityError):
        record_predictions(predictions, manifest, {}, ledger)
    with sqlite3.connect(ledger) as connection:
        assert connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM predictions")
    with pytest.raises(ValueError, match="before kickoff"):
        record_predictions(
            predictions.assign(kickoff_at=(now - pd.Timedelta(seconds=1)).isoformat()),
            manifest,
            {},
            ledger,
        )
    results = fixtures[MATCH_KEYS].assign(home_goals=2, away_goals=0)
    results.to_csv(tmp_path / "results.csv", index=False)
    report = score_ledger(tmp_path / "results.csv", ledger, tmp_path / "scored")
    assert report.pending.eq(1).all()
    assert report.scored.eq(0).all()


def test_score_ledger_joins_results_without_changing_predictions(tmp_path):
    # A historical ledger fixture represents a forecast that was recorded before kickoff.
    now, fixtures, _, manifest = live_frames()
    past = now - pd.Timedelta(days=2)
    fixtures["date"] = past.tz_localize(None).normalize()
    fixtures["kickoff_at"] = past.isoformat()
    row = json.loads(fixtures.to_json(orient="records", date_format="iso"))[0]
    row.update(
        model_id="test",
        predicted_result=2,
        selected=True,
        prob_0=0.2,
        prob_1=0.3,
        prob_2=0.5,
    )
    ledger = tmp_path / "historical.sqlite"
    with sqlite3.connect(ledger) as connection:
        connection.execute("CREATE TABLE predictions (payload TEXT, recorded_at TEXT)")
        connection.execute(
            "INSERT INTO predictions VALUES (?, ?)",
            (
                json.dumps({"forecast": row}),
                (past - pd.Timedelta(minutes=10)).isoformat(),
            ),
        )
    results = fixtures[MATCH_KEYS].assign(home_goals=2, away_goals=0)
    results.to_csv(tmp_path / "results.csv", index=False)
    report = score_ledger(tmp_path / "results.csv", ledger, tmp_path / "scored")
    assert report.accuracy.eq(1).all()
    assert report.scored.eq(1).all()


def test_live_forecast_writes_all_and_selected_views_and_records_inputs(
    tmp_path, monkeypatch
):
    from src.evaluation.confidence import ConfidencePolicy
    from src.models import live

    now, fixtures, quotes, manifest = live_frames()
    fixtures = fixtures.assign(season="2026_27", stage="league_phase")
    fixtures.to_csv(tmp_path / "fixtures.csv", index=False)
    quotes.to_csv(tmp_path / "quotes.csv", index=False)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    history = tmp_path / "history.csv"
    history.write_text("date\n")
    monkeypatch.setattr(live, "HISTORY", history)
    xg = tmp_path / "xg.csv"
    xg.write_text("date\n")
    monkeypatch.setattr(live, "DEFAULT_XG", xg)
    policy = ConfidencePolicy(
        "closing_odds",
        "Champions League",
        "2020-01-01",
        None,
        None,
        threshold=0.7,
        status="ready",
    )
    joblib.dump(
        {"model": ClosingModel("closing_odds"), "policy": policy},
        model_dir / "model.joblib",
    )
    manifest.update(
        bundle_sha256=live.sha256(model_dir / "model.joblib"),
        source_sha256={},
        packages={},
    )
    (model_dir / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        live,
        "prepare_forecast_features",
        lambda data, as_of, **kw: data.assign(forecast_at=as_of),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "live",
            "forecast",
            str(tmp_path / "fixtures.csv"),
            "--quotes",
            str(tmp_path / "quotes.csv"),
            "--model-dir",
            str(model_dir),
            "--output",
            str(tmp_path / "forecasts.csv"),
            "--ledger",
            str(tmp_path / "ledger.sqlite"),
        ],
    )
    live.main()
    result = pd.read_csv(tmp_path / "forecasts.csv")
    assert len(result) == 1
    assert not result.selected.any()
    assert pd.read_csv(tmp_path / "forecasts_selected.csv").empty
    with sqlite3.connect(tmp_path / "ledger.sqlite") as connection:
        payload = json.loads(
            connection.execute("SELECT payload FROM predictions").fetchone()[0]
        )
    assert payload["provenance"]["quotes_sha256"] == live.sha256(
        tmp_path / "quotes.csv"
    )
    assert payload["forecast"]["quote_source"] == "test"
    (model_dir / "model.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        live.load_frozen(model_dir)
