import pandas as pd

from src.features.model_dataset import FEATURE_COLUMNS, MODEL_COLUMNS, build_model_dataset


def make_feature_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "date": "2020-01-01",
        "season": "2020_21",
        "competition": "Premier League",
        "stage": None,
        "home_team": "Home",
        "away_team": "Away",
        "result": 1,
    }
    defaults.update({feature: 0.0 for feature in FEATURE_COLUMNS})
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_build_model_dataset_selects_expected_columns() -> None:
    matches = make_feature_rows(
        [
            {
                "extra_column": "ignored",
            }
        ]
    )

    model_data = build_model_dataset(matches)

    assert list(model_data.columns) == MODEL_COLUMNS


def test_build_model_dataset_drops_rows_with_missing_features() -> None:
    matches = make_feature_rows(
        [
            {
                "date": "2020-01-01",
                "home_team": "Team A",
                "away_team": "Team B",
                "ppg_5_diff": None,
            },
            {
                "date": "2020-01-02",
                "home_team": "Team C",
                "away_team": "Team D",
                "ppg_5_diff": 1.5,
            },
        ]
    )

    model_data = build_model_dataset(matches)

    assert len(model_data) == 1
    assert model_data.loc[0, "home_team"] == "Team C"


def test_build_model_dataset_sorts_chronologically() -> None:
    matches = make_feature_rows(
        [
            {
                "date": "2020-01-02",
                "home_team": "Team C",
                "away_team": "Team D",
            },
            {
                "date": "2020-01-01",
                "home_team": "Team A",
                "away_team": "Team B",
            },
        ]
    )

    model_data = build_model_dataset(matches)

    assert model_data.loc[0, "date"] == pd.Timestamp("2020-01-01")
    assert model_data.loc[0, "home_team"] == "Team A"


def test_build_model_dataset_allows_custom_feature_columns() -> None:
    matches = make_feature_rows(
        [
            {
                "elo_diff": 20,
                "custom_feature": 4.2,
            }
        ]
    )

    model_data = build_model_dataset(matches, feature_columns=["elo_diff", "custom_feature"])

    assert "custom_feature" in model_data.columns
    assert "elo_diff" in FEATURE_COLUMNS
    assert "is_champions_league" in FEATURE_COLUMNS
