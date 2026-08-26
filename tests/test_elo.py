import pandas as pd

from src.features.elo import add_elo_features, expected_score, score_from_result


def make_matches(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "season": "2020_21",
        "competition": "Premier League",
        "stage": None,
        "home_team_raw": None,
        "away_team_raw": None,
        "home_goals": 0,
        "away_goals": 0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_expected_score_is_even_for_equal_ratings() -> None:
    assert expected_score(1500, 1500) == 0.5


def test_score_from_result_encoding() -> None:
    assert score_from_result(2) == (1.0, 0.0)
    assert score_from_result(1) == (0.5, 0.5)
    assert score_from_result(0) == (0.0, 1.0)


def test_first_match_starts_both_teams_at_initial_rating() -> None:
    matches = make_matches(
        [
            {
                "date": "2020-01-01",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "result": 2,
            }
        ]
    )

    featured = add_elo_features(matches)

    assert featured.loc[0, "home_elo_before"] == 1500
    assert featured.loc[0, "away_elo_before"] == 1500
    assert featured.loc[0, "elo_diff"] == 0


def test_second_match_uses_updated_pre_match_ratings() -> None:
    matches = make_matches(
        [
            {
                "date": "2020-01-01",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "result": 2,
            },
            {
                "date": "2020-01-08",
                "home_team": "Chelsea",
                "away_team": "Arsenal",
                "result": 1,
            },
        ]
    )

    featured = add_elo_features(matches)

    assert featured.loc[1, "home_elo_before"] == 1490
    assert featured.loc[1, "away_elo_before"] == 1510
    assert featured.loc[1, "elo_diff"] == -20


def test_draw_moves_ratings_toward_each_other() -> None:
    matches = make_matches(
        [
            {
                "date": "2020-01-01",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "result": 2,
            },
            {
                "date": "2020-01-08",
                "home_team": "Chelsea",
                "away_team": "Arsenal",
                "result": 1,
            },
            {
                "date": "2020-01-15",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "result": 1,
            },
        ]
    )

    featured = add_elo_features(matches)

    assert featured.loc[2, "home_elo_before"] < 1510
    assert featured.loc[2, "away_elo_before"] > 1490
    assert featured.loc[2, "elo_diff"] < 20


def test_matches_are_sorted_chronologically_before_elo_updates() -> None:
    matches = make_matches(
        [
            {
                "date": "2020-01-08",
                "home_team": "Chelsea",
                "away_team": "Arsenal",
                "result": 1,
            },
            {
                "date": "2020-01-01",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "result": 2,
            },
        ]
    )

    featured = add_elo_features(matches)

    assert featured.loc[0, "date"] == pd.Timestamp("2020-01-01")
    assert featured.loc[1, "home_team"] == "Chelsea"
    assert featured.loc[1, "home_elo_before"] == 1490
