from pathlib import Path

import pandas as pd

from src.data.build_master_matches import build_master_matches, derive_result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def make_raw_tree(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    competitions = ["premier-league", "la-liga", "bundesliga", "serie-a", "ligue-1"]

    for competition in competitions:
        write_csv(
            raw_dir / competition / "2020_21.csv",
            [
                {
                    "Date": "02/01/21",
                    "HomeTeam": f"{competition} Home",
                    "AwayTeam": f"{competition} Away",
                    "FTHG": 1,
                    "FTAG": 1,
                    "FTR": "D",
                }
            ],
        )

    return raw_dir


def test_derive_result_encoding() -> None:
    result = derive_result(
        pd.Series([0, 1, 3]),
        pd.Series([2, 1, 0]),
    )

    assert result.tolist() == [0, 1, 2]


def test_build_master_matches_drops_blank_rows_and_sorts_by_date(tmp_path: Path) -> None:
    raw_dir = make_raw_tree(tmp_path)
    write_csv(
        raw_dir / "premier-league" / "2021_22.csv",
        [
            {
                "Date": None,
                "HomeTeam": None,
                "AwayTeam": None,
                "FTHG": None,
                "FTAG": None,
                "FTR": None,
            },
            {
                "Date": "01/01/20",
                "HomeTeam": "Early Home",
                "AwayTeam": "Early Away",
                "FTHG": 2,
                "FTAG": 0,
                "FTR": "H",
            },
        ],
    )

    matches, report = build_master_matches(raw_dir=raw_dir)

    assert report.blank_rows_dropped == 1
    assert "Early Home" in set(matches["home_team"])
    assert matches["date"].is_monotonic_increasing
    assert not matches[["date", "home_team", "away_team", "home_goals", "away_goals"]].isna().any().any()


def test_build_master_matches_validates_ftr_against_goals(tmp_path: Path) -> None:
    raw_dir = make_raw_tree(tmp_path)
    write_csv(
        raw_dir / "serie-a" / "2021_22.csv",
        [
            {
                "Date": "01/01/21",
                "HomeTeam": "Mismatch Home",
                "AwayTeam": "Mismatch Away",
                "FTHG": 2,
                "FTAG": 0,
                "FTR": "A",
            }
        ],
    )

    try:
        build_master_matches(raw_dir=raw_dir)
    except ValueError as exc:
        assert "FTR/result mismatches" in str(exc)
    else:
        raise AssertionError("Expected FTR/result mismatch validation to fail")


def test_build_master_matches_includes_champions_league_rows(tmp_path: Path) -> None:
    raw_dir = make_raw_tree(tmp_path)
    write_csv(
        raw_dir / "champions_league" / "2020_21.csv",
        [
            {
                "date": "2020-12-01",
                "season": "2020_21",
                "competition": "Champions League",
                "stage_raw": "Group A",
                "stage": "league_phase",
                "home_team": "UCL Home",
                "away_team": "UCL Away",
                "home_goals": 0,
                "away_goals": 1,
            }
        ],
    )

    matches, report = build_master_matches(raw_dir=raw_dir)
    ucl_rows = matches[matches["competition"] == "Champions League"]

    assert report.input_files == 6
    assert len(ucl_rows) == 1
    assert ucl_rows.iloc[0]["stage"] == "league_phase"
    assert ucl_rows.iloc[0]["result"] == 0
    assert "stage" in matches.columns


def test_build_master_matches_normalizes_ucl_gruppe_stage(tmp_path: Path) -> None:
    raw_dir = make_raw_tree(tmp_path)
    write_csv(
        raw_dir / "champions_league" / "2019_20.csv",
        [
            {
                "date": "2019-09-17",
                "season": "2019_20",
                "competition": "Champions League",
                "stage_raw": "Gruppe G",
                "stage": "Gruppe G",
                "home_team": "RB Leipzig (GER)",
                "away_team": "Benfica (POR)",
                "home_goals": 2,
                "away_goals": 1,
            }
        ],
    )

    matches, _ = build_master_matches(raw_dir=raw_dir)
    ucl_row = matches[matches["competition"] == "Champions League"].iloc[0]

    assert ucl_row["stage"] == "league_phase"
    assert ucl_row["result"] == 2
