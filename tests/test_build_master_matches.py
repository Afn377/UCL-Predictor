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
