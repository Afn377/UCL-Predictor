from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_OUTPUT_PATH = PROCESSED_DIR / "matches.csv"

REQUIRED_DOMESTIC_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

COMPETITION_NAMES = {
    "premier-league": "Premier League",
    "la-liga": "La Liga",
    "bundesliga": "Bundesliga",
    "serie-a": "Serie A",
    "ligue-1": "Ligue 1",
}

RESULT_FROM_GOALS = {
    "away": 0,
    "draw": 1,
    "home": 2,
}

RESULT_FROM_FTR = {
    "A": RESULT_FROM_GOALS["away"],
    "D": RESULT_FROM_GOALS["draw"],
    "H": RESULT_FROM_GOALS["home"],
}


@dataclass(frozen=True)
class BuildReport:
    rows_read: int
    rows_written: int
    blank_rows_dropped: int
    incomplete_rows_dropped: int
    input_files: int


def parse_match_dates(values: pd.Series) -> pd.Series:
    date_text = values.astype("string").str.strip()
    parsed = pd.to_datetime(date_text, format="%d/%m/%Y", errors="coerce")
    missing_mask = parsed.isna()

    if missing_mask.any():
        parsed = parsed.mask(
            missing_mask,
            pd.to_datetime(date_text[missing_mask], format="%d/%m/%y", errors="coerce"),
        )

    return parsed


def derive_result(home_goals: pd.Series, away_goals: pd.Series) -> pd.Series:
    result = pd.Series(RESULT_FROM_GOALS["draw"], index=home_goals.index, dtype="int64")
    result = result.mask(home_goals > away_goals, RESULT_FROM_GOALS["home"])
    result = result.mask(home_goals < away_goals, RESULT_FROM_GOALS["away"])
    return result


def load_domestic_file(path: Path, competition: str) -> tuple[pd.DataFrame, int, int, int]:
    df = pd.read_csv(path)
    missing_columns = [col for col in REQUIRED_DOMESTIC_COLUMNS if col not in df.columns]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{path} is missing required columns: {missing}")

    rows_read = len(df)
    core = df[REQUIRED_DOMESTIC_COLUMNS].copy()

    blank_mask = core.isna().all(axis=1)
    blank_rows = int(blank_mask.sum())
    core = core.loc[~blank_mask].copy()

    core["date"] = parse_match_dates(core["Date"])
    core["home_goals"] = pd.to_numeric(core["FTHG"], errors="coerce")
    core["away_goals"] = pd.to_numeric(core["FTAG"], errors="coerce")

    incomplete_mask = (
        core["date"].isna()
        | core["HomeTeam"].isna()
        | core["AwayTeam"].isna()
        | core["home_goals"].isna()
        | core["away_goals"].isna()
    )
    incomplete_rows = int(incomplete_mask.sum())
    core = core.loc[~incomplete_mask].copy()

    core["home_goals"] = core["home_goals"].astype("int64")
    core["away_goals"] = core["away_goals"].astype("int64")
    core["result"] = derive_result(core["home_goals"], core["away_goals"])

    ftr_result = core["FTR"].map(RESULT_FROM_FTR)
    mismatches = core[ftr_result.notna() & (ftr_result != core["result"])]
    if not mismatches.empty:
        examples = mismatches[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].head()
        raise ValueError(f"{path} has FTR/result mismatches:\n{examples.to_string(index=False)}")

    normalized = pd.DataFrame(
        {
            "date": core["date"].dt.normalize(),
            "season": path.stem,
            "competition": competition,
            "home_team": core["HomeTeam"].astype(str).str.strip(),
            "away_team": core["AwayTeam"].astype(str).str.strip(),
            "home_goals": core["home_goals"],
            "away_goals": core["away_goals"],
            "result": core["result"],
        }
    )

    return normalized, rows_read, blank_rows, incomplete_rows


def iter_domestic_files(raw_dir: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []

    for folder_name, competition_name in COMPETITION_NAMES.items():
        folder = raw_dir / folder_name
        if not folder.exists():
            raise FileNotFoundError(f"Expected raw data folder does not exist: {folder}")

        for csv_path in sorted(folder.glob("*.csv")):
            files.append((csv_path, competition_name))

    return files


def build_master_matches(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, BuildReport]:
    frames: list[pd.DataFrame] = []
    rows_read = 0
    blank_rows = 0
    incomplete_rows = 0
    input_files = iter_domestic_files(raw_dir)

    for csv_path, competition in input_files:
        frame, file_rows_read, file_blank_rows, file_incomplete_rows = load_domestic_file(
            csv_path,
            competition,
        )
        frames.append(frame)
        rows_read += file_rows_read
        blank_rows += file_blank_rows
        incomplete_rows += file_incomplete_rows

    if not frames:
        raise ValueError(f"No domestic CSV files found under {raw_dir}")

    matches = pd.concat(frames, ignore_index=True)
    matches = matches.sort_values(
        ["date", "competition", "home_team", "away_team"],
        kind="mergesort",
    ).reset_index(drop=True)

    duplicates = matches.duplicated(
        subset=["date", "competition", "home_team", "away_team"],
        keep=False,
    )
    if duplicates.any():
        examples = matches.loc[
            duplicates,
            ["date", "competition", "home_team", "away_team", "home_goals", "away_goals"],
        ].head()
        raise ValueError(f"Unexpected duplicate matches found:\n{examples.to_string(index=False)}")

    report = BuildReport(
        rows_read=rows_read,
        rows_written=len(matches),
        blank_rows_dropped=blank_rows,
        incomplete_rows_dropped=incomplete_rows,
        input_files=len(input_files),
    )
    return matches, report


def write_master_matches(output_path: Path = DEFAULT_OUTPUT_PATH, raw_dir: Path = RAW_DIR) -> BuildReport:
    matches, report = build_master_matches(raw_dir=raw_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output_path, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the processed domestic master match dataset.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = write_master_matches(output_path=args.output, raw_dir=args.raw_dir)
    print(f"Read {report.rows_read} rows from {report.input_files} files.")
    print(f"Dropped {report.blank_rows_dropped} blank rows.")
    print(f"Dropped {report.incomplete_rows_dropped} incomplete rows.")
    print(f"Wrote {report.rows_written} matches to {args.output}.")


if __name__ == "__main__":
    main()
