from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.team_names import (
    find_unmapped_ucl_style_names,
    load_team_name_mapping,
    normalize_team_names,
)

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_OUTPUT_PATH = PROCESSED_DIR / "matches.csv"
DEFAULT_TEAM_MAPPING_PATH = DATA_DIR / "team_name_mapping.csv"

REQUIRED_DOMESTIC_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
OPTIONAL_DOMESTIC_COLUMNS = {
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
}
REQUIRED_UCL_COLUMNS = [
    "date",
    "season",
    "competition",
    "stage",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
]

COMPETITION_NAMES = {
    "premier-league": "Premier League",
    "la-liga": "La Liga",
    "bundesliga": "Bundesliga",
    "serie-a": "Serie A",
    "ligue-1": "Ligue 1",
}
EXTRA_COMPETITIONS = {
    "primeira-liga": "Primeira Liga",
    "eredivisie": "Eredivisie",
    "belgian-pro-league": "Belgian Pro League",
    "scottish-premiership": "Scottish Premiership",
    "super-lig": "Super Lig",
    "austrian-bundesliga": "Austrian Bundesliga",
    "swiss-super-league": "Swiss Super League",
    "danish-superliga": "Danish Superliga",
    "greek-super-league": "Greek Super League",
}
SCORE_METADATA = [
    "home_extra_time_goals",
    "away_extra_time_goals",
    "home_penalties",
    "away_penalties",
    "went_to_extra_time",
    "score_raw",
]

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

STAGE_NORMALIZATIONS = {
    "gruppe g": "league_phase",
    "gruppe h": "league_phase",
}


@dataclass(frozen=True)
class BuildReport:
    rows_read: int
    rows_written: int
    blank_rows_dropped: int
    incomplete_rows_dropped: int
    input_files: int
    unmapped_ucl_team_names: tuple[str, ...]


def parse_match_dates(values: pd.Series) -> pd.Series:
    # football-data mixes 4- and 2-digit years
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


def normalize_stage_values(values: pd.Series) -> pd.Series:
    # map known variants, keep unknown text for inspection
    stage_text = values.astype("string").str.strip()
    normalized = stage_text.str.lower().map(STAGE_NORMALIZATIONS)
    return normalized.fillna(stage_text)


def load_domestic_file(
    path: Path, competition: str
) -> tuple[pd.DataFrame, int, int, int]:
    raw_matches = pd.read_csv(path)
    missing_columns = [
        col for col in REQUIRED_DOMESTIC_COLUMNS if col not in raw_matches.columns
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{path} is missing required columns: {missing}")

    rows_read = len(raw_matches)
    available_optional_columns = [
        column for column in OPTIONAL_DOMESTIC_COLUMNS if column in raw_matches.columns
    ]
    core = raw_matches[[*REQUIRED_DOMESTIC_COLUMNS, *available_optional_columns]].copy()

    # some provider files end with blank rows; count them
    blank_mask = core.isna().all(axis=1)
    blank_rows = int(blank_mask.sum())
    core = core.loc[~blank_mask].copy()

    core["date"] = parse_match_dates(core["Date"])
    core["home_goals"] = pd.to_numeric(core["FTHG"], errors="coerce")
    core["away_goals"] = pd.to_numeric(core["FTAG"], errors="coerce")
    for raw_column, normalized_column in OPTIONAL_DOMESTIC_COLUMNS.items():
        if raw_column in core.columns:
            core[normalized_column] = pd.to_numeric(core[raw_column], errors="coerce")
        else:
            core[normalized_column] = pd.NA

    # need both clubs, a date, and a regulation-time score
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
        examples = mismatches[
            ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
        ].head()
        raise ValueError(
            f"{path} has FTR/result mismatches:\n{examples.to_string(index=False)}"
        )

    normalized = pd.DataFrame(
        {
            "date": core["date"].dt.normalize(),
            "season": path.stem,
            "competition": competition,
            "stage": pd.NA,
            "home_team_raw": core["HomeTeam"].astype(str).str.strip(),
            "away_team_raw": core["AwayTeam"].astype(str).str.strip(),
            "home_goals": core["home_goals"],
            "away_goals": core["away_goals"],
            "home_shots": core["home_shots"],
            "away_shots": core["away_shots"],
            "home_shots_on_target": core["home_shots_on_target"],
            "away_shots_on_target": core["away_shots_on_target"],
            "home_corners": core["home_corners"],
            "away_corners": core["away_corners"],
            "result": core["result"],
        }
    )

    return normalized, rows_read, blank_rows, incomplete_rows


def load_ucl_file(
    path: Path, competition="Champions League"
) -> tuple[pd.DataFrame, int, int, int]:
    # downloader normalizes UCL, but schema still checked
    raw_matches = pd.read_csv(path)
    missing_columns = [
        col for col in REQUIRED_UCL_COLUMNS if col not in raw_matches.columns
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{path} is missing required columns: {missing}")

    rows_read = len(raw_matches)
    core = raw_matches[
        REQUIRED_UCL_COLUMNS + [c for c in SCORE_METADATA if c in raw_matches]
    ].copy()

    blank_mask = core.isna().all(axis=1)
    blank_rows = int(blank_mask.sum())
    core = core.loc[~blank_mask].copy()

    core["date"] = pd.to_datetime(core["date"], format="%Y-%m-%d", errors="coerce")
    core["home_goals"] = pd.to_numeric(core["home_goals"], errors="coerce")
    core["away_goals"] = pd.to_numeric(core["away_goals"], errors="coerce")

    incomplete_mask = (
        core["date"].isna()
        | core["season"].isna()
        | core["competition"].isna()
        | core["stage"].isna()
        | core["home_team"].isna()
        | core["away_team"].isna()
        | core["home_goals"].isna()
        | core["away_goals"].isna()
    )
    incomplete_rows = int(incomplete_mask.sum())
    core = core.loc[~incomplete_mask].copy()

    core["home_goals"] = core["home_goals"].astype("int64")
    core["away_goals"] = core["away_goals"].astype("int64")
    core["result"] = derive_result(core["home_goals"], core["away_goals"])

    normalized = pd.DataFrame(
        {
            "date": core["date"].dt.normalize(),
            "season": core["season"].astype(str).str.strip(),
            "competition": competition,
            "stage": normalize_stage_values(core["stage"]),
            "home_team_raw": core["home_team"].astype(str).str.strip(),
            "away_team_raw": core["away_team"].astype(str).str.strip(),
            "home_goals": core["home_goals"],
            "away_goals": core["away_goals"],
            "home_shots": pd.NA,
            "away_shots": pd.NA,
            "home_shots_on_target": pd.NA,
            "away_shots_on_target": pd.NA,
            "home_corners": pd.NA,
            "away_corners": pd.NA,
            "result": core["result"],
        }
    )

    for column in SCORE_METADATA:
        normalized[column] = core[column] if column in core else pd.NA
    return normalized, rows_read, blank_rows, incomplete_rows


def iter_domestic_files(raw_dir: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []

    for folder_name, competition_name in (
        COMPETITION_NAMES | EXTRA_COMPETITIONS
    ).items():
        folder = raw_dir / folder_name
        if not folder.exists():
            if folder_name in EXTRA_COMPETITIONS:
                continue
            raise FileNotFoundError(
                f"Expected raw data folder does not exist: {folder}"
            )

        for csv_path in sorted(folder.glob("*.csv")):
            files.append((csv_path, competition_name))

    return files


def iter_ucl_files(raw_dir: Path) -> list[Path]:
    folder = raw_dir / "champions_league"
    if not folder.exists():
        return []

    return sorted(folder.glob("*.csv"))


def build_master_matches(
    raw_dir: Path = RAW_DIR,
    team_mapping_path: Path = DEFAULT_TEAM_MAPPING_PATH,
) -> tuple[pd.DataFrame, BuildReport]:
    frames: list[pd.DataFrame] = []
    rows_read = 0
    blank_rows = 0
    incomplete_rows = 0
    input_files = iter_domestic_files(raw_dir)

    for csv_path, competition in input_files:
        frame, file_rows_read, file_blank_rows, file_incomplete_rows = (
            load_domestic_file(
                csv_path,
                competition,
            )
        )
        frames.append(frame)
        rows_read += file_rows_read
        blank_rows += file_blank_rows
        incomplete_rows += file_incomplete_rows

    ucl_files = iter_ucl_files(raw_dir)
    for csv_path in ucl_files:
        frame, file_rows_read, file_blank_rows, file_incomplete_rows = load_ucl_file(
            csv_path
        )
        frames.append(frame)
        rows_read += file_rows_read
        blank_rows += file_blank_rows
        incomplete_rows += file_incomplete_rows

    european_files = []
    for folder, competition in [
        ("europa_league", "Europa League"),
        ("conference_league", "Conference League"),
    ]:
        for csv_path in sorted((raw_dir / folder).glob("*.csv")):
            frame, read, blank, incomplete = load_ucl_file(csv_path, competition)
            frames.append(frame)
            rows_read += read
            blank_rows += blank
            incomplete_rows += incomplete
            european_files.append(csv_path)

    if not frames:
        raise ValueError(f"No match CSV files found under {raw_dir}")

    # map names before sorting and duplicate checks
    matches = pd.concat(frames, ignore_index=True)
    team_mapping = load_team_name_mapping(team_mapping_path)
    matches["home_team"] = normalize_team_names(matches["home_team_raw"], team_mapping)
    matches["away_team"] = normalize_team_names(matches["away_team_raw"], team_mapping)
    matches = matches.sort_values(
        ["date", "competition", "home_team", "away_team"],
        kind="mergesort",
    ).reset_index(drop=True)

    # dup fixture keys would repeat results in rolling features
    duplicates = matches.duplicated(
        subset=["date", "competition", "home_team", "away_team"],
        keep=False,
    )
    if duplicates.any():
        examples = matches.loc[
            duplicates,
            [
                "date",
                "competition",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
            ],
        ].head()
        raise ValueError(
            f"Unexpected duplicate matches found:\n{examples.to_string(index=False)}"
        )

    column_order = [
        "date",
        "season",
        "competition",
        "stage",
        "home_team_raw",
        "away_team_raw",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "home_shots",
        "away_shots",
        "home_shots_on_target",
        "away_shots_on_target",
        "home_corners",
        "away_corners",
        "result",
    ]
    matches = matches[column_order + [c for c in SCORE_METADATA if c in matches]]
    # ids survive provider display-name changes
    from src.data.team_names import club_id

    for side in ("home", "away"):
        matches[f"{side}_team_id"] = matches[f"{side}_team"].map(club_id)
    unmapped_ucl_team_names = find_unmapped_ucl_style_names(matches)

    report = BuildReport(
        rows_read=rows_read,
        rows_written=len(matches),
        blank_rows_dropped=blank_rows,
        incomplete_rows_dropped=incomplete_rows,
        input_files=len(input_files) + len(ucl_files) + len(european_files),
        unmapped_ucl_team_names=unmapped_ucl_team_names,
    )
    return matches, report


def write_master_matches(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    raw_dir: Path = RAW_DIR,
    team_mapping_path: Path = DEFAULT_TEAM_MAPPING_PATH,
) -> BuildReport:
    matches, report = build_master_matches(
        raw_dir=raw_dir, team_mapping_path=team_mapping_path
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output_path, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the processed master match dataset."
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--team-mapping", type=Path, default=DEFAULT_TEAM_MAPPING_PATH)
    args = parser.parse_args()

    report = write_master_matches(
        output_path=args.output,
        raw_dir=args.raw_dir,
        team_mapping_path=args.team_mapping,
    )
    print(f"Read {report.rows_read} rows from {report.input_files} files.")
    print(f"Dropped {report.blank_rows_dropped} blank rows.")
    print(f"Dropped {report.incomplete_rows_dropped} incomplete rows.")
    if report.unmapped_ucl_team_names:
        preview = ", ".join(report.unmapped_ucl_team_names[:20])
        suffix = " ..." if len(report.unmapped_ucl_team_names) > 20 else ""
        print(
            f"Unmapped UCL-style team names ({len(report.unmapped_ucl_team_names)}): {preview}{suffix}"
        )
    print(f"Wrote {report.rows_written} matches to {args.output}.")


if __name__ == "__main__":
    main()
