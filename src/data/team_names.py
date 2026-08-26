from __future__ import annotations

from pathlib import Path

import pandas as pd


MAPPING_COLUMNS = ["source_name", "canonical_name"]


def load_team_name_mapping(path: Path) -> dict[str, str]:
    if not path.exists() or path.stat().st_size == 0:
        return {}

    mapping_df = pd.read_csv(path)
    missing_columns = [col for col in MAPPING_COLUMNS if col not in mapping_df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{path} is missing required columns: {missing}")

    mapping_df = mapping_df.dropna(subset=MAPPING_COLUMNS)
    mapping_df["source_name"] = mapping_df["source_name"].astype(str).str.strip()
    mapping_df["canonical_name"] = mapping_df["canonical_name"].astype(str).str.strip()

    duplicates = mapping_df[mapping_df.duplicated("source_name", keep=False)]
    if not duplicates.empty:
        examples = duplicates["source_name"].head().to_list()
        raise ValueError(f"{path} has duplicate source_name values: {examples}")

    return dict(zip(mapping_df["source_name"], mapping_df["canonical_name"], strict=False))


def normalize_team_name(name: object, mapping: dict[str, str]) -> object:
    if pd.isna(name):
        return name

    team_name = str(name).strip()
    return mapping.get(team_name, team_name)


def normalize_team_names(values: pd.Series, mapping: dict[str, str]) -> pd.Series:
    return values.map(lambda value: normalize_team_name(value, mapping))


def find_unmapped_ucl_style_names(matches: pd.DataFrame) -> tuple[str, ...]:
    ucl_matches = matches[matches["competition"] == "Champions League"]
    if ucl_matches.empty:
        return ()

    raw_names = pd.concat(
        [ucl_matches["home_team_raw"], ucl_matches["away_team_raw"]],
        ignore_index=True,
    ).dropna()
    canonical_names = pd.concat(
        [ucl_matches["home_team"], ucl_matches["away_team"]],
        ignore_index=True,
    ).dropna()

    unmapped = raw_names[(raw_names == canonical_names) & raw_names.astype(str).str.contains(r"\([A-Z]{3}\)$")]
    return tuple(sorted(unmapped.astype(str).unique()))
