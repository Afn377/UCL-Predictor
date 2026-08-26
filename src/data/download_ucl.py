from pathlib import Path
import re
import subprocess
import tempfile
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent

RAW_DIR = DATA_DIR / "raw" / "champions_league"
TEMP_DIR = Path(tempfile.gettempdir()) / "ucl_forecast_openfootball_champions_league"

RAW_DIR.mkdir(parents=True, exist_ok=True)

REPO_URL = "https://github.com/openfootball/champions-league.git"

SEASONS = [
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]


def clone_or_update_repo():
    if TEMP_DIR.exists():
        print("Updating OpenFootball repository...")
        subprocess.run(
            ["git", "-C", str(TEMP_DIR), "pull"],
            check=True,
        )
    else:
        print("Cloning OpenFootball repository...")
        subprocess.run(
            ["git", "clone", REPO_URL, str(TEMP_DIR)],
            check=True,
        )


def normalize_stage(stage):
    if stage is None:
        return None

    stage_lower = stage.lower()

    if "qualif" in stage_lower:
        return "qualifying"

    if "playoff" in stage_lower or "play-off" in stage_lower:
        return "playoff"

    if "league" in stage_lower or "group" in stage_lower:
        return "league_phase"

    if "round of 16" in stage_lower or "last 16" in stage_lower:
        return "round_of_16"

    if "quarter" in stage_lower:
        return "quarterfinal"

    if "semi" in stage_lower:
        return "semifinal"

    if "final" in stage_lower:
        return "final"

    return stage.strip()


def parse_season(season):
    file_path = TEMP_DIR / season / "cl.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {file_path}")

    rows = []

    current_stage = None
    current_date = None

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            continue

        # Stage headings such as:
        # ▪ League, Matchday 1
        # ▪ Round of 16
        if line.startswith("▪"):
            current_stage = line.lstrip("▪").strip()
            continue

        # Older files may use headings without the bullet.
        stage_keywords = [
            "group",
            "league",
            "round of 16",
            "quarter-final",
            "quarterfinal",
            "semi-final",
            "semifinal",
            "final",
            "playoff",
            "play-off",
            "qualifying",
        ]

        line_lower = line.lower()

        if (
            not re.search(r"\d+-\d+", line)
            and any(keyword in line_lower for keyword in stage_keywords)
            and not line.startswith("#")
        ):
            current_stage = line
            continue

        # Date formats such as:
        # Tue Sep 16 2025
        # Wed Sep 17
        date_match = re.match(
            r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
            r"([A-Z][a-z]{2})\s+"
            r"(\d{1,2})(?:\s+(\d{4}))?$",
            line,
        )

        if date_match:
            weekday, month, day, year = date_match.groups()

            if year is None:
                start_year = int(season.split("-")[0])

                # Aug-Dec belong to the first year,
                # Jan-Jul belong to the second year.
                month_num = pd.to_datetime(month, format="%b").month

                if month_num >= 7:
                    year = start_year
                else:
                    year = start_year + 1

            current_date = pd.to_datetime(
                f"{day} {month} {year}",
                format="%d %b %Y",
            )

            continue

        # Skip comments / metadata
        if line.startswith("#") or line.startswith("="):
            continue

        # Remove kickoff time if present:
        # 21:00 Arsenal FC ... 2-1 ...
        line_without_time = re.sub(
            r"^\d{1,2}[:.]\d{2}\s+",
            "",
            line,
        )

        # Match format generally contains:
        # Home Team v Away Team 2-1
        #
        # Score may optionally be followed by halftime score:
        # 2-1 (1-0)
        match = re.match(
            r"^(.*?)\s+v\s+(.*?)\s+"
            r"(\d+)-(\d+)"
            r"(?:\s+\([^)]*\))?"
            r"(?:\s+.*)?$",
            line_without_time,
        )

        if not match:
            continue

        home_team, away_team, home_goals, away_goals = match.groups()

        rows.append(
            {
                "date": current_date,
                "season": season.replace("-", "_"),
                "competition": "Champions League",
                "stage_raw": current_stage,
                "stage": normalize_stage(current_stage),
                "home_team": home_team.strip(),
                "away_team": away_team.strip(),
                "home_goals": int(home_goals),
                "away_goals": int(away_goals),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No matches parsed for {season}")

    return df


def main():
    clone_or_update_repo()

    all_seasons = []

    for season in SEASONS:
        print(f"Parsing {season}...")

        df = parse_season(season)

        output_path = RAW_DIR / f"{season.replace('-', '_')}.csv"

        df.to_csv(output_path, index=False)

        print(
            f"Saved {len(df)} matches -> "
            f"{output_path.relative_to(DATA_DIR)}"
        )

        all_seasons.append(df)

    combined = pd.concat(all_seasons, ignore_index=True)

    combined = combined.sort_values("date").reset_index(drop=True)

    print()
    print("Finished.")
    print(f"Total UCL matches: {len(combined)}")
    print(
        f"Date range: "
        f"{combined['date'].min().date()} "
        f"to {combined['date'].max().date()}"
    )

    print()
    print("Matches by season:")
    print(combined["season"].value_counts().sort_index())

    print()
    print("Stages:")
    print(combined["stage"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
