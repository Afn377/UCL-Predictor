import re
import subprocess
import tempfile
from pathlib import Path

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
    # keep checkout in temp dir, outside project data
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
    # collapse era-specific wording into stable stage keys
    if stage is None:
        return None

    stage_lower = stage.lower()

    if "qualif" in stage_lower:
        return "qualifying"

    if "playoff" in stage_lower or "play-off" in stage_lower:
        return "playoff"

    if "league" in stage_lower or "group" in stage_lower or "gruppe" in stage_lower:
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


def parse_score(text):
    """Read football.txt scores without treating shoot-outs as match goals."""
    # shoot-out totals can precede the match score
    score = r"(\d+)-(\d+)"
    penalties = re.match(score + r"\s+pen\.\s+", text)
    penalty_goals = tuple(map(int, penalties.groups())) if penalties else (None, None)
    remaining = text[penalties.end() :] if penalties else text
    score_match = re.match(score, remaining)
    if score_match is None:
        raise ValueError(f"Unrecognized score: {text}")
    final_score = tuple(map(int, score_match.groups()))
    extra_time = "a.e.t." in remaining
    if extra_time:
        # parenthesized score is the model's regulation result
        regulation = re.search(r"a\.e\.t\.\s*\(" + score + r"(?:,|\))", remaining)
        if regulation is None:
            raise ValueError(f"Extra-time score lacks a regulation score: {text}")
        regulation_score = tuple(map(int, regulation.groups()))
    else:
        regulation_score = final_score
    return {
        "home_goals": regulation_score[0],
        "away_goals": regulation_score[1],
        "home_extra_time_goals": final_score[0] - regulation_score[0],
        "away_extra_time_goals": final_score[1] - regulation_score[1],
        "home_penalties": penalty_goals[0],
        "away_penalties": penalty_goals[1],
        "went_to_extra_time": extra_time,
        "score_raw": text,
    }


def parse_season(season, filename="cl.txt", competition="Champions League"):
    file_path = TEMP_DIR / season / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {file_path}")

    rows = []

    current_stage = None
    current_date = None

    with open(file_path, "r", encoding="utf-8") as season_file:
        lines = season_file.readlines()

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("▪"):
            current_stage = line.lstrip("▪").strip()
            continue

        # older files omit the bullet on stage headings
        stage_keywords = [
            "group",
            "gruppe",
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

                # aug-dec = first year, jan-jul = second year
                month_num = pd.to_datetime(month, format="%b").month

                if month_num >= 7 and normalize_stage(current_stage) in {
                    "round_of_16",
                    "quarterfinal",
                    "semifinal",
                    "final",
                }:
                    year = start_year + 1
                elif month_num >= 7:
                    year = start_year
                else:
                    year = start_year + 1

            current_date = pd.to_datetime(
                f"{day} {month} {year}",
                format="%d %b %Y",
            )

            continue

        if line.startswith("#") or line.startswith("="):
            continue

        line_without_time = re.sub(
            r"^\d{1,2}[:.]\d{2}\s+",
            "",
            line,
        )

        match = re.match(
            r"^(.*?)\s+v\s+(.*?)\s+" r"(\d+-\d+.*)$",
            line_without_time,
        )

        if not match:
            continue

        home_team, away_team, score_text = match.groups()

        # keep raw stage text for auditing
        rows.append(
            {
                "date": current_date,
                "season": season.replace("-", "_"),
                "competition": competition,
                "stage_raw": current_stage,
                "stage": normalize_stage(current_stage),
                "home_team": home_team.strip(),
                "away_team": away_team.strip(),
                **parse_score(score_text),
            }
        )

    season_matches = pd.DataFrame(rows)

    if season_matches.empty:
        raise ValueError(f"No matches parsed for {season}")

    return season_matches


def main():
    clone_or_update_repo()

    all_seasons = []

    for season in SEASONS:
        # one CSV per season simplifies repairs and coverage checks
        print(f"Parsing {season}...")

        season_matches = parse_season(season)

        output_path = RAW_DIR / f"{season.replace('-', '_')}.csv"

        season_matches.to_csv(output_path, index=False)

        print(
            f"Saved {len(season_matches)} matches -> "
            f"{output_path.relative_to(DATA_DIR)}"
        )

        all_seasons.append(season_matches)

    # combined frame only used for this summary
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
