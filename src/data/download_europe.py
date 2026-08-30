"""Reuse the public OpenFootball source for Europa and Conference League results."""

from src.data.download_ucl import (
    DATA_DIR,
    SEASONS,
    TEMP_DIR,
    clone_or_update_repo,
    parse_season,
)


def download_europe():
    if not TEMP_DIR.exists():
        clone_or_update_repo()
    for filename, directory, competition in [
        ("el.txt", "europa_league", "Europa League"),
        ("conf.txt", "conference_league", "Conference League"),
    ]:
        output = DATA_DIR / "raw" / directory
        output.mkdir(parents=True, exist_ok=True)
        for season in SEASONS:
            # conference league missing from the earliest seasons
            if not (TEMP_DIR / season / filename).exists():
                continue
            frame = parse_season(season, filename, competition)
            frame.to_csv(output / f"{season.replace('-', '_')}.csv", index=False)
            print(competition, season, len(frame), flush=True)


if __name__ == "__main__":
    download_europe()
