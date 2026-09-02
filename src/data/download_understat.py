"""Cache public Understat league responses and normalize completed matches."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "src/data/raw/understat"
LEAGUES = {
    "EPL": "Premier League",
    "La_liga": "La Liga",
    "Bundesliga": "Bundesliga",
    "Serie_A": "Serie A",
    "Ligue_1": "Ligue 1",
}


def normalize_league_data(data: dict, league: str, year: int) -> pd.DataFrame:
    # understat stores fixture xG and team npxG in separate sections
    if not isinstance(data.get("dates"), list) or not isinstance(
        data.get("teams"), dict
    ):
        raise ValueError("Understat response must contain dates and teams")
    history = {}
    for team in data["teams"].values():
        for match in team.get("history", []):
            key = (str(team["id"]), str(match["date"])[:10], match["h_a"])
            if key in history:
                raise ValueError(f"Ambiguous Understat team history: {key}")
            history[key] = match
    rows = []
    for match in data["dates"]:
        # scheduled fixtures have no final goals or real xG
        if match.get("isResult") not in (True, "true", 1):
            continue
        date = str(match["datetime"])[:10]
        row = {
            "understat_id": str(match["id"]),
            "date": date,
            "season": f"{year}_{str(year + 1)[-2:]}",
            "competition": LEAGUES[league],
            "home_team": match["h"]["title"],
            "away_team": match["a"]["title"],
            "home_goals": int(match["goals"]["h"]),
            "away_goals": int(match["goals"]["a"]),
            "home_xg": float(match["xG"]["h"]),
            "away_xg": float(match["xG"]["a"]),
        }
        for side, short in [("home", "h"), ("away", "a")]:
            stats = history.get((str(match[short]["id"]), date, short), {})
            row[f"{side}_npxg"] = stats.get("npxG", float("nan"))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No completed matches returned for {league}/{year}")
    if frame.understat_id.duplicated().any():
        raise ValueError("Duplicate Understat match IDs")
    numeric = ["home_xg", "away_xg", "home_npxg", "away_npxg"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if frame[numeric].lt(0).any().any():
        raise ValueError("Negative xG in Understat response")
    return frame


def download_understat(
    start_year: int = 2014,
    end_year: int = 2025,
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> pd.DataFrame:
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    with requests.Session() as session:
        # endpoint needs browser-like AJAX headers
        session.headers.update(
            {
                "User-Agent": "UCL-Forecast research",
                "X-Requested-With": "XMLHttpRequest",
            }
        )
        for league in LEAGUES:
            for year in range(start_year, end_year + 1):
                path = cache_dir / f"{league}_{year}.json"
                if path.exists() and not refresh:
                    # cache keeps feature rebuilds reproducible
                    payload = json.loads(path.read_text())
                else:
                    url = f"https://understat.com/getLeagueData/{league}/{year}"
                    response = session.get(
                        url,
                        timeout=45,
                        headers={
                            "Referer": f"https://understat.com/league/{league}/{year}"
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    normalize_league_data(data, league, year)
                    payload = {
                        "source_url": url,
                        "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
                        "data": data,
                    }
                    # temp file first in case the download is interrupted
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(payload, ensure_ascii=True))
                    temporary.replace(path)
                    time.sleep(0.35)
                frames.append(normalize_league_data(payload["data"], league, year))
                print(
                    f"Understat {league}/{year}: {len(frames[-1])} matches", flush=True
                )
    matches = pd.concat(frames, ignore_index=True)
    matches.to_csv(cache_dir / "matches.csv", index=False)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    download_understat(args.start_year, args.end_year, args.cache_dir, args.refresh)


if __name__ == "__main__":
    main()
