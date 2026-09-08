"""Download public ClubElo histories for clubs in the local UCL dataset."""

import argparse
import re
import time
import unicodedata
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from src.data.team_names import load_team_name_mapping, normalize_team_names

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "src/data/raw/clubelo"
ALIASES = {
    "Bayern Munich": "Bayern",
    "Man United": "ManUnited",
    "Man City": "ManCity",
    "Paris SG": "PSG",
    "Ath Madrid": "Atletico",
    "Ath Bilbao": "Bilbao",
    "Sp Lisbon": "Sporting",
    "Sp Braga": "Braga",
    "PSV Eindhoven": "PSV",
    "M'gladbach": "Gladbach",
    "Ein Frankfurt": "Frankfurt",
    "Sociedad": "Sociedad",
    "FC Copenhagen": "Kobenhavn",
    "Club Brugge": "ClubBrugge",
    "Buyuksehyr": "Basaksehir",
}


def provider_name(name):
    # known clubs use clubelo slugs, else ascii-strip the name
    if name in ALIASES:
        return ALIASES[name]
    name = re.sub(r"\s*\([A-Z]{3}\)$", "", name)
    return re.sub(
        r"[^A-Za-z0-9]",
        "",
        unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode(),
    )


def download_clubelo(refresh=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    mapping = load_team_name_mapping(ROOT / "src/data/team_name_mapping.csv")
    raw = pd.concat(
        [
            pd.read_csv(csv_path)
            for csv_path in (ROOT / "src/data/raw/champions_league").glob("*.csv")
        ]
    )
    teams = sorted(
        set(normalize_team_names(raw.home_team, mapping))
        | set(normalize_team_names(raw.away_team, mapping))
    )
    frames, audit = [], []
    session = requests.Session()
    for team in teams:
        # per-club cache file so interrupted runs can resume
        name = provider_name(team)
        path = CACHE / f"{name}.csv"
        try:
            if not path.exists() or refresh:
                response = session.get(f"http://api.clubelo.com/{name}", timeout=15)
                response.raise_for_status()
                frame = pd.read_csv(StringIO(response.text))
                if not {"From", "Elo"} <= set(frame):
                    raise ValueError("No recognized club history returned")
                frame.to_csv(path, index=False)
                time.sleep(0.2)
            frame = pd.read_csv(path)
            normalized = pd.DataFrame(
                {
                    "team": team,
                    "date": pd.to_datetime(frame.From),
                    "elo": pd.to_numeric(frame.Elo, errors="raise"),
                }
            )
            frames.append(normalized[normalized.date >= "2009-01-01"])
            audit.append({"team": team, "provider_team": name, "status": "downloaded"})
        except (requests.RequestException, ValueError) as exc:
            # log per-team failures to keep partial coverage visible
            audit.append(
                {"team": team, "provider_team": name, "status": type(exc).__name__}
            )
        print(team, audit[-1]["status"], flush=True)
    pd.DataFrame(audit).to_csv(CACHE / "coverage.csv", index=False)
    if not frames:
        raise RuntimeError("No ClubElo histories downloaded; inspect coverage.csv")
    result = pd.concat(frames, ignore_index=True).drop_duplicates(["team", "date"])
    result.to_csv(CACHE / "history.csv", index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    download_clubelo(parser.parse_args().refresh)
