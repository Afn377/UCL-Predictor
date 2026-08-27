"""Download additional Football-Data leagues, retaining original columns for odds."""

import argparse
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RAW_DIR = Path(__file__).resolve().parent / "raw"

LEAGUES = {
    "primeira-liga": "P1",
    "eredivisie": "N1",
    "belgian-pro-league": "B1",
    "scottish-premiership": "SC0",
    "super-lig": "T1",
    "greek-super-league": "G1",
}
EXTRA = {
    "austrian-bundesliga": "AUT",
    "swiss-super-league": "SWZ",
    "danish-superliga": "DNK",
}


def read_download(content):
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        # some older exports use windows encoding
        text = content.decode("cp1252")
    return pd.read_csv(StringIO(text))


def normalize_extra(raw):
    columns = {
        "Date": "Date",
        "Home": "HomeTeam",
        "Away": "AwayTeam",
        "HG": "FTHG",
        "AG": "FTAG",
        "Res": "FTR",
        "Season": "Season",
    }
    if set(columns) - set(raw):
        raise ValueError("Extra-league download lacks result columns")
    data = raw.rename(columns=columns).copy()
    data["Date"] = pd.to_datetime(data.Date, dayfirst=True, errors="raise").dt.strftime(
        "%d/%m/%Y"
    )
    return data


def download_domestic(start=2015, end=2025, refresh=False):
    # retry transient errors; this loop hits many season files
    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=3,
                backoff_factor=2,
                status_forcelist=[429, 500, 502, 503, 504],
            )
        ),
    )
    for directory, code in LEAGUES.items():
        for year in range(start, end + 1):
            path = RAW_DIR / directory / f"{year}_{str(year + 1)[2:]}.csv"
            if path.exists() and not refresh:
                continue
            url = f"https://football-data.co.uk/mmz4281/{str(year)[2:]}{str(year + 1)[2:]}/{code}.csv"
            response = session.get(url, timeout=20)
            response.raise_for_status()
            frame = read_download(response.content)
            if {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"} - set(frame):
                raise ValueError(f"Invalid result download: {url}")
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
            print(f"Downloaded {path}: {len(frame)} rows", flush=True)
            time.sleep(0.3)
    for directory, code in EXTRA.items():
        # these leagues ship as one combined per-country export
        if not refresh and all(
            (RAW_DIR / directory / f"{y}_{str(y+1)[2:]}.csv").exists()
            for y in range(start, end + 1)
        ):
            continue
        response = session.get(
            f"https://football-data.co.uk/new/{code}.csv", timeout=20
        )
        response.raise_for_status()
        frame = normalize_extra(read_download(response.content))
        for season, data in frame.groupby("Season"):
            year = int(str(season)[:4])
            if not start <= year <= end:
                continue
            path = RAW_DIR / directory / f"{year}_{str(year+1)[2:]}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(path, index=False)
            print(f"Downloaded {path}: {len(data)} rows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    download_domestic(args.start_year, args.end_year, args.refresh)
