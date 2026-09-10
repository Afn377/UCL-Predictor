"""Download the public UCL historical odds table, preserving source responses."""

import json
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://footiqo.com/database/leagues/europe-champions-league/"
OUTPUT = Path(__file__).parent / "raw" / "footiqo"


def download():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    response = session.get(URL, timeout=40)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    # page has several tables; only some expose closing odds
    candidates = []
    for element in soup.select("input[id$='_desc']"):
        config = json.loads(element["value"])
        params = config["dataTableParams"]
        names = [column["name"] for column in params["columnDefs"]]
        if "xbetClose1FT" in names:
            candidates.append((config, names))
    if not candidates:
        raise ValueError("Public odds table was not found")
    frames, counts = [], []
    (OUTPUT / "source.html").write_text(response.text)
    for config, names in candidates:
        params = config["dataTableParams"]
        payload = {
            "draw": 1,
            "start": 0,
            "length": -1,
            "search[value]": "",
            "search[regex]": "false",
        }
        nonce = soup.select_one(f"#wdtNonceFrontendServerSide_{config['tableWpId']}")
        if nonce is not None:
            payload["wdtNonce"] = nonce.get("value", "")
        for i, name in enumerate(names):
            # DataTables needs per-column metadata
            payload.update(
                {
                    f"columns[{i}][data]": i,
                    f"columns[{i}][name]": name,
                    f"columns[{i}][searchable]": "true",
                    f"columns[{i}][orderable]": "true",
                    f"columns[{i}][search][value]": "",
                    f"columns[{i}][search][regex]": "false",
                }
            )
        response = session.post(params["ajax"]["url"], data=payload, timeout=60)
        response.raise_for_status()
        (OUTPUT / f"{config['tableId']}_response.txt").write_text(response.text)
        result = response.json()
        (OUTPUT / f"{config['tableId']}.json").write_text(json.dumps(result, indent=2))
        rows = result.get("data", result.get("aaData"))
        if rows is None:
            raise ValueError(f"Unexpected table response: {list(result)}")
        expected = int(
            result.get("recordsFiltered", result.get("iTotalDisplayRecords", len(rows)))
        )
        # catch silent first-page-only exports
        if len(rows) != expected:
            raise ValueError(f"Incomplete export: {len(rows)} of {expected} rows")
        frame = pd.DataFrame(rows)
        if frame.shape[1] != len(names):
            raise ValueError("Source column count changed")
        frame.columns = names
        frame = frame.map(
            lambda value: BeautifulSoup(str(value), "html.parser").get_text(strip=True)
        )
        frames.append(frame)
        counts.append({"table": config["tableId"], "rows": len(frame)})
    raw = pd.concat(frames, ignore_index=True)
    if raw.id.duplicated().any():
        raise ValueError("Overlapping source match IDs")
    raw.to_csv(OUTPUT / "ucl_odds_original.csv", index=False)
    normalized = raw.rename(
        columns={
            "id": "source_match_id",
            "homeTeam": "home_team_raw",
            "awayTeam": "away_team_raw",
            "xbetClose1FT": "odds_home",
            "xbetCloseXFT": "odds_draw",
            "xbetClose2FT": "odds_away",
        }
    ).copy()
    normalized["date"] = pd.to_datetime(
        raw.matchDate, format="%d-%m-%y %H:%M", errors="raise"
    ).dt.normalize()
    normalized["competition"] = "Champions League"
    for column in ["odds_home", "odds_draw", "odds_away"]:
        normalized[column] = pd.to_numeric(
            normalized[column].replace({"": None, "-": None}), errors="coerce"
        )
    normalized["source"] = "footiqo_1xbet_closing"
    normalized["snapshot_at"] = pd.NaT
    normalized.to_csv(OUTPUT / "ucl_closing_odds.csv", index=False)
    manifest = {
        "url": URL,
        "downloaded_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "tables": counts,
        "rows": len(raw),
        "timing": "closing; quote timestamp unavailable",
        "usage": "Benchmark only until fixture matching and source validation are complete",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    download()
