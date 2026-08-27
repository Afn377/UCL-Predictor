"""Download additional Football-Data leagues, retaining original columns for odds."""

from io import StringIO

import pandas as pd

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
