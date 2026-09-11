"""Match Footiqo's closing odds to our UCL results and report unmatched fixtures.

Source: https://footiqo.com/database/leagues/europe-champions-league/
"""

from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOOTIQO_ODDS_PATH = (
    ROOT / "src" / "data" / "raw" / "footiqo" / "ucl_closing_odds.csv"
)
DEFAULT_MASTER_MATCHES_PATH = (
    ROOT / "src" / "data" / "processed" / "matches_with_features.csv"
)
DEFAULT_TEAM_MAPPING_PATH = ROOT / "src" / "data" / "team_name_mapping.csv"


# names must match the results and feature files
FOOTIQO_TO_CANONICAL = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Chelsea": "Chelsea",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Manchester City": "Man City",
    "Manchester Utd": "Man United",
    "Newcastle": "Newcastle",
    "Tottenham": "Tottenham",
    "Ath Bilbao": "Ath Bilbao",
    "Atl. Madrid": "Ath Madrid",
    "Barcelona": "Barcelona",
    "Betis": "Betis",
    "Girona": "Girona",
    "Real Madrid": "Real Madrid",
    "Real Sociedad": "Sociedad",
    "Sevilla": "Sevilla",
    "Valencia": "Valencia",
    "Villarreal": "Villarreal",
    "B. Monchengladbach": "M'gladbach",
    "Bayer Leverkusen": "Leverkusen",
    "Bayern Munich": "Bayern Munich",
    "Dortmund": "Dortmund",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Hoffenheim": "Hoffenheim",
    "RB Leipzig": "RB Leipzig",
    "Schalke": "Schalke 04",
    "Stuttgart": "Stuttgart",
    "Union Berlin": "Union Berlin",
    "Wolfsburg": "Wolfsburg",
    "AC Milan": "Milan",
    "AS Roma": "Roma",
    "Atalanta": "Atalanta",
    "Bologna": "Bologna",
    "Inter": "Inter",
    "Juventus": "Juventus",
    "Lazio": "Lazio",
    "Napoli": "Napoli",
    "Brest": "Brest",
    "Lens": "Lens",
    "Lille": "Lille",
    "Lyon": "Lyon",
    "Marseille": "Marseille",
    "Monaco": "Monaco",
    "Paris SG": "Paris SG",
    "PSG": "Paris SG",
    "Rennes": "Rennes",
    "Benfica": "Benfica",
    "Braga": "Sp Braga",
    "FC Porto": "Porto",
    "Sporting CP": "Sp Lisbon",
    "Ajax": "Ajax",
    "Feyenoord": "Feyenoord",
    "PSV": "PSV Eindhoven",
    "Anderlecht": "Anderlecht",
    "Antwerp": "Antwerp",
    "Club Brugge": "Club Brugge",
    "Club Brugge KV": "Club Brugge",
    "Gent": "Gent",
    "Genk": "Genk",
    "Royale Union SG": "St. Gilloise",
    "Basaksehir": "Buyuksehyr",
    "Besiktas": "Besiktas",
    "Fenerbahce": "Fenerbahce",
    "Galatasaray": "Galatasaray",
    "Olympiacos Piraeus": "Olympiakos",
    "AEK Athens": "AEK",
    "AEK Athens FC": "AEK",
    "APOEL": "APOEL Nikosia (CYP)",
    "BATE": "BATE Borisov (BLR)",
    "Bodo/Glimt": "FK Bodø/Glimt (NOR)",
    "Celtic": "Celtic",
    "CSKA Moscow": "CSKA Moskva (RUS)",
    "Crvena zvezda": "Crvena Zvezda",
    "D. Zagreb": "Dinamo Zagreb",
    "Dinamo Kiev": "Dinamo Kiev (UKR)",
    "Dinamo Zagreb": "Dinamo Zagreb",
    "Dyn. Kyiv": "Dinamo Kiev (UKR)",
    "Ferencvaros": "Ferencvárosi TC (HUN)",
    "FC Astana": "FK Astana (KAZ)",
    "FK Astana": "FK Astana (KAZ)",
    "FK Kairat": "FK Kairat (KAZ)",
    "FK Krasnodar": "FK Krasnodar (RUS)",
    "FK Rostov": "FK Rostov (RUS)",
    "Kairat Almaty": "FK Kairat (KAZ)",
    "Krasnodar": "FK Krasnodar (RUS)",
    "LASK": "LASK",
    "Legia": "Legia Warszawa (POL)",
    "Lokomotiv Moscow": "Lokomotiv Moskva (RUS)",
    "Ludogorets": "PFC Ludogorets Razgrad (BUL)",
    "Maccabi Haifa": "Maccabi Haifa (ISR)",
    "Maccabi Tel Aviv": "Maccabi Tel Aviv (ISR)",
    "Malmo FF": "Malmö FF (SWE)",
    "Maribor": "NK Maribor (SVN)",
    "Midtjylland": "Midtjylland",
    "Pafos": "Paphos FC (CYP)",
    "Plzen": "Viktoria Plzeň (CZE)",
    "Qarabag": "Qarabag",
    "Rangers": "Rangers",
    "Salzburg": "Salzburg",
    "Sheriff": "FC Sheriff (MDA)",
    "Sheriff Tiraspol": "FC Sheriff (MDA)",
    "Shakhtar Donetsk": "Shakhtar Donetsk",
    "Slavia Prague": "Slavia Praha",
    "Slovan Bratislava": "ŠK Slovan Bratislava (SVK)",
    "Sparta Prague": "AC Sparta Praha (CZE)",
    "Spartak Moscow": "Spartak Moskva (RUS)",
    "Sturm Graz": "Sturm Graz",
    "Young Boys": "Young Boys",
    "Zenit": "Zenit St. Petersburg (RUS)",
    "Basel": "Basel",
    "FC Copenhagen": "FC Copenhagen",
}


def normalize_footiqo_team_name(team: str) -> Optional[str]:
    """Return our club name, or None so an unknown name shows up in the audit."""
    return FOOTIQO_TO_CANONICAL.get(team)
