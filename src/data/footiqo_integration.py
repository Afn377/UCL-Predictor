"""Match Footiqo's closing odds to our UCL results and report unmatched fixtures.

Source: https://footiqo.com/database/leagues/europe-champions-league/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

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


def load_footiqo_odds(path: Path = DEFAULT_FOOTIQO_ODDS_PATH) -> pd.DataFrame:
    """Read the odds export, match club names, and remove the bookmaker margin."""
    # keep season as text so 2020/2021 isn't read as a number
    odds = pd.read_csv(path, dtype={"Season": str})

    # results key on date; drop the source's time of day
    odds["date"] = pd.to_datetime(odds["matchDate"], format="%d-%m-%y %H:%M").dt.date

    odds["season"] = (
        odds["Season"].str.split("/").str[0]
        + "_"
        + odds["Season"].str.split("/").str[1].str[-2:]
    )

    # implied probs carry the bookmaker margin; rescale to sum 1
    home_prob = 1 / odds["odds_home"]
    draw_prob = 1 / odds["odds_draw"]
    away_prob = 1 / odds["odds_away"]
    total = home_prob + draw_prob + away_prob

    odds["closing_prob_0"] = away_prob / total  # away win
    odds["closing_prob_1"] = draw_prob / total  # draw
    odds["closing_prob_2"] = home_prob / total  # home win
    odds["closing_source"] = "footiqo_1xbet"

    odds["home_team"] = odds["home_team_raw"].apply(normalize_footiqo_team_name)
    odds["away_team"] = odds["away_team_raw"].apply(normalize_footiqo_team_name)

    # future fixtures have no result; can't enter a backtest
    odds = odds[odds["season"] != "2026_27"].copy()

    return odds[
        [
            "date",
            "season",
            "home_team",
            "away_team",
            "closing_prob_0",
            "closing_prob_1",
            "closing_prob_2",
            "closing_source",
            "home_team_raw",
            "away_team_raw",
            "odds_home",
            "odds_draw",
            "odds_away",
        ]
    ].reset_index(drop=True)


def validate_and_join_footiqo_odds(
    footiqo_odds: pd.DataFrame,
    master_matches: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Join known clubs to results and keep unmatched fixtures for inspection."""
    audit = {
        "total_footiqo_matches": len(footiqo_odds),
        "unmapped_teams": set(),
        "matched_fixtures": 0,
        "unmatched_fixtures": [],
        "ambiguous_fixtures": [],
        "duplicate_fixtures": [],
    }

    unmapped_home = footiqo_odds[footiqo_odds["home_team"].isna()][
        "home_team_raw"
    ].unique()
    unmapped_away = footiqo_odds[footiqo_odds["away_team"].isna()][
        "away_team_raw"
    ].unique()
    audit["unmapped_teams"] = set(unmapped_home) | set(unmapped_away)

    # don't guess a club when its name isn't in the mapping
    valid_odds = footiqo_odds.dropna(subset=["home_team", "away_team"]).copy()

    matches = master_matches.copy()
    matches["date"] = pd.to_datetime(matches["date"]).dt.date
    matches["season"] = matches["season"].astype(str)

    if "competition" in matches.columns:
        matches = matches[matches["competition"] == "Champions League"]

    # left join keeps odds rows still needing a manual mapping
    joined = valid_odds.merge(
        matches[["date", "season", "home_team", "away_team"]],
        on=["date", "season", "home_team", "away_team"],
        how="left",
        indicator=True,
    )

    audit["matched_fixtures"] = (joined["_merge"] == "both").sum()
    unmatched = joined[joined["_merge"] == "left_only"]
    audit["unmatched_fixtures"] = unmatched[
        [
            "date",
            "season",
            "home_team_raw",
            "away_team_raw",
            "odds_home",
            "odds_draw",
            "odds_away",
        ]
    ].to_dict("records")

    # duplicate odds are ambiguous, so flag them in the audit
    duplicates = valid_odds[
        valid_odds.duplicated(
            subset=["date", "season", "home_team", "away_team"], keep=False
        )
    ]
    if len(duplicates) > 0:
        audit["duplicate_fixtures"] = duplicates[
            ["date", "season", "home_team_raw", "away_team_raw", "odds_home"]
        ].to_dict("records")

    return joined.drop("_merge", axis=1), audit


def write_footiqo_integration_report(
    audit: dict,
    output_path: Path = ROOT
    / "src"
    / "data"
    / "processed"
    / "footiqo_integration_audit.md",
) -> None:
    """Write the matching problems to a small report we can inspect."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = f"""# Footiqo UCL Closing Odds Integration Audit

## Summary
- Total Footiqo matches (2015/16-2025/26): {audit['total_footiqo_matches']}
- Successfully matched with master matches: {audit['matched_fixtures']}
- Unmatched fixtures: {len(audit['unmatched_fixtures'])}
- Duplicate fixtures in source: {len(audit['duplicate_fixtures'])}
- Unmapped team names: {len(audit['unmapped_teams'])}

## Unmapped Teams
These Footiqo team names could not be matched to canonical names:
"""

    if audit["unmapped_teams"]:
        # sort so report diffs stay stable
        for team in sorted(audit["unmapped_teams"]):
            report += f"\n- {team}"
    else:
        report += "\nNone - all teams successfully mapped."

    if audit["unmatched_fixtures"]:
        # cap the preview so one issue can't bloat the report
        report += f"\n\n## Unmatched Fixtures ({len(audit['unmatched_fixtures'])})\n"
        report += "\nThese fixtures have Footiqo odds but no corresponding entry in master matches:\n\n"
        for fixture in sorted(
            audit["unmatched_fixtures"], key=lambda fixture: fixture["date"]
        )[:20]:
            report += f"- {fixture['date']}: {fixture['home_team_raw']} vs {fixture['away_team_raw']} "
            report += f"({fixture['odds_home']:.2f}/{fixture['odds_draw']:.2f}/{fixture['odds_away']:.2f})\n"
        if len(audit["unmatched_fixtures"]) > 20:
            report += f"\n... and {len(audit['unmatched_fixtures']) - 20} more\n"

    if audit["duplicate_fixtures"]:
        report += f"\n## Duplicate Fixtures ({len(audit['duplicate_fixtures'])})\n"
        report += "\nThese fixtures appear more than once in Footiqo data:\n\n"
        for fixture in audit["duplicate_fixtures"][:10]:
            report += f"- {fixture['date']}: {fixture['home_team_raw']} vs {fixture['away_team_raw']}\n"

    with open(output_path, "w") as report_file:
        report_file.write(report)

    print(f"Audit report written to {output_path}")


if __name__ == "__main__":
    odds = load_footiqo_odds()
    print(f"Loaded {len(odds)} Footiqo odds records")

    matches = pd.read_csv(DEFAULT_MASTER_MATCHES_PATH)
    joined, audit = validate_and_join_footiqo_odds(odds, matches)

    print("\nIntegration Results:")
    print(f"  Matched: {audit['matched_fixtures']}/{audit['total_footiqo_matches']}")
    print(f"  Unmapped teams: {len(audit['unmapped_teams'])}")
    if audit["unmapped_teams"]:
        print(f"  Unmapped: {', '.join(sorted(audit['unmapped_teams'])[:5])}")

    write_footiqo_integration_report(audit)
