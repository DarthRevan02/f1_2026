# ============================================================
# grid_normalizer.py — Remap historical data to 2026 grid
# ============================================================
# Problem: historical data (2022–2024) has drivers at their
# OLD teams (e.g. Hamilton at Mercedes, Sainz at Ferrari).
# The model must learn driver SKILL independent of team,
# and team PERFORMANCE independent of which driver was there.
#
# This module does two things:
#
#   1. DRIVER COLUMN  → always the driver's abbreviation (unchanged)
#      Driver skill stats are computed from their own lap times,
#      regardless of which car they were in. This is correct.
#
#   2. TEAM COLUMN    → remapped to the 2026 team that driver
#      NOW races for, so the model's "team_encoded" feature
#      reflects the 2026 constructor, not a historical one.
#
# Additionally, drivers who are NEW to F1 in 2026 (no historical
# data) are handled via synthetic baselines derived from their
# junior career / F2 performance tier.
# ============================================================

# ── Complete 2026 driver → team mapping ───────────────────
# This is the single source of truth for the 2026 grid.
# Update here if any mid-season driver changes happen.
DRIVER_2026_TEAM = {
    # Red Bull Racing  (Honda RBPT power unit)
    "VER": "Red Bull Racing",
    "LAW": "Red Bull Racing",

    # Ferrari  (Ferrari power unit)
    "HAM": "Ferrari",          # moved from Mercedes
    "LEC": "Ferrari",

    # McLaren  (Mercedes power unit)
    "NOR": "McLaren",
    "PIA": "McLaren",

    # Mercedes  (Mercedes power unit)
    "ANT": "Mercedes",         # Andrea Kimi Antonelli — rookie
    "RUS": "Mercedes",

    # Aston Martin  (Honda power unit)
    "ALO": "Aston Martin",
    "STR": "Aston Martin",

    # Alpine  (Renault power unit)
    "GAS": "Alpine",
    "COL": "Alpine",           # Franco Colapinto — moved from Williams

    # Williams  (Mercedes power unit)
    "ALB": "Williams",
    "SAI": "Williams",         # moved from Ferrari

    # Haas  (Ferrari power unit)
    "BEA": "Haas",             # Oliver Bearman — promoted
    "OCO": "Haas",             # moved from Alpine

    # Racing Bulls  (Honda RBPT power unit)
    "TSU": "Racing Bulls",
    "HAD": "Racing Bulls",     # Isack Hadjar — rookie

    # Audi  (Audi power unit — new manufacturer)
    "BOT": "Audi",             # moved from Alfa Romeo / Kick Sauber
    "BOR": "Audi",             # Nico Hülkenberg moved; Gabriel Bortoleto — rookie

    # Cadillac  (GM/Cadillac — new team & manufacturer)
    "DOO": "Cadillac",         # Jack Doohan
    "MAL": "Cadillac",         # Marcus Armstrong (placeholder)
}

# ── Historical team name aliases → canonical 2026 name ────
# FastF1 uses inconsistent team names across years.
# Map all historical variants to the closest 2026 equivalent
# so team_encoded is consistent across the full training set.
TEAM_NAME_ALIASES = {
    # Mercedes works team
    "Mercedes":                     "Mercedes",
    "Mercedes AMG":                 "Mercedes",
    "Mercedes-AMG Petronas":        "Mercedes",
    "Mercedes-AMG Petronas F1 Team":"Mercedes",

    # Ferrari
    "Ferrari":                      "Ferrari",
    "Scuderia Ferrari":             "Ferrari",
    "Scuderia Ferrari HP":          "Ferrari",

    # Red Bull
    "Red Bull Racing":              "Red Bull Racing",
    "Red Bull":                     "Red Bull Racing",
    "Oracle Red Bull Racing":       "Red Bull Racing",

    # McLaren
    "McLaren":                      "McLaren",
    "McLaren F1 Team":              "McLaren",

    # Aston Martin
    "Aston Martin":                 "Aston Martin",
    "Aston Martin Aramco":          "Aston Martin",
    "Aston Martin Aramco Cognizant":"Aston Martin",
    "Aston Martin F1 Team":         "Aston Martin",

    # Alpine / Renault
    "Alpine":                       "Alpine",
    "Alpine F1 Team":               "Alpine",
    "Alpine F1":                    "Alpine",
    "Renault":                      "Alpine",          # pre-2021 name

    # Williams
    "Williams":                     "Williams",
    "Williams Racing":              "Williams",

    # Haas
    "Haas":                         "Haas",
    "Haas F1 Team":                 "Haas",
    "MoneyGram Haas F1 Team":       "Haas",

    # Racing Bulls / AlphaTauri / Toro Rosso
    "Racing Bulls":                 "Racing Bulls",
    "RB":                           "Racing Bulls",
    "AlphaTauri":                   "Racing Bulls",
    "Scuderia AlphaTauri":          "Racing Bulls",
    "Toro Rosso":                   "Racing Bulls",
    "Scuderia Toro Rosso":          "Racing Bulls",
    "Visa Cash App RB":             "Racing Bulls",

    # Audi (was Alfa Romeo / Sauber / Kick Sauber)
    "Audi":                         "Audi",
    "Sauber":                       "Audi",
    "Alfa Romeo":                   "Audi",
    "Alfa Romeo Racing":            "Audi",
    "Alfa Romeo F1 Team ORLEN":     "Audi",
    "Kick Sauber":                  "Audi",
    "Stake F1 Team Kick Sauber":    "Audi",

    # Cadillac (new team — no historical data)
    "Cadillac":                     "Cadillac",
    "Andretti":                     "Cadillac",
    "Andretti Global":              "Cadillac",
}

# ── Drivers with NO historical F1 race data ───────────────
# These drivers have no laps in 2022–2024 FastF1 data.
# We assign synthetic baseline gap-to-pole values derived
# from their F2 championship performance tier.
#
# Tiers (based on F2 / junior series results):
#   "top"    : F2 champion / dominant season     → 1.2s avg gap
#   "strong" : F2 podium regular / top 3         → 1.6s avg gap
#   "midfield": F2 mid-table / first season      → 2.2s avg gap
#
# These are deliberately conservative — the model will
# update them rapidly once 2026 race data comes in.
NEW_DRIVER_BASELINES = {
    "ANT": {  # Andrea Kimi Antonelli — F2 2024 strong season
        "tier":                "strong",
        "driver_avg_gap_hist": 1.55,
        "driver_consistency":  0.48,
        "driver_wet_skill":    0.05,
        "driver_dnf_rate":     0.06,
    },
    "HAD": {  # Isack Hadjar — F2 2024 champion
        "tier":                "top",
        "driver_avg_gap_hist": 1.25,
        "driver_consistency":  0.42,
        "driver_wet_skill":    0.03,
        "driver_dnf_rate":     0.05,
    },
    "BOR": {  # Gabriel Bortoleto — F2 2024 champion
        "tier":                "top",
        "driver_avg_gap_hist": 1.20,
        "driver_consistency":  0.40,
        "driver_wet_skill":    0.04,
        "driver_dnf_rate":     0.05,
    },
    "BEA": {  # Oliver Bearman — partial 2024 F1 appearances, F2 runner-up
        "tier":                "strong",
        "driver_avg_gap_hist": 1.50,
        "driver_consistency":  0.50,
        "driver_wet_skill":    0.02,
        "driver_dnf_rate":     0.07,
    },
    "DOO": {  # Jack Doohan — F2 podiums, Alpine reserve
        "tier":                "strong",
        "driver_avg_gap_hist": 1.65,
        "driver_consistency":  0.55,
        "driver_wet_skill":    0.01,
        "driver_dnf_rate":     0.08,
    },
    "MAL": {  # Marcus Armstrong / Cadillac seat placeholder
        "tier":                "midfield",
        "driver_avg_gap_hist": 2.10,
        "driver_consistency":  0.65,
        "driver_wet_skill":    0.00,
        "driver_dnf_rate":     0.10,
    },
    "COL": {  # Franco Colapinto — partial 2024 Williams races
        "tier":                "strong",
        "driver_avg_gap_hist": 1.70,
        "driver_consistency":  0.58,
        "driver_wet_skill":    0.02,
        "driver_dnf_rate":     0.09,
    },
}

# ── Drivers who changed teams for 2026 ────────────────────
# For reference / audit — these are the key moves.
# The normalizer remaps their historical Team column to their
# 2026 team so team_encoded is consistent.
TEAM_CHANGES_2026 = {
    "HAM": {"from": "Mercedes",       "to": "Ferrari"},
    "SAI": {"from": "Ferrari",        "to": "Williams"},
    "OCO": {"from": "Alpine",         "to": "Haas"},
    "BOT": {"from": "Kick Sauber",    "to": "Audi"},
    "TSU": {"from": "Racing Bulls",   "to": "Racing Bulls"},  # stayed
    "COL": {"from": "Williams",       "to": "Alpine"},
    "BEA": {"from": "Haas (reserve)", "to": "Haas"},
    "ANT": {"from": "none (rookie)",  "to": "Mercedes"},
    "HAD": {"from": "none (rookie)",  "to": "Racing Bulls"},
    "BOR": {"from": "none (rookie)",  "to": "Audi"},
    "DOO": {"from": "Alpine",         "to": "Cadillac"},
}


import pandas as pd


def normalize_teams(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remap the Team column in historical training data so that:
    - All team name variants → canonical 2026 constructor name
    - Each driver's team → their 2026 team (not their historical team)

    This ensures team_encoded always represents the 2026 constructor,
    making historical and 2026 data directly comparable.

    Called in 2_feature_engineering.py before encoding.
    """
    original_teams = df["Team"].nunique()

    # Step 1: normalize team name spelling variants
    df["Team"] = df["Team"].map(TEAM_NAME_ALIASES).fillna(df["Team"])

    # Step 2: remap each driver's team to their 2026 team
    # This handles Hamilton's laps at Mercedes → now labelled Ferrari, etc.
    df["Team"] = df.apply(
        lambda row: DRIVER_2026_TEAM.get(row["Driver"], row["Team"]),
        axis=1
    )

    remapped = df["Team"].nunique()
    print(f"  Team normalization: {original_teams} variants → {remapped} canonical 2026 teams")

    # Audit: show which driver→team remappings were applied
    for driver, change in TEAM_CHANGES_2026.items():
        if driver in df["Driver"].values and change["from"] != change["to"]:
            count = (df["Driver"] == driver).sum()
            print(f"    {driver}: {change['from']} → {change['to']}  ({count} laps remapped)")

    return df


def inject_new_driver_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """
    For drivers with no historical F1 data (rookies / new entrants),
    inject synthetic baseline stats so they appear in the training set
    with reasonable prior values instead of NaN.

    Creates one synthetic 'lap' row per rookie per circuit so they
    appear in the encoder and get a valid driver_encoded value.
    The model will rapidly override these with real 2026 data.

    Called in 2_feature_engineering.py after driver baselines are computed.
    """
    new_rows = []
    circuits = df["circuit"].dropna().unique()[:5]  # use first 5 circuits as anchors

    for driver, baseline in NEW_DRIVER_BASELINES.items():
        if driver in df["Driver"].values:
            continue  # already has real data (e.g. Bearman's 2024 appearances)

        team = DRIVER_2026_TEAM.get(driver, "Unknown")
        print(f"  Injecting baseline for new driver: {driver} ({team}) — tier: {baseline['tier']}")

        for circuit in circuits:
            new_rows.append({
                "Driver":              driver,
                "Team":                team,
                "year":                2024,     # attach to last historical year
                "circuit":             circuit,
                "LapTime_s":           90.0 + baseline["driver_avg_gap_hist"],
                "gap_to_pole":         baseline["driver_avg_gap_hist"],
                "driver_avg_gap_hist": baseline["driver_avg_gap_hist"],
                "driver_consistency":  baseline["driver_consistency"],
                "driver_wet_skill":    baseline["driver_wet_skill"],
                "driver_dnf_rate":     baseline["driver_dnf_rate"],
                "is_synthetic":        1,  # flag so we can audit/remove later
            })

    if new_rows:
        synthetic_df = pd.DataFrame(new_rows)
        df = pd.concat([df, synthetic_df], ignore_index=True)
        print(f"  → {len(new_rows)} synthetic rows injected for {len(NEW_DRIVER_BASELINES)} new drivers")
    else:
        print("  → All 2026 drivers already have historical data")

    return df


def filter_to_2026_grid_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove any laps from drivers who are NOT on the 2026 grid.
    This keeps the training data strictly relevant — no Bottas,
    no Ricciardo, no Magnussen, no retired drivers.

    Called in 2_feature_engineering.py after normalization.
    """
    grid_drivers = set(DRIVER_2026_TEAM.keys())
    before = len(df)
    df = df[df["Driver"].isin(grid_drivers)].copy()
    after  = len(df)
    removed_drivers = set(df["Driver"].unique()) - grid_drivers
    print(f"  Grid filter: {before} → {after} laps "
          f"({before - after} laps removed from non-2026 drivers)")
    return df


def print_grid_audit():
    """Print a human-readable audit of the 2026 grid and team changes."""
    print("\n" + "="*60)
    print("  2026 F1 Grid — Driver → Team Mapping")
    print("="*60)

    teams = {}
    for driver, team in DRIVER_2026_TEAM.items():
        teams.setdefault(team, []).append(driver)

    for team, drivers in sorted(teams.items()):
        print(f"  {team:<25} {' / '.join(drivers)}")

    print("\n  Key team changes from 2025 → 2026:")
    for driver, change in TEAM_CHANGES_2026.items():
        if change["from"] != change["to"] and "rookie" not in change["from"]:
            print(f"    {driver}: {change['from']} → {change['to']}")

    print("\n  New drivers (synthetic baselines):")
    for driver, baseline in NEW_DRIVER_BASELINES.items():
        team = DRIVER_2026_TEAM.get(driver, "?")
        print(f"    {driver} ({team}) — tier: {baseline['tier']}, "
              f"baseline gap: {baseline['driver_avg_gap_hist']}s")
    print("="*60 + "\n")
