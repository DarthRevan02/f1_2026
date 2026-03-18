# ============================================================
# grid_normalizer.py — Remap historical data to 2026 grid
# ============================================================
# Correct 2026 grid changes vs original:
#
#   RED BULL    : VER + HAD  (Hadjar promoted from Racing Bulls)
#   RACING BULLS: LAW + LIN  (Lindblad replaces Hadjar — only rookie)
#   AUDI        : HUL + BOR  (Hulkenberg, NOT Bottas)
#   CADILLAC    : BOT + PER  (Bottas + Perez returning after year out)
#   (DOO/MAL removed — Doohan/Armstrong not on 2026 grid)
# ============================================================

# ── Complete 2026 driver → team mapping ───────────────────
DRIVER_2026_TEAM = {
    # Red Bull Racing  (Honda RBPT power unit)
    "VER": "Red Bull Racing",
    "HAD": "Red Bull Racing",      # Isack Hadjar — promoted from Racing Bulls

    # Ferrari  (Ferrari power unit)
    "HAM": "Ferrari",              # moved from Mercedes
    "LEC": "Ferrari",

    # McLaren  (Mercedes power unit)
    "NOR": "McLaren",
    "PIA": "McLaren",

    # Mercedes  (Mercedes power unit)
    "ANT": "Mercedes",             # Andrea Kimi Antonelli — sophomore season
    "RUS": "Mercedes",

    # Aston Martin  (Honda power unit)
    "ALO": "Aston Martin",
    "STR": "Aston Martin",

    # Alpine  (Mercedes power unit — switched from Renault)
    "GAS": "Alpine",
    "COL": "Alpine",               # Franco Colapinto

    # Williams  (Mercedes power unit)
    "ALB": "Williams",
    "SAI": "Williams",             # moved from Ferrari

    # Haas  (Ferrari power unit)
    "BEA": "Haas",                 # Oliver Bearman
    "OCO": "Haas",                 # moved from Alpine

    # Racing Bulls  (Honda RBPT power unit)
    "LAW": "Racing Bulls",
    "LIN": "Racing Bulls",         # Arvid Lindblad — only rookie on 2026 grid

    # Audi  (Audi power unit — new manufacturer)
    "HUL": "Audi",                 # Nico Hulkenberg — moved from Haas/Kick Sauber
    "BOR": "Audi",                 # Gabriel Bortoleto

    # Cadillac  (Ferrari power unit — new team)
    "BOT": "Cadillac",             # Valtteri Bottas — returning after year out
    "PER": "Cadillac",             # Sergio Perez — returning after year out
}

# ── Historical team name aliases → canonical 2026 name ────
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
    "Renault":                      "Alpine",

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

    # Cadillac (new team)
    "Cadillac":                     "Cadillac",
    "Andretti":                     "Cadillac",
    "Andretti Global":              "Cadillac",
}

# ── Drivers with NO (or very limited) historical F1 race data ─
# Synthetic baseline gap-to-pole values derived from junior career.
#
# Tiers:
#   "top"      : F2 champion / dominant season     → ~1.2s avg gap
#   "strong"   : F2 podium regular / top 3         → ~1.6s avg gap
#   "midfield" : Limited F1 / first full season    → ~2.2s avg gap
#   "veteran"  : Experienced F1 driver returning   → uses real history
NEW_DRIVER_BASELINES = {
    "ANT": {  # Kimi Antonelli — 1 F1 season (2025), strong rookie
        "tier":                "strong",
        "driver_avg_gap_hist": 1.45,
        "driver_consistency":  0.46,
        "driver_wet_skill":    0.05,
        "driver_dnf_rate":     0.06,
    },
    "HAD": {  # Isack Hadjar — F2 2024 champion, promoted to Red Bull
        "tier":                "top",
        "driver_avg_gap_hist": 1.20,
        "driver_consistency":  0.40,
        "driver_wet_skill":    0.04,
        "driver_dnf_rate":     0.05,
    },
    "BOR": {  # Gabriel Bortoleto — F2 2024 champion
        "tier":                "top",
        "driver_avg_gap_hist": 1.20,
        "driver_consistency":  0.40,
        "driver_wet_skill":    0.04,
        "driver_dnf_rate":     0.05,
    },
    "BEA": {  # Oliver Bearman — partial 2024 appearances + full 2025
        "tier":                "strong",
        "driver_avg_gap_hist": 1.50,
        "driver_consistency":  0.50,
        "driver_wet_skill":    0.02,
        "driver_dnf_rate":     0.07,
    },
    "LIN": {  # Arvid Lindblad — only rookie, Red Bull junior, F2 2025
        "tier":                "top",
        "driver_avg_gap_hist": 1.30,
        "driver_consistency":  0.45,
        "driver_wet_skill":    0.03,
        "driver_dnf_rate":     0.08,
    },
    "COL": {  # Franco Colapinto — partial 2024 Williams + 2025 Alpine
        "tier":                "strong",
        "driver_avg_gap_hist": 1.65,
        "driver_consistency":  0.55,
        "driver_wet_skill":    0.02,
        "driver_dnf_rate":     0.09,
    },
}

# ── Key team/driver changes for 2026 ──────────────────────
TEAM_CHANGES_2026 = {
    "HAM": {"from": "Mercedes",           "to": "Ferrari"},
    "SAI": {"from": "Ferrari",            "to": "Williams"},
    "OCO": {"from": "Alpine",             "to": "Haas"},
    "HAD": {"from": "Racing Bulls",       "to": "Red Bull Racing"},
    "LAW": {"from": "Red Bull Racing",    "to": "Racing Bulls"},
    "HUL": {"from": "Haas / Kick Sauber", "to": "Audi"},
    "BOT": {"from": "Kick Sauber",        "to": "Cadillac"},
    "PER": {"from": "Red Bull Racing",    "to": "Cadillac"},
    "COL": {"from": "Williams",           "to": "Alpine"},
    "BEA": {"from": "Haas (partial)",     "to": "Haas"},
    "ANT": {"from": "none (rookie→soph)", "to": "Mercedes"},
    "LIN": {"from": "none (rookie)",      "to": "Racing Bulls"},
    "BOR": {"from": "none (rookie)",      "to": "Audi"},
}


import pandas as pd


def normalize_teams(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remap Team column so all historical and 2026 data uses
    canonical 2026 constructor names consistently.
    """
    original_teams = df["Team"].nunique()

    # Step 1: normalise team name spelling variants
    df["Team"] = df["Team"].map(TEAM_NAME_ALIASES).fillna(df["Team"])

    # Step 2: remap each driver's team to their 2026 team
    df["Team"] = df.apply(
        lambda row: DRIVER_2026_TEAM.get(row["Driver"], row["Team"]),
        axis=1
    )

    remapped = df["Team"].nunique()
    print(f"  Team normalization: {original_teams} variants → {remapped} canonical 2026 teams")

    for driver, change in TEAM_CHANGES_2026.items():
        if driver in df["Driver"].values and change["from"] != change["to"]:
            count = (df["Driver"] == driver).sum()
            print(f"    {driver}: {change['from']} → {change['to']}  ({count} laps remapped)")

    return df


def inject_new_driver_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject synthetic baseline rows for drivers with no / minimal
    historical F1 data. Replaced rapidly once 2026 race data comes in.
    """
    new_rows = []
    circuits = df["circuit"].dropna().unique()[:5]

    for driver, baseline in NEW_DRIVER_BASELINES.items():
        if driver in df["Driver"].values:
            continue  # already has real data

        team = DRIVER_2026_TEAM.get(driver, "Unknown")
        print(f"  Injecting baseline for new driver: {driver} ({team}) — tier: {baseline['tier']}")

        for circuit in circuits:
            new_rows.append({
                "Driver":              driver,
                "Team":                team,
                "year":                2024,
                "circuit":             circuit,
                "LapTime_s":           90.0 + baseline["driver_avg_gap_hist"],
                "gap_to_pole":         baseline["driver_avg_gap_hist"],
                "driver_avg_gap_hist": baseline["driver_avg_gap_hist"],
                "driver_consistency":  baseline["driver_consistency"],
                "driver_wet_skill":    baseline["driver_wet_skill"],
                "driver_dnf_rate":     baseline["driver_dnf_rate"],
                "is_synthetic":        1,
            })

    if new_rows:
        synthetic_df = pd.DataFrame(new_rows)
        df = pd.concat([df, synthetic_df], ignore_index=True)
        print(f"  → {len(new_rows)} synthetic rows injected for {len(NEW_DRIVER_BASELINES)} drivers")
    else:
        print("  → All 2026 drivers already have historical data")

    return df


def filter_to_2026_grid_only(df: pd.DataFrame) -> pd.DataFrame:
    """Remove laps from drivers not on the 2026 grid."""
    grid_drivers = set(DRIVER_2026_TEAM.keys())
    before = len(df)
    df = df[df["Driver"].isin(grid_drivers)].copy()
    after = len(df)
    print(f"  Grid filter: {before} → {after} laps "
          f"({before - after} laps removed from non-2026 drivers)")
    return df


def print_grid_audit():
    """Print a human-readable audit of the 2026 grid."""
    print("\n" + "=" * 60)
    print("  2026 F1 Grid — Driver → Team Mapping")
    print("=" * 60)

    teams = {}
    for driver, team in DRIVER_2026_TEAM.items():
        teams.setdefault(team, []).append(driver)

    for team, drivers in sorted(teams.items()):
        print(f"  {team:<25} {' / '.join(drivers)}")

    print("\n  Key changes from 2025 → 2026:")
    for driver, change in TEAM_CHANGES_2026.items():
        if change["from"] != change["to"] and "rookie" not in change["from"]:
            print(f"    {driver}: {change['from']} → {change['to']}")

    print("\n  New/synthetic baseline drivers:")
    for driver, baseline in NEW_DRIVER_BASELINES.items():
        team = DRIVER_2026_TEAM.get(driver, "?")
        print(f"    {driver} ({team}) — tier: {baseline['tier']}, "
              f"baseline gap: {baseline['driver_avg_gap_hist']}s")
    print("=" * 60 + "\n")
