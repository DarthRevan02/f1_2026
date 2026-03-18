# ============================================================
# reliability.py — Engine & mechanical reliability scoring
# ============================================================
# This module computes reliability scores used as features
# in the lap time / gap-to-pole prediction model.
#
# Three scores are computed:
#
#   1. car_reliability_score  (team)
#      Rolling DNF rate for the team over the past N races.
#      Lower = more reliable. Used as a continuous feature.
#
#   2. driver_dnf_rate  (driver)
#      Driver's personal DNF rate (includes crashes + mech).
#      Separate from team score — some drivers bin it more.
#
#   3. engine_failure_prob  (team × circuit_type)
#      Probability of engine-related retirement based on:
#        - team's engine DNF history
#        - circuit thermal / power demand classification
#      Used in 5_predict_top10.py to apply a DNF risk penalty.
#
# All scores are rolling windows so they update as 2026 data
# comes in — early season = rely on 2022–2024 baseline,
# mid/late season = 2026 data dominates.
# ============================================================

import pandas as pd
import numpy as np
import fastf1
import os
import joblib

from config import SEASONS, CACHE_DIR, F1_2026_CALENDAR, F1_2026_DRIVERS

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

RELIABILITY_PATH     = "./data/reliability_scores.csv"
DNF_RAW_PATH         = "./data/dnf_raw.csv"

# ── Circuit power / thermal demand classification ─────────
# Higher demand = more stress on PU = higher engine failure risk
# Scale: 1 (low) → 3 (high)
CIRCUIT_PU_DEMAND = {
    "Australia":     2, "China":        2, "Japan":        3,
    "Bahrain":       3, "Saudi Arabia": 3, "Miami":        2,
    "Canada":        2, "Monaco":       1, "Barcelona":    2,
    "Austria":       3, "Great Britain":3, "Belgium":      3,
    "Hungary":       1, "Netherlands":  2, "Italy":        3,
    "Madrid":        2, "Azerbaijan":   3, "Singapore":    1,
    "Austin":        2, "Mexico City":  2, "Sao Paulo":    2,
    "Las Vegas":     3, "Qatar":        3, "Abu Dhabi":    2,
}

# ── DNF cause classification ──────────────────────────────
# FastF1 result status strings that indicate engine/PU failures
ENGINE_DNF_KEYWORDS = [
    "power unit", "engine", "hybrid", "electrical", "mgu",
    "ers", "turbo", "exhaust", "overheating", "water leak",
    "oil leak", "hydraulics", "gearbox",  # gearbox included — PU-adjacent
]

MECHANICAL_DNF_KEYWORDS = ENGINE_DNF_KEYWORDS + [
    "mechanical", "suspension", "brakes", "driveshaft",
    "puncture", "wheel", "accident", "collision", "dnf",
]


def classify_dnf(status: str) -> str:
    """Classify a FastF1 result status string into a DNF category."""
    if pd.isna(status):
        return "finished"
    s = str(status).lower()
    if any(k in s for k in ENGINE_DNF_KEYWORDS):
        return "engine_dnf"
    if any(k in s for k in MECHANICAL_DNF_KEYWORDS):
        return "mechanical_dnf"
    if s in ("finished", "+1 lap", "+2 laps", "+3 laps"):
        return "finished"
    return "other_dnf"


def fetch_dnf_history(seasons: list[int]) -> pd.DataFrame:
    """
    Pull race results for all given seasons and classify each
    driver's finish status per race.
    Returns a flat DataFrame with one row per driver per race.
    """
    records = []
    for year in seasons:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, event in schedule.iterrows():
            rnd = event["RoundNumber"]
            print(f"  Reliability: [{year}] Round {rnd} {event['Location']}")
            try:
                session = fastf1.get_session(year, rnd, "R")
                session.load(laps=False, telemetry=False,
                             weather=False, messages=False)
                results = session.results[
                    ["Abbreviation", "TeamName", "Status", "GridPosition", "ClassifiedPosition"]
                ].copy()
                results.columns = ["driver", "team", "status", "grid", "finish_pos"]
                results["year"]    = year
                results["round"]   = rnd
                results["circuit"] = event["Location"]
                results["dnf_type"]= results["status"].apply(classify_dnf)
                results["is_dnf"]  = (results["dnf_type"] != "finished").astype(int)
                results["is_engine_dnf"] = (results["dnf_type"] == "engine_dnf").astype(int)
                records.append(results)
            except Exception as e:
                print(f"    ⚠ Skipped — {e}")
    
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def compute_rolling_reliability(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Compute rolling reliability scores per team and per driver.

    car_reliability_score: team's rolling DNF rate (all mechanical)
    engine_dnf_rate:        team's rolling engine-specific DNF rate
    driver_dnf_rate:        driver's personal rolling DNF rate

    Window = last N races. Lower score = more reliable.
    """
    df = df.sort_values(["year", "round"])

    # ── Team reliability ──────────────────────────────────
    team_rel = (
        df.groupby(["team", "year", "round"])
        .agg(
            any_dnf     =("is_dnf",        "max"),
            engine_dnf  =("is_engine_dnf", "max"),
        )
        .reset_index()
        .sort_values(["team", "year", "round"])
    )
    team_rel["car_reliability_score"] = (
        team_rel.groupby("team")["any_dnf"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    team_rel["engine_dnf_rate"] = (
        team_rel.groupby("team")["engine_dnf"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )

    # ── Driver reliability ────────────────────────────────
    driver_rel = (
        df.sort_values(["driver", "year", "round"])
        .copy()
    )
    driver_rel["driver_dnf_rate"] = (
        driver_rel.groupby("driver")["is_dnf"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )

    # Merge team scores back onto driver-level df
    df = df.merge(
        team_rel[["team","year","round","car_reliability_score","engine_dnf_rate"]],
        on=["team","year","round"], how="left"
    )
    df = df.merge(
        driver_rel[["driver","year","round","driver_dnf_rate"]],
        on=["driver","year","round"], how="left"
    )
    return df


def compute_engine_failure_prob(df: pd.DataFrame) -> pd.DataFrame:
    """
    engine_failure_prob = engine_dnf_rate × circuit_pu_demand_factor
    Normalised to [0, 1].
    This is used in prediction to penalise a driver's expected finish.
    """
    df["circuit_pu_demand"] = df["circuit"].map(CIRCUIT_PU_DEMAND).fillna(2)
    # Demand 1→0.8x, 2→1.0x, 3→1.3x multiplier
    demand_factor = {1: 0.8, 2: 1.0, 3: 1.3}
    df["pu_demand_factor"]      = df["circuit_pu_demand"].map(demand_factor)
    df["engine_failure_prob"]   = (
        df["engine_dnf_rate"] * df["pu_demand_factor"]
    ).clip(0, 1)
    return df


def build_and_save(seasons: list[int] = None):
    """
    Full pipeline: fetch → classify → compute scores → save.
    Call this from 1_fetch_data.py or standalone.
    """
    if seasons is None:
        seasons = SEASONS

    print("\n=== Building Reliability Scores ===")
    os.makedirs("./data", exist_ok=True)

    # Check if raw DNF data already exists (avoid re-downloading)
    if os.path.exists(DNF_RAW_PATH):
        print(f"  Loading cached DNF data from {DNF_RAW_PATH}")
        df = pd.read_csv(DNF_RAW_PATH)
    else:
        df = fetch_dnf_history(seasons)
        if df.empty:
            print("  ⚠ No DNF data fetched — reliability features will default to 0")
            return
        df.to_csv(DNF_RAW_PATH, index=False)
        print(f"  Saved raw DNF data → {DNF_RAW_PATH}")

    df = compute_rolling_reliability(df)
    df = compute_engine_failure_prob(df)
    df.to_csv(RELIABILITY_PATH, index=False)
    print(f"  ✅ Saved reliability scores → {RELIABILITY_PATH}")
    return df


def get_latest_scores(as_of_round: int = None, year: int = None) -> dict:
    """
    Get the most recent reliability scores for each team and driver.
    Used by 5_predict_top10.py to fill features before prediction.

    Returns:
        {
          "team": { "Ferrari": 0.05, "Red Bull Racing": 0.03, ... },
          "driver": { "VER": 0.02, "HAM": 0.08, ... },
          "engine_failure_prob": { "Ferrari": 0.04, ... }
        }
    """
    if not os.path.exists(RELIABILITY_PATH):
        print("  ⚠ No reliability data — run build_and_save() first. Defaulting to 0.")
        teams   = {t: 0.0 for t in set(F1_2026_DRIVERS.values())}
        drivers = {d: 0.0 for d in F1_2026_DRIVERS.keys()}
        return {"team": teams, "driver": drivers, "engine_failure_prob": teams}

    df = pd.read_csv(RELIABILITY_PATH)

    # Filter to most recent data available
    if year and as_of_round:
        df = df[(df["year"] < year) | ((df["year"] == year) & (df["round"] <= as_of_round))]

    # Latest score per team
    team_scores = (
        df.sort_values(["year","round"])
        .groupby("team")
        .last()
        [["car_reliability_score","engine_failure_prob"]]
        .to_dict()
    )
    # Latest score per driver
    driver_scores = (
        df.sort_values(["year","round"])
        .groupby("driver")
        .last()
        ["driver_dnf_rate"]
        .to_dict()
    )

    return {
        "team":               team_scores.get("car_reliability_score", {}),
        "driver":             driver_scores,
        "engine_failure_prob":team_scores.get("engine_failure_prob", {}),
    }


def append_2026_race(round_number: int):
    """
    After a 2026 race, append its DNF data to the reliability dataset
    and recompute rolling scores.
    Called automatically by 4_fetch_2026_race.py.
    """
    print(f"  Updating reliability with 2026 Round {round_number}...")
    try:
        session = fastf1.get_session(2026, round_number, "R")
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        
        cal     = {r[0]: r[1] for r in F1_2026_CALENDAR}
        circuit = cal.get(round_number, "Unknown")
        
        results = session.results[
            ["Abbreviation","TeamName","Status","GridPosition","ClassifiedPosition"]
        ].copy()
        results.columns = ["driver","team","status","grid","finish_pos"]
        results["year"]         = 2026
        results["round"]        = round_number
        results["circuit"]      = circuit
        results["dnf_type"]     = results["status"].apply(classify_dnf)
        results["is_dnf"]       = (results["dnf_type"] != "finished").astype(int)
        results["is_engine_dnf"]= (results["dnf_type"] == "engine_dnf").astype(int)

        # Append to raw DNF file
        if os.path.exists(DNF_RAW_PATH):
            existing = pd.read_csv(DNF_RAW_PATH)
            existing = existing[~((existing["year"]==2026) & (existing["round"]==round_number))]
            combined = pd.concat([existing, results], ignore_index=True)
        else:
            combined = results
        combined.to_csv(DNF_RAW_PATH, index=False)

        # Recompute scores
        combined = compute_rolling_reliability(combined)
        combined = compute_engine_failure_prob(combined)
        combined.to_csv(RELIABILITY_PATH, index=False)
        print(f"  ✅ Reliability scores updated with Round {round_number}")

    except Exception as e:
        print(f"  ⚠ Could not update reliability: {e}")


# ── Standalone usage ──────────────────────────────────────
if __name__ == "__main__":
    df = build_and_save()
    if df is not None:
        print("\nSample scores:")
        cols = ["team","driver","year","round","car_reliability_score",
                "driver_dnf_rate","engine_failure_prob"]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].dropna().tail(20).to_string(index=False))
