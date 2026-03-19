# ============================================================
# 2_feature_engineering.py  —  Build all model features
# ============================================================
# Fixes applied:
#   [1] compute_gap_to_pole: drop pre-existing fastest_lap_s
#       before merge to prevent _x/_y suffix KeyError
#   [2] compute_tyre_features: use .infer_objects() before
#       .astype(int) to silence pandas FutureWarning
#   [3] compute_driver_baselines: guard against empty wet-lap
#       dataset causing KeyError: driver_wet_skill
#
# Usage: python 2_feature_engineering.py
# Input:  ./data/raw_laps.csv
# Output: ./data/features.csv  +  ./models/encoder_*.pkl
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib, os

from config import (
    RAW_DATA_PATH, FEATURES_DATA_PATH, FEATURE_COLS, TARGET, SEASONS,
    FILTER_MIN_LAP_NUMBER, FILTER_MIN_TYRE_LIFE, FILTER_MAX_GAP_TO_POLE,
    FILTER_LAPTIME_QUANTILE, FILTER_TRACK_STATUS,
)
from reliability import build_and_save as build_reliability, RELIABILITY_PATH
from grid_normalizer import (
    normalize_teams, filter_to_2026_grid_only,
    inject_new_driver_baselines, print_grid_audit,
)

os.makedirs("./models", exist_ok=True)


# ── Circuit characteristics ────────────────────────────────
CIRCUIT_DEG = {
    "Bahrain": 2,      "Saudi Arabia": 0, "Australia": 1,
    "Japan": 0,        "China": 2,        "Miami": 1,
    "Monaco": 0,       "Barcelona": 2,    "Canada": 0,
    "Austria": 1,      "Great Britain": 1,"Hungary": 2,
    "Belgium": 0,      "Netherlands": 2,  "Italy": 0,
    "Singapore": 0,    "Qatar": 2,        "Austin": 1,
    "Mexico City": 0,  "Brazil": 1,       "Las Vegas": 0,
    "Abu Dhabi": 1,    "Azerbaijan": 0,   "Madrid": 1,
}

STREET_CIRCUITS = {
    "Monaco", "Singapore", "Azerbaijan", "Las Vegas", "Saudi Arabia", "Madrid"
}


# ══════════════════════════════════════════════════════════
# [1] TARGET — gap_to_pole
# ══════════════════════════════════════════════════════════

def compute_gap_to_pole(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX: drop pre-existing fastest_lap_s column (written by
    1_fetch_data.py) before the merge to prevent pandas from
    producing fastest_lap_s_x / fastest_lap_s_y columns which
    cause KeyError: 'fastest_lap_s' on the next line.
    """
    if "fastest_lap_s" in df.columns:
        df = df.drop(columns=["fastest_lap_s"])

    fastest = (
        df.groupby(["year", "round", "LapNumber"])["LapTime_s"]
        .min().reset_index()
        .rename(columns={"LapTime_s": "fastest_lap_s"})
    )
    df = df.merge(fastest, on=["year", "round", "LapNumber"], how="left")
    df["gap_to_pole"] = (df["LapTime_s"] - df["fastest_lap_s"]).clip(lower=0)
    return df


# ══════════════════════════════════════════════════════════
# [2] NOISE FILTERING
# ══════════════════════════════════════════════════════════

def filter_noise(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    if "session_type" not in df.columns:
        df["session_type"] = "race"

    is_quali_type = df["session_type"].isin(["quali", "sprint_quali"])

    race_clean  = df[~is_quali_type & df["TrackStatus_encoded"].isin(FILTER_TRACK_STATUS)]
    quali_clean = df[is_quali_type]
    df = pd.concat([race_clean, quali_clean], ignore_index=True)
    print(f"    Track status   : -{n - len(df):>6} laps  (SC/VSC — race/sprint only)")
    n = len(df)

    race_df  = df[~df["session_type"].isin(["quali", "sprint_quali"])]
    quali_df = df[ df["session_type"].isin(["quali", "sprint_quali"])]
    race_df  = race_df[race_df["LapNumber"] >= FILTER_MIN_LAP_NUMBER]
    df = pd.concat([race_df, quali_df], ignore_index=True)
    print(f"    Opening laps   : -{n - len(df):>6} laps  (first {FILTER_MIN_LAP_NUMBER} — race/sprint only)")
    n = len(df)

    df = df[df["TyreLife"] >= FILTER_MIN_TYRE_LIFE]
    print(f"    Out-laps       : -{n - len(df):>6} laps  (TyreLife < {FILTER_MIN_TYRE_LIFE})")
    n = len(df)

    clean_parts = []
    for stype, grp in df.groupby("session_type"):
        ceil = grp["LapTime_s"].quantile(FILTER_LAPTIME_QUANTILE)
        clean_parts.append(grp[grp["LapTime_s"] <= ceil])
    df = pd.concat(clean_parts, ignore_index=True)
    print(f"    Lap time p98   : -{n - len(df):>6} laps  (per session type)")
    n = len(df)

    df = df[df["gap_to_pole"] <= FILTER_MAX_GAP_TO_POLE]
    df = df[df["gap_to_pole"] >= 0]
    print(f"    Gap ceiling    : -{n - len(df):>6} laps  (>{FILTER_MAX_GAP_TO_POLE}s off pace)")

    summary = df.groupby("session_type").size().to_dict()
    print(f"    → {len(df):,} clean laps  {summary}")
    return df.copy()


# ══════════════════════════════════════════════════════════
# CORE FEATURES
# ══════════════════════════════════════════════════════════

def compute_tyre_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX: use .infer_objects(copy=False) before .astype(int) on
    FreshTyre to silence pandas FutureWarning about downcasting
    object dtype arrays via fillna.
    """
    compound_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3,
                    "INTERMEDIATE": 4, "WET": 5}
    df["tyre_compound_encoded"] = (
        df["Compound"].str.upper().map(compound_map).fillna(2).astype(int)
    )
    df["is_fresh_tyre"] = (
        df["FreshTyre"].fillna(False).infer_objects(copy=False).astype(int)
    )

    df = df.sort_values(["year", "round", "Driver", "LapNumber"])
    df["tyre_deg_rate"] = (
        df.groupby(["year", "round", "Driver", "stint_number"])["LapTime_s"]
        .transform(lambda x: x.diff().fillna(0))
        .clip(-2, 2)
    )
    return df


def compute_driver_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX: guard against empty wet-lap subset. If no wet-condition
    races exist in the training data the wet Series is empty and
    (dry - wet) produces nothing — merge adds no column —
    causing KeyError on the subsequent fillna line.
    Check wet.empty first and fall back to 0.0.
    """
    stats = (
        df.groupby("Driver")["gap_to_pole"]
        .agg(driver_avg_gap_hist="mean", driver_consistency="std")
        .reset_index()
    )
    df = df.merge(stats, on="Driver", how="left")

    dry = df[df["Rainfall"] == 0].groupby("Driver")["gap_to_pole"].mean()
    wet = df[df["Rainfall"] == 1].groupby("Driver")["gap_to_pole"].mean()

    if not wet.empty:
        wet_skill = (dry - wet).rename("driver_wet_skill")
        df = df.merge(wet_skill.reset_index(), on="Driver", how="left")
    else:
        df["driver_wet_skill"] = 0.0

    df["driver_wet_skill"] = df["driver_wet_skill"].fillna(0.0)
    return df


def compute_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    df["total_race_laps"] = df.groupby(["year", "round"])["LapNumber"].transform("max")
    df["fuel_load_proxy"] = (
        (df["total_race_laps"] - df["lap_number_in_session"]) * 1.8
    ).clip(lower=0)
    return df


def compute_track_evolution(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df.groupby(["year", "round"])["TrackTemp"].transform("first")
    df["track_temp_delta"] = (df["TrackTemp"] - baseline).fillna(0)
    return df


# ══════════════════════════════════════════════════════════
# [3] INTERACTION FEATURES
# ══════════════════════════════════════════════════════════

def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df["tyre_temp_interaction"]   = df["TyreLife"] * df["TrackTemp"]
    df["wet_deg_interaction"]     = df["Rainfall"] * df["tyre_deg_class"]
    df["is_street_circuit"]       = df["circuit"].isin(STREET_CIRCUITS).astype(int)
    df["grid_street_interaction"] = df["grid_position"] * (
        df["is_street_circuit"].map({1: 1.5, 0: 1.0})
    )
    df["driver_tyre_interaction"] = df["driver_avg_gap_hist"] * df["TyreLife"]
    df["speed_temp_interaction"]  = df["SpeedST"].fillna(300) * df["TrackTemp"]
    df["fuel_stint_interaction"]  = df["fuel_load_proxy"] * df["stint_number"]
    return df


# ══════════════════════════════════════════════════════════
# RELIABILITY MERGE
# ══════════════════════════════════════════════════════════

def merge_reliability_scores(df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(RELIABILITY_PATH):
        print("  Reliability file not found — building now (takes a few minutes)...")
        build_reliability(SEASONS)

    rel = pd.read_csv(RELIABILITY_PATH)

    team_rel = (
        rel.sort_values(["year", "round"])
        .groupby("team")[["car_reliability_score", "engine_dnf_rate", "engine_failure_prob"]]
        .last().reset_index()
        .rename(columns={"team": "Team"})
    )
    driver_rel = (
        rel.sort_values(["year", "round"])
        .groupby("driver")[["driver_dnf_rate"]]
        .last().reset_index()
        .rename(columns={"driver": "Driver"})
    )
    df = df.merge(team_rel,   on="Team",   how="left")
    df = df.merge(driver_rel, on="Driver", how="left")

    df["car_reliability_score"] = df["car_reliability_score"].fillna(0.10)
    df["engine_dnf_rate"]       = df["engine_dnf_rate"].fillna(0.05)
    df["engine_failure_prob"]   = df["engine_failure_prob"].fillna(0.05)
    df["driver_dnf_rate"]       = df["driver_dnf_rate"].fillna(0.08)
    return df


# ══════════════════════════════════════════════════════════
# ENCODING
# ══════════════════════════════════════════════════════════

def encode_and_save(df: pd.DataFrame, col: str) -> pd.DataFrame:
    le = LabelEncoder()
    df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
    joblib.dump(le, f"./models/encoder_{col}.pkl")
    print(f"  Encoded '{col}' → {len(le.classes_)} classes: {list(le.classes_)}")
    return df


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    sep = "=" * 58
    print(f"\n{sep}")
    print("  Feature Engineering Pipeline")
    print(sep)

    # ── Load ──────────────────────────────────────────────
    print("\n[1/9] Loading raw laps...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"  {len(df):,} raw laps loaded from {RAW_DATA_PATH}")

    df["TrackStatus_encoded"] = (
        pd.to_numeric(df["TrackStatus"], errors="coerce").fillna(1).astype(int)
    )
    df["LapNumber"]             = df["LapNumber"].fillna(0).astype(int)
    df["TyreLife"]              = df["TyreLife"].fillna(0).astype(int)
    df["lap_number_in_session"] = df["LapNumber"]
    df["Rainfall"]              = df["Rainfall"].fillna(0).astype(float)
    df["stint_number"] = df.groupby(["year", "round", "Driver"])["Stint"].transform(
        lambda x: pd.factorize(x)[0] + 1
    )
    df["pit_count"] = df.groupby(["year", "round", "Driver"])["PitOutTime"].transform(
        lambda x: x.notna().cumsum()
    )

    # ── Grid normalisation ────────────────────────────────
    print("\n[2/9] Normalising to 2026 grid...")

    if "session_type" not in df.columns:
        df["session_type"] = "race"
        print("  session_type column missing — defaulting to 'race'")
    else:
        session_summary = df["session_type"].value_counts().to_dict()
        print(f"  Session types in data: {session_summary}")

    print_grid_audit()
    df = filter_to_2026_grid_only(df)
    df = normalize_teams(df)

    # ── Target variable ───────────────────────────────────
    print("\n[3/9] Computing gap_to_pole target...")
    df = compute_gap_to_pole(df)
    print(f"  gap_to_pole range: 0.0 – {df['gap_to_pole'].max():.2f}s")

    # ── Noise filtering ───────────────────────────────────
    print("\n[4/9] Filtering noisy laps...")
    before = len(df)
    df = filter_noise(df)
    print(f"  Removed {before - len(df):,} noisy laps "
          f"({(before - len(df)) / before * 100:.1f}% of data)")

    df = inject_new_driver_baselines(df)

    # ── Core features ─────────────────────────────────────
    print("\n[5/9] Computing core features...")
    df["tyre_deg_class"] = df["circuit"].map(CIRCUIT_DEG).fillna(1).astype(int)
    df = compute_tyre_features(df)
    df = compute_driver_baselines(df)
    df = compute_strategy_features(df)
    df = compute_track_evolution(df)
    print(f"  Core features done")

    # ── Interaction features ──────────────────────────────
    print("\n[6/9] Computing interaction features...")
    df = compute_interaction_features(df)

    # ── Reliability ───────────────────────────────────────
    print("\n[7/9] Merging reliability scores...")
    df = merge_reliability_scores(df)

    # ── Encode categoricals ───────────────────────────────
    print("\n[8/9] Encoding categoricals...")
    df = encode_and_save(df, "Driver")
    df = encode_and_save(df, "Team")
    df = encode_and_save(df, "circuit")
    df = encode_and_save(df, "session_type")
    df = df.rename(columns={
        "Driver_encoded":       "driver_encoded",
        "Team_encoded":         "team_encoded",
        "circuit_encoded":      "circuit_encoded",
        "session_type_encoded": "session_type_encoded",
    })

    # ── Fill NaNs ─────────────────────────────────────────
    print("\n[9/9] Filling missing values...")
    fill_zero = [
        "Rainfall", "WindSpeed", "WindDirection", "track_temp_delta",
        "undercut_window", "overcut_window", "is_safety_car_lap",
        "incidents_in_race", "position_delta_sc",
        "last_pit_stop_time", "pit_delta_vs_field",
        "overtake_mode_laps", "active_aero_mode",
        "energy_store_pct", "fuel_remaining_kg",
        "driver_wet_skill", "tyre_deg_rate", "gap_to_pole_quali",
        "wet_deg_interaction", "is_street_circuit",
    ]
    for col in fill_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df["grid_position"]      = df["grid_position"].fillna(20)
    df["AirTemp"]            = df["AirTemp"].fillna(25)
    df["TrackTemp"]          = df["TrackTemp"].fillna(35)
    df["Humidity"]           = df["Humidity"].fillna(50)
    df["avg_pit_time_team"]  = df["avg_pit_time_team"].fillna(2.5)
    df["driver_consistency"] = df["driver_consistency"].fillna(
        df["driver_consistency"].median()
    )

    missing_created = []
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
            missing_created.append(col)
    if missing_created:
        print(f"  ⚠ Created as 0: {missing_created}")

    df = df[df[TARGET].notna() & (df[TARGET] >= 0)]

    # ── Save ──────────────────────────────────────────────
    df.to_csv(FEATURES_DATA_PATH, index=False)

    print(f"\n{sep}")
    print(f"  ✅ Feature engineering complete")
    print(f"  Laps       : {len(df):,}")
    print(f"  Features   : {len(FEATURE_COLS)}")
    print(f"  Target     : {TARGET}  (mean={df[TARGET].mean():.3f}s  "
          f"std={df[TARGET].std():.3f}s)")
    print(f"  Saved to   : {FEATURES_DATA_PATH}")
    print(sep + "\n")


if __name__ == "__main__":
    main()
