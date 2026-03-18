# ============================================================
# 4_fetch_2026_race.py  —  Fetch one completed 2026 race,
#      compute all 8-factor features, append to 2026 dataset
# ============================================================
# Usage:
#   python 4_fetch_2026_race.py <round_number>
#   e.g.  python 4_fetch_2026_race.py 1   ← after Australia
#         python 4_fetch_2026_race.py 2   ← after China
# ============================================================

import fastf1
import pandas as pd
import numpy as np
import joblib, os, sys

from config import (
    CACHE_DIR, SEASON_2026_DATA, FEATURE_COLS, TARGET,
    F1_2026_CALENDAR, F1_2026_DRIVERS
)
from reliability import append_2026_race as reliability_update
# Reuse helper functions from 1_fetch_data
from importlib import import_module
fetch_mod = import_module("1_fetch_data")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("./data", exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

CIRCUIT_DEG = {
    "Bahrain": 2, "Saudi Arabia": 0, "Australia": 1, "Japan": 0,
    "China": 2, "Miami": 1, "Monaco": 0, "Barcelona": 2,
    "Canada": 0, "Austria": 1, "Great Britain": 1, "Hungary": 2,
    "Belgium": 0, "Netherlands": 2, "Italy": 0, "Singapore": 0,
    "Qatar": 2, "Austin": 1, "Mexico City": 0, "Sao Paulo": 1,
    "Las Vegas": 0, "Abu Dhabi": 1, "Azerbaijan": 0, "Madrid": 1,
}


def encode_col(df, col_raw, col_enc):
    enc_path = f"./models/encoder_{col_raw}.pkl"
    if os.path.exists(enc_path):
        le = joblib.load(enc_path)
        df[col_enc] = df[col_raw].apply(
            lambda x: int(le.transform([str(x)])[0])
                      if str(x) in le.classes_ else -1
        )
    else:
        df[col_enc] = 0
    return df


def apply_driver_baselines(df: pd.DataFrame, feat_path: str) -> pd.DataFrame:
    """Pull driver baseline stats from historical feature set."""
    if not os.path.exists(feat_path):
        df["driver_avg_gap_hist"] = 0
        df["driver_consistency"]  = 0
        df["driver_wet_skill"]    = 0
        return df

    hist = pd.read_csv(feat_path, usecols=["Driver","driver_avg_gap_hist",
                                            "driver_consistency","driver_wet_skill"])
    hist = hist.dropna().groupby("Driver").first().reset_index()
    df   = df.merge(hist, on="Driver", how="left")
    df["driver_avg_gap_hist"] = df["driver_avg_gap_hist"].fillna(1.5)
    df["driver_consistency"]  = df["driver_consistency"].fillna(0.5)
    df["driver_wet_skill"]    = df["driver_wet_skill"].fillna(0)
    return df


def apply_car_reliability(df: pd.DataFrame, feat_path: str) -> pd.DataFrame:
    if not os.path.exists(feat_path):
        df["car_reliability_score"] = 0.95
        return df
    hist = pd.read_csv(feat_path, usecols=["Team","car_reliability_score"])
    hist = hist.dropna().groupby("Team").first().reset_index()
    df   = df.merge(hist, on="Team", how="left")
    df["car_reliability_score"] = df["car_reliability_score"].fillna(0.95)
    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python 4_fetch_2026_race.py <round_number>")
        sys.exit(1)

    round_number = int(sys.argv[1])
    cal = {r[0]: r for r in F1_2026_CALENDAR}
    if round_number not in cal:
        raise ValueError(f"Round {round_number} not in calendar")

    _, city, fastf1_name, race_date, _ = cal[round_number]
    print(f"\nFetching 2026 Round {round_number}: {city}...")

    # Use the same core fetch function from 1_fetch_data.py
    df = fetch_mod.fetch_session(2026, round_number, city)

    if df.empty:
        print("  ⚠ No data returned. Is the race session available on FastF1 yet?")
        sys.exit(1)

    print(f"  {len(df)} laps fetched")

    # Encode categoricals using saved historical encoders
    df = encode_col(df, "Driver",  "driver_encoded")
    df = encode_col(df, "Team",    "team_encoded")
    df = encode_col(df, "circuit", "circuit_encoded")

    # Driver baselines from historical data
    from config import FEATURES_DATA_PATH
    df = apply_driver_baselines(df, FEATURES_DATA_PATH)
    df = apply_car_reliability(df,  FEATURES_DATA_PATH)

    # Tyre deg class
    df["tyre_deg_class"] = df["circuit"].map(CIRCUIT_DEG).fillna(1).astype(int)

    # Track evolution delta
    baseline = df.groupby("LapNumber")["TrackTemp"].transform("first")
    if "track_temp_delta" not in df.columns:
        df["track_temp_delta"] = df["TrackTemp"] - baseline

    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    # Fill NaNs
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    df = df[df[TARGET].notna() & (df[TARGET] >= 0)]

    # Keep only needed columns
    keep = FEATURE_COLS + [TARGET, "LapTime_s", "Driver", "Team",
                            "year", "round", "circuit", "LapNumber"]
    keep = [c for c in keep if c in df.columns]
    df   = df[keep].copy()

    # Append to 2026 dataset (replace if round already exists)
    if os.path.exists(SEASON_2026_DATA):
        existing = pd.read_csv(SEASON_2026_DATA)
        existing = existing[existing["round"] != round_number]
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(SEASON_2026_DATA, index=False)
    rounds_done = sorted(df["round"].unique().tolist())
    print(f"\n✅ 2026 dataset updated → {SEASON_2026_DATA}")
    print(f"   Completed rounds: {rounds_done}")
    print(f"   Total 2026 laps : {len(df[df['year']==2026])}")

    # Update rolling reliability scores with this race's DNF data
    reliability_update(round_number)


if __name__ == "__main__":
    main()
