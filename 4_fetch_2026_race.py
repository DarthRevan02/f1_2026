# ============================================================
# 4_fetch_2026_race.py  —  Fetch ALL sessions for a completed
#                           2026 race weekend and append to
#                           the growing 2026 dataset
# ============================================================
# Sessions fetched per weekend:
#   Race         — always
#   Qualifying   — always
#   Sprint       — sprint weekends only (auto-detected)
#   Sprint Quali — sprint weekends only (auto-detected)
#
# Usage:
#   python 4_fetch_2026_race.py <round_number>
#   e.g.  python 4_fetch_2026_race.py 1   ← Australia
#         python 4_fetch_2026_race.py 2   ← China (sprint weekend)
#
# Output:
#   Appends to ./data/2026_laps.csv
#   Updates ./data/reliability_scores.csv
# ============================================================

import fastf1
import pandas as pd
import numpy as np
import joblib, os, sys

from config import (
    CACHE_DIR, SEASON_2026_DATA, FEATURE_COLS, TARGET,
    F1_2026_CALENDAR, F1_2026_DRIVERS,
)
from reliability import append_2026_race as reliability_update

# Reuse all session-fetch helpers from 1_fetch_data
# (gap_to_pole, weather merge, tyre encoding, strategy, incidents etc.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
_fetch = import_module("1_fetch_data")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("./data", exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# Sprint weekends in 2026 — auto-detected from calendar
SPRINT_ROUNDS_2026 = {
    r[0] for r in F1_2026_CALENDAR if r[4]  # r[4] = has_sprint
}


def encode_with_saved(df: pd.DataFrame, col_raw: str,
                      col_enc: str) -> pd.DataFrame:
    """Use encoder trained on historical data. Unknown values → -1."""
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


def apply_encoders(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all saved label encoders to a 2026 lap DataFrame."""
    df = encode_with_saved(df, "Driver",       "driver_encoded")
    df = encode_with_saved(df, "Team",         "team_encoded")
    df = encode_with_saved(df, "circuit",      "circuit_encoded")
    df = encode_with_saved(df, "session_type", "session_type_encoded")
    return df


def build_interaction_features(df: pd.DataFrame,
                                circuit: str) -> pd.DataFrame:
    """Compute the 6 interaction features used by the model."""
    STREET = {"Monaco","Singapore","Azerbaijan","Las Vegas","Saudi Arabia","Madrid"}
    is_street = 1 if circuit in STREET else 0

    df["tyre_temp_interaction"]   = df["TyreLife"].fillna(1) * df["TrackTemp"].fillna(35)
    df["wet_deg_interaction"]     = df["Rainfall"].fillna(0) * df["tyre_deg_class"].fillna(1)
    df["grid_street_interaction"] = df["grid_position"].fillna(10) * (1.5 if is_street else 1.0)
    df["driver_tyre_interaction"] = (df["driver_avg_gap_hist"].fillna(1.5) *
                                     df["TyreLife"].fillna(1))
    df["speed_temp_interaction"]  = df["SpeedST"].fillna(300) * df["TrackTemp"].fillna(35)
    df["fuel_stint_interaction"]  = df["fuel_load_proxy"].fillna(0) * df["stint_number"].fillna(1)
    df["is_street_circuit"]       = is_street
    return df


def fetch_weekend(round_number: int) -> pd.DataFrame:
    """
    Fetch all available sessions for a 2026 race weekend.
    Uses the shared fetch helpers from 1_fetch_data.py so
    feature computation is identical to historical training data.
    """
    cal = {r[0]: r for r in F1_2026_CALENDAR}
    if round_number not in cal:
        raise ValueError(f"Round {round_number} not in F1_2026_CALENDAR")

    _, city, fastf1_name, race_date, _ = cal[round_number]
    has_sprint = round_number in SPRINT_ROUNDS_2026

    sprint_str = " [SPRINT WEEKEND]" if has_sprint else ""
    print(f"\n  Fetching 2026 Round {round_number}: {city}{sprint_str}")

    df = _fetch.fetch_round(
        year=2026,
        round_number=round_number,
        circuit=city,
        has_sprint=has_sprint,
    )

    if df.empty:
        raise ValueError(f"No data returned for Round {round_number}")

    # Show what was fetched
    summary = df.groupby("session_type").size().to_dict()
    print(f"  Sessions fetched: {summary}")

    return df, city


def post_process(df: pd.DataFrame, round_number: int,
                 city: str) -> pd.DataFrame:
    """
    Apply 2026-specific post-processing:
      - Apply saved label encoders (Driver, Team, circuit, session_type)
      - Build interaction features
      - Add 2026 regulation placeholders
      - Ensure all FEATURE_COLS exist
    """
    from grid_normalizer import normalize_teams, DRIVER_2026_TEAM

    # Remap teams to 2026 grid
    df["Team"] = df.apply(
        lambda row: DRIVER_2026_TEAM.get(row.get("Driver",""), row.get("Team","")),
        axis=1
    )

    # Apply all encoders
    df["circuit"] = city   # ensure consistent circuit name
    df = apply_encoders(df)

    # Interaction features
    df = build_interaction_features(df, city)

    # 2026 regulation placeholders (update when FastF1 logs telemetry)
    for col in ["overtake_mode_laps","active_aero_mode",
                "energy_store_pct","fuel_remaining_kg"]:
        if col not in df.columns:
            df[col] = 0

    # Ensure every model feature column exists
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    # Keep only valid target rows
    df = df[df[TARGET].notna() & (df[TARGET] >= 0)].copy()

    # Attach metadata
    df["year"]  = 2026
    df["round"] = round_number
    df["circuit"] = city

    return df


def save_to_2026_dataset(df: pd.DataFrame, round_number: int):
    """Append this weekend to 2026_laps.csv, replacing if already exists."""
    keep_cols = (FEATURE_COLS +
                 [TARGET, "LapTime_s", "Driver", "Team",
                  "year", "round", "circuit", "session_type",
                  "session_type_encoded", "LapNumber"])
    keep_cols = list(dict.fromkeys(  # dedup while preserving order
        c for c in keep_cols if c in df.columns
    ))
    df = df[keep_cols].copy()

    if os.path.exists(SEASON_2026_DATA):
        existing = pd.read_csv(SEASON_2026_DATA)
        # Drop this round if already present (allows re-fetching)
        existing = existing[existing["round"] != round_number]
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(SEASON_2026_DATA, index=False)

    rounds_done   = sorted(df["round"].unique().astype(int).tolist())
    session_counts = df[df["round"]==round_number].groupby("session_type").size().to_dict()

    print(f"\n  ✅ 2026 dataset updated → {SEASON_2026_DATA}")
    print(f"  Completed rounds : {rounds_done}")
    print(f"  This weekend     : {session_counts}")
    print(f"  Total 2026 laps  : {len(df[df['year']==2026]):,}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python 4_fetch_2026_race.py <round_number>")
        print("Example: python 4_fetch_2026_race.py 2")
        sys.exit(1)

    round_number = int(sys.argv[1])

    print(f"\n{'='*55}")
    print(f"  2026 Race Weekend Fetch — Round {round_number}")
    print(f"  Sessions: Race + Qualifying + Sprint (if applicable)")
    print(f"{'='*55}")

    # 1. Fetch all sessions
    df, city = fetch_weekend(round_number)

    # 2. Post-process (encoders + interactions + placeholders)
    print(f"\n  Post-processing {len(df):,} laps...")
    df = post_process(df, round_number, city)
    print(f"  Clean laps after target filter: {len(df):,}")

    # 3. Save
    save_to_2026_dataset(df, round_number)

    # 4. Update reliability scores with this race's DNF data
    print(f"\n  Updating reliability scores...")
    reliability_update(round_number)

    print(f"\n{'='*55}")
    print(f"  ✅ Round {round_number} complete — ready to retrain")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
