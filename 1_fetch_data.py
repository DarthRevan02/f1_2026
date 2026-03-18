# ============================================================
# 1_fetch_data.py  —  Fetch raw data covering all 8 key F1
#                     performance factors from FastF1
# ============================================================
# Run ONCE before the season starts.
# Downloads every Race session for seasons in config.SEASONS.
#
# Usage:
#   python 1_fetch_data.py
#
# Output:  ./data/raw_laps.csv
# ============================================================

import fastf1
import pandas as pd
import numpy as np
import os

from config import SEASONS, CACHE_DIR, RAW_DATA_PATH

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("./data", exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


# ── Helpers ───────────────────────────────────────────────

def safe_total_seconds(td):
    """Convert timedelta to float seconds safely."""
    try:
        return td.total_seconds()
    except Exception:
        return np.nan


def _get_weather_data(session):
    """
    Safely retrieve session.weather_data.

    Newer FastF1 versions raise DataNotLoadedError instead of
    returning None when weather data is unavailable — this wrapper
    catches that and any other exception, returning None on failure.
    """
    try:
        w = session.weather_data
        if w is None or (hasattr(w, "empty") and w.empty):
            return None
        return w
    except Exception:
        return None


def extract_weather(session) -> dict:
    """Return mean weather values across the session."""
    defaults = {"AirTemp": 25, "TrackTemp": 35, "Humidity": 50,
                "WindSpeed": 10, "WindDirection": 180, "Rainfall": 0}

    w = _get_weather_data(session)
    if w is None:
        return defaults

    return {
        "AirTemp":       float(w["AirTemp"].mean()),
        "TrackTemp":     float(w["TrackTemp"].mean()),
        "Humidity":      float(w["Humidity"].mean()),
        "WindSpeed":     float(w["WindSpeed"].mean()),
        "WindDirection": float(w["WindDirection"].mean()),
        "Rainfall":      int(w["Rainfall"].any()),
    }


def extract_lap_weather(session, laps: pd.DataFrame) -> pd.DataFrame:
    """
    Map per-lap weather by finding the closest weather timestamp
    to each lap's start time. Falls back to session average.
    """
    w = _get_weather_data(session)

    if w is None:
        avg = extract_weather(session)
        for k, v in avg.items():
            laps[k] = v
        laps["track_temp_delta"] = 0
        return laps

    weather = w.reset_index()
    wcols   = ["Time", "AirTemp", "TrackTemp", "Humidity",
               "WindSpeed", "WindDirection", "Rainfall"]
    weather = weather[[c for c in wcols if c in weather.columns]].copy()

    # Per-lap weather via merge_asof on Time
    if "Time" in laps.columns and "Time" in weather.columns:
        laps_sorted    = laps.sort_values("Time")
        weather_sorted = weather.sort_values("Time")
        merged = pd.merge_asof(
            laps_sorted, weather_sorted,
            on="Time", direction="nearest"
        )
        # Compute track temp delta from lap 1 baseline
        baseline = merged.loc[merged["LapNumber"] == merged["LapNumber"].min(),
                               "TrackTemp"].mean()
        merged["track_temp_delta"] = merged["TrackTemp"] - baseline
        return merged.sort_values("LapNumber").reset_index(drop=True)

    # Fallback
    avg = extract_weather(session)
    for k, v in avg.items():
        laps[k] = v
    laps["track_temp_delta"] = 0
    return laps


def pit_stop_features(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Extract pit stop execution quality per lap.
    Factors: pit stop duration, team average, delta to field.
    """
    laps = laps.sort_values(["Driver", "LapNumber"]).copy()

    # PitInTime / PitOutTime difference → pit stop duration
    if "PitInTime" in laps.columns and "PitOutTime" in laps.columns:
        pit_in  = laps["PitInTime"].apply(safe_total_seconds)
        pit_out = laps["PitOutTime"].apply(safe_total_seconds)
        laps["raw_pit_duration"] = pit_out - pit_in
        # Only valid where both exist and result is positive and < 60s
        laps["raw_pit_duration"] = laps["raw_pit_duration"].where(
            (laps["raw_pit_duration"] > 1.5) & (laps["raw_pit_duration"] < 60)
        )
    else:
        laps["raw_pit_duration"] = np.nan

    # Carry last pit duration forward within driver stint
    laps["last_pit_stop_time"] = (
        laps.groupby("Driver")["raw_pit_duration"]
        .transform(lambda x: x.ffill())
        .fillna(0)
    )

    # Team average pit time this race
    team_avg = (
        laps.groupby("Team")["raw_pit_duration"]
        .mean().rename("avg_pit_time_team")
    )
    laps = laps.merge(team_avg, on="Team", how="left")
    laps["avg_pit_time_team"] = laps["avg_pit_time_team"].fillna(2.5)

    # Delta to field average
    field_avg = laps["raw_pit_duration"].mean()
    laps["pit_delta_vs_field"] = laps["last_pit_stop_time"] - (field_avg if not np.isnan(field_avg) else 2.5)

    return laps


def strategy_features(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Undercut / overcut windows, fuel proxy, stint counting.
    """
    laps = laps.sort_values(["Driver", "LapNumber"]).copy()

    # Stint number and pit count
    laps["stint_number"] = laps.groupby("Driver")["Stint"].transform(
        lambda x: pd.factorize(x)[0] + 1
    )
    laps["pit_count"] = laps.groupby("Driver")["PitOutTime"].transform(
        lambda x: x.notna().cumsum()
    )

    # Total race laps (max lap number in session)
    total_laps = laps["LapNumber"].max()
    laps["total_race_laps"] = total_laps

    # Fuel load proxy: ~1.8 kg burned per lap, starts at ~100 kg
    laps["fuel_load_proxy"] = (total_laps - laps["LapNumber"]).clip(lower=0) * 1.8

    # Undercut/overcut: did driver pit 1-2 laps before/after field average?
    # Mark laps where pit-in occurred
    pit_laps = laps[laps["raw_pit_duration"].notna()][["Driver", "LapNumber"]].copy()
    pit_laps = pit_laps.rename(columns={"LapNumber": "pit_lap"})

    laps["undercut_window"] = 0
    laps["overcut_window"]  = 0

    if not pit_laps.empty:
        field_pit_avg = pit_laps.groupby("Driver")["pit_lap"].mean().mean()
        for driver, grp in laps.groupby("Driver"):
            driver_pits = pit_laps[pit_laps["Driver"] == driver]["pit_lap"].values
            for pit_lap in driver_pits:
                delta = pit_lap - field_pit_avg
                if -2 <= delta <= -1:
                    laps.loc[(laps["Driver"] == driver) &
                             (laps["LapNumber"] >= pit_lap) &
                             (laps["LapNumber"] <= pit_lap + 5), "undercut_window"] = 1
                elif 1 <= delta <= 2:
                    laps.loc[(laps["Driver"] == driver) &
                             (laps["LapNumber"] >= pit_lap) &
                             (laps["LapNumber"] <= pit_lap + 5), "overcut_window"] = 1

    return laps


def incident_features(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Safety car flags, incident counter, position gains at restart.
    """
    laps["TrackStatus"] = pd.to_numeric(laps["TrackStatus"], errors="coerce").fillna(1)
    laps["TrackStatus_encoded"] = laps["TrackStatus"].astype(int)
    laps["is_safety_car_lap"]   = laps["TrackStatus"].isin([2, 3, 4, 5]).astype(int)

    # Cumulative incidents
    laps = laps.sort_values("LapNumber")
    laps["incidents_in_race"] = laps["is_safety_car_lap"].cumsum()

    # Position delta at last SC restart (position change next lap after SC ends)
    # Initialise as float64 to avoid FutureWarning when assigning float mean values
    laps["position_delta_sc"] = 0.0
    if "Position" in laps.columns:
        sc_end_laps = laps[
            (laps["is_safety_car_lap"] == 0) &
            (laps["is_safety_car_lap"].shift(1) == 1)
        ]["LapNumber"].values
        for lap in sc_end_laps:
            mask = laps["LapNumber"] == lap
            prev = laps[laps["LapNumber"] == lap - 1][["Driver", "Position"]] \
                      .rename(columns={"Position": "prev_pos"})
            curr = laps[mask][["Driver", "Position"]] \
                      .rename(columns={"Position": "curr_pos"})
            merged = curr.merge(prev, on="Driver", how="left")
            if not merged.empty:
                delta_map = (merged["prev_pos"] - merged["curr_pos"]).values
                laps.loc[mask, "position_delta_sc"] = float(delta_map.mean()) if len(delta_map) else 0.0

    return laps


def fetch_session(year: int, round_number: int, circuit: str) -> pd.DataFrame:
    """Core function: load one race session and extract all features."""
    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load(telemetry=True, weather=True, laps=True)
    except Exception as e:
        print(f"    ⚠ Could not load session: {e}")
        return pd.DataFrame()

    laps = session.laps.copy()
    laps["LapTime_s"] = laps["LapTime"].apply(safe_total_seconds)

    # Filter: keep only valid racing laps
    laps = laps[laps["LapTime_s"].between(55, 200)].copy()
    if laps.empty:
        return pd.DataFrame()

    # ── Gap to pole (target variable) ─────────────────────
    fastest = (
        laps.groupby("LapNumber")["LapTime_s"].min()
        .reset_index().rename(columns={"LapTime_s": "fastest_lap_s"})
    )
    laps = laps.merge(fastest, on="LapNumber", how="left")
    laps["gap_to_pole"] = (laps["LapTime_s"] - laps["fastest_lap_s"]).clip(lower=0)

    # ── Weather (per lap) ─────────────────────────────────
    laps = extract_lap_weather(session, laps)

    # ── Grid + quali gap ──────────────────────────────────
    try:
        results = session.results[["Abbreviation", "GridPosition", "Q3", "Q2", "Q1"]].copy()
        results.columns = ["Driver", "grid_position", "Q3", "Q2", "Q1"]
        # Use best quali time available
        results["best_quali_s"] = (
            results["Q3"].apply(safe_total_seconds)
            .fillna(results["Q2"].apply(safe_total_seconds))
            .fillna(results["Q1"].apply(safe_total_seconds))
        )
        pole_time = results["best_quali_s"].min()
        results["gap_to_pole_quali"] = (results["best_quali_s"] - pole_time).fillna(0)
        laps = laps.merge(results[["Driver", "grid_position", "gap_to_pole_quali"]],
                          on="Driver", how="left")
    except Exception:
        laps["grid_position"]     = 10
        laps["gap_to_pole_quali"] = 0

    # ── Tyre features ─────────────────────────────────────
    compound_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5, "UNKNOWN": 0}
    laps["tyre_compound_encoded"] = (
        laps["Compound"].str.upper().map(compound_map).fillna(0).astype(int)
    )
    laps["is_fresh_tyre"] = laps["FreshTyre"].fillna(False).astype(int)

    # Tyre degradation rate: slope of lap time vs TyreLife per driver per stint
    laps = laps.sort_values(["Driver", "LapNumber"])
    laps["tyre_deg_rate"] = (
        laps.groupby(["Driver", "Stint"])["LapTime_s"]
        .transform(lambda x: x.diff().fillna(0))
    )

    # ── Speed columns ─────────────────────────────────────
    for col in ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]:
        if col not in laps.columns:
            laps[col] = np.nan
    laps[["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]] = (
        laps[["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]].fillna(
            laps[["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]].mean()
        )
    )

    # ── Pit stop features ─────────────────────────────────
    laps = pit_stop_features(laps)

    # ── Strategy features ─────────────────────────────────
    laps = strategy_features(laps)

    # ── Incident / safety car ─────────────────────────────
    laps = incident_features(laps)

    # ── Session metadata ──────────────────────────────────
    laps["year"]                  = year
    laps["round"]                 = round_number
    laps["circuit"]               = circuit
    laps["lap_number_in_session"] = laps["LapNumber"]

    # ── Tyre deg class placeholder (set in feature_engineering) ──
    laps["tyre_deg_class"] = 1  # 0=low, 1=med, 2=high — overridden in step 2

    # ── 2026 regulation placeholders ──────────────────────
    laps["overtake_mode_laps"] = 0
    laps["active_aero_mode"]   = 0
    laps["energy_store_pct"]   = 0
    laps["fuel_remaining_kg"]  = 0

    # Placeholder driver-level features (computed in step 2 from full dataset)
    laps["driver_avg_gap_hist"]   = 0
    laps["driver_consistency"]    = 0
    laps["driver_wet_skill"]      = 0
    laps["car_reliability_score"] = 0

    return laps


def fetch_season(year: int) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    all_laps  = []

    for _, event in schedule.iterrows():
        rnd     = event["RoundNumber"]
        circuit = event["Location"]
        print(f"  [{year}] Round {rnd:2d}: {circuit}")
        df = fetch_session(year, rnd, circuit)
        if not df.empty:
            print(f"           → {len(df)} laps")
            all_laps.append(df)
        else:
            print(f"           → skipped")

    return pd.concat(all_laps, ignore_index=True) if all_laps else pd.DataFrame()


def main():
    all_seasons = []
    for year in SEASONS:
        print(f"\n{'='*50}")
        print(f"  Season {year}")
        print(f"{'='*50}")
        df = fetch_season(year)
        all_seasons.append(df)
        print(f"  ✓ {len(df)} laps for {year}")

    full = pd.concat(all_seasons, ignore_index=True)
    full.to_csv(RAW_DATA_PATH, index=False)
    print(f"\n✅ Saved {len(full)} total laps → {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
