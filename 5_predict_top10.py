# ============================================================
# 5_predict_top10.py  —  Predict top 10 finishing positions
#                         + running driver championship standings
# ============================================================
# Usage:
#   python 5_predict_top10.py <round_number> [quali order...]
#
# Examples:
#   python 5_predict_top10.py 2
#   python 5_predict_top10.py 2 RUS ANT LEC HAM NOR PIA VER LAW ALO STR
# ============================================================

import pandas as pd
import numpy as np
import joblib, os, sys

from config import (
    MODEL_PATH, SCALER_PATH, FEATURE_COLS,
    F1_2026_CALENDAR, F1_2026_DRIVERS,
    PREDICTIONS_LOG, STANDINGS_LOG,
    POINTS_MAP, FASTEST_LAP_POINT
)
from reliability import get_latest_scores as get_reliability

os.makedirs("./outputs", exist_ok=True)


# ── Circuit characteristics ────────────────────────────────
CIRCUIT_META = {
    "Australia":    {"tyre_life": 15, "compound": 2, "track_temp": 38, "air_temp": 22},
    "China":        {"tyre_life": 12, "compound": 2, "track_temp": 40, "air_temp": 18},
    "Japan":        {"tyre_life": 18, "compound": 1, "track_temp": 35, "air_temp": 16},
    "Bahrain":      {"tyre_life": 10, "compound": 2, "track_temp": 48, "air_temp": 30},
    "Saudi Arabia": {"tyre_life": 20, "compound": 1, "track_temp": 32, "air_temp": 28},
    "Miami":        {"tyre_life": 15, "compound": 2, "track_temp": 52, "air_temp": 30},
    "Canada":       {"tyre_life": 22, "compound": 1, "track_temp": 38, "air_temp": 24},
    "Monaco":       {"tyre_life": 25, "compound": 1, "track_temp": 42, "air_temp": 22},
    "Barcelona":    {"tyre_life": 10, "compound": 2, "track_temp": 46, "air_temp": 28},
    "Austria":      {"tyre_life": 15, "compound": 2, "track_temp": 44, "air_temp": 24},
    "Great Britain":{"tyre_life": 15, "compound": 2, "track_temp": 36, "air_temp": 20},
    "Belgium":      {"tyre_life": 20, "compound": 1, "track_temp": 32, "air_temp": 20},
    "Hungary":      {"tyre_life": 10, "compound": 2, "track_temp": 50, "air_temp": 32},
    "Netherlands":  {"tyre_life": 10, "compound": 2, "track_temp": 38, "air_temp": 22},
    "Italy":        {"tyre_life": 22, "compound": 1, "track_temp": 38, "air_temp": 26},
    "Madrid":       {"tyre_life": 14, "compound": 2, "track_temp": 44, "air_temp": 28},
    "Azerbaijan":   {"tyre_life": 22, "compound": 1, "track_temp": 38, "air_temp": 26},
    "Singapore":    {"tyre_life": 24, "compound": 1, "track_temp": 34, "air_temp": 30},
    "Austin":       {"tyre_life": 15, "compound": 2, "track_temp": 46, "air_temp": 28},
    "Mexico City":  {"tyre_life": 22, "compound": 1, "track_temp": 28, "air_temp": 22},
    "Sao Paulo":    {"tyre_life": 15, "compound": 2, "track_temp": 44, "air_temp": 26},
    "Las Vegas":    {"tyre_life": 22, "compound": 1, "track_temp": 18, "air_temp": 14},
    "Qatar":        {"tyre_life": 8,  "compound": 3, "track_temp": 52, "air_temp": 28},
    "Abu Dhabi":    {"tyre_life": 16, "compound": 2, "track_temp": 36, "air_temp": 28},
}

DEG_CLASS = {
    "Australia": 1, "China": 2, "Japan": 0, "Bahrain": 2,
    "Saudi Arabia": 0, "Miami": 1, "Canada": 0, "Monaco": 0,
    "Barcelona": 2, "Austria": 1, "Great Britain": 1, "Belgium": 0,
    "Hungary": 2, "Netherlands": 2, "Italy": 0, "Madrid": 1,
    "Azerbaijan": 0, "Singapore": 0, "Austin": 1, "Mexico City": 0,
    "Sao Paulo": 1, "Las Vegas": 0, "Qatar": 2, "Abu Dhabi": 1,
}


def encode_val(col_raw, value):
    enc_path = f"./models/encoder_{col_raw}.pkl"
    if os.path.exists(enc_path):
        le = joblib.load(enc_path)
        return int(le.transform([str(value)])[0]) if str(value) in le.classes_ else -1
    return 0


def get_driver_baselines() -> dict:
    """Load driver baseline stats from features.csv."""
    from config import FEATURES_DATA_PATH
    if not os.path.exists(FEATURES_DATA_PATH):
        return {}
    hist = pd.read_csv(FEATURES_DATA_PATH,
                       usecols=["Driver","driver_avg_gap_hist",
                                "driver_consistency","driver_wet_skill"])
    return hist.dropna().groupby("Driver").first().to_dict("index")


def get_team_speed_proxies() -> dict:
    """Average speed trap values per team from historical data."""
    from config import FEATURES_DATA_PATH
    if not os.path.exists(FEATURES_DATA_PATH):
        return {}
    hist = pd.read_csv(FEATURES_DATA_PATH,
                       usecols=["Team","SpeedST","SpeedI1","SpeedI2","SpeedFL"])
    return hist.dropna().groupby("Team").mean().to_dict("index")


def build_race_scenario(
    circuit: str,
    quali_order: list[str],
    weather_override: dict = None,
    as_of_round: int = None,
) -> pd.DataFrame:
    """
    Build a per-driver feature row representing a mid-race lap
    for all factors considered by the model.
    """
    meta       = CIRCUIT_META.get(circuit, CIRCUIT_META["Australia"])
    deg_class  = DEG_CLASS.get(circuit, 1)
    circuit_enc = encode_val("circuit", circuit)
    driver_baselines = get_driver_baselines()
    team_speeds      = get_team_speed_proxies()

    # ── Load live reliability scores (updates after each race) ──
    rel_scores = get_reliability(as_of_round=as_of_round, year=2026)
    team_reliability   = rel_scores.get("team", {})
    driver_dnf_rates   = rel_scores.get("driver", {})
    engine_fail_probs  = rel_scores.get("engine_failure_prob", {})

    weather = {
        "AirTemp":       meta["air_temp"],
        "TrackTemp":     meta["track_temp"],
        "Humidity":      50,
        "WindSpeed":     12,
        "WindDirection": 180,
        "Rainfall":      0,
    }
    if weather_override:
        weather.update(weather_override)

    rows = []
    for pos, driver in enumerate(quali_order, start=1):
        team = F1_2026_DRIVERS.get(driver, "Unknown")

        # Baseline stats
        db = driver_baselines.get(driver, {})
        ts = team_speeds.get(team, {"SpeedST": 310, "SpeedI1": 195,
                                    "SpeedI2": 200, "SpeedFL": 280})

        # Reliability scores from the reliability module
        car_rel      = team_reliability.get(team, 0.10)     # rolling DNF rate
        driver_dnf   = driver_dnf_rates.get(driver, 0.08)   # personal DNF rate
        engine_prob  = engine_fail_probs.get(team, 0.05)    # engine-specific risk

        # Quali gap approximation from grid position
        gap_to_pole_quali = (pos - 1) * 0.05

        # Fuel proxy: mid-race
        mid_lap   = 30
        total_lap = 57
        fuel_proxy = (total_lap - mid_lap) * 1.8

        rows.append({
            # ── Car Performance ──
            "SpeedST":               ts.get("SpeedST", 310),
            "SpeedI1":               ts.get("SpeedI1", 195),
            "SpeedI2":               ts.get("SpeedI2", 200),
            "SpeedFL":               ts.get("SpeedFL", 280),
            "team_encoded":          encode_val("Team", team),
            "car_reliability_score": car_rel,

            # ── Driver Ability ──
            "driver_encoded":        encode_val("Driver", driver),
            "driver_avg_gap_hist":   db.get("driver_avg_gap_hist", 1.5),
            "driver_consistency":    db.get("driver_consistency", 0.5),
            "driver_wet_skill":      db.get("driver_wet_skill", 0),

            # ── Reliability (stored for penalty step) ──
            "_driver_dnf_rate":      driver_dnf,
            "_engine_failure_prob":  engine_prob,

            # ── Strategy ──
            "stint_number":          2,
            "pit_count":             1,
            "undercut_window":       0,
            "overcut_window":        0,
            "total_race_laps":       total_lap,
            "fuel_load_proxy":       fuel_proxy,

            # ── Qualifying ──
            "grid_position":         pos,
            "gap_to_pole_quali":     gap_to_pole_quali,

            # ── Tyre Management ──
            "TyreLife":              meta["tyre_life"],
            "tyre_compound_encoded": meta["compound"],
            "is_fresh_tyre":         0,
            "tyre_deg_rate":         0.05 * deg_class,      # approx deg rate
            "tyre_deg_class":        deg_class,

            # ── Incidents ──
            "TrackStatus_encoded":   1,
            "is_safety_car_lap":     0,
            "incidents_in_race":     1,                      # 1 average incident
            "position_delta_sc":     0,

            # ── Pit Stop ──
            "last_pit_stop_time":    2.4,
            "avg_pit_time_team":     2.4,
            "pit_delta_vs_field":    0.0,

            # ── Environment ──
            "AirTemp":               weather["AirTemp"],
            "TrackTemp":             weather["TrackTemp"],
            "Humidity":              weather["Humidity"],
            "WindSpeed":             weather["WindSpeed"],
            "WindDirection":         weather["WindDirection"],
            "Rainfall":              weather["Rainfall"],
            "track_temp_delta":      5,                      # track warms ~5°C by mid-race
            "circuit_encoded":       circuit_enc,
            "lap_number_in_session": mid_lap,

            # ── 2026 Regs ──
            "overtake_mode_laps":    0,
            "active_aero_mode":      0,
            "energy_store_pct":      0,
            "fuel_remaining_kg":     0,

            # Metadata
            "Driver": driver,
            "Team":   team,
        })

    return pd.DataFrame(rows)


def predict_top10(
    round_number: int,
    quali_order: list[str] | None = None,
    weather_override: dict = None,
    is_sprint: bool = False,
) -> pd.DataFrame:

    cal = {r[0]: r for r in F1_2026_CALENDAR}
    if round_number not in cal:
        raise ValueError(f"Round {round_number} not in calendar")

    _, city, _, race_date, _ = cal[round_number]

    if quali_order is None:
        quali_order = list(F1_2026_DRIVERS.keys())
        print("  ⚠  No qualifying order — using default driver list")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("No model found. Run 3_train_model.py first.")

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Pass as_of_round so reliability uses data only from completed races
    completed_round = round_number - 1
    df = build_race_scenario(city, quali_order, weather_override,
                             as_of_round=completed_round)
    X  = df[FEATURE_COLS].fillna(0)
    df["predicted_gap_s"] = model.predict(scaler.transform(X))

    # ── Reliability penalty ────────────────────────────────────
    # Higher engine_failure_prob + driver_dnf_rate → add gap penalty
    # This pushes unreliable cars further back in predicted order.
    #
    # Formula:
    #   dnf_combined = 0.6 × engine_failure_prob + 0.4 × driver_dnf_rate
    #   penalty_s    = dnf_combined × RELIABILITY_PENALTY_SCALE
    #
    # Scale = 5s: a team with 20% failure rate gets +1s added to their gap.
    # Adjust RELIABILITY_PENALTY_SCALE to make reliability more/less impactful.
    RELIABILITY_PENALTY_SCALE = 5.0
    df["_dnf_combined"] = (
        0.6 * df["_engine_failure_prob"].fillna(0.05) +
        0.4 * df["_driver_dnf_rate"].fillna(0.08)
    )
    df["reliability_penalty_s"] = df["_dnf_combined"] * RELIABILITY_PENALTY_SCALE
    df["predicted_gap_s"]       = df["predicted_gap_s"] + df["reliability_penalty_s"]

    # Rank: lower gap = better position
    df = df.sort_values("predicted_gap_s").reset_index(drop=True)
    df["predicted_position"] = range(1, len(df) + 1)

    top10 = df.head(10).copy()
    top10["Gap to Leader (s)"] = (
        top10["predicted_gap_s"] - top10["predicted_gap_s"].iloc[0]
    ).round(3)

    # ── Print ─────────────────────────────────────────────
    label = "SPRINT" if is_sprint else "RACE"
    print(f"\n{'═'*70}")
    print(f"  🏁  2026 Round {round_number} — {city}  [{label} PREDICTION]")
    print(f"  📅  {race_date}")
    print(f"{'═'*70}")
    header = f"  {'Pos':>3}  {'Driver':<6}  {'Team':<20}  {'Gap (s)':>8}  {'Rel.Risk':>9}  {'Eng.Risk':>9}"
    print(header)
    print(f"  {'─'*65}")
    for _, row in top10.iterrows():
        gap_str = "LEADER" if row["predicted_position"] == 1 else f"+{row['Gap to Leader (s)']:.3f}"
        rel_pct  = f"{row['_driver_dnf_rate']*100:.1f}%"
        eng_pct  = f"{row['_engine_failure_prob']*100:.1f}%"
        print(f"  {int(row['predicted_position']):>3}  {row['Driver']:<6}  "
              f"{row['Team']:<20}  {gap_str:>8}  {rel_pct:>9}  {eng_pct:>9}")
    print(f"{'═'*70}")
    print(f"  Rel.Risk = driver personal DNF rate (rolling last 10 races)")
    print(f"  Eng.Risk = engine/PU failure prob (team rate × circuit PU demand)")
    print(f"{'═'*70}\n")

    # ── Log predictions ───────────────────────────────────
    log_row = top10[["predicted_position","Driver","Team",
                      "Gap to Leader (s)","_driver_dnf_rate","_engine_failure_prob"]].copy()
    log_row.columns = ["Pos","Driver","Team","Gap_s","driver_dnf_rate","engine_failure_prob"]
    log_row["round"]        = round_number
    log_row["circuit"]      = city
    log_row["race_date"]    = race_date
    log_row["is_sprint"]    = is_sprint
    log_row["predicted_on"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    if os.path.exists(PREDICTIONS_LOG):
        existing = pd.read_csv(PREDICTIONS_LOG)
        existing = existing[~((existing["round"] == round_number) &
                              (existing["is_sprint"] == is_sprint))]
        log_row = pd.concat([existing, log_row], ignore_index=True)
    log_row.to_csv(PREDICTIONS_LOG, index=False)

    return top10


def update_standings(round_number: int, top10: pd.DataFrame, is_sprint: bool = False):
    """
    Compute and display the projected championship standings
    by accumulating predicted points from all predicted races so far.
    """
    pts_map = POINTS_MAP

    # Load existing standings or start fresh
    if os.path.exists(STANDINGS_LOG):
        standings = pd.read_csv(STANDINGS_LOG)
    else:
        standings = pd.DataFrame(columns=["Driver","Team","Points","Wins","Podiums"])

    # Points for this race
    new_pts = []
    for _, row in top10.iterrows():
        pos    = int(row["predicted_position"])
        points = pts_map.get(pos, 0)
        if pos == 1:
            points += FASTEST_LAP_POINT  # assume race winner also sets fastest lap
        new_pts.append({
            "Driver": row["Driver"],
            "Team":   row["Team"],
            "points": points,
            "win":    1 if pos == 1 else 0,
            "podium": 1 if pos <= 3 else 0,
        })

    new_df = pd.DataFrame(new_pts)

    # Merge with existing
    if standings.empty:
        standings = new_df.rename(columns={
            "points":"Points","win":"Wins","podium":"Podiums"
        })
    else:
        merged = standings.merge(
            new_df, on=["Driver","Team"], how="outer"
        ).fillna(0)
        merged["Points"]  = merged["Points"]  + merged["points"]
        merged["Wins"]    = merged["Wins"]    + merged["win"]
        merged["Podiums"] = merged["Podiums"] + merged["podium"]
        standings = merged[["Driver","Team","Points","Wins","Podiums"]]

    standings = standings.sort_values("Points", ascending=False).reset_index(drop=True)
    standings["Position"] = range(1, len(standings) + 1)

    # Print standings
    rounds_done = round_number
    print(f"\n{'═'*60}")
    print(f"  🏆  PROJECTED DRIVER STANDINGS  (after Round {rounds_done})")
    print(f"{'═'*60}")
    print(f"  {'Pos':>3}  {'Driver':<6}  {'Team':<22}  {'Pts':>5}  {'W':>3}  {'Pod':>4}")
    print(f"  {'─'*54}")
    for _, row in standings.head(20).iterrows():
        print(f"  {int(row['Position']):>3}  {row['Driver']:<6}  {row['Team']:<22}"
              f"  {int(row['Points']):>5}  {int(row['Wins']):>3}  {int(row['Podiums']):>4}")
    print(f"{'═'*60}\n")

    standings.to_csv(STANDINGS_LOG, index=False)
    print(f"  Standings saved → {STANDINGS_LOG}")
    return standings


def main():
    if len(sys.argv) < 2:
        print("Usage: python 5_predict_top10.py <round> [quali_order...]")
        print("  e.g. python 5_predict_top10.py 2 RUS ANT LEC HAM NOR PIA VER LAW ALO STR")
        sys.exit(1)

    round_number = int(sys.argv[1])
    quali_order  = sys.argv[2:] if len(sys.argv) > 2 else None

    cal = {r[0]: r for r in F1_2026_CALENDAR}
    _, city, _, _, is_sprint = cal[round_number]

    top10 = predict_top10(round_number, quali_order, is_sprint=is_sprint)
    update_standings(round_number, top10, is_sprint=is_sprint)


if __name__ == "__main__":
    main()
