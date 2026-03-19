#!/usr/bin/env python3
# ============================================================
# run.py  —  F1 2026 Predictor  |  Interactive Terminal UI
# ============================================================
# Single entry point for everything. Run once, use all season.
#
# Usage:
#   python run.py
#
# What it does:
#   - Shows current driver & constructor championship standings
#   - Predicts top 10 for the next race (or any race you choose)
#   - After a race completes: fetches data, updates reliability,
#     retrains model, updates standings — all in one flow
#   - After race 10: model fully switches to 2026-only training
#     (drops all pre-2026 historical data)
# ============================================================

import os, sys, time, json, subprocess
import pandas as pd
import numpy as np
from datetime import datetime

# ── Make sure we're running from the project directory ─────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ── Colour codes (work on all major terminals) ─────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    GREY   = "\033[90m"

    # Team colours (approximate terminal equivalent)
    TEAM_COLORS = {
        "Red Bull Racing": "\033[94m",   # blue
        "Ferrari":         "\033[91m",   # red
        "McLaren":         "\033[93m",   # orange/yellow
        "Mercedes":        "\033[96m",   # cyan
        "Aston Martin":    "\033[92m",   # green
        "Alpine":          "\033[94m",   # blue
        "Williams":        "\033[94m",   # blue
        "Haas":            "\033[90m",   # grey
        "Racing Bulls":    "\033[94m",   # blue
        "Audi":            "\033[97m",   # white
        "Cadillac":        "\033[95m",   # magenta
    }

    @staticmethod
    def team(name: str, text: str) -> str:
        return C.TEAM_COLORS.get(name, C.WHITE) + text + C.RESET


# ── Constants ──────────────────────────────────────────────
STANDINGS_FILE   = "./data/standings_2026.json"
PREDICTIONS_LOG  = "./outputs/predictions_log.csv"
SEASON_2026_DATA = "./data/2026_laps.csv"
MODEL_LOG        = "./models/training_log.csv"
RETRAIN_THRESHOLD = 10   # races before model goes 2026-only


# ══════════════════════════════════════════════════════════
# STANDINGS  STATE
# ══════════════════════════════════════════════════════════

def load_standings() -> dict:
    """Load or initialise championship standings."""
    if os.path.exists(STANDINGS_FILE):
        with open(STANDINGS_FILE) as f:
            return json.load(f)
    return {
        "driver":       {},   # abbr → {"name": str, "team": str, "pts": int, "wins": int, "podiums": int, "dnfs": int}
        "constructor":  {},   # team  → {"pts": int, "wins": int}
        "races_done":   [],   # list of completed round numbers
        "last_updated": None,
    }


def save_standings(s: dict):
    os.makedirs("./data", exist_ok=True)
    with open(STANDINGS_FILE, "w") as f:
        json.dump(s, f, indent=2)


def update_standings_from_result(standings: dict, round_num: int,
                                 circuit: str, results: list[dict]) -> dict:
    """
    Update standings from a race result.
    results: list of dicts with keys: driver, team, position, fastest_lap, dnf
    Positions 1–10 score points. Fastest lap scores +1 if in top 10.
    """
    from config import POINTS_MAP, FASTEST_LAP_POINT

    if round_num in standings["races_done"]:
        return standings   # already processed

    for entry in results:
        drv  = entry["driver"]
        team = entry["team"]
        pos  = entry.get("position")
        fl   = entry.get("fastest_lap", False)
        dnf  = entry.get("dnf", False)

        # Driver standings
        if drv not in standings["driver"]:
            standings["driver"][drv] = {
                "team": team, "pts": 0, "wins": 0, "podiums": 0, "dnfs": 0
            }
        d = standings["driver"][drv]
        if pos and not dnf:
            pts = POINTS_MAP.get(pos, 0)
            if fl and pos <= 10:
                pts += FASTEST_LAP_POINT
            d["pts"]    += pts
            d["wins"]   += 1 if pos == 1 else 0
            d["podiums"]+= 1 if pos <= 3 else 0
        if dnf:
            d["dnfs"] += 1

        # Constructor standings
        if team not in standings["constructor"]:
            standings["constructor"][team] = {"pts": 0, "wins": 0}
        c = standings["constructor"][team]
        if pos and not dnf:
            c["pts"]  += POINTS_MAP.get(pos, 0)
            c["wins"] += 1 if pos == 1 else 0

    standings["races_done"].append(round_num)
    standings["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return standings


# ══════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def banner():
    print(f"""
{C.RED}{'═'*62}{C.RESET}
{C.BOLD}{C.WHITE}   🏎   F1 2026 RACE PREDICTOR   —   SEASON TRACKER   🏆{C.RESET}
{C.RED}{'═'*62}{C.RESET}""")


def section(title: str):
    print(f"\n{C.YELLOW}{'─'*62}{C.RESET}")
    print(f"{C.BOLD}{C.WHITE}  {title}{C.RESET}")
    print(f"{C.YELLOW}{'─'*62}{C.RESET}")


def info(msg: str):  print(f"  {C.CYAN}ℹ{C.RESET}  {msg}")
def ok(msg: str):    print(f"  {C.GREEN}✓{C.RESET}  {msg}")
def warn(msg: str):  print(f"  {C.YELLOW}⚠{C.RESET}  {msg}")
def err(msg: str):   print(f"  {C.RED}✗{C.RESET}  {msg}")
def step(n, total, msg: str):
    print(f"\n  {C.GREY}[{n}/{total}]{C.RESET} {C.BOLD}{msg}{C.RESET}")


def prompt(msg: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    try:
        val = input(f"\n  {C.GREEN}▶{C.RESET}  {msg}{hint}: ").strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        return default


def confirm(msg: str) -> bool:
    val = prompt(f"{msg} (y/n)", "y")
    return val.lower().startswith("y")


def spinner(msg: str, func, *args, **kwargs):
    """Run func() with a simple progress indicator."""
    frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    import threading, itertools

    result = [None]
    exc    = [None]

    def run():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=run)
    t.start()
    i = 0
    while t.is_alive():
        print(f"\r  {C.CYAN}{frames[i % len(frames)]}{C.RESET}  {msg}...", end="", flush=True)
        time.sleep(0.1)
        i += 1
    print(f"\r  {C.GREEN}✓{C.RESET}  {msg}   {' '*20}")
    t.join()
    if exc[0]:
        raise exc[0]
    return result[0]


# ══════════════════════════════════════════════════════════
# DISPLAY: CALENDAR
# ══════════════════════════════════════════════════════════

def show_calendar(standings: dict):
    from config import F1_2026_CALENDAR
    section("2026 SEASON CALENDAR")
    done = set(standings["races_done"])

    print(f"\n  {'Rd':<4} {'Circuit':<20} {'Date':<12} {'Sprint':<7} {'Status'}")
    print(f"  {'──':<4} {'──────────────────':<20} {'──────────':<12} {'──────':<7} {'──────'}")

    for r, city, _, date, sprint in F1_2026_CALENDAR:
        if r in done:
            status = f"{C.GREEN}Done ✓{C.RESET}"
        elif r == min(set(range(1, 25)) - done, default=None):
            status = f"{C.YELLOW}◀ Next{C.RESET}"
        else:
            status = f"{C.GREY}Upcoming{C.RESET}"
        sp = f"{C.CYAN}Sprint{C.RESET}" if sprint else ""
        print(f"  {r:<4} {city:<20} {date:<12} {sp:<14} {status}")


# ══════════════════════════════════════════════════════════
# DISPLAY: DRIVER STANDINGS
# ══════════════════════════════════════════════════════════

def show_driver_standings(standings: dict):
    section("DRIVER CHAMPIONSHIP STANDINGS")
    d = standings["driver"]
    if not d:
        warn("No races completed yet — standings will appear after Round 1")
        return

    rows = sorted(d.items(), key=lambda x: (-x[1]["pts"], -x[1]["wins"]))
    print(f"\n  {'Pos':<4} {'Driver':<6} {'Team':<22} {'Pts':>5} {'W':>3} {'Pod':>4} {'DNF':>4}")
    print(f"  {'───':<4} {'──────':<6} {'────────────────────':<22} {'───':>5} {'─':>3} {'───':>4} {'───':>4}")

    leader_pts = rows[0][1]["pts"] if rows else 0
    for i, (abbr, info_d) in enumerate(rows, 1):
        team  = info_d["team"]
        pts   = info_d["pts"]
        gap   = f"-{leader_pts - pts}" if i > 1 else "LEAD"
        pos_col = C.YELLOW if i == 1 else (C.WHITE if i <= 3 else C.GREY)
        name_col = C.team(team, abbr)
        print(f"  {pos_col}{i:<4}{C.RESET} {name_col:<17} "
              f"{C.DIM}{team:<22}{C.RESET} "
              f"{C.BOLD}{pts:>5}{C.RESET} "
              f"{info_d['wins']:>3} {info_d['podiums']:>4} "
              f"{C.RED if info_d['dnfs'] > 0 else C.GREY}{info_d['dnfs']:>4}{C.RESET}")

    races = len(standings["races_done"])
    print(f"\n  {C.DIM}After {races} race{'s' if races != 1 else ''}  "
          f"•  {24 - races} remaining{C.RESET}")


# ══════════════════════════════════════════════════════════
# DISPLAY: CONSTRUCTOR STANDINGS
# ══════════════════════════════════════════════════════════

def show_constructor_standings(standings: dict):
    section("CONSTRUCTOR CHAMPIONSHIP STANDINGS")
    c = standings["constructor"]
    if not c:
        warn("No races completed yet")
        return

    rows = sorted(c.items(), key=lambda x: (-x[1]["pts"], -x[1]["wins"]))
    print(f"\n  {'Pos':<4} {'Team':<26} {'Pts':>5} {'Wins':>5}")
    print(f"  {'───':<4} {'────────────────────────':<26} {'───':>5} {'────':>5}")

    leader_pts = rows[0][1]["pts"] if rows else 0
    for i, (team, cd) in enumerate(rows, 1):
        gap     = f"-{leader_pts - cd['pts']}" if i > 1 else "LEAD"
        pos_col = C.YELLOW if i == 1 else (C.WHITE if i <= 3 else C.GREY)
        tc      = C.team(team, f"{team:<26}")
        print(f"  {pos_col}{i:<4}{C.RESET} {tc} "
              f"{C.BOLD}{cd['pts']:>5}{C.RESET} {cd['wins']:>5}")


# ══════════════════════════════════════════════════════════
# DISPLAY: PREDICTION TOP 10
# ══════════════════════════════════════════════════════════

def show_prediction(round_num: int, quali_order: list[str],
                    weather: dict = None):
    """Run prediction for a given round and display the top 10."""
    from config import F1_2026_CALENDAR, F1_2026_DRIVERS

    cal = {r[0]: r for r in F1_2026_CALENDAR}
    if round_num not in cal:
        err(f"Round {round_num} not in 2026 calendar")
        return None

    _, city, _, race_date, _ = cal[round_num]

    import joblib
    from config import MODEL_PATH, SCALER_PATH, FEATURE_COLS
    from reliability import get_latest_scores

    if not os.path.exists(MODEL_PATH):
        err("No trained model found. Run option [S] Setup first.")
        return None

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Build scenario
    df = _build_scenario(city, quali_order, weather, round_num - 1)
    X  = df[FEATURE_COLS].fillna(0)
    df["raw_gap"] = model.predict(scaler.transform(X))

    # Reliability penalty
    PENALTY_SCALE = 5.0
    df["penalty"] = (
        0.6 * df["_eng_prob"].fillna(0.05) +
        0.4 * df["_drv_dnf"].fillna(0.08)
    ) * PENALTY_SCALE
    df["final_gap"] = df["raw_gap"] + df["penalty"]
    df = df.sort_values("final_gap").reset_index(drop=True)
    df["pos"] = range(1, len(df) + 1)

    top10 = df.head(10)
    leader_gap = top10.iloc[0]["final_gap"]

    # ── Print prediction table ──────────────────────────
    section(f"🏁  ROUND {round_num}: {city.upper()}  —  TOP 10 PREDICTION")
    print(f"  {C.DIM}Race date: {race_date}   "
          f"Model trained on: {_model_data_summary()}{C.RESET}\n")

    header = (f"  {'Pos':<4} {'Driver':<6} {'Team':<22} "
              f"{'Gap (s)':>8}  {'Rel%':>5}  {'Eng%':>5}")
    print(f"{C.BOLD}{header}{C.RESET}")
    print(f"  {'─'*60}")

    medals = {1: f"{C.YELLOW}🥇{C.RESET}", 2: f"{C.WHITE}🥈{C.RESET}",
              3: f"\033[38;5;208m🥉{C.RESET}"}

    for _, row in top10.iterrows():
        pos     = int(row["pos"])
        gap_str = "LEADER" if pos == 1 else f"+{row['final_gap'] - leader_gap:.3f}s"
        medal   = medals.get(pos, f"  {pos} ")
        tc      = C.team(row["Team"], row["Team"][:20])
        drv_col = C.BOLD + C.team(row["Team"], row["Driver"]) + C.RESET
        rel_pct = f"{row['_drv_dnf']*100:.1f}%"
        eng_pct = f"{row['_eng_prob']*100:.1f}%"
        gap_col = C.GREEN if pos == 1 else (C.YELLOW if pos <= 3 else C.WHITE)
        print(f"  {medal}  {drv_col:<17} {tc:<28} "
              f"{gap_col}{gap_str:>8}{C.RESET}  "
              f"{C.GREY}{rel_pct:>5}  {eng_pct:>5}{C.RESET}")

    print(f"\n  {C.DIM}Rel% = driver personal DNF rate  "
          f"•  Eng% = engine failure prob × circuit PU demand{C.RESET}")

    # Save to log
    _log_prediction(top10, round_num, city, race_date, leader_gap)
    return top10


def _model_data_summary() -> str:
    """One-line summary of what data the current model was trained on."""
    if not os.path.exists(MODEL_LOG):
        return "unknown"
    log = pd.read_csv(MODEL_LOG)
    if log.empty:
        return "unknown"
    last = log.iloc[-1]
    return f"R²={last['r2']:.3f}  MAE={last['mae']:.3f}s  @ {last['timestamp']}"


def _build_scenario(circuit: str, quali_order: list,
                    weather: dict, as_of_round: int) -> pd.DataFrame:
    """Build per-driver feature rows for prediction."""
    import joblib
    from config import FEATURE_COLS, F1_2026_DRIVERS
    from reliability import get_latest_scores

    # Circuit meta
    CIRCUIT_META = {
        "Australia":    {"tyre_life":15, "compound":2, "track_temp":38, "air_temp":22, "pu_demand":2},
        "China":        {"tyre_life":12, "compound":2, "track_temp":40, "air_temp":18, "pu_demand":2},
        "Japan":        {"tyre_life":18, "compound":1, "track_temp":35, "air_temp":16, "pu_demand":3},
        "Bahrain":      {"tyre_life":10, "compound":2, "track_temp":48, "air_temp":30, "pu_demand":3},
        "Saudi Arabia": {"tyre_life":20, "compound":1, "track_temp":32, "air_temp":28, "pu_demand":3},
        "Miami":        {"tyre_life":15, "compound":2, "track_temp":52, "air_temp":30, "pu_demand":2},
        "Canada":       {"tyre_life":22, "compound":1, "track_temp":38, "air_temp":24, "pu_demand":2},
        "Monaco":       {"tyre_life":25, "compound":1, "track_temp":42, "air_temp":22, "pu_demand":1},
        "Barcelona":    {"tyre_life":10, "compound":2, "track_temp":46, "air_temp":28, "pu_demand":2},
        "Austria":      {"tyre_life":15, "compound":2, "track_temp":44, "air_temp":24, "pu_demand":3},
        "Great Britain":{"tyre_life":15, "compound":2, "track_temp":36, "air_temp":20, "pu_demand":3},
        "Belgium":      {"tyre_life":20, "compound":1, "track_temp":32, "air_temp":20, "pu_demand":3},
        "Hungary":      {"tyre_life":10, "compound":2, "track_temp":50, "air_temp":30, "pu_demand":2},
        "Netherlands":  {"tyre_life":12, "compound":2, "track_temp":38, "air_temp":22, "pu_demand":2},
        "Italy":        {"tyre_life":20, "compound":1, "track_temp":40, "air_temp":26, "pu_demand":3},
        "Madrid":       {"tyre_life":14, "compound":2, "track_temp":44, "air_temp":28, "pu_demand":2},
        "Azerbaijan":   {"tyre_life":22, "compound":1, "track_temp":35, "air_temp":22, "pu_demand":3},
        "Singapore":    {"tyre_life":25, "compound":1, "track_temp":40, "air_temp":30, "pu_demand":1},
        "Austin":       {"tyre_life":16, "compound":2, "track_temp":44, "air_temp":28, "pu_demand":2},
        "Mexico City":  {"tyre_life":20, "compound":1, "track_temp":38, "air_temp":22, "pu_demand":2},
        "Sao Paulo":    {"tyre_life":15, "compound":2, "track_temp":42, "air_temp":26, "pu_demand":2},
        "Las Vegas":    {"tyre_life":22, "compound":1, "track_temp":18, "air_temp":12, "pu_demand":3},
        "Qatar":        {"tyre_life":10, "compound":2, "track_temp":50, "air_temp":28, "pu_demand":3},
        "Abu Dhabi":    {"tyre_life":15, "compound":2, "track_temp":38, "air_temp":26, "pu_demand":2},
    }
    STREET = {"Monaco","Singapore","Azerbaijan","Las Vegas","Saudi Arabia","Madrid"}
    DEG_CLASS = {
        "Bahrain":2,"Saudi Arabia":0,"Australia":1,"Japan":0,"China":2,"Miami":1,
        "Monaco":0,"Barcelona":2,"Canada":0,"Austria":1,"Great Britain":1,"Hungary":2,
        "Belgium":0,"Netherlands":2,"Italy":0,"Singapore":0,"Qatar":2,"Austin":1,
        "Mexico City":0,"Brazil":1,"Las Vegas":0,"Abu Dhabi":1,"Azerbaijan":0,"Madrid":1,
    }

    meta = CIRCUIT_META.get(circuit, CIRCUIT_META["Australia"])
    if weather:
        meta.update(weather)

    deg_class    = DEG_CLASS.get(circuit, 1)
    is_street    = 1 if circuit in STREET else 0
    circuit_enc  = _encode_val("circuit", circuit)
    rel          = get_latest_scores(as_of_round=as_of_round, year=2026)
    team_rel     = rel.get("team", {})
    drv_dnf      = rel.get("driver", {})
    eng_probs    = rel.get("engine_failure_prob", {})

    # Load driver baselines from features.csv
    driver_base = {}
    team_speeds = {}
    feats_path  = "./data/features.csv"
    if os.path.exists(feats_path):
        hist = pd.read_csv(feats_path,
                           usecols=["Driver","Team","driver_avg_gap_hist",
                                    "driver_consistency","driver_wet_skill",
                                    "SpeedST","SpeedI1","SpeedI2","SpeedFL"])
        driver_base = (hist.dropna(subset=["Driver"])
                       .groupby("Driver")
                       .first()
                       .to_dict("index"))
        team_speeds = (hist.dropna(subset=["Team"])
                       .groupby("Team")[["SpeedST","SpeedI1","SpeedI2","SpeedFL"]]
                       .mean()
                       .to_dict("index"))

    rows = []
    mid_lap   = 30
    total_lap = 57
    for pos, driver in enumerate(quali_order, 1):
        team  = F1_2026_DRIVERS.get(driver, "Unknown")
        db    = driver_base.get(driver, {})
        ts    = team_speeds.get(team, {"SpeedST":310,"SpeedI1":195,"SpeedI2":200,"SpeedFL":280})
        car_r = team_rel.get(team, 0.10)
        d_dnf = drv_dnf.get(driver, 0.08)
        e_prb = eng_probs.get(team, 0.05)
        fuel  = (total_lap - mid_lap) * 1.8

        rows.append({
            "Driver": driver, "Team": team,
            # model features
            "SpeedST":               ts.get("SpeedST", 310),
            "SpeedI1":               ts.get("SpeedI1", 195),
            "SpeedI2":               ts.get("SpeedI2", 200),
            "SpeedFL":               ts.get("SpeedFL", 280),
            "team_encoded":          _encode_val("Team", team),
            "car_reliability_score": car_r,
            "driver_encoded":        _encode_val("Driver", driver),
            "driver_avg_gap_hist":   db.get("driver_avg_gap_hist", 1.5),
            "driver_consistency":    db.get("driver_consistency", 0.5),
            "driver_wet_skill":      db.get("driver_wet_skill", 0),
            "stint_number":          2,
            "pit_count":             1,
            "undercut_window":       0,
            "overcut_window":        0,
            "total_race_laps":       total_lap,
            "fuel_load_proxy":       fuel,
            "grid_position":         pos,
            "gap_to_pole_quali":     (pos - 1) * 0.05,
            "TyreLife":              meta["tyre_life"],
            "tyre_compound_encoded": meta["compound"],
            "is_fresh_tyre":         0,
            "tyre_deg_rate":         0,
            "tyre_deg_class":        deg_class,
            "TrackStatus_encoded":   1,
            "is_safety_car_lap":     0,
            "incidents_in_race":     0,
            "position_delta_sc":     0,
            "last_pit_stop_time":    2.3,
            "avg_pit_time_team":     2.5,
            "pit_delta_vs_field":    0,
            "AirTemp":               meta["air_temp"],
            "TrackTemp":             meta["track_temp"],
            "Humidity":              50,
            "WindSpeed":             10,
            "WindDirection":         180,
            "Rainfall":              0,
            "track_temp_delta":      0,
            "circuit_encoded":       circuit_enc,
            "lap_number_in_session": mid_lap,
            # interaction features
            "tyre_temp_interaction":  meta["tyre_life"] * meta["track_temp"],
            "wet_deg_interaction":    0,
            "grid_street_interaction":pos * (1.5 if is_street else 1.0),
            "driver_tyre_interaction":db.get("driver_avg_gap_hist",1.5)*meta["tyre_life"],
            "speed_temp_interaction": ts.get("SpeedST",310) * meta["track_temp"],
            "fuel_stint_interaction": fuel * 2,
            # 2026 placeholders
            "overtake_mode_laps":    0,
            "active_aero_mode":      0,
            "energy_store_pct":      0,
            "fuel_remaining_kg":     0,
            # session type — always "race" for predictions
            "session_type_encoded":  _encode_val("session_type", "race"),
            # reliability (not model features — used for penalty)
            "_drv_dnf":  d_dnf,
            "_eng_prob": e_prb,
        })

    return pd.DataFrame(rows)


def _encode_val(col: str, value: str) -> int:
    import joblib
    path = f"./models/encoder_{col}.pkl"
    if os.path.exists(path):
        le = joblib.load(path)
        return int(le.transform([str(value)])[0]) if str(value) in le.classes_ else -1
    return 0


def _log_prediction(top10: pd.DataFrame, round_num: int,
                    city: str, race_date: str, leader_gap: float):
    os.makedirs("./outputs", exist_ok=True)
    log = top10[["pos","Driver","Team","final_gap","_drv_dnf","_eng_prob"]].copy()
    log.columns = ["Pos","Driver","Team","Gap_s","Driver_DNF_Rate","Engine_Prob"]
    log["Gap_to_Leader"] = (log["Gap_s"] - leader_gap).round(3)
    log["round"]         = round_num
    log["circuit"]       = city
    log["race_date"]     = race_date
    log["predicted_on"]  = datetime.now().strftime("%Y-%m-%d %H:%M")

    if os.path.exists(PREDICTIONS_LOG):
        existing = pd.read_csv(PREDICTIONS_LOG)
        existing = existing[existing["round"] != round_num]
        log = pd.concat([existing, log], ignore_index=True)
    log.to_csv(PREDICTIONS_LOG, index=False)


# ══════════════════════════════════════════════════════════
# TRAINING LOGIC
# ══════════════════════════════════════════════════════════

def get_completed_2026_rounds() -> list[int]:
    if not os.path.exists(SEASON_2026_DATA):
        return []
    df = pd.read_csv(SEASON_2026_DATA)
    return sorted(df["round"].dropna().unique().astype(int).tolist())


def retrain_model(completed_rounds: list[int], verbose: bool = True):
    """
    Intelligent retraining logic:

    Races 1–9:   train on historical (2022–2024) + 2026 races done
                 Reliability scores from historical DNF data + 2026 updates

    Race 10+:    FULL SWITCH — train ONLY on 2026 data
                 The model has seen enough 2026 car/reg behaviour
                 to stop relying on pre-2026 data entirely
                 Reliability recalculated from 2026 DNF data only
    """
    n_done = len(completed_rounds)

    if n_done >= RETRAIN_THRESHOLD:
        mode   = "2026-only"
        detail = (f"Race {n_done} ≥ {RETRAIN_THRESHOLD} — "
                  f"switching to 2026-only training (dropping pre-2026 history)")
    else:
        mode   = "historical+2026"
        detail = (f"Race {n_done} < {RETRAIN_THRESHOLD} — "
                  f"augmenting historical data with {n_done} completed 2026 races")

    if verbose:
        info(f"Training mode: {C.BOLD}{mode}{C.RESET}")
        info(detail)

    # Call the training script with the right flag
    flag = "--2026only" if n_done >= RETRAIN_THRESHOLD else "--update"
    result = subprocess.run(
        [sys.executable, "3_train_model.py", flag],
        capture_output=False
    )
    return result.returncode == 0


# ══════════════════════════════════════════════════════════
# RACE WORKFLOW  (the main action)
# ══════════════════════════════════════════════════════════

def run_post_race_workflow(standings: dict) -> dict:
    """
    Full post-race pipeline:
    1. Ask which round just finished
    2. Fetch race data from FastF1
    3. Update reliability scores (engine DNF focused)
    4. Retrain model (2026-only if ≥ 10 races done)
    5. Update championship standings
    6. Predict top 10 for next race
    """
    from config import F1_2026_CALENDAR

    done = set(standings["races_done"])
    cal  = {r[0]: r for r in F1_2026_CALENDAR}
    all_rounds = [r[0] for r in F1_2026_CALENDAR]

    # ── Step 1: which round? ──────────────────────────────
    section("POST-RACE UPDATE")
    remaining = [r for r in all_rounds if r not in done]
    if not remaining:
        ok("All 24 races completed! Season over.")
        return standings

    print(f"\n  Completed rounds : {sorted(done) if done else 'None yet'}")
    print(f"  Remaining rounds : {remaining[:6]}{'...' if len(remaining)>6 else ''}")

    round_input = prompt("Which round just finished?",
                         str(min(remaining)))
    try:
        completed_round = int(round_input)
    except ValueError:
        err("Invalid round number")
        return standings

    if completed_round not in cal:
        err(f"Round {completed_round} not found in 2026 calendar")
        return standings

    _, city, _, race_date, _ = cal[completed_round]
    print(f"\n  {C.BOLD}Round {completed_round}: {city}  ({race_date}){C.RESET}")

    # ── Step 2: fetch race data ───────────────────────────
    step(1, 5, f"Fetching race data from FastF1 — Round {completed_round}: {city}")
    result = subprocess.run(
        [sys.executable, "4_fetch_2026_race.py", str(completed_round)],
        capture_output=False
    )
    if result.returncode != 0:
        err("Data fetch failed. Race may not be available yet on FastF1.")
        if not confirm("Continue anyway with last known model?"):
            return standings

    # ── Step 3: update reliability ────────────────────────
    step(2, 5, "Updating engine & driver reliability scores")
    try:
        from reliability import append_2026_race
        append_2026_race(completed_round)
        ok("Reliability scores updated")
    except Exception as e:
        warn(f"Reliability update skipped: {e}")

    # ── Step 4: retrain ───────────────────────────────────
    completed_rounds = get_completed_2026_rounds()
    n_done = len(completed_rounds)
    step(3, 5,
         f"Retraining model "
         f"({'2026-ONLY MODE' if n_done >= RETRAIN_THRESHOLD else f'historical + {n_done} 2026 races'})")

    if n_done >= RETRAIN_THRESHOLD:
        print(f"\n  {C.CYAN}{'═'*50}{C.RESET}")
        print(f"  {C.BOLD}{C.YELLOW}  🔄  MILESTONE: {n_done} 2026 RACES COMPLETE{C.RESET}")
        print(f"  {C.WHITE}  Model now trains exclusively on 2026 data.{C.RESET}")
        print(f"  {C.WHITE}  Pre-2026 historical data is no longer used.{C.RESET}")
        print(f"  {C.CYAN}{'═'*50}{C.RESET}")

    retrain_model(completed_rounds)
    ok("Model retrained")

    # ── Step 5: update standings ──────────────────────────
    step(4, 5, "Updating championship standings")
    standings = _update_standings_from_fastf1(standings, completed_round, city)
    save_standings(standings)
    ok("Standings updated")

    # ── Step 6: predict next race ─────────────────────────
    next_round = completed_round + 1
    if next_round in cal:
        _, next_city, _, next_date, _ = cal[next_round]
        step(5, 5, f"Predicting Round {next_round}: {next_city}  ({next_date})")

        print(f"\n  {C.DIM}Enter qualifying order (space-separated driver codes).{C.RESET}")
        print(f"  {C.DIM}Leave blank to use estimated order.{C.RESET}")
        print(f"  {C.DIM}e.g.  NOR VER HAM LEC PIA ANT RUS ALO SAI STR{C.RESET}")
        quali_input = prompt("Qualifying order", "")
        quali_order = quali_input.split() if quali_input.strip() else None

        if not quali_order:
            from config import F1_2026_DRIVERS
            quali_order = list(F1_2026_DRIVERS.keys())
            warn("No qualifying order given — using estimated order")

        # Optional weather override
        if confirm("Enter custom weather for this race?"):
            weather = _get_weather_input()
        else:
            weather = None

        show_prediction(next_round, quali_order, weather)
    else:
        ok("Season complete — no more races to predict!")

    return standings


def _update_standings_from_fastf1(standings: dict,
                                   round_num: int, city: str) -> dict:
    """Pull actual race results from FastF1 and update standings."""
    try:
        import fastf1
        fastf1.Cache.enable_cache("./cache")
        session = fastf1.get_session(2026, round_num, "R")
        session.load(laps=False, telemetry=False, weather=False)
        results = session.results

        entries = []
        for _, row in results.iterrows():
            pos = row.get("ClassifiedPosition")
            try:
                pos = int(pos)
            except (ValueError, TypeError):
                pos = None

            entries.append({
                "driver":      row["Abbreviation"],
                "team":        row["TeamName"],
                "position":    pos,
                "fastest_lap": bool(row.get("FastestLap", False)),
                "dnf":         pos is None,
            })

        standings = update_standings_from_result(standings, round_num, city, entries)
        ok(f"Standings updated from FastF1 results")
    except Exception as e:
        warn(f"Could not fetch results from FastF1: {e}")
        warn("Standings not updated — you can update manually from the menu")
    return standings


def _get_weather_input() -> dict:
    """Prompt for weather override values."""
    print(f"  {C.DIM}Press Enter to keep defaults{C.RESET}")
    weather = {}
    try:
        at = prompt("Air temp (°C)", "")
        if at: weather["air_temp"] = float(at)
        tt = prompt("Track temp (°C)", "")
        if tt: weather["track_temp"] = float(tt)
        rain = prompt("Rain? (0=dry, 1=wet)", "0")
        weather["Rainfall"] = float(rain)
    except ValueError:
        warn("Invalid input — using defaults")
        return {}
    return weather


# ══════════════════════════════════════════════════════════
# SETUP  (first-time / re-run)
# ══════════════════════════════════════════════════════════

def run_setup():
    """First-time setup: fetch data → feature engineering → train."""
    section("INITIAL SETUP")
    print(f"""
  This will:
    1. Download 2021–2024 race data from FastF1  (~30–60 min, cached)
    2. Build features
    3. Train the base XGBoost model

  {C.DIM}The cache means step 1 only takes long once.
  After that, re-running setup is fast.{C.RESET}
""")
    if not confirm("Start setup?"):
        return

    scripts = [
        ("1_fetch_data.py",            "Fetching 2021–2024 race data"),
        ("2_feature_engineering.py",   "Building features"),
        ("3_train_model.py",           "Training base model"),
    ]

    for script, label in scripts:
        print(f"\n{'─'*58}")
        print(f"  {C.BOLD}{label}{C.RESET}")
        print(f"{'─'*58}")
        result = subprocess.run([sys.executable, script], capture_output=False)
        if result.returncode != 0:
            err(f"{script} failed. Check errors above.")
            return

    ok("Setup complete! You can now predict races.")


# ══════════════════════════════════════════════════════════
# MODEL STATUS
# ══════════════════════════════════════════════════════════

def show_model_status(standings: dict):
    section("MODEL STATUS")
    completed = get_completed_2026_rounds()
    n_done    = len(completed)

    import os
    model_exists = os.path.exists("./models/f1_model.pkl")
    data_exists  = os.path.exists("./data/features.csv")

    print(f"\n  Model file     : {'✓ exists' if model_exists else '✗ not found — run Setup'}")
    print(f"  Feature data   : {'✓ exists' if data_exists else '✗ not found — run Setup'}")

    if os.path.exists(MODEL_LOG):
        log  = pd.read_csv(MODEL_LOG)
        last = log.iloc[-1]
        print(f"\n  Last training  : {last['timestamp']}")
        print(f"  R²             : {C.BOLD}{last['r2']:.4f}{C.RESET}  ", end="")
        r2 = float(last["r2"])
        if r2 >= 0.90:   print(f"{C.GREEN}Excellent ✓{C.RESET}")
        elif r2 >= 0.85: print(f"{C.GREEN}Good ✓{C.RESET}")
        elif r2 >= 0.75: print(f"{C.YELLOW}Acceptable{C.RESET}")
        else:            print(f"{C.RED}Low — consider tuning{C.RESET}")
        print(f"  MAE            : {last['mae']:.3f}s  "
              f"{C.DIM}(avg error in predicted gap-to-pole){C.RESET}")

        if len(log) > 1:
            print(f"\n  Training history (last 5 runs):")
            for _, row in log.tail(5).iterrows():
                print(f"    {row['timestamp']}   R²={row['r2']:.4f}   MAE={row['mae']:.3f}s")

    print(f"\n  2026 races in model: {completed if completed else 'None yet'}")

    if n_done < RETRAIN_THRESHOLD:
        remaining = RETRAIN_THRESHOLD - n_done
        print(f"  Training mode  : {C.CYAN}Historical + 2026 augmentation{C.RESET}")
        print(f"  {C.DIM}→ {remaining} more races until 2026-only mode activates{C.RESET}")
    else:
        print(f"  Training mode  : {C.YELLOW}{C.BOLD}2026-ONLY  "
              f"(≥{RETRAIN_THRESHOLD} races — pre-2026 history dropped){C.RESET}")

    if model_exists and (not os.path.exists(MODEL_LOG) or
                         float(pd.read_csv(MODEL_LOG).iloc[-1]["r2"]) < 0.85):
        print(f"\n  {C.YELLOW}Tip: R² < 0.85 → run hyperparameter tuning:{C.RESET}")
        print(f"       python 3b_tune_hyperparams.py --quick")


# ══════════════════════════════════════════════════════════
# MANUAL PREDICT  (pick any round)
# ══════════════════════════════════════════════════════════

def run_manual_predict(standings: dict):
    from config import F1_2026_CALENDAR, F1_2026_DRIVERS

    section("MANUAL PREDICTION")
    cal = {r[0]: r for r in F1_2026_CALENDAR}

    print(f"\n  Available rounds: 1–24")
    round_input = prompt("Predict which round?", "1")
    try:
        round_num = int(round_input)
    except ValueError:
        err("Invalid input")
        return

    if round_num not in cal:
        err(f"Round {round_num} not in calendar")
        return

    _, city, _, _, _ = cal[round_num]
    print(f"\n  {C.BOLD}Round {round_num}: {city}{C.RESET}")
    print(f"  Enter qualifying order (space-separated driver codes).")
    print(f"  Leave blank for estimated order.")
    print(f"  e.g.  NOR VER HAM LEC PIA ANT RUS ALO SAI STR BOT BOR")

    quali_input = prompt("Qualifying order", "")
    quali_order = quali_input.split() if quali_input.strip() else list(F1_2026_DRIVERS.keys())

    if confirm("Enter custom weather?"):
        weather = _get_weather_input()
    else:
        weather = None

    show_prediction(round_num, quali_order, weather)


# ══════════════════════════════════════════════════════════
# MAIN MENU
# ══════════════════════════════════════════════════════════

def main():
    standings = load_standings()

    while True:
        clear()
        banner()

        # Quick status bar
        done    = len(standings["races_done"])
        next_r  = min((set(range(1, 25)) - set(standings["races_done"])), default=None)
        from config import F1_2026_CALENDAR
        cal     = {r[0]: r for r in F1_2026_CALENDAR}
        next_str = f"Round {next_r}: {cal[next_r][1]}" if next_r else "Season complete"

        print(f"\n  {C.DIM}Races done: {done}/24   "
              f"Next: {next_str}   "
              f"{'Model: ✓' if os.path.exists('./models/f1_model.pkl') else 'Model: ✗ run Setup'}"
              f"{C.RESET}")

        # Menu
        print(f"""
  {C.BOLD}MENU{C.RESET}
  ─────────────────────────────────────
  {C.GREEN}[1]{C.RESET}  Predict next race top 10
  {C.GREEN}[2]{C.RESET}  Post-race update  (fetch + retrain + standings + predict next)
  {C.YELLOW}[3]{C.RESET}  Driver championship standings
  {C.YELLOW}[4]{C.RESET}  Constructor standings
  {C.YELLOW}[5]{C.RESET}  Full season calendar
  {C.CYAN}[6]{C.RESET}  Model status & training history
  {C.CYAN}[7]{C.RESET}  Predict any specific race
  {C.MAGENTA}[S]{C.RESET}  Initial setup  (first time only)
  {C.RED}[Q]{C.RESET}  Quit
""")

        choice = prompt("Choose", "1").upper()

        if choice == "1":
            # Predict next race
            if next_r:
                print(f"\n  {C.DIM}Enter qualifying order or press Enter for estimate:{C.RESET}")
                qi = prompt("Qualifying order", "")
                from config import F1_2026_DRIVERS
                qo = qi.split() if qi.strip() else list(F1_2026_DRIVERS.keys())
                show_prediction(next_r, qo)
            else:
                warn("Season complete — no more races to predict")
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")

        elif choice == "2":
            standings = run_post_race_workflow(standings)
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")

        elif choice == "3":
            show_driver_standings(standings)
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")

        elif choice == "4":
            show_constructor_standings(standings)
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")

        elif choice == "5":
            show_calendar(standings)
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")

        elif choice == "6":
            show_model_status(standings)
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")

        elif choice == "7":
            run_manual_predict(standings)
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")

        elif choice == "S":
            run_setup()
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")

        elif choice == "Q":
            clear()
            print(f"\n  {C.GREEN}Goodbye! 🏎{C.RESET}\n")
            sys.exit(0)

        else:
            warn("Invalid option")
            time.sleep(0.8)


if __name__ == "__main__":
    main()
