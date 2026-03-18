# ============================================================
# config.py  —  Master configuration for F1 2026 Predictor
# ============================================================

# ── Seasons for base historical training ──────────────────
SEASONS = [2022, 2023, 2024]

# ── Paths ─────────────────────────────────────────────────
CACHE_DIR          = "./cache"
RAW_DATA_PATH      = "./data/raw_laps.csv"
FEATURES_DATA_PATH = "./data/features.csv"
SEASON_2026_DATA   = "./data/2026_laps.csv"
MODEL_PATH         = "./models/f1_model.pkl"
SCALER_PATH        = "./models/scaler.pkl"
PREDICTIONS_LOG    = "./outputs/predictions_log.csv"
STANDINGS_LOG      = "./outputs/standings_log.csv"

# ── 2026 Season Calendar ───────────────────────────────────
# (round, city, fastf1_name, date, has_sprint)
F1_2026_CALENDAR = [
    (1,  "Australia",     "Australia",    "2026-03-08", False),
    (2,  "China",         "China",        "2026-03-15", True),
    (3,  "Japan",         "Japan",        "2026-03-29", False),
    (4,  "Bahrain",       "Bahrain",      "2026-04-12", False),
    (5,  "Saudi Arabia",  "Saudi Arabia", "2026-04-19", False),
    (6,  "Miami",         "Miami",        "2026-05-03", True),
    (7,  "Canada",        "Canada",       "2026-05-24", True),
    (8,  "Monaco",        "Monaco",       "2026-06-07", False),
    (9,  "Barcelona",     "Spain",        "2026-06-14", False),
    (10, "Austria",       "Austria",      "2026-06-28", False),
    (11, "Great Britain", "Great Britain","2026-07-05", True),
    (12, "Belgium",       "Belgium",      "2026-07-19", False),
    (13, "Hungary",       "Hungary",      "2026-07-26", False),
    (14, "Netherlands",   "Netherlands",  "2026-08-23", True),
    (15, "Italy",         "Italy",        "2026-09-06", False),
    (16, "Madrid",        "Spain",        "2026-09-13", False),
    (17, "Azerbaijan",    "Azerbaijan",   "2026-09-26", False),
    (18, "Singapore",     "Singapore",    "2026-10-11", True),
    (19, "Austin",        "USA",          "2026-10-25", False),
    (20, "Mexico City",   "Mexico",       "2026-11-01", False),
    (21, "Sao Paulo",     "Brazil",       "2026-11-08", False),
    (22, "Las Vegas",     "Las Vegas",    "2026-11-21", False),
    (23, "Qatar",         "Qatar",        "2026-11-29", False),
    (24, "Abu Dhabi",     "Abu Dhabi",    "2026-12-06", False),
]

# ── 2026 Driver → Team mapping ─────────────────────────────
# Single source of truth lives in grid_normalizer.py
from grid_normalizer import DRIVER_2026_TEAM as F1_2026_DRIVERS

# ── F1 Points system ───────────────────────────────────────
POINTS_MAP = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
              6: 8,  7: 6,  8: 4,  9: 2,  10: 1}
SPRINT_POINTS_MAP = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
FASTEST_LAP_POINT = 1

# ── Target variable ────────────────────────────────────────
# gap_to_pole = seconds behind the fastest lap on that lap number.
# Always 0–5s regardless of circuit → model doesn't waste capacity
# learning circuit baseline times. Biggest single R² improvement.
TARGET = "gap_to_pole"

# ── Noise filtering thresholds ─────────────────────────────
# Applied in 2_feature_engineering.py BEFORE training.
# Removing these laps is one of the highest-impact R² improvements.
#
# Why each filter matters:
#   MIN_LAP_NUMBER  — first 3 laps have cold tyres + max fuel → outliers
#   MIN_TYRE_LIFE   — lap 1 on a tyre = out-lap warming lap → not representative
#   MAX_GAP_TO_POLE — >8s off pace = DNF crawl / massive incident → not learnable
#   LAPTIME_QUANTILE— top 2% raw times = traffic / error / SC without status update
#   TRACK_STATUS    — SC/VSC laps are artificially slow and corrupt deg learning
FILTER_MIN_LAP_NUMBER   = 3
FILTER_MIN_TYRE_LIFE    = 2
FILTER_MAX_GAP_TO_POLE  = 8.0
FILTER_LAPTIME_QUANTILE = 0.98
FILTER_TRACK_STATUS     = [1, 2]   # 1=clear, 2=yellow. Drop 3=SC, 4=red, 5=VSC.

# ── Feature columns (42 total) ─────────────────────────────
# Grouped by the 10 factors that determine F1 lap time.
# Order does not matter for XGBoost but grouping aids readability.
FEATURE_COLS = [

    # ── 1. CAR PERFORMANCE ────────────────────────────────
    "SpeedST",               # Speed trap → power unit output proxy
    "SpeedI1",               # Sector 1 speed → aero efficiency
    "SpeedI2",               # Sector 2 speed
    "SpeedFL",               # Final sector speed → mechanical grip
    "team_encoded",          # Car baseline (aero package + PU)
    "car_reliability_score", # Team rolling DNF rate (from reliability.py)

    # ── 2. DRIVER ABILITY ─────────────────────────────────
    "driver_encoded",        # Driver identity
    "driver_avg_gap_hist",   # Career avg gap to pole — strongest driver feature
    "driver_consistency",    # Lap time std dev (lower = more consistent)
    "driver_wet_skill",      # Delta gap in wet vs dry (positive = better in wet)

    # ── 3. STRATEGY ───────────────────────────────────────
    "stint_number",          # 1st/2nd/3rd stint — strategy phase
    "pit_count",             # Total stops made so far
    "undercut_window",       # 1 if pitted 1–2 laps before a nearby rival
    "overcut_window",        # 1 if stayed out 1–2 laps longer than rival
    "total_race_laps",       # Circuit length proxy
    "fuel_load_proxy",       # (total_laps − lap_num) × 1.8 kg/lap

    # ── 4. QUALIFYING ─────────────────────────────────────
    "grid_position",         # Starting grid slot (1=pole) — strong pace proxy
    "gap_to_pole_quali",     # Quali delta in seconds — direct pace signal

    # ── 5. TYRE MANAGEMENT ────────────────────────────────
    "TyreLife",              # Laps on current set
    "tyre_compound_encoded", # SOFT=1 MEDIUM=2 HARD=3 INTER=4 WET=5
    "is_fresh_tyre",         # 1 = new unused set
    "tyre_deg_rate",         # Lap time slope per tyre lap (per-stint)
    "tyre_deg_class",        # Circuit severity: 0=low 1=medium 2=high

    # ── 6. INCIDENTS / LUCK ───────────────────────────────
    "TrackStatus_encoded",   # 1=clear 2=yellow 3=SC 4=red 5=VSC
    "is_safety_car_lap",     # Binary SC/VSC flag for this lap
    "incidents_in_race",     # Cumulative incidents up to this lap
    "position_delta_sc",     # Positions gained/lost at last SC restart

    # ── 7. PIT STOP EXECUTION ─────────────────────────────
    "last_pit_stop_time",    # Duration of most recent stop (seconds)
    "avg_pit_time_team",     # Team's rolling average stop time this season
    "pit_delta_vs_field",    # Last stop time minus field average

    # ── 8. ENVIRONMENT ────────────────────────────────────
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "WindSpeed",
    "WindDirection",
    "Rainfall",
    "track_temp_delta",      # Track temp change since lap 1 (rubbering-in)
    "circuit_encoded",
    "lap_number_in_session",

    # ── 9. INTERACTION FEATURES ───────────────────────────
    # Explicitly constructed cross-feature signals.
    # These give XGBoost better split candidates for non-linear effects.
    "tyre_temp_interaction",    # TyreLife × TrackTemp — deg accelerates in heat
    "wet_deg_interaction",      # Rainfall × tyre_deg_class — wet on high-deg = chaos
    "grid_street_interaction",  # grid_position × street_factor — overtaking penalty
    "driver_tyre_interaction",  # driver_avg_gap_hist × TyreLife — elite tyre mgmt
    "speed_temp_interaction",   # SpeedST × TrackTemp — PU output drops in heat
    "fuel_stint_interaction",   # fuel_load_proxy × stint_number — fuel effect by phase

    # ── 10. 2026 REGULATION PLACEHOLDERS ──────────────────
    # All set to 0 until FastF1 logs 2026 telemetry channels.
    # Already in feature set so model starts using them immediately
    # once real data arrives — no code change required.
    "overtake_mode_laps",    # Laps with Overtake Mode active (replaces DRS)
    "active_aero_mode",      # 0=efficiency wing, 1=high downforce mode
    "energy_store_pct",      # ERS charge % (MGU-K only from 2026)
    "fuel_remaining_kg",     # Actual fuel load (smaller tanks in 2026)
]

# ── XGBoost hyperparameters ────────────────────────────────
# Default best-practice values. Run 3b_tune_hyperparams.py once
# after initial training to find optimal values for your dataset
# and update these.
XGBOOST_PARAMS = {
    "n_estimators":          1000,   # max trees; early stopping finds the real best
    "max_depth":             6,      # depth 6 = good balance, avoids overfitting
    "learning_rate":         0.02,   # low LR + many trees = better generalisation
    "subsample":             0.8,    # row sampling per tree → reduces overfitting
    "colsample_bytree":      0.75,   # feature sampling per tree → reduces overfitting
    "min_child_weight":      3,      # min samples per leaf → avoids tiny splits
    "gamma":                 0.1,    # min loss reduction to make a split
    "reg_alpha":             0.05,   # L1 regularisation
    "reg_lambda":            1.2,    # L2 regularisation
    "random_state":          42,
    "tree_method":           "hist", # faster training, same accuracy as exact
    "early_stopping_rounds": 50,     # stop if no improvement for 50 rounds
}

# ── Train / test split strategy ───────────────────────────
# TIME-BASED: train on 2022+2023, validate on 2024.
# Never random-split F1 data — laps from the same race would
# appear in both train and test, inflating R².
TRAIN_YEARS   = [2022, 2023]
VALIDATE_YEAR = 2024

TEST_SIZE    = 0.2      # fallback if year column is missing
RANDOM_STATE = 42
