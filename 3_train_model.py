# ============================================================
# 3_train_model.py  —  Train / retrain the XGBoost model
# ============================================================
# R² improvements in this file vs a basic train script:
#
#   [1] XGBoost instead of Linear Regression
#       Handles non-linear tyre deg curves, weather interactions.
#       Impact: 0.48 → ~0.75 R²
#
#   [2] Time-based split (train 2022+2023, test 2024)
#       More honest than random split — tests generalisation to
#       an unseen season, which is exactly what 2026 requires.
#       Also prevents data leakage (same-race laps in both splits).
#
#   [3] Early stopping
#       Finds the optimal number of trees automatically.
#       Avoids overfitting to training data.
#
#   [4] Per-season R² diagnostics
#       Shows whether the model generalises across years or
#       just memorises one season.
#
#   [5] Training log
#       Tracks MAE / R² improvement across every retrain.
#       Shows whether adding 2026 race data is helping.
#
# Usage:
#   python 3_train_model.py             ← historical data only
#   python 3_train_model.py --update    ← include completed 2026 races
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, os, sys

from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import mean_absolute_error, r2_score
from xgboost               import XGBRegressor

from config import (
    FEATURES_DATA_PATH, MODEL_PATH, SCALER_PATH,
    SEASON_2026_DATA, FEATURE_COLS, TARGET,
    XGBOOST_PARAMS, TRAIN_YEARS, VALIDATE_YEAR, RANDOM_STATE,
)

os.makedirs("./outputs", exist_ok=True)
os.makedirs("./models",  exist_ok=True)

SEP = "=" * 58


# ══════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════

def load_data(include_2026: bool, only_2026: bool = False) -> pd.DataFrame:
    """
    Load training data with three modes:

    only_2026=True  (--2026only flag, activated after race 10):
        Train ONLY on 2026 completed race data.
        Pre-2026 history is completely dropped.
        The model now reflects only what 2026 cars actually do.
        Uses last 2 rounds as validation set.

    include_2026=True  (--update flag, races 1–9):
        Train on historical 2022–2024 + completed 2026 races.
        Augments historical patterns with new-reg behaviour.

    neither flag (baseline):
        Train on historical 2022–2024 only.
    """
    if only_2026:
        if not os.path.exists(SEASON_2026_DATA):
            print("  ⚠ No 2026 data found — falling back to historical")
            only_2026 = False
        else:
            df_26  = pd.read_csv(SEASON_2026_DATA)
            rounds = sorted(df_26["round"].unique().tolist())
            print(f"  {C.YELLOW}2026-ONLY MODE{C.RESET}  — "
                  f"{len(df_26):,} laps from rounds {rounds}")
            print(f"  Pre-2026 historical data: DROPPED")
            df = df_26
            for col in FEATURE_COLS:
                if col not in df.columns:
                    df[col] = 0
            return df

    df = pd.read_csv(FEATURES_DATA_PATH)
    print(f"  Historical data  : {len(df):,} laps ({TRAIN_YEARS + [VALIDATE_YEAR]})")

    if include_2026 and os.path.exists(SEASON_2026_DATA):
        df_26  = pd.read_csv(SEASON_2026_DATA)
        rounds = sorted(df_26["round"].unique().tolist())
        print(f"  + 2026 data      : {len(df_26):,} laps  rounds {rounds}")
        df = pd.concat([df, df_26], ignore_index=True)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    return df


# ══════════════════════════════════════════════════════════
# [2] TIME-BASED SPLIT
# ══════════════════════════════════════════════════════════

def time_based_split(df: pd.DataFrame):
    """
    Train on TRAIN_YEARS, validate on VALIDATE_YEAR.

    Why time-based beats random split:
      - Random split leaks same-race laps into both train/test
        (laps 1–30 in train, laps 31–57 in test → artificially
        high R² because the model has "seen" the race conditions)
      - Time-based split tests whether the model predicts an
        ENTIRELY UNSEEN season — much more honest
      - Matches real use: you'll predict 2026 races having trained
        only on pre-2026 data
    """
    if "year" not in df.columns:
        from sklearn.model_selection import train_test_split
        print("  ⚠ No year column — falling back to random 80/20 split")
        X = df[FEATURE_COLS].fillna(0)
        y = df[TARGET].fillna(0)
        return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    train_df = df[df["year"].isin(TRAIN_YEARS)]
    val_df   = df[df["year"] == VALIDATE_YEAR]

    # Any extra years (e.g. 2026 completed races) go into training
    extra_df = df[~df["year"].isin(TRAIN_YEARS + [VALIDATE_YEAR])]
    if not extra_df.empty:
        extra_years = sorted(extra_df["year"].unique().tolist())
        print(f"  + Extra years in train: {extra_years}")
        train_df = pd.concat([train_df, extra_df], ignore_index=True)

    print(f"  Train : years {TRAIN_YEARS}  →  {len(train_df):,} laps")
    print(f"  Test  : year  {VALIDATE_YEAR}  →  {len(val_df):,} laps")

    X_train = train_df[FEATURE_COLS].fillna(0)
    y_train = train_df[TARGET].fillna(0)
    X_test  = val_df[FEATURE_COLS].fillna(0)
    y_test  = val_df[TARGET].fillna(0)

    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════
# DIAGNOSTICS
# ══════════════════════════════════════════════════════════

def print_r2_summary(r2: float, mae: float):
    """Print R² with interpretation and guidance."""
    print(f"\n  {'─'*40}")
    print(f"  MAE = {mae:.3f} s   (avg error in gap-to-pole prediction)")
    print(f"  R²  = {r2:.4f}")
    print(f"  {'─'*40}")
    if r2 >= 0.90:
        print("  ✅ Excellent (>0.90) — model explains >90% of variance")
    elif r2 >= 0.85:
        print("  ✅ Good (0.85–0.90) — suitable for race predictions")
    elif r2 >= 0.75:
        print("  ⚠  Acceptable (0.75–0.85)")
        print("     → Run: python 3b_tune_hyperparams.py")
    elif r2 >= 0.60:
        print("  ⚠  Weak (0.60–0.75)")
        print("     → Check: were noise filters applied in step 2?")
        print("     → Check: is TARGET = gap_to_pole (not LapTime_s)?")
    else:
        print("  ❌ Low (<0.60)")
        print("     → Re-run 2_feature_engineering.py from scratch")
        print("     → Check raw_laps.csv has enough races (≥ 2 seasons)")


def print_per_season_r2(model, scaler, df: pd.DataFrame):
    """
    R² broken down by season.
    If 2022 and 2023 are high but 2024 (test year) is low,
    the model is overfitting to training seasons.
    Target: all years within 0.05 of each other.
    """
    print("\n  Per-season R² (gap between seasons = overfitting signal):")
    for yr in sorted(df["year"].unique()):
        yr_df = df[df["year"] == yr]
        if len(yr_df) < 50:
            continue
        X = yr_df[FEATURE_COLS].fillna(0)
        y = yr_df[TARGET].fillna(0)
        preds = model.predict(scaler.transform(X))
        r2  = r2_score(y, preds)
        mae = mean_absolute_error(y, preds)
        split_label = "(train)" if yr in TRAIN_YEARS else "(test) "
        print(f"    {yr} {split_label}:  R²={r2:.4f}  MAE={mae:.3f}s  "
              f"n={len(yr_df):,}")


def plot_feature_importance(model, path):
    """Color-coded feature importance chart by factor group."""
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importance = importance.sort_values(ascending=True)

    # Color each bar by which factor group it belongs to
    factor_colors = {
        "#1E88E5": ["SpeedST","SpeedI1","SpeedI2","SpeedFL",
                    "team_encoded","car_reliability_score"],
        "#43A047": ["driver_encoded","driver_avg_gap_hist",
                    "driver_consistency","driver_wet_skill"],
        "#FB8C00": ["stint_number","pit_count","undercut_window",
                    "overcut_window","total_race_laps","fuel_load_proxy"],
        "#8E24AA": ["grid_position","gap_to_pole_quali"],
        "#E53935": ["TyreLife","tyre_compound_encoded","is_fresh_tyre",
                    "tyre_deg_rate","tyre_deg_class"],
        "#00ACC1": ["TrackStatus_encoded","is_safety_car_lap",
                    "incidents_in_race","position_delta_sc"],
        "#F4511E": ["last_pit_stop_time","avg_pit_time_team","pit_delta_vs_field"],
        "#6D4C41": ["AirTemp","TrackTemp","Humidity","WindSpeed","WindDirection",
                    "Rainfall","track_temp_delta","circuit_encoded",
                    "lap_number_in_session"],
        "#FF6F00": ["tyre_temp_interaction","wet_deg_interaction",
                    "grid_street_interaction","driver_tyre_interaction",
                    "speed_temp_interaction","fuel_stint_interaction"],
        "#757575": ["overtake_mode_laps","active_aero_mode",
                    "energy_store_pct","fuel_remaining_kg"],
    }
    color_labels = {
        "#1E88E5": "Car Performance",   "#43A047": "Driver Ability",
        "#FB8C00": "Strategy",          "#8E24AA": "Qualifying",
        "#E53935": "Tyre Management",   "#00ACC1": "Incidents / Luck",
        "#F4511E": "Pit Stop",          "#6D4C41": "Environment",
        "#FF6F00": "Interaction",        "#757575": "2026 Regs",
    }
    colors = []
    for feat in importance.index:
        c = "#BDBDBD"
        for hex_c, feats in factor_colors.items():
            if feat in feats:
                c = hex_c
                break
        colors.append(c)

    fig, ax = plt.subplots(figsize=(11, max(12, len(FEATURE_COLS) * 0.32)))
    ax.barh(importance.index, importance.values, color=colors)
    ax.set_title("Feature Importance — F1 2026 Predictor", fontsize=14, pad=12)
    ax.set_xlabel("XGBoost Importance Score")

    from matplotlib.patches import Patch
    legend = [Patch(color=c, label=lbl) for c, lbl in color_labels.items()]
    ax.legend(handles=legend, loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_diagnostics(y_test, preds, r2: float, path: str):
    """
    Two-panel diagnostic:
      Left  — predicted vs actual (tight cluster on diagonal = good)
      Right — residual histogram (bell curve centred on 0 = good)

    Fat tails in the residual plot reveal which lap types the
    model still struggles with (usually extreme weather or SC chaos).
    """
    residuals = np.array(y_test) - np.array(preds)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Predicted vs actual
    axes[0].scatter(y_test, preds, alpha=0.15, s=6, color="#1E88E5")
    lim = [0, max(float(np.max(y_test)), float(np.max(preds)))]
    axes[0].plot(lim, lim, "r--", lw=1, label="Perfect prediction")
    axes[0].set_xlabel("Actual gap to pole (s)")
    axes[0].set_ylabel("Predicted gap to pole (s)")
    axes[0].set_title(f"Predicted vs Actual  (R²={r2:.4f})")
    axes[0].legend(fontsize=8)

    # Residuals
    axes[1].hist(residuals, bins=80, color="#43A047", alpha=0.85, edgecolor="none")
    axes[1].axvline(0, color="red", lw=1.2, linestyle="--")
    axes[1].set_xlabel("Residual: actual − predicted (seconds)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(
        f"Residual Distribution\n"
        f"σ={residuals.std():.3f}s   "
        f"median={np.median(residuals):.3f}s   "
        f"p95={np.percentile(np.abs(residuals), 95):.3f}s"
    )

    plt.suptitle("Model Diagnostics — F1 2026 Predictor", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def save_training_log(mae: float, r2: float):
    """Track every training run's metrics so you can see improvement over time."""
    log_path = "./models/training_log.csv"
    row = pd.DataFrame([{
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "mae":       round(mae, 4),
        "r2":        round(r2, 4),
    }])
    if os.path.exists(log_path):
        log  = pd.read_csv(log_path)
        prev = log.iloc[-1]
        delta_r2  = r2  - prev["r2"]
        delta_mae = mae - prev["mae"]
        sign_r2   = "+" if delta_r2  >= 0 else ""
        sign_mae  = "+" if delta_mae >= 0 else ""
        print(f"\n  vs previous model:")
        print(f"    R²  {prev['r2']:.4f} → {r2:.4f}  ({sign_r2}{delta_r2:.4f})")
        print(f"    MAE {prev['mae']:.4f} → {mae:.4f}  ({sign_mae}{delta_mae:.4f}s)")
        log = pd.concat([log, row], ignore_index=True)
    else:
        print("  (First training run — no previous model to compare)")
        log = row
    log.to_csv(log_path, index=False)


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

# Simple colour codes for terminal output
class C:
    RESET  = "\033[0m"; BOLD = "\033[1m"
    YELLOW = "\033[93m"; CYAN = "\033[96m"; GREEN = "\033[92m"


def main():
    only_2026   = "--2026only" in sys.argv
    update_mode = "--update"   in sys.argv or only_2026

    if only_2026:
        label = "2026-ONLY  (pre-2026 history dropped)"
    elif update_mode:
        label = "historical + 2026 augmentation"
    else:
        label = "historical only"

    print(f"\n{SEP}")
    print(f"  Training XGBoost Model  [{label}]")
    print(SEP + "\n")

    # 1. Load — passes only_2026 flag
    df = load_data(include_2026=update_mode, only_2026=only_2026)
    print(f"  Total laps for training: {len(df):,}")

    # 2. Split
    # In 2026-only mode: use last 2 rounds as validation set
    if only_2026 and "round" in df.columns:
        rounds    = sorted(df["round"].unique())
        val_round = rounds[-2:] if len(rounds) >= 3 else rounds[-1:]
        trn_round = [r for r in rounds if r not in val_round]
        train_df  = df[df["round"].isin(trn_round)]
        val_df    = df[df["round"].isin(val_round)]
        print(f"  Train rounds : {trn_round}")
        print(f"  Val rounds   : {val_round}")
        X_train = train_df[FEATURE_COLS].fillna(0)
        y_train = train_df[TARGET].fillna(0)
        X_test  = val_df[FEATURE_COLS].fillna(0)
        y_test  = val_df[TARGET].fillna(0)
        print(f"  Train: {len(X_train):,} laps  |  Test: {len(X_test):,} laps")
    else:
        print(f"\n  Split strategy: time-based (not random)")
        X_train, X_test, y_train, y_test = time_based_split(df)

    # 3. Scale
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 4. Train
    print(f"\n  Training XGBoost "
          f"(max {XGBOOST_PARAMS['n_estimators']} trees, "
          f"early stop @{XGBOOST_PARAMS.get('early_stopping_rounds', 50)})...")

    model = XGBRegressor(**XGBOOST_PARAMS, verbosity=0)
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=False,
    )
    best_n = getattr(model, "best_iteration", XGBOOST_PARAMS["n_estimators"])
    print(f"  Best number of trees: {best_n}")

    # 5. Evaluate
    preds = model.predict(X_test_s)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)

    val_label = "2026 validation rounds" if only_2026 else f"{VALIDATE_YEAR} test season"
    print(f"\n  ── Results on {val_label} ──")
    print_r2_summary(r2, mae)

    # 6. Per-season / per-round breakdown
    print_per_season_r2(model, scaler, df)

    # 7. Save
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n  Model saved  → {MODEL_PATH}")
    print(f"  Scaler saved → {SCALER_PATH}")

    # 8. Training log
    save_training_log(mae, r2)

    # 9. Plots
    print("\n  Generating diagnostic plots...")
    plot_feature_importance(model, "./outputs/feature_importance.png")
    plot_diagnostics(y_test, preds, r2, "./outputs/residuals.png")

    print(f"\n{SEP}")
    print(f"  ✅ Done  MAE={mae:.3f}s  R²={r2:.4f}")
    if only_2026:
        print(f"  {C.YELLOW}2026-only model active — "
              f"pre-2026 history no longer used{C.RESET}")
    elif r2 < 0.85:
        print(f"  → Next step: python 3b_tune_hyperparams.py")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
