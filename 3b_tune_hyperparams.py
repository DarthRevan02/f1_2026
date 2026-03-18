# ============================================================
# 3b_tune_hyperparams.py  —  Find optimal XGBoost parameters
# ============================================================
# Run this ONCE after initial training if R² < 0.85.
# It searches the hyperparameter space and updates config.py
# with the best-found values automatically.
#
# Uses RandomizedSearchCV (faster than grid search).
# Takes ~15–40 minutes depending on your CPU.
#
# Usage:
#   python 3b_tune_hyperparams.py
#   python 3b_tune_hyperparams.py --quick   ← fewer iterations, ~5 min
#
# Output:
#   - Prints best parameters
#   - Updates XGBOOST_PARAMS in config.py automatically
#   - Saves tuning results to ./outputs/tuning_results.csv
# ============================================================

import pandas as pd
import numpy as np
import joblib, os, sys

from sklearn.preprocessing    import StandardScaler
from sklearn.model_selection  import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics          import make_scorer, r2_score
from xgboost                  import XGBRegressor

from config import (
    FEATURES_DATA_PATH, FEATURE_COLS, TARGET,
    TRAIN_YEARS, VALIDATE_YEAR, RANDOM_STATE,
)

os.makedirs("./outputs", exist_ok=True)

SEP = "=" * 58


def load_train_data() -> tuple:
    """Load training data only (not validation year)."""
    df = pd.read_csv(FEATURES_DATA_PATH)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    train_df = df[df["year"].isin(TRAIN_YEARS)] if "year" in df.columns else df
    X = train_df[FEATURE_COLS].fillna(0)
    y = train_df[TARGET].fillna(0)
    print(f"  Training data: {len(X):,} laps from years {TRAIN_YEARS}")
    return X, y


def run_tuning(X, y, quick: bool = False) -> dict:
    """
    RandomizedSearchCV over XGBoost hyperparameters.

    Uses TimeSeriesSplit so validation folds are always
    chronologically after training folds — no data leakage.

    n_iter=20 (quick) or n_iter=60 (full):
      quick = ~5 min   — useful for a first pass
      full  = ~30 min  — for final tuning
    """
    param_distributions = {
        "n_estimators":      [400, 600, 800, 1000, 1200],
        "max_depth":         [4, 5, 6, 7, 8],
        "learning_rate":     [0.005, 0.01, 0.02, 0.03, 0.05],
        "subsample":         [0.7, 0.75, 0.8, 0.85, 0.9],
        "colsample_bytree":  [0.65, 0.7, 0.75, 0.8, 0.85],
        "min_child_weight":  [2, 3, 4, 5, 7],
        "gamma":             [0.0, 0.05, 0.1, 0.15, 0.2],
        "reg_alpha":         [0.0, 0.01, 0.05, 0.1, 0.2],
        "reg_lambda":        [0.8, 1.0, 1.2, 1.5, 2.0],
    }

    n_iter = 20 if quick else 60
    cv     = TimeSeriesSplit(n_splits=3)  # time-aware CV — no leakage

    base_model = XGBRegressor(
        tree_method="hist",
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=make_scorer(r2_score),
        n_jobs=-1,          # use all CPU cores
        random_state=RANDOM_STATE,
        verbose=2,
        refit=True,
    )

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    print(f"\n  Running {n_iter} iterations × {cv.n_splits} folds "
          f"= {n_iter * cv.n_splits} model fits...")
    print("  (This will take a few minutes — grab a coffee)\n")

    search.fit(X_s, y)

    return search, scaler


def update_config(best_params: dict):
    """
    Automatically write the best-found hyperparameters back into config.py.
    Reads the file as text, replaces the XGBOOST_PARAMS block.
    """
    config_path = "./config.py"

    # Build the new XGBOOST_PARAMS block
    lines = [
        "XGBOOST_PARAMS = {\n",
        f"    \"n_estimators\":          {best_params.get('n_estimators', 1000)},\n",
        f"    \"max_depth\":             {best_params.get('max_depth', 6)},\n",
        f"    \"learning_rate\":         {best_params.get('learning_rate', 0.02)},\n",
        f"    \"subsample\":             {best_params.get('subsample', 0.8)},\n",
        f"    \"colsample_bytree\":      {best_params.get('colsample_bytree', 0.75)},\n",
        f"    \"min_child_weight\":      {best_params.get('min_child_weight', 3)},\n",
        f"    \"gamma\":                 {best_params.get('gamma', 0.1)},\n",
        f"    \"reg_alpha\":             {best_params.get('reg_alpha', 0.05)},\n",
        f"    \"reg_lambda\":            {best_params.get('reg_lambda', 1.2)},\n",
        "    \"random_state\":          42,\n",
        "    \"tree_method\":           \"hist\",\n",
        "    \"early_stopping_rounds\": 50,\n",
        "}\n",
    ]
    new_block = "".join(lines)

    with open(config_path, "r") as f:
        content = f.read()

    # Find and replace the XGBOOST_PARAMS block
    import re
    pattern = r"XGBOOST_PARAMS\s*=\s*\{[^}]+\}"
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_block.rstrip("\n"), content, flags=re.DOTALL)
        with open(config_path, "w") as f:
            f.write(content)
        print(f"  ✅ config.py updated with best parameters")
    else:
        print("  ⚠ Could not auto-update config.py — copy params manually:")
        print(new_block)


def main():
    quick = "--quick" in sys.argv

    print(f"\n{SEP}")
    print(f"  Hyperparameter Tuning  {'[QUICK MODE]' if quick else '[FULL MODE]'}")
    print(SEP + "\n")

    # Load
    X, y = load_train_data()

    # Tune
    search, scaler = run_tuning(X, y, quick=quick)

    # Results
    best_params = search.best_params_
    best_r2     = search.best_score_

    print(f"\n{SEP}")
    print(f"  Best CV R² : {best_r2:.4f}")
    print(f"  Best params:")
    for k, v in sorted(best_params.items()):
        print(f"    {k:<25} = {v}")
    print(SEP)

    # Save full results CSV
    results = pd.DataFrame(search.cv_results_)
    results = results.sort_values("rank_test_score")[
        ["rank_test_score", "mean_test_score", "std_test_score", "params"]
    ]
    results.to_csv("./outputs/tuning_results.csv", index=False)
    print(f"\n  Full results → ./outputs/tuning_results.csv")

    # Update config.py automatically
    print("\n  Updating config.py...")
    update_config(best_params)

    print(f"\n  Now retrain with the new params:")
    print(f"  python 3_train_model.py")
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
