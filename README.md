# 🏎 F1 2026 Race Predictor

A supervised machine learning system that predicts the **top 10 finishing
positions** for every 2026 Formula 1 race. The model retrains after each
race using real FastF1 data and maintains live championship standings
throughout the season.

---

## Table of Contents

1. [What this does](#1-what-this-does)
2. [Requirements](#2-requirements)
3. [Installation](#3-installation)
4. [Project structure](#4-project-structure)
5. [First-time setup](#5-first-time-setup)
6. [Running the app](#6-running-the-app)
7. [Menu guide](#7-menu-guide)
8. [Your weekly workflow](#8-your-weekly-workflow)
9. [How the model works](#9-how-the-model-works)
10. [How training evolves across the season](#10-how-training-evolves-across-the-season)
11. [How reliability scoring works](#11-how-reliability-scoring-works)
12. [Understanding the prediction output](#12-understanding-the-prediction-output)
13. [Customisation](#13-customisation)
14. [Troubleshooting](#14-troubleshooting)
15. [File reference](#15-file-reference)

---

## 1. What this does

| Capability | Detail |
|---|---|
| Predicts top 10 | For every 2026 race, with gap-to-leader in seconds |
| Live standings | Driver and constructor championship, updates after each race |
| Incremental learning | Model retrains after every race using real 2026 data |
| Regulation-aware | Historical data remapped to 2026 grid; 2026 cars treated separately |
| Reliability scoring | Engine failure probability per team × circuit PU demand |
| 2026-only mode | After race 10, model drops all pre-2026 history automatically |
| Single command | `python run.py` — interactive terminal menu for everything |

---

## 2. Requirements

- **Python 3.10 or newer**
- Internet connection (for FastF1 to download race data)
- ~2–4 GB disk space (FastF1 cache of 3 seasons)
- Windows / macOS / Linux — all supported

Check your Python version:
```bash
python --version
```

If it shows Python 3.9 or older, install a newer version from python.org.

---

## 3. Installation

### Step 1 — Download the project files

Put all these files into one folder, e.g. `f1_predictor/`:

```
f1_predictor/
├── run.py
├── config.py
├── grid_normalizer.py
├── reliability.py
├── 1_fetch_data.py
├── 2_feature_engineering.py
├── 3_train_model.py
├── 3b_tune_hyperparams.py
├── 4_fetch_2026_race.py
├── 5_predict_top10.py
├── 6_race_workflow.py
└── requirements.txt
```

### Step 2 — Open a terminal in that folder

**Windows:**
- Open the folder in File Explorer
- Click the address bar, type `cmd`, press Enter

**macOS / Linux:**
```bash
cd ~/wherever/you/put/f1_predictor
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `fastf1`, `xgboost`, `scikit-learn`, `pandas`, `numpy`,
`matplotlib`, `joblib`.

If `pip` is not found, try `pip3` or `python -m pip`.

---

## 4. Project structure

```
f1_predictor/
│
├── run.py                    ← MAIN ENTRY POINT — run this
│
├── config.py                 ← All settings: seasons, features, thresholds
├── grid_normalizer.py        ← 2026 driver→team map + rookie baselines
├── reliability.py            ← Engine & driver reliability scoring
│
├── 1_fetch_data.py           ← Downloads 2022–2024 race data (runs once)
├── 2_feature_engineering.py  ← Builds all model features from raw data
├── 3_train_model.py          ← Trains / retrains the XGBoost model
├── 3b_tune_hyperparams.py    ← Optional: finds best hyperparameters
├── 4_fetch_2026_race.py      ← Fetches one completed 2026 race
├── 5_predict_top10.py        ← Generates top 10 prediction for a round
├── 6_race_workflow.py        ← (Legacy) chains fetch → retrain → predict
│
├── requirements.txt          ← Python package list
│
├── data/                     ← Created automatically
│   ├── raw_laps.csv          ← Historical lap data (2022–2024)
│   ├── features.csv          ← Engineered training features
│   ├── 2026_laps.csv         ← Grows as 2026 races complete
│   ├── standings_2026.json   ← Live championship standings
│   ├── dnf_raw.csv           ← Historical DNF data for reliability
│   └── reliability_scores.csv
│
├── models/                   ← Created automatically
│   ├── f1_model.pkl          ← Trained XGBoost model
│   ├── scaler.pkl            ← StandardScaler
│   ├── training_log.csv      ← R² and MAE for every training run
│   └── encoder_*.pkl         ← Label encoders for Driver / Team / circuit
│
├── cache/                    ← FastF1 cache (created automatically)
│
└── outputs/                  ← Created automatically
    ├── predictions_log.csv   ← Every top 10 prediction all season
    ├── feature_importance.png
    └── residuals.png
```

---

## 5. First-time setup

This only needs to be done **once at the start of the season**.

### Option A — via the menu (easiest)

```bash
python run.py
```

Choose `[S] Initial setup` from the menu. It will run all three steps
automatically and tell you when it is done.

### Option B — manually, step by step

```bash
# Step 1: Download 2022–2024 race data from FastF1
# This takes 30–60 minutes the first time (downloads ~3 seasons)
# All data is cached locally so it only runs slow once
python 1_fetch_data.py

# Step 2: Build features from the raw data
python 2_feature_engineering.py

# Step 3: Train the base model on historical data
python 3_train_model.py
```

**Expected output from Step 3:**
```
========================================================
  Training XGBoost Model  [historical only]
========================================================
  Historical data  : 284,xxx laps
  Train : [2022, 2023]  →  189,xxx laps
  Test  : 2024          →   94,xxx laps

  MAE = 0.312 s
  R²  = 0.8764
  ✅ Good — suitable for race predictions
```

If R² is below 0.85 after setup, run the optional tuning step:
```bash
python 3b_tune_hyperparams.py --quick   # ~5 min
python 3_train_model.py                 # retrain with found params
```

---

## 6. Running the app

```bash
python run.py
```

Every interaction this season happens through this one command.
The menu stays open until you press Q.

---

## 7. Menu guide

```
══════════════════════════════════════════════════════════════
   🏎   F1 2026 RACE PREDICTOR   —   SEASON TRACKER   🏆
══════════════════════════════════════════════════════════════

  Races done: 3/24   Next: Round 4: Bahrain   Model: ✓

  MENU
  ─────────────────────────────────────
  [1]  Predict next race top 10
  [2]  Post-race update  (fetch + retrain + standings + predict next)
  [3]  Driver championship standings
  [4]  Constructor standings
  [5]  Full season calendar
  [6]  Model status & training history
  [7]  Predict any specific race
  [S]  Initial setup  (first time only)
  [Q]  Quit
```

### [1] Predict next race top 10

Predicts the top 10 for the next race on the calendar.
You will be asked to enter the qualifying order if known.

```
Enter qualifying order (space-separated driver codes):
▶  NOR VER HAM LEC PIA ANT RUS ALO SAI STR
```

Driver codes are 3-letter abbreviations: VER, HAM, NOR, LEC, etc.
See the full list in the [Driver codes](#driver-codes) section.

Leave blank to use an estimated order based on current championship standings.

### [2] Post-race update

The main button you press every race weekend. Does everything automatically:

1. Asks which round just finished
2. Downloads race data from FastF1
3. Updates engine & driver reliability scores
4. Retrains the model
5. Updates championship standings from official results
6. Predicts the top 10 for the next race

Takes about 5–15 minutes depending on internet speed.

### [3] Driver championship standings

```
  ── DRIVER CHAMPIONSHIP STANDINGS ──────────────────────
  Pos  Driver  Team                   Pts    W   Pod   DNF
  ───  ──────  ─────────────────────  ───    ─   ───   ───
   1   NOR     McLaren                 75    3     3     0
   2   VER     Red Bull Racing         68   -7     2     1
   3   HAM     Ferrari                 61  -14     2     0
  ...
  After 3 races  •  21 remaining
```

Updates automatically after each post-race update (Option 2).

### [4] Constructor standings

Same as driver standings but for teams. Both drivers' points are
combined into the constructor total.

### [5] Full season calendar

Shows all 24 rounds with dates, sprint flag, and completion status.

### [6] Model status

Shows:
- Whether the model file exists
- Current R² and MAE
- Training history (last 5 runs)
- Which training mode is active (historical+2026 or 2026-only)
- How many races until 2026-only mode activates

### [7] Predict any specific race

Predict any round (1–24) regardless of whether it is the next one.
Useful for:
- Back-testing against a race that already happened
- Previewing later races in the season

### [S] Initial setup

Only needed once. Downloads historical data, builds features,
trains base model.

---

## 8. Your weekly workflow

### Race weekend (recommended sequence)

**Thursday / Friday (FP1–FP2)**
- No action needed yet.
- Optionally use `[7]` to preview the race with estimated quali order.

**Saturday after qualifying**
```
python run.py  →  [1]  Predict next race top 10
```
Enter the actual qualifying order when prompted.
This gives the most accurate pre-race prediction.

**Sunday after the race**
```
python run.py  →  [2]  Post-race update
```
Enter the round number that just finished.
The system handles everything from there.

**Typical session for a race weekend:**
```
1. python run.py
2. Press 1 → enter qualifying order → view top 10 prediction
3. (Race happens)
4. Press 2 → enter round number → system fetches, retrains, predicts next
5. Press 3 → view updated standings
6. Press Q
```

---

## 9. How the model works

### Algorithm
XGBoost (Extreme Gradient Boosting) — a tree-based ensemble model that
handles non-linear relationships. Chosen because lap time in F1 is
non-linear: tyre deg accelerates exponentially, temperature affects
different teams differently, reliability risk compounds over a stint.

### Target variable: `gap_to_pole`
The model predicts **how many seconds slower than the race leader** a
driver will run on a representative mid-race lap. This is better than
predicting raw lap time because:
- Monaco laps are ~75s, Monza ~82s, Bahrain ~92s
- `gap_to_pole` is always 0–5s regardless of circuit
- The model only needs to learn *relative* pace, not circuit baselines

### 42 features across 10 factor groups

| Group | Features | Key signal |
|---|---|---|
| Car performance | SpeedST, SpeedI1–FL, team_encoded | Power unit + aero |
| Driver ability | driver_encoded, driver_avg_gap_hist, consistency | Pure skill |
| Strategy | stint_number, pit_count, fuel_load_proxy | Race phase |
| Qualifying | grid_position, gap_to_pole_quali | Pre-race pace signal |
| Tyre management | TyreLife, compound, deg_rate, deg_class | Tyre state |
| Incidents | TrackStatus, SC laps, incidents_in_race | Safety car effect |
| Pit execution | last_pit_time, avg_pit_time_team | Stop quality |
| Environment | AirTemp, TrackTemp, Humidity, Rainfall | Conditions |
| Interactions | 6 cross-feature products | Non-linear effects |
| 2026 regs | overtake_mode, active_aero, ERS_pct | New regs (placeholder) |

### Noise filtering
Before training, these laps are removed because the model cannot learn
from them:
- Safety car and VSC laps (artificially slow)
- First 3 laps of a race (cold tyres + max fuel spike)
- Out-laps after a pit stop (tyre warming)
- Laps more than 8 seconds off the pace (DNF crawls, severe incidents)
- Top 2% of raw lap times (traffic, unreported incidents)

### Train / test split
The model is evaluated using **time-based splitting**, never random:
- Train on 2022 + 2023 seasons
- Test on 2024 season (completely unseen)

This is more honest than random splitting because it tests whether the
model generalises to an entire unseen season — which is what 2026
prediction requires.

---

## 10. How training evolves across the season

The model uses three different training modes depending on how many
2026 races have been completed:

### Mode 1 — Before any 2026 races
**Data:** 2022 + 2023 + 2024 historical races (all remapped to 2026 grid)

The model knows nothing about 2026 car behaviour yet. Predictions
rely entirely on historical driver and team patterns. Less accurate
for early-season races because 2026 regulations are completely new.

### Mode 2 — Races 1 to 9 (historical + 2026 augmentation)
**Data:** Historical 2022–2024 + completed 2026 races

As each 2026 race completes, it is added to the training set.
The model starts learning actual 2026 car behaviour — power unit
characteristics, tyre deg curves under new regulations, team
performance hierarchy. Predictions become more accurate each week.

### Mode 3 — Race 10 onwards (2026-only)
**Data:** Only the completed 2026 races (pre-2026 history dropped)

After 10 races the model has enough real 2026 data that historical
patterns from different regulations become noise rather than signal.
The model fully switches to 2026-only training. Validation uses the
last 2 completed rounds as the test set.

When this switch happens, the terminal shows:

```
  ══════════════════════════════════════════════════════
    🔄  MILESTONE: 10 2026 RACES COMPLETE
    Model now trains exclusively on 2026 data.
    Pre-2026 historical data is no longer used.
  ══════════════════════════════════════════════════════
```

### How accuracy is expected to improve

| Point in season | Training data | Expected R² |
|---|---|---|
| Before race 1 | Historical only | ~0.85–0.88 |
| After race 3 | Hist + 3 races | ~0.87–0.90 |
| After race 10 | 2026-only | ~0.88–0.92 |
| After race 20 | 2026-only | ~0.90–0.93 |

---

## 11. How reliability scoring works

Reliability is tracked separately from the main lap time model and
applied as a penalty to the predicted gap.

### Engine / car reliability (`reliability.py`)

For each team, the system computes a **rolling 10-race window** DNF rate.
DNF types are classified from FastF1 result status strings:

| Category | Status strings matched |
|---|---|
| Engine DNF | "power unit", "engine", "hybrid", "electrical", "MGU", "ERS", "turbo", "overheating", "hydraulics", "gearbox" |
| Mechanical DNF | All of the above + "suspension", "brakes", "driveshaft", "puncture" |
| Finished | "+1 lap", "+2 laps", classified positions |

`engine_failure_prob` multiplies the engine DNF rate by the circuit's
power unit demand level:

| Circuit PU demand | Multiplier | Examples |
|---|---|---|
| Low (1) | 0.8× | Monaco, Singapore, Hungary |
| Medium (2) | 1.0× | Australia, China, Abu Dhabi |
| High (3) | 1.3× | Japan, Monza, Spa, Austria |

So a team with a 15% engine DNF rate faces 15% × 1.3 = 19.5% risk at
Japan, but 15% × 0.8 = 12% risk at Monaco.

### Driver personal DNF rate

Separate from car reliability — captures driver errors and bad luck.
Computed as a rolling 10-race window of all DNF types for that driver.

### How it affects predictions

Both scores are combined into a gap penalty applied after the model
predicts the raw lap time gap:

```
dnf_combined   = 0.6 × engine_failure_prob + 0.4 × driver_dnf_rate
penalty_s      = dnf_combined × 5.0
final_gap      = raw_model_gap + penalty_s
```

A team with 20% combined risk gets +1.0s added to their gap, pushing
them lower in the predicted order. This reflects the reality that
unreliable cars are less likely to finish well even when fast.

### Updates after each race

When you run the post-race update (Option 2), the system automatically:
1. Downloads the race results from FastF1
2. Classifies each driver's retirement cause
3. Appends to the DNF history file
4. Recomputes rolling reliability scores
5. These updated scores are used in the next prediction

This means if a new team suffers two engine failures in the first four
races, their reliability risk increases and the model penalises them
more heavily in subsequent predictions.

---

## 12. Understanding the prediction output

```
══════════════════════════════════════════════════════════════════════
  🏁  2026 Round 4 — BAHRAIN  [RACE PREDICTION]
  📅  2026-04-12
  Model trained on: R²=0.891  MAE=0.287s  @ 2026-03-17 09:14
══════════════════════════════════════════════════════════════════════
  Pos  Driver  Team                  Gap (s)  Rel%   Eng%
────────────────────────────────────────────────────────────────────
🥇     NOR     McLaren               LEADER    2.1%   1.8%
🥈     VER     Red Bull Racing       +0.412    1.5%   2.3%
🥉     HAM     Ferrari               +0.638    3.2%   4.1%
  4    LEC     Ferrari               +0.821    2.8%   4.1%
  5    PIA     McLaren               +1.044    1.9%   1.8%
  6    ANT     Mercedes              +1.203    6.0%   2.9%
  7    RUS     Mercedes              +1.445    2.4%   2.9%
  8    ALO     Aston Martin          +1.712    4.1%   3.5%
  9    BOR     Audi                  +1.980    5.0%   8.2%
 10    SAI     Williams              +2.201    3.3%   2.1%
══════════════════════════════════════════════════════════════════════
  Rel% = driver personal DNF rate  •  Eng% = engine failure prob
══════════════════════════════════════════════════════════════════════
```

**Gap (s):** How many seconds behind the predicted race leader.
This is the model's predicted average gap during a representative
mid-race lap, adjusted for the reliability penalty.

**Rel%:** The driver's personal DNF rate from their last 10 races.
ANT at 6% is high because he is a rookie with limited data — will
decrease as real 2026 data comes in.

**Eng%:** The team's engine failure probability at this specific circuit,
accounting for that circuit's power unit demand level.
BOR/Audi at 8.2% reflects Bahrain being a high PU demand circuit and
Audi being a new manufacturer with unknown reliability baseline.

---

## 13. Customisation

### Change training seasons
In `config.py`:
```python
SEASONS = [2022, 2023, 2024]   # add or remove years
```

### Change noise filtering aggressiveness
In `config.py`:
```python
FILTER_MIN_LAP_NUMBER   = 3     # increase to 4 to skip more opening laps
FILTER_MAX_GAP_TO_POLE  = 8.0   # decrease to 5.0 for stricter outlier removal
FILTER_LAPTIME_QUANTILE = 0.98  # decrease to 0.95 for more aggressive filtering
```

### Change when 2026-only mode activates
In `run.py`:
```python
RETRAIN_THRESHOLD = 10   # change to 8 for earlier switch, 12 for later
```

### Update for mid-season driver changes
In `grid_normalizer.py`, update `DRIVER_2026_TEAM`:
```python
"NEW_DRIVER_CODE": "Team Name",   # add replacement driver
```

### Add weather forecast before a race
When predicting (Option 1 or 2), answer `y` when asked:
```
Enter custom weather for this race? (y/n): y
Air temp (°C): 34
Track temp (°C): 52
Rain? (0=dry, 1=wet): 0
```

### Run hyperparameter tuning
If R² is below 0.85:
```bash
python 3b_tune_hyperparams.py          # full search, ~30 min
python 3b_tune_hyperparams.py --quick  # fast search, ~5 min
python 3_train_model.py                # retrain with new params
```
This auto-updates `config.py` with the best parameters found.

---

## 14. Troubleshooting

### "No module named fastf1" / "No module named xgboost"
```bash
pip install -r requirements.txt
```

### First run is very slow
Normal. FastF1 downloads ~3 seasons of race data (2–4 GB).
This only happens once — all data is cached locally.
Subsequent runs are instant.

### "Session not available" error
The race has not been processed by FastF1 yet.
FastF1 data is usually available 1–2 hours after the race ends.
Wait and try again.

### "No trained model found"
You have not run setup yet.
```
python run.py → choose [S]
```

### R² is below 0.75 after setup
1. Check that `2_feature_engineering.py` completed without errors
2. Ensure `TARGET = "gap_to_pole"` in `config.py` (not `LapTime_s`)
3. Run hyperparameter tuning: `python 3b_tune_hyperparams.py --quick`

### Standings are wrong / not updating
Option 2 (post-race update) must be run after each completed race.
Standings only update when real FastF1 results are available.

### Windows terminal shows garbled characters instead of colours
The coloured output uses ANSI escape codes. Enable them in Windows:
```
Settings → Terminal → Use legacy console (uncheck it)
```
Or run in Windows Terminal instead of Command Prompt.

### "Round X not in 2026 calendar"
Check the round number. The 2026 season has rounds 1–24.
If a round was cancelled or rescheduled, update `F1_2026_CALENDAR`
in `config.py`.

---

## 15. File reference

| File | Edit? | Purpose |
|---|---|---|
| `run.py` | No | Main terminal UI — single entry point |
| `config.py` | Yes | Seasons, features, filter thresholds, hyperparams |
| `grid_normalizer.py` | If driver changes | 2026 grid, team aliases, rookie baselines |
| `reliability.py` | No | DNF classification, rolling reliability scores |
| `1_fetch_data.py` | No | Downloads 2022–2024 data via FastF1 |
| `2_feature_engineering.py` | No | Noise filtering, feature computation |
| `3_train_model.py` | No | XGBoost training, all three modes |
| `3b_tune_hyperparams.py` | No | Hyperparameter search (optional) |
| `4_fetch_2026_race.py` | No | Fetches one 2026 race, appends to dataset |
| `5_predict_top10.py` | No | Prediction logic, reliability penalty |
| `6_race_workflow.py` | No | Legacy CLI workflow (replaced by run.py) |

---

## Driver codes

| Code | Driver | Team |
|---|---|---|
| VER | Max Verstappen | Red Bull Racing |
| LAW | Liam Lawson | Red Bull Racing |
| HAM | Lewis Hamilton | Ferrari |
| LEC | Charles Leclerc | Ferrari |
| NOR | Lando Norris | McLaren |
| PIA | Oscar Piastri | McLaren |
| ANT | Andrea Kimi Antonelli | Mercedes |
| RUS | George Russell | Mercedes |
| ALO | Fernando Alonso | Aston Martin |
| STR | Lance Stroll | Aston Martin |
| GAS | Pierre Gasly | Alpine |
| COL | Franco Colapinto | Alpine |
| ALB | Alexander Albon | Williams |
| SAI | Carlos Sainz | Williams |
| BEA | Oliver Bearman | Haas |
| OCO | Esteban Ocon | Haas |
| TSU | Yuki Tsunoda | Racing Bulls |
| HAD | Isack Hadjar | Racing Bulls |
| BOT | Valtteri Bottas | Audi |
| BOR | Gabriel Bortoleto | Audi |
| DOO | Jack Doohan | Cadillac |
| MAL | Marcus Armstrong | Cadillac |

---

## Quick command reference

```bash
# Everything (use this all season)
python run.py

# Initial setup only
python 1_fetch_data.py
python 2_feature_engineering.py
python 3_train_model.py

# Optional: improve R² if below 0.85
python 3b_tune_hyperparams.py --quick
python 3_train_model.py

# Manual individual steps (if needed)
python 4_fetch_2026_race.py 3          # fetch round 3
python 3_train_model.py --update       # retrain with 2026 data
python 3_train_model.py --2026only     # retrain on 2026 data only
python 5_predict_top10.py 4 NOR VER HAM LEC  # predict round 4
```
