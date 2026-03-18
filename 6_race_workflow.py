# ============================================================
# 6_race_workflow.py  —  YOUR MAIN SCRIPT after every race
# ============================================================
#
# After each race completes, run ONE command:
#
#   python 6_race_workflow.py <completed_round> [quali_order...]
#
# What it does automatically:
#   1. Fetches the completed race from FastF1
#   2. Retrains model with all 2026 data accumulated so far
#   3. Predicts top 10 for the NEXT race
#   4. Prints updated driver championship standings
#
# Examples:
#   After Australia, no China quali yet:
#     python 6_race_workflow.py 1
#
#   After Australia, China quali known:
#     python 6_race_workflow.py 1 RUS ANT LEC HAM NOR PIA VER LAW ALO STR BOT BOR
#
#   After China, Japan quali known:
#     python 6_race_workflow.py 2 VER NOR HAM LEC PIA ANT RUS ALO STR BOT
#
# ── Weather override (optional) ────────────────────────────
#   Edit WEATHER_OVERRIDES below to inject forecast data
#   for the next race before running the script.
# ============================================================

import sys, os, subprocess
from config import F1_2026_CALENDAR


# ── Optional: inject weather forecast for upcoming race ────
# Keys = city name, values = weather dict
# Update before each race weekend if forecast is available
WEATHER_OVERRIDES = {
    # Example:
    # "China": {"AirTemp": 16, "TrackTemp": 28, "Humidity": 70,
    #           "WindSpeed": 18, "Rainfall": 1},
}


def run(cmd: list[str]):
    print(f"\n  $ {' '.join(cmd)}")
    print(f"  {'─'*50}")
    subprocess.run(cmd, check=True)


def get_next(completed: int):
    cal = {r[0]: r for r in F1_2026_CALENDAR}
    return cal.get(completed + 1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python 6_race_workflow.py <completed_round> [quali_order...]")
        print("  e.g. python 6_race_workflow.py 1 RUS ANT LEC HAM NOR PIA VER LAW")
        sys.exit(1)

    completed    = int(sys.argv[1])
    quali_order  = sys.argv[2:]

    cal = {r[0]: r for r in F1_2026_CALENDAR}
    done_city = cal.get(completed, (None,"?"))[1]
    next_race = get_next(completed)

    print(f"\n{'═'*60}")
    print(f"  🔄  F1 2026 Race Workflow")
    print(f"  ✅  Completed : Round {completed} — {done_city}")
    if next_race:
        print(f"  🔮  Predicting: Round {next_race[0]} — {next_race[1]}")
    print(f"{'═'*60}")

    # ── Step 1: Fetch completed race ──────────────────────
    print(f"\n[1/3]  Fetching race data for Round {completed} ({done_city})...")
    run(["python", "4_fetch_2026_race.py", str(completed)])

    # ── Step 2: Retrain with accumulated 2026 data ────────
    print(f"\n[2/3]  Retraining model (historical + all 2026 races up to R{completed})...")
    run(["python", "3_train_model.py", "--update"])

    # ── Step 3: Predict next race ─────────────────────────
    if next_race:
        next_round = next_race[0]
        next_city  = next_race[1]
        print(f"\n[3/3]  Predicting top 10 for Round {next_round} ({next_city})...")

        predict_cmd = ["python", "5_predict_top10.py", str(next_round)] + list(quali_order)
        run(predict_cmd)
    else:
        print("\n[3/3]  ── Season complete! ──")
        print("       All predictions saved in ./outputs/predictions_log.csv")
        print("       Final standings in    ./outputs/standings_log.csv")

    print(f"\n{'═'*60}")
    print(f"  ✅  Workflow complete for Round {completed}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
