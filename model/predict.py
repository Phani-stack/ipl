"""
IPL Win Probability — Prediction Script
Loads the artifacts saved by train_model.py and predicts for any match state.
"""

import pickle
import numpy as np

# ── LOAD ──────────────────────────────────────────────────────────────────────
model    = pickle.load(open("model.pkl",    "rb"))
encoder  = pickle.load(open("encoder.pkl",  "rb"))
FEATURES = pickle.load(open("features.pkl", "rb"))

TEAM_RENAME = {
    "Delhi Daredevils":            "Delhi Capitals",
    "Kings XI Punjab":             "Punjab Kings",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    "Rising Pune Supergiants":     "Rising Pune Supergiant",
}

# ── AVAILABLE TEAMS (for reference) ───────────────────────────────────────────
print("Available teams:", list(encoder.classes_))

# ════════════════════════════════════════════════════════════════════════
#  ✏️  CHANGE THESE VALUES FOR EACH PREDICTION
# ════════════════════════════════════════════════════════════════════════
batting_team   = "Chennai Super Kings"   # team currently batting (chasing)
bowling_team   = "Mumbai Indians"        # team currently bowling (defending)

target         = 180     # 1st innings total (what the batting team needs)
current_score  = 54      # runs scored so far in the chase
balls_used     = 36      # balls faced so far  (36 = end of powerplay = 6 overs)

# Optional live context — set to None to auto-estimate from run rate
last6_runs     = None    # runs scored in the last 6 balls (1 over)
last6_dots     = None    # dot balls in the last 6 balls
last6_bat      = None    # batsman runs (excl. extras) in last 6 balls
# ════════════════════════════════════════════════════════════════════════


def predict(batting_team, bowling_team, target,
            current_score, balls_used,
            last6_runs=None, last6_dots=None, last6_bat=None):

    batting_team = TEAM_RENAME.get(batting_team, batting_team)
    bowling_team = TEAM_RENAME.get(bowling_team, bowling_team)

    if batting_team not in encoder.classes_:
        raise ValueError(f"Unknown batting team: '{batting_team}'\n"
                         f"Available: {list(encoder.classes_)}")
    if bowling_team not in encoder.classes_:
        raise ValueError(f"Unknown bowling team: '{bowling_team}'\n"
                         f"Available: {list(encoder.classes_)}")

    balls_left   = max(120 - balls_used, 0)
    runs_left    = max(target - current_score, 0)
    overs_used   = balls_used / 6
    overs_left   = balls_left / 6

    crr = current_score / (overs_used + 1e-6)
    rrr = runs_left     / (overs_left + 1e-6)

    # Auto-estimate rolling features when not supplied
    if last6_runs is None:
        last6_runs = min(crr, 20) * 1.0   # rough: 1 over at current rate
    if last6_dots is None:
        last6_dots = max(0, 6 - last6_runs)   # balls without a run
    if last6_bat is None:
        last6_bat  = last6_runs * 0.85    # ~85% of runs from bat

    momentum = last6_runs / 6

    bt_enc = encoder.transform([batting_team])[0]
    bl_enc = encoder.transform([bowling_team])[0]

    row = {
        "bt_enc":          bt_enc,
        "bl_enc":          bl_enc,
        "runs_left":       runs_left,
        "balls_left":      balls_left,
        "crr":             crr,
        "rrr":             rrr,
        "pressure":        rrr - crr,
        "target_ratio":    current_score / target,
        "required_ratio":  rrr / (crr + 1e-6),
        "is_powerplay":    int(balls_used <= 36),
        "is_middle":       int(36 < balls_used <= 96),
        "is_death":        int(balls_used > 96),
        "momentum":        momentum,
        "last6_runs":      last6_runs,
        "last6_dots":      last6_dots,
        "last6_bat":       last6_bat,
        "high_target":     int(target >= 180),
        "low_target":      int(target <= 140),
        "target":          target,
    }

    X = np.array([[row[f] for f in FEATURES]])
    p_bat  = round(model.predict_proba(X)[0][1] * 100, 1)
    p_bowl = round(100 - p_bat, 1)
    return p_bat, p_bowl


# ── SINGLE PREDICTION ─────────────────────────────────────────────────────────
p_bat, p_bowl = predict(
    batting_team, bowling_team, target,
    current_score, balls_used,
    last6_runs, last6_dots, last6_bat
)

overs_str  = f"{balls_used // 6}.{balls_used % 6}"
runs_left  = max(target - current_score, 0)
balls_left = max(120 - balls_used, 0)
rrr        = runs_left / (balls_left / 6 + 1e-6)

print("\n🏏  IPL WIN PROBABILITY")
print("═" * 44)
print(f"  Batting  : {batting_team}")
print(f"  Bowling  : {bowling_team}")
print(f"  Target   : {target}")
print(f"  Score    : {current_score}  ({overs_str} ov)")
print(f"  Need     : {runs_left} off {balls_left} balls  (RRR {rrr:.2f})")
print("─" * 44)
print(f"  {batting_team:<34} {p_bat:>5.1f}%")
print(f"  {bowling_team:<34} {p_bowl:>5.1f}%")
print("═" * 44)
winner = batting_team if p_bat > 50 else bowling_team
print(f"  👉  Predicted winner : {winner}\n")


# ── OVER-BY-OVER SIMULATION ───────────────────────────────────────────────────
def simulate(batting_team, bowling_team, target,
             runs_per_over=8.0, wicket_every_n=4):
    """
    Simulate a chase at a fixed run rate for quick visualisation.
    Replace with real ball-by-ball data in a live app.
    """
    print(f"📈  Over-by-over simulation  "
          f"(pace={runs_per_over:.1f} rpo, wicket every {wicket_every_n} ov)")
    print(f"     {'Ov':<4} {'Score':<10} {'W':<4} {'Chase%':>7}  Bar")
    print("     " + "─" * 42)

    score = 0
    wkts  = 0
    for ov in range(1, 21):
        score += runs_per_over
        if ov % wicket_every_n == 0 and wkts < 9:
            wkts += 1
        p, _ = predict(batting_team, bowling_team, target,
                       int(score), ov * 6)
        bar = "█" * int(p / 5) + "░" * (20 - int(p / 5))
        print(f"     {ov:<4} {int(score):>3}/{wkts:<6} {'':4} {p:>5.1f}%  {bar}")

simulate(batting_team, bowling_team, target,
         runs_per_over=target / 20)
