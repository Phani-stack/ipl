"""
IPL Win Probability — Training Script
Tailored exactly to your CSV columns:

balls.csv  : id, batter_name, bowler_name, non_striker_name,
             batsman_run, extra_run, total_run, batting_team, bowling_team

matches.csv: id, city, gender, pom, toss_decision, winner,
             team_type, won_by
"""

import pandas as pd
import numpy as np
import pickle
import ast
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
BALLS_CSV   = "../data/balls.csv"
MATCHES_CSV = "../data/matches.csv"

TEAM_RENAME = {
    "Delhi Daredevils":            "Delhi Capitals",
    "Kings XI Punjab":             "Punjab Kings",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    "Rising Pune Supergiants":     "Rising Pune Supergiant",
}

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("📂 Loading CSVs...")
balls   = pd.read_csv(BALLS_CSV,   index_col=0)
matches = pd.read_csv(MATCHES_CSV, index_col=0)

balls["batting_team"] = balls["batting_team"].replace(TEAM_RENAME)
balls["bowling_team"] = balls["bowling_team"].replace(TEAM_RENAME)
matches["winner"]     = matches["winner"].replace(TEAM_RENAME)

print(f"   balls: {len(balls):,} rows  |  matches: {len(matches)}")

# ── PARSE won_by  e.g. {'runs': 35}  or  {'wickets': 7} ─────────────────────
def parse_won_by(s):
    try:
        d = ast.literal_eval(str(s))
        if "runs"    in d: return "runs",    int(d["runs"])
        if "wickets" in d: return "wickets", int(d["wickets"])
    except Exception:
        pass
    return None, None

matches[["win_type", "win_margin"]] = matches["won_by"].apply(
    lambda s: pd.Series(parse_won_by(s))
)

match_info = matches[["winner", "win_type", "win_margin"]].reset_index()
match_info.rename(columns={"index": "id"}, inplace=True)

# ── MERGE ─────────────────────────────────────────────────────────────────────
df = balls.reset_index(drop=True).merge(match_info, on="id", how="inner")
df = df.dropna(subset=["batting_team", "bowling_team", "winner"])
print(f"   After merge: {len(df):,} rows  |  {df['id'].nunique()} matches")

# ── INFER INNINGS ─────────────────────────────────────────────────────────────
# Rows are already in ball order. Within each match the batting_team changes
# exactly once (1st → 2nd innings). Tag each row accordingly.
df = df.reset_index(drop=True)
df["ball_num"] = df.groupby("id").cumcount()   # 0-based sequential ball index

first_bat = df.groupby("id")["batting_team"].first().rename("first_bat")
df = df.join(first_bat, on="id")
df["inning"] = np.where(df["batting_team"] == df["first_bat"], 1, 2)

# ── 1ST INNINGS TARGET ────────────────────────────────────────────────────────
inn1 = df[df["inning"] == 1].copy()
target_map = (
    inn1.groupby("id")["total_run"]
        .sum()
        .rename("target")
        .reset_index()
)

# ── 2ND INNINGS ───────────────────────────────────────────────────────────────
inn2 = df[df["inning"] == 2].copy()
inn2 = inn2.merge(target_map, on="id", how="inner")
inn2 = inn2[inn2["target"] > 0]

# Ball index within the 2nd innings (1-based)
inn2["ball_in_inn2"] = inn2.groupby("id").cumcount() + 1

# Cumulative score in the chase
inn2["current_score"] = inn2.groupby("id")["total_run"].cumsum()

inn2["runs_left"]  = (inn2["target"] - inn2["current_score"]).clip(0)
inn2["balls_left"] = (120 - inn2["ball_in_inn2"]).clip(0)
inn2["overs_used"] = inn2["ball_in_inn2"] / 6
inn2["overs_left"] = inn2["balls_left"]   / 6

# ── RUN RATES ─────────────────────────────────────────────────────────────────
inn2["crr"] = inn2["current_score"] / (inn2["overs_used"] + 1e-6)
inn2["rrr"] = inn2["runs_left"]     / (inn2["overs_left"] + 1e-6)

# ── PHASE FLAGS ───────────────────────────────────────────────────────────────
inn2["is_powerplay"] = (inn2["ball_in_inn2"] <= 36).astype(int)
inn2["is_middle"]    = ((inn2["ball_in_inn2"] > 36) & (inn2["ball_in_inn2"] <= 96)).astype(int)
inn2["is_death"]     = (inn2["ball_in_inn2"] > 96).astype(int)

# ── PRESSURE & RATIO FEATURES ─────────────────────────────────────────────────
inn2["pressure"]       = inn2["rrr"] - inn2["crr"]
inn2["target_ratio"]   = inn2["current_score"] / inn2["target"]
inn2["required_ratio"] = inn2["rrr"] / (inn2["crr"] + 1e-6)

# ── MOMENTUM (rolling last 6 balls) ──────────────────────────────────────────
def roll6(s):
    return s.rolling(6, min_periods=1).sum()

inn2["last6_runs"] = inn2.groupby("id")["total_run"].transform(roll6)
inn2["momentum"]   = inn2["last6_runs"] / 6        # avg runs per ball last over
inn2["is_dot"]     = (inn2["total_run"] == 0).astype(int)
inn2["last6_dots"] = inn2.groupby("id")["is_dot"].transform(roll6)

# Batsman run rate in last 6 balls
inn2["last6_bat"] = inn2.groupby("id")["batsman_run"].transform(roll6)

# ── TARGET DIFFICULTY ─────────────────────────────────────────────────────────
inn2["high_target"] = (inn2["target"] >= 180).astype(int)
inn2["low_target"]  = (inn2["target"] <= 140).astype(int)

# ── LABEL ─────────────────────────────────────────────────────────────────────
inn2["result"] = (inn2["batting_team"] == inn2["winner"]).astype(int)

# ── CLEAN ─────────────────────────────────────────────────────────────────────
inn2 = inn2[(inn2["balls_left"] > 0) & (inn2["runs_left"] >= 0)]
print(f"   Training rows: {len(inn2):,}  |  Win rate: {inn2['result'].mean():.3f}")

# ── ENCODE TEAMS ──────────────────────────────────────────────────────────────
le = LabelEncoder()
le.fit(sorted(pd.concat([inn2["batting_team"], inn2["bowling_team"]]).unique()))

inn2["bt_enc"] = le.transform(inn2["batting_team"])
inn2["bl_enc"] = le.transform(inn2["bowling_team"])

print(f"   Teams ({len(le.classes_)}): {list(le.classes_)}")

# ── FEATURE MATRIX ────────────────────────────────────────────────────────────
FEATURES = [
    "bt_enc", "bl_enc",
    "runs_left", "balls_left",
    "crr", "rrr",
    "pressure", "target_ratio", "required_ratio",
    "is_powerplay", "is_middle", "is_death",
    "momentum", "last6_runs", "last6_dots", "last6_bat",
    "high_target", "low_target",
    "target",
]

X      = inn2[FEATURES].fillna(0).astype(float)
y      = inn2["result"]
groups = inn2["id"]

# ── TRAIN / TEST SPLIT (group by match → zero leakage) ───────────────────────
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print(f"\n   Train rows: {len(X_train):,}  |  Test rows: {len(X_test):,}")

# ── MODEL ─────────────────────────────────────────────────────────────────────
base = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=15,
    gamma=1.0,
    reg_alpha=0.5,
    reg_lambda=2.0,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

# CalibratedClassifierCV → win % are statistically honest (60% really means 60%)
model = CalibratedClassifierCV(base, cv=3, method="sigmoid")

print("\n⏳ Training (may take ~1-2 min)...")
model.fit(X_train, y_train)

# ── EVALUATE ──────────────────────────────────────────────────────────────────
probs = model.predict_proba(X_test)[:, 1]

print("\n📊 Test-set metrics")
print(f"   ROC-AUC     : {roc_auc_score(y_test, probs):.4f}")
print(f"   Log-loss    : {log_loss(y_test, probs):.4f}")
print(f"   Brier score : {brier_score_loss(y_test, probs):.4f}  (lower = better calibration)")

test_snap      = inn2.iloc[test_idx].copy()
test_snap["pred"] = probs
last_ball      = test_snap.sort_values("ball_in_inn2").groupby("id").last()
end_acc        = (last_ball["pred"].round() == last_ball["result"]).mean()
print(f"   End-of-match accuracy: {end_acc:.3f}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
pickle.dump(model,    open("model.pkl",    "wb"))
pickle.dump(le,       open("encoder.pkl",  "wb"))
pickle.dump(FEATURES, open("features.pkl", "wb"))

print("\n✅ Saved: model.pkl  encoder.pkl  features.pkl")

