#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entraine le modele de forecast BE imbalance.
Features : 16 QH lags (4h) de volume BE + 6 features cycliques = 22 features.
Meme architecture que le modele FR (HistGradientBoostingRegressor).
Sauvegarde : models/be_imbalance_model.joblib
"""

import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

REPO    = Path(__file__).resolve().parents[2]
DATA    = REPO / "data_ve_2025"
OUT     = REPO / "models" / "be_imbalance_model.joblib"
OUT.parent.mkdir(exist_ok=True)

MAX_LAG_BE = 16   # 4 heures de 15-min history

# =============================================================================
# CHARGEMENT
# =============================================================================
print("Chargement donnees BE...")
parts = []
for f in sorted(DATA.glob("imbalance_be_2025-*.csv")):
    try:
        parts.append(pd.read_csv(f, parse_dates=["timestamp"]))
    except Exception:
        pass

df = (pd.concat(parts, ignore_index=True)
      .sort_values("timestamp")
      .drop_duplicates("timestamp")
      .reset_index(drop=True))
print(f"  {len(df)} QH charges  ({df['timestamp'].min().date()} -> {df['timestamp'].max().date()})")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
X, y, timestamps = [], [], []
for i in range(MAX_LAG_BE, len(df)):
    row_ts = df.loc[i, "timestamp"]
    lags   = df.loc[i - MAX_LAG_BE : i - 1, "volume"].values[::-1]  # plus recent en premier
    slot   = row_ts.hour * 4 + row_ts.minute // 15
    feats  = list(lags) + [
        np.sin(2 * np.pi * slot / 96),           np.cos(2 * np.pi * slot / 96),
        np.sin(2 * np.pi * row_ts.dayofweek / 7), np.cos(2 * np.pi * row_ts.dayofweek / 7),
        np.sin(2 * np.pi * row_ts.month / 12),    np.cos(2 * np.pi * row_ts.month / 12),
    ]
    X.append(feats)
    y.append(df.loc[i, "volume"])
    timestamps.append(row_ts)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
print(f"  {X.shape[0]} samples, {X.shape[1]} features")

# =============================================================================
# TRAIN / VALIDATION (derniers 30 jours = validation)
# =============================================================================
n_val   = 30 * 96
n_train = len(X) - n_val

X_train, y_train = X[:n_train], y[:n_train]
X_val,   y_val   = X[n_train:], y[n_train:]

print(f"\nEntraine sur {n_train} QH, valide sur {n_val} QH (derniers 30j)...")
model = HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.05, max_depth=5,
    min_samples_leaf=20, random_state=42,
)
model.fit(X_train, y_train)

# =============================================================================
# EVALUATION
# =============================================================================
pred_train = model.predict(X_train)
pred_val   = model.predict(X_val)

mae_t   = mean_absolute_error(y_train, pred_train)
mae_v   = mean_absolute_error(y_val,   pred_val)
dacc_t  = ((pred_train > 0) == (y_train > 0)).mean() * 100
dacc_v  = ((pred_val   > 0) == (y_val   > 0)).mean() * 100

print(f"\n  Train  : MAE={mae_t:.1f} MW  dir_acc={dacc_t:.1f}%")
print(f"  Val    : MAE={mae_v:.1f} MW  dir_acc={dacc_v:.1f}%")

# =============================================================================
# SAUVEGARDE
# =============================================================================
joblib.dump(model, OUT)
print(f"\n  Modele sauvegarde : {OUT}")
print(f"  n_features_in_   : {model.n_features_in_}")
