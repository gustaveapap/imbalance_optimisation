#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Barre de progression en temps reel pour les 4 simulations.
Usage : python scripts/status.py          (affichage unique)
        python scripts/status.py --watch  (rafraichissement toutes les 60s)
"""

import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
from datetime import datetime
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# Periode de reference
T_START = pd.Timestamp("2026-01-01")
T_END   = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=1)  # hier

BAR_W = 30  # largeur barre

def bar(n, total, width=BAR_W):
    if total <= 0:
        return "[" + "?" * width + "]  ?%"
    pct  = min(n / total, 1.0)
    fill = int(pct * width)
    return f"[{'#' * fill}{'-' * (width - fill)}] {pct*100:5.1f}%"

def days_covered(total_days):
    return f"{total_days}/{(T_END - T_START).days + 1} j"

# =============================================================================
# VE BE
# =============================================================================
def status_ve_be():
    path = REPO / "outputs" / "ve_be" / "simulation_ve_be_2026.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None, "ABSENT"
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        last_ts   = df["timestamp"].max()
        total_qh  = (T_END + pd.Timedelta(days=1) - T_START).total_seconds() / 900
        done_qh   = len(df[df["strategy"] == "S_BE_OPT"])
        opt = df[df["strategy"] == "S_BE_OPT"]["cost_eur"].sum()
        s1  = df[df["strategy"] == "S1"]["cost_eur"].sum()
        detail = (f"dernier: {last_ts.strftime('%Y-%m-%d %H:%M')}  |  "
                  f"OPT={opt:+.2f}EUR  S1={s1:+.2f}EUR")
        return done_qh / total_qh, detail
    except Exception as e:
        return None, f"ERREUR: {e}"

# =============================================================================
# VE FR
# =============================================================================
def status_ve_fr():
    path = REPO / "outputs" / "ve_fr" / "simulation_ve_fr_2026.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None, "ABSENT"
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        last_ts  = df["timestamp"].max()
        total_qh = (T_END + pd.Timedelta(days=1) - T_START).total_seconds() / 900
        done_qh  = len(df[df["strategy"] == "S_BE_OPT"])
        opt = df[df["strategy"] == "S_BE_OPT"]["cost_eur"].sum()
        s1  = df[df["strategy"] == "S1"]["cost_eur"].sum()
        detail = (f"dernier: {last_ts.strftime('%Y-%m-%d %H:%M')}  |  "
                  f"OPT={opt:+.2f}EUR  S1={s1:+.2f}EUR")
        return done_qh / total_qh, detail
    except Exception as e:
        return None, f"ERREUR: {e}"

# =============================================================================
# SOLAR BE
# =============================================================================
def status_solar_be():
    path = REPO / "outputs" / "solar_be" / "simulation_2026" / "summary_2026.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None, "ABSENT"
    try:
        df       = pd.read_csv(path)
        last_day = pd.Timestamp(df["date"].max())
        total_d  = (T_END - T_START).days + 1
        done_d   = len(df)
        s3_total = df["s3_total"].sum()
        s1_total = df["s1_total"].sum()
        detail   = (f"dernier: {last_day.date()}  |  {days_covered(done_d)}  |  "
                    f"S3={s3_total:+.0f}EUR  S1={s1_total:+.0f}EUR  /GW")
        return done_d / total_d, detail
    except Exception as e:
        return None, f"ERREUR: {e}"

# =============================================================================
# SOLAR FR
# =============================================================================
def status_solar_fr():
    path = REPO / "outputs" / "solar_fr" / "simulation_2026" / "summary_2026.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None, "ABSENT"
    try:
        df       = pd.read_csv(path)
        last_day = pd.Timestamp(df["date"].max())
        total_d  = (T_END - T_START).days + 1
        done_d   = len(df)
        s2_total = df["s2_total"].sum()
        s1_total = df["s1_total"].sum()
        detail   = (f"dernier: {last_day.date()}  |  {days_covered(done_d)}  |  "
                    f"S2={s2_total:+.0f}EUR  S1={s1_total:+.0f}EUR  /GW")
        return done_d / total_d, detail
    except Exception as e:
        return None, f"ERREUR: {e}"

# =============================================================================
# DISPLAY
# =============================================================================
def display():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ref = f"2026-01-01 -> {T_END.date()}"
    print(f"\n{'='*80}")
    print(f"  ETAT DES SIMULATIONS  --  {now}  --  ref: {ref}")
    print(f"{'='*80}")

    sims = [
        ("VE BE      ", status_ve_be),
        ("VE FR      ", status_ve_fr),
        ("Solar BE   ", status_solar_be),
        ("Solar FR   ", status_solar_fr),
    ]

    for label, fn in sims:
        pct, detail = fn()
        b = bar(pct if pct is not None else 0, 1.0) if pct is not None else "[" + "?" * BAR_W + "]   ?%"
        print(f"  {label}  {b}")
        print(f"             {detail}")
        print()

    print(f"{'='*80}\n")

def main():
    watch = "--watch" in sys.argv
    if watch:
        interval = 60
        print(f"Mode watch actif (rafraichissement toutes les {interval}s). Ctrl+C pour quitter.")
        while True:
            display()
            time.sleep(interval)
    else:
        display()

if __name__ == "__main__":
    main()
