#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIMULATION SOLAIRE FR 2025 -- S1/S2/S3/S4
Utilise les donnees cachees dans data_ve_2025/ (imbalance_fr, prix_da,
afrr/mfrr BE+DE+NL) -- aucun telechargement imbalance/merit necessaire.
Resultats dans outputs/solar_fr/simulation_2025/
"""

import sys, os, warnings, time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================
REPO_DIR = Path(__file__).resolve().parents[2]   # imbalance_optimisation/
sys.path.insert(0, str(Path(__file__).parent))    # pour importer test_logic_fr

os.chdir(str(REPO_DIR))

DATA_2025   = REPO_DIR / "data_ve_2025"
DATA_SOLAR  = REPO_DIR / "data_solar_optim"      # contient pvgis + solar cache
RESULTS_DIR = REPO_DIR / "outputs" / "solar_fr" / "simulation_2025"
MERIT_CACHE = DATA_2025 / "merit_cache"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MERIT_CACHE.mkdir(exist_ok=True)

SUMMARY_FILE = RESULTS_DIR / "summary_2025.csv"
ERRORS_FILE  = RESULTS_DIR / "errors_2025.csv"

START_DATE = "2025-01-01"
END_DATE   = "2025-12-31"

# =============================================================================
# IMPORT fonctions utilitaires depuis test_logic_fr
# (on n'importe PAS dl_imbalance_fr / dl_da / build_merit_metrics
#  car on les remplace par des versions qui lisent data_ve_2025/)
# =============================================================================
from test_logic_fr import (
    dl_pvgis, dl_solar, pvgis_day,
    compute_strategies_v2, build_forecast_parc,
    qh_range, align_qh, file_ok, safe_float,
    forecast_fr, compute_ratios,
    PVGIS_FILE, MODEL_PATH,
)

SEP = "=" * 100

# =============================================================================
# LECTURE IMBALANCE FR 2025 (cache data_ve_2025/)
# =============================================================================

def dl_imbalance_fr_2025(date_str):
    p = DATA_2025 / f"imbalance_fr_{date_str}.csv"
    if not file_ok(p):
        print(f"    imbalance_fr  : MANQUANT ({p.name})")
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["timestamp"])
    print(f"    imbalance_fr  : cache ({len(df)} lignes)")
    return df

# =============================================================================
# LECTURE PRIX DA 2025 (cache data_ve_2025/)
# =============================================================================

def dl_da_2025(date_str):
    p = DATA_2025 / f"prix_da_{date_str}.csv"
    if not file_ok(p):
        print(f"    prix_da       : MANQUANT ({p.name})")
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["timestamp"])
    print(f"    prix_da       : cache ({len(df)} lignes)")
    return df

# =============================================================================
# MERIT METRICS 2025 (depuis fichiers raw data_ve_2025/)
# =============================================================================

def build_merit_metrics_2025(date_str):
    cache = MERIT_CACHE / f"merit_{date_str}.csv"
    if file_ok(cache):
        df = pd.read_csv(cache, parse_dates=["timestamp"])
        for col in ["afrr_ratio_negative", "afrr_ratio_very_negative",
                    "mfrr_ratio_negative",  "mfrr_ratio_very_negative"]:
            if col not in df.columns:
                df[col] = 0.0
        print(f"    merit order EU : cache")
        return df

    dfs_afrr, dfs_mfrr = [], []
    for country in ["be", "de", "nl"]:
        for reserve, lst in [("afrr", dfs_afrr), ("mfrr", dfs_mfrr)]:
            f = DATA_2025 / f"{reserve}_{country}_{date_str}.csv"
            if file_ok(f):
                try:
                    lst.append(pd.read_csv(f, parse_dates=["timestamp"]))
                except Exception:
                    pass

    df_afrr = pd.concat(dfs_afrr, ignore_index=True) if dfs_afrr else pd.DataFrame()
    df_mfrr = pd.concat(dfs_mfrr, ignore_index=True) if dfs_mfrr else pd.DataFrame()

    afrr_dn = compute_ratios(df_afrr, "DOWN").rename(columns={
        "ratio_negative":      "afrr_ratio_negative",
        "ratio_very_negative": "afrr_ratio_very_negative"})
    mfrr_dn = compute_ratios(df_mfrr, "DOWN").rename(columns={
        "ratio_negative":      "mfrr_ratio_negative",
        "ratio_very_negative": "mfrr_ratio_very_negative"})

    full = pd.DataFrame({"timestamp": qh_range(date_str)})
    for dfx in [afrr_dn, mfrr_dn]:
        if dfx is not None and not dfx.empty:
            full = full.merge(dfx, on="timestamp", how="left")
    for col in ["afrr_ratio_negative", "afrr_ratio_very_negative",
                "mfrr_ratio_negative",  "mfrr_ratio_very_negative"]:
        if col not in full.columns:
            full[col] = 0.0
        else:
            full[col] = full[col].fillna(0.0)

    full.to_csv(cache, index=False)
    n_countries = len(dfs_afrr) + len(dfs_mfrr)
    print(f"    merit order EU : {n_countries} fichiers -> {len(full)} QH metriques")
    return full

# =============================================================================
# SIMULATION UN JOUR
# =============================================================================

def simulate_day_2025(date_str, df_pvgis, df_hist):
    result_path = RESULTS_DIR / f"day_{date_str}.csv"
    if file_ok(result_path):
        df = pd.read_csv(result_path, parse_dates=["timestamp"])
        print(f"  {date_str}  -> cache ({len(df)} QH)")
        return df

    print(f"\n  {date_str}")
    df_imb    = dl_imbalance_fr_2025(date_str)
    df_da     = dl_da_2025(date_str)
    df_solar  = dl_solar(date_str)       # ENTSO-E, gracieux si echec
    df_pv_day = pvgis_day(date_str, df_pvgis)

    if df_imb.empty or df_da.empty or df_pv_day.empty:
        print(f"    -> SKIP (imb={df_imb.empty} da={df_da.empty} pv={df_pv_day.empty})")
        return None

    metrics = build_merit_metrics_2025(date_str)

    df_fc = forecast_fr(date_str, df_hist)
    if df_fc is not None:
        print(f"    forecast_fr   : {len(df_fc)} QH")
    else:
        print(f"    forecast_fr   : historique insuffisant -> volume=0")

    full = pd.DataFrame({"timestamp": qh_range(date_str)})
    full = full.merge(df_imb[["timestamp", "volume", "prix_positif", "prix_negatif"]],
                      on="timestamp", how="left")
    full = full.merge(df_da[["timestamp", "price_eur_mwh"]], on="timestamp", how="left")
    full = full.merge(df_pv_day[["timestamp", "production_mw"]], on="timestamp", how="left")
    full = full.merge(metrics, on="timestamp", how="left")

    if df_fc is not None and not df_fc.empty:
        full = full.merge(df_fc[["timestamp", "forecast_volume", "forecast_direction"]],
                          on="timestamp", how="left")
    else:
        full["forecast_volume"]    = 0.0
        full["forecast_direction"] = "UP"

    if not df_solar.empty:
        full = full.merge(df_solar[["timestamp", "forecast_mw", "actual_mw"]],
                          on="timestamp", how="left")
    else:
        full["forecast_mw"] = 0.0
        full["actual_mw"]   = 0.0

    full["production_mw"]      = full["production_mw"].fillna(0)
    full["price_eur_mwh"]      = full["price_eur_mwh"].ffill().fillna(0)
    full["prix_positif"]       = full["prix_positif"].ffill().bfill().fillna(0)
    full["prix_negatif"]       = full["prix_negatif"].ffill().bfill().fillna(0)
    full["forecast_volume"]    = full["forecast_volume"].fillna(0)
    full["forecast_direction"] = full["forecast_direction"].fillna("UP")
    full["forecast_mw"]        = full["forecast_mw"].fillna(0)
    full["actual_mw"]          = full["actual_mw"].fillna(1).replace(0, 1)

    full["forecast_parc"] = build_forecast_parc(
        full["production_mw"], full[["timestamp", "forecast_mw", "actual_mw"]])
    full["date"] = date_str
    full = compute_strategies_v2(full)

    full.to_csv(result_path, index=False)
    return full

# =============================================================================
# SUMMARY
# =============================================================================

def save_summary(results, errors):
    df_all = pd.concat(results, ignore_index=True)
    rows = []
    for date_str, grp in df_all.groupby("date"):
        row = {"date": date_str}
        for s in ["s1", "s2", "s3"]:
            row[f"{s}_total"] = grp[f"{s}_total"].sum()
            row[f"{s}_da"]    = grp[f"{s}_revenue_da"].sum()
            row[f"{s}_imb"]   = grp[f"{s}_revenue_imb"].sum()
            if s != "s1":
                row[f"{s}_curt"] = int(grp[f"{s}_curtail"].sum())
        row["n_solar_qh"] = int((grp["production_mw"] > 0.01).sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(SUMMARY_FILE, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(ERRORS_FILE, index=False)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print(SEP)
    print("  SIMULATION SOLAIRE FR 2025  --  S1 / S2 / S3")
    print(f"  Periode : {START_DATE} -> {END_DATE}")
    print(f"  Data    : {DATA_2025}")
    print(f"  Outputs : {RESULTS_DIR}")
    print(SEP)

    # PVGIS (cache dans data_solar_optim/)
    print("\n  PVGIS...")
    df_pvgis = dl_pvgis()

    # Historique imbalance pour forecast walk-forward
    print("\n  Chargement historique imbalance FR 2025...")
    parts = []
    for f in sorted(DATA_2025.glob("imbalance_fr_2025-*.csv")):
        try:
            parts.append(pd.read_csv(f, parse_dates=["timestamp"]))
        except Exception:
            pass
    df_hist = (pd.concat(parts, ignore_index=True)
               .sort_values("timestamp").drop_duplicates("timestamp")
               .reset_index(drop=True)) if parts else pd.DataFrame()
    print(f"  {len(df_hist)} QH historiques charges")

    # Generer toutes les dates
    all_dates = [d.strftime("%Y-%m-%d")
                 for d in pd.date_range(START_DATE, END_DATE, freq="D")]
    print(f"\n  {len(all_dates)} jours a simuler\n")

    results, errors = [], []

    for i, date_str in enumerate(all_dates):
        try:
            df_day = simulate_day_2025(date_str, df_pvgis, df_hist)
        except Exception as e:
            print(f"  {date_str}  -> ERREUR: {e}")
            errors.append({"date": date_str, "error": str(e)})
            continue

        if df_day is None:
            errors.append({"date": date_str, "error": "donnees manquantes"})
            continue

        # Mise a jour historique pour forecast des jours suivants
        new_rows = df_day[["timestamp", "volume"]].dropna()
        if not df_hist.empty:
            df_hist = (pd.concat([df_hist, new_rows], ignore_index=True)
                       .sort_values("timestamp").drop_duplicates("timestamp"))
        else:
            df_hist = new_rows.copy()

        results.append(df_day)

        n_sol = int((df_day["production_mw"] > 0.01).sum())
        s1 = df_day["s1_total"].sum()
        s2 = df_day["s2_total"].sum()
        s3 = df_day["s3_total"].sum()
        best = ["S1","S2","S3"][[s1,s2,s3].index(max(s1,s2,s3))]
        print(f"    {date_str}: {n_sol} QH sol  S1={s1:+.0f}  S2={s2:+.0f}  S3={s3:+.0f}  best={best}")

        if (i + 1) % 14 == 0 or i == len(all_dates) - 1:
            save_summary(results, errors)
            print(f"  [SAVE] {i+1}/{len(all_dates)} jours")

    # Resume final
    print(f"\n{SEP}")
    print("  RESUME FINAL")
    print(SEP)
    if not results:
        print("  Aucun resultat.")
        return

    df_all = pd.concat(results, ignore_index=True)
    save_summary(results, errors)

    df_all["month"] = pd.to_datetime(df_all["timestamp"]).dt.to_period("M").astype(str)
    print()
    for mois in sorted(df_all["month"].unique()):
        dm   = df_all[df_all["month"] == mois]
        vals = [dm[f"s{k}_total"].sum() for k in [1,2,3]]
        best = ["S1","S2","S3"][vals.index(max(vals))]
        print(f"  {mois}  S1={vals[0]:+7.0f}  S2={vals[1]:+7.0f}  S3={vals[2]:+7.0f}  best={best}")

    print()
    for s, name in [("s1","S1 Baseline          "),
                    ("s2","S2 DA-adapt+curt150MW"),
                    ("s3","S3 50%fix+curt150MW  ")]:
        tot = df_all[f"{s}_total"].sum()
        da  = df_all[f"{s}_revenue_da"].sum()
        imb = df_all[f"{s}_revenue_imb"].sum()
        nc  = int(df_all[f"{s}_curtail"].sum()) if s != "s1" else 0
        print(f"  {name}  total={tot:+9.2f}  DA={da:+9.2f}  imb={imb:+9.2f}  curt={nc}")

    if errors:
        print(f"\n  {len(errors)} jours en erreur / manquants")

    print(f"\n  Resultats dans : {RESULTS_DIR}")
    print(SEP)


if __name__ == "__main__":
    main()
