#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIMULATION ANNUELLE FR 2026 -- S1/S2/S3/S4
Telecharge toutes donnees 2026-01-01 -> 2026-04-13, simule chaque jour.
Resultats incrementaux dans simulation_solar_intelligent/
"""

import sys, os
# Importer toutes les fonctions de test_logic_fr
sys.path.insert(0, "C:/Users/gusta")
os.chdir("C:/Users/gusta")

import base64, time, warnings
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import requests

warnings.filterwarnings("ignore")

# Importer depuis test_logic_fr.py
from test_logic_fr import (
    get_rte_token, dl_pvgis, dl_imbalance_fr, dl_da, dl_solar,
    pvgis_day, build_merit_metrics, forecast_fr, compute_strategies_v2,
    build_forecast_parc, qh_range, simulate_day,
    DATA_DIR, OUT_DIR, PVGIS_FILE, MODEL_PATH,
    CLIENT_ID_IMBALANCE, CLIENT_SECRET_IMBALANCE,
)

SEP = "=" * 100

# =============================================================================
# CONFIG
# =============================================================================

START_DATE = "2026-01-01"
END_DATE   = "2026-04-13"   # hier (aujourd'hui=14/04 potentiellement incomplet)

RESULTS_DIR  = Path("simulation_solar_intelligent")
RESULTS_DIR.mkdir(exist_ok=True)
SUMMARY_FILE = RESULTS_DIR / "summary_2026.csv"
MERIT_CACHE  = DATA_DIR / "merit_cache"
MERIT_CACHE.mkdir(exist_ok=True)

TOKEN_REFRESH_INTERVAL = 3000  # refresh token toutes les 50 min

# =============================================================================
# CACHE MERIT ORDER
# =============================================================================

def get_merit_metrics(date_str):
    """Merit order avec cache CSV par jour."""
    cache_path = MERIT_CACHE / f"merit_{date_str}.csv"
    if cache_path.exists() and cache_path.stat().st_size > 200:
        df = pd.read_csv(cache_path, parse_dates=["timestamp"])
        return df
    print(f"    merit order EU : telechargement...")
    metrics = build_merit_metrics(date_str)
    if metrics is not None and not metrics.empty:
        metrics.to_csv(cache_path, index=False)
        print(f"    merit order EU : {len(metrics)} QH metriques -> cache")
    else:
        print(f"    merit order EU : ECHEC")
    return metrics

# =============================================================================
# SIMULATION UN JOUR (avec cache merit)
# =============================================================================

def simulate_day_full(date_str, token, df_pvgis, df_hist):
    """Simule un jour complet avec cache merit order."""
    result_path = RESULTS_DIR / f"day_{date_str}.csv"
    if result_path.exists() and result_path.stat().st_size > 500:
        df = pd.read_csv(result_path, parse_dates=["timestamp"])
        print(f"  {date_str}  -> cache complet ({len(df)} QH)")
        return df

    print(f"\n  {date_str}")
    df_imb   = dl_imbalance_fr(date_str, token)
    df_da    = dl_da(date_str)
    df_solar = dl_solar(date_str)
    df_pv_day = pvgis_day(date_str, df_pvgis)

    if df_imb.empty or df_da.empty or df_pv_day.empty:
        print(f"    -> SKIP (donnees manquantes: imb={df_imb.empty} da={df_da.empty} pv={df_pv_day.empty})")
        return None

    metrics = get_merit_metrics(date_str)

    df_fc = forecast_fr(date_str, df_hist)
    if df_fc is not None:
        n_fc = len(df_fc)
        print(f"    forecast_fr   : {n_fc} QH")
    else:
        print(f"    forecast_fr   : historique insuffisant -> volume=0")

    full = pd.DataFrame({"timestamp": qh_range(date_str)})
    full = full.merge(df_imb[["timestamp","volume","prix_positif","prix_negatif"]],
                      on="timestamp", how="left")
    full = full.merge(df_da[["timestamp","price_eur_mwh"]], on="timestamp", how="left")
    full = full.merge(df_pv_day[["timestamp","production_mw"]], on="timestamp", how="left")
    full = full.merge(metrics, on="timestamp", how="left")
    if df_fc is not None and not df_fc.empty:
        full = full.merge(df_fc[["timestamp","forecast_volume","forecast_direction"]],
                          on="timestamp", how="left")
    else:
        full["forecast_volume"]    = 0.0
        full["forecast_direction"] = "UP"
    if not df_solar.empty:
        full = full.merge(df_solar[["timestamp","forecast_mw","actual_mw"]],
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
        full["production_mw"], full[["timestamp","forecast_mw","actual_mw"]])
    full["date"] = date_str
    full = compute_strategies_v2(full)

    full.to_csv(result_path, index=False)
    return full

# =============================================================================
# MAIN
# =============================================================================

def main():
    print(SEP)
    print("  SIMULATION ANNUELLE FR 2026  --  S1/S2/S3/S4")
    print(f"  Periode : {START_DATE} -> {END_DATE}")
    print(SEP)

    # Auth RTE initiale
    print("\n  Auth RTE...")
    token = get_rte_token()
    token_ts = time.time()
    print("  OK")

    # PVGIS
    print("\n  PVGIS...")
    df_pvgis = dl_pvgis()

    # Historique imbalance existant (pour forecast)
    print("\n  Chargement historique imbalance FR...")
    parts = []
    for f in sorted(DATA_DIR.glob("imbalance_fr_2026-*.csv")):
        try:
            parts.append(pd.read_csv(f, parse_dates=["timestamp"]))
        except Exception:
            pass
    df_hist = (pd.concat(parts, ignore_index=True)
               .sort_values("timestamp").drop_duplicates("timestamp")
               .reset_index(drop=True)) if parts else pd.DataFrame()
    print(f"  {len(df_hist)} QH historiques charges")

    # Generer toutes les dates
    all_dates = []
    d = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    while d <= end:
        all_dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    print(f"\n  {len(all_dates)} jours a simuler\n")

    results = []
    errors  = []

    for i, date_str in enumerate(all_dates):
        # Refresh token si besoin
        if time.time() - token_ts > TOKEN_REFRESH_INTERVAL:
            try:
                token = get_rte_token()
                token_ts = time.time()
                print(f"  [TOKEN] Refresh OK")
            except Exception as e:
                print(f"  [TOKEN] WARN refresh echoue: {e}")

        try:
            df_day = simulate_day_full(date_str, token, df_pvgis, df_hist)
        except Exception as e:
            print(f"  {date_str}  -> ERREUR: {e}")
            errors.append({"date": date_str, "error": str(e)})
            continue

        if df_day is None:
            errors.append({"date": date_str, "error": "donnees manquantes"})
            continue

        # Mettre a jour historique pour forecast des jours suivants
        new_rows = df_day[["timestamp","volume"]].dropna()
        if not df_hist.empty:
            df_hist = (pd.concat([df_hist, new_rows], ignore_index=True)
                       .sort_values("timestamp").drop_duplicates("timestamp"))
        else:
            df_hist = new_rows.copy()

        results.append(df_day)

        # Synthese rapide du jour
        n_sol = (df_day["production_mw"] > 0.01).sum()
        n_curt_s3 = int(df_day["s3_curtail"].sum())
        s1 = df_day["s1_total"].sum()
        s2 = df_day["s2_total"].sum()
        s3 = df_day["s3_total"].sum()
        s4 = df_day["s4_total"].sum()
        best = ["S1","S2","S3","S4"][[s1,s2,s3,s4].index(max(s1,s2,s3,s4))]
        print(f"    Synthese {date_str}: {n_sol} QH solaires  "
              f"S1={s1:+.0f}  S2={s2:+.0f}  S3={s3:+.0f}  S4={s4:+.0f}  "
              f"curt_S3={n_curt_s3}  best={best}")

        # Sauvegarde resume incrementale toutes les 7 jours
        if (i + 1) % 7 == 0 or i == len(all_dates) - 1:
            df_all = pd.concat(results, ignore_index=True)
            save_summary(df_all, errors)
            print(f"  [SAVE] Resume intermediaire sauvegarde ({i+1}/{len(all_dates)} jours)")

    # Resume final
    print(f"\n{SEP}")
    print("  RESUME FINAL")
    print(SEP)
    if results:
        df_all = pd.concat(results, ignore_index=True)
        save_summary(df_all, errors)
        print_final(df_all)
    if errors:
        print(f"\n  {len(errors)} jours en erreur:")
        for e in errors:
            print(f"    {e['date']}: {e['error']}")


def save_summary(df_all, errors):
    summary_rows = []
    for date_str, grp in df_all.groupby("date"):
        row = {"date": date_str}
        for s in ["s1","s2","s3","s4"]:
            row[f"{s}_total"]  = grp[f"{s}_total"].sum()
            row[f"{s}_da"]     = grp[f"{s}_revenue_da"].sum()
            row[f"{s}_imb"]    = grp[f"{s}_revenue_imb"].sum()
            if s != "s1":
                row[f"{s}_curt"] = int(grp[f"{s}_curtail"].sum())
        row["n_solar_qh"] = int((grp["production_mw"] > 0.01).sum())
        summary_rows.append(row)
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(SUMMARY_FILE, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(RESULTS_DIR / "errors_2026.csv", index=False)


def print_final(df_all):
    df_all["month"] = pd.to_datetime(df_all["timestamp"]).dt.to_period("M").astype(str)
    print()
    for mois in sorted(df_all["month"].unique()):
        dm = df_all[df_all["month"] == mois]
        vals = [dm[f"{s}_total"].sum() for s in ["s1","s2","s3","s4"]]
        best = ["S1","S2","S3","S4"][vals.index(max(vals))]
        print(f"  {mois}   "
              + "  ".join(f"{s}={v:+7.0f}" for s,v in zip(["S1","S2","S3","S4"],vals))
              + f"   best={best}")
    print()
    for s, name in [("s1","S1 Baseline      "),
                    ("s2","S2 DA-adapt 150MW"),
                    ("s3","S3 50%fix  150MW "),
                    ("s4","S4 DA-adapt 100MW")]:
        tot  = df_all[f"{s}_total"].sum()
        da   = df_all[f"{s}_revenue_da"].sum()
        imb  = df_all[f"{s}_revenue_imb"].sum()
        nc   = int(df_all[f"{s}_curtail"].sum()) if s != "s1" else 0
        print(f"  {name}  total={tot:+9.2f}  DA={da:+9.2f}  imb={imb:+9.2f}  curt={nc}")


if __name__ == "__main__":
    main()
