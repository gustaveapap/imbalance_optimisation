#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIMULATION SOLAIRE FR 2026 -- S1/S2/S3
Meme logique que sim_fr_2025.py : strategies VE-proven.
Telechargement RTE + ENTSO-E pour jours manquants dans data_ve_2026/.
Resultats dans outputs/solar_fr/simulation_2026/
"""

import sys, os, warnings, time, base64
from datetime import datetime, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================
REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(str(REPO_DIR))

DATA_2026   = REPO_DIR / "data_ve_2026"
DATA_2025   = REPO_DIR / "data_ve_2025"
RESULTS_DIR = REPO_DIR / "outputs" / "solar_fr" / "simulation_2026"
MERIT_CACHE = DATA_2026 / "merit_cache"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MERIT_CACHE.mkdir(exist_ok=True)

SUMMARY_FILE = RESULTS_DIR / "summary_2026.csv"
ERRORS_FILE  = RESULTS_DIR / "errors_2026.csv"

START_DATE = "2026-01-01"
END_DATE   = "2026-05-11"

from test_logic_fr import (
    dl_pvgis, dl_solar, pvgis_day,
    compute_strategies_v2, build_forecast_parc,
    qh_range, align_qh, file_ok, safe_float,
    forecast_fr, compute_ratios,
    PVGIS_FILE, MODEL_PATH,
)

CLIENT_ID     = "bdc03388-6c93-46f6-adbf-1a77d5b89684"
CLIENT_SECRET = "da352352-63f8-42ce-9a34-0624f7560a72"
ENTSOE_TOKEN  = "dcc0c513-dab6-4add-82be-7b693f2b7fab"
ENTSOE_API    = "https://web-api.tp.entsoe.eu/api"
FR_DOMAIN     = "10YFR-RTE------C"

SEP = "=" * 100

# =============================================================================
# AUTH RTE
# =============================================================================

def get_rte_token():
    basic = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    r = requests.post(
        "https://digital.iservices.rte-france.com/token/oauth/",
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Authorization": f"Basic {basic}"},
        data={"grant_type": "client_credentials"}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]

# =============================================================================
# IMBALANCE FR 2026 (cache ou telechargement RTE)
# =============================================================================

_rte_token = {"v": None, "ts": 0}

def ensure_token():
    if _rte_token["v"] is None or time.time() - _rte_token["ts"] > 3000:
        _rte_token["v"]  = get_rte_token()
        _rte_token["ts"] = time.time()
    return _rte_token["v"]

def dl_imbalance_fr_2026(date_str):
    p = DATA_2026 / f"imbalance_fr_{date_str}.csv"
    if file_ok(p):
        print(f"    imbalance_fr  : cache")
        return pd.read_csv(p, parse_dates=["timestamp"])
    token = ensure_token()
    s = datetime.strptime(date_str, "%Y-%m-%d")
    e = s + timedelta(days=1)
    for attempt in range(3):
        try:
            r = requests.get(
                "https://digital.iservices.rte-france.com/open_api/balancing_energy/v5/imbalance_data",
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                params={"start_date": s.strftime("%Y-%m-%dT%H:%M:%S+01:00"),
                        "end_date":   e.strftime("%Y-%m-%dT%H:%M:%S+01:00")},
                timeout=30)
            if r.status_code == 401:
                _rte_token["v"] = None; token = ensure_token(); continue
            rows = []
            for item in r.json().get("imbalance_data", []):
                for v in item.get("values", []):
                    rows.append({
                        "timestamp":    pd.to_datetime(v.get("start_date")),
                        "volume":       safe_float(v.get("imbalance")),
                        "prix_positif": safe_float(v.get("positive_imbalance_settlement_price")),
                        "prix_negatif": safe_float(v.get("negative_imbalance_settlement_price")),
                    })
            if not rows:
                print(f"    imbalance_fr  : VIDE"); return pd.DataFrame()
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df["timestamp"] = (df["timestamp"].dt.tz_convert("Europe/Paris")
                               .dt.tz_localize(None).dt.floor("15min"))
            df = align_qh(df, date_str, "ffill")
            df.to_csv(p, index=False)
            print(f"    imbalance_fr  : {len(df)} lignes -> cache")
            return df
        except Exception as ex:
            print(f"    imbalance_fr  : attempt {attempt+1} WARN {ex}")
            time.sleep(5 * (attempt + 1))
    print(f"    imbalance_fr  : ECHEC"); return pd.DataFrame()

# =============================================================================
# PRIX DA FR 2026 (cache ou ENTSO-E)
# =============================================================================

def dl_da_2026(date_str):
    p = DATA_2026 / f"prix_da_{date_str}.csv"
    if file_ok(p):
        print(f"    prix_da       : cache")
        return pd.read_csv(p, parse_dates=["timestamp"])
    start = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d%H%M")
    end   = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y%m%d%H%M")
    for attempt in range(3):
        try:
            r = requests.get(ENTSOE_API, params={
                "securityToken": ENTSOE_TOKEN, "documentType": "A44",
                "in_Domain": FR_DOMAIN, "out_Domain": FR_DOMAIN,
                "periodStart": start, "periodEnd": end,
            }, timeout=60)
            if r.status_code != 200:
                time.sleep(5 * (attempt + 1)); continue
            rows = []
            for period in ET.fromstring(r.content).findall(".//{*}Period"):
                s_el = period.find(".//{*}start")
                r_el = period.find(".//{*}resolution")
                if s_el is None: continue
                s_dt = pd.to_datetime(s_el.text, utc=True)
                res  = r_el.text if r_el is not None else "PT60M"
                for pt in period.findall(".//{*}Point"):
                    pos = int(pt.find(".//{*}position").text)
                    px  = float(pt.find(".//{*}price.amount").text)
                    off = (timedelta(minutes=(pos-1)*15) if res == "PT15M"
                           else timedelta(hours=(pos-1)))
                    ts  = (s_dt + off).tz_convert("Europe/Paris").tz_localize(None)
                    rows.append({"timestamp": ts, "price_eur_mwh": px})
            if not rows:
                print(f"    prix_da       : VIDE"); return pd.DataFrame()
            df = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
            df = align_qh(df, date_str, "ffill")
            df.to_csv(p, index=False)
            print(f"    prix_da       : {len(df)} lignes -> cache")
            return df
        except Exception as ex:
            time.sleep(5 * (attempt + 1))
    print(f"    prix_da       : ECHEC"); return pd.DataFrame()

# =============================================================================
# MERIT METRICS 2026
# =============================================================================

def build_merit_metrics_2026(date_str):
    cache = MERIT_CACHE / f"merit_{date_str}.csv"
    if file_ok(cache):
        df = pd.read_csv(cache, parse_dates=["timestamp"])
        for col in ["afrr_ratio_negative", "afrr_ratio_very_negative",
                    "mfrr_ratio_negative",  "mfrr_ratio_very_negative"]:
            if col not in df.columns: df[col] = 0.0
        print(f"    merit order EU : cache")
        return df

    dfs_afrr, dfs_mfrr = [], []
    for country in ["be", "de", "nl"]:
        for reserve, lst in [("afrr", dfs_afrr), ("mfrr", dfs_mfrr)]:
            f = DATA_2026 / f"{reserve}_{country}_{date_str}.csv"
            if file_ok(f):
                try: lst.append(pd.read_csv(f, parse_dates=["timestamp"]))
                except Exception: pass

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
        if col not in full.columns: full[col] = 0.0
        else: full[col] = full[col].fillna(0.0)

    full.to_csv(cache, index=False)
    print(f"    merit order EU : {len(dfs_afrr)+len(dfs_mfrr)} fichiers -> {len(full)} QH")
    return full

# =============================================================================
# SIMULATION UN JOUR
# =============================================================================

def simulate_day_2026(date_str, df_pvgis, df_hist):
    result_path = RESULTS_DIR / f"day_{date_str}.csv"
    if file_ok(result_path):
        df = pd.read_csv(result_path, parse_dates=["timestamp"])
        print(f"  {date_str}  -> cache ({len(df)} QH)")
        return df

    print(f"\n  {date_str}")
    df_imb    = dl_imbalance_fr_2026(date_str)
    df_da     = dl_da_2026(date_str)
    df_solar  = dl_solar(date_str)
    df_pv_day = pvgis_day(date_str, df_pvgis)

    if df_imb.empty or df_da.empty or df_pv_day.empty:
        print(f"    -> SKIP (imb={df_imb.empty} da={df_da.empty} pv={df_pv_day.empty})")
        return None

    metrics = build_merit_metrics_2026(date_str)
    df_fc   = forecast_fr(date_str, df_hist)
    if df_fc is not None: print(f"    forecast_fr   : {len(df_fc)} QH")
    else:                 print(f"    forecast_fr   : historique insuffisant")

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
        full["forecast_volume"] = 0.0; full["forecast_direction"] = "UP"

    if not df_solar.empty:
        full = full.merge(df_solar[["timestamp", "forecast_mw", "actual_mw"]],
                          on="timestamp", how="left")
    else:
        full["forecast_mw"] = 0.0; full["actual_mw"] = 0.0

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
            if s != "s1": row[f"{s}_curt"] = int(grp[f"{s}_curtail"].sum())
        row["n_solar_qh"] = int((grp["production_mw"] > 0.01).sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(SUMMARY_FILE, index=False)
    if errors: pd.DataFrame(errors).to_csv(ERRORS_FILE, index=False)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print(SEP)
    print("  SIMULATION SOLAIRE FR 2026  --  S1 / S2 / S3")
    print(f"  Periode : {START_DATE} -> {END_DATE}")
    print(f"  Data    : {DATA_2026}")
    print(f"  Outputs : {RESULTS_DIR}")
    print(SEP)

    print("\n  PVGIS..."); df_pvgis = dl_pvgis()
    print("\n  Auth RTE..."); ensure_token(); print("  OK")

    print("\n  Chargement historique imbalance FR (2025 + 2026)...")
    parts = []
    for f in sorted(DATA_2025.glob("imbalance_fr_2025-*.csv")):
        try: parts.append(pd.read_csv(f, parse_dates=["timestamp"]))
        except Exception: pass
    for f in sorted(DATA_2026.glob("imbalance_fr_2026-*.csv")):
        try: parts.append(pd.read_csv(f, parse_dates=["timestamp"]))
        except Exception: pass
    df_hist = (pd.concat(parts, ignore_index=True)
               .sort_values("timestamp").drop_duplicates("timestamp")
               .reset_index(drop=True)) if parts else pd.DataFrame(columns=["timestamp", "volume"])
    print(f"  {len(df_hist)} QH historiques charges")

    all_dates = [d.strftime("%Y-%m-%d")
                 for d in pd.date_range(START_DATE, END_DATE, freq="D")]
    print(f"\n  {len(all_dates)} jours a simuler\n")

    results, errors = [], []

    for i, date_str in enumerate(all_dates):
        try:
            df_day = simulate_day_2026(date_str, df_pvgis, df_hist)
        except Exception as e:
            print(f"  {date_str}  -> ERREUR: {e}")
            errors.append({"date": date_str, "error": str(e)}); continue

        if df_day is None:
            errors.append({"date": date_str, "error": "donnees manquantes"}); continue

        new_rows = df_day[["timestamp", "volume"]].dropna()
        df_hist = (pd.concat([df_hist, new_rows], ignore_index=True)
                   .sort_values("timestamp").drop_duplicates("timestamp"))
        results.append(df_day)

        s1 = df_day["s1_total"].sum(); s2 = df_day["s2_total"].sum(); s3 = df_day["s3_total"].sum()
        best = ["S1","S2","S3"][[s1,s2,s3].index(max(s1,s2,s3))]
        n_sol = int((df_day["production_mw"] > 0.01).sum())
        print(f"    {date_str}: {n_sol} QH sol  S1={s1:+.0f}  S2={s2:+.0f}  S3={s3:+.0f}  best={best}")

        if (i + 1) % 14 == 0 or i == len(all_dates) - 1:
            save_summary(results, errors)
            print(f"  [SAVE] {i+1}/{len(all_dates)} jours")

    print(f"\n{SEP}\n  RESUME FINAL\n{SEP}")
    if not results:
        print("  Aucun resultat."); return
    save_summary(results, errors)
    df_all = pd.concat(results, ignore_index=True)
    df_all["month"] = pd.to_datetime(df_all["timestamp"]).dt.to_period("M").astype(str)
    for mois in sorted(df_all["month"].unique()):
        dm   = df_all[df_all["month"] == mois]
        vals = [dm[f"s{k}_total"].sum() for k in [1,2,3]]
        best = ["S1","S2","S3"][vals.index(max(vals))]
        print(f"  {mois}  S1={vals[0]:+7.0f}  S2={vals[1]:+7.0f}  S3={vals[2]:+7.0f}  best={best}")

if __name__ == "__main__":
    main()
