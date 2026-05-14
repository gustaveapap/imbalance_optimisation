#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIMULATION SOLAIRE BE 2026 -- S1/S2/S3
Meme logique que sim_be_2025.py : strategies VE-proven.
Telecharge imbalance BE depuis ODS133 (volume) + ODS134 (ISP) si manquant.
Resultats dans outputs/solar_be/simulation_2026/
"""

import sys, os, warnings, time
from datetime import datetime, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import joblib

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================
REPO_DIR  = Path(__file__).resolve().parents[2]
SOLAR_FR  = REPO_DIR / "optimizers" / "solar_fr"
sys.path.insert(0, str(SOLAR_FR))
os.chdir(str(REPO_DIR))

DATA_2026   = REPO_DIR / "data_ve_2026"
DATA_2025   = REPO_DIR / "data_ve_2025"
DATA_SOLAR  = REPO_DIR / "data" / "raw" / "solar_be"
RESULTS_DIR = REPO_DIR / "outputs" / "solar_be" / "simulation_2026"
MERIT_CACHE = DATA_2026 / "merit_cache"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MERIT_CACHE.mkdir(exist_ok=True)

PVGIS_FILE   = DATA_SOLAR / "pvgis_be_2019.csv"
SUMMARY_FILE = RESULTS_DIR / "summary_2026.csv"
ERRORS_FILE  = RESULTS_DIR / "errors_2026.csv"

START_DATE = "2026-01-01"
END_DATE   = "2026-05-11"

MODEL_PATH_BE = REPO_DIR / "models" / "be_imbalance_model.joblib"
MAX_LAG_BE    = 16

ENTSOE_TOKEN     = "dcc0c513-dab6-4add-82be-7b693f2b7fab"
ENTSOE_API       = "https://web-api.tp.entsoe.eu/api"
ENTSOE_SOLAR_URL = "https://transparency.entsoe.eu/generation/forecast/windAndSolar/solar/load"
BE_DOMAIN        = "10YBE----------2"
BE_AREA          = "BZN|10YBE----------2"
BRUSSELS         = "Europe/Brussels"
ODS133_URL       = "https://external-elia.opendatasoft.com/api/records/1.0/search/"
ODS134_URL       = "https://external-elia.opendatasoft.com/api/records/1.0/search/"
SEUIL_DA_MID     = 30

SEP = "=" * 100

from test_logic_fr import (
    pvgis_day, qh_range, align_qh, file_ok, compute_ratios,
    build_forecast_parc, safe_float,
)

# =============================================================================
# PVGIS BE
# =============================================================================

def dl_pvgis_be():
    if not file_ok(PVGIS_FILE):
        raise FileNotFoundError(f"PVGIS BE introuvable: {PVGIS_FILE}")
    df = pd.read_csv(PVGIS_FILE, parse_dates=["timestamp"])
    print(f"  pvgis_be        : {len(df)} heures  max={df['production_mw'].max():.3f} MW/GW")
    return df

# =============================================================================
# IMBALANCE BE 2026 — ODS133 (volume SI) + ODS134 (ISP QH)
# =============================================================================

def _fetch_ods133(date_str):
    import datetime as dt_mod
    d = dt_mod.date.fromisoformat(date_str)
    start_utc = dt_mod.datetime(d.year, d.month, d.day, tzinfo=dt_mod.timezone.utc) - dt_mod.timedelta(hours=2)
    end_utc   = dt_mod.datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=dt_mod.timezone.utc) + dt_mod.timedelta(hours=2)
    q = (f"datetime:[{start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')} "
         f"TO {end_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}]")
    frames, start = [], 0
    while True:
        for attempt in range(3):
            try:
                r = requests.get(ODS133_URL, params={
                    "dataset": "ods133", "q": q,
                    "rows": 1000, "start": start, "sort": "datetime",
                }, timeout=60)
                if r.status_code == 429: time.sleep(10); continue
                r.raise_for_status()
                records = r.json().get("records", [])
                break
            except Exception as e:
                time.sleep(5 * (attempt + 1))
        else:
            return pd.DataFrame()
        rows = []
        for rec in records:
            f = rec.get("fields", {})
            dt_raw = f.get("datetime", "")
            si_val = f.get("systemimbalance")
            if dt_raw and si_val is not None:
                try:
                    ts = pd.to_datetime(dt_raw, utc=True).tz_convert(BRUSSELS).tz_localize(None)
                    rows.append({"datetime": ts, "volume": float(si_val)})
                except Exception: pass
        if rows: frames.append(pd.DataFrame(rows))
        if len(records) < 1000: break
        start += 1000
    if not frames: return pd.DataFrame()
    df = (pd.concat(frames, ignore_index=True)
          .drop_duplicates("datetime").sort_values("datetime"))
    df = df[df["datetime"].dt.date == d]
    # Aggreger en QH (moyenne)
    df["timestamp"] = df["datetime"].dt.floor("15min")
    return df.groupby("timestamp", as_index=False)["volume"].mean()

def _fetch_ods134(date_str):
    ws = (pd.Timestamp(date_str).tz_localize(BRUSSELS, nonexistent="shift_forward")
          .tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
    we = ((pd.Timestamp(date_str) + pd.Timedelta(days=1))
          .tz_localize(BRUSSELS, nonexistent="shift_forward")
          .tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
    all_records, start_i = [], 0
    while True:
        for attempt in range(3):
            try:
                r = requests.get(ODS134_URL, params={
                    "dataset": "ods134",
                    "q": f"datetime:[{ws} TO {we}]",
                    "rows": 1000, "start": start_i, "sort": "datetime",
                }, timeout=30)
                r.raise_for_status()
                data = r.json().get("records", [])
                break
            except Exception: time.sleep(5)
        else: break
        all_records.extend(data)
        if len(data) < 1000: break
        start_i += 1000
        if start_i >= 9000: break
    rows = []
    for rec in all_records:
        f = rec.get("fields", {})
        dt_raw = f.get("datetime")
        if dt_raw:
            try:
                ts = pd.to_datetime(dt_raw, utc=True).tz_convert(BRUSSELS).tz_localize(None)
                rows.append({"timestamp": ts.floor("15min"), "isp": f.get("imbalanceprice")})
            except Exception: pass
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["isp"] = pd.to_numeric(df["isp"], errors="coerce")
    return df.drop_duplicates("timestamp")

def dl_imbalance_be_2026(date_str):
    p = DATA_2026 / f"imbalance_be_{date_str}.csv"
    if file_ok(p):
        df = pd.read_csv(p, parse_dates=["timestamp"])
        print(f"    imbalance_be  : cache ({len(df)} lignes)")
        return df
    print(f"    imbalance_be  : telechargement ODS133+ODS134...")
    df_vol = _fetch_ods133(date_str)
    df_isp = _fetch_ods134(date_str)
    if df_vol.empty:
        print(f"    imbalance_be  : ODS133 VIDE"); return pd.DataFrame()
    if df_isp.empty:
        df_vol["isp"] = 0.0
        df = align_qh(df_vol, date_str, "ffill")
    else:
        df = df_vol.merge(df_isp, on="timestamp", how="left")
        df = align_qh(df, date_str, "ffill")
        df["isp"] = df["isp"].fillna(0)
    df["volume"] = df["volume"].fillna(0)
    df.to_csv(p, index=False)
    print(f"    imbalance_be  : {len(df)} QH -> cache")
    return df

# =============================================================================
# PRIX DA BE 2026 (ENTSO-E A44)
# =============================================================================

def dl_da_be_2026(date_str):
    cache = DATA_2026 / f"prix_da_be_{date_str}.csv"
    if file_ok(cache):
        df = pd.read_csv(cache, parse_dates=["timestamp"])
        print(f"    prix_da_be    : cache ({len(df)} lignes)")
        return df
    start = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d%H%M")
    end   = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y%m%d%H%M")
    for attempt in range(5):
        try:
            r = requests.get(ENTSOE_API, params={
                "securityToken": ENTSOE_TOKEN, "documentType": "A44",
                "in_Domain": BE_DOMAIN, "out_Domain": BE_DOMAIN,
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
                    off = (timedelta(minutes=(pos-1)*15) if res=="PT15M"
                           else timedelta(hours=(pos-1)))
                    ts  = (s_dt + off).tz_convert(BRUSSELS).tz_localize(None)
                    rows.append({"timestamp": ts, "price_eur_mwh": px})
            if not rows:
                print(f"    prix_da_be    : VIDE"); return pd.DataFrame()
            df = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
            df.to_csv(cache, index=False)
            print(f"    prix_da_be    : {len(df)} lignes -> cache")
            return df
        except Exception as e:
            time.sleep(5 * (attempt + 1))
    print(f"    prix_da_be    : ECHEC"); return pd.DataFrame()

# =============================================================================
# SOLAR ENTSO-E BE 2026
# =============================================================================

def dl_solar_be_2026(date_str):
    cache = DATA_2026 / f"solar_be_{date_str}.csv"
    if file_ok(cache):
        print(f"    solar_be      : cache")
        return pd.read_csv(cache, parse_dates=["timestamp"])
    from datetime import timezone
    date_from = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    date_to   = date_from + timedelta(days=1)
    try:
        r = requests.post(ENTSOE_SOLAR_URL, json={
            "dateTimeRange": {
                "from": date_from.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "to":   date_to.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            },
            "areaList": [BE_AREA], "timeZone": "CET",
            "sorterList": [], "filterMap": {},
        }, headers={"accept": "application/json",
                    "content-type": "application/json; charset=utf-8"}, timeout=90)
        if r.status_code != 200:
            print(f"    solar_be      : HTTP {r.status_code}"); return pd.DataFrame()
        rows = []
        for inst in r.json().get("instanceList", []):
            if inst.get("businessDimensionMap", {}).get("PRODUCTION_TYPE") != "B16": continue
            for period in inst.get("curveData", {}).get("periodList", []):
                st  = period.get("timeInterval", {}).get("from")
                if not st: continue
                res  = period.get("resolution")
                s_dt = pd.to_datetime(st, utc=True)
                for pos_str, vals in period.get("pointMap", {}).items():
                    pos = int(pos_str)
                    off = timedelta(minutes=pos*15) if res=="PT15M" else timedelta(hours=pos)
                    ts  = (s_dt + off).tz_convert(BRUSSELS).tz_localize(None)
                    v   = vals if isinstance(vals, list) else []
                    rows.append({
                        "timestamp":   ts,
                        "forecast_mw": safe_float(v[2]) if len(v) > 2 else np.nan,
                        "actual_mw":   safe_float(v[3]) if len(v) > 3 else np.nan,
                    })
        if not rows:
            print(f"    solar_be      : VIDE"); return pd.DataFrame()
        df = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
        df = align_qh(df, date_str, "interp")
        df["forecast_mw"] = df["forecast_mw"].fillna(0)
        df["actual_mw"]   = df["actual_mw"].fillna(0)
        df.to_csv(cache, index=False)
        print(f"    solar_be      : {len(df)} QH -> cache")
        return df
    except Exception as e:
        print(f"    solar_be      : WARN {e}"); return pd.DataFrame()

# =============================================================================
# MERIT METRICS BE 2026
# =============================================================================

def build_merit_metrics_be_2026(date_str):
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
# FORECAST BE (walk-forward, modele pre-entraine 2025)
# =============================================================================

_be_model_cache = {}

def forecast_be(date_str, df_hist):
    if not MODEL_PATH_BE.exists(): return None
    if df_hist is None or df_hist.empty: return None
    try:
        if "model" not in _be_model_cache:
            _be_model_cache["model"] = joblib.load(MODEL_PATH_BE)
        model = _be_model_cache["model"]
        X, valid_ts = [], []
        for ts in qh_range(date_str):
            hist = df_hist[df_hist["timestamp"] < ts].tail(MAX_LAG_BE)
            if len(hist) < MAX_LAG_BE: continue
            lags  = hist["volume"].values[::-1]
            slot  = ts.hour * 4 + ts.minute // 15
            feats = list(lags) + [
                np.sin(2*np.pi*slot/96),            np.cos(2*np.pi*slot/96),
                np.sin(2*np.pi*ts.dayofweek/7),     np.cos(2*np.pi*ts.dayofweek/7),
                np.sin(2*np.pi*ts.month/12),        np.cos(2*np.pi*ts.month/12),
            ]
            X.append(feats); valid_ts.append(ts)
        if not X: return None
        preds = model.predict(np.array(X, dtype=np.float32))
        return pd.DataFrame([{
            "timestamp": ts, "forecast_volume": float(p),
            "forecast_direction": "DOWN" if p > 0 else "UP",
        } for ts, p in zip(valid_ts, preds)])
    except Exception as e:
        print(f"    forecast_be   : WARN {e}"); return None

# =============================================================================
# REVENUE IMBALANCE BE
# =============================================================================

def rev_imb_be(ecart, isp):
    rev = np.zeros(len(ecart))
    m = (ecart > 0) & (isp < 0);  rev[m] =  ecart[m] * np.abs(isp[m]) / 4
    m = (ecart < 0) & (isp > 0);  rev[m] =  np.abs(ecart[m]) * isp[m] / 4
    m = (ecart > 0) & (isp > 0);  rev[m] = -ecart[m] * isp[m] / 4
    m = (ecart < 0) & (isp < 0);  rev[m] = -np.abs(ecart[m]) * np.abs(isp[m]) / 4
    return rev

# =============================================================================
# STRATEGIES BE (identiques a sim_be_2025)
# =============================================================================

def compute_strategies_be(df):
    fp   = df["forecast_parc"].values
    prod = df["production_mw"].values
    da   = df["price_eur_mwh"].values
    isp  = df["isp"].fillna(0).values
    vol   = np.abs(df["forecast_volume"].fillna(0).values)
    fdir  = df["forecast_direction"].fillna("UP").values
    mfrr_neg  = df["mfrr_ratio_negative"].fillna(0).values
    afrr_neg  = df["afrr_ratio_negative"].fillna(0).values
    solar_on  = prod > 0.01

    sd = (solar_on & (fdir == "DOWN")
          & (afrr_neg > 50) & (mfrr_neg > 50) & (vol > 300))
    df["signal_down_150"] = sd

    df["s1_nomination"]  = fp
    df["s1_production"]  = prod
    df["s1_ecart"]       = fp - prod
    df["s1_revenue_imb"] = rev_imb_be(df["s1_ecart"].values, isp)
    df["s1_revenue_da"]  = fp * da / 4
    df["s1_total"]       = df["s1_revenue_da"] + df["s1_revenue_imb"]

    curtail_s2 = sd & solar_on
    nom_s2 = np.where(da < 0, 0.0, np.where(da < SEUIL_DA_MID, fp * 0.5, fp))
    df["s2_curtail"]     = curtail_s2
    df["s2_nomination"]  = nom_s2
    df["s2_production"]  = np.where(curtail_s2, 0.0, prod)
    df["s2_ecart"]       = nom_s2 - df["s2_production"].values
    df["s2_revenue_imb"] = rev_imb_be(df["s2_ecart"].values, isp)
    df["s2_revenue_da"]  = nom_s2 * da / 4
    df["s2_total"]       = df["s2_revenue_da"] + df["s2_revenue_imb"]

    curtail_s3 = sd & solar_on
    nom_s3 = fp * 0.1
    df["s3_curtail"]     = curtail_s3
    df["s3_nomination"]  = nom_s3
    df["s3_production"]  = np.where(curtail_s3, 0.0, prod)
    df["s3_ecart"]       = nom_s3 - df["s3_production"].values
    df["s3_revenue_imb"] = rev_imb_be(df["s3_ecart"].values, isp)
    df["s3_revenue_da"]  = nom_s3 * da / 4
    df["s3_total"]       = df["s3_revenue_da"] + df["s3_revenue_imb"]

    return df

# =============================================================================
# SIMULATION UN JOUR
# =============================================================================

def simulate_day_be_2026(date_str, df_pvgis, df_hist):
    result_path = RESULTS_DIR / f"day_{date_str}.csv"
    if file_ok(result_path):
        df = pd.read_csv(result_path, parse_dates=["timestamp"])
        print(f"  {date_str}  -> cache ({len(df)} QH)")
        return df

    print(f"\n  {date_str}")
    df_imb    = dl_imbalance_be_2026(date_str)
    df_da     = dl_da_be_2026(date_str)
    df_pv_day = pvgis_day(date_str, df_pvgis)
    df_solar  = dl_solar_be_2026(date_str)

    if df_imb.empty or df_da.empty or df_pv_day.empty:
        print(f"    -> SKIP (imb={df_imb.empty} da={df_da.empty} pv={df_pv_day.empty})")
        return None

    metrics = build_merit_metrics_be_2026(date_str)
    df_fc   = forecast_be(date_str, df_hist)
    if df_fc is not None: print(f"    forecast_be   : {len(df_fc)} QH")
    else:                 print(f"    forecast_be   : historique insuffisant -> volume=0")

    full = pd.DataFrame({"timestamp": qh_range(date_str)})
    full = full.merge(df_imb[["timestamp", "volume", "isp"]], on="timestamp", how="left")
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
    full["isp"]                = full["isp"].ffill().bfill().fillna(0)
    full["forecast_volume"]    = full["forecast_volume"].fillna(0)
    full["forecast_direction"] = full["forecast_direction"].fillna("UP")
    full["forecast_mw"]        = full["forecast_mw"].fillna(0)
    full["actual_mw"]          = full["actual_mw"].fillna(1).replace(0, 1)

    full["forecast_parc"] = build_forecast_parc(
        full["production_mw"], full[["timestamp", "forecast_mw", "actual_mw"]])
    full["date"] = date_str
    full = compute_strategies_be(full)
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
        row["n_solar_qh"]   = int((grp["production_mw"] > 0.01).sum())
        row["prod_mwh_day"] = grp["production_mw"].sum() / 4.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(SUMMARY_FILE, index=False)
    if errors: pd.DataFrame(errors).to_csv(ERRORS_FILE, index=False)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print(SEP)
    print("  SIMULATION SOLAIRE BE 2026  --  S1 / S2 / S3")
    print(f"  Periode : {START_DATE} -> {END_DATE}")
    print(f"  Data    : {DATA_2026}")
    print(f"  Outputs : {RESULTS_DIR}")
    print(SEP)

    print("\n  PVGIS BE..."); df_pvgis = dl_pvgis_be()

    # df_hist : 2025 complet + walk-forward 2026
    print("\n  Chargement historique imbalance BE (2025 + 2026)...")
    parts = []
    for f in sorted(DATA_2025.glob("imbalance_be_2025-*.csv")):
        try: parts.append(pd.read_csv(f, parse_dates=["timestamp"]))
        except Exception: pass
    for f in sorted(DATA_2026.glob("imbalance_be_2026-*.csv")):
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
            df_day = simulate_day_be_2026(date_str, df_pvgis, df_hist)
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

# =============================================================================
# MODE CONTINU  (python sim_be_2026.py --continuous)
# =============================================================================

def run_continuous():
    """Simule chaque nouveau jour des que les donnees sont disponibles (vers 02:00)."""
    import sys as _sys
    print(SEP)
    print("  SIMULATION SOLAIRE BE 2026 -- MODE CONTINU")
    print(SEP)

    print("\n  PVGIS BE..."); df_pvgis = dl_pvgis_be()

    while True:
        today     = datetime.now().date()
        yesterday = today - timedelta(days=1)

        # Dernier jour simule
        last_sim = (datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=1)).date()
        if SUMMARY_FILE.exists():
            try:
                df_s = pd.read_csv(SUMMARY_FILE)
                if not df_s.empty:
                    last_sim = datetime.strptime(df_s["date"].max(), "%Y-%m-%d").date()
            except Exception:
                pass

        new_dates = [d.strftime("%Y-%m-%d")
                     for d in pd.date_range(
                         datetime.strptime(START_DATE, "%Y-%m-%d").date(),
                         yesterday, freq="D")
                     if d.date() > last_sim]

        if not new_dates:
            # Attendre demain 02:00
            wake = datetime.combine(today + timedelta(days=1),
                                    datetime.min.time()) + timedelta(hours=2)
            delta = max(300, (wake - datetime.now()).total_seconds())
            print(f"  [{datetime.now().strftime('%H:%M')}] BE Solar a jour ({last_sim}). "
                  f"Prochain cycle : {wake.strftime('%Y-%m-%d %H:%M')} ({delta/3600:.1f}h)",
                  flush=True)
            time.sleep(delta)
            continue

        print(f"\n  {len(new_dates)} jours a traiter : {new_dates[0]} -> {new_dates[-1]}")

        # Recharger historique imbalance
        parts = []
        for f in sorted(DATA_2025.glob("imbalance_be_2025-*.csv")):
            try: parts.append(pd.read_csv(f, parse_dates=["timestamp"]))
            except Exception: pass
        for f in sorted(DATA_2026.glob("imbalance_be_2026-*.csv")):
            try: parts.append(pd.read_csv(f, parse_dates=["timestamp"]))
            except Exception: pass
        df_hist = (pd.concat(parts, ignore_index=True)
                   .sort_values("timestamp").drop_duplicates("timestamp")
                   .reset_index(drop=True)) if parts else pd.DataFrame(columns=["timestamp", "volume"])

        # Charger resultats deja calcules
        results = []
        for f in sorted(RESULTS_DIR.glob("day_2026-*.csv")):
            try: results.append(pd.read_csv(f, parse_dates=["timestamp"]))
            except Exception: pass
        errors = []

        for i, date_str in enumerate(new_dates):
            try:
                df_day = simulate_day_be_2026(date_str, df_pvgis, df_hist)
            except Exception as e:
                print(f"  {date_str} -> ERREUR: {e}")
                errors.append({"date": date_str, "error": str(e)}); continue

            if df_day is None:
                print(f"  {date_str} -> donnees manquantes, reessai au prochain cycle")
                errors.append({"date": date_str, "error": "donnees manquantes"}); continue

            df_hist = (pd.concat([df_hist, df_day[["timestamp", "volume"]].dropna()],
                                 ignore_index=True)
                       .sort_values("timestamp").drop_duplicates("timestamp"))
            results.append(df_day)

            s1 = df_day["s1_total"].sum(); s2 = df_day["s2_total"].sum()
            s3 = df_day["s3_total"].sum()
            best = ["S1","S2","S3"][[s1,s2,s3].index(max(s1,s2,s3))]
            n_sol = int((df_day["production_mw"] > 0.01).sum())
            print(f"    {date_str}: {n_sol} QH sol  "
                  f"S1={s1:+.0f}  S2={s2:+.0f}  S3={s3:+.0f}  best={best}", flush=True)

        if results:
            save_summary(results, errors)
            print(f"  [SAVE] {len(results)} jours dans summary")

if __name__ == "__main__":
    import sys as _sys
    if "--continuous" in _sys.argv:
        run_continuous()
    else:
        main()
