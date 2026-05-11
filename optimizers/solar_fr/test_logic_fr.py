#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST LOGIQUE FR -- S1/S2/S3/S4 CORRIGEES (autonome)
====================================================
Telecharge 6 jours cibles + teste les 4 strategies corrigees.

Jours cibles :
  Jan 2026 : 2026-01-02 (S2>S1), 2026-01-04 (signal_DOWN), 2026-01-16 (DA max)
  Avr 2026 : 2026-04-05, 2026-04-06, 2026-04-07 (prix negatifs printemps)

Localisation : C:/Users/ApapGustave/test_logic_fr.py
Commande     : python test_logic_fr.py
"""

import base64
import time
import warnings
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import joblib

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
CLIENT_ID_IMBALANCE     = "bdc03388-6c93-46f6-adbf-1a77d5b89684"
CLIENT_SECRET_IMBALANCE = "da352352-63f8-42ce-9a34-0624f7560a72"
API_KEY_TENNET          = "18fe140e-12a2-446e-ab89-455d33709fac"
ENTSOE_TOKEN            = "dcc0c513-dab6-4add-82be-7b693f2b7fab"

BASE_URL_RTE          = "https://digital.iservices.rte-france.com/open_api"
BASE_URL_ENTSOE_API   = "https://web-api.tp.entsoe.eu/api"
BASE_URL_ENTSOE_SOLAR = "https://transparency.entsoe.eu/generation/forecast/windAndSolar/solar/load"
PVGIS_URL             = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"

DATA_DIR   = Path("data_solar_optim")
OUT_DIR    = Path("simulation_solar_intelligent")
MODEL_PATH = Path("C:/Users/gusta/rte_forecaster/artifacts/fr_imbalance_full_model.pkl")
PVGIS_FILE = DATA_DIR / "pvgis_2019_hourly.csv"

DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

SEUIL_DA_MID  = 30
DEVIATION_MAX = 0.5284
MAX_LAG       = 96

import pandas as _pd
TEST_DAYS = [d.strftime("%Y-%m-%d") for d in _pd.date_range("2026-01-01", "2026-04-15")]

SEP  = "=" * 100
SEP2 = "-" * 100

# =============================================================================
# UTILITAIRES
# =============================================================================

def safe_float(x):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def retry(func, n=3, delay=3):
    for i in range(n):
        try:
            return func()
        except Exception as e:
            if i < n - 1:
                time.sleep(delay)
            else:
                raise

def qh_range(date_str):
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return pd.date_range(d, d + timedelta(days=1), freq="15min", inclusive="left")

def align_qh(df, date_str, method="ffill"):
    if df is None or df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df["timestamp"] = df["timestamp"].dt.floor("15min")
    target = datetime.strptime(date_str, "%Y-%m-%d").date()
    df = df[df["timestamp"].dt.date == target]
    if df.empty:
        return pd.DataFrame()
    num_cols = [c for c in df.columns if c != "timestamp"
                and pd.api.types.is_numeric_dtype(df[c])]
    if df["timestamp"].duplicated().any():
        df = df.groupby("timestamp", as_index=False)[num_cols].mean()
    full = pd.DataFrame({"timestamp": qh_range(date_str)})
    df   = full.merge(df, on="timestamp", how="left")
    if method == "ffill":
        df[num_cols] = df[num_cols].ffill()
    else:
        df[num_cols] = df[num_cols].interpolate(method="linear").ffill().bfill()
    return df

def file_ok(p):
    return p.exists() and p.stat().st_size > 50

# =============================================================================
# TELECHARGEMENTS
# =============================================================================

def get_rte_token():
    basic = base64.b64encode(
        f"{CLIENT_ID_IMBALANCE}:{CLIENT_SECRET_IMBALANCE}".encode()).decode()
    r = requests.post(
        "https://digital.iservices.rte-france.com/token/oauth/",
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Authorization": f"Basic {basic}"},
        data={"grant_type": "client_credentials"}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]

def dl_imbalance_fr(date_str, token):
    p = DATA_DIR / f"imbalance_fr_{date_str}.csv"
    if file_ok(p):
        print(f"    imbalance_fr  : cache")
        return pd.read_csv(p, parse_dates=["timestamp"])
    s = datetime.strptime(date_str, "%Y-%m-%d")
    e = s + timedelta(days=1)
    r = retry(lambda: requests.get(
        f"{BASE_URL_RTE}/balancing_energy/v5/imbalance_data",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params={"start_date": s.strftime("%Y-%m-%dT%H:%M:%S+01:00"),
                "end_date":   e.strftime("%Y-%m-%dT%H:%M:%S+01:00")},
        timeout=30))
    try:
        data = r.json()
    except Exception:
        print(f"    imbalance_fr  : reponse vide/invalide, refresh token...")
        return None
    rows = []
    for item in data.get("imbalance_data", []):
        for v in item.get("values", []):
            rows.append({
                "timestamp":    pd.to_datetime(v.get("start_date")),
                "volume":       safe_float(v.get("imbalance")),
                "prix_positif": safe_float(v.get("positive_imbalance_settlement_price")),
                "prix_negatif": safe_float(v.get("negative_imbalance_settlement_price")),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"    imbalance_fr  : VIDE")
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return pd.DataFrame()
    df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Paris").dt.tz_localize(None)
    df["timestamp"] = df["timestamp"].dt.floor("15min")
    df = align_qh(df, date_str, "ffill")
    df.to_csv(p, index=False)
    print(f"    imbalance_fr  : {len(df)} lignes")
    return df

def dl_da(date_str):
    p = DATA_DIR / f"prix_da_{date_str}.csv"
    if file_ok(p):
        print(f"    prix_da       : cache")
        return pd.read_csv(p, parse_dates=["timestamp"])
    start = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d%H%M")
    end   = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y%m%d%H%M")
    r = retry(lambda: requests.get(BASE_URL_ENTSOE_API, params={
        "securityToken": ENTSOE_TOKEN, "documentType": "A44",
        "in_Domain": "10YFR-RTE------C", "out_Domain": "10YFR-RTE------C",
        "periodStart": start, "periodEnd": end,
    }, timeout=60))
    if r.status_code != 200:
        print(f"    prix_da       : ERREUR {r.status_code}")
        return pd.DataFrame()
    rows = []
    for period in ET.fromstring(r.content).findall(".//{*}Period"):
        s_el = period.find(".//{*}start")
        r_el = period.find(".//{*}resolution")
        if s_el is None:
            continue
        s_dt = pd.to_datetime(s_el.text, utc=True)
        res  = r_el.text if r_el is not None else "PT60M"
        for pt in period.findall(".//{*}Point"):
            pos = int(pt.find(".//{*}position").text)
            px  = float(pt.find(".//{*}price.amount").text)
            off = timedelta(minutes=(pos-1)*15) if res == "PT15M" else timedelta(hours=(pos-1))
            ts  = (s_dt + off).tz_convert("Europe/Paris").tz_localize(None)
            rows.append({"timestamp": ts, "price_eur_mwh": px})
    if not rows:
        print(f"    prix_da       : VIDE")
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
    df = align_qh(df, date_str, "ffill")
    df.to_csv(p, index=False)
    print(f"    prix_da       : {len(df)} lignes")
    return df

def dl_solar(date_str):
    p = DATA_DIR / f"solar_{date_str}.csv"
    if file_ok(p):
        print(f"    solar         : cache")
        return pd.read_csv(p, parse_dates=["timestamp"])
    date_from = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    date_to   = date_from + timedelta(days=1)
    r = retry(lambda: requests.post(BASE_URL_ENTSOE_SOLAR, json={
        "dateTimeRange": {
            "from": date_from.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "to":   date_to.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        },
        "areaList": ["BZN|10YFR-RTE------C"], "timeZone": "CET",
        "sorterList": [], "filterMap": {},
    }, headers={"accept": "application/json",
                "content-type": "application/json; charset=utf-8"}, timeout=90))
    rows = []
    for inst in r.json().get("instanceList", []):
        if inst.get("businessDimensionMap", {}).get("PRODUCTION_TYPE") != "B16":
            continue
        for period in inst.get("curveData", {}).get("periodList", []):
            st   = period.get("timeInterval", {}).get("from")
            if not st:
                continue
            res  = period.get("resolution")
            s_dt = pd.to_datetime(st, utc=True)
            for pos_str, vals in period.get("pointMap", {}).items():
                pos = int(pos_str)
                off = timedelta(minutes=pos*15) if res == "PT15M" else timedelta(hours=pos)
                ts  = (s_dt + off).tz_convert("Europe/Paris").tz_localize(None)
                v   = vals if isinstance(vals, list) else []
                rows.append({
                    "timestamp":   ts,
                    "forecast_mw": safe_float(v[2]) if len(v) > 2 else np.nan,
                    "actual_mw":   safe_float(v[3]) if len(v) > 3 else np.nan,
                })
    if not rows:
        print(f"    solar         : VIDE")
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
    df = align_qh(df, date_str, "interp")
    df["forecast_mw"] = df["forecast_mw"].fillna(0)
    df["actual_mw"]   = df["actual_mw"].fillna(0)
    df.to_csv(p, index=False)
    print(f"    solar         : {len(df)} lignes")
    return df

def dl_pvgis():
    if file_ok(PVGIS_FILE):
        print(f"  pvgis           : cache")
        return pd.read_csv(PVGIS_FILE, parse_dates=["timestamp"])
    print(f"  pvgis           : telechargement...")
    r = requests.get(PVGIS_URL, params={
        "lat": 46.8, "lon": 1.7, "startyear": 2019, "endyear": 2019,
        "pvcalculation": 1, "peakpower": 1000, "loss": 14,
        "angle": 35, "aspect": 0, "outputformat": "json",
    }, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["outputs"]["hourly"])
    df["timestamp"]     = (pd.to_datetime(df["time"], format="%Y%m%d:%H%M")
                           - pd.Timedelta(minutes=10))
    df["production_mw"] = df["P"] / 1_000_000.0
    df = df[["timestamp", "production_mw"]].sort_values("timestamp")
    df.to_csv(PVGIS_FILE, index=False)
    print(f"  pvgis           : {len(df)} heures  max={df['production_mw'].max():.3f}")
    return df

def pvgis_day(date_str, df_pvgis):
    date_obj   = datetime.strptime(date_str, "%Y-%m-%d")
    target_doy = date_obj.timetuple().tm_yday
    df_day = df_pvgis[df_pvgis["timestamp"].dt.dayofyear == target_doy].copy()
    if df_day.empty:
        return pd.DataFrame()
    df_day["timestamp"] = df_day["timestamp"].apply(
        lambda x: x.replace(year=date_obj.year, month=date_obj.month, day=date_obj.day))
    df_day = (df_day.set_index("timestamp")
              .resample("15min").interpolate(method="linear")
              .reset_index())
    df_day = align_qh(df_day, date_str, "interp")
    df_day["production_mw"] = df_day["production_mw"].fillna(0)
    return df_day[["timestamp", "production_mw"]]

def dl_elia(date_str, product, direction):
    date_obj  = datetime.strptime(date_str, "%Y-%m-%d")
    if date_obj < datetime(2026, 1, 22):
        ds_up, ds_dn = "ods156", "ods157"
    else:
        ds_up, ds_dn = "ods163", "ods164"
    dataset   = ds_up if direction == "incremental" else ds_dn
    dir_label = "UP" if direction == "incremental" else "DOWN"
    date_end  = date_obj + timedelta(days=1)
    base_url  = "https://opendata.elia.be/api/records/1.0/search/"
    all_recs, idx = [], 0
    while True:
        r = retry(lambda: requests.get(base_url, params={
            "dataset": dataset, "rows": 1000, "start": idx,
            "refine.balancingproduct": product, "sort": "datetime",
            "q": (f"datetime:[{date_str}T00:00:00 TO "
                  f"{date_end.strftime('%Y-%m-%d')}T00:00:00]"),
        }, timeout=45))
        data = r.json().get("records", [])
        if not data:
            break
        all_recs.extend(data)
        if len(data) < 1000:
            break
        idx += 1000
        time.sleep(0.2)
    offers = []
    for rec in all_recs:
        f  = rec.get("fields", {})
        dt = pd.to_datetime(f.get("datetime", ""), errors="coerce", utc=True)
        if pd.isna(dt):
            continue
        ts    = dt.tz_convert("Europe/Brussels").tz_localize(None).floor("15min")
        price = safe_float(f.get("energybidmarginalprice"))
        qty   = safe_float(f.get("energybidvolume"))
        if np.isnan(price) or np.isnan(qty) or qty <= 0:
            continue
        offers.append({"timestamp": ts, "direction": dir_label,
                       "price": price, "quantity": qty,
                       "country": "BE", "reserve": product})
    return pd.DataFrame(offers)

def dl_regelleistung(date_str, product):
    r = retry(lambda: requests.get(
        "https://www.regelleistung.net/apps/cpp-publisher/api/v2/tenders/results/anonymous",
        params={"productType": product, "market": "ENERGY",
                "exportFormat": "xlsx", "deliveryDate": date_str},
        timeout=60))
    if r.status_code != 200 or r.content[:2] != b"PK":
        return pd.DataFrame()
    tmp = DATA_DIR / f"_tmp_{product}_de_{date_str}.xlsx"
    tmp.write_bytes(r.content)
    try:
        xde = pd.read_excel(tmp, sheet_name=0)
    except Exception:
        tmp.unlink(missing_ok=True)
        return pd.DataFrame()
    tmp.unlink(missing_ok=True)
    if "PRODUCT" not in xde.columns:
        return pd.DataFrame()
    xde["QH_IDX"] = (xde["PRODUCT"].astype(str).str.extract(r"(\d{3})")
                     .astype(float).astype("Int64"))
    xde = xde.dropna(subset=["QH_IDX"])
    xde["DeliveryDate"] = pd.to_datetime(xde["DELIVERY_DATE"]).dt.normalize()
    xde["timestamp"]    = (xde["DeliveryDate"]
                           + pd.to_timedelta((xde["QH_IDX"] - 1) * 15, unit="m"))
    offers = []
    for prefix, direction in [("POS", "UP"), ("NEG", "DOWN")]:
        sub = xde[xde["PRODUCT"].astype(str).str.startswith(prefix)].copy()
        if sub.empty or "ENERGY_PRICE_[EUR/MWh]" not in sub.columns:
            continue
        if "ALLOCATED_CAPACITY_[MW]" not in sub.columns:
            continue
        sub = sub.copy()
        # Regelleistung DE stocke les prix comme positifs avec direction GRID_TO_PROVIDER
        # pour les bids NEG (DOWN). Il faut inverser le signe comme le fait le code BE.
        if "ENERGY_PRICE_PAYMENT_DIRECTION" in sub.columns:
            mask_g2p = sub["ENERGY_PRICE_PAYMENT_DIRECTION"] == "GRID_TO_PROVIDER"
            sub.loc[mask_g2p, "ENERGY_PRICE_[EUR/MWh]"] = \
                sub.loc[mask_g2p, "ENERGY_PRICE_[EUR/MWh]"].astype(float) * -1
        offers.append(pd.DataFrame({
            "timestamp": sub["timestamp"],
            "direction": direction,
            "price":     sub["ENERGY_PRICE_[EUR/MWh]"].astype(float),
            "quantity":  sub["ALLOCATED_CAPACITY_[MW]"].astype(float),
            "country":   "DE",
            "reserve":   product,
        }).dropna())
    return pd.concat(offers, ignore_index=True) if offers else pd.DataFrame()

def dl_tennet(date_str):
    p = DATA_DIR / f"tennet_{date_str}.csv"
    if file_ok(p):
        print(f"    tennet NL     : cache")
        df = pd.read_csv(p, parse_dates=["timestamp"])
        df_a = df[df["reserve"] == "aFRR"].reset_index(drop=True)
        df_m = df[df["reserve"] == "mFRR"].reset_index(drop=True)
        return df_a, df_m

    base = datetime.strptime(date_str + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    r = retry(lambda: requests.get(
        "https://api.tennet.eu/publications/v1/merit-order-list",
        headers={"apikey": API_KEY_TENNET, "Accept": "text/csv"},
        params={"date_from": base.strftime("%d-%m-%Y %H:%M:%S"),
                "date_to":   (base + timedelta(days=1)).strftime("%d-%m-%Y %H:%M:%S")},
        timeout=60))
    if r.status_code != 200 or not r.content.strip():
        return pd.DataFrame(), pd.DataFrame()
    mo = pd.read_csv(StringIO(r.content.decode("utf-8")))
    time_col = next((c for c in mo.columns if "Start" in c and ("Local" in c or "Loc" in c)), None)
    if not time_col:
        return pd.DataFrame(), pd.DataFrame()
    mo["timestamp"] = pd.to_datetime(mo[time_col], errors="coerce").dt.tz_localize(None)
    cap_col = next((c for c in mo.columns if "Capacity" in c and "Threshold" in c), None)
    if cap_col is None:
        print(f"    tennet NL     : colonne Capacity Threshold absente ({list(mo.columns)[:5]})")
        return pd.DataFrame(), pd.DataFrame()
    mo = mo.dropna(subset=["timestamp", cap_col])
    mo = mo.rename(columns={cap_col: "Capacity Threshold"})
    mo = (mo[mo["timestamp"].dt.date == base.date()]
          .sort_values(["timestamp", "Capacity Threshold"]))
    mo["volume"] = (mo.groupby("timestamp")["Capacity Threshold"]
                    .diff().fillna(mo["Capacity Threshold"]))
    mo = mo[mo["volume"] > 0]
    afrr_all, mfrr_all = [], []
    for ts in mo["timestamp"].unique():
        mo_ts = mo[mo["timestamp"] == ts]
        for price_col, direction in [("Price Up", "UP"), ("Price Down", "DOWN")]:
            if price_col not in mo_ts.columns:
                continue
            sub = mo_ts[[price_col, "volume"]].dropna().copy()
            sub.columns = ["price", "quantity"]
            sub["timestamp"] = ts
            sub = sub.sort_values("price", ascending=(direction == "UP"))
            sub["cumul"] = sub["quantity"].cumsum()
            for df_t, lst, reserve in [
                (sub[sub["cumul"] <= 110].copy(), afrr_all, "aFRR"),
                (sub[sub["cumul"] > 110].copy(),  mfrr_all, "mFRR"),
            ]:
                if not df_t.empty:
                    df_t = df_t[["timestamp","price","quantity"]].copy()
                    df_t["direction"] = direction
                    df_t["country"]   = "NL"
                    df_t["reserve"]   = reserve
                    lst.append(df_t)
    df_a = pd.concat(afrr_all, ignore_index=True) if afrr_all else pd.DataFrame()
    df_m = pd.concat(mfrr_all, ignore_index=True) if mfrr_all else pd.DataFrame()
    df_all = pd.concat([df_a, df_m], ignore_index=True)
    if not df_all.empty:
        df_all.to_csv(p, index=False)
    return df_a, df_m

# =============================================================================
# METRIQUES EU
# =============================================================================

def compute_ratios(df_merit, direction):
    if df_merit is None or df_merit.empty:
        return pd.DataFrame()
    df_dir = df_merit[df_merit["direction"] == direction].copy()
    if df_dir.empty:
        return pd.DataFrame()
    out = []
    for ts in sorted(df_dir["timestamp"].unique()):
        offers = df_dir[df_dir["timestamp"] == ts]
        total  = offers["quantity"].sum()
        if total <= 0:
            continue
        if direction == "DOWN":
            out.append({
                "timestamp":           ts,
                "ratio_negative":      100*offers[offers["price"]<0]["quantity"].sum()/total,
                "ratio_very_negative": 100*offers[offers["price"]<-50]["quantity"].sum()/total,
            })
        else:
            out.append({
                "timestamp": ts,
                "ratio_60":  100*offers[offers["price"]>60]["quantity"].sum()/total,
                "ratio_80":  100*offers[offers["price"]>80]["quantity"].sum()/total,
            })
    return pd.DataFrame(out)

def build_merit_metrics(date_str):
    p = DATA_DIR / f"merit_{date_str}.csv"
    if file_ok(p):
        print(f"    merit order EU : cache")
        df = pd.read_csv(p, parse_dates=["timestamp"])
        for col in ["afrr_ratio_negative","afrr_ratio_very_negative",
                    "mfrr_ratio_negative","mfrr_ratio_very_negative"]:
            if col not in df.columns:
                df[col] = 0.0
        return df

    dfs_afrr, dfs_mfrr = [], []
    for prod, direction in [("aFRR","incremental"),("aFRR","decremental"),
                             ("mFRR","incremental"),("mFRR","decremental")]:
        df = dl_elia(date_str, prod, direction)
        if not df.empty:
            (dfs_afrr if prod == "aFRR" else dfs_mfrr).append(df)
    for prod, lst in [("aFRR", dfs_afrr), ("mFRR", dfs_mfrr)]:
        df = dl_regelleistung(date_str, prod)
        if not df.empty:
            lst.append(df)
    df_a_nl, df_m_nl = dl_tennet(date_str)
    if not df_a_nl.empty:
        dfs_afrr.append(df_a_nl)
    if not df_m_nl.empty:
        dfs_mfrr.append(df_m_nl)

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
    for col in ["afrr_ratio_negative","afrr_ratio_very_negative",
                "mfrr_ratio_negative","mfrr_ratio_very_negative"]:
        if col not in full.columns:
            full[col] = 0.0
        else:
            full[col] = full[col].fillna(0.0)
    full.to_csv(p, index=False)
    return full

# =============================================================================
# STRATEGIES FR V2
# =============================================================================

def rev_imb_fr(ecart, ppos, pneg):
    rev = np.zeros(len(ecart), dtype=float)
    m = (ecart > 0) & (pneg < 0);  rev[m] =  ecart[m] * np.abs(pneg[m]) / 4
    m = (ecart < 0) & (ppos > 0);  rev[m] =  np.abs(ecart[m]) * ppos[m] / 4
    m = (ecart > 0) & (ppos > 0);  rev[m] = -ecart[m] * ppos[m] / 4
    m = (ecart < 0) & (pneg < 0);  rev[m] = -np.abs(ecart[m]) * np.abs(pneg[m]) / 4
    return rev

def compute_strategies_v2(df):
    df   = df.copy()
    prod = df["production_mw"].fillna(0)
    fp   = df["forecast_parc"].fillna(0)
    da   = df["price_eur_mwh"].fillna(0)
    vol  = df["forecast_volume"].abs().fillna(0)
    fdir = df["forecast_direction"].fillna("UP")
    afrr_neg  = df["afrr_ratio_negative"].fillna(0)
    mfrr_neg  = df["mfrr_ratio_negative"].fillna(0)
    afrr_vneg = df["afrr_ratio_very_negative"].fillna(0)
    ppos = df["prix_positif"].fillna(0).values
    pneg = df["prix_negatif"].fillna(0).values

    # Gate nuit
    solar_on = prod > 0.01

    # Signal DOWN : risque prix négatifs (curtailment)
    def sig_dn(thr):
        return (
            solar_on
            & (fdir == "DOWN")
            & (
                ((vol > thr) & (mfrr_neg > 85) & (afrr_neg > 70))
                | ((vol > thr) & (afrr_vneg > 75))
            )
        )

    sd150 = sig_dn(150)
    df["signal_down_150"] = sd150

    # S1 Baseline — nomination 100%, pas d'ajustement
    df["s1_nomination"]  = fp.values
    df["s1_production"]  = prod.values
    df["s1_ecart"]       = (fp - prod).values
    df["s1_revenue_imb"] = rev_imb_fr(df["s1_ecart"].values, ppos, pneg)
    df["s1_revenue_da"]  = (fp * da / 4).values
    df["s1_total"]       = df["s1_revenue_da"] + df["s1_revenue_imb"]

    # S2 Active — nomination DA-adapt + curtail DOWN > 150 MW (même logique production que S3)
    curtail_s2 = sd150 & solar_on
    nom_s2     = np.where(da < 0, 0.0, np.where(da < SEUIL_DA_MID, fp * 0.5, fp))
    df["s2_curtail"]     = curtail_s2
    df["s2_nomination"]  = nom_s2
    df["s2_production"]  = np.where(curtail_s2, 0.0, prod)
    df["s2_ecart"]       = nom_s2 - df["s2_production"]
    df["s2_revenue_imb"] = rev_imb_fr(df["s2_ecart"].values, ppos, pneg)
    df["s2_revenue_da"]  = nom_s2 * da / 4
    df["s2_total"]       = df["s2_revenue_da"] + df["s2_revenue_imb"]

    # S3 Active — nomination 50% fixe + curtail DOWN > 150 MW
    curtail_s3 = sd150 & solar_on
    df["s3_curtail"]     = curtail_s3
    nom_s3               = (fp * 0.5).values
    df["s3_nomination"]  = nom_s3
    df["s3_production"]  = np.where(curtail_s3, 0.0, prod)
    df["s3_ecart"]       = nom_s3 - df["s3_production"]
    df["s3_revenue_imb"] = rev_imb_fr(df["s3_ecart"].values, ppos, pneg)
    df["s3_revenue_da"]  = nom_s3 * da / 4
    df["s3_total"]       = df["s3_revenue_da"] + df["s3_revenue_imb"]

    return df

# =============================================================================
# FORECAST PARC
# =============================================================================

def build_forecast_parc(prod, df_solar):
    if df_solar.empty or "forecast_mw" not in df_solar.columns:
        return prod
    fc_nat = df_solar["forecast_mw"].fillna(0)
    ac_nat = df_solar["actual_mw"].replace(0, np.nan).fillna(1)
    ratio  = (fc_nat / ac_nat).replace([np.inf, -np.inf], np.nan).fillna(1)
    raw    = prod * ratio
    lo     = prod * (1 - DEVIATION_MAX)
    hi     = prod * (1 + DEVIATION_MAX)
    return np.clip(raw, lo, hi).clip(upper=1.0)

# =============================================================================
# FORECAST MODELE FR
# =============================================================================

def forecast_fr(date_str, df_hist):
    if not MODEL_PATH.exists():
        return None
    if df_hist is None or df_hist.empty:
        return None
    try:
        model = joblib.load(MODEL_PATH)
        rows  = []
        for ts in qh_range(date_str):
            hist = df_hist[df_hist["timestamp"] < ts].tail(MAX_LAG)
            if len(hist) < MAX_LAG:
                continue
            exp    = pd.date_range(end=ts - pd.Timedelta(minutes=15),
                                   periods=MAX_LAG, freq="15min")
            series = (hist.set_index("timestamp")["volume"]
                      .reindex(exp).interpolate(limit=2).ffill().bfill())
            if series.isna().any():
                continue
            slot  = ts.hour * 4 + ts.minute // 15
            feats = list(series.values[::-1]) + [
                np.sin(2*np.pi*slot/96),  np.cos(2*np.pi*slot/96),
                np.sin(2*np.pi*ts.dayofweek/7), np.cos(2*np.pi*ts.dayofweek/7),
                np.sin(2*np.pi*ts.month/12),    np.cos(2*np.pi*ts.month/12),
            ]
            pred = float(model.predict(np.array(feats).reshape(1, -1))[0])
            rows.append({"timestamp": ts, "forecast_volume": pred,
                         "forecast_direction": "DOWN" if pred > 0 else "UP"})
        return pd.DataFrame(rows) if rows else None
    except Exception as e:
        print(f"    forecast_fr   : WARN {e}")
        return None

# =============================================================================
# SIMULATION D'UN JOUR
# =============================================================================

def simulate_day(date_str, token, df_pvgis, df_hist):
    print(f"\n  {date_str}")
    df_imb    = dl_imbalance_fr(date_str, token)
    df_da     = dl_da(date_str)
    df_solar  = dl_solar(date_str)
    df_pv_day = pvgis_day(date_str, df_pvgis)

    if df_imb is None or df_da is None:
        return None
    if df_imb.empty or df_da.empty or df_pv_day.empty:
        print(f"    -> SKIP (donnees manquantes)")
        return None

    print(f"    merit order EU : telechargement...")
    metrics = build_merit_metrics(date_str)
    print(f"    merit order EU : {len(metrics)} QH metriques")

    df_fc = forecast_fr(date_str, df_hist)
    if df_fc is not None:
        print(f"    forecast_fr   : {len(df_fc)} QH")
    else:
        print(f"    forecast_fr   : modele absent -> volume=0")

    full = pd.DataFrame({"timestamp": qh_range(date_str)})
    full = full.merge(df_imb[["timestamp","volume","prix_positif","prix_negatif"]],
                      on="timestamp", how="left")
    full = full.merge(df_da[["timestamp","price_eur_mwh"]],
                      on="timestamp", how="left")
    full = full.merge(df_pv_day[["timestamp","production_mw"]],
                      on="timestamp", how="left")
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

    full["production_mw"]       = full["production_mw"].fillna(0)
    full["price_eur_mwh"]       = full["price_eur_mwh"].ffill().fillna(0)
    full["prix_positif"]        = full["prix_positif"].ffill().bfill().fillna(0)
    full["prix_negatif"]        = full["prix_negatif"].ffill().bfill().fillna(0)
    full["forecast_volume"]     = full["forecast_volume"].fillna(0)
    full["forecast_direction"]  = full["forecast_direction"].fillna("UP")
    full["forecast_mw"]         = full["forecast_mw"].fillna(0)
    full["actual_mw"]           = full["actual_mw"].fillna(1).replace(0, 1)

    full["forecast_parc"] = build_forecast_parc(
        full["production_mw"],
        full[["timestamp","forecast_mw","actual_mw"]])

    full["date"] = date_str
    full = compute_strategies_v2(full)
    return full

# =============================================================================
# AFFICHAGE
# =============================================================================

def cas_imb(ecart, ppos, pneg):
    if ecart > 0 and pneg < 0:
        return f"SHORT + prix_neg<0 -> GAIN  (+{abs(ecart)*abs(pneg)/4:.4f})"
    if ecart < 0 and ppos > 0:
        return f"LONG  + prix_pos>0 -> GAIN  (+{abs(ecart)*ppos/4:.4f})"
    if ecart > 0 and ppos > 0:
        return f"SHORT + prix_pos>0 -> PENA  (-{ecart*ppos/4:.4f})"
    if ecart < 0 and pneg < 0:
        return f"LONG  + prix_neg<0 -> PENA  (-{abs(ecart)*abs(pneg)/4:.4f})"
    return "neutre (ecart=0 ou prix=0)"

def print_qh(row, label=""):
    ts   = row["timestamp"]
    da   = row["price_eur_mwh"]
    ppos = row["prix_positif"]
    pneg = row["prix_negatif"]
    prod = row["production_mw"]
    fp   = row["forecast_parc"]
    fvol = row["forecast_volume"]
    fdir = row["forecast_direction"]
    afrr = row["afrr_ratio_negative"]
    mfrr = row["mfrr_ratio_negative"]
    d150 = bool(row["signal_down_150"])
    d100 = bool(row["signal_down_100"])

    note_da = ("  [DA<0  -> nom=0%  -> LONG max]"
               if da < 0 else
               "  [DA<30 -> nom=50% -> LONG partiel]"
               if da < SEUIL_DA_MID else "")

    print(f"\n{SEP}")
    print(f"  {label} -- {ts}")
    print(SEP)
    print(f"  DA price      : {da:+8.2f} EUR/MWh{note_da}")
    print(f"  prix_positif  : {ppos:+8.2f} EUR/MWh")
    print(f"  prix_negatif  : {pneg:+8.2f} EUR/MWh")
    print(f"  production    : {prod:.4f} MW/GWp")
    print(f"  forecast_parc : {fp:.4f} MW/GWp")
    print(f"  forecast imb  : {fdir}  {fvol:.0f} MW")
    print(f"  aFRR%neg      : {afrr:.1f}%  |  mFRR%neg : {mfrr:.1f}%")
    print(f"  signal_DN_150 : {'OUI' if d150 else 'NON'}")
    print(f"  signal_DN_100 : {'OUI' if d100 else 'NON'}")
    print()
    for s, name in [("s1","S1 Baseline      "),
                    ("s2","S2 DA-adapt 150MW"),
                    ("s3","S3 50%fix  150MW "),
                    ("s4","S4 DA-adapt 100MW")]:
        nom  = row[f"{s}_nomination"]
        prds = row[f"{s}_production"]
        ec   = row[f"{s}_ecart"]
        rda  = row[f"{s}_revenue_da"]
        rib  = row[f"{s}_revenue_imb"]
        rt   = row[f"{s}_total"]
        curt = bool(row.get(f"{s}_curtail", False)) if s != "s1" else False
        print(f"  {name}  "
              f"nom={nom:.4f}  "
              f"prod={prds:.4f}{'[CURTAIL]' if curt else '         '}  "
              f"ecart={ec:+.4f}  "
              f"DA={rda:+.4f}  imb={rib:+.4f}  "
              f"TOT={rt:+.4f}  <- {cas_imb(ec, ppos, pneg)}")
    print()

def print_day_summary(df_day):
    date = df_day["date"].iloc[0]
    n_sol = (df_day["production_mw"] > 0.01).sum()
    print(f"\n  Synthese {date}  ({n_sol} QH solaires) :")
    for s, name in [("s1","S1"),("s2","S2"),("s3","S3"),("s4","S4")]:
        tot = df_day[f"{s}_total"].sum()
        da  = df_day[f"{s}_revenue_da"].sum()
        imb = df_day[f"{s}_revenue_imb"].sum()
        nc  = int(df_day[f"{s}_curtail"].sum()) if s != "s1" else 0
        curt_str = f"  curt={nc}" if s != "s1" else ""
        print(f"    {name}  total={tot:+7.2f}  DA={da:+7.2f}  imb={imb:+7.2f}{curt_str}")

def print_global(results):
    print(f"\n{SEP}")
    print("  RESUME GLOBAL -- 6 jours cibles")
    print(SEP)
    df_all = pd.concat(results, ignore_index=True)
    df_all["month"] = df_all["timestamp"].dt.to_period("M").astype(str)

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
        tot = df_all[f"{s}_total"].sum()
        da  = df_all[f"{s}_revenue_da"].sum()
        imb = df_all[f"{s}_revenue_imb"].sum()
        nc  = int(df_all[f"{s}_curtail"].sum()) if s != "s1" else 0
        print(f"  {name}  total={tot:+8.2f}  DA={da:+8.2f}  imb={imb:+8.2f}  curt={nc}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print(SEP)
    print("  TEST LOGIQUE FR -- S1/S2/S3/S4 V2  (6 jours cibles)")
    print("  Corrections : seuils 150/100MW, gate nuit, gate nomination>0, signal_UP supprime")
    print(SEP)

    print("\n  Auth RTE...")
    token = get_rte_token()
    print("  OK")

    print("\n  PVGIS...")
    df_pvgis = dl_pvgis()

    print("\n  Historique imbalance FR existant...")
    parts = []
    for f in sorted(DATA_DIR.glob("imbalance_fr_2026-*.csv")):
        try:
            parts.append(pd.read_csv(f, parse_dates=["timestamp"]))
        except Exception:
            pass
    df_hist = (pd.concat(parts, ignore_index=True)
               .sort_values("timestamp").drop_duplicates("timestamp")
               .reset_index(drop=True)) if parts else pd.DataFrame()
    print(f"  {len(df_hist)} QH historiques")

    results = []
    for i, date_str in enumerate(TEST_DAYS):
        # Refresh token toutes les 20 jours ou si simulate_day retourne None
        if i > 0 and i % 20 == 0:
            try:
                token = get_rte_token()
                print(f"  [token refresh jour {i}]")
            except Exception as e:
                print(f"  [token refresh ECHEC: {e}]")
        df_day = simulate_day(date_str, token, df_pvgis, df_hist)
        if df_day is None:
            # Token probablement expire - refresh et retry
            try:
                token = get_rte_token()
                print(f"  [token refresh apres None - retry {date_str}]")
                df_day = simulate_day(date_str, token, df_pvgis, df_hist)
            except Exception as e:
                print(f"  [retry ECHEC: {e}]")
        if df_day is None:
            continue

        # Met a jour historique pour forecast des jours suivants
        if not df_hist.empty:
            new_rows = df_day[["timestamp","volume"]].dropna()
            df_hist = (pd.concat([df_hist, new_rows], ignore_index=True)
                       .sort_values("timestamp").drop_duplicates("timestamp"))
        else:
            df_hist = df_day[["timestamp","volume"]].dropna().copy()

        results.append(df_day)
        print_day_summary(df_day)

        # 3 QH representatifs par jour
        df_sol = df_day[df_day["production_mw"] > 0.05]
        if df_sol.empty:
            continue

        picks = {}
        # prix DA le plus eleve
        picks["DA_MAX"] = df_sol.loc[df_sol["price_eur_mwh"].idxmax()]
        # prix DA le plus bas
        picks["DA_MIN"] = df_sol.loc[df_sol["price_eur_mwh"].idxmin()]
        # signal_DOWN si present
        sd = df_sol[df_sol["signal_down_100"] == True]
        if not sd.empty:
            picks["SIGNAL_DOWN"] = sd.iloc[0]
        # milieu de journee
        mid = df_sol.iloc[len(df_sol)//2]
        picks["MIDDAY"] = mid

        for label, row in picks.items():
            print_qh(row, label=f"{date_str} [{label}]")

    if results:
        print_global(results)

    print(f"\n{SEP}")
    print("  FIN DU TEST")
    print("  Si logique correcte -> integrer compute_strategies_v2() dans pipeline principal")
    print(SEP)


if __name__ == "__main__":
    main()