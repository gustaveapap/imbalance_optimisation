#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BACKFILL 2026 - telecharge toutes les donnees manquantes pour solar_be.

Donnees manquantes identifiees (15 avril 2026) :
  si_1min  : 2026-04-12, 13, 14  (ODS133, 1min)
  isp      : 2026-04-12, 13, 14  (ODS134, QH)
  aFRR/mFRR/DA/solar : 2026-04-13, 14, 15

Usage : python backfill_2026.py
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import time
import warnings
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

# --- Paths ---
BASE_DIR = Path("C:/Users/gusta/imbalance_optimisation")
DATA_DIR = BASE_DIR / "data/raw/solar_be"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Config ---
BRUSSELS      = "Europe/Brussels"
TZ_BRUSSELS   = ZoneInfo(BRUSSELS)
ELIA_OPENDATA = "https://external-elia.opendatasoft.com/api/records/1.0/search/"
ENTSOE_API    = "https://web-api.tp.entsoe.eu/api"
ENTSOE_SOLAR  = "https://transparency.entsoe.eu/generation/forecast/windAndSolar/solar/load"
TENNET_URL    = "https://api.tennet.eu/publications/v1/merit-order-list"
TENNET_KEY    = "18fe140e-12a2-446e-ab89-455d33709fac"
DE_API_URL    = "https://www.regelleistung.net/apps/cpp-publisher/api/v2/tenders/results/anonymous"
ENTSOE_TOKEN  = "dcc0c513-dab6-4add-82be-7b693f2b7fab"
BE_DOMAIN     = "10YBE----------2"
BE_AREA       = "BZN|10YBE----------2"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# --- SI 1min (ODS133, resolution 1 min, historique) ---

def download_si_1min(date_str):
    out = DATA_DIR / f"si_1min_{date_str}.csv"
    if out.exists() and out.stat().st_size > 10000:
        log(f"  si_1min {date_str} - deja present ({out.stat().st_size:,} B), skip")
        return

    log(f"  si_1min {date_str} - telechargement ODS133...")
    frames, start = [], 0
    MAX_START = 10000

    while start < MAX_START:
        params = {
            "dataset":           "ods133",
            "rows":              1000,
            "start":             start,
            "refine.date_start": date_str,
            "sort":              "datetime",
        }
        for attempt in range(3):
            try:
                r = requests.get(ELIA_OPENDATA, params=params, timeout=60)
                if r.status_code == 429:
                    time.sleep(5)
                    continue
                if r.status_code == 400 and "10000" in r.text:
                    start = MAX_START
                    break
                r.raise_for_status()
                records = r.json().get("records", [])
                if not records:
                    start = MAX_START
                    break
                rows = []
                for rec in records:
                    f = rec.get("fields", {})
                    dt_raw = f.get("datetime", "")
                    si_val = f.get("systemimbalance")
                    if dt_raw and si_val is not None:
                        try:
                            dt = pd.to_datetime(dt_raw, utc=True).tz_convert(BRUSSELS).tz_localize(None)
                            rows.append({"datetime": dt, "actual_system_imbalance": float(si_val)})
                        except Exception:
                            pass
                if rows:
                    frames.append(pd.DataFrame(rows))
                if len(records) < 1000:
                    start = MAX_START
                    break
                start += 1000
                time.sleep(0.5)
                break
            except Exception as e:
                log(f"    WARNING attempt {attempt+1}: {e}")
                time.sleep(3)

    if frames:
        df = (pd.concat(frames, ignore_index=True)
                .drop_duplicates("datetime")
                .sort_values("datetime")
                .reset_index(drop=True))
        target = pd.Timestamp(date_str).date()
        df = df[df["datetime"].dt.date == target]
        df.to_csv(out, index=False)
        log(f"  si_1min {date_str} - {len(df)} lignes sauvegardees")
    else:
        log(f"  si_1min {date_str} - AUCUNE donnee obtenue")


# --- ISP (ODS134, resolution QH, historique) ---

def download_isp(date_str):
    out = DATA_DIR / f"isp_{date_str}.csv"
    if out.exists() and out.stat().st_size > 500:
        log(f"  isp {date_str} - deja present, skip")
        return

    log(f"  isp {date_str} - telechargement ODS134...")
    frames, start = [], 0

    while start < 10000:
        params = {
            "dataset":           "ods134",
            "rows":              1000,
            "start":             start,
            "refine.date_start": date_str,
            "sort":              "datetime",
        }
        for attempt in range(3):
            try:
                r = requests.get(ELIA_OPENDATA, params=params, timeout=60)
                if r.status_code == 429:
                    time.sleep(5)
                    continue
                if r.status_code == 400 and "10000" in r.text:
                    start = 10000
                    break
                r.raise_for_status()
                records = r.json().get("records", [])
                if not records:
                    start = 10000
                    break
                rows = []
                for rec in records:
                    f = rec.get("fields", {})
                    dt_raw = f.get("datetime", "")
                    isp_val = f.get("imbalanceprice")
                    if dt_raw and isp_val is not None:
                        try:
                            dt = pd.to_datetime(dt_raw, utc=True).tz_convert(BRUSSELS).tz_localize(None)
                            rows.append({"datetime": dt, "imbalanceprice": float(isp_val)})
                        except Exception:
                            pass
                if rows:
                    frames.append(pd.DataFrame(rows))
                if len(records) < 1000:
                    start = 10000
                    break
                start += 1000
                time.sleep(0.4)
                break
            except Exception as e:
                log(f"    WARNING attempt {attempt+1}: {e}")
                time.sleep(3)

    if frames:
        df = (pd.concat(frames, ignore_index=True)
                .drop_duplicates("datetime")
                .sort_values("datetime")
                .reset_index(drop=True))
        target = pd.Timestamp(date_str).date()
        df = df[df["datetime"].dt.date == target]
        df.to_csv(out, index=False)
        log(f"  isp {date_str} - {len(df)} QH sauvegardes")
    else:
        log(f"  isp {date_str} - AUCUNE donnee")


# --- aFRR / mFRR BE (ODS163/164) ---

def download_be_bids(date_str, product, out_inc, out_dec):
    DATASETS = [("incremental", "ods163", out_inc),
                ("decremental", "ods164", out_dec)]
    MAX_START = 10000

    for direction, dataset, out_path in DATASETS:
        if out_path.exists() and out_path.stat().st_size > 500:
            log(f"  BE {product} {direction} {date_str} - skip")
            continue
        log(f"  BE {product} {direction} {date_str} - telechargement {dataset}...")
        frames, start = [], 0
        while start < MAX_START:
            params = {
                "dataset":                 dataset,
                "rows":                    1000,
                "start":                   start,
                "refine.date_start":       date_str,
                "refine.balancingproduct": product,
                "sort":                    "datetime",
            }
            try:
                r = requests.get(ELIA_OPENDATA, params=params, timeout=60)
                if r.status_code == 429:
                    time.sleep(3)
                    r = requests.get(ELIA_OPENDATA, params=params, timeout=60)
                if r.status_code == 400 and "10000" in r.text:
                    break
                r.raise_for_status()
                records = r.json().get("records", [])
                if not records:
                    break
                df_page = pd.DataFrame([rec.get("fields", {}) for rec in records])
                if "datetime" in df_page.columns:
                    df_page["datetime"] = pd.to_datetime(df_page["datetime"], errors="coerce", utc=True)
                frames.append(df_page)
                if len(records) < 1000:
                    break
                start += 1000
                time.sleep(0.6)
            except Exception as e:
                log(f"    WARNING BE {product} {direction}: {e}")
                break
        df_out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        df_out.to_csv(out_path, index=False)
        log(f"  BE {product} {direction} {date_str} - {len(df_out)} lignes")


# --- DE aFRR / mFRR (regelleistung.net) ---

def download_de_xlsx(date_str, product, out_path):
    if out_path.exists() and out_path.stat().st_size > 1000:
        log(f"  DE {product} {date_str} - skip")
        return
    log(f"  DE {product} {date_str} - telechargement xlsx...")
    for product_type in ([product, "MRL"] if product == "mFRR" else [product]):
        try:
            r = requests.get(DE_API_URL, params={
                "productType": product_type, "market": "ENERGY",
                "exportFormat": "xlsx", "deliveryDate": date_str,
            }, timeout=60)
            if r.status_code == 200 and r.content[:4] == b"PK\x03\x04":
                out_path.write_bytes(r.content)
                log(f"  DE {product} {date_str} - {len(r.content):,} B")
                return
            log(f"  DE {product} {date_str} productType={product_type}: HTTP {r.status_code}")
        except Exception as e:
            log(f"    WARNING DE {product}: {e}")
    log(f"  DE {product} {date_str} - ECHEC")


# --- NL aFRR (TenNET) ---

def download_nl_csv(date_str, out_path):
    if out_path.exists() and out_path.stat().st_size > 100:
        log(f"  NL {date_str} - skip")
        return
    log(f"  NL {date_str} - telechargement TenNET heure par heure...")
    base     = datetime.strptime(date_str + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    now_lo   = datetime.now(TZ_BRUSSELS)
    max_hour = now_lo.hour if date_str == now_lo.strftime("%Y-%m-%d") else 23
    headers  = {"apikey": TENNET_KEY, "Accept": "text/csv"}
    all_dfs  = []

    for hour in range(max_hour + 1):
        from_t = (base + timedelta(hours=hour)).strftime("%d-%m-%Y %H:%M:%S")
        to_t   = (base + timedelta(hours=hour + 1)).strftime("%d-%m-%Y %H:%M:%S")
        for attempt in range(3):
            try:
                r = requests.get(TENNET_URL, headers=headers,
                                 params={"date_from": from_t, "date_to": to_t}, timeout=30)
                if r.status_code == 200 and r.content.strip():
                    all_dfs.append(pd.read_csv(StringIO(r.content.decode("utf-8"))))
                    break
                elif r.status_code == 429:
                    time.sleep(6)
                else:
                    break
            except Exception as e:
                log(f"    WARNING NL {from_t}: {e}")
            time.sleep(6)
        time.sleep(0.5)

    if all_dfs:
        pd.concat(all_dfs, ignore_index=True).to_csv(out_path, index=False)
        log(f"  NL {date_str} - {sum(len(d) for d in all_dfs)} lignes")
    else:
        pd.DataFrame().to_csv(out_path, index=False)
        log(f"  NL {date_str} - 0 lignes")


# --- DA price (ENTSO-E A44) ---

def download_da(date_str):
    cache = DATA_DIR / f"da_{date_str}.csv"
    if cache.exists() and cache.stat().st_size > 100:
        log(f"  DA {date_str} - skip")
        return
    log(f"  DA {date_str} - telechargement ENTSO-E A44...")
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
                log(f"    DA HTTP {r.status_code}, retry {attempt+1}/5")
                time.sleep(5 * (attempt + 1))
                continue
            root = ET.fromstring(r.content)
            rows = []
            bx = ZoneInfo(BRUSSELS)
            for period in root.findall(".//{*}Period"):
                s_el = period.find(".//{*}start")
                r_el = period.find(".//{*}resolution")
                if s_el is None:
                    continue
                s_dt = pd.to_datetime(s_el.text, utc=True)
                res  = r_el.text if r_el is not None else "PT60M"
                for p in period.findall(".//{*}Point"):
                    pos = int(p.find(".//{*}position").text)
                    px  = float(p.find(".//{*}price.amount").text)
                    off = (timedelta(minutes=(pos - 1) * 15) if res == "PT15M"
                           else timedelta(hours=(pos - 1)))
                    ts  = (s_dt + off).astimezone(bx).replace(tzinfo=None)
                    rows.append({"timestamp": pd.Timestamp(ts), "price_eur_mwh": px})
            if not rows:
                log(f"  DA {date_str} - 0 points dans la reponse XML")
                return
            df = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
            df.to_csv(cache, index=False)
            log(f"  DA {date_str} - {len(df)} QH sauvegardes")
            return
        except Exception as e:
            log(f"    WARNING DA attempt {attempt+1}: {e}")
            time.sleep(5 * (attempt + 1))
    log(f"  DA {date_str} - ECHEC apres 5 tentatives")


# --- Solar ENTSO-E ---

def download_solar(date_str):
    cache = DATA_DIR / f"solar_{date_str}.csv"
    if cache.exists() and cache.stat().st_size > 100:
        log(f"  solar {date_str} - skip")
        return
    log(f"  solar {date_str} - telechargement ENTSO-E...")
    date_from = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    date_to   = date_from + timedelta(days=1)
    bx = ZoneInfo(BRUSSELS)
    for attempt in range(3):
        try:
            r = requests.post(
                ENTSOE_SOLAR,
                json={
                    "dateTimeRange": {
                        "from": date_from.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                        "to":   date_to.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    },
                    "areaList": [BE_AREA], "timeZone": "CET",
                    "sorterList": [], "filterMap": {},
                },
                headers={"accept": "application/json",
                         "content-type": "application/json; charset=utf-8"},
                timeout=60)
            if r.status_code != 200:
                log(f"    solar HTTP {r.status_code}, retry {attempt+1}")
                time.sleep(3)
                continue
            rows = []
            for inst in r.json().get("instanceList", []):
                if inst.get("businessDimensionMap", {}).get("PRODUCTION_TYPE") != "B16":
                    continue
                for period in inst.get("curveData", {}).get("periodList", []):
                    st = period.get("timeInterval", {}).get("from")
                    if not st:
                        continue
                    res  = period.get("resolution")
                    s_dt = pd.to_datetime(st, utc=True)
                    for pos_str, vals in period.get("pointMap", {}).items():
                        pos = int(pos_str)
                        off = (timedelta(minutes=pos * 15) if res == "PT15M"
                               else timedelta(hours=pos))
                        ts = (s_dt + off).astimezone(bx).replace(tzinfo=None)
                        v  = vals if isinstance(vals, list) else []
                        rows.append({
                            "timestamp":   pd.Timestamp(ts),
                            "forecast_mw": float(v[0]) if len(v) > 0 else np.nan,
                            "actual_mw":   float(v[3]) if len(v) > 3 else np.nan,
                        })
            if rows:
                df = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
                df.to_csv(cache, index=False)
                log(f"  solar {date_str} - {len(df)} QH sauvegardes")
                return
            log(f"  solar {date_str} - 0 points dans la reponse")
            return
        except Exception as e:
            log(f"    WARNING solar attempt {attempt+1}: {e}")
            time.sleep(3)
    log(f"  solar {date_str} - ECHEC")


# --- MAIN ---

if __name__ == "__main__":
    today = datetime.now(TZ_BRUSSELS).strftime("%Y-%m-%d")
    log(f"=== BACKFILL 2026 - demarrage (aujourd'hui={today}) ===")

    log("\n--- SI 1min (ODS133) ---")
    for d in ["2026-04-12", "2026-04-13", "2026-04-14"]:
        download_si_1min(d)

    log("\n--- ISP (ODS134) ---")
    for d in ["2026-04-12", "2026-04-13", "2026-04-14"]:
        download_isp(d)

    log("\n--- aFRR BE (ODS163/164) ---")
    for d in ["2026-04-13", "2026-04-14", "2026-04-15"]:
        download_be_bids(d, "aFRR",
                         DATA_DIR / f"afrr_be_inc_{d}.csv",
                         DATA_DIR / f"afrr_be_dec_{d}.csv")

    log("\n--- DE aFRR (regelleistung.net) ---")
    for d in ["2026-04-13", "2026-04-14", "2026-04-15"]:
        download_de_xlsx(d, "aFRR", DATA_DIR / f"afrr_de_{d}.xlsx")

    log("\n--- NL aFRR (TenNET) ---")
    for d in ["2026-04-13", "2026-04-14", "2026-04-15"]:
        download_nl_csv(d, DATA_DIR / f"afrr_nl_{d}.csv")

    log("\n--- mFRR BE (ODS163/164) ---")
    for d in ["2026-04-13", "2026-04-14", "2026-04-15"]:
        download_be_bids(d, "mFRR",
                         DATA_DIR / f"mfrr_be_inc_{d}.csv",
                         DATA_DIR / f"mfrr_be_dec_{d}.csv")

    log("\n--- DE mFRR (regelleistung.net) ---")
    for d in ["2026-04-13", "2026-04-14", "2026-04-15"]:
        download_de_xlsx(d, "mFRR", DATA_DIR / f"mfrr_de_{d}.xlsx")

    log("\n--- DA price (ENTSO-E A44) ---")
    for d in ["2026-04-13", "2026-04-14", "2026-04-15"]:
        download_da(d)

    log("\n--- Solar ENTSO-E ---")
    for d in ["2026-04-13", "2026-04-14", "2026-04-15"]:
        download_solar(d)

    # Resume
    log("\n=== BACKFILL TERMINE - resume ===")
    prefixes_check = {
        "si_1min":     (["2026-04-12", "2026-04-13", "2026-04-14"], "csv"),
        "isp":         (["2026-04-12", "2026-04-13", "2026-04-14"], "csv"),
        "afrr_be_inc": (["2026-04-13", "2026-04-14", "2026-04-15"], "csv"),
        "afrr_be_dec": (["2026-04-13", "2026-04-14", "2026-04-15"], "csv"),
        "afrr_de":     (["2026-04-13", "2026-04-14", "2026-04-15"], "xlsx"),
        "afrr_nl":     (["2026-04-13", "2026-04-14", "2026-04-15"], "csv"),
        "mfrr_be_inc": (["2026-04-13", "2026-04-14", "2026-04-15"], "csv"),
        "mfrr_be_dec": (["2026-04-13", "2026-04-14", "2026-04-15"], "csv"),
        "mfrr_de":     (["2026-04-13", "2026-04-14", "2026-04-15"], "xlsx"),
        "da":          (["2026-04-13", "2026-04-14", "2026-04-15"], "csv"),
        "solar":       (["2026-04-13", "2026-04-14", "2026-04-15"], "csv"),
    }
    ok, ko = 0, 0
    for prefix, (dates, ext) in prefixes_check.items():
        for d in dates:
            p = DATA_DIR / f"{prefix}_{d}.{ext}"
            if p.exists() and p.stat().st_size > 100:
                ok += 1
            else:
                log(f"  MANQUANT : {p.name}")
                ko += 1
    log(f"\n  {ok} fichiers OK  |  {ko} fichiers manquants")
