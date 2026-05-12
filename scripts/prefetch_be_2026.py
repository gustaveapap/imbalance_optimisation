#!/usr/bin/env python3
"""
Pré-télécharge en parallèle imbalance BE + DA BE + solar BE pour 2026.
4 threads = ~4x plus rapide que le sim sequentiel.
"""
import sys, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "optimizers" / "solar_fr"))
os.chdir(str(REPO))

from test_logic_fr import align_qh, file_ok, safe_float, qh_range

DATA_2026    = REPO / "data_ve_2026"
ENTSOE_TOKEN = "dcc0c513-dab6-4add-82be-7b693f2b7fab"
ENTSOE_API   = "https://web-api.tp.entsoe.eu/api"
SOLAR_URL    = "https://transparency.entsoe.eu/generation/forecast/windAndSolar/solar/load"
BE_DOMAIN    = "10YBE----------2"
BE_AREA      = "BZN|10YBE----------2"
BRUSSELS     = "Europe/Brussels"
ODS133_URL   = "https://external-elia.opendatasoft.com/api/records/1.0/search/"
ODS134_URL   = "https://external-elia.opendatasoft.com/api/records/1.0/search/"

START = "2026-01-01"
END   = "2026-05-11"

all_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(START, END, freq="D")]

# ── ODS133 ────────────────────────────────────────────────────────────────────
def fetch_ods133(date_str):
    p = DATA_2026 / f"_ods133_{date_str}.csv"
    if file_ok(p):
        return pd.read_csv(p, parse_dates=["timestamp"])
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
            except Exception: time.sleep(5)
        else: return pd.DataFrame()
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
    df = pd.concat(frames, ignore_index=True).drop_duplicates("datetime").sort_values("datetime")
    df = df[df["datetime"].dt.date == d]
    df["timestamp"] = df["datetime"].dt.floor("15min")
    df = df.groupby("timestamp", as_index=False)["volume"].mean()
    df.to_csv(p, index=False)
    return df

# ── ODS134 ────────────────────────────────────────────────────────────────────
def fetch_ods134(date_str):
    p = DATA_2026 / f"_ods134_{date_str}.csv"
    if file_ok(p):
        return pd.read_csv(p, parse_dates=["timestamp"])
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
    df = df.drop_duplicates("timestamp")
    df.to_csv(p, index=False)
    return df

# ── IMBALANCE BE (merge ODS133 + ODS134) ─────────────────────────────────────
def fetch_imbalance_be(date_str):
    out = DATA_2026 / f"imbalance_be_{date_str}.csv"
    if file_ok(out):
        return True
    df_vol = fetch_ods133(date_str)
    df_isp = fetch_ods134(date_str)
    if df_vol.empty:
        return False
    if not df_isp.empty:
        df = df_vol.merge(df_isp, on="timestamp", how="left")
    else:
        df = df_vol.copy(); df["isp"] = 0.0
    df = align_qh(df, date_str, "ffill")
    df["volume"] = df["volume"].fillna(0)
    df["isp"]    = df["isp"].fillna(0)
    df.to_csv(out, index=False)
    return True

# ── DA BE (ENTSO-E A44) ───────────────────────────────────────────────────────
def fetch_da_be(date_str):
    out = DATA_2026 / f"prix_da_be_{date_str}.csv"
    if file_ok(out):
        return True
    start = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d%H%M")
    end   = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y%m%d%H%M")
    for attempt in range(3):
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
            if not rows: return False
            pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp").to_csv(out, index=False)
            return True
        except Exception: time.sleep(5)
    return False

# ── SOLAR BE ──────────────────────────────────────────────────────────────────
def fetch_solar_be(date_str):
    out = DATA_2026 / f"solar_be_{date_str}.csv"
    if file_ok(out):
        return True
    date_from = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    date_to   = date_from + timedelta(days=1)
    try:
        r = requests.post(SOLAR_URL, json={
            "dateTimeRange": {
                "from": date_from.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "to":   date_to.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            },
            "areaList": [BE_AREA], "timeZone": "CET",
            "sorterList": [], "filterMap": {},
        }, headers={"accept": "application/json",
                    "content-type": "application/json; charset=utf-8"}, timeout=90)
        if r.status_code != 200: return False
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
                    rows.append({"timestamp": ts,
                                 "forecast_mw": safe_float(v[2]) if len(v)>2 else np.nan,
                                 "actual_mw":   safe_float(v[3]) if len(v)>3 else np.nan})
        if not rows: return False
        df = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
        df = align_qh(df, date_str, "interp")
        df["forecast_mw"] = df["forecast_mw"].fillna(0)
        df["actual_mw"]   = df["actual_mw"].fillna(0)
        df.to_csv(out, index=False)
        return True
    except Exception: return False

# ── MAIN ──────────────────────────────────────────────────────────────────────
def process_day(date_str):
    ok_imb    = fetch_imbalance_be(date_str)
    ok_da     = fetch_da_be(date_str)
    ok_solar  = fetch_solar_be(date_str)
    return date_str, ok_imb, ok_da, ok_solar

if __name__ == "__main__":
    missing = [d for d in all_dates
               if not file_ok(DATA_2026 / f"imbalance_be_{d}.csv")]
    print(f"Prefetch BE 2026: {len(missing)} jours manquants sur {len(all_dates)}")

    done, failed = 0, 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(process_day, d): d for d in missing}
        for fut in as_completed(futures):
            d, ok_imb, ok_da, ok_solar = fut.result()
            status = "OK" if ok_imb else "FAIL-imb"
            done += 1
            print(f"  [{done}/{len(missing)}] {d}  imb={ok_imb} da={ok_da} solar={ok_solar}")

    n_imb   = sum(file_ok(DATA_2026 / f"imbalance_be_{d}.csv") for d in all_dates)
    n_da    = sum(file_ok(DATA_2026 / f"prix_da_be_{d}.csv")   for d in all_dates)
    n_solar = sum(file_ok(DATA_2026 / f"solar_be_{d}.csv")     for d in all_dates)
    print(f"\nResultat : imbalance={n_imb}/{len(all_dates)}  DA={n_da}/{len(all_dates)}  solar={n_solar}/{len(all_dates)}")
