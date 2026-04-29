#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill price_min / price_max for 2026 rows in forecast_log_full.csv.

Sources:
  - merit_price : local afrr_be/nl CSV + afrr_de XLSX (data_ve_2026/)
  - voaa proxy  : ODS134 marginalincrementalprice / marginaldecrementalprice

DE XLSX format (2026):  _tmp_afrr_de_YYYY-MM-DD.xlsx
  PRODUCT   : NEG_001 .. NEG_096 -> DOWN direction, QH index 0..95
              POS_001 .. POS_096 -> UP direction,   QH index 0..95
  ENERGY_PRICE_[EUR/MWh], OFFERED_CAPACITY_[MW]
  No sign correction needed (prices already positive for both directions).

Formula (same as live compute_price_range):
  INC: price_min = voaa_dec,  price_max = merit_Inc_1000
  DEC: price_min = voaa_inc,  price_max = merit_Dec_1000
"""

import sys, time, logging, datetime as dt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

BASE_DIR   = Path("C:/Users/gusta/imbalance_optimisation")
VE_DIR     = BASE_DIR / "data_ve_2026"
DE_CSV_DIR = VE_DIR / "_de_csv_cache"
LOG_CSV    = BASE_DIR / "forecasters" / "elia_forecaster" / "forecast_log_full.csv"

THRESHOLD  = 1000
BRUSSELS   = "Europe/Brussels"
ODS134_URL = "https://external-elia.opendatasoft.com/api/records/1.0/search/"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Merit order ────────────────────────────────────────────────────────────────

def price_at(stack: pd.DataFrame, direction: str, threshold: float) -> float:
    sub = stack[stack["direction"] == direction][["price", "quantity"]].copy()
    sub["price"]    = pd.to_numeric(sub["price"],    errors="coerce")
    sub["quantity"] = pd.to_numeric(sub["quantity"], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return np.nan
    asc = direction == "UP"
    sub = sub.sort_values("price", ascending=asc)
    sub["cum"] = sub["quantity"].cumsum()
    eligible = sub[sub["cum"] <= threshold]
    return float(eligible["price"].iloc[-1]) if not eligible.empty else np.nan


def _convert_one(date: dt.date) -> tuple[dt.date, bool]:
    """Convert a single DE xlsx -> CSV. Returns (date, success)."""
    csv_path = DE_CSV_DIR / f"afrr_de_{date.isoformat()}.csv"
    if csv_path.exists() and csv_path.stat().st_size > 500:
        return date, True

    xlsx = VE_DIR / f"_tmp_afrr_de_{date.isoformat()}.xlsx"
    if not xlsx.exists():
        xlsx = VE_DIR / f"temp_afrr_de_{date.isoformat()}.xlsx"
    if not xlsx.exists():
        return date, False

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_excel(xlsx, usecols=["PRODUCT",
                                               "ENERGY_PRICE_[EUR/MWh]",
                                               "OFFERED_CAPACITY_[MW]"])
        df = df.copy()
        df["direction"] = df["PRODUCT"].apply(lambda p: "UP" if str(p).startswith("POS") else "DOWN")
        df["qh_idx"]    = df["PRODUCT"].str[4:].astype(int) - 1
        df["timestamp"] = pd.Timestamp(date) + pd.to_timedelta(df["qh_idx"] * 15, unit="min")
        df = df.rename(columns={
            "ENERGY_PRICE_[EUR/MWh]": "price",
            "OFFERED_CAPACITY_[MW]":  "quantity",
        })[["timestamp", "direction", "price", "quantity"]]
        df.to_csv(csv_path, index=False)
        return date, True
    except Exception as exc:
        log.warning("  DE convert %s failed: %s", date, exc)
        return date, False


def preconvert_de_xlsx(dates: list[dt.date], workers: int = 4) -> None:
    DE_CSV_DIR.mkdir(parents=True, exist_ok=True)
    need = [d for d in dates
            if not (DE_CSV_DIR / f"afrr_de_{d.isoformat()}.csv").exists()]
    if not need:
        log.info("DE CSV cache already complete (%d dates).", len(dates))
        return
    log.info("Pre-converting %d DE xlsx files (parallel=%d)...", len(need), workers)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_convert_one, d): d for d in need}
        for fut in as_completed(futures):
            date, ok = fut.result()
            done += 1
            if done % 10 == 0 or done == len(need):
                log.info("  DE convert: %d/%d", done, len(need))
    log.info("DE xlsx -> CSV conversion done.")


def load_stack_2026(date: dt.date) -> pd.DataFrame:
    frames = []

    for prefix in ("afrr_be", "afrr_nl"):
        f = VE_DIR / f"{prefix}_{date.isoformat()}.csv"
        if f.exists() and f.stat().st_size > 100:
            try:
                df = pd.read_csv(f, parse_dates=["timestamp"])
                df = df[["timestamp", "direction", "price", "quantity"]].copy()
                df["price"]    = pd.to_numeric(df["price"],    errors="coerce")
                df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
                frames.append(df)
            except Exception:
                pass

    de_csv = DE_CSV_DIR / f"afrr_de_{date.isoformat()}.csv"
    if de_csv.exists() and de_csv.stat().st_size > 100:
        try:
            df_de = pd.read_csv(de_csv, parse_dates=["timestamp"])
            frames.append(df_de)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame(columns=["timestamp", "direction", "price", "quantity"])
    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    return combined


# ── ODS134 VoAA ───────────────────────────────────────────────────────────────

def fetch_ods134_chunk(start_dt, end_dt, page_size=1000) -> pd.DataFrame:
    ws = pd.Timestamp(start_dt).tz_localize(BRUSSELS, nonexistent="shift_forward").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    we = pd.Timestamp(end_dt).tz_localize(BRUSSELS, nonexistent="shift_forward").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    all_records, start_i = [], 0
    while True:
        for attempt in range(3):
            try:
                r = requests.get(ODS134_URL, params={
                    "dataset": "ods134",
                    "q": f"datetime:[{ws} TO {we}]",
                    "rows": page_size, "start": start_i, "sort": "datetime",
                }, timeout=30)
                r.raise_for_status()
                data = r.json().get("records", [])
                break
            except Exception as e:
                log.warning("ODS134 attempt %d/3: %s", attempt + 1, e)
                time.sleep(5)
        else:
            break
        all_records.extend(data)
        if len(data) < page_size:
            break
        start_i += page_size
        if start_i + page_size > 9000:
            log.warning("ODS134 pagination cap at start=%d", start_i)
            break

    if not all_records:
        return pd.DataFrame()
    rows = []
    for rec in all_records:
        f = rec.get("fields", {})
        dt_raw = f.get("datetime")
        if dt_raw:
            try:
                ts = pd.to_datetime(dt_raw, utc=True).tz_convert(BRUSSELS).tz_localize(None)
                rows.append({
                    "forecast_time": ts,
                    "voaa_inc": f.get("marginalincrementalprice"),
                    "voaa_dec": f.get("marginaldecrementalprice"),
                })
            except Exception:
                pass
    df = pd.DataFrame(rows)
    if not df.empty:
        df["voaa_inc"] = pd.to_numeric(df["voaa_inc"], errors="coerce")
        df["voaa_dec"] = pd.to_numeric(df["voaa_dec"], errors="coerce")
    return df


def fetch_ods134_all(start_dt, end_dt, chunk_days=60) -> pd.DataFrame:
    frames = []
    cursor = pd.Timestamp(start_dt)
    end_ts = pd.Timestamp(end_dt)
    n = int((end_ts - cursor).days / chunk_days) + 1
    i = 0
    while cursor < end_ts:
        i += 1
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days), end_ts)
        log.info("ODS134 chunk [%d/%d] %s -> %s", i, n, cursor.date(), chunk_end.date())
        chunk = fetch_ods134_chunk(cursor, chunk_end)
        if not chunk.empty:
            frames.append(chunk)
        cursor = chunk_end
        time.sleep(1)

    if not frames:
        return pd.DataFrame(columns=["forecast_time", "voaa_inc", "voaa_dec"])
    return (pd.concat(frames, ignore_index=True)
              .drop_duplicates("forecast_time")
              .sort_values("forecast_time")
              .reset_index(drop=True))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("Loading %s ...", LOG_CSV)
    df = pd.read_csv(LOG_CSV, parse_dates=["forecast_time"])
    log.info("Total rows: %d", len(df))

    mask = (df["forecast_time"].dt.year == 2026) & df["price_min"].isna() & df["forecast_value"].notna()
    n_target = mask.sum()
    log.info("2026 rows to backfill: %d", n_target)
    if n_target == 0:
        log.info("Nothing to do.")
        return

    targets = df[mask][["forecast_time", "price_direction", "forecast_value"]].copy()
    unique_dates = sorted(targets["forecast_time"].dt.date.unique())
    log.info("Unique dates: %d  (%s -> %s)", len(unique_dates), unique_dates[0], unique_dates[-1])

    start_bt = targets["forecast_time"].min()
    end_bt   = targets["forecast_time"].max() + pd.Timedelta(minutes=15)

    # Step 0 — Pre-convert DE xlsx to CSV in parallel
    log.info("=" * 55)
    log.info("STEP 0: Pre-convert DE xlsx -> CSV cache")
    log.info("=" * 55)
    preconvert_de_xlsx(unique_dates, workers=8)

    # Step 1 — ODS134 VoAA
    log.info("=" * 55)
    log.info("STEP 1: Fetch ODS134 VoAA proxies (%s -> %s)", start_bt.date(), end_bt.date())
    log.info("=" * 55)
    ods = fetch_ods134_all(start_bt, end_bt)
    log.info("ODS134 rows: %d", len(ods))
    ods_idx = ods.set_index("forecast_time") if not ods.empty else pd.DataFrame()

    # Step 2 — Merit prices from local files
    log.info("=" * 55)
    log.info("STEP 2: Compute merit prices from local files")
    log.info("=" * 55)

    merit_map = {}
    stack_cache = {}

    for i, date in enumerate(unique_dates):
        if date not in stack_cache:
            stack_cache[date] = load_stack_2026(date)
        stack = stack_cache[date]

        qhs_on_date = targets[targets["forecast_time"].dt.date == date]
        for _, row in qhs_on_date.iterrows():
            qh        = row["forecast_time"]
            direction = row["price_direction"]
            if pd.isna(direction):
                fv = row["forecast_value"]
                direction = ("INC" if float(fv) < 0 else "DEC") if pd.notna(fv) else None

            voaa_inc = voaa_dec = np.nan
            if not ods_idx.empty and qh in ods_idx.index:
                voaa_inc = float(ods_idx.at[qh, "voaa_inc"]) if pd.notna(ods_idx.at[qh, "voaa_inc"]) else np.nan
                voaa_dec = float(ods_idx.at[qh, "voaa_dec"]) if pd.notna(ods_idx.at[qh, "voaa_dec"]) else np.nan

            merit_price = np.nan
            if not stack.empty:
                qh_stack = stack[stack["timestamp"].dt.floor("15min") == pd.Timestamp(qh).floor("15min")]
                if not qh_stack.empty:
                    bid_dir    = "UP" if direction == "INC" else "DOWN"
                    merit_price = price_at(qh_stack, bid_dir, THRESHOLD)

            if direction == "INC":
                p_min, p_max = voaa_dec, merit_price
            elif direction == "DEC":
                p_min, p_max = voaa_inc, merit_price
            else:
                p_min = p_max = np.nan

            merit_map[qh] = (p_min, p_max, voaa_inc, voaa_dec)

        if (i + 1) % 10 == 0 or (i + 1) == len(unique_dates):
            log.info("  [%d/%d] dates processed", i + 1, len(unique_dates))

    # Step 3 — Write back
    log.info("=" * 55)
    log.info("STEP 3: Update forecast_log_full.csv")
    log.info("=" * 55)

    updated = 0
    for idx in df.index[mask]:
        qh = df.at[idx, "forecast_time"]
        if qh in merit_map:
            p_min, p_max, vi, vd = merit_map[qh]
            df.at[idx, "price_min"] = round(p_min, 4) if pd.notna(p_min) else np.nan
            df.at[idx, "price_max"] = round(p_max, 4) if pd.notna(p_max) else np.nan
            df.at[idx, "voaa_inc"]  = round(vi, 4)    if pd.notna(vi)    else np.nan
            df.at[idx, "voaa_dec"]  = round(vd, 4)    if pd.notna(vd)    else np.nan
            updated += 1

    df.to_csv(LOG_CSV, index=False)
    log.info("Saved -> %s  (%d rows updated)", LOG_CSV, updated)

    # Coverage report
    total    = len(df)
    has_pm   = df["price_min"].notna().sum()
    has_pmax = df["price_max"].notna().sum()
    log.info("=" * 55)
    log.info("FINAL COVERAGE")
    log.info("  Total rows   : %d", total)
    log.info("  price_min    : %d (%.1f%%)", has_pm,   has_pm   / total * 100)
    log.info("  price_max    : %d (%.1f%%)", has_pmax, has_pmax / total * 100)
    log.info("=" * 55)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
