#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill price_min / price_max for 2025 rows in forecast_log_full.csv.

Sources:
  - merit_price  : local afrr_de/nl/be_2025-*.csv  (Inc_1000 / Dec_1000)
  - voaa proxy   : ODS134 marginalincrementalprice / marginaldecrementalprice

Formula (same as live compute_price_range):
  INC: price_min = voaa_dec,  price_max = merit_Inc_1000
  DEC: price_min = voaa_inc,  price_max = merit_Dec_1000
"""

import sys, time, logging
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path("C:/Users/gusta/imbalance_optimisation")
VE_DIR   = BASE_DIR / "data_ve_2025"
LOG_CSV  = BASE_DIR / "forecasters" / "elia_forecaster" / "forecast_log_full.csv"

THRESHOLD  = 1000          # MW  (Inc_1000 / Dec_1000 — last threshold in live code)
BRUSSELS   = "Europe/Brussels"
ODS134_URL = "https://external-elia.opendatasoft.com/api/records/1.0/search/"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Merit order helpers ────────────────────────────────────────────────────────

def price_at(stack: pd.DataFrame, direction: str, threshold: float) -> float:
    """Price at cumulative volume threshold for a given direction (UP/DOWN)."""
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


def load_stack(date: dt.date) -> pd.DataFrame:
    """Load combined BE + NL + DE aFRR bid stack for a calendar date.

    DE sign correction: the local afrr_de CSV stores UP bid prices in the
    'provider-to-grid' convention (all negative), opposite to BE/NL which use
    positive prices. We negate DE UP prices to align the convention.
    Sentinel bids with |price| > 3000 €/MWh are filtered from all countries.
    """
    frames = []
    for prefix in ("afrr_be", "afrr_nl", "afrr_de"):
        f = VE_DIR / f"{prefix}_{date.isoformat()}.csv"
        if f.exists() and f.stat().st_size > 100:
            try:
                df = pd.read_csv(f, parse_dates=["timestamp"])
                df = df[["timestamp", "direction", "price", "quantity"]].copy()
                df["price"]    = pd.to_numeric(df["price"],    errors="coerce")
                df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
                if prefix == "afrr_de":
                    # DE UP prices are negative (provider-to-grid convention) → negate
                    df.loc[df["direction"] == "UP", "price"] *= -1
                frames.append(df)
            except Exception:
                pass
    if not frames:
        return pd.DataFrame(columns=["timestamp", "direction", "price", "quantity"])
    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    return combined


# ── ODS134 fetch ───────────────────────────────────────────────────────────────

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
        if start_i + page_size > 9000:   # stay under 10 000 hard limit
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
    """Fetch ODS134 over a long range, chunked to avoid 10 000-record API limit."""
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

    # Only rows that have a forecast but no price range yet
    mask = df["forecast_value"].notna() & df["price_min"].isna() & df["price_max"].isna()
    n_target = mask.sum()
    log.info("Rows to backfill: %d", n_target)
    if n_target == 0:
        log.info("Nothing to do.")
        return

    start_bt = df.loc[mask, "forecast_time"].min()
    end_bt   = df.loc[mask, "forecast_time"].max() + pd.Timedelta(minutes=15)
    log.info("Backfill range: %s -> %s", start_bt.date(), end_bt.date())

    # Step 1 — ODS134 voaa proxies
    log.info("=" * 50)
    log.info("STEP 1: Fetch ODS134 voaa proxies")
    log.info("=" * 50)
    ods = fetch_ods134_all(start_bt, end_bt)
    log.info("ODS134 rows: %d", len(ods))
    ods_idx = ods.set_index("forecast_time") if not ods.empty else pd.DataFrame()

    # Step 2 — Merit order from local files
    log.info("=" * 50)
    log.info("STEP 2: Compute merit order prices from local files")
    log.info("=" * 50)

    targets = df[mask][["forecast_time", "price_direction", "forecast_value"]].copy()
    unique_dates = sorted(targets["forecast_time"].dt.date.unique())
    log.info("Unique dates: %d", len(unique_dates))

    merit_map = {}   # forecast_time -> (price_min, price_max)
    stack_cache = {}

    for i, date in enumerate(unique_dates):
        if date not in stack_cache:
            stack_cache[date] = load_stack(date)
        stack = stack_cache[date]

        qhs_on_date = targets[targets["forecast_time"].dt.date == date]
        for _, row in qhs_on_date.iterrows():
            qh  = row["forecast_time"]
            direction = row["price_direction"]
            if pd.isna(direction):
                fv = row["forecast_value"]
                direction = ("INC" if float(fv) < 0 else "DEC") if pd.notna(fv) else None

            # VoAA proxy from ODS134
            voaa_inc = voaa_dec = np.nan
            if not ods_idx.empty and qh in ods_idx.index:
                voaa_inc = float(ods_idx.at[qh, "voaa_inc"])
                voaa_dec = float(ods_idx.at[qh, "voaa_dec"])

            # Merit price from local stack at this QH
            merit_price = np.nan
            if not stack.empty:
                qh_stack = stack[stack["timestamp"].dt.floor("15min") == pd.Timestamp(qh).floor("15min")]
                if not qh_stack.empty:
                    bid_dir = "UP" if direction == "INC" else "DOWN"
                    merit_price = price_at(qh_stack, bid_dir, THRESHOLD)

            # price_min / price_max (same formula as live compute_price_range)
            if direction == "INC":
                p_min, p_max = voaa_dec, merit_price
            elif direction == "DEC":
                p_min, p_max = voaa_inc, merit_price
            else:  # direction unknown (no forecast_value to infer from)
                p_min = p_max = np.nan

            merit_map[qh] = (p_min, p_max)

        if (i + 1) % 30 == 0 or (i + 1) == len(unique_dates):
            log.info("  [%d/%d] dates processed", i + 1, len(unique_dates))

    # Step 3 — Write back
    log.info("=" * 50)
    log.info("STEP 3: Update forecast_log_full.csv")
    log.info("=" * 50)

    updated = 0
    for idx in df.index[mask]:
        qh = df.at[idx, "forecast_time"]
        if qh in merit_map:
            p_min, p_max = merit_map[qh]
            df.at[idx, "price_min"] = round(p_min, 4) if pd.notna(p_min) else np.nan
            df.at[idx, "price_max"] = round(p_max, 4) if pd.notna(p_max) else np.nan
            updated += 1

    filled = df.loc[mask, "price_min"].notna().sum()
    log.info("Rows updated: %d  |  price_min non-null: %d / %d", updated, filled, n_target)

    df.to_csv(LOG_CSV, index=False)
    log.info("Saved -> %s", LOG_CSV)
    log.info("Done.")


if __name__ == "__main__":
    main()
