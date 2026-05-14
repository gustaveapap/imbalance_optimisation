#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fill remaining missing QHs in forecast_log_full.csv between 2026-02-01 and 2026-03-31.
SI 1-min should already be cached in data/backfill_elia/ from the previous fill run.
For any dates where the cache has gaps, attempts a fresh download from ODS133.
"""

import sys, time, logging, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import joblib

BASE_DIR = Path("C:/Users/gusta/imbalance_optimisation")
ELIA_DIR = BASE_DIR / "forecasters" / "elia_forecaster"
SI_DIR   = BASE_DIR / "data" / "backfill_elia"
LOG_CSV  = ELIA_DIR / "forecast_log_full.csv"
MODEL    = ELIA_DIR / "models" / "imbalance_forecaster_v1.joblib"

sys.path.insert(0, str(ELIA_DIR))
from system_imbalance_forecaster.utils.feature_utils import extract_features

BRUSSELS   = "Europe/Brussels"
ODS133_URL = "https://external-elia.opendatasoft.com/api/records/1.0/search/"
ODS134_URL = "https://external-elia.opendatasoft.com/api/records/1.0/search/"

RANGE_START = pd.Timestamp("2026-02-01")
RANGE_END   = pd.Timestamp("2026-03-31 23:45")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _download_ods133_day(date_str: str, out_path: Path, force: bool = False) -> int:
    if out_path.exists() and out_path.stat().st_size > 10_000 and not force:
        return -1  # cached
    d = dt.date.fromisoformat(date_str)
    start_utc = dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc) - dt.timedelta(hours=2)
    end_utc   = dt.datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=dt.timezone.utc) + dt.timedelta(hours=2)
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
                if r.status_code == 429:
                    time.sleep(10)
                    continue
                r.raise_for_status()
                records = r.json().get("records", [])
                break
            except Exception as e:
                log.warning("ODS133 %s attempt %d/3: %s", date_str, attempt + 1, e)
                time.sleep(5 * (attempt + 1))
        else:
            return 0
        rows = []
        for rec in records:
            f = rec.get("fields", {})
            dt_raw = f.get("datetime", "")
            si_val = f.get("systemimbalance")
            if dt_raw and si_val is not None:
                try:
                    ts = pd.to_datetime(dt_raw, utc=True).tz_convert(BRUSSELS).tz_localize(None)
                    rows.append({"datetime": ts, "actual_system_imbalance": float(si_val)})
                except Exception:
                    pass
        if rows:
            frames.append(pd.DataFrame(rows))
        if len(records) < 1000:
            break
        start += 1000
    if not frames:
        return 0
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates("datetime")
            .sort_values("datetime")
            .reset_index(drop=True))
    df = df[df["datetime"].dt.date == d]
    df.to_csv(out_path, index=False)
    return len(df)


def load_si_for_dates(dates_needed: set, force_dates: set = None) -> pd.DataFrame:
    SI_DIR.mkdir(parents=True, exist_ok=True)
    if force_dates is None:
        force_dates = set()
    frames = []
    for d in sorted(dates_needed):
        date_str = d.isoformat()
        out = SI_DIR / f"si_1min_{date_str}.csv"
        force = d in force_dates
        if force or not (out.exists() and out.stat().st_size > 10_000):
            log.info("  Downloading ODS133 %s (force=%s)...", date_str, force)
            n = _download_ods133_day(date_str, out, force=force)
            if n == 0:
                log.warning("  %s: 0 rows returned", date_str)
            time.sleep(1)
        if out.exists() and out.stat().st_size > 100:
            try:
                frames.append(pd.read_csv(out, parse_dates=["datetime"]))
            except Exception:
                pass
    if not frames:
        return pd.DataFrame()
    return (pd.concat(frames, ignore_index=True)
              .drop_duplicates("datetime")
              .sort_values("datetime")
              .set_index("datetime"))


def fetch_ods134_actuals(start_dt, end_dt) -> pd.DataFrame:
    ws = pd.Timestamp(start_dt).tz_localize(BRUSSELS, nonexistent="shift_forward").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    we = pd.Timestamp(end_dt).tz_localize(BRUSSELS, nonexistent="shift_forward").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
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
            except Exception as e:
                log.warning("ODS134 attempt %d/3: %s", attempt + 1, e)
                time.sleep(5)
        else:
            break
        all_records.extend(data)
        if len(data) < 1000:
            break
        start_i += 1000
        if start_i >= 9000:
            break
    rows = []
    for rec in all_records:
        f = rec.get("fields", {})
        dt_raw = f.get("datetime")
        if dt_raw:
            try:
                ts = pd.to_datetime(dt_raw, utc=True).tz_convert(BRUSSELS).tz_localize(None)
                rows.append({
                    "forecast_time": ts.floor("15min"),
                    "actual_volume": f.get("systemimbalance"),
                    "actual_isp":    f.get("imbalanceprice"),
                })
            except Exception:
                pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["actual_volume"] = pd.to_numeric(df["actual_volume"], errors="coerce")
    df["actual_isp"]    = pd.to_numeric(df["actual_isp"],    errors="coerce")
    return df.drop_duplicates("forecast_time").set_index("forecast_time")


def main():
    log.info("Loading %s ...", LOG_CSV)
    df = pd.read_csv(LOG_CSV, parse_dates=["forecast_time"])

    expected = pd.date_range(RANGE_START, RANGE_END, freq="15min")
    present  = set(df["forecast_time"])
    missing  = [t for t in expected if t not in present]
    log.info("Missing QHs in %s - %s: %d", RANGE_START.date(), RANGE_END.date(), len(missing))
    if not missing:
        log.info("Nothing to do.")
        return

    # Identify dates where extract_features might fail due to SI gaps
    dates_needed: set = set()
    for qh in missing:
        dates_needed.add(qh.date())
        if qh.hour == 0:
            dates_needed.add((qh - pd.Timedelta(days=1)).date())

    # First pass: try with existing cache
    log.info("Loading SI 1-min from cache (%d dates)...", len(dates_needed))
    si_all = load_si_for_dates(dates_needed)
    log.info("SI loaded: %d rows", len(si_all))

    # Identify which QHs still fail (SI gap dates) — force re-download for those dates
    failed_dates: set = set()
    for qh in missing:
        features = extract_features(si_all, qh) if not si_all.empty else None
        if features is None:
            failed_dates.add(qh.date())
            if qh.hour == 0:
                failed_dates.add((qh - pd.Timedelta(days=1)).date())

    if failed_dates:
        log.info("Re-downloading SI for %d dates with gaps: %s",
                 len(failed_dates), sorted(failed_dates))
        si_all = load_si_for_dates(dates_needed, force_dates=failed_dates)
        log.info("SI reloaded: %d rows", len(si_all))

    # Fetch actuals from ODS134
    log.info("Fetching ODS134 actuals (%s -> %s)...", missing[0].date(), missing[-1].date())
    actuals = fetch_ods134_actuals(missing[0], missing[-1] + pd.Timedelta(minutes=15))
    log.info("ODS134 actuals: %d QHs", len(actuals))

    model   = joblib.load(MODEL)
    now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    buf     = []
    n_ok = n_err = 0

    for qh in missing:
        features = extract_features(si_all, qh) if not si_all.empty else None
        if features is None:
            n_err += 1
            continue
        try:
            forecast_value  = float(model.predict(features.reshape(1, -1))[0])
            price_direction = "INC" if forecast_value < 0 else "DEC"
        except Exception:
            n_err += 1
            continue

        actual_volume = actual_isp = None
        if not actuals.empty and qh in actuals.index:
            av = actuals.at[qh, "actual_volume"]
            ai = actuals.at[qh, "actual_isp"]
            actual_volume = float(av) if pd.notna(av) else None
            actual_isp    = float(ai) if pd.notna(ai) else None

        buf.append({
            "forecast_time":         qh,
            "forecast_value":        round(forecast_value, 6),
            "price_direction":       price_direction,
            "actual_volume":         actual_volume,
            "actual_isp":            actual_isp,
            "forecast_generated_at": now_str,
        })
        n_ok += 1

    log.info("Forecast done: %d ok / %d err", n_ok, n_err)

    if not buf:
        log.info("No new rows to add.")
        return

    new_rows = pd.DataFrame(buf)
    full = (pd.concat([df, new_rows], ignore_index=True)
              .drop_duplicates(subset=["forecast_time"])
              .sort_values("forecast_time")
              .reset_index(drop=True))
    full.to_csv(LOG_CSV, index=False)

    total     = len(full)
    has_fcast = full["forecast_value"].notna().sum()
    log.info("=" * 55)
    log.info("FINAL COVERAGE")
    log.info("  Total rows     : %d", total)
    log.info("  forecast_value : %d (%.1f%%)", has_fcast, has_fcast / total * 100)
    log.info("  Still missing  : %d", len(expected) - full["forecast_time"].isin(expected).sum())
    log.info("=" * 55)


if __name__ == "__main__":
    main()
