#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4-step Elia + RTE backfill pipeline.
Run from: C:/Users/gusta/imbalance_optimisation/

Step 1 — Merge RTE logs (backfill + live) → forecast_log_full.csv
Step 2 — Download ODS133 SI 1-min for 2025-01-01 → 2025-10-26
Step 3 — Elia walk-forward backfill → forecast_log_backfill.csv
Step 4 — Merge Elia logs (backfill + forecastV3) → forecast_log_full.csv
"""

import sys, time, logging
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import joblib

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path("C:/Users/gusta/imbalance_optimisation")
ELIA_DIR = BASE_DIR / "forecasters/elia_forecaster"
RTE_DIR  = BASE_DIR / "forecasters/rte_forecaster"
VE_DIR   = BASE_DIR / "data_ve_2025"
SI_DIR   = BASE_DIR / "data/backfill_elia"

sys.path.insert(0, str(ELIA_DIR))
from system_imbalance_forecaster.utils.feature_utils import extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ODS133_URL = "https://external-elia.opendatasoft.com/api/records/1.0/search/"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Merge RTE logs
# ══════════════════════════════════════════════════════════════════════════════
def step1_merge_rte():
    log.info("=" * 60)
    log.info("STEP 1: Merge RTE logs")
    log.info("=" * 60)

    backfill = pd.read_csv(RTE_DIR / "forecast_log_backfill.csv", parse_dates=["timestamp"])
    live     = pd.read_csv(RTE_DIR / "forecast_log.csv",          parse_dates=["timestamp"])

    full = (pd.concat([backfill, live], ignore_index=True)
              .drop_duplicates(subset=["timestamp"])
              .sort_values("timestamp")
              .reset_index(drop=True))

    out = RTE_DIR / "forecast_log_full.csv"
    full.to_csv(out, index=False)

    log.info("STEP 1 done: %d rows → %s", len(full), out)
    log.info("  backfill : %d rows  (%s → %s)",
             len(backfill), backfill['timestamp'].min(), backfill['timestamp'].max())
    log.info("  live     : %d rows  (%s → %s)",
             len(live), live['timestamp'].min(), live['timestamp'].max())
    log.info("  full     : %d rows  (%s → %s)",
             len(full), full['timestamp'].min(), full['timestamp'].max())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Download ODS133 SI 1-min
# ══════════════════════════════════════════════════════════════════════════════
def _download_ods133_day(date_str: str, out_path: Path) -> int:
    d = dt.date.fromisoformat(date_str)
    # UTC window: ±2h around calendar day to cover all CET/CEST transitions
    start_utc = dt.datetime(d.year, d.month, d.day, 0, 0, 0,
                            tzinfo=dt.timezone.utc) - dt.timedelta(hours=2)
    end_utc   = dt.datetime(d.year, d.month, d.day, 23, 59, 59,
                            tzinfo=dt.timezone.utc) + dt.timedelta(hours=2)
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
                log.warning("  ODS133 %s attempt %d/3: %s", date_str, attempt + 1, e)
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
                    ts = (pd.to_datetime(dt_raw, utc=True)
                            .tz_convert("Europe/Brussels")
                            .tz_localize(None))
                    rows.append({"datetime": ts,
                                 "actual_system_imbalance": float(si_val)})
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
    df = df[df["datetime"].dt.date == d]   # filter to exact CET calendar date
    df.to_csv(out_path, index=False)
    return len(df)


def step2_download_si1min():
    log.info("=" * 60)
    log.info("STEP 2: Download ODS133 SI 1-min (2025-01-01 → 2025-10-26)")
    log.info("=" * 60)
    SI_DIR.mkdir(parents=True, exist_ok=True)

    dates = [dt.date(2025, 1, 1) + dt.timedelta(days=i)
             for i in range((dt.date(2025, 10, 26) - dt.date(2025, 1, 1)).days + 1)]
    total = len(dates)
    done = skipped = errors = 0

    for i, d in enumerate(dates):
        date_str = d.isoformat()
        out = SI_DIR / f"si_1min_{date_str}.csv"

        if out.exists() and out.stat().st_size > 10_000:
            skipped += 1
            continue

        n = _download_ods133_day(date_str, out)
        if n > 0:
            done += 1
        else:
            errors += 1
            log.warning("  %s: 0 rows returned", date_str)

        if (i + 1) % 10 == 0:
            log.info("  [%d/%d] %s — done=%d skip=%d err=%d",
                     i + 1, total, date_str, done, skipped, errors)
        time.sleep(1)

    log.info("STEP 2 done: %d downloaded, %d skipped, %d errors.", done, skipped, errors)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Elia walk-forward backfill
# ══════════════════════════════════════════════════════════════════════════════
def step3_elia_backfill():
    log.info("=" * 60)
    log.info("STEP 3: Elia walk-forward backfill (2025-01-01 → 2025-10-26)")
    log.info("=" * 60)

    model = joblib.load(ELIA_DIR / "models" / "imbalance_forecaster_v1.joblib")
    log.info("Model loaded.")

    OUTPUT = ELIA_DIR / "forecast_log_backfill.csv"
    done: set = set()
    if OUTPUT.exists():
        prev = pd.read_csv(OUTPUT, parse_dates=["timestamp"])
        done = set(pd.to_datetime(prev["timestamp"]))
        log.info("Resuming: %d QHs already done.", len(done))
    else:
        pd.DataFrame(columns=[
            "timestamp", "forecast_value", "price_direction",
            "actual_volume", "actual_isp"
        ]).to_csv(OUTPUT, index=False)

    # Load all SI 1-min into memory
    log.info("Loading SI 1-min files from %s ...", SI_DIR)
    si_files = sorted(SI_DIR.glob("si_1min_*.csv"))
    if not si_files:
        log.error("No SI 1-min files found — run STEP 2 first.")
        return

    si_all = (pd.concat(
                  [pd.read_csv(f, parse_dates=["datetime"]) for f in si_files],
                  ignore_index=True)
               .drop_duplicates("datetime")
               .sort_values("datetime")
               .set_index("datetime"))
    log.info("SI 1-min: %d rows  (%s → %s)",
             len(si_all), si_all.index.min(), si_all.index.max())

    # Load imbalance_be actuals (QH resolution, CET naive)
    log.info("Loading imbalance_be actuals from %s ...", VE_DIR)
    imbalance_cache: dict = {}
    for f in sorted(VE_DIR.glob("imbalance_be_2025-*.csv")):
        date_key = dt.date.fromisoformat(f.stem.replace("imbalance_be_", ""))
        df_act = pd.read_csv(f, parse_dates=["timestamp"]).set_index("timestamp")
        imbalance_cache[date_key] = df_act
    log.info("Loaded %d imbalance_be files.", len(imbalance_cache))

    qh_range = pd.date_range(
        dt.datetime(2025, 1, 1, 0, 0),
        dt.datetime(2025, 10, 26, 23, 45),
        freq="15min"
    )
    total = len(qh_range)
    log.info("QHs to forecast: %d", total)

    buf = []
    n_ok = n_skip = n_err = 0

    for i, qh in enumerate(qh_range):
        if qh in done:
            n_skip += 1
            continue

        # ── features: 60 SI 1-min values in [qh-65min, qh-5min) ─────────────
        features = extract_features(si_all, qh)
        if features is None:
            n_err += 1
            continue

        # ── forecast ──────────────────────────────────────────────────────────
        try:
            forecast_value  = float(model.predict(features.reshape(1, -1))[0])
            price_direction = "INC" if forecast_value < 0 else "DEC"
        except Exception as e:
            log.warning("QH %s predict failed: %s", qh, e)
            n_err += 1
            continue

        # ── actual volume + ISP from imbalance_be ─────────────────────────────
        actual_volume = actual_isp = None
        qh_date = qh.date()
        if qh_date in imbalance_cache:
            try:
                row = imbalance_cache[qh_date].loc[qh]
                actual_volume = float(row["volume"])
                actual_isp    = float(row["isp"])
            except KeyError:
                pass

        buf.append({
            "timestamp":       qh,
            "forecast_value":  round(forecast_value, 6),
            "price_direction": price_direction,
            "actual_volume":   actual_volume,
            "actual_isp":      actual_isp,
        })
        n_ok += 1

        if len(buf) >= 50:
            pd.DataFrame(buf).to_csv(OUTPUT, mode="a", header=False, index=False)
            buf.clear()

        if (i + 1) % 500 == 0:
            log.info("  [%d/%d] %.1f%%  ok=%d  skip=%d  err=%d",
                     i + 1, total, 100 * (i + 1) / total, n_ok, n_skip, n_err)

    if buf:
        pd.DataFrame(buf).to_csv(OUTPUT, mode="a", header=False, index=False)

    log.info("STEP 3 done: %d ok / %d skip / %d err → %s", n_ok, n_skip, n_err, OUTPUT)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Merge Elia logs
# ══════════════════════════════════════════════════════════════════════════════
def step4_merge_elia():
    log.info("=" * 60)
    log.info("STEP 4: Merge Elia logs")
    log.info("=" * 60)

    backfill = pd.read_csv(ELIA_DIR / "forecast_log_backfill.csv",
                           parse_dates=["timestamp"])
    backfill = backfill.rename(columns={"timestamp": "forecast_time"})

    live = pd.read_csv(ELIA_DIR / "forecastV3.csv",
                       parse_dates=["forecast_time"])

    full = (pd.concat([backfill, live], ignore_index=True)
              .drop_duplicates(subset=["forecast_time"])
              .sort_values("forecast_time")
              .reset_index(drop=True))

    out = ELIA_DIR / "forecast_log_full.csv"
    full.to_csv(out, index=False)

    log.info("STEP 4 done: %d rows → %s", len(full), out)
    log.info("  backfill : %d rows  (%s → %s)",
             len(backfill), backfill['forecast_time'].min(), backfill['forecast_time'].max())
    log.info("  live     : %d rows  (%s → %s)",
             len(live), live['forecast_time'].min(), live['forecast_time'].max())
    log.info("  full     : %d rows  (%s → %s)",
             len(full), full['forecast_time'].min(), full['forecast_time'].max())


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    step1_merge_rte()
    step2_download_si1min()
    step3_elia_backfill()
    step4_merge_elia()
    log.info("All steps complete.")
