#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill RTE imbalance forecasts: 2025-01-01 00:00 CET → 2025-07-10 23:45 CET.

Strategy:
  1. Download the full data range once and cache as parquet (resumable).
  2. For each 15-min QH target, slice the 27-h lag window from the cache,
     run build_features() + forecast_next() with the pre-trained model,
     and record the actual value (already published for all past QHs).
  3. Append results to forecast_log_backfill.csv; safe to re-run.

Output: forecasters/rte_forecaster/forecast_log_backfill.csv
Cache:  _backfill_rte_cache.parquet  (deleted after successful run if desired)
"""

import sys, time, logging
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

BASE_DIR = Path(__file__).parent
RTE_DIR  = BASE_DIR / "forecasters" / "rte_forecaster"
sys.path.insert(0, str(RTE_DIR))

from run_forecast_scheduler import (
    CLIENT_ID, CLIENT_SECRET,
    get_token, download_imbalance, build_features, forecast_next,
)

# ── constants ──────────────────────────────────────────────────────────────────
MODEL_PATH  = RTE_DIR / "artifacts" / "fr_imbalance_full_model.pkl"
OUTPUT_CSV  = RTE_DIR / "forecast_log_backfill.csv"
CACHE_FILE  = BASE_DIR / "_backfill_rte_cache.csv"

BACKFILL_START = dt.datetime(2025, 1,  1,  0,  0)   # first QH to forecast (CET naive)
BACKFILL_END   = dt.datetime(2026, 4, 28, 17, 30)   # last  QH to forecast (CET naive)
LAG_WINDOW     = dt.timedelta(hours=27)              # history window per QH (matches live)
MIN_ROWS       = 96                                  # lag_96 needs 96 past QHs minimum
CHUNK_DAYS     = 7                                   # days per API call
API_SLEEP      = 2.0                                 # seconds between chunk calls

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── token management ───────────────────────────────────────────────────────────
_tok: str | None = None
_tok_ts: dt.datetime | None = None
TOKEN_TTL = dt.timedelta(minutes=50)   # refresh before 1-hour API expiry

def fresh_token() -> str:
    global _tok, _tok_ts
    if _tok is None or dt.datetime.now() - _tok_ts > TOKEN_TTL:
        log.info("Requesting new OAuth token…")
        _tok = get_token(CLIENT_ID, CLIENT_SECRET)
        _tok_ts = dt.datetime.now()
    return _tok


# ── data download ──────────────────────────────────────────────────────────────
def download_chunk_safe(start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    for attempt in range(4):
        try:
            df = download_imbalance(fresh_token(), start, end)
            return df
        except Exception as exc:
            log.warning("Chunk %s→%s attempt %d/4 failed: %s", start, end, attempt + 1, exc)
            time.sleep(15 * (attempt + 1))
    log.error("Chunk %s→%s permanently failed, skipping.", start, end)
    return pd.DataFrame({"imbalance_mwh": pd.Series(dtype=float)})


def build_or_load_cache() -> pd.DataFrame:
    # need data from 27h before first QH through 15 min after last QH (for actuals)
    data_start = BACKFILL_START - LAG_WINDOW - dt.timedelta(minutes=15)
    data_end   = BACKFILL_END   + dt.timedelta(minutes=15)

    if CACHE_FILE.exists():
        cache = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        if (not cache.empty
                and cache.index.min() <= pd.Timestamp(data_start) + dt.timedelta(hours=2)
                and cache.index.max() >= pd.Timestamp(data_end)   - dt.timedelta(hours=2)):
            log.info("Cache OK — %d rows, %s → %s.", len(cache), cache.index.min(), cache.index.max())
            return cache
        log.info("Cache incomplete, re-downloading.")

    n_chunks = int((data_end - data_start).days / CHUNK_DAYS) + 1
    log.info("Downloading %s → %s in %d-day chunks (%d total)…",
             data_start.date(), data_end.date(), CHUNK_DAYS, n_chunks)

    frames = []
    cursor = data_start
    chunk_i = 0
    while cursor < data_end:
        chunk_end = min(cursor + dt.timedelta(days=CHUNK_DAYS), data_end)
        chunk_i += 1
        log.info("[%d/%d] %s → %s", chunk_i, n_chunks, cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
        df = download_chunk_safe(cursor, chunk_end)
        if not df.empty:
            frames.append(df)
        cursor = chunk_end
        time.sleep(API_SLEEP)

    if not frames:
        raise RuntimeError("No data downloaded — check API credentials / connectivity.")

    cache = pd.concat(frames)
    cache = cache[~cache.index.duplicated(keep="last")].sort_index()
    cache.to_csv(CACHE_FILE)
    log.info("Cache saved: %d rows, %s → %s.", len(cache), cache.index.min(), cache.index.max())
    return cache


# ── backfill loop ──────────────────────────────────────────────────────────────
def run_backfill(cache: pd.DataFrame) -> None:
    model = joblib.load(MODEL_PATH)
    log.info("Model loaded: %s", MODEL_PATH)

    # load already-processed timestamps for resumability
    done: set[pd.Timestamp] = set()
    if OUTPUT_CSV.exists():
        prev = pd.read_csv(OUTPUT_CSV, parse_dates=["timestamp"])
        done = set(pd.to_datetime(prev["timestamp"]))
        log.info("Resuming: %d QHs already in backfill output.", len(done))

    # also skip QHs already captured in the live log
    live_log = RTE_DIR / "forecast_log.csv"
    if live_log.exists():
        live_df = pd.read_csv(live_log, parse_dates=["timestamp"])
        live_done = set(pd.to_datetime(live_df["timestamp"]))
        log.info("Skipping %d QHs already in live log.", len(live_done))
        done |= live_done
    else:
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["timestamp", "forecast_mwh", "actual_mwh"]).to_csv(OUTPUT_CSV, index=False)
        log.info("Created output: %s", OUTPUT_CSV)

    qh_range = pd.date_range(BACKFILL_START, BACKFILL_END, freq="15min")
    total    = len(qh_range)
    log.info("QHs to process: %d  (%s → %s)", total, BACKFILL_START, BACKFILL_END)

    buf: list[dict] = []
    n_ok = n_skip = n_err = 0

    for i, qh in enumerate(qh_range):
        # ── skip if already done ──────────────────────────────────────────────
        if qh in done:
            n_skip += 1
            continue

        # ── slice 27-h lag window (strictly before qh, no future leakage) ────
        win_start = qh - LAG_WINDOW
        raw = cache.loc[(cache.index >= win_start) & (cache.index < qh)].copy()

        if len(raw) < MIN_ROWS:
            log.warning("[%d/%d] %s: only %d rows in window, skip.", i + 1, total, qh, len(raw))
            n_err += 1
            continue

        # ── actual value (published — all QHs are in the past) ───────────────
        actual = float(cache.at[qh, "imbalance_mwh"]) if qh in cache.index else np.nan

        # ── forecast ──────────────────────────────────────────────────────────
        try:
            feat = build_features(raw)

            # ensure feat ends exactly at qh - 15min (trim any overshoot)
            expected_latest = qh - pd.Timedelta(minutes=15)
            if feat.index[-1] != expected_latest:
                feat = feat[feat.index <= expected_latest]
            if feat.empty:
                raise ValueError(f"No data at or before expected latest {expected_latest}")

            next_ts, forecast, _ = forecast_next(feat, model)

            buf.append({
                "timestamp":    next_ts,
                "forecast_mwh": round(float(forecast), 6),
                "actual_mwh":   round(actual, 4) if pd.notna(actual) else None,
            })
            n_ok += 1

        except Exception as exc:
            log.warning("[%d/%d] %s: forecast failed — %s", i + 1, total, qh, exc)
            n_err += 1
            continue

        # ── flush buffer ──────────────────────────────────────────────────────
        if len(buf) >= 50:
            pd.DataFrame(buf).to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
            buf.clear()

        # ── progress ──────────────────────────────────────────────────────────
        if (i + 1) % 100 == 0:
            log.info("Progress: %d/%d (%.1f%%) | ok=%d  skip=%d  err=%d",
                     i + 1, total, 100 * (i + 1) / total, n_ok, n_skip, n_err)

    # flush remainder
    if buf:
        pd.DataFrame(buf).to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

    log.info("Backfill complete: %d ok / %d skipped / %d errors → %s",
             n_ok, n_skip, n_err, OUTPUT_CSV)


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cache = build_or_load_cache()
    run_backfill(cache)
