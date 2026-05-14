#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download BE/NL/DE aFRR merit order for 2026-04-21 to 2026-04-28,
normalize to data_ve_2026 CSV format (same columns as existing files).
Does NOT write to forecast_log_full.csv — run backfill_price_range_2026.py after.
"""

import sys, time, shutil, logging, datetime as dt
from pathlib import Path

import pandas as pd
import requests

BASE_DIR     = Path("C:/Users/gusta/imbalance_optimisation")
ELIA_DIR     = BASE_DIR / "forecasters" / "elia_forecaster"
VE_DIR       = BASE_DIR / "data_ve_2026"
DOWNLOAD_DIR = Path.home() / "Downloads"
BRUSSELS     = "Europe/Brussels"

sys.path.insert(0, str(ELIA_DIR))
from afrr_merit_order import download_de_xlsx, download_be_csvs, download_nl_csv_for_day

DATES = [dt.date(2026, 4, d) for d in range(21, 29)]  # Apr 21-28 inclusive

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def normalize_be(inc_path: Path, dec_path: Path, out_path: Path, date: dt.date) -> bool:
    rows = []
    for direction, path in [("UP", inc_path), ("DOWN", dec_path)]:
        if not path.exists() or path.stat().st_size < 50:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "datetime" not in df.columns or "energybidmarginalprice" not in df.columns:
            continue
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df["timestamp"] = df["datetime"].dt.tz_convert(BRUSSELS).dt.tz_localize(None)
        if "balancingproduct" in df.columns:
            df = df[df["balancingproduct"] == "aFRR"]
        df["price"]    = pd.to_numeric(df["energybidmarginalprice"], errors="coerce")
        df["quantity"] = pd.to_numeric(df["energybidvolume"],        errors="coerce")
        df = df.dropna(subset=["price", "quantity", "timestamp"])
        df = df[df["timestamp"].dt.date == date]
        df["direction"] = direction
        df["country"]   = "BE"
        df["reserve"]   = "aFRR"
        rows.append(df[["timestamp", "direction", "price", "quantity", "country", "reserve"]])

    if not rows:
        return False
    pd.concat(rows, ignore_index=True).to_csv(out_path, index=False)
    return True


def normalize_nl(nl_path: Path, out_path: Path, date: dt.date) -> bool:
    if not nl_path.exists() or nl_path.stat().st_size < 50:
        return False
    try:
        df = pd.read_csv(nl_path)
    except Exception:
        return False

    time_col = next(
        (c for c in df.columns if "Start" in c and ("Loc" in c or "Local" in c)),
        next((c for c in df.columns if "Start" in c), None),
    )
    if time_col is None or "Capacity Threshold" not in df.columns:
        return False

    df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["timestamp"])
    df = df[df["timestamp"].dt.date == date]

    rows = []
    for ts, grp in df.groupby("timestamp"):
        grp = grp.sort_values("Capacity Threshold").reset_index(drop=True)
        grp["vol"] = grp["Capacity Threshold"] - grp["Capacity Threshold"].shift(1).fillna(0)
        for _, row in grp.iterrows():
            if "Price Up" in row.index and pd.notna(row["Price Up"]):
                rows.append({"timestamp": ts, "price": row["Price Up"],   "quantity": row["vol"],
                             "direction": "UP",   "country": "NL", "reserve": "aFRR"})
            if "Price Down" in row.index and pd.notna(row["Price Down"]):
                rows.append({"timestamp": ts, "price": row["Price Down"],  "quantity": row["vol"],
                             "direction": "DOWN", "country": "NL", "reserve": "aFRR"})
    if not rows:
        return False
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return True


def main():
    VE_DIR.mkdir(parents=True, exist_ok=True)

    for date in DATES:
        date_str = date.isoformat()
        log.info("=== %s ===", date_str)

        # ── DE ──────────────────────────────────────────────────────────────
        de_out = VE_DIR / f"_tmp_afrr_de_{date_str}.xlsx"
        if de_out.exists() and de_out.stat().st_size > 10_000:
            log.info("  DE: already cached.")
        else:
            try:
                dl_path = download_de_xlsx(str(DOWNLOAD_DIR), date_str)
                shutil.copy2(dl_path, de_out)
                log.info("  DE: downloaded (%d KB).", de_out.stat().st_size // 1024)
            except Exception as e:
                log.warning("  DE download failed: %s", e)

        # ── BE ──────────────────────────────────────────────────────────────
        be_out = VE_DIR / f"afrr_be_{date_str}.csv"
        if be_out.exists() and be_out.stat().st_size > 100:
            log.info("  BE: already cached.")
        else:
            inc_path = DOWNLOAD_DIR / f"afrr_incremental_bids_BE_{date_str}.csv"
            dec_path = DOWNLOAD_DIR / f"afrr_decremental_bids_BE_{date_str}.csv"
            try:
                download_be_csvs(date_str, inc_path, dec_path)
                ok = normalize_be(inc_path, dec_path, be_out, date)
                log.info("  BE: %s (%s).", "saved" if ok else "empty", be_out.name)
            except Exception as e:
                log.warning("  BE failed: %s", e)

        # ── NL ──────────────────────────────────────────────────────────────
        nl_out = VE_DIR / f"afrr_nl_{date_str}.csv"
        if nl_out.exists() and nl_out.stat().st_size > 100:
            log.info("  NL: already cached.")
        else:
            nl_dl = DOWNLOAD_DIR / f"tennet_merit_order_full_{date_str.replace('-', '')}.csv"
            try:
                download_nl_csv_for_day(date_str, nl_dl)
                ok = normalize_nl(nl_dl, nl_out, date)
                log.info("  NL: %s (%s).", "saved" if ok else "empty", nl_out.name)
            except Exception as e:
                log.warning("  NL failed: %s", e)

        time.sleep(2)

    log.info("All downloads done. Run backfill_price_range_2026.py to fill prices.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
