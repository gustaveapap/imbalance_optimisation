#!/usr/bin/env python3
"""
BE SOLAR 2026 — CONTINUOUS MODE
================================
APScheduler fires every 15 min at :00 :15 :30 :45 (Europe/Brussels).

Each tick:
  1. Fetch rolling 90-min SI 1min window from Elia ODS133 (always live)
  2. Run imbalance forecast for current QH (same model as elia_forecaster)
  3. Download EU aFRR + mFRR decremental bid stacks (BE/DE/NL),
     compute real-time metrics per QH:
       afrr_ratio_negative, afrr_prix_min, afrr_vol_avant_0
       mfrr_ratio_negative, mfrr_prix_min, mfrr_vol_avant_0
  4. Fetch ISP (ODS134), DA price (ENTSO-E A44), solar ENTSO-E, PVGIS
  5. Compute strategies S1 / S2 / S3 / S4
  6. Append one QH row to outputs/solar_be/forecast_log.csv

PATHS
-----
  Model   : forecasters/elia_forecaster/models/imbalance_forecaster_v1.joblib
  Log     : outputs/solar_be/forecast_log.csv
  Cache   : data/raw/solar_be/   ← all downloaded files land here

EU MERIT ORDER COUNTRY COVERAGE
---------------------------------
  aFRR: BE (Elia ODS163/164) + DE (regelleistung.net) + NL (TenNET)
  mFRR: BE (Elia, balancingproduct=mFRR on ODS163/164) + DE (regelleistung.net, productType=mFRR)
        NL not separated by product on TenNET → NL Dec stack shared with aFRR

NOTE: DE mFRR productType string on regelleistung.net may need updating if
      the API rejects "mFRR". Accepted values observed: "aFRR", "mFRR", "MRL".
"""

import time
import warnings
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import pytz
import requests
import xml.etree.ElementTree as ET
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR            = Path("C:/Users/gusta/imbalance_optimisation")
ELIA_FORECASTER_DIR = BASE_DIR / "forecasters/elia_forecaster"
MODEL_PATH          = ELIA_FORECASTER_DIR / "models/imbalance_forecaster_v1.joblib"
OUTPUT_DIR          = BASE_DIR / "outputs/solar_be"
DATA_DIR            = BASE_DIR / "data/raw/solar_be"
LOG_FILE            = OUTPUT_DIR / "forecast_log.csv"
PVGIS_CACHE         = DATA_DIR / "pvgis_be_2019.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

ENTSOE_TOKEN  = "dcc0c513-dab6-4add-82be-7b693f2b7fab"
ENTSOE_API    = "https://web-api.tp.entsoe.eu/api"
ENTSOE_SOLAR  = "https://transparency.entsoe.eu/generation/forecast/windAndSolar/solar/load"
ELIA_URL      = "https://external-elia.opendatasoft.com/api/records/1.0/search/"
ELIA_OPENDATA = "https://opendata.elia.be/api/records/1.0/search/"
PVGIS_URL     = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
TENNET_URL    = "https://api.tennet.eu/publications/v1/merit-order-list"
TENNET_KEY    = "18fe140e-12a2-446e-ab89-455d33709fac"
DE_API_URL    = "https://www.regelleistung.net/apps/cpp-publisher/api/v2/tenders/results/anonymous"

BE_DOMAIN     = "10YBE----------2"
BE_AREA       = "BZN|10YBE----------2"
BRUSSELS      = "Europe/Brussels"
TZ_BRUSSELS   = ZoneInfo(BRUSSELS)
PVGIS_LAT     = 50.5
PVGIS_LON     = 4.5

SEUIL_DA_MID  = 30          # EUR/MWh threshold for DA nomination logic
FEATURE_WIN   = 60          # minutes of SI history fed to the model
DEVIATION_MAX = 0.5284      # max deviation band for forecast_parc clipping
SI_WINDOW_MIN = 90          # minutes of rolling SI 1min window to fetch
BID_MAX_AGE   = 45          # minutes; re-download bid files if older

LOG_COLUMNS = [
    "timestamp",
    "forecast_volume", "forecast_direction",
    "afrr_ratio_negative", "afrr_prix_min", "afrr_vol_avant_0",
    "mfrr_ratio_negative", "mfrr_prix_min", "mfrr_vol_avant_0",
    "isp", "price_eur_mwh",
    "production_mw", "forecast_mw", "actual_mw", "forecast_parc",
    "curtail_v8_300", "curtail_v8_150",
    "s1_nomination", "s1_production", "s1_ecart",
    "s1_revenue_da", "s1_revenue_imb", "s1_total",
    "s2_nomination", "s2_production", "s2_ecart",
    "s2_revenue_da", "s2_revenue_imb", "s2_total",
    "s3_nomination", "s3_production", "s3_ecart",
    "s3_revenue_da", "s3_revenue_imb", "s3_total",
    "s4_nomination", "s4_production", "s4_ecart",
    "s4_revenue_da", "s4_revenue_imb", "s4_total",
]

# =============================================================================
# HELPERS
# =============================================================================

def safe_float(x):
    if x is None: return np.nan
    if isinstance(x, (int, float, np.floating)): return float(x)
    if isinstance(x, str) and x.strip() == "": return np.nan
    try: return float(x)
    except: return np.nan


def file_is_recent(path: Path, max_age_minutes: int = BID_MAX_AGE,
                   min_size_bytes: int = 100) -> bool:
    if not path.exists(): return False
    age_ok  = (time.time() - path.stat().st_mtime) < (max_age_minutes * 60)
    size_ok = path.stat().st_size >= min_size_bytes
    return age_ok and size_ok


def current_qh() -> pd.Timestamp:
    """Current QH start as naive Europe/Brussels timestamp."""
    now = datetime.now(TZ_BRUSSELS).replace(second=0, microsecond=0)
    return pd.Timestamp(now.replace(minute=(now.minute // 15) * 15, tzinfo=None))


def to_utc_str(ts: pd.Timestamp) -> str:
    """Naive Brussels local timestamp → UTC string for Elia ODS queries."""
    bx = pytz.timezone(BRUSSELS)
    return ts.tz_localize(bx).tz_convert(pytz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

# =============================================================================
# EU MERIT ORDER — DOWNLOADS
# =============================================================================

# ---------- BE aFRR / mFRR (Elia OpenData ODS163 inc / ODS164 dec) ----------

def _download_be_bids(date_str: str, product: str,
                      out_inc: Path, out_dec: Path) -> None:
    """
    Download BE incremental + decremental bid CSVs from Elia OpenData.
    product = 'aFRR' → datasets ods163/ods164
    product = 'mFRR' → same datasets with balancingproduct=mFRR filter
    (Elia stores both products in the same endpoint.)
    """
    DATASETS = [("incremental", "ods163", out_inc),
                ("decremental", "ods164", out_dec)]
    MAX_START = 10000   # Elia v1 API hard limit; 400 fires at start>=10000

    for direction, dataset, out_path in DATASETS:
        frames, start = [], 0
        while start < MAX_START:
            params = {
                "dataset":  dataset,
                "rows":     1000,
                "start":    start,
                "refine.date_start":       date_str,
                "refine.balancingproduct": product,
                "sort":     "datetime",
            }
            try:
                r = requests.get(ELIA_OPENDATA, params=params, timeout=60)
                if r.status_code == 429:
                    time.sleep(3)
                    r = requests.get(ELIA_OPENDATA, params=params, timeout=60)
                if r.status_code != 200:
                    if r.status_code == 400 and "10000" in r.text:
                        break
                    raise RuntimeError(f"HTTP {r.status_code}")
                records = r.json().get("records", [])
                if not records:
                    break
                df_page = pd.DataFrame([rec.get("fields", {}) for rec in records])
                if "datetime" in df_page.columns:
                    df_page["datetime"] = pd.to_datetime(
                        df_page["datetime"], errors="coerce", utc=True)
                frames.append(df_page)
                if len(records) < 1000:
                    break
                start += 1000
                time.sleep(0.6)
            except Exception as e:
                print(f"  WARNING BE {product} {direction}: {e}")
                break

        df_out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        df_out.to_csv(out_path, index=False)
        print(f"  [BE {product}] {direction}: {len(df_out)} rows")


# ---------- DE aFRR / mFRR (regelleistung.net API, no Selenium fallback) ----

def _download_de_xlsx(date_str: str, product: str, out_path: Path) -> bool:
    """
    Download DE aFRR or mFRR results from regelleistung.net direct API.
    product: 'aFRR' or 'mFRR'  (API also accepts 'MRL' for mFRR if 'mFRR' fails)
    Returns True on success.
    """
    for product_type in ([product, "MRL"] if product == "mFRR" else [product]):
        try:
            r = requests.get(DE_API_URL, params={
                "productType":   product_type,
                "market":        "ENERGY",
                "exportFormat":  "xlsx",
                "deliveryDate":  date_str,
            }, timeout=60)
            if r.status_code == 200 and r.content[:4] == b"PK\x03\x04":
                out_path.write_bytes(r.content)
                print(f"  [DE {product}] xlsx downloaded ({len(r.content):,} B)")
                return True
            print(f"  [DE {product}] productType={product_type}: HTTP {r.status_code}")
        except Exception as e:
            print(f"  WARNING DE {product} download: {e}")
    return False


def _parse_de_xlsx(xlsx_path: Path, date_str: str) -> pd.DataFrame:
    """
    Parse DE xlsx result file.  Returns DataFrame with columns
    [Timestamp, price, volume] for the DECREMENTAL (NEG) stack only,
    filtered to date_str.  Sign convention applied: NEG bids where
    payment direction is GRID_TO_PROVIDER get their price negated.
    """
    if not xlsx_path.exists() or xlsx_path.stat().st_size < 1000:
        return pd.DataFrame(columns=["Timestamp", "price", "volume"])
    try:
        xde = pd.read_excel(xlsx_path, sheet_name=0)
        xde["QH_IDX"] = (xde["PRODUCT"].astype(str)
                         .str.extract(r"(\d{3})").astype(float).astype("Int64"))
        xde = xde.dropna(subset=["QH_IDX"]).copy()
        xde["QH_IDX"] = xde["QH_IDX"].astype(int)

        # Resolve delivery date
        date_candidates = [c for c in xde.columns
                           if "DELIVERY" in c.upper() and "DATE" in c.upper()]
        if "DATE" in xde.columns and "DATE" not in date_candidates:
            date_candidates.append("DATE")
        delivery = None
        for c in date_candidates:
            try:
                col = pd.to_datetime(xde[c], errors="coerce")
                if col.notna().sum() > 0:
                    delivery = col
                    break
            except Exception:
                pass
        if delivery is None:
            delivery = pd.Series(pd.to_datetime(date_str), index=xde.index)
        xde["DeliveryDate"] = delivery.dt.normalize()
        xde["Timestamp"] = (xde["DeliveryDate"]
                            + pd.to_timedelta(xde["QH_IDX"] * 15, unit="m"))

        day = pd.Timestamp(date_str).date()
        xde_day = xde[xde["DeliveryDate"].dt.date == day].copy()
        if xde_day.empty:
            vc = xde["DeliveryDate"].dt.date.value_counts()
            if not vc.empty:
                xde_day = xde[xde["DeliveryDate"].dt.date == vc.index[0]].copy()

        # Keep only decremental (NEG) products
        dec = xde_day[xde_day["PRODUCT"].astype(str).str.startswith("NEG")].copy()
        if dec.empty:
            return pd.DataFrame(columns=["Timestamp", "price", "volume"])

        # Sign: if payment direction is GRID_TO_PROVIDER the price is negative
        mask = dec["ENERGY_PRICE_PAYMENT_DIRECTION"] == "GRID_TO_PROVIDER"
        dec.loc[mask, "ENERGY_PRICE_[EUR/MWh]"] *= -1
        dec = dec.rename(columns={
            "ENERGY_PRICE_[EUR/MWh]":      "price",
            "ALLOCATED_CAPACITY_[MW]":     "volume",
        })
        return dec[["Timestamp", "price", "volume"]].dropna()
    except Exception as e:
        print(f"  WARNING parse DE xlsx: {e}")
        return pd.DataFrame(columns=["Timestamp", "price", "volume"])


# ---------- NL aFRR (TenNET merit-order-list, shared for aFRR/mFRR) ---------

def _download_nl_csv(date_str: str, out_path: Path) -> None:
    """
    Download NL aFRR merit order from TenNET hour by hour (same as 2025 code).
    NL data is not split by product on this endpoint; used for aFRR Dec stack.
    """
    base   = datetime.strptime(date_str + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    now_lo = datetime.now(TZ_BRUSSELS)
    max_hour = now_lo.hour if date_str == now_lo.strftime("%Y-%m-%d") else 23
    headers  = {"apikey": TENNET_KEY, "Accept": "text/csv"}
    all_dfs  = []

    for hour in range(max_hour + 1):
        from_t = (base + timedelta(hours=hour)).strftime("%d-%m-%Y %H:%M:%S")
        to_t   = (base + timedelta(hours=hour+1)).strftime("%d-%m-%Y %H:%M:%S")
        for attempt in range(3):
            try:
                r = requests.get(TENNET_URL, headers=headers,
                                 params={"date_from": from_t, "date_to": to_t},
                                 timeout=30)
                if r.status_code == 200 and r.content.strip():
                    all_dfs.append(pd.read_csv(StringIO(r.content.decode("utf-8"))))
                    break
                elif r.status_code == 429:
                    time.sleep(6)
                else:
                    break
            except Exception as e:
                print(f"  WARNING NL {from_t}: {e}")
            time.sleep(6)

    if all_dfs:
        pd.concat(all_dfs, ignore_index=True).to_csv(out_path, index=False)
        print(f"  [NL] {sum(len(d) for d in all_dfs)} rows downloaded")
    else:
        pd.DataFrame().to_csv(out_path, index=False)
        print(f"  [NL] 0 rows for {date_str}")


def _parse_nl_dec(nl_path: Path, qh_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Extract the decremental (Price Down) stack for one QH from TenNET CSV.
    Returns DataFrame [price, volume].
    """
    if not nl_path.exists() or nl_path.stat().st_size < 100:
        return pd.DataFrame(columns=["price", "volume"])
    try:
        mo = pd.read_csv(nl_path)
        time_col = next(
            (c for c in mo.columns if "Start" in c and "Loc" in c or "Local" in c),
            None)
        if time_col is None:
            return pd.DataFrame(columns=["price", "volume"])
        mo["Start_dt"] = pd.to_datetime(mo[time_col], errors="coerce").dt.tz_localize(None)
        sel = mo[mo["Start_dt"] == qh_ts].copy()
        if sel.empty or "Price Down" not in sel.columns or "Capacity Threshold" not in sel.columns:
            return pd.DataFrame(columns=["price", "volume"])
        sel = sel.sort_values("Capacity Threshold").copy()
        sel["prev_cap"] = sel["Capacity Threshold"].shift(1).fillna(0)
        sel["volume"]   = sel["Capacity Threshold"] - sel["prev_cap"]
        return (sel.rename(columns={"Price Down": "price"})[["price", "volume"]]
                .dropna())
    except Exception as e:
        print(f"  WARNING parse NL Dec: {e}")
        return pd.DataFrame(columns=["price", "volume"])


# ---------- Build EU Dec stack for one QH + compute metrics -----------------

def _be_dec_for_qh(csv_path: Path, product: str,
                   qh_ts: pd.Timestamp) -> pd.DataFrame:
    """Extract BE [price, volume] for the decremental product at qh_ts."""
    if not csv_path.exists() or csv_path.stat().st_size < 100:
        return pd.DataFrame(columns=["price", "volume"])
    try:
        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df["ts_local"] = (df["datetime"]
                          .dt.tz_convert(BRUSSELS).dt.tz_localize(None))
        if "balancingproduct" in df.columns:
            df = df[df["balancingproduct"] == product]
        qh_df = df[df["ts_local"].dt.floor("15min") == qh_ts].copy()
        if qh_df.empty:
            return pd.DataFrame(columns=["price", "volume"])
        return (qh_df.rename(columns={
                    "energybidmarginalprice": "price",
                    "energybidvolume":        "volume",
                })[["price", "volume"]]
                .pipe(lambda d: d.assign(
                    price=pd.to_numeric(d["price"], errors="coerce"),
                    volume=pd.to_numeric(d["volume"], errors="coerce")))
                .dropna())
    except Exception as e:
        print(f"  WARNING BE Dec parse {csv_path.name}: {e}")
        return pd.DataFrame(columns=["price", "volume"])


def _stack_metrics(frames: list) -> tuple:
    """
    Build combined decremental stack from a list of [price, volume] DataFrames,
    sort descending by price, then compute:
      ratio_negative  — % of total volume (MW) priced below zero
      prix_min        — minimum bid price in the combined stack
      vol_avant_0     — cumulative volume when price first crosses below zero

    Returns (ratio_negative, prix_min, vol_avant_0) or (0.0, nan, 0.0) if empty.
    """
    valid = [f[["price", "volume"]].dropna() for f in frames if not f.empty]
    if not valid:
        return 0.0, np.nan, 0.0

    stack = pd.concat(valid, ignore_index=True)
    stack["price"]  = pd.to_numeric(stack["price"],  errors="coerce")
    stack["volume"] = pd.to_numeric(stack["volume"], errors="coerce").abs()
    stack = (stack.dropna()
             .sort_values("price", ascending=False)
             .reset_index(drop=True))

    if stack.empty:
        return 0.0, np.nan, 0.0

    stack["cum_vol"] = stack["volume"].cumsum()
    total_vol = stack["volume"].sum()
    if total_vol == 0:
        return 0.0, np.nan, 0.0

    neg_mask = stack["price"] < 0
    neg_vol  = stack.loc[neg_mask, "volume"].sum()
    ratio_negative = float(neg_vol / total_vol * 100)

    prix_min = float(stack["price"].min())

    # vol_avant_0: cumulative MW at the boundary where price first goes negative
    first_neg = stack[neg_mask]
    if first_neg.empty:
        vol_avant_0 = float(total_vol)
    elif first_neg.index[0] == 0:
        vol_avant_0 = 0.0
    else:
        prev_idx    = first_neg.index[0] - 1
        vol_avant_0 = float(stack.loc[prev_idx, "cum_vol"])

    return ratio_negative, prix_min, vol_avant_0


def fetch_eu_metrics(date_str: str, qh_ts: pd.Timestamp) -> dict:
    """
    Download/refresh EU aFRR + mFRR bid stacks and compute per-QH metrics
    from the combined decremental stack (BE + DE + NL).

    Files are cached in DATA_DIR with a BID_MAX_AGE-minute freshness check.
    Any per-product failure is isolated: the scheduler never stops.

    Returns dict with keys:
      afrr_ratio_negative, afrr_prix_min, afrr_vol_avant_0
      mfrr_ratio_negative, mfrr_prix_min, mfrr_vol_avant_0
    """
    result = {
        "afrr_ratio_negative": 0.0, "afrr_prix_min": np.nan, "afrr_vol_avant_0": 0.0,
        "mfrr_ratio_negative": 0.0, "mfrr_prix_min": np.nan, "mfrr_vol_avant_0": 0.0,
    }

    # ── aFRR ──────────────────────────────────────────────────────────────── #
    p_be_inc   = DATA_DIR / f"afrr_be_inc_{date_str}.csv"
    p_be_dec   = DATA_DIR / f"afrr_be_dec_{date_str}.csv"
    p_de_xlsx  = DATA_DIR / f"afrr_de_{date_str}.xlsx"
    p_nl_csv   = DATA_DIR / f"afrr_nl_{date_str}.csv"

    try:
        if not (file_is_recent(p_be_inc) and file_is_recent(p_be_dec)):
            _download_be_bids(date_str, "aFRR", p_be_inc, p_be_dec)
    except Exception as e:
        print(f"  WARNING aFRR BE download: {e}")

    try:
        if not file_is_recent(p_de_xlsx, min_size_bytes=1000):
            _download_de_xlsx(date_str, "aFRR", p_de_xlsx)
    except Exception as e:
        print(f"  WARNING aFRR DE download: {e}")

    try:
        if not file_is_recent(p_nl_csv):
            _download_nl_csv(date_str, p_nl_csv)
    except Exception as e:
        print(f"  WARNING aFRR NL download: {e}")

    try:
        be_dec   = _be_dec_for_qh(p_be_dec, "aFRR", qh_ts)
        de_dec   = (_parse_de_xlsx(p_de_xlsx, date_str)
                    .pipe(lambda df: df[df["Timestamp"] == qh_ts]
                          [["price", "volume"]].dropna()))
        nl_dec   = _parse_nl_dec(p_nl_csv, qh_ts)
        r, p, v  = _stack_metrics([be_dec, de_dec, nl_dec])
        result.update({"afrr_ratio_negative": r, "afrr_prix_min": p, "afrr_vol_avant_0": v})
    except Exception as e:
        print(f"  WARNING aFRR metrics: {e}")

    # ── mFRR ──────────────────────────────────────────────────────────────── #
    p_mfrr_inc  = DATA_DIR / f"mfrr_be_inc_{date_str}.csv"
    p_mfrr_dec  = DATA_DIR / f"mfrr_be_dec_{date_str}.csv"
    p_mfrr_de   = DATA_DIR / f"mfrr_de_{date_str}.xlsx"
    # NL TenNET merit-order-list does not separate aFRR/mFRR:
    # reuse the same NL Dec stack as a conservative approximation.

    try:
        if not (file_is_recent(p_mfrr_inc) and file_is_recent(p_mfrr_dec)):
            _download_be_bids(date_str, "mFRR", p_mfrr_inc, p_mfrr_dec)
    except Exception as e:
        print(f"  WARNING mFRR BE download: {e}")

    try:
        if not file_is_recent(p_mfrr_de, min_size_bytes=1000):
            _download_de_xlsx(date_str, "mFRR", p_mfrr_de)
    except Exception as e:
        print(f"  WARNING mFRR DE download: {e}")

    try:
        be_mfrr_dec = _be_dec_for_qh(p_mfrr_dec, "mFRR", qh_ts)
        de_mfrr_dec = (_parse_de_xlsx(p_mfrr_de, date_str)
                       .pipe(lambda df: df[df["Timestamp"] == qh_ts]
                             [["price", "volume"]].dropna()))
        nl_dec_reuse = _parse_nl_dec(p_nl_csv, qh_ts)
        r, p, v  = _stack_metrics([be_mfrr_dec, de_mfrr_dec, nl_dec_reuse])
        result.update({"mfrr_ratio_negative": r, "mfrr_prix_min": p, "mfrr_vol_avant_0": v})
    except Exception as e:
        print(f"  WARNING mFRR metrics: {e}")

    return result

# =============================================================================
# MARKET DATA FETCHERS
# =============================================================================

ODS169_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods169/records"
ODS162_URL = "https://external-elia.opendatasoft.com/api/records/1.0/search/"


def load_1min_window(end_ts: pd.Timestamp, minutes: int = SI_WINDOW_MIN) -> pd.DataFrame:
    """
    Fetch SI 1min data from Elia ODS169 (v2.1 API, real-time) for the rolling
    window [end_ts - minutes, end_ts].  Always live — no file cache.

    ODS133 (v1 API) has a ~24h publication lag and cannot be used for live
    forecasting.  ODS169 is the live equivalent used by elia_forecaster/app.py.
    """
    start_ts = end_ts - pd.Timedelta(minutes=minutes)
    s_utc    = to_utc_str(start_ts)
    e_utc    = to_utc_str(end_ts)
    all_recs, offset = [], 0
    limit = 100  # v2.1 API page size

    for attempt in range(5):
        try:
            r = requests.get(ODS169_URL, params={
                "where":    f'datetime >= "{s_utc}" and datetime <= "{e_utc}"',
                "limit":    limit,
                "offset":   offset,
                "order_by": "datetime",
            }, timeout=60)
            r.raise_for_status()
            page = r.json().get("results", [])
            if not page:
                break
            all_recs.extend(page)
            if len(page) < limit:
                break
            offset += limit
        except Exception as e:
            print(f"  WARNING SI 1min window attempt {attempt+1}: {e}")
            time.sleep(5)

    if not all_recs:
        return pd.DataFrame()

    df = pd.DataFrame(all_recs)
    df["datetime"] = (pd.to_datetime(df["datetime"], utc=True)
                      .dt.tz_convert(BRUSSELS).dt.tz_localize(None))
    df["actual_system_imbalance"] = pd.to_numeric(
        df["systemimbalance"], errors="coerce")
    return (df[["datetime", "actual_system_imbalance"]]
            .dropna().sort_values("datetime")
            .drop_duplicates("datetime").reset_index(drop=True))


def load_isp_live(qh_ts: pd.Timestamp) -> float:
    """
    Fetch ISP for a specific QH from Elia ODS162 (QH-resolution, published
    shortly after each QH ends).  Falls back to the previous QH if the current
    one is not yet available.  Returns 0.0 on complete failure.

    ODS162 at external-elia.opendatasoft.com is the live source used by
    elia_forecaster/app.py; ODS134 at the same endpoint is an alternative
    but has fewer records intraday.
    """
    for offset_min in [0, 15, 30]:          # try current QH, then step back
        check_ts = qh_ts - pd.Timedelta(minutes=offset_min)
        s_utc = to_utc_str(check_ts)
        e_utc = to_utc_str(check_ts + pd.Timedelta(minutes=15))
        try:
            r = requests.get(ODS162_URL, params={
                "dataset": "ods162",
                "q":       f"datetime:[{s_utc} TO {e_utc}]",
                "rows":    10, "sort": "datetime",
            }, timeout=30)
            r.raise_for_status()
            records = r.json().get("records", [])
            vals = [safe_float(rec["fields"].get("imbalanceprice")) for rec in records]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                if offset_min > 0:
                    print(f"  ISP: current QH unavailable, using QH-{offset_min}min")
                return float(np.mean(vals))
        except Exception as e:
            print(f"  WARNING ISP live (offset={offset_min}min): {e}")
    return 0.0


def load_da_live(date_str: str) -> pd.Series:
    """
    Fetch ENTSO-E Day-Ahead prices for BE (A44) for the full day.
    Cached per day in DATA_DIR since DA prices are fixed once D-1 published.
    Returns Series indexed by QH Timestamp -> EUR/MWh (empty on failure).
    """
    cache = DATA_DIR / f"da_{date_str}.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["timestamp"])
        return df.set_index("timestamp")["price_eur_mwh"]

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
                time.sleep(5 * (attempt + 1))
                continue
            root = ET.fromstring(r.content)
            rows = []
            for period in root.findall(".//{*}Period"):
                s_el = period.find(".//{*}start")
                r_el = period.find(".//{*}resolution")
                if s_el is None: continue
                s_dt = pd.to_datetime(s_el.text, utc=True)
                res  = r_el.text if r_el is not None else "PT60M"
                for p in period.findall(".//{*}Point"):
                    pos = int(p.find(".//{*}position").text)
                    px  = float(p.find(".//{*}price.amount").text)
                    off = (timedelta(minutes=(pos-1)*15) if res == "PT15M"
                           else timedelta(hours=(pos-1)))
                    ts  = (s_dt + off).tz_convert(BRUSSELS).tz_localize(None)
                    rows.append({"timestamp": ts, "price_eur_mwh": px})
            if not rows:
                return pd.Series(dtype=float)
            df = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
            df.to_csv(cache, index=False)
            return df.set_index("timestamp")["price_eur_mwh"]
        except Exception as e:
            print(f"  WARNING DA live: {e}")
            time.sleep(5 * (attempt + 1))
    return pd.Series(dtype=float)


def load_solar_day(date_str: str) -> pd.DataFrame:
    """
    Fetch ENTSO-E solar forecast + actual for BE for a full day.
    Cached per day in DATA_DIR.  Returns DataFrame with [timestamp, forecast_mw, actual_mw].
    """
    cache = DATA_DIR / f"solar_{date_str}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["timestamp"])

    date_from = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    date_to   = date_from + timedelta(days=1)
    for _ in range(2):
        try:
            r = requests.post(ENTSOE_SOLAR,
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
                time.sleep(3)
                continue
            rows = []
            for inst in r.json().get("instanceList", []):
                if inst.get("businessDimensionMap", {}).get("PRODUCTION_TYPE") != "B16":
                    continue
                for period in inst.get("curveData", {}).get("periodList", []):
                    st = period.get("timeInterval", {}).get("from")
                    if not st: continue
                    res  = period.get("resolution")
                    s_dt = pd.to_datetime(st, utc=True)
                    for pos_str, vals in period.get("pointMap", {}).items():
                        pos = int(pos_str)
                        off = (timedelta(minutes=pos*15) if res == "PT15M"
                               else timedelta(hours=pos))
                        ts  = (s_dt + off).tz_convert(BRUSSELS).tz_localize(None)
                        v   = vals if isinstance(vals, list) else []
                        rows.append({
                            "timestamp":   ts,
                            "forecast_mw": safe_float(v[0]) if len(v) > 0 else np.nan,
                            "actual_mw":   safe_float(v[3]) if len(v) > 3 else np.nan,
                        })
            if rows:
                df = (pd.DataFrame(rows)
                      .drop_duplicates("timestamp").sort_values("timestamp"))
                df.to_csv(cache, index=False)
                return df
            break
        except Exception as e:
            print(f"  WARNING solar day: {e}")
            time.sleep(3)
    return pd.DataFrame(columns=["timestamp", "forecast_mw", "actual_mw"])


def load_pvgis_be() -> pd.DataFrame:
    """
    Load/download PVGIS hourly production profile for BE (2019 reference year).
    Returns DataFrame [timestamp, production_mw] in MW/GWp. Cached permanently.
    """
    if PVGIS_CACHE.exists():
        return pd.read_csv(PVGIS_CACHE, parse_dates=["timestamp"])
    print("  PVGIS BE -> downloading...")
    r = requests.get(PVGIS_URL, params={
        "lat": PVGIS_LAT, "lon": PVGIS_LON,
        "startyear": 2019, "endyear": 2019,
        "pvcalculation": 1, "peakpower": 1000,
        "loss": 14, "angle": 35, "aspect": 0, "outputformat": "json",
    }, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["outputs"]["hourly"])
    df["timestamp"]     = (pd.to_datetime(df["time"], format="%Y%m%d:%H%M")
                           - pd.Timedelta(minutes=10))
    df["production_mw"] = df["P"] / 1_000_000
    out = df[["timestamp", "production_mw"]].sort_values("timestamp").reset_index(drop=True)
    out.to_csv(PVGIS_CACHE, index=False)
    print(f"  PVGIS: {len(out)} hourly points cached")
    return out


def get_pvgis_production(qh_ts: pd.Timestamp, df_pvgis: pd.DataFrame) -> float:
    """Look up MW/GWp for the given QH by day-of-year and hour from the PVGIS profile."""
    mask = ((df_pvgis["timestamp"].dt.dayofyear == qh_ts.dayofyear) &
            (df_pvgis["timestamp"].dt.hour == qh_ts.hour))
    sub = df_pvgis[mask]
    return float(sub.iloc[0]["production_mw"]) if not sub.empty else 0.0

# =============================================================================
# IMBALANCE FORECAST MODEL
# =============================================================================

class ImbalanceForecasterBE:
    """
    Thin wrapper around the sklearn pipeline in imbalance_forecaster_v1.joblib.
    Identical feature engineering to the 2025 simulation (BUG 1-5 all corrected):
      - floor("1min") alignment
      - interpolate → NaN check → ffill/bfill
      - feature window [target_qh - 65min, target_qh - 5min) = 60 samples
    """

    def __init__(self, path: Path = MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        self.model    = joblib.load(path)
        self._indexed = None
        print(f"  Model loaded: {path.name}")

    def prepare(self, df_1min: pd.DataFrame) -> None:
        """Index a 1-min SI window DataFrame for predict_qh."""
        df = df_1min.copy()
        if "datetime" not in df.columns and "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)
        df["datetime"] = df["datetime"].dt.floor("1min")
        df["actual_system_imbalance"] = pd.to_numeric(
            df["actual_system_imbalance"], errors="coerce")
        self._indexed = (df[["datetime", "actual_system_imbalance"]]
                         .dropna()
                         .drop_duplicates("datetime")
                         .sort_values("datetime")
                         .set_index("datetime"))

    def predict_qh(self, target_qh: pd.Timestamp):
        """
        Return forecast volume (MW) for target_qh, or None if data insufficient.
        DOWN = positive forecast (system is long, expects excess generation).
        """
        if self._indexed is None:
            return None
        feature_end   = target_qh - pd.Timedelta(minutes=5)
        feature_start = feature_end - pd.Timedelta(minutes=FEATURE_WIN)
        window = self._indexed.loc[
            (self._indexed.index >= feature_start) &
            (self._indexed.index <  feature_end),
            "actual_system_imbalance",
        ]
        if len(window) < 55:
            return None
        expected = pd.date_range(
            feature_start, feature_end, freq="1min", inclusive="left")
        series = window.reindex(expected).interpolate(limit=5)
        if series.isna().sum() > 10:
            return None
        features = series.ffill().bfill().values
        if len(features) != FEATURE_WIN:
            return None
        return float(self.model.predict(features.reshape(1, -1))[0])

# =============================================================================
# STRATEGIES
# =============================================================================

def _curtail_v8(fc_direction: str, fc_volume: float,
                mfrr: float, afrr: float, vol_seuil: float = 300) -> bool:
    """
    Curtailment signal for V8+ strategies.
    Vol is from the imbalance forecast (fc_volume, realistic at decision time).
    mFRR/aFRR ratios are from the EU decremental bid stacks (real-time).
    """
    if fc_direction != "DOWN":
        return False
    vol  = abs(fc_volume or 0)
    mfrr = mfrr or 0
    afrr = afrr or 0
    if vol > vol_seuil and mfrr > 75  and afrr > 65:  return True
    if (mfrr > 95 or afrr > 95) and vol > 50:          return True
    if mfrr > 75  and afrr > 75  and vol > 100:        return True
    return False


def _imb_revenue(ecart: pd.Series, isp: pd.Series) -> pd.Series:
    """
    Imbalance revenue (EUR per QH) per row.
    SHORT + ISP<0 → gain    LONG + ISP>0 → gain
    SHORT + ISP>0 → penalty LONG + ISP<0 → penalty
    Division by 4 converts MW × EUR/MWh → EUR per 15-min QH.
    """
    rev = pd.Series(0.0, index=ecart.index)
    m = (ecart > 0) & (isp < 0);  rev[m] =  ecart[m] * isp[m].abs() / 4
    m = (ecart < 0) & (isp > 0);  rev[m] =  ecart[m].abs() * isp[m] / 4
    m = (ecart > 0) & (isp > 0);  rev[m] = -ecart[m] * isp[m] / 4
    m = (ecart < 0) & (isp < 0);  rev[m] = -ecart[m].abs() * isp[m].abs() / 4
    return rev


def compute_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forecast_parc then S1/S2/S3/S4 columns in-place.
    Works on a single-row or multi-row DataFrame.
    """
    # forecast_parc: PVGIS production scaled by ENTSO-E fc/ac ratio, clipped
    fc_mw  = df["forecast_mw"].fillna(0)
    ac_mw  = df["actual_mw"].fillna(1).replace(0, 1)
    ratio  = (fc_mw / ac_mw).replace([np.inf, -np.inf], 1).fillna(1)
    raw    = df["production_mw"] * ratio
    lo     = df["production_mw"] * (1 - DEVIATION_MAX)
    hi     = df["production_mw"] * (1 + DEVIATION_MAX)
    df["forecast_parc"] = np.clip(raw, lo, hi).clip(upper=1.0)

    isp  = df["isp"].fillna(0)
    prod = df["production_mw"]

    # ── S1 — Baseline: nominate forecast_parc ────────────────────────────── #
    df["s1_nomination"]  = df["forecast_parc"]
    df["s1_production"]  = prod
    df["s1_ecart"]       = df["s1_nomination"] - df["s1_production"]
    df["s1_revenue_imb"] = _imb_revenue(df["s1_ecart"], isp)
    df["s1_revenue_da"]  = df["s1_nomination"] * df["price_eur_mwh"] / 4
    df["s1_total"]       = df["s1_revenue_da"] + df["s1_revenue_imb"]

    # DA nomination for S2/S4 (price-adaptive)
    def _nom_da(row):
        p, fc = row["price_eur_mwh"], row["forecast_parc"]
        if pd.isna(p) or pd.isna(fc): return 0.0
        if p < 0:             return 0.0
        if p < SEUIL_DA_MID:  return fc * 0.5
        return fc

    # Curtailment is only meaningful when there is actual solar production.
    # Firing the signal at night (prod ~ 0) creates no ecart and no revenue.
    solar_on = prod > 0.01   # boolean mask: daytime QHs with real production

    # ── S2 — V8+ 300 MW ──────────────────────────────────────────────────── #
    df["curtail_v8_300"] = df.apply(lambda r: _curtail_v8(
        r["forecast_direction"], r["forecast_volume"],
        r["mfrr_ratio_negative"], r["afrr_ratio_negative"], vol_seuil=300), axis=1)
    df["curtail_v8_300"] = df["curtail_v8_300"] & solar_on   # day-only gate
    df["s2_nomination"]  = df.apply(_nom_da, axis=1)
    df["s2_production"]  = np.where(df["curtail_v8_300"], 0.0, prod)
    df["s2_ecart"]       = df["s2_nomination"] - df["s2_production"]
    df["s2_revenue_imb"] = _imb_revenue(df["s2_ecart"], isp)
    df["s2_revenue_da"]  = df["s2_nomination"] * df["price_eur_mwh"] / 4
    df["s2_total"]       = df["s2_revenue_da"] + df["s2_revenue_imb"]

    # ── S3 — 50%+V8: fixed 50% nomination, same curtail as S2 ───────────── #
    df["s3_nomination"]  = df["forecast_parc"] * 0.5
    df["s3_production"]  = df["s2_production"]
    df["s3_ecart"]       = df["s3_nomination"] - df["s3_production"]
    df["s3_revenue_imb"] = _imb_revenue(df["s3_ecart"], isp)
    df["s3_revenue_da"]  = df["s3_nomination"] * df["price_eur_mwh"] / 4
    df["s3_total"]       = df["s3_revenue_da"] + df["s3_revenue_imb"]

    # ── S4 — V8+ 150 MW ──────────────────────────────────────────────────── #
    df["curtail_v8_150"] = df.apply(lambda r: _curtail_v8(
        r["forecast_direction"], r["forecast_volume"],
        r["mfrr_ratio_negative"], r["afrr_ratio_negative"], vol_seuil=150), axis=1)
    df["curtail_v8_150"] = df["curtail_v8_150"] & solar_on   # day-only gate
    df["s4_nomination"]  = df.apply(_nom_da, axis=1)
    df["s4_production"]  = np.where(df["curtail_v8_150"], 0.0, prod)
    df["s4_ecart"]       = df["s4_nomination"] - df["s4_production"]
    df["s4_revenue_imb"] = _imb_revenue(df["s4_ecart"], isp)
    df["s4_revenue_da"]  = df["s4_nomination"] * df["price_eur_mwh"] / 4
    df["s4_total"]       = df["s4_revenue_da"] + df["s4_revenue_imb"]

    return df

# =============================================================================
# LOG
# =============================================================================

def append_to_log(row: dict) -> None:
    """Append one QH result dict to LOG_FILE, creating file + header if needed."""
    df_row = pd.DataFrame([row])
    for col in LOG_COLUMNS:
        if col not in df_row.columns:
            df_row[col] = np.nan
    df_row[LOG_COLUMNS].to_csv(
        LOG_FILE, mode="a", index=False,
        header=not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0)

# =============================================================================
# SCHEDULER TICK
# =============================================================================

def run_qh_tick(forecaster: ImbalanceForecasterBE,
                df_pvgis: pd.DataFrame) -> None:
    """
    One 15-minute cycle.  Each step is independently try/except so a data
    outage in one source (e.g. DE xlsx) never crashes the scheduler.
    """
    qh_ts    = current_qh()
    date_str = qh_ts.strftime("%Y-%m-%d")
    now_str  = datetime.now(TZ_BRUSSELS).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now_str}] === QH {qh_ts} ===")

    row: dict = {"timestamp": qh_ts}

    # ── 1. Rolling SI 1min window → imbalance forecast ───────────────────── #
    try:
        df_1min = load_1min_window(qh_ts, minutes=SI_WINDOW_MIN)
        if df_1min.empty:
            print(f"  SKIP: no SI 1min data for window [{qh_ts - pd.Timedelta(minutes=SI_WINDOW_MIN)}, {qh_ts}]")
            return
        print(f"  SI 1min: {len(df_1min)} pts  "
              f"({df_1min['datetime'].min().strftime('%H:%M')} -> "
              f"{df_1min['datetime'].max().strftime('%H:%M')})")
        forecaster.prepare(df_1min)
        fc_vol = forecaster.predict_qh(qh_ts)
        if fc_vol is None:
            print("  Forecast: window too sparse (<55 valid pts)")
            row.update({"forecast_volume": np.nan, "forecast_direction": "UP"})
        else:
            fc_dir = "DOWN" if fc_vol > 0 else "UP"
            print(f"  Forecast: {fc_dir}  {fc_vol:+.1f} MW")
            row.update({"forecast_volume": fc_vol, "forecast_direction": fc_dir})
    except Exception as e:
        print(f"  ERROR forecast: {e}")
        row.update({"forecast_volume": np.nan, "forecast_direction": "UP"})

    # ── 2. EU aFRR/mFRR metrics from decremental bid stacks ──────────────── #
    try:
        metrics = fetch_eu_metrics(date_str, qh_ts)
        row.update(metrics)
        print(f"  aFRR  ratio={metrics['afrr_ratio_negative']:.1f}%  "
              f"vol0={metrics['afrr_vol_avant_0']:.0f}MW  "
              f"min={metrics['afrr_prix_min']:.1f} EUR")
        print(f"  mFRR  ratio={metrics['mfrr_ratio_negative']:.1f}%  "
              f"vol0={metrics['mfrr_vol_avant_0']:.0f}MW  "
              f"min={metrics['mfrr_prix_min']:.1f} EUR")
    except Exception as e:
        print(f"  ERROR metrics: {e}")
        row.update({
            "afrr_ratio_negative": 0.0, "afrr_prix_min": np.nan, "afrr_vol_avant_0": 0.0,
            "mfrr_ratio_negative": 0.0, "mfrr_prix_min": np.nan, "mfrr_vol_avant_0": 0.0,
        })

    # ── 3. ISP ────────────────────────────────────────────────────────────── #
    try:
        isp = load_isp_live(qh_ts)
        row["isp"] = isp
        print(f"  ISP:  {isp:+.2f} EUR/MWh")
    except Exception as e:
        print(f"  ERROR ISP: {e}")
        row["isp"] = 0.0

    # ── 4. DA price ───────────────────────────────────────────────────────── #
    try:
        da_series = load_da_live(date_str)
        da_price  = 0.0
        if not da_series.empty:
            # Latest QH index that is <= current QH (forward-fill within day)
            past = da_series.index[da_series.index <= qh_ts + pd.Timedelta(minutes=15)]
            if len(past):
                da_price = float(da_series[past[-1]])
        row["price_eur_mwh"] = da_price
        print(f"  DA:   {da_price:.2f} EUR/MWh")
    except Exception as e:
        print(f"  ERROR DA: {e}")
        row["price_eur_mwh"] = 0.0

    # ── 5. Solar production (PVGIS profile + ENTSO-E solar fc/ac) ─────────── #
    try:
        row["production_mw"] = get_pvgis_production(qh_ts, df_pvgis)
    except Exception as e:
        print(f"  ERROR PVGIS production: {e}")
        row["production_mw"] = 0.0

    try:
        df_solar = load_solar_day(date_str)
        if not df_solar.empty:
            sol_row = df_solar[df_solar["timestamp"].dt.floor("15min") == qh_ts]
            row["forecast_mw"] = safe_float(sol_row.iloc[0]["forecast_mw"]) if not sol_row.empty else 0.0
            row["actual_mw"]   = safe_float(sol_row.iloc[0]["actual_mw"])   if not sol_row.empty else 1.0
        else:
            row["forecast_mw"] = 0.0
            row["actual_mw"]   = 1.0
    except Exception as e:
        print(f"  ERROR solar ENTSO-E: {e}")
        row["forecast_mw"] = 0.0
        row["actual_mw"]   = 1.0

    # actual_mw = 0 would cause division by zero in forecast_parc; floor to 1
    if row.get("actual_mw", 1.0) == 0:
        row["actual_mw"] = 1.0

    # ── 6. Strategies ─────────────────────────────────────────────────────── #
    try:
        df_row = pd.DataFrame([row])
        df_row["production_mw"]       = df_row["production_mw"].fillna(0)
        df_row["isp"]                 = df_row["isp"].fillna(0)
        df_row["price_eur_mwh"]       = df_row["price_eur_mwh"].fillna(0)
        df_row["forecast_volume"]     = df_row["forecast_volume"].fillna(0)
        df_row["forecast_direction"]  = df_row["forecast_direction"].fillna("UP")
        df_row["mfrr_ratio_negative"] = df_row["mfrr_ratio_negative"].fillna(0)
        df_row["afrr_ratio_negative"] = df_row["afrr_ratio_negative"].fillna(0)
        df_row["forecast_mw"]         = df_row["forecast_mw"].fillna(0)
        df_row["actual_mw"]           = df_row["actual_mw"].fillna(1).replace(0, 1)
        df_row = compute_strategies(df_row)

        curtail = ("CURTAIL" if df_row["curtail_v8_300"].iloc[0] else "      ")
        print(f"  {curtail}  "
              f"S1={df_row['s1_total'].iloc[0]:+.3f}  "
              f"S2={df_row['s2_total'].iloc[0]:+.3f}  "
              f"S3={df_row['s3_total'].iloc[0]:+.3f}  "
              f"S4={df_row['s4_total'].iloc[0]:+.3f}  EUR/GWp")
        row = df_row.iloc[0].to_dict()
    except Exception as e:
        print(f"  ERROR strategies: {e}")

    # ── 7. Append to log ──────────────────────────────────────────────────── #
    try:
        append_to_log(row)
        print(f"  -> {LOG_FILE.name}")
    except Exception as e:
        print(f"  ERROR append: {e}")

# =============================================================================
# BATCH MODE — HISTORICAL SIMULATION 2026
# =============================================================================

ODS133_V1_URL = "https://opendata.elia.be/api/records/1.0/search/"
BATCH_OUTPUT  = OUTPUT_DIR / "simulation_be_2026.csv"


# ── 1. _file_cached ──────────────────────────────────────────────────────── #

def _file_cached(path: Path, min_size: int = 100) -> bool:
    """Historical file check: exists + meets minimum byte size, no age limit."""
    return path.exists() and path.stat().st_size >= min_size


# ── 2. load_1min_day_hist ────────────────────────────────────────────────── #

def load_1min_day_hist(date_str: str) -> pd.DataFrame:
    """
    Fetch full-day SI 1min from Elia ODS133 (v1 API, confirmed to hold
    2025-2026 data at 1440 pts/day).  Results cached per day in DATA_DIR.
    Returns DataFrame with columns [datetime, actual_system_imbalance].
    """
    cache = DATA_DIR / f"si_1min_{date_str}.csv"
    if _file_cached(cache, min_size=10_000):
        try:
            df = pd.read_csv(cache, parse_dates=["datetime"])
            df["actual_system_imbalance"] = pd.to_numeric(
                df["actual_system_imbalance"], errors="coerce")
            return df
        except Exception:
            cache.unlink(missing_ok=True)  # broken cache → re-download

    bx    = pytz.timezone(BRUSSELS)
    s_utc = (pd.Timestamp(date_str)
             .tz_localize(bx).tz_convert(pytz.UTC)
             .strftime("%Y-%m-%dT%H:%M:%SZ"))
    e_utc = ((pd.Timestamp(date_str) + pd.Timedelta(days=1))
             .tz_localize(bx).tz_convert(pytz.UTC)
             .strftime("%Y-%m-%dT%H:%M:%SZ"))

    all_recs, idx = [], 0
    while True:
        try:
            r = requests.get(ODS133_V1_URL, params={
                "dataset": "ods133",
                "q":       f"datetime:[{s_utc} TO {e_utc}]",
                "rows":    1000, "start": idx, "sort": "datetime",
            }, timeout=60)
            r.raise_for_status()
            data = r.json().get("records", [])
            if not data:
                break
            all_recs.extend(data)
            if len(data) < 1000:
                break
            idx += 1000
            time.sleep(0.2)
        except Exception as e:
            print(f"    WARNING SI 1min {date_str}: {e}")
            break

    if not all_recs:
        return pd.DataFrame()

    df = pd.json_normalize([rec["fields"] for rec in all_recs])
    df["datetime"] = (pd.to_datetime(df["datetime"], utc=True)
                      .dt.tz_convert(BRUSSELS).dt.tz_localize(None))
    df["actual_system_imbalance"] = pd.to_numeric(
        df["systemimbalance"], errors="coerce")
    out = (df[["datetime", "actual_system_imbalance"]]
           .dropna().sort_values("datetime").drop_duplicates("datetime")
           .reset_index(drop=True))
    out.to_csv(cache, index=False)
    return out


# ── 3. load_isp_day_hist ─────────────────────────────────────────────────── #

def load_isp_day_hist(date_str: str) -> pd.Series:
    """
    Fetch all QH ISP prices for a day from ODS134 (opendata.elia.be, v1 API).
    ODS162 (external-elia) does not cover 2026 historical dates in batch mode.
    ODS134 has ~96 QH rows/day with imbalanceprice column.
    Returns Series indexed by naive Brussels 15-min Timestamp -> EUR/MWh.
    """
    cache = DATA_DIR / f"isp_{date_str}.csv"
    if _file_cached(cache):
        try:
            df = pd.read_csv(cache, parse_dates=["datetime"])
            return df.set_index("datetime")["imbalanceprice"]
        except Exception:
            cache.unlink(missing_ok=True)  # broken cache → re-download

    bx    = pytz.timezone(BRUSSELS)
    s_utc = (pd.Timestamp(date_str)
             .tz_localize(bx).tz_convert(pytz.UTC)
             .strftime("%Y-%m-%dT%H:%M:%SZ"))
    e_utc = ((pd.Timestamp(date_str) + pd.Timedelta(days=1))
             .tz_localize(bx).tz_convert(pytz.UTC)
             .strftime("%Y-%m-%dT%H:%M:%SZ"))

    all_recs, offset = [], 0
    while True:
        try:
            r = requests.get(ODS133_V1_URL, params={
                "dataset": "ods134",
                "q":       f"datetime:[{s_utc} TO {e_utc}]",
                "rows":    1000, "start": offset, "sort": "datetime",
            }, timeout=60)
            r.raise_for_status()
            data = r.json().get("records", [])
            if not data:
                break
            all_recs.extend(data)
            if len(data) < 1000:
                break
            offset += 1000
        except Exception as e:
            print(f"    WARNING ISP {date_str}: {e}")
            time.sleep(3)
            break

    if not all_recs:
        return pd.Series(dtype=float)

    df = pd.json_normalize([rec["fields"] for rec in all_recs])
    df["datetime"] = (pd.to_datetime(df["datetime"], utc=True)
                      .dt.tz_convert(BRUSSELS).dt.tz_localize(None)
                      .dt.floor("15min"))
    df["imbalanceprice"] = pd.to_numeric(df["imbalanceprice"], errors="coerce")
    out = (df[["datetime", "imbalanceprice"]].dropna()
           .drop_duplicates("datetime").sort_values("datetime"))
    out.to_csv(cache, index=False)
    return out.set_index("datetime")["imbalanceprice"]


# ── 4. _download_nl_csv_batch ─────────────────────────────────────────────── #

def _download_nl_csv_batch(date_str: str, out_path: Path) -> None:
    """
    NL merit order for batch mode.
    Tries one full-day API request first (fast).  Falls back to hourly
    with 1 s sleep (vs 6 s in real-time) if the full-day response is
    empty or too small.
    """
    base    = datetime.strptime(date_str + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    headers = {"apikey": TENNET_KEY, "Accept": "text/csv"}

    # Attempt 1: full day in one request
    try:
        from_t = base.strftime("%d-%m-%Y %H:%M:%S")
        to_t   = (base + timedelta(hours=24)).strftime("%d-%m-%Y %H:%M:%S")
        r = requests.get(TENNET_URL, headers=headers,
                         params={"date_from": from_t, "date_to": to_t},
                         timeout=60)
        if r.status_code == 200 and len(r.content.strip()) > 500:
            df = pd.read_csv(StringIO(r.content.decode("utf-8")))
            if len(df) >= 24:
                df.to_csv(out_path, index=False)
                return
    except Exception:
        pass

    # Fallback: hourly, 1 s sleep
    all_dfs = []
    for hour in range(24):
        from_h = (base + timedelta(hours=hour)).strftime("%d-%m-%Y %H:%M:%S")
        to_h   = (base + timedelta(hours=hour+1)).strftime("%d-%m-%Y %H:%M:%S")
        for attempt in range(2):
            try:
                r = requests.get(TENNET_URL, headers=headers,
                                 params={"date_from": from_h, "date_to": to_h},
                                 timeout=20)
                if r.status_code == 200 and r.content.strip():
                    all_dfs.append(pd.read_csv(StringIO(r.content.decode("utf-8"))))
                    break
                if r.status_code == 429:
                    time.sleep(3)
            except Exception:
                pass
            time.sleep(1)

    df_out = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    df_out.to_csv(out_path, index=False)


# ── 5. Pre-parsers + _parse_nl_dec_from_df ──────────────────────────────── #

def _preparse_be_dec(csv_path: Path, product: str) -> pd.DataFrame:
    """
    Load a BE bid CSV once, normalise datetimes, filter to product.
    Returns DataFrame [ts_qh, price, volume] for fast per-QH filtering.
    """
    if not _file_cached(csv_path):
        return pd.DataFrame(columns=["ts_qh", "price", "volume"])
    try:
        df = pd.read_csv(csv_path)
        if "datetime" not in df.columns:
            return pd.DataFrame(columns=["ts_qh", "price", "volume"])
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df["ts_qh"]    = (df["datetime"]
                          .dt.tz_convert(BRUSSELS).dt.tz_localize(None)
                          .dt.floor("15min"))
        if "balancingproduct" in df.columns:
            df = df[df["balancingproduct"] == product]
        df["price"]  = pd.to_numeric(df["energybidmarginalprice"], errors="coerce")
        df["volume"] = pd.to_numeric(df["energybidvolume"],        errors="coerce")
        return df[["ts_qh", "price", "volume"]].dropna()
    except Exception as e:
        print(f"    WARNING _preparse_be_dec {csv_path.name}: {e}")
        return pd.DataFrame(columns=["ts_qh", "price", "volume"])


def _preparse_nl_dec(csv_path: Path) -> pd.DataFrame:
    """Load NL CSV once and normalise Start_dt for per-QH lookup."""
    if not _file_cached(csv_path):
        return pd.DataFrame()
    try:
        mo = pd.read_csv(csv_path)
        time_col = next(
            (c for c in mo.columns if "Start" in c and ("Loc" in c or "Local" in c)),
            None)
        if time_col is None:
            return pd.DataFrame()
        mo["Start_dt"] = (pd.to_datetime(mo[time_col], errors="coerce")
                          .dt.tz_localize(None))
        return mo
    except Exception:
        return pd.DataFrame()


def _parse_nl_dec_from_df(nl_df: pd.DataFrame,
                           qh_ts: pd.Timestamp) -> pd.DataFrame:
    """Filter pre-loaded NL DataFrame to one QH; return [price, volume]."""
    if nl_df.empty:
        return pd.DataFrame(columns=["price", "volume"])
    try:
        sel = nl_df[nl_df["Start_dt"] == qh_ts].copy()
        if sel.empty or "Price Down" not in sel.columns or \
                "Capacity Threshold" not in sel.columns:
            return pd.DataFrame(columns=["price", "volume"])
        sel = sel.sort_values("Capacity Threshold").copy()
        sel["prev_cap"] = sel["Capacity Threshold"].shift(1).fillna(0)
        sel["volume"]   = sel["Capacity Threshold"] - sel["prev_cap"]
        return (sel.rename(columns={"Price Down": "price"})[["price", "volume"]]
                .dropna())
    except Exception:
        return pd.DataFrame(columns=["price", "volume"])


# ── 6. compute_day_eu_metrics ────────────────────────────────────────────── #

def compute_day_eu_metrics(date_str: str) -> pd.DataFrame:
    """
    Download bid stacks for the day (all files cached in DATA_DIR), then
    compute aFRR + mFRR metrics for all 96 QHs.

    Each bid CSV/xlsx is loaded once; per-QH filtering happens in memory.
    Returns DataFrame [timestamp, afrr_*, mfrr_*] with 96 rows.
    """
    qh_times = pd.date_range(date_str, periods=96, freq="15min")

    # ── file paths ──────────────────────────────────────────────────────── #
    p_be_afrr_inc = DATA_DIR / f"afrr_be_inc_{date_str}.csv"
    p_be_afrr_dec = DATA_DIR / f"afrr_be_dec_{date_str}.csv"
    p_de_afrr     = DATA_DIR / f"afrr_de_{date_str}.xlsx"
    p_nl          = DATA_DIR / f"afrr_nl_{date_str}.csv"
    p_be_mfrr_inc = DATA_DIR / f"mfrr_be_inc_{date_str}.csv"
    p_be_mfrr_dec = DATA_DIR / f"mfrr_be_dec_{date_str}.csv"
    p_de_mfrr     = DATA_DIR / f"mfrr_de_{date_str}.xlsx"

    # ── download if not cached (no age limit for historical data) ────────── #
    if not (_file_cached(p_be_afrr_inc) and _file_cached(p_be_afrr_dec)):
        try:
            _download_be_bids(date_str, "aFRR", p_be_afrr_inc, p_be_afrr_dec)
        except Exception as e:
            print(f"    WARNING BE aFRR {date_str}: {e}")

    if not _file_cached(p_de_afrr, min_size=1000):
        try:
            _download_de_xlsx(date_str, "aFRR", p_de_afrr)
        except Exception as e:
            print(f"    WARNING DE aFRR {date_str}: {e}")

    if not _file_cached(p_nl):
        try:
            _download_nl_csv_batch(date_str, p_nl)
        except Exception as e:
            print(f"    WARNING NL {date_str}: {e}")

    if not (_file_cached(p_be_mfrr_inc) and _file_cached(p_be_mfrr_dec)):
        try:
            _download_be_bids(date_str, "mFRR", p_be_mfrr_inc, p_be_mfrr_dec)
        except Exception as e:
            print(f"    WARNING BE mFRR {date_str}: {e}")

    if not _file_cached(p_de_mfrr, min_size=1000):
        try:
            _download_de_xlsx(date_str, "mFRR", p_de_mfrr)
        except Exception as e:
            print(f"    WARNING DE mFRR {date_str}: {e}")

    # ── load each file once ──────────────────────────────────────────────── #
    be_afrr = _preparse_be_dec(p_be_afrr_dec, "aFRR")
    de_afrr = _parse_de_xlsx(p_de_afrr, date_str)   # returns [Timestamp, price, volume]
    nl_day  = _preparse_nl_dec(p_nl)
    be_mfrr = _preparse_be_dec(p_be_mfrr_dec, "mFRR")
    de_mfrr = _parse_de_xlsx(p_de_mfrr, date_str)

    # ── compute metrics per QH ───────────────────────────────────────────── #
    rows = []
    _empty = pd.DataFrame(columns=["price", "volume"])

    for qh_ts in qh_times:
        # aFRR dec stack
        be_a = (be_afrr[be_afrr["ts_qh"] == qh_ts][["price","volume"]]
                if not be_afrr.empty else _empty)
        de_a = (de_afrr[de_afrr["Timestamp"] == qh_ts][["price","volume"]]
                if not de_afrr.empty else _empty)
        nl_a = _parse_nl_dec_from_df(nl_day, qh_ts)
        ar, ap, av = _stack_metrics([be_a, de_a, nl_a])

        # mFRR dec stack (reuse NL — TenNET does not split by product)
        be_m = (be_mfrr[be_mfrr["ts_qh"] == qh_ts][["price","volume"]]
                if not be_mfrr.empty else _empty)
        de_m = (de_mfrr[de_mfrr["Timestamp"] == qh_ts][["price","volume"]]
                if not de_mfrr.empty else _empty)
        mr, mp, mv = _stack_metrics([be_m, de_m, nl_a])

        rows.append({
            "timestamp":           qh_ts,
            "afrr_ratio_negative": ar, "afrr_prix_min": ap, "afrr_vol_avant_0": av,
            "mfrr_ratio_negative": mr, "mfrr_prix_min": mp, "mfrr_vol_avant_0": mv,
        })

    return pd.DataFrame(rows)


# ── 7. run_batch ─────────────────────────────────────────────────────────── #

def run_batch(start_date: str = "2026-01-01",
              end_date:   str = "2026-04-11",
              output_csv: Path = BATCH_OUTPUT) -> None:
    """
    Historical batch simulation for [start_date, end_date].

    Per day:
      • SI 1min downloaded from ODS133 (all days first, forecaster prepared once)
      • Bid stacks downloaded per day (cached); metrics computed for all 96 QHs
      • ISP (ODS162), DA (ENTSO-E A44), solar (ENTSO-E) fetched per day
      • Strategies S1-S4 computed for all 96 QHs
    Results saved to output_csv; monthly summary printed.
    """
    dates     = [d.strftime("%Y-%m-%d")
                 for d in pd.date_range(start_date, end_date, freq="D")]
    n_total   = len(dates)

    print("=" * 65)
    print(f"BE SOLAR 2026 — BATCH  {start_date} -> {end_date}  ({n_total} days)")
    print("=" * 65)

    forecaster = ImbalanceForecasterBE(MODEL_PATH)

    # ── PVGIS (cached permanently) ──────────────────────────────────────── #
    print("\n[1/4] PVGIS profile...")
    df_pvgis = load_pvgis_be()
    print(f"  {len(df_pvgis)} hourly points")

    # ── SI 1min: download all days, prepare forecaster once ─────────────── #
    print(f"\n[2/4] SI 1min ({n_total} days)...")
    parts = []
    for i, ds in enumerate(dates, 1):
        print(f"  [{i:>3}/{n_total}] {ds}", end="  ", flush=True)
        df_d = load_1min_day_hist(ds)
        if df_d.empty:
            print("SKIP (no data)")
        else:
            print(f"{len(df_d)} pts")
            parts.append(df_d)

    if not parts:
        print("ERROR: No SI 1min data available for any day.")
        return

    df_1min_all = (pd.concat(parts, ignore_index=True)
                   .sort_values("datetime").drop_duplicates("datetime")
                   .reset_index(drop=True))
    print(f"  Total: {len(df_1min_all):,} pts  "
          f"({df_1min_all['datetime'].min().date()} -> "
          f"{df_1min_all['datetime'].max().date()})")
    forecaster.prepare(df_1min_all)

    # ── Per-day simulation ───────────────────────────────────────────────── #
    print(f"\n[3/4] Simulation ({n_total} days)...")
    all_results, skipped = [], []

    for i, ds in enumerate(dates, 1):
        print(f"  [{i:>3}/{n_total}] {ds} ", end="", flush=True)
        qh_times = pd.date_range(ds, periods=96, freq="15min")

        # --- Imbalance forecasts for all 96 QHs ---
        fc_rows = []
        for qh in qh_times:
            pred = forecaster.predict_qh(qh)
            fc_rows.append({
                "timestamp":          qh,
                "forecast_volume":    pred if pred is not None else np.nan,
                "forecast_direction": ("DOWN" if pred is not None and pred > 0
                                       else "UP"),
            })
        df_fc = pd.DataFrame(fc_rows)
        n_valid = df_fc["forecast_volume"].notna().sum()

        # --- EU metrics for all 96 QHs ---
        try:
            df_met = compute_day_eu_metrics(ds)
        except Exception as e:
            print(f"\n    WARNING metrics {ds}: {e}")
            df_met = pd.DataFrame({"timestamp": qh_times})

        # --- ISP ---
        isp_series = load_isp_day_hist(ds)

        # --- DA price ---
        da_series = load_da_live(ds)

        # --- Solar ENTSO-E ---
        df_solar = load_solar_day(ds)

        # --- Assemble day DataFrame ---
        full = pd.DataFrame({"timestamp": qh_times})
        full = full.merge(df_fc, on="timestamp", how="left")

        met_cols = ["timestamp","afrr_ratio_negative","afrr_prix_min","afrr_vol_avant_0",
                    "mfrr_ratio_negative","mfrr_prix_min","mfrr_vol_avant_0"]
        if not df_met.empty and "afrr_ratio_negative" in df_met.columns:
            full = full.merge(df_met[met_cols], on="timestamp", how="left")
        else:
            for c in met_cols[1:]:
                full[c] = np.nan

        # ISP: reindex QH timestamps, ffill within day
        if not isp_series.empty:
            isp_reindexed = isp_series.reindex(qh_times).ffill().fillna(0)
            full["isp"] = full["timestamp"].map(isp_reindexed.to_dict()).fillna(0)
        else:
            full["isp"] = 0.0

        # DA: reindex + ffill (hourly source → 15min QHs)
        if not da_series.empty:
            da_reindexed = da_series.reindex(qh_times).ffill().bfill().fillna(0)
            full["price_eur_mwh"] = full["timestamp"].map(
                da_reindexed.to_dict()).fillna(0)
        else:
            full["price_eur_mwh"] = 0.0

        # PVGIS production
        full["production_mw"] = full["timestamp"].map(
            lambda t: get_pvgis_production(t, df_pvgis))

        # Solar ENTSO-E
        if not df_solar.empty:
            df_solar_ts = df_solar.copy()
            df_solar_ts["timestamp"] = (pd.to_datetime(df_solar_ts["timestamp"])
                                        .dt.floor("15min"))
            full = full.merge(
                df_solar_ts[["timestamp","forecast_mw","actual_mw"]],
                on="timestamp", how="left")
        else:
            full["forecast_mw"] = np.nan
            full["actual_mw"]   = np.nan

        # Fill safe defaults
        full["production_mw"]       = full["production_mw"].fillna(0)
        full["isp"]                 = full["isp"].fillna(0)
        full["price_eur_mwh"]       = full["price_eur_mwh"].ffill().fillna(0)
        full["forecast_volume"]     = full["forecast_volume"].fillna(0)
        full["forecast_direction"]  = full["forecast_direction"].fillna("UP")
        full["mfrr_ratio_negative"] = full["mfrr_ratio_negative"].fillna(0)
        full["afrr_ratio_negative"] = full["afrr_ratio_negative"].fillna(0)
        full["afrr_vol_avant_0"]    = full["afrr_vol_avant_0"].fillna(0)
        full["mfrr_vol_avant_0"]    = full["mfrr_vol_avant_0"].fillna(0)
        full["forecast_mw"]         = full["forecast_mw"].fillna(0)
        full["actual_mw"]           = full["actual_mw"].fillna(1).replace(0, 1)
        full["date"] = ds

        full = compute_strategies(full)
        all_results.append(full)

        s1 = full["s1_total"].sum()
        s2 = full["s2_total"].sum()
        n_curt = int(full["curtail_v8_300"].sum())
        print(f"fc={n_valid}/96  S1={s1:+.1f}  S2={s2:+.1f}  curt={n_curt}")

    if not all_results:
        print("No results.")
        return

    # ── Save CSV ─────────────────────────────────────────────────────────── #
    print(f"\n[4/4] Saving results...")
    df_all = pd.concat(all_results, ignore_index=True)

    # Ensure all LOG_COLUMNS present
    for col in LOG_COLUMNS:
        if col not in df_all.columns:
            df_all[col] = np.nan
    extra = [c for c in df_all.columns if c not in LOG_COLUMNS and c != "date"]
    save_cols = LOG_COLUMNS + ["date"] + extra
    df_all[[c for c in save_cols if c in df_all.columns]].to_csv(
        output_csv, index=False)
    print(f"  {output_csv}")
    print(f"  {len(df_all):,} QH rows  |  {len(all_results)} days")
    if skipped:
        print(f"  Skipped: {len(skipped)} days")

    print_batch_summary(df_all)


# ── 8. print_batch_summary ───────────────────────────────────────────────── #

def print_batch_summary(df_all: pd.DataFrame) -> None:
    """Monthly summary table, curtailment counts, and per-strategy breakdown."""
    df = df_all.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["month"]     = df["timestamp"].dt.to_period("M")
    df["month_str"] = df["month"].dt.strftime("%b %Y")

    monthly = (df.groupby("month_str", sort=False).agg(
        s1_da   =("s1_revenue_da",  "sum"), s1_imb=("s1_revenue_imb","sum"),
        s1_total=("s1_total",       "sum"),
        s2_da   =("s2_revenue_da",  "sum"), s2_imb=("s2_revenue_imb","sum"),
        s2_total=("s2_total",       "sum"),
        s3_da   =("s3_revenue_da",  "sum"), s3_imb=("s3_revenue_imb","sum"),
        s3_total=("s3_total",       "sum"),
        s4_da   =("s4_revenue_da",  "sum"), s4_imb=("s4_revenue_imb","sum"),
        s4_total=("s4_total",       "sum"),
        n_days      =("date",           "nunique"),
        n_curt_300  =("curtail_v8_300", "sum"),
        n_curt_150  =("curtail_v8_150", "sum"),
    ).reset_index())

    # Sort chronologically
    period_map = (df.drop_duplicates("month_str")
                  .set_index("month_str")["month"])
    monthly["_sort"] = monthly["month_str"].map(period_map)
    monthly = (monthly.sort_values("_sort")
               .drop(columns="_sort").reset_index(drop=True))

    print("\n" + "=" * 70)
    print("RESULTS BE SOLAR 2026")
    print("=" * 70)
    print(f"  {'Month':<10} {'S1':>9} {'S2':>9} {'S3':>9} {'S4':>9}  Best  Days")
    print("  " + "-" * 64)
    for _, row in monthly.iterrows():
        s1, s2 = row["s1_total"], row["s2_total"]
        s3, s4 = row["s3_total"], row["s4_total"]
        best   = ["S1","S2","S3","S4"][[s1,s2,s3,s4].index(max(s1,s2,s3,s4))]
        print(f"  {row['month_str']:<10} "
              f"{s1:>8,.0f}  {s2:>8,.0f}  {s3:>8,.0f}  {s4:>8,.0f}  "
              f"{best:>4}  {int(row['n_days']):>2}d")
    print("  " + "-" * 64)

    s1t = monthly["s1_total"].sum()
    s2t = monthly["s2_total"].sum()
    s3t = monthly["s3_total"].sum()
    s4t = monthly["s4_total"].sum()
    best_t = ["S1","S2","S3","S4"][[s1t,s2t,s3t,s4t].index(max(s1t,s2t,s3t,s4t))]
    print(f"  {'TOTAL':<10} "
          f"{s1t:>8,.0f}  {s2t:>8,.0f}  {s3t:>8,.0f}  {s4t:>8,.0f}  {best_t:>4}")

    print(f"\n  {'Strategy':<16} {'Total':>10} {'DA rev':>10} {'Imb rev':>10}")
    print("  " + "-" * 50)
    for key, name in [("s1","S1 Baseline"),("s2","S2 V8+ 300MW"),
                      ("s3","S3 50%+V8"), ("s4","S4 V8+ 150MW")]:
        t  = monthly[f"{key}_total"].sum()
        da = monthly[f"{key}_da"].sum()
        ib = monthly[f"{key}_imb"].sum()
        print(f"  {name:<16} {t:>10,.1f} {da:>10,.1f} {ib:>10,.1f}")

    n_total   = len(df)
    nc300 = monthly["n_curt_300"].sum()
    nc150 = monthly["n_curt_150"].sum()
    print(f"\n  Curtailments S2 (>300MW): {nc300:.0f} QH  "
          f"({nc300/n_total*100:.1f}% of all QHs)")
    print(f"  Curtailments S4 (>150MW): {nc150:.0f} QH  "
          f"({nc150/n_total*100:.1f}% of all QHs)")
    print(f"\n  QHs computed : {n_total:,}  |  Days: {df['date'].nunique()}")
    print("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="BE Solar 2026 optimizer")
    parser.add_argument("--batch", action="store_true",
                        help="Run historical batch simulation 2026-01-01 -> 2026-04-11")
    parser.add_argument("--start", default="2026-01-01",
                        help="Batch start date (YYYY-MM-DD)")
    parser.add_argument("--end",   default="2026-04-11",
                        help="Batch end date (YYYY-MM-DD)")
    parser.add_argument("--out",   default=str(BATCH_OUTPUT),
                        help="Batch output CSV path")
    args = parser.parse_args()

    if args.batch:
        run_batch(args.start, args.end, Path(args.out))
        return

    # ── Continuous scheduler mode ─────────────────────────────────────────── #
    print("=" * 65)
    print("BE SOLAR 2026 — CONTINUOUS MODE")
    print("=" * 65)
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Log     : {LOG_FILE}")
    print(f"  Cache   : {DATA_DIR}")
    print()

    forecaster = ImbalanceForecasterBE(MODEL_PATH)

    print("  Loading PVGIS profile...")
    df_pvgis = load_pvgis_be()
    print(f"  PVGIS: {len(df_pvgis)} hourly points")

    print("\n  Running initial tick...")
    run_qh_tick(forecaster, df_pvgis)

    scheduler = BlockingScheduler(timezone=str(TZ_BRUSSELS))
    scheduler.add_job(
        func=lambda: run_qh_tick(forecaster, df_pvgis),
        trigger=CronTrigger(minute="0,15,30,45", timezone=str(TZ_BRUSSELS)),
        id="solar_be_qh",
        name="BE Solar 2026 QH tick",
        misfire_grace_time=120,
        coalesce=True,
    )

    print("\n  Scheduler started — firing every 15 min at :00 :15 :30 :45")
    print("  Ctrl-C to stop.\n")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\nScheduler stopped.")


if __name__ == "__main__":
    main()
