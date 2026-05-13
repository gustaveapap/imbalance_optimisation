#!/usr/bin/env python3
"""
FR SOLAR — CONTINUOUS SCHEDULER
================================
APScheduler fires every 15 min at :00 :15 :30 :45 (Europe/Paris).

Each tick:
  1. Fetch rolling 6h of FR QH imbalance from RTE API (OAuth2)
  2. Run imbalance forecast for current QH (FR model, 96-lag QH features)
  3. Download EU aFRR + mFRR decremental bid stacks (BE/DE/NL),
     compute per-QH metrics: afrr_ratio_negative, afrr_prix_min, afrr_vol_avant_0
                              mfrr_ratio_negative, mfrr_prix_min, mfrr_vol_avant_0
  4. Fetch DA price (ENTSO-E A44 FR), solar (ENTSO-E FR B16), PVGIS
  5. Compute strategies S1 / S2 / S3
  6. Append one QH row to outputs/solar_fr/forecast_log.csv

PATHS
-----
  Model   : forecasters/rte_forecaster/artifacts/fr_imbalance_full_model.pkl
  Log     : outputs/solar_fr/forecast_log.csv
  Cache   : data/raw/solar_fr/

EU MERIT ORDER COUNTRY COVERAGE
---------------------------------
  aFRR: BE (Elia ODS163/164) + DE (regelleistung.net) + NL (TenNET)
  mFRR: BE + DE + NL (TenNET dec stack shared with aFRR)
"""

import base64
import time
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import pytz
import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR    = Path("C:/Users/gusta/imbalance_optimisation")
MODEL_PATH  = BASE_DIR / "forecasters/rte_forecaster/artifacts/fr_imbalance_full_model.pkl"
OUTPUT_DIR  = BASE_DIR / "outputs/solar_fr"
DATA_DIR    = BASE_DIR / "data/raw/solar_fr"
LOG_FILE    = OUTPUT_DIR / "forecast_log.csv"
PVGIS_CACHE = DATA_DIR / "pvgis_fr_2019.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

CLIENT_ID     = "bdc03388-6c93-46f6-adbf-1a77d5b89684"
CLIENT_SECRET = "da352352-63f8-42ce-9a34-0624f7560a72"
ENTSOE_TOKEN  = "dcc0c513-dab6-4add-82be-7b693f2b7fab"
ENTSOE_API    = "https://web-api.tp.entsoe.eu/api"
ENTSOE_SOLAR  = "https://transparency.entsoe.eu/generation/forecast/windAndSolar/solar/load"
TENNET_URL    = "https://api.tennet.eu/publications/v1/merit-order-list"
TENNET_KEY    = "18fe140e-12a2-446e-ab89-455d33709fac"
DE_API_URL    = "https://www.regelleistung.net/apps/cpp-publisher/api/v2/tenders/results/anonymous"
ELIA_OPENDATA = "https://opendata.elia.be/api/records/1.0/search/"

FR_DOMAIN  = "10YFR-RTE------C"
FR_AREA    = "BZN|10YFR-RTE------C"
PARIS      = "Europe/Paris"
TZ_PARIS   = ZoneInfo(PARIS)
PVGIS_LAT  = 46.8
PVGIS_LON  = 1.7

MAX_LAG       = 96          # FR model feature window (QH lags)
FEATURE_DIM   = MAX_LAG + 6 # 96 lags + 6 cyclical time features
BID_MAX_AGE   = 45          # minutes; re-download bid files if older
SEUIL_DA_MID  = 30          # EUR/MWh — DA nomination threshold
DEVIATION_MAX = 0.5284

LOG_COLUMNS = [
    "timestamp",
    "forecast_volume", "forecast_direction",
    "afrr_ratio_negative", "afrr_prix_min", "afrr_vol_avant_0",
    "mfrr_ratio_negative", "mfrr_prix_min", "mfrr_vol_avant_0",
    "isp_positif", "isp_negatif", "price_eur_mwh",
    "production_mw", "forecast_mw", "actual_mw", "forecast_parc",
    "s1_nomination", "s1_production", "s1_ecart",
    "s1_revenue_da", "s1_revenue_imb", "s1_total",
    "s2_nomination", "s2_production", "s2_ecart", "s2_curtail",
    "s2_revenue_da", "s2_revenue_imb", "s2_total",
    "s3_nomination", "s3_production", "s3_ecart", "s3_curtail",
    "s3_revenue_da", "s3_revenue_imb", "s3_total",
]

# =============================================================================
# HELPERS
# =============================================================================

def safe_float(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    if isinstance(x, str) and x.strip() == "":
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def file_is_recent(path: Path, max_age_minutes: int = BID_MAX_AGE,
                   min_size_bytes: int = 100) -> bool:
    if not path.exists():
        return False
    age_ok  = (time.time() - path.stat().st_mtime) < (max_age_minutes * 60)
    size_ok = path.stat().st_size >= min_size_bytes
    return age_ok and size_ok


def current_qh() -> pd.Timestamp:
    now = datetime.now(TZ_PARIS).replace(second=0, microsecond=0)
    return pd.Timestamp(now.replace(minute=(now.minute // 15) * 15, tzinfo=None))


def to_paris_str(ts: pd.Timestamp) -> str:
    px = pytz.timezone(PARIS)
    return ts.tz_localize(px).tz_convert(pytz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

# =============================================================================
# RTE OAUTH2 TOKEN
# =============================================================================

_rte_token: dict = {"value": None, "expires_at": 0.0}


def get_rte_token() -> str:
    if time.time() < _rte_token["expires_at"] - 60:
        return _rte_token["value"]
    basic = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    r = requests.post(
        "https://digital.iservices.rte-france.com/token/oauth/",
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Authorization": f"Basic {basic}"},
        data={"grant_type": "client_credentials"}, timeout=30)
    r.raise_for_status()
    data = r.json()
    _rte_token["value"]      = data["access_token"]
    _rte_token["expires_at"] = time.time() + data.get("expires_in", 1800)
    return _rte_token["value"]

# =============================================================================
# EU MERIT ORDER — DOWNLOADS  (BE + DE + NL, identical to solar_be_scheduler)
# =============================================================================

def _download_be_bids(date_str: str, product: str,
                      out_inc: Path, out_dec: Path) -> None:
    DATASETS = [("incremental", "ods163", out_inc),
                ("decremental", "ods164", out_dec)]
    MAX_START = 10000
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


def _download_de_xlsx(date_str: str, product: str, out_path: Path) -> bool:
    for product_type in ([product, "MRL"] if product == "mFRR" else [product]):
        try:
            r = requests.get(DE_API_URL, params={
                "productType":  product_type,
                "market":       "ENERGY",
                "exportFormat": "xlsx",
                "deliveryDate": date_str,
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
    if not xlsx_path.exists() or xlsx_path.stat().st_size < 1000:
        return pd.DataFrame(columns=["Timestamp", "price", "volume"])
    try:
        xde = pd.read_excel(xlsx_path, sheet_name=0)
        xde["QH_IDX"] = (xde["PRODUCT"].astype(str)
                         .str.extract(r"(\d{3})").astype(float).astype("Int64"))
        xde = xde.dropna(subset=["QH_IDX"]).copy()
        xde["QH_IDX"] = xde["QH_IDX"].astype(int)
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
        dec = xde_day[xde_day["PRODUCT"].astype(str).str.startswith("NEG")].copy()
        if dec.empty:
            return pd.DataFrame(columns=["Timestamp", "price", "volume"])
        mask = dec["ENERGY_PRICE_PAYMENT_DIRECTION"] == "GRID_TO_PROVIDER"
        dec.loc[mask, "ENERGY_PRICE_[EUR/MWh]"] *= -1
        dec = dec.rename(columns={
            "ENERGY_PRICE_[EUR/MWh]":  "price",
            "ALLOCATED_CAPACITY_[MW]": "volume",
        })
        return dec[["Timestamp", "price", "volume"]].dropna()
    except Exception as e:
        print(f"  WARNING parse DE xlsx: {e}")
        return pd.DataFrame(columns=["Timestamp", "price", "volume"])


def _download_nl_csv(date_str: str, out_path: Path) -> None:
    base    = datetime.strptime(date_str + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    now_lo  = datetime.now(TZ_PARIS)
    max_hour = now_lo.hour if date_str == now_lo.strftime("%Y-%m-%d") else 23
    headers = {"apikey": TENNET_KEY, "Accept": "text/csv"}
    all_dfs = []
    for hour in range(max_hour + 1):
        from_t = (base + timedelta(hours=hour)).strftime("%d-%m-%Y %H:%M:%S")
        to_t   = (base + timedelta(hours=hour + 1)).strftime("%d-%m-%Y %H:%M:%S")
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
    if not nl_path.exists() or nl_path.stat().st_size < 100:
        return pd.DataFrame(columns=["price", "volume"])
    try:
        mo = pd.read_csv(nl_path)
        time_col = next(
            (c for c in mo.columns if "Start" in c and ("Loc" in c or "Local" in c)),
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


def _be_dec_for_qh(csv_path: Path, product: str,
                   qh_ts: pd.Timestamp) -> pd.DataFrame:
    if not csv_path.exists() or csv_path.stat().st_size < 100:
        return pd.DataFrame(columns=["price", "volume"])
    try:
        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df["ts_local"] = (df["datetime"]
                          .dt.tz_convert("Europe/Brussels").dt.tz_localize(None))
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
    first_neg = stack[neg_mask]
    if first_neg.empty:
        vol_avant_0 = float(total_vol)
    elif first_neg.index[0] == 0:
        vol_avant_0 = 0.0
    else:
        vol_avant_0 = float(stack.loc[first_neg.index[0] - 1, "cum_vol"])
    return ratio_negative, prix_min, vol_avant_0


def fetch_eu_metrics(date_str: str, qh_ts: pd.Timestamp) -> dict:
    result = {
        "afrr_ratio_negative": 0.0, "afrr_prix_min": np.nan, "afrr_vol_avant_0": 0.0,
        "mfrr_ratio_negative": 0.0, "mfrr_prix_min": np.nan, "mfrr_vol_avant_0": 0.0,
    }
    p_be_inc  = DATA_DIR / f"afrr_be_inc_{date_str}.csv"
    p_be_dec  = DATA_DIR / f"afrr_be_dec_{date_str}.csv"
    p_de_xlsx = DATA_DIR / f"afrr_de_{date_str}.xlsx"
    p_nl_csv  = DATA_DIR / f"afrr_nl_{date_str}.csv"

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
        be_dec = _be_dec_for_qh(p_be_dec, "aFRR", qh_ts)
        de_dec = (_parse_de_xlsx(p_de_xlsx, date_str)
                  .pipe(lambda df: df[df["Timestamp"] == qh_ts][["price", "volume"]].dropna()))
        nl_dec = _parse_nl_dec(p_nl_csv, qh_ts)
        r, p, v = _stack_metrics([be_dec, de_dec, nl_dec])
        result.update({"afrr_ratio_negative": r, "afrr_prix_min": p, "afrr_vol_avant_0": v})
    except Exception as e:
        print(f"  WARNING aFRR metrics: {e}")

    p_mfrr_inc = DATA_DIR / f"mfrr_be_inc_{date_str}.csv"
    p_mfrr_dec = DATA_DIR / f"mfrr_be_dec_{date_str}.csv"
    p_mfrr_de  = DATA_DIR / f"mfrr_de_{date_str}.xlsx"

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
        be_mfrr_dec  = _be_dec_for_qh(p_mfrr_dec, "mFRR", qh_ts)
        de_mfrr_dec  = (_parse_de_xlsx(p_mfrr_de, date_str)
                        .pipe(lambda df: df[df["Timestamp"] == qh_ts][["price", "volume"]].dropna()))
        nl_dec_reuse = _parse_nl_dec(p_nl_csv, qh_ts)
        r, p, v = _stack_metrics([be_mfrr_dec, de_mfrr_dec, nl_dec_reuse])
        result.update({"mfrr_ratio_negative": r, "mfrr_prix_min": p, "mfrr_vol_avant_0": v})
    except Exception as e:
        print(f"  WARNING mFRR metrics: {e}")

    return result

# =============================================================================
# FR MARKET DATA FETCHERS
# =============================================================================

def load_imbalance_fr_live(hours_back: int = 8) -> pd.DataFrame:
    """
    Fetch last N hours of FR QH imbalance (volume + settlement prices) from RTE API.
    Returns DataFrame [timestamp, volume, prix_positif, prix_negatif].
    """
    now   = datetime.now(pytz.timezone(PARIS))
    start = now - timedelta(hours=hours_back)
    token = get_rte_token()
    r = requests.get(
        "https://digital.iservices.rte-france.com/open_api/balancing_energy/v5/imbalance_data",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params={
            "start_date": start.strftime("%Y-%m-%dT%H:%M:%S+01:00"),
            "end_date":   now.strftime("%Y-%m-%dT%H:%M:%S+01:00"),
        }, timeout=30)
    r.raise_for_status()
    rows = []
    for item in r.json().get("imbalance_data", []):
        for v in item.get("values", []):
            rows.append({
                "timestamp":    pd.to_datetime(v.get("start_date")),
                "volume":       safe_float(v.get("imbalance")),
                "prix_positif": safe_float(v.get("positive_imbalance_settlement_price")),
                "prix_negatif": safe_float(v.get("negative_imbalance_settlement_price")),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = (pd.to_datetime(df["timestamp"], utc=True)
                       .dt.tz_convert(PARIS).dt.tz_localize(None).dt.floor("15min"))
    return (df.sort_values("timestamp")
            .drop_duplicates("timestamp")
            .reset_index(drop=True))


def load_da_live(date_str: str) -> pd.Series:
    """ENTSO-E A44 Day-Ahead price for FR. Cached per day."""
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
                "in_Domain": FR_DOMAIN, "out_Domain": FR_DOMAIN,
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
                if s_el is None:
                    continue
                s_dt = pd.to_datetime(s_el.text, utc=True)
                res  = r_el.text if r_el is not None else "PT60M"
                for pt in period.findall(".//{*}Point"):
                    pos = int(pt.find(".//{*}position").text)
                    px  = float(pt.find(".//{*}price.amount").text)
                    off = (timedelta(minutes=(pos - 1) * 15) if res == "PT15M"
                           else timedelta(hours=(pos - 1)))
                    ts  = (s_dt + off).tz_convert(PARIS).tz_localize(None)
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
    """ENTSO-E solar forecast + actual for FR. Cached per day."""
    cache = DATA_DIR / f"solar_{date_str}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["timestamp"])
    date_from = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    date_to   = date_from + timedelta(days=1)
    for _ in range(2):
        try:
            r = requests.post(
                ENTSOE_SOLAR,
                json={
                    "dateTimeRange": {
                        "from": date_from.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                        "to":   date_to.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    },
                    "areaList": [FR_AREA], "timeZone": "CET",
                    "sorterList": [], "filterMap": {},
                },
                headers={"accept": "application/json",
                         "content-type": "application/json; charset=utf-8"},
                timeout=90)
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
                        ts  = (s_dt + off).tz_convert(PARIS).tz_localize(None)
                        v   = vals if isinstance(vals, list) else []
                        rows.append({
                            "timestamp":   ts,
                            "forecast_mw": safe_float(v[2]) if len(v) > 2 else np.nan,
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


def load_pvgis_fr() -> pd.DataFrame:
    """PVGIS hourly production profile for FR (2019 reference, center of France)."""
    if PVGIS_CACHE.exists():
        return pd.read_csv(PVGIS_CACHE, parse_dates=["timestamp"])
    print("  PVGIS FR -> downloading...")
    r = requests.get(
        "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc",
        params={
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
    mask = ((df_pvgis["timestamp"].dt.dayofyear == qh_ts.dayofyear) &
            (df_pvgis["timestamp"].dt.hour == qh_ts.hour))
    sub = df_pvgis[mask]
    return float(sub.iloc[0]["production_mw"]) if not sub.empty else 0.0

# =============================================================================
# FR IMBALANCE FORECASTER  (96-lag QH features, same as forecast_fr in test_logic_fr)
# =============================================================================

class ImbalanceForecasterFR:
    """
    Thin wrapper around the RTE sklearn model (fr_imbalance_full_model.pkl).
    Features: 96 reversed QH lags + 6 cyclical time encodings = 102 total.
    """

    def __init__(self, path: Path = MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        self.model = joblib.load(path)
        self._hist = pd.DataFrame(columns=["timestamp", "volume"])
        print(f"  Model loaded: {path.name}")

    def update(self, df_new: pd.DataFrame) -> None:
        """Merge new QH rows into rolling history. Keeps last 200 QH (50h)."""
        if df_new.empty or "timestamp" not in df_new.columns:
            return
        df_new = df_new[["timestamp", "volume"]].dropna()
        self._hist = (pd.concat([self._hist, df_new], ignore_index=True)
                      .drop_duplicates("timestamp")
                      .sort_values("timestamp")
                      .tail(200)
                      .reset_index(drop=True))

    def predict_qh(self, target_qh: pd.Timestamp):
        """Return forecast volume (MW) or None if history insufficient."""
        hist = self._hist[self._hist["timestamp"] < target_qh].tail(MAX_LAG)
        if len(hist) < MAX_LAG:
            return None
        exp    = pd.date_range(end=target_qh - pd.Timedelta(minutes=15),
                               periods=MAX_LAG, freq="15min")
        series = (hist.set_index("timestamp")["volume"]
                  .reindex(exp).interpolate(limit=2).ffill().bfill())
        if series.isna().any():
            return None
        slot  = target_qh.hour * 4 + target_qh.minute // 15
        feats = list(series.values[::-1]) + [
            np.sin(2 * np.pi * slot / 96),         np.cos(2 * np.pi * slot / 96),
            np.sin(2 * np.pi * target_qh.dayofweek / 7),
            np.cos(2 * np.pi * target_qh.dayofweek / 7),
            np.sin(2 * np.pi * target_qh.month / 12),
            np.cos(2 * np.pi * target_qh.month / 12),
        ]
        return float(self.model.predict(np.array([feats], dtype=np.float32))[0])

# =============================================================================
# STRATEGIES  (compute_strategies_v2 — VE-proven FR signals, from test_logic_fr)
# =============================================================================

def _rev_imb_fr(ecart: np.ndarray,
                ppos: np.ndarray, pneg: np.ndarray) -> np.ndarray:
    """FR-specific imbalance revenue. Two settlement prices: positif + negatif."""
    rev = np.zeros(len(ecart), dtype=float)
    m = (ecart > 0) & (pneg < 0); rev[m] =  ecart[m] * np.abs(pneg[m]) / 4
    m = (ecart < 0) & (ppos > 0); rev[m] =  np.abs(ecart[m]) * ppos[m] / 4
    m = (ecart > 0) & (ppos > 0); rev[m] = -ecart[m] * ppos[m] / 4
    m = (ecart < 0) & (pneg < 0); rev[m] = -np.abs(ecart[m]) * np.abs(pneg[m]) / 4
    return rev


def compute_strategies_fr(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy()
    prod = df["production_mw"].fillna(0)
    fc_mw = df["forecast_mw"].fillna(0)
    ac_mw = df["actual_mw"].fillna(1).replace(0, 1)
    ratio = (fc_mw / ac_mw).replace([np.inf, -np.inf], 1).fillna(1)
    raw   = prod * ratio
    lo    = prod * (1 - DEVIATION_MAX)
    hi    = prod * (1 + DEVIATION_MAX)
    df["forecast_parc"] = np.clip(raw, lo, hi).clip(upper=1.0)

    fp       = df["forecast_parc"].fillna(0)
    da       = df["price_eur_mwh"].fillna(0)
    vol      = df["forecast_volume"].abs().fillna(0)
    fdir     = df["forecast_direction"].fillna("UP")
    afrr_neg = df["afrr_ratio_negative"].fillna(0)
    mfrr_neg = df["mfrr_ratio_negative"].fillna(0)
    ppos     = df["isp_positif"].fillna(0).values
    pneg     = df["isp_negatif"].fillna(0).values

    solar_on = prod > 0.01

    # Curtailment signals (VE-proven FR triggers)
    sd_prudent = (
        solar_on & (fdir == "DOWN") & (
            ((vol > 300) & (mfrr_neg > 75) & (afrr_neg > 65))
            | (((mfrr_neg > 95) | (afrr_neg > 95)) & (vol > 50))
            | ((mfrr_neg > 75) & (afrr_neg > 75) & (vol > 100))
        )
    )
    sd_ultra = (
        solar_on & (fdir == "DOWN") & (
            (mfrr_neg > 80) | ((mfrr_neg > 60) & (vol > 250))
        )
    )

    # S1 — Baseline: nominate forecast_parc 100%
    df["s1_nomination"]  = fp.values
    df["s1_production"]  = prod.values
    df["s1_ecart"]       = (fp - prod).values
    df["s1_revenue_imb"] = _rev_imb_fr(df["s1_ecart"].values, ppos, pneg)
    df["s1_revenue_da"]  = (fp * da / 4).values
    df["s1_total"]       = df["s1_revenue_da"] + df["s1_revenue_imb"]

    # S2 — Active: DA-adapt (0 if DA<0) + curtail ULTRA signal
    curtail_s2       = sd_ultra & solar_on
    nom_s2           = np.where(da < 0, 0.0, fp)
    df["s2_curtail"]     = curtail_s2
    df["s2_nomination"]  = nom_s2
    df["s2_production"]  = np.where(curtail_s2, 0.0, prod)
    df["s2_ecart"]       = nom_s2 - df["s2_production"]
    df["s2_revenue_imb"] = _rev_imb_fr(df["s2_ecart"].values, ppos, pneg)
    df["s2_revenue_da"]  = nom_s2 * da / 4
    df["s2_total"]       = df["s2_revenue_da"] + df["s2_revenue_imb"]

    # S3 — 70% fixed nomination + curtail PRUDENT signal
    curtail_s3       = sd_prudent & solar_on
    nom_s3           = (fp * 0.7).values
    df["s3_curtail"]     = curtail_s3
    df["s3_nomination"]  = nom_s3
    df["s3_production"]  = np.where(curtail_s3, 0.0, prod)
    df["s3_ecart"]       = nom_s3 - df["s3_production"]
    df["s3_revenue_imb"] = _rev_imb_fr(df["s3_ecart"].values, ppos, pneg)
    df["s3_revenue_da"]  = nom_s3 * da / 4
    df["s3_total"]       = df["s3_revenue_da"] + df["s3_revenue_imb"]

    return df

# =============================================================================
# LOG
# =============================================================================

def append_to_log(row: dict) -> None:
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

def run_qh_tick(forecaster: ImbalanceForecasterFR,
                df_pvgis: pd.DataFrame) -> None:
    """One 15-minute cycle. Every step is independently try/except."""
    qh_ts    = current_qh()
    date_str = qh_ts.strftime("%Y-%m-%d")
    now_str  = datetime.now(TZ_PARIS).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now_str}] === QH {qh_ts} ===")

    row: dict = {"timestamp": qh_ts}

    # ── 1. RTE imbalance → update history → forecast ─────────────────────── #
    try:
        df_imb = load_imbalance_fr_live(hours_back=8)
        if df_imb.empty:
            print("  SKIP: no FR imbalance data from RTE")
            return
        forecaster.update(df_imb[["timestamp", "volume"]])
        print(f"  RTE imb: {len(df_imb)} QH  "
              f"({df_imb['timestamp'].min().strftime('%H:%M')} -> "
              f"{df_imb['timestamp'].max().strftime('%H:%M')})")

        # ISP from most recently completed QH in RTE response
        past_imb = df_imb[df_imb["timestamp"] < qh_ts]
        if not past_imb.empty:
            last_row       = past_imb.iloc[-1]
            row["isp_positif"] = safe_float(last_row.get("prix_positif", np.nan))
            row["isp_negatif"] = safe_float(last_row.get("prix_negatif", np.nan))
            print(f"  ISP+: {row['isp_positif']:+.2f}  ISP-: {row['isp_negatif']:+.2f}  EUR/MWh")
        else:
            row["isp_positif"] = 0.0
            row["isp_negatif"] = 0.0

        fc_vol = forecaster.predict_qh(qh_ts)
        if fc_vol is None:
            print("  Forecast: history too short (<96 QH)")
            row.update({"forecast_volume": np.nan, "forecast_direction": "UP"})
        else:
            fc_dir = "DOWN" if fc_vol > 0 else "UP"
            print(f"  Forecast: {fc_dir}  {fc_vol:+.1f} MW")
            row.update({"forecast_volume": fc_vol, "forecast_direction": fc_dir})
    except Exception as e:
        print(f"  ERROR imbalance/forecast: {e}")
        row.update({"forecast_volume": np.nan, "forecast_direction": "UP",
                    "isp_positif": 0.0, "isp_negatif": 0.0})

    # ── 2. EU aFRR / mFRR metrics ─────────────────────────────────────────── #
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

    # ── 3. DA price (ENTSO-E FR) ──────────────────────────────────────────── #
    try:
        da_series = load_da_live(date_str)
        da_price  = 0.0
        if not da_series.empty:
            past = da_series.index[da_series.index <= qh_ts + pd.Timedelta(minutes=15)]
            if len(past):
                da_price = float(da_series[past[-1]])
        row["price_eur_mwh"] = da_price
        print(f"  DA:   {da_price:.2f} EUR/MWh")
    except Exception as e:
        print(f"  ERROR DA: {e}")
        row["price_eur_mwh"] = 0.0

    # ── 4. Solar production (PVGIS profile + ENTSO-E solar fc/ac) ────────── #
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

    if row.get("actual_mw", 1.0) == 0:
        row["actual_mw"] = 1.0

    # ── 5. Strategies ─────────────────────────────────────────────────────── #
    try:
        df_row = pd.DataFrame([row])
        for col, default in [
            ("production_mw", 0), ("price_eur_mwh", 0), ("isp_positif", 0),
            ("isp_negatif", 0), ("forecast_volume", 0), ("forecast_mw", 0),
            ("mfrr_ratio_negative", 0), ("afrr_ratio_negative", 0),
        ]:
            df_row[col] = df_row.get(col, pd.Series([default])).fillna(default)
        df_row["forecast_direction"] = df_row.get(
            "forecast_direction", pd.Series(["UP"])).fillna("UP")
        df_row["actual_mw"] = df_row.get(
            "actual_mw", pd.Series([1.0])).fillna(1.0).replace(0, 1.0)

        df_row = compute_strategies_fr(df_row)

        curt2 = "CURTAIL" if df_row["s2_curtail"].iloc[0] else "      "
        curt3 = "CURTAIL" if df_row["s3_curtail"].iloc[0] else "      "
        print(f"  S1={df_row['s1_total'].iloc[0]:+.3f}  "
              f"S2[{curt2}]={df_row['s2_total'].iloc[0]:+.3f}  "
              f"S3[{curt3}]={df_row['s3_total'].iloc[0]:+.3f}  EUR/GWp")
        row = df_row.iloc[0].to_dict()
    except Exception as e:
        print(f"  ERROR strategies: {e}")

    # ── 6. Append to log ──────────────────────────────────────────────────── #
    try:
        append_to_log(row)
        print(f"  -> {LOG_FILE.name}")
    except Exception as e:
        print(f"  ERROR append: {e}")

# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    print("=" * 65)
    print("FR SOLAR 2026 — CONTINUOUS MODE")
    print("=" * 65)
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Log     : {LOG_FILE}")
    print(f"  Cache   : {DATA_DIR}")
    print()

    print("  Auth RTE...")
    get_rte_token()
    print("  OK")

    forecaster = ImbalanceForecasterFR(MODEL_PATH)

    print("  Loading PVGIS profile (FR)...")
    df_pvgis = load_pvgis_fr()
    print(f"  PVGIS: {len(df_pvgis)} hourly points")

    print("\n  Loading initial FR imbalance history...")
    try:
        df_init = load_imbalance_fr_live(hours_back=30)
        forecaster.update(df_init[["timestamp", "volume"]])
        print(f"  {len(forecaster._hist)} QH loaded into forecaster history")
    except Exception as e:
        print(f"  WARNING initial history: {e}")

    print("\n  Running initial tick...")
    run_qh_tick(forecaster, df_pvgis)

    scheduler = BlockingScheduler(timezone=str(TZ_PARIS))
    scheduler.add_job(
        func=lambda: run_qh_tick(forecaster, df_pvgis),
        trigger=CronTrigger(minute="0,15,30,45", timezone=str(TZ_PARIS)),
        id="solar_fr_qh",
        name="FR Solar 2026 QH tick",
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
