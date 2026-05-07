"""
=============================================================================
PIPELINE VE 2025 BELGIQUE — OPTIMISATION CHARGE INTELLIGENTE

Changes vs FR version:
  - MODEL_PATH : forecasters/elia_forecaster/models/imbalance_forecaster_v1.joblib
  - FORECAST_FILE : forecast_volume_ve_2025_be.csv
  - SAVE_DIR : simulation_ve_2025_be/
  - intelligent_download : imbalance_be from ODS133+ODS134 (no RTE OAuth)
  - build_features : 60 x 1-min SI in [qh-65min, qh-5min) — same as extract_features()
  - ISP : directly from isp column in imbalance_be files
  - DA prices : ENTSO-E zone 10YBE----------2
=============================================================================
"""

import time, warnings
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

API_KEY_TENNET      = "18fe140e-12a2-446e-ab89-455d33709fac"
ENTSOE_TOKEN        = "dcc0c513-dab6-4add-82be-7b693f2b7fab"
ODS133_URL          = "https://external-elia.opendatasoft.com/api/records/1.0/search/"
ODS134_URL          = "https://external-elia.opendatasoft.com/api/records/1.0/search/"
BASE_URL_ENTSOE_API = "https://web-api.tp.entsoe.eu/api"

BASE_DIR      = Path("C:/Users/gusta/imbalance_optimisation")
MODEL_PATH    = BASE_DIR / "forecasters/elia_forecaster/models/imbalance_forecaster_v1.joblib"
FORECAST_FILE = BASE_DIR / "forecast_volume_ve_2025_be.csv"
DATA_DIR      = BASE_DIR / "data_ve_2025"
SAVE_DIR      = BASE_DIR / "simulation_ve_2025_be"
SI_CACHE_DIR  = BASE_DIR / "data" / "backfill_elia"

DATA_DIR.mkdir(exist_ok=True)
SAVE_DIR.mkdir(exist_ok=True)
SI_CACHE_DIR.mkdir(parents=True, exist_ok=True)

BRUSSELS         = "Europe/Brussels"
CHECKPOINT_EVERY = 100

EV_CAPACITY_KWH          = 66.0
CHARGER_POWER_PER_QH_KWH = 11.0 / 4
CONSUMPTION_KWH_PER_KM   = 66.0 / 440.0
INITIAL_SOC_KWH          = 33.0
FORCED_THRESHOLD_WEEKDAY = 22.0
FORCED_THRESHOLD_WEEKEND = 33.0
FORCED_HOURS_WEEKDAY     = [4, 5, 16, 17]
FORCED_HOURS_WEEKEND     = [7, 8, 9]

# =============================================================================
# UTILITAIRES
# =============================================================================

def safe_float(x):
    if x is None: return np.nan
    if isinstance(x, (int, float, np.floating)): return float(x)
    if isinstance(x, str) and x.strip() == "": return np.nan
    try: return float(x)
    except: return np.nan

def normalize_ts(s):
    s = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_localize(None)
    except: pass
    return s.dt.floor("15min")

def create_full_qh_range(date_str):
    d  = datetime.strptime(date_str, "%Y-%m-%d")
    ts = pd.date_range(start=d, end=d + timedelta(days=1), freq="15min", inclusive="left")
    return pd.DataFrame({"timestamp": ts})

def dedup(df, value_cols):
    if df.empty: return df
    df = df.copy()
    df["timestamp"] = normalize_ts(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    keep = ["timestamp"] + [c for c in value_cols if c in df.columns]
    return df[keep].groupby("timestamp", as_index=False).mean(numeric_only=True) \
                   .sort_values("timestamp").reset_index(drop=True)

def ensure_qh(df, date_str, kind):
    if df is None or df.empty or "timestamp" not in df.columns: return pd.DataFrame()
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    df = df.copy()
    df["timestamp"] = normalize_ts(df["timestamp"])
    df = df[df["timestamp"].dt.date == date_obj].copy()
    if df.empty: return pd.DataFrame()
    if kind == "da": vcols = ["price_eur_mwh"]
    else:            vcols = [c for c in df.columns if c != "timestamp"]
    df    = dedup(df, vcols)
    full  = create_full_qh_range(date_str).set_index("timestamp")
    df_qh = df.set_index("timestamp").reindex(full.index)
    if kind == "da": df_qh[vcols] = df_qh[vcols].ffill()
    elif kind in ("imb", "metrics"): pass
    else: df_qh[vcols] = df_qh[vcols].interpolate(method="linear", limit_direction="both")
    return df_qh.reset_index()

def retry(fn, n=3, delay=2):
    err = None
    for i in range(n):
        try: return fn()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            err = e
            if i < n - 1: time.sleep(delay)
    raise err

# =============================================================================
# DOWNLOAD ODS133 (SI 1-MIN)
# =============================================================================

def _download_ods133_day(date_str, out_path):
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
            except Exception as e:
                print(f"    ODS133 {date_str} attempt {attempt+1}/3: {e}")
                time.sleep(5 * (attempt + 1))
        else:
            return 0
        rows = []
        for rec in records:
            f      = rec.get("fields", {})
            dt_raw = f.get("datetime", "")
            si_val = f.get("systemimbalance")
            if dt_raw and si_val is not None:
                try:
                    ts = pd.to_datetime(dt_raw, utc=True).tz_convert(BRUSSELS).tz_localize(None)
                    rows.append({"datetime": ts, "actual_system_imbalance": float(si_val)})
                except: pass
        if rows: frames.append(pd.DataFrame(rows))
        if len(records) < 1000: break
        start += 1000
    if not frames: return 0
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates("datetime")
            .sort_values("datetime")
            .reset_index(drop=True))
    df = df[df["datetime"].dt.date == d]
    df.to_csv(out_path, index=False)
    return len(df)

# =============================================================================
# FETCH ODS134 (QH ISP)
# =============================================================================

def _fetch_ods134_day(date_str):
    ws = pd.Timestamp(date_str).tz_localize(BRUSSELS, nonexistent="shift_forward").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    we = (pd.Timestamp(date_str) + pd.Timedelta(days=1)).tz_localize(BRUSSELS, nonexistent="shift_forward").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
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
            except Exception:
                time.sleep(5)
        else:
            break
        all_records.extend(data)
        if len(data) < 1000: break
        start_i += 1000
        if start_i >= 9000: break
    rows = []
    for rec in all_records:
        f      = rec.get("fields", {})
        dt_raw = f.get("datetime")
        if dt_raw:
            try:
                ts = pd.to_datetime(dt_raw, utc=True).tz_convert(BRUSSELS).tz_localize(None)
                rows.append({"timestamp": ts.floor("15min"), "isp": f.get("imbalanceprice")})
            except: pass
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["isp"] = pd.to_numeric(df["isp"], errors="coerce")
    return df.drop_duplicates("timestamp")

# =============================================================================
# DOWNLOAD MERIT ORDER BE (Elia ODS156/157)
# =============================================================================

def download_elia_dir(date_str, product, direction_name):
    date_obj        = datetime.strptime(date_str, "%Y-%m-%d")
    direction_label = "UP" if direction_name == "incremental" else "DOWN"
    dataset_id      = "ods156" if direction_name == "incremental" else "ods157"
    try:
        date_end = date_obj + timedelta(days=1)
        params   = {"dataset": dataset_id, "rows": 1000, "start": 0,
                    "refine.balancingproduct": product, "sort": "datetime",
                    "q": f"datetime:[{date_str}T00:00:00 TO {date_end.strftime('%Y-%m-%d')}T00:00:00]"}
        r = requests.get("https://opendata.elia.be/api/records/1.0/search/",
                         params=params, timeout=45)
        if r.status_code != 200: return pd.DataFrame()
        data  = r.json(); nhits = data.get("nhits", 0)
        if nhits == 0: return pd.DataFrame()
        recs = []; start = 0
        while start < nhits and start < 10000:
            params["start"] = start
            rp = requests.get("https://opendata.elia.be/api/records/1.0/search/",
                              params=params, timeout=45)
            if rp.status_code == 200:
                recs.extend(rp.json().get("records", [])); start += 1000
            else: break
            time.sleep(0.3)
        offers = []
        for rec in recs:
            f  = rec.get("fields", {})
            dt = pd.to_datetime(f.get("datetime", ""), errors="coerce", utc=True)
            if pd.isna(dt): continue
            ts = dt.tz_convert("Europe/Brussels").tz_localize(None).floor("15min")
            if ts.date() != date_obj.date(): continue
            try:
                price    = float(f.get("energybidmarginalprice"))
                quantity = float(f.get("energybidvolume"))
            except: continue
            if quantity <= 0: continue
            offers.append({"timestamp": ts, "direction": direction_label,
                           "price": price, "quantity": quantity,
                           "country": "BE", "reserve": product})
        df = pd.DataFrame(offers)
        return df.drop_duplicates(
            subset=["timestamp", "direction", "price", "quantity", "reserve"]
        ) if not df.empty else pd.DataFrame()
    except: return pd.DataFrame()

def download_elia(date_str, product="aFRR"):
    try:
        up   = download_elia_dir(date_str, product, "incremental")
        down = download_elia_dir(date_str, product, "decremental")
        return pd.concat([up, down], ignore_index=True)
    except: return pd.DataFrame()

# =============================================================================
# DOWNLOAD MERIT ORDER NL (TenneT)
# =============================================================================

def download_tennet(date_str):
    try:
        base = datetime.strptime(date_str + " 00:00:00", "%Y-%m-%d %H:%M:%S")
        resp = requests.get(
            "https://api.tennet.eu/publications/v1/merit-order-list",
            headers={"apikey": API_KEY_TENNET, "Accept": "text/csv"},
            params={"date_from": base.strftime("%d-%m-%Y %H:%M:%S"),
                    "date_to":   (base + timedelta(days=1)).strftime("%d-%m-%Y %H:%M:%S")},
            timeout=60)
        if resp.status_code != 200 or not resp.content.strip():
            return pd.DataFrame(), pd.DataFrame()
        mo = pd.read_csv(StringIO(resp.content.decode("utf-8")))
        tc = "Timeinterval Start Loc" if "Timeinterval Start Loc" in mo.columns \
             else "Timeinterval Start (Local Time)"
        if tc not in mo.columns: return pd.DataFrame(), pd.DataFrame()
        mo["timestamp"] = pd.to_datetime(mo[tc], errors="coerce").dt.tz_localize(None)
        mo = mo.dropna(subset=["timestamp", "Capacity Threshold"])
        mo = mo[mo["timestamp"].dt.date == datetime.strptime(date_str, "%Y-%m-%d").date()]
        mo = mo.sort_values(["timestamp", "Capacity Threshold"]).reset_index(drop=True)
        mo["volume"] = mo.groupby("timestamp")["Capacity Threshold"] \
                         .diff().fillna(mo["Capacity Threshold"])
        mo = mo[mo["volume"] > 0]
        all_a, all_m = [], []
        for ts in mo["timestamp"].unique():
            mt = mo[mo["timestamp"] == ts]
            for pc, d in [("Price Up", "UP"), ("Price Down", "DOWN")]:
                if pc not in mt.columns: continue
                ots = mt[[pc, "volume"]].dropna().copy()
                ots.columns = ["price", "quantity"]
                ots["timestamp"] = ts
                ots = ots.sort_values("price", ascending=(d == "UP"))
                ots["cumul"] = ots["quantity"].cumsum()
                af = ots[ots["cumul"] <= 110].copy()
                mf = ots[ots["cumul"] > 110].copy()
                cross = ots[(ots["cumul"] > 110) & ((ots["cumul"] - ots["quantity"]) < 110)]
                if not cross.empty:
                    rc = cross.iloc[0]
                    qa = 110 - (rc["cumul"] - rc["quantity"]); qm = rc["quantity"] - qa
                    if qa > 0:
                        af = pd.concat([af, pd.DataFrame([{
                            "timestamp": ts, "price": rc["price"], "quantity": qa}])],
                            ignore_index=True)
                    if qm > 0 and not mf.empty:
                        mf.iloc[0, mf.columns.get_loc("quantity")] = qm
                for dft, rt in [(af, "aFRR"), (mf, "mFRR")]:
                    if dft.empty: continue
                    dft = dft[["timestamp", "price", "quantity"]].copy()
                    dft["direction"] = d; dft["country"] = "NL"; dft["reserve"] = rt
                    (all_a if rt == "aFRR" else all_m).append(dft)
        def cat(lst):
            if not lst: return pd.DataFrame()
            df = pd.concat(lst, ignore_index=True)
            df.drop_duplicates(
                subset=["timestamp", "direction", "price", "quantity", "reserve"],
                inplace=True)
            return df
        return cat(all_a), cat(all_m)
    except: return pd.DataFrame(), pd.DataFrame()

# =============================================================================
# DOWNLOAD PRIX DA BE (ENTSO-E A44 — zone 10YBE----------2)
# =============================================================================

def download_prix_da(date_str):
    try:
        def fetch():
            start = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d%H%M")
            end   = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y%m%d%H%M")
            resp  = requests.get(BASE_URL_ENTSOE_API,
                params={"securityToken": ENTSOE_TOKEN, "documentType": "A44",
                        "in_Domain":  "10YBE----------2",
                        "out_Domain": "10YBE----------2",
                        "periodStart": start, "periodEnd": end}, timeout=120)
            return resp.content if resp.status_code == 200 else None
        content = retry(fetch, n=3, delay=3)
        if content is None: return pd.DataFrame()
        root = ET.fromstring(content)
        ns   = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}
        rows = []
        for ts_el in root.findall(".//ns:TimeSeries", ns):
            for period in ts_el.findall(".//ns:Period", ns):
                start_el = period.find(".//ns:timeInterval/ns:start", ns)
                if start_el is None: continue
                start_dt   = pd.to_datetime(start_el.text, utc=True)
                res_el     = period.find(".//ns:resolution", ns)
                resolution = res_el.text if res_el is not None else None
                points     = period.findall(".//ns:Point", ns)
                for p in points:
                    pos   = int(p.find(".//ns:position", ns).text)
                    price = float(p.find(".//ns:price.amount", ns).text)
                    offset = timedelta(minutes=(pos - 1) * 15) \
                             if (resolution == "PT15M" or len(points) > 24) \
                             else timedelta(hours=(pos - 1))
                    ts_loc = (start_dt + offset).tz_convert(BRUSSELS).tz_localize(None)
                    rows.append({"timestamp": ts_loc, "price_eur_mwh": price})
        df = pd.DataFrame(rows)
        if df.empty: return pd.DataFrame()
        df = df.drop_duplicates(subset=["timestamp"]) \
               .sort_values("timestamp").reset_index(drop=True)
        df = ensure_qh(df, date_str, kind="da")
        return df[["timestamp", "price_eur_mwh"]]
    except: return pd.DataFrame()

# =============================================================================
# DOWNLOAD INTELLIGENT (BE — ODS133 + ODS134 only, no RTE OAuth)
# =============================================================================

def intelligent_download():
    all_days    = [d.strftime("%Y-%m-%d") for d in pd.date_range("2025-01-01", "2025-12-31", freq="D")]
    missing_imb = [ds for ds in all_days if not (DATA_DIR / f"imbalance_be_{ds}.csv").exists()]
    missing_si  = [ds for ds in all_days
                   if not (SI_CACHE_DIR / f"si_1min_{ds}.csv").exists() or
                      (SI_CACHE_DIR / f"si_1min_{ds}.csv").stat().st_size < 10_000]
    print(f"  imbalance_be : {len(all_days)-len(missing_imb)}/{len(all_days)} presents")
    print(f"  si_1min cache: {len(all_days)-len(missing_si)}/{len(all_days)} presents")

    to_fetch = sorted(set(missing_imb) | set(missing_si))
    if not to_fetch:
        print("  Toutes les donnees sont a jour."); return

    print(f"  {len(to_fetch)} jours a traiter...")
    for ds in to_fetch:
        print(f"    {ds}", end=" | ")
        si_path = SI_CACHE_DIR / f"si_1min_{ds}.csv"
        if ds in set(missing_si):
            n = _download_ods133_day(ds, si_path)
            print(f"si:{n}", end=" ")
        if ds in set(missing_imb) and si_path.exists() and si_path.stat().st_size > 100:
            df_si = pd.read_csv(si_path, parse_dates=["datetime"])
            df_si["ts_qh"] = pd.to_datetime(df_si["datetime"]).dt.floor("15min")
            df_qh = df_si.groupby("ts_qh")["actual_system_imbalance"].mean().reset_index()
            df_qh = df_qh.rename(columns={"ts_qh": "timestamp", "actual_system_imbalance": "volume"})
            ods134 = _fetch_ods134_day(ds)
            if not ods134.empty:
                df_qh = df_qh.merge(ods134[["timestamp", "isp"]], on="timestamp", how="left")
            else:
                df_qh["isp"] = np.nan
            df_qh.to_csv(DATA_DIR / f"imbalance_be_{ds}.csv", index=False)
            print(f"imb:{len(df_qh)}", end=" ")
        print("OK")
        time.sleep(1)

# =============================================================================
# BUILD FEATURES — 60 x 1-min SI in [qh-65min, qh-5min)
# =============================================================================

def build_features(si_all, target_ts):
    feature_end   = target_ts - pd.Timedelta(minutes=5)
    feature_start = feature_end - pd.Timedelta(minutes=60)
    window = si_all[
        (si_all.index >= feature_start) &
        (si_all.index < feature_end)
    ]["actual_system_imbalance"].values
    if len(window) != 60:
        return None
    return window.reshape(1, -1)

# =============================================================================
# LOAD SI 1-MIN CACHE
# =============================================================================

def load_si_all():
    si_files = sorted(SI_CACHE_DIR.glob("si_1min_2025-*.csv"))
    if not si_files:
        raise FileNotFoundError(f"No si_1min_2025-*.csv in {SI_CACHE_DIR}")
    frames = []
    for f in si_files:
        try:
            df = pd.read_csv(f, parse_dates=["datetime"])
            frames.append(df)
        except: pass
    return (pd.concat(frames, ignore_index=True)
              .drop_duplicates("datetime")
              .sort_values("datetime")
              .set_index("datetime"))

# =============================================================================
# GENERATE FORECASTS (walk-forward, BE)
# =============================================================================

def generate_forecasts(si_all):
    imb_files = sorted(DATA_DIR.glob("imbalance_be_2025-*.csv"))
    if not imb_files:
        raise FileNotFoundError("No imbalance_be_2025-*.csv files found.")
    all_qh = []
    for f in imb_files:
        try:
            df = pd.read_csv(f, parse_dates=["timestamp"])
            if "volume" in df.columns:
                df["timestamp"] = normalize_ts(df["timestamp"])
                all_qh.append(df[["timestamp", "volume"]])
        except: pass
    df_qh_all = (pd.concat(all_qh, ignore_index=True)
                   .sort_values("timestamp")
                   .reset_index(drop=True))

    already = set()
    if FORECAST_FILE.exists():
        try:
            done   = pd.read_csv(FORECAST_FILE, parse_dates=["timestamp"])
            done["timestamp"] = normalize_ts(done["timestamp"])
            n_fc   = done["timestamp"].dt.date.nunique()
            n_imb  = len(imb_files)
            if n_fc < n_imb - 2:
                FORECAST_FILE.unlink()
                print(f"  Forecast incomplet ({n_fc}j vs {n_imb}j) — regeneration.")
            else:
                already = set(done["timestamp"].tolist())
                print(f"  Checkpoint: {len(already):,} QH — reprise.")
        except:
            print("  Checkpoint illisible — reprise.")

    model   = joblib.load(MODEL_PATH)
    wmode   = "a" if already else "w"
    wheader = not bool(already)
    batch   = []; n_done = len(already); n_err = 0

    for _, row in df_qh_all.iterrows():
        ts = row["timestamp"]
        if ts in already: continue
        features = build_features(si_all, ts)
        if features is None: n_err += 1; continue
        try:
            pred = float(model.predict(features)[0])
        except Exception:
            n_err += 1; continue
        vol = safe_float(row["volume"])
        batch.append({
            "timestamp":          ts,
            "forecast_volume":    pred,
            "forecast_direction": "INC" if pred < 0 else "DEC",
            "actual_volume":      vol,
            "actual_direction":   "INC" if (pd.notna(vol) and vol < 0) else "DEC",
        })
        n_done += 1
        if len(batch) >= CHECKPOINT_EVERY:
            pd.DataFrame(batch).to_csv(FORECAST_FILE, mode=wmode, header=wheader, index=False)
            wmode = "a"; wheader = False; batch = []
            print(f"  checkpoint: {n_done:,} QH", end="\r")
    if batch:
        pd.DataFrame(batch).to_csv(FORECAST_FILE, mode=wmode, header=wheader, index=False)
    print(f"\n  Termine: {n_done:,} forecasts | {n_err} erreurs.")

# =============================================================================
# MERIT ORDERS — RATIOS
# =============================================================================

def load_merit_day(date_str, reserve):
    parts = []
    for c in ["de", "be", "nl"]:
        f = DATA_DIR / f"{reserve.lower()}_{c}_{date_str}.csv"
        if f.exists():
            try: parts.append(pd.read_csv(f, parse_dates=["timestamp"]))
            except: pass
    if not parts: return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["timestamp"] = normalize_ts(df["timestamp"])
    df["price"]    = pd.to_numeric(df.get("price",    np.nan), errors="coerce")
    df["quantity"] = pd.to_numeric(df.get("quantity", np.nan), errors="coerce")
    return df.dropna(subset=["timestamp", "direction", "price", "quantity"]).query("quantity > 0")

def compute_ratio_negative(df_offers, date_str, label):
    col  = f"{label}_ratio_neg"
    full = create_full_qh_range(date_str)
    if df_offers.empty or "direction" not in df_offers.columns:
        full[col] = 0.0; return full[["timestamp", col]]
    df = df_offers[df_offers["direction"] == "DOWN"].copy()
    if df.empty:
        full[col] = 0.0; return full[["timestamp", col]]
    agg = []
    for ts in df["timestamp"].unique():
        sub   = df[df["timestamp"] == ts]; total = sub["quantity"].sum()
        ratio = 100.0 * sub[sub["price"] < 0]["quantity"].sum() / total if total > 0 else 0.0
        agg.append({"timestamp": ts, col: ratio})
    out = dedup(pd.DataFrame(agg), [col])
    out = ensure_qh(out, date_str, kind="metrics")
    out[col] = out[col].fillna(0.0)
    return out[["timestamp", col]]

# =============================================================================
# STRATEGIES (same as FR)
# =============================================================================

def s8v2(row):
    fv = abs(row.get("forecast_volume", 0) or 0)
    return fv > 300 \
        and (row.get("mfrr_ratio_neg", 0) or 0) > 75 \
        and (row.get("afrr_ratio_neg", 0) or 0) > 75

def s8v4(row):
    fv   = abs(row.get("forecast_volume", 0) or 0)
    da   = row.get("price_eur_mwh", np.nan)
    afrr = row.get("afrr_ratio_neg", 0) or 0
    mfrr = row.get("mfrr_ratio_neg", 0) or 0
    b1   = fv > 300 and mfrr > 75 and afrr > 75
    b2   = not pd.isna(da) and da < 0 and fv > 200 and afrr > 75
    return b1 or b2

def s9_hybrid(row):
    da   = row.get("price_eur_mwh", np.nan)
    fv   = abs(row.get("forecast_volume", 0) or 0)
    afrr = row.get("afrr_ratio_neg", 0) or 0
    da_ok = not pd.isna(da) and da < 0
    return (da_ok and fv > 150) or (fv > 450 and afrr > 75)

def s1_prudent(row):
    mfrr = row.get("mfrr_ratio_neg", 0) or 0
    afrr = row.get("afrr_ratio_neg", 0) or 0
    vol  = abs(row.get("forecast_volume", 0) or 0)
    if vol > 300 and mfrr > 75 and afrr > 65: return True
    if (mfrr > 95 or afrr > 95) and vol > 50: return True
    if mfrr > 75 and afrr > 75 and vol > 100: return True
    return False

def s2_ultra(row):
    mfrr = row.get("mfrr_ratio_neg", 0) or 0
    vol  = abs(row.get("forecast_volume", 0) or 0)
    return mfrr > 80 or (mfrr > 60 and vol > 250)

STRATEGIES = {
    "S8v2":       s8v2,
    "S8v4":       s8v4,
    "S9_Hybrid":  s9_hybrid,
    "S1_Prudent": s1_prudent,
    "S2_Ultra":   s2_ultra,
}

# =============================================================================
# SOC LOGIC (same as FR)
# =============================================================================

def discharge_kwh(ts):
    h, dow = ts.hour, ts.dayofweek
    if dow < 5:
        if h in [6, 18]: return 50 * CONSUMPTION_KWH_PER_KM / 4
    else:
        if 10 <= h < 20: return (200 / 10) * CONSUMPTION_KWH_PER_KM / 4
    return 0.0

def forced_charge_kwh(ts, soc):
    h, dow = ts.hour, ts.dayofweek
    if dow < 5 and h in FORCED_HOURS_WEEKDAY:
        if soc < FORCED_THRESHOLD_WEEKDAY:
            return min(FORCED_THRESHOLD_WEEKDAY - soc, CHARGER_POWER_PER_QH_KWH)
    if dow >= 5 and h in FORCED_HOURS_WEEKEND:
        if soc < FORCED_THRESHOLD_WEEKEND:
            return min(FORCED_THRESHOLD_WEEKEND - soc, CHARGER_POWER_PER_QH_KWH)
    return 0.0

def smart_window(ts):
    h, dow = ts.hour, ts.dayofweek
    if dow < 5: return (8 <= h < 18) or h >= 20 or h < 6
    return h >= 20 or h < 10

# =============================================================================
# REPORTING
# =============================================================================

def print_results(df_all):
    sep = "=" * 80
    print(f"\n{sep}\nRESULTATS — VE 2025 BE (DE+BE+NL)\n{sep}")
    rows = []
    for name in STRATEGIES:
        df_s = df_all[df_all["strategy"] == name]
        ch   = df_s[df_s["total_kwh"] > 0]
        sm   = df_s[df_s["smart_kwh"] > 0]
        neg  = sm[sm["isp"] < 0]
        rows.append({
            "Strategie":       name,
            "Cout total EUR":  round(df_s["cost_eur"].sum(), 2),
            "EUR/MWh moy":     round(df_s["cost_eur"].sum() / ch["total_kwh"].sum() * 1000
                                     if ch["total_kwh"].sum() > 0 else 0, 2),
            "Smart evt":       int(df_s["smart_triggered"].sum()),
            "Smart cout EUR":  round(sm["cost_eur"].sum(), 2),
            "Neg ISP evt":     len(neg),
            "Gain neg EUR":    round(neg["cost_eur"].sum(), 2),
            "Forced cout EUR": round(df_s[df_s["forced_kwh"] > 0]["cost_eur"].sum(), 2),
        })
    df_res = pd.DataFrame(rows).sort_values("Cout total EUR")
    print(df_res.to_string(index=False))
    best = df_res.iloc[0]
    print(f"\n  => Meilleure : {best['Strategie']} ({best['Cout total EUR']:.2f} EUR)")
    print(sep)
    print("\nCOUT PAR MOIS (EUR)\n" + "-" * 70)
    mois = {name: df_all[df_all["strategy"] == name].groupby("month")["cost_eur"].sum()
            for name in STRATEGIES}
    print(pd.DataFrame(mois).round(2).to_string())
    return df_res

# =============================================================================
# MAIN
# =============================================================================

def main():
    sep = "=" * 80
    print(sep)
    print("PIPELINE VE 2025 BELGIQUE — (DE+BE+NL)")
    print(sep)

    # 1. Download missing data
    print("\n[1/5] Telechargement donnees manquantes (ODS133/ODS134)...")
    intelligent_download()

    # 2. Load SI 1-min
    print("\n[2/5] Chargement SI 1-min pour features...")
    si_all = load_si_all()
    print(f"      {len(si_all):,} lignes SI 1-min chargees.")

    # 3. Forecasts
    print("\n[3/5] Forecasts walk-forward (BE)...")
    generate_forecasts(si_all)
    df_fc = pd.read_csv(FORECAST_FILE, parse_dates=["timestamp"])
    df_fc["timestamp"] = normalize_ts(df_fc["timestamp"])
    print(f"      {len(df_fc):,} QH de forecasts charges.")

    # 4. Simulation — SOC continu jour apres jour
    print("\n[4/5] Simulation VE (SOC continu, DE+BE+NL)...")
    days = sorted(f.stem.replace("imbalance_be_", "")
                  for f in DATA_DIR.glob("imbalance_be_2025-*.csv"))
    print(f"      {len(days)} jours candidats.")

    soc_state = {name: INITIAL_SOC_KWH for name in STRATEGIES}
    all_recs  = {name: [] for name in STRATEGIES}
    sim = skip = 0

    for ds in days:
        imb_file = DATA_DIR / f"imbalance_be_{ds}.csv"
        if not imb_file.exists(): skip += 1; continue
        try: df_imb = pd.read_csv(imb_file, parse_dates=["timestamp"])
        except: skip += 1; continue
        df_imb = ensure_qh(df_imb, ds, kind="imb")
        if df_imb.empty or int(df_imb["volume"].isna().sum()) > 48:
            skip += 1; continue
        day  = datetime.strptime(ds, "%Y-%m-%d").date()
        df_f = df_fc[df_fc["timestamp"].dt.date == day].copy()
        if df_f.empty: skip += 1; continue
        df_f = dedup(df_f, ["forecast_volume"])

        rat_a = compute_ratio_negative(load_merit_day(ds, "afrr"), ds, "afrr")
        rat_m = compute_ratio_negative(load_merit_day(ds, "mfrr"), ds, "mfrr")

        df_da = pd.DataFrame()
        da_f  = DATA_DIR / f"prix_da_{ds}.csv"
        if da_f.exists():
            try: df_da = ensure_qh(pd.read_csv(da_f, parse_dates=["timestamp"]), ds, kind="da")
            except: pass

        isp_cols = ["timestamp", "volume", "isp"] if "isp" in df_imb.columns \
                   else ["timestamp", "volume"]
        df_sim = create_full_qh_range(ds)
        df_sim = df_sim.merge(df_imb[isp_cols], on="timestamp", how="left")
        df_sim = df_sim.merge(df_f[["timestamp", "forecast_volume"]], on="timestamp", how="left")
        df_sim = df_sim.merge(rat_a, on="timestamp", how="left")
        df_sim = df_sim.merge(rat_m, on="timestamp", how="left")
        if not df_da.empty:
            df_sim = df_sim.merge(df_da[["timestamp", "price_eur_mwh"]], on="timestamp", how="left")
        else:
            df_sim["price_eur_mwh"] = np.nan

        df_sim["forecast_volume"] = df_sim["forecast_volume"].fillna(0.0)
        df_sim["afrr_ratio_neg"]  = pd.to_numeric(df_sim.get("afrr_ratio_neg"), errors="coerce").fillna(0.0)
        df_sim["mfrr_ratio_neg"]  = pd.to_numeric(df_sim.get("mfrr_ratio_neg"), errors="coerce").fillna(0.0)
        if "isp" not in df_sim.columns:
            df_sim["isp"] = np.nan
        df_sim["isp"] = pd.to_numeric(df_sim["isp"], errors="coerce")

        soc = {name: soc_state[name] for name in STRATEGIES}

        for _, row in df_sim.iterrows():
            ts  = row["timestamp"]
            isp = row.get("isp", np.nan)
            if pd.isna(isp) or pd.isna(row.get("volume", np.nan)):
                for name in STRATEGIES:
                    all_recs[name].append({
                        "timestamp": ts, "cost_eur": 0.0, "forced_kwh": 0.0,
                        "smart_kwh": 0.0, "total_kwh": 0.0, "soc_kwh": soc[name],
                        "smart_triggered": False, "isp": np.nan,
                        "month": ts.month, "strategy": name, "date": ds})
                continue
            disc = discharge_kwh(ts)
            for name in STRATEGIES:
                soc[name] = max(0.0, soc[name] - disc)
            for name, fn in STRATEGIES.items():
                s   = soc[name]
                fkw = min(forced_charge_kwh(ts, s), EV_CAPACITY_KWH - s)
                s  += fkw
                skw = 0.0; triggered = False
                if smart_window(ts) and s < EV_CAPACITY_KWH:
                    if abs(row.get("forecast_volume", 0) or 0) > 50:
                        if fn(row):
                            skw = min(CHARGER_POWER_PER_QH_KWH, EV_CAPACITY_KWH - s)
                            s  += skw; triggered = True
                tkw = fkw + skw; soc[name] = s
                all_recs[name].append({
                    "timestamp": ts, "cost_eur": tkw * float(isp) / 1000.0,
                    "forced_kwh": fkw, "smart_kwh": skw, "total_kwh": tkw,
                    "soc_kwh": s, "isp": float(isp), "smart_triggered": triggered,
                    "month": ts.month, "strategy": name, "date": ds})

        for name in STRATEGIES:
            soc_state[name] = soc[name]
        sim += 1

        costs = {n: sum(r["cost_eur"] for r in all_recs[n] if r["date"] == ds)
                 for n in STRATEGIES}
        print(f"      {ds}: " + " | ".join(f"{n}={v:+.2f}" for n, v in costs.items()))

    print(f"\n      Simules: {sim}/{len(days)}  Skippes: {skip}")
    if not any(all_recs[n] for n in STRATEGIES):
        print("Aucun resultat."); return None, None

    print("\n[5/5] Sauvegarde et reporting...")
    df_all = pd.concat([pd.DataFrame(all_recs[n]) for n in STRATEGIES], ignore_index=True)
    df_all.to_csv(SAVE_DIR / "simulation_complete.csv", index=False)
    df_res = print_results(df_all)
    print(f"\n  Fichiers dans {SAVE_DIR}/")
    return df_all, df_res


df_all, df_res = main()
