"""
=============================================================================
PIPELINE VE 2025 BELGIQUE — v2 (BUGS CORRIGES)

BUG #1 CORRIGE : suppression de abs() sur forecast_volume
  - avant : abs(forecast_volume) > seuil  → charge dans les deux directions
  - apres :     forecast_volume  > seuil  → charge uniquement quand reseau
                                             excedentaire (ISP tend negatif)

BUG #2 CONFIRME OK : compute_ratio_negative filtre deja direction=="DOWN"

NOUVELLE STRATEGIE : S_BE_opt
  afrr_ratio_neg > 50 AND mfrr_ratio_neg > 50 AND forecast_volume > 200
  Trouvee par grid search — cible ~63 EUR/an vs ~406 EUR baseline
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
SAVE_DIR      = BASE_DIR / "simulation_ve_2025_be_v2"       # <-- v2
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
# DOWNLOAD INTELLIGENT (ODS133 + ODS134)
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
# MERIT ORDERS — ratios sur bids DOWN uniquement
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
    """Ratio (%) de la capacite decrementale (DOWN) a prix < 0 par QH."""
    col  = f"{label}_ratio_neg"
    full = create_full_qh_range(date_str)
    if df_offers.empty or "direction" not in df_offers.columns:
        full[col] = 0.0; return full[["timestamp", col]]
    # BUG #2 FIX : uniquement bids decrementaux (DOWN)
    df = df_offers[df_offers["direction"] == "DOWN"].copy()
    if df.empty:
        full[col] = 0.0; return full[["timestamp", col]]
    agg = []
    for ts in df["timestamp"].unique():
        sub   = df[df["timestamp"] == ts]
        total = sub["quantity"].sum()
        ratio = 100.0 * sub[sub["price"] < 0]["quantity"].sum() / total if total > 0 else 0.0
        agg.append({"timestamp": ts, col: ratio})
    out = dedup(pd.DataFrame(agg), [col])
    out = ensure_qh(out, date_str, kind="metrics")
    out[col] = out[col].fillna(0.0)
    return out[["timestamp", col]]

# =============================================================================
# STRATEGIES — BUG #1 CORRIGE : plus de abs() sur forecast_volume
# =============================================================================

def s_be_opt(row):
    """NOUVELLE — grid search optimal : aFRR>50, mFRR>50, fv>200."""
    fv   = (row.get("forecast_volume", 0) or 0)   # FIX: pas de abs()
    afrr = row.get("afrr_ratio_neg", 0) or 0
    mfrr = row.get("mfrr_ratio_neg", 0) or 0
    return fv > 200 and afrr > 50 and mfrr > 50

def s8v2(row):
    fv   = (row.get("forecast_volume", 0) or 0)   # FIX
    mfrr = row.get("mfrr_ratio_neg", 0) or 0
    afrr = row.get("afrr_ratio_neg", 0) or 0
    return fv > 300 and mfrr > 75 and afrr > 75

def s8v4(row):
    fv   = (row.get("forecast_volume", 0) or 0)   # FIX
    da   = row.get("price_eur_mwh", np.nan)
    afrr = row.get("afrr_ratio_neg", 0) or 0
    mfrr = row.get("mfrr_ratio_neg", 0) or 0
    b1   = fv > 300 and mfrr > 75 and afrr > 75
    b2   = not pd.isna(da) and da < 0 and fv > 200 and afrr > 75
    return b1 or b2

def s9_hybrid(row):
    fv   = (row.get("forecast_volume", 0) or 0)   # FIX
    da   = row.get("price_eur_mwh", np.nan)
    afrr = row.get("afrr_ratio_neg", 0) or 0
    da_ok = not pd.isna(da) and da < 0
    return (da_ok and fv > 150) or (fv > 450 and afrr > 75)

def s1_prudent(row):
    fv   = (row.get("forecast_volume", 0) or 0)   # FIX
    mfrr = row.get("mfrr_ratio_neg", 0) or 0
    afrr = row.get("afrr_ratio_neg", 0) or 0
    if fv > 300 and mfrr > 75 and afrr > 65: return True
    if (mfrr > 95 or afrr > 95) and fv > 50: return True
    if mfrr > 75 and afrr > 75 and fv > 100: return True
    return False

def s2_ultra(row):
    fv   = (row.get("forecast_volume", 0) or 0)   # FIX
    mfrr = row.get("mfrr_ratio_neg", 0) or 0
    return mfrr > 80 or (mfrr > 60 and fv > 250)

STRATEGIES = {
    "S_BE_opt":   s_be_opt,
    "S8v2":       s8v2,
    "S8v4":       s8v4,
    "S9_Hybrid":  s9_hybrid,
    "S1_Prudent": s1_prudent,
    "S2_Ultra":   s2_ultra,
}

# =============================================================================
# SOC LOGIC
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
    sep = "=" * 100
    print(f"\n{sep}")
    print("RESULTATS — VE 2025 BE v2 (BUG #1 CORRIGE — abs() supprime)")
    print(sep)
    rows = []
    for name in STRATEGIES:
        df_s  = df_all[df_all["strategy"] == name]
        smart = df_s[df_s["smart_kwh"] > 0]
        neg   = smart[smart["isp"] < 0]
        total_eur   = df_s["cost_eur"].sum()
        forced_eur  = df_s[df_s["forced_kwh"] > 0]["cost_eur"].sum()
        smart_eur   = smart["cost_eur"].sum()
        total_kwh   = df_s["total_kwh"].sum()
        forced_kwh  = df_s["forced_kwh"].sum()
        smart_kwh_  = df_s["smart_kwh"].sum()
        smart_evt   = int(df_s["smart_triggered"].sum())
        neg_evt     = len(neg)
        hit_rate    = 100.0 * neg_evt / smart_evt if smart_evt > 0 else 0.0
        rows.append({
            "Strategie":        name,
            "Total EUR":        round(total_eur,  2),
            "Forced EUR":       round(forced_eur, 2),
            "Smart EUR":        round(smart_eur,  2),
            "Total kWh":        round(total_kwh,  1),
            "Forced kWh":       round(forced_kwh, 1),
            "Smart kWh":        round(smart_kwh_, 1),
            "Smart evt":        smart_evt,
            "Neg ISP evt":      neg_evt,
            "Hit rate %":       round(hit_rate, 1),
        })
    df_res = pd.DataFrame(rows).sort_values("Total EUR")
    print(df_res.to_string(index=False))

    best = df_res.iloc[0]
    print(f"\n  => Meilleure : {best['Strategie']} ({best['Total EUR']:.2f} EUR)")
    print(sep)

    # Monthly breakdown — toutes stratégies
    print("\nCOUT PAR MOIS (EUR)")
    print("-" * 70)
    mois = {name: df_all[df_all["strategy"] == name].groupby("month")["cost_eur"].sum()
            for name in STRATEGIES}
    print(pd.DataFrame(mois).round(2).to_string())

    # Monthly detail for best strategy
    best_name = best["Strategie"]
    df_best = df_all[df_all["strategy"] == best_name].copy()
    print(f"\nDETAIL MENSUEL — {best_name}")
    print("-" * 70)
    monthly = df_best.groupby("month").agg(
        total_eur   = ("cost_eur",       "sum"),
        smart_evt   = ("smart_triggered","sum"),
        smart_kwh   = ("smart_kwh",      "sum"),
        forced_kwh  = ("forced_kwh",     "sum"),
    ).round(2)
    print(monthly.to_string())
    return df_res

# =============================================================================
# MAIN
# =============================================================================

def main():
    sep = "=" * 80
    print(sep)
    print("PIPELINE VE 2025 BELGIQUE v2 — (DE+BE+NL) — BUG #1 CORRIGE")
    print(sep)

    print("\n[1/5] Verification donnees (ODS133/ODS134)...")
    intelligent_download()

    print("\n[2/5] Chargement SI 1-min pour features...")
    si_all = load_si_all()
    print(f"      {len(si_all):,} lignes SI 1-min chargees.")

    print("\n[3/5] Forecasts walk-forward (BE)...")
    generate_forecasts(si_all)
    df_fc = pd.read_csv(FORECAST_FILE, parse_dates=["timestamp"])
    df_fc["timestamp"] = normalize_ts(df_fc["timestamp"])
    print(f"      {len(df_fc):,} QH de forecasts charges.")

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
                    # BUG #1 FIX : forecast_volume > 50 (plus abs())
                    # Charge uniquement quand le reseau est excedentaire
                    if (row.get("forecast_volume", 0) or 0) > 50:
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
