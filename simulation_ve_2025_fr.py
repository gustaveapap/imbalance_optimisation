"""
=============================================================================
PIPELINE VE 2025 FRANCE — OPTIMISATION CHARGE INTELLIGENTE
Coller dans une cellule de notebook et executer.

Corrections vs version precedente :
  - DE Regelleistung : endpoint crds/api/v2 (sert l historique 2025)
  - DE Regelleistung : parsing ZipFile+regex (bypass bug openpyxl inlineStr)
  - DE Regelleistung : colonne OFFERED_CAPACITY_[MW]
  - Strategies : S8v2 (meilleure) + S8v4 + S9_Hybrid
  - Gate smart : abs(fv) > 50
  - SOC continu jour apres jour

Resultats attendus (DE+BE+NL complet) :
  S8v2      : ~206.89 EUR
  S8v4      : ~206.98 EUR
  S9_Hybrid : ~244.77 EUR
=============================================================================
"""

import re, time, base64, warnings
from datetime import datetime, timedelta
from io import StringIO, BytesIO
from pathlib import Path
from zipfile import ZipFile

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

CLIENT_ID_IMBALANCE     = "bdc03388-6c93-46f6-adbf-1a77d5b89684"
CLIENT_SECRET_IMBALANCE = "da352352-63f8-42ce-9a34-0624f7560a72"
API_KEY_TENNET          = "18fe140e-12a2-446e-ab89-455d33709fac"
ENTSOE_TOKEN            = "dcc0c513-dab6-4add-82be-7b693f2b7fab"

BASE_URL_RTE        = "https://digital.iservices.rte-france.com/open_api"
BASE_URL_ENTSOE_API = "https://web-api.tp.entsoe.eu/api"

MODEL_PATH    = Path(r"C:\Users\gusta\rte_forecaster\artifacts\fr_imbalance_full_model.pkl")
FORECAST_FILE = Path(r"C:\Users\gusta\imbalance_optimisation\forecast_volume_ve_2025.csv")
DATA_DIR      = Path(r"C:\Users\gusta\imbalance_optimisation\data_ve_2025")
SAVE_DIR      = Path(r"C:\Users\gusta\imbalance_optimisation\simulation_ve_2025_corrige")
DATA_DIR.mkdir(exist_ok=True)
SAVE_DIR.mkdir(exist_ok=True)

MAX_LAG          = 96
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
    return df[keep].groupby("timestamp", as_index=False).mean(numeric_only=True)\
                   .sort_values("timestamp").reset_index(drop=True)

def ensure_qh(df, date_str, kind):
    if df is None or df.empty or "timestamp" not in df.columns: return pd.DataFrame()
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    df = df.copy()
    df["timestamp"] = normalize_ts(df["timestamp"])
    df = df[df["timestamp"].dt.date == date_obj].copy()
    if df.empty: return pd.DataFrame()
    if kind == "da":    vcols = ["price_eur_mwh"]
    elif kind == "imb": vcols = ["volume", "prix_positif", "prix_negatif"]
    else:               vcols = [c for c in df.columns if c != "timestamp"]
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
# AUTH RTE
# =============================================================================

def get_rte_token():
    basic = base64.b64encode(
        f"{CLIENT_ID_IMBALANCE}:{CLIENT_SECRET_IMBALANCE}".encode()).decode()
    r = requests.post(
        "https://digital.iservices.rte-france.com/token/oauth/",
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Authorization": f"Basic {basic}"},
        data={"grant_type": "client_credentials"}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]

# =============================================================================
# DOWNLOAD IMBALANCE FR (RTE)
# =============================================================================

def download_imbalance_fr(date_str, token):
    try:
        start_dt = datetime.strptime(date_str, "%Y-%m-%d")
        r = requests.get(
            f"{BASE_URL_RTE}/balancing_energy/v5/imbalance_data",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            params={"start_date": start_dt.strftime("%Y-%m-%dT%H:%M:%S+01:00"),
                    "end_date":   (start_dt + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S+01:00")},
            timeout=30)
        r.raise_for_status()
        rows = []
        for item in r.json().get("imbalance_data", []):
            for v in item.get("values", []):
                rows.append({
                    "timestamp":    pd.to_datetime(v["start_date"]),
                    "volume":       safe_float(v.get("imbalance")),
                    "prix_positif": safe_float(v.get("positive_imbalance_settlement_price")),
                    "prix_negatif": safe_float(v.get("negative_imbalance_settlement_price"))})
        df = pd.DataFrame(rows)
        if df.empty: return pd.DataFrame()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)\
                            .dt.tz_convert("Europe/Paris").dt.tz_localize(None).dt.floor("15min")
        df = ensure_qh(df, date_str, kind="imb")
        return df[["timestamp", "volume", "prix_positif", "prix_negatif"]]
    except: return pd.DataFrame()

# =============================================================================
# DOWNLOAD MERIT ORDER DE (Regelleistung)
# =============================================================================

def download_regelleistung(date_str, product_type="aFRR"):
    try:
        r = retry(lambda: requests.get(
            "https://www.regelleistung.net/apps/crds/api/v2/tenders/results/anonymous",
            params={"productType": product_type, "market": "ENERGY",
                    "exportFormat": "xlsx", "deliveryDate": date_str},
            timeout=60))
        if r.status_code != 200 or r.content[:2] != b"PK":
            return pd.DataFrame()

        with ZipFile(BytesIO(r.content)) as z:
            xml = z.read("xl/worksheets/sheet1.xml").decode("utf-8", errors="ignore")

        rows = []
        for row_match in re.finditer(r"<row r=\"\d+\">(.*?)</row>", xml, re.DOTALL):
            cells = []
            for cell in re.finditer(r"<c r=\"[A-Z]+\d+\"[^>]*>(.*?)</c>",
                                    row_match.group(1), re.DOTALL):
                t = re.search(r"<t>(.*?)</t>", cell.group(1))
                v = re.search(r"<v>(.*?)</v>", cell.group(1))
                cells.append(t.group(1) if t else (v.group(1) if v else ""))
            rows.append(cells)

        if len(rows) < 2: return pd.DataFrame()
        headers = rows[0]
        max_len = max(len(r) for r in rows)
        data    = [r + [""] * (max_len - len(r)) for r in rows[1:]]
        if max_len > len(headers):
            headers = headers + [f"col_{i}" for i in range(max_len - len(headers))]
        xde = pd.DataFrame(data, columns=headers)
        if "PRODUCT" not in xde.columns: return pd.DataFrame()

        xde["QH_IDX"] = xde["PRODUCT"].astype(str)\
                          .str.extract(r"(\d{3})").astype(float).astype("Int64")
        xde = xde.dropna(subset=["QH_IDX"])
        xde["DeliveryDate"] = pd.to_datetime(
            pd.to_numeric(xde["DELIVERY_DATE"], errors="coerce"),
            unit="D", origin="1899-12-30").dt.normalize()
        xde["timestamp"] = xde["DeliveryDate"] + \
                           pd.to_timedelta((xde["QH_IDX"] - 1) * 15, unit="m")

        qty_col = next((c for c in xde.columns if "CAPACITY" in c.upper()), None)
        if qty_col is None: return pd.DataFrame()

        offers = []
        for pfx, direction in [("POS", "UP"), ("NEG", "DOWN")]:
            sub = xde[xde["PRODUCT"].astype(str).str.startswith(pfx)].copy()
            if sub.empty or "ENERGY_PRICE_[EUR/MWh]" not in sub.columns: continue
            if "ENERGY_PRICE_PAYMENT_DIRECTION" in sub.columns:
                mask = sub["ENERGY_PRICE_PAYMENT_DIRECTION"] == "GRID_TO_PROVIDER"
                sub.loc[mask, "ENERGY_PRICE_[EUR/MWh]"] = \
                    pd.to_numeric(sub.loc[mask, "ENERGY_PRICE_[EUR/MWh]"], errors="coerce") * -1
            of = pd.DataFrame({
                "timestamp": pd.to_datetime(sub["timestamp"]).dt.floor("15min"),
                "direction": direction,
                "price":     pd.to_numeric(sub["ENERGY_PRICE_[EUR/MWh]"], errors="coerce"),
                "quantity":  pd.to_numeric(sub[qty_col], errors="coerce"),
                "country":   "DE", "reserve": product_type,
            }).dropna()
            of = of[of["quantity"] > 0]
            if not of.empty: offers.append(of)

        if not offers: return pd.DataFrame()
        return pd.concat(offers, ignore_index=True).drop_duplicates(
            subset=["timestamp", "direction", "price", "quantity", "reserve"])
    except: return pd.DataFrame()

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
        mo["volume"] = mo.groupby("timestamp")["Capacity Threshold"]\
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
# DOWNLOAD PRIX DA FR (ENTSO-E A44)
# =============================================================================

def download_prix_da(date_str):
    try:
        def fetch():
            start = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d%H%M")
            end   = (datetime.strptime(date_str, "%Y-%m-%d") +
                     timedelta(days=1)).strftime("%Y%m%d%H%M")
            resp  = requests.get(BASE_URL_ENTSOE_API,
                params={"securityToken": ENTSOE_TOKEN, "documentType": "A44",
                        "in_Domain": "10YFR-RTE------C",
                        "out_Domain": "10YFR-RTE------C",
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
                    offset = timedelta(minutes=(pos-1)*15) \
                             if (resolution == "PT15M" or len(points) > 24) \
                             else timedelta(hours=(pos-1))
                    ts_loc = (start_dt + offset).tz_convert("Europe/Paris").tz_localize(None)
                    rows.append({"timestamp": ts_loc, "price_eur_mwh": price})
        df = pd.DataFrame(rows)
        if df.empty: return pd.DataFrame()
        df = df.drop_duplicates(subset=["timestamp"])\
               .sort_values("timestamp").reset_index(drop=True)
        df = ensure_qh(df, date_str, kind="da")
        return df[["timestamp", "price_eur_mwh"]]
    except: return pd.DataFrame()

# =============================================================================
# DOWNLOAD INTELLIGENT
# =============================================================================

def intelligent_download(token):
    all_days = [d.strftime("%Y-%m-%d")
                for d in pd.date_range("2025-01-01", "2025-12-31", freq="D")]
    days = [ds for ds in all_days
            if not (DATA_DIR / f"imbalance_fr_{ds}.csv").exists()]
    print(f"  {len(all_days)-len(days)}/{len(all_days)} jours deja presents.")
    if not days:
        print("  Toutes les donnees sont a jour."); return

    print(f"  {len(days)} jours manquants...")
    for i, ds in enumerate(days):
        if i > 0 and i % 30 == 0:
            try: token = get_rte_token(); print("  [token renouvele]")
            except: pass
        print(f"    {ds}", end=" | ")

        df = download_imbalance_fr(ds, token)
        if not df.empty:
            df.to_csv(DATA_DIR / f"imbalance_fr_{ds}.csv", index=False)
            print(f"imb:{len(df)}", end=" ")
        else: print("imb:--", end=" ")

        for product, prefix in [("aFRR","afrr"), ("mFRR","mfrr")]:
            out = DATA_DIR / f"{prefix}_de_{ds}.csv"
            if not out.exists() or out.stat().st_size < 200:
                df = download_regelleistung(ds, product)
                if not df.empty: df.to_csv(out, index=False)

        for product, prefix in [("aFRR","afrr"), ("mFRR","mfrr")]:
            out = DATA_DIR / f"{prefix}_be_{ds}.csv"
            if not out.exists():
                df = download_elia(ds, product)
                if not df.empty: df.to_csv(out, index=False)

        af_nl, mf_nl = download_tennet(ds)
        if not af_nl.empty: af_nl.to_csv(DATA_DIR / f"afrr_nl_{ds}.csv", index=False)
        if not mf_nl.empty: mf_nl.to_csv(DATA_DIR / f"mfrr_nl_{ds}.csv", index=False)

        out = DATA_DIR / f"prix_da_{ds}.csv"
        if not out.exists():
            df = download_prix_da(ds)
            if not df.empty: df.to_csv(out, index=False)

        print("OK"); time.sleep(1)

# =============================================================================
# FIX SKLEARN
# =============================================================================

def fix_hgb_model(model):
    UINT32 = {"left_child", "right_child", "feature_idx", "n_samples", "bitset_idx"}
    for stage in model._predictors:
        for tree in stage:
            nodes = tree.nodes
            new_dtype = np.dtype([(n, np.uint32 if n in UINT32 else nodes.dtype[n])
                                   for n in nodes.dtype.names])
            tree.nodes = nodes.astype(new_dtype)
    return model

# =============================================================================
# FORECASTS WALK-FORWARD
# =============================================================================

def build_features(historical, target_ts):
    hist = historical[historical["timestamp"] < target_ts]
    if len(hist) < MAX_LAG: return None
    series   = hist.tail(MAX_LAG).set_index("timestamp")["volume"]
    expected = pd.date_range(end=target_ts - pd.Timedelta(minutes=15),
                             periods=MAX_LAG, freq="15min")
    series   = series.reindex(expected).interpolate(limit=2).ffill().bfill()
    slot     = target_ts.hour * 4 + target_ts.minute // 15
    feats    = list(series.values[::-1]) + [
        np.sin(2*np.pi*slot/96),    np.cos(2*np.pi*slot/96),
        np.sin(2*np.pi*target_ts.dayofweek/7), np.cos(2*np.pi*target_ts.dayofweek/7),
        np.sin(2*np.pi*target_ts.month/12),    np.cos(2*np.pi*target_ts.month/12),
    ]
    return np.array(feats).reshape(1, -1)

def generate_forecasts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modele introuvable : {MODEL_PATH}")
    files = sorted(DATA_DIR.glob("imbalance_fr_2025-*.csv"))
    if not files: raise FileNotFoundError("Aucun fichier imbalance.")
    all_df = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["timestamp"])
        if "volume" not in df.columns: continue
        df["timestamp"] = normalize_ts(df["timestamp"])
        df["volume"]    = pd.to_numeric(df["volume"], errors="coerce")
        all_df.append(df[["timestamp", "volume"]])
    df_full = pd.concat(all_df, ignore_index=True)\
                .dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    total = len(df_full)

    already = set()
    if FORECAST_FILE.exists():
        try:
            done = pd.read_csv(FORECAST_FILE, parse_dates=["timestamp"])
            done["timestamp"] = normalize_ts(done["timestamp"])
            already = set(done["timestamp"].tolist())
            last_imb = df_full["timestamp"].max()
            last_fc  = max(already) if already else pd.Timestamp("2000-01-01")
            if (len(already) >= total - MAX_LAG) and \
               (last_fc >= last_imb - pd.Timedelta(minutes=15)):
                print(f"  Forecasts complets : {len(already):,} QH."); return
            print(f"  Checkpoint : {len(already):,}/{total:,} QH — reprise.")
        except:
            print("  Checkpoint illisible — reprise.")

    model  = fix_hgb_model(joblib.load(MODEL_PATH))
    wmode  = "a" if already else "w"; wheader = not bool(already)
    batch  = []; n_done = len(already); n_skip = 0; n_err = 0

    for _, row in df_full.iterrows():
        ts = row["timestamp"]
        if ts in already: continue
        vol = safe_float(row["volume"])
        if np.isnan(vol): n_skip += 1; continue
        X = build_features(df_full, ts)
        if X is None: continue
        try: pred = float(model.predict(X)[0])
        except Exception as e:
            n_err += 1
            if n_err <= 3: print(f"\n  WARN {ts}: {e}")
            continue
        batch.append({"timestamp": ts, "forecast_volume": pred,
                      "forecast_direction": "DOWN" if pred > 0 else "UP",
                      "actual_volume": vol,
                      "actual_direction": "DOWN" if vol > 0 else "UP"})
        n_done += 1
        if len(batch) >= CHECKPOINT_EVERY:
            pd.DataFrame(batch).to_csv(FORECAST_FILE, mode=wmode, header=wheader, index=False)
            wmode = "a"; wheader = False; batch = []
            print(f"  checkpoint : {n_done:,}/{total:,} QH", end="\r")
    if batch:
        pd.DataFrame(batch).to_csv(FORECAST_FILE, mode=wmode, header=wheader, index=False)
    print(f"\n  Termine : {n_done:,} forecasts | {n_skip} NaN skips | {n_err} erreurs.")

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
    return df.dropna(subset=["timestamp","direction","price","quantity"]).query("quantity > 0")

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
        sub = df[df["timestamp"] == ts]; total = sub["quantity"].sum()
        ratio = 100.0 * sub[sub["price"] < 0]["quantity"].sum() / total if total > 0 else 0.0
        agg.append({"timestamp": ts, col: ratio})
    out = dedup(pd.DataFrame(agg), [col])
    out = ensure_qh(out, date_str, kind="metrics")
    out[col] = out[col].fillna(0.0)
    return out[["timestamp", col]]

# =============================================================================
# STRATEGIES
# =============================================================================

def s8v2(row):
    """Meilleure — calibree sur DE+BE+NL : P(ISP<0)=43.4%"""
    fv = abs(row.get("forecast_volume", 0) or 0)
    return fv > 300 \
        and (row.get("mfrr_ratio_neg", 0) or 0) > 75 \
        and (row.get("afrr_ratio_neg", 0) or 0) > 75

def s8v4(row):
    """S8v2 + branche DA : P(ISP<0 branche DA)=66.9%"""
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
# LOGIQUE SOC
# =============================================================================

def discharge_kwh(ts):
    h, dow = ts.hour, ts.dayofweek
    if dow < 5:
        if h in [6, 18]: return 50 * CONSUMPTION_KWH_PER_KM / 4
    else:
        if 10 <= h < 20: return (200/10) * CONSUMPTION_KWH_PER_KM / 4
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
    print(f"\n{sep}\nRESULTATS — VE 2025 FR (DE+BE+NL)\n{sep}")
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
            "Forced cout EUR": round(df_s[df_s["forced_kwh"]>0]["cost_eur"].sum(), 2),
        })
    df_res = pd.DataFrame(rows).sort_values("Cout total EUR")
    print(df_res.to_string(index=False))
    best = df_res.iloc[0]
    print(f"\n  => Meilleure : {best['Strategie']} ({best['Cout total EUR']:.2f} EUR)")
    print(sep)
    print("\nCOUT PAR MOIS (EUR)\n" + "-"*70)
    mois = {name: df_all[df_all["strategy"]==name].groupby("month")["cost_eur"].sum()
            for name in STRATEGIES}
    print(pd.DataFrame(mois).round(2).to_string())
    return df_res

# =============================================================================
# MAIN
# =============================================================================

def main():
    sep = "=" * 80
    print(sep)
    print("PIPELINE VE 2025 FRANCE — version corrigee (DE+BE+NL)")
    print(sep)

    # 1. Auth
    print("\n[1/5] Authentification RTE...")
    token = get_rte_token()
    print("      OK.")

    # 2. Download
    print("\n[2/5] Telechargement donnees manquantes...")
    intelligent_download(token)

    # 3. Forecasts
    print("\n[3/5] Forecasts walk-forward...")
    if FORECAST_FILE.exists():
        try:
            df_check = pd.read_csv(FORECAST_FILE, parse_dates=["timestamp"])
            df_check["timestamp"] = normalize_ts(df_check["timestamp"])
            n_imb = len(list(DATA_DIR.glob("imbalance_fr_2025-*.csv")))
            n_fc  = df_check["timestamp"].dt.date.nunique()
            if n_fc < n_imb - 2:
                FORECAST_FILE.unlink()
                print(f"  Forecast incomplet ({n_fc}j vs {n_imb}j) — regeneration.")
        except:
            FORECAST_FILE.unlink()
    generate_forecasts()
    df_fc = pd.read_csv(FORECAST_FILE, parse_dates=["timestamp"])
    df_fc["timestamp"] = normalize_ts(df_fc["timestamp"])
    print(f"      {len(df_fc):,} QH de forecasts charges.")

    # 4. Simulation — SOC continu jour apres jour
    print("\n[4/5] Simulation VE (SOC continu, DE+BE+NL)...")
    days = sorted(f.stem.replace("imbalance_fr_", "")
                  for f in DATA_DIR.glob("imbalance_fr_2025-*.csv"))
    print(f"      {len(days)} jours candidats.")

    soc_state = {name: INITIAL_SOC_KWH for name in STRATEGIES}
    all_recs  = {name: [] for name in STRATEGIES}
    sim = skip = 0

    for ds in days:
        imb_file = DATA_DIR / f"imbalance_fr_{ds}.csv"
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

        df_sim = create_full_qh_range(ds)
        df_sim = df_sim.merge(df_imb, on="timestamp", how="left")
        df_sim = df_sim.merge(df_f[["timestamp","forecast_volume"]], on="timestamp", how="left")
        df_sim = df_sim.merge(rat_a, on="timestamp", how="left")
        df_sim = df_sim.merge(rat_m, on="timestamp", how="left")
        if not df_da.empty:
            df_sim = df_sim.merge(df_da[["timestamp","price_eur_mwh"]], on="timestamp", how="left")
        else: df_sim["price_eur_mwh"] = np.nan

        df_sim["forecast_volume"] = df_sim["forecast_volume"].fillna(0.0)
        df_sim["afrr_ratio_neg"]  = pd.to_numeric(df_sim.get("afrr_ratio_neg"),  errors="coerce").fillna(0.0)
        df_sim["mfrr_ratio_neg"]  = pd.to_numeric(df_sim.get("mfrr_ratio_neg"),  errors="coerce").fillna(0.0)
        df_sim["isp"] = pd.to_numeric(
            np.where(df_sim["volume"].fillna(0) > 0,
                     df_sim["prix_positif"], df_sim["prix_negatif"]), errors="coerce")

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
                    "timestamp": ts, "cost_eur": tkw*float(isp)/1000.0,
                    "forced_kwh": fkw, "smart_kwh": skw, "total_kwh": tkw,
                    "soc_kwh": s, "isp": float(isp), "smart_triggered": triggered,
                    "month": ts.month, "strategy": name, "date": ds})

        for name in STRATEGIES:
            soc_state[name] = soc[name]
        sim += 1

        costs = {n: sum(r["cost_eur"] for r in all_recs[n] if r["date"]==ds)
                 for n in STRATEGIES}
        print(f"      {ds}: " + " | ".join(f"{n}={v:+.2f}" for n,v in costs.items()))

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
