#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIMULATION SOLAIRE BE 2025 -- S1/S2/S3
Lit imbalance_be_*.csv + merit_cache depuis data_ve_2025/
Telecharge les prix DA BE depuis ENTSO-E A44 (cache par jour dans data_ve_2025/)
Production PVGIS depuis data/raw/solar_be/pvgis_be_2019.csv
Resultats dans outputs/solar_be/simulation_2025/
"""

import sys, os, warnings, time
from datetime import datetime, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import joblib

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================
REPO_DIR   = Path(__file__).resolve().parents[2]
SOLAR_FR   = REPO_DIR / "optimizers" / "solar_fr"
sys.path.insert(0, str(SOLAR_FR))
os.chdir(str(REPO_DIR))

DATA_2025   = REPO_DIR / "data_ve_2025"
DATA_SOLAR  = REPO_DIR / "data" / "raw" / "solar_be"
RESULTS_DIR = REPO_DIR / "outputs" / "solar_be" / "simulation_2025"
MERIT_CACHE = DATA_2025 / "merit_cache"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MERIT_CACHE.mkdir(exist_ok=True)

SUMMARY_FILE = RESULTS_DIR / "summary_2025.csv"
ERRORS_FILE  = RESULTS_DIR / "errors_2025.csv"
PVGIS_FILE   = DATA_SOLAR / "pvgis_be_2019.csv"

START_DATE = "2025-01-01"
END_DATE   = "2025-12-31"

MODEL_PATH_BE = REPO_DIR / "models" / "be_imbalance_model.joblib"
MAX_LAG_BE    = 16   # 4h de 15-min history (meme structure que forecast_fr)

ENTSOE_TOKEN      = "dcc0c513-dab6-4add-82be-7b693f2b7fab"
ENTSOE_API        = "https://web-api.tp.entsoe.eu/api"
ENTSOE_SOLAR_URL  = "https://transparency.entsoe.eu/generation/forecast/windAndSolar/solar/load"
BE_DOMAIN         = "10YBE----------2"
BE_AREA           = "BZN|10YBE----------2"
BRUSSELS          = "Europe/Brussels"
SEUIL_DA_MID      = 30
DEVIATION_MAX     = 0.5284

SEP = "=" * 100

# =============================================================================
# IMPORT utilitaires partagees depuis solar_fr/test_logic_fr
# =============================================================================
from test_logic_fr import (
    pvgis_day, qh_range, align_qh, file_ok, compute_ratios,
    build_forecast_parc, safe_float,
)

# =============================================================================
# LECTURE IMBALANCE BE
# =============================================================================

def dl_imbalance_be(date_str):
    p = DATA_2025 / f"imbalance_be_{date_str}.csv"
    if not file_ok(p):
        print(f"    imbalance_be  : MANQUANT ({p.name})")
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["timestamp"])
    print(f"    imbalance_be  : cache ({len(df)} lignes)")
    return df

# =============================================================================
# LECTURE PVGIS BE
# =============================================================================

def dl_pvgis_be():
    if not file_ok(PVGIS_FILE):
        raise FileNotFoundError(f"PVGIS BE introuvable: {PVGIS_FILE}")
    df = pd.read_csv(PVGIS_FILE, parse_dates=["timestamp"])
    print(f"  pvgis_be        : {len(df)} heures  max={df['production_mw'].max():.3f} MW/GW")
    return df

# =============================================================================
# SOLAR ENTSO-E BE (forecast_national + actual_national, pour forecast_parc)
# =============================================================================

def dl_solar_be(date_str):
    cache = DATA_2025 / f"solar_be_{date_str}.csv"
    if file_ok(cache):
        print(f"    solar_be      : cache")
        return pd.read_csv(cache, parse_dates=["timestamp"])

    from datetime import timezone
    date_from = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    date_to   = date_from + timedelta(days=1)
    try:
        r = requests.post(ENTSOE_SOLAR_URL, json={
            "dateTimeRange": {
                "from": date_from.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "to":   date_to.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            },
            "areaList": [BE_AREA], "timeZone": "CET",
            "sorterList": [], "filterMap": {},
        }, headers={"accept": "application/json",
                    "content-type": "application/json; charset=utf-8"}, timeout=90)
        if r.status_code != 200:
            print(f"    solar_be      : HTTP {r.status_code}")
            return pd.DataFrame()
        rows = []
        for inst in r.json().get("instanceList", []):
            if inst.get("businessDimensionMap", {}).get("PRODUCTION_TYPE") != "B16":
                continue
            for period in inst.get("curveData", {}).get("periodList", []):
                st  = period.get("timeInterval", {}).get("from")
                if not st:
                    continue
                res  = period.get("resolution")
                s_dt = pd.to_datetime(st, utc=True)
                for pos_str, vals in period.get("pointMap", {}).items():
                    pos = int(pos_str)
                    off = timedelta(minutes=pos * 15) if res == "PT15M" else timedelta(hours=pos)
                    ts  = (s_dt + off).tz_convert(BRUSSELS).tz_localize(None)
                    v   = vals if isinstance(vals, list) else []
                    rows.append({
                        "timestamp":   ts,
                        "forecast_mw": safe_float(v[2]) if len(v) > 2 else np.nan,
                        "actual_mw":   safe_float(v[3]) if len(v) > 3 else np.nan,
                    })
        if not rows:
            print(f"    solar_be      : VIDE")
            return pd.DataFrame()
        df = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
        df = align_qh(df, date_str, "interp")
        df["forecast_mw"] = df["forecast_mw"].fillna(0)
        df["actual_mw"]   = df["actual_mw"].fillna(0)
        df.to_csv(cache, index=False)
        print(f"    solar_be      : {len(df)} QH -> cache")
        return df
    except Exception as e:
        print(f"    solar_be      : WARN {e}")
        return pd.DataFrame()

# =============================================================================
# PRIX DA BE (ENTSO-E A44, cache par jour)
# =============================================================================

def dl_da_be(date_str):
    cache = DATA_2025 / f"prix_da_be_{date_str}.csv"
    if file_ok(cache):
        df = pd.read_csv(cache, parse_dates=["timestamp"])
        print(f"    prix_da_be    : cache ({len(df)} lignes)")
        return df

    start = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d%H%M")
    end   = (datetime.strptime(date_str, "%Y-%m-%d")
             + timedelta(days=1)).strftime("%Y%m%d%H%M")

    for attempt in range(5):
        try:
            r = requests.get(ENTSOE_API, params={
                "securityToken": ENTSOE_TOKEN,
                "documentType":  "A44",
                "in_Domain":     BE_DOMAIN,
                "out_Domain":    BE_DOMAIN,
                "periodStart":   start,
                "periodEnd":     end,
            }, timeout=60)
            if r.status_code != 200:
                print(f"    prix_da_be    : HTTP {r.status_code}, retry {attempt+1}")
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
                    ts  = (s_dt + off).tz_convert(BRUSSELS).tz_localize(None)
                    rows.append({"timestamp": ts, "price_eur_mwh": px})
            if not rows:
                print(f"    prix_da_be    : VIDE")
                return pd.DataFrame()
            df = (pd.DataFrame(rows)
                  .drop_duplicates("timestamp")
                  .sort_values("timestamp"))
            df.to_csv(cache, index=False)
            print(f"    prix_da_be    : {len(df)} lignes -> cache")
            return df
        except Exception as e:
            print(f"    prix_da_be    : WARN {e}")
            time.sleep(5 * (attempt + 1))
    print(f"    prix_da_be    : ECHEC")
    return pd.DataFrame()

# =============================================================================
# MERIT METRICS 2025 (reuse du cache commun data_ve_2025/merit_cache/)
# =============================================================================

def build_merit_metrics_be(date_str):
    cache = MERIT_CACHE / f"merit_{date_str}.csv"
    if file_ok(cache):
        df = pd.read_csv(cache, parse_dates=["timestamp"])
        for col in ["afrr_ratio_negative", "afrr_ratio_very_negative",
                    "mfrr_ratio_negative",  "mfrr_ratio_very_negative"]:
            if col not in df.columns:
                df[col] = 0.0
        print(f"    merit order EU : cache")
        return df

    dfs_afrr, dfs_mfrr = [], []
    for country in ["be", "de", "nl"]:
        for reserve, lst in [("afrr", dfs_afrr), ("mfrr", dfs_mfrr)]:
            f = DATA_2025 / f"{reserve}_{country}_{date_str}.csv"
            if file_ok(f):
                try:
                    lst.append(pd.read_csv(f, parse_dates=["timestamp"]))
                except Exception:
                    pass

    df_afrr = pd.concat(dfs_afrr, ignore_index=True) if dfs_afrr else pd.DataFrame()
    df_mfrr = pd.concat(dfs_mfrr, ignore_index=True) if dfs_mfrr else pd.DataFrame()

    afrr_dn = compute_ratios(df_afrr, "DOWN").rename(columns={
        "ratio_negative":      "afrr_ratio_negative",
        "ratio_very_negative": "afrr_ratio_very_negative"})
    mfrr_dn = compute_ratios(df_mfrr, "DOWN").rename(columns={
        "ratio_negative":      "mfrr_ratio_negative",
        "ratio_very_negative": "mfrr_ratio_very_negative"})

    full = pd.DataFrame({"timestamp": qh_range(date_str)})
    for dfx in [afrr_dn, mfrr_dn]:
        if dfx is not None and not dfx.empty:
            full = full.merge(dfx, on="timestamp", how="left")
    for col in ["afrr_ratio_negative", "afrr_ratio_very_negative",
                "mfrr_ratio_negative",  "mfrr_ratio_very_negative"]:
        if col not in full.columns:
            full[col] = 0.0
        else:
            full[col] = full[col].fillna(0.0)

    full.to_csv(cache, index=False)
    n_countries = len(dfs_afrr) + len(dfs_mfrr)
    print(f"    merit order EU : {n_countries} fichiers -> {len(full)} QH metriques")
    return full

# =============================================================================
# FORECAST BE (walk-forward, meme structure que forecast_fr)
# =============================================================================

_be_model_cache = {}

def forecast_be(date_str, df_hist):
    """
    Forecast imbalance BE pour date_str en utilisant le modele pre-entraine.
    Features : 16 QH lags de volume BE + 6 features cycliques.
    Retourne DataFrame [timestamp, forecast_volume, forecast_direction] ou None.
    """
    if not MODEL_PATH_BE.exists():
        return None
    if df_hist is None or df_hist.empty:
        return None
    try:
        if "model" not in _be_model_cache:
            _be_model_cache["model"] = joblib.load(MODEL_PATH_BE)
        model = _be_model_cache["model"]
        X, valid_ts = [], []
        for ts in qh_range(date_str):
            hist = df_hist[df_hist["timestamp"] < ts].tail(MAX_LAG_BE)
            if len(hist) < MAX_LAG_BE:
                continue
            lags  = hist["volume"].values[::-1]
            slot  = ts.hour * 4 + ts.minute // 15
            feats = list(lags) + [
                np.sin(2 * np.pi * slot / 96),            np.cos(2 * np.pi * slot / 96),
                np.sin(2 * np.pi * ts.dayofweek / 7),     np.cos(2 * np.pi * ts.dayofweek / 7),
                np.sin(2 * np.pi * ts.month / 12),        np.cos(2 * np.pi * ts.month / 12),
            ]
            X.append(feats)
            valid_ts.append(ts)
        if not X:
            return None
        preds = model.predict(np.array(X, dtype=np.float32))
        rows = [{
            "timestamp":          ts,
            "forecast_volume":    float(p),
            "forecast_direction": "DOWN" if p > 0 else "UP",
        } for ts, p in zip(valid_ts, preds)]
        return pd.DataFrame(rows) if rows else None
    except Exception as e:
        print(f"    forecast_be   : WARN {e}")
        return None

# =============================================================================
# REVENUE IMBALANCE BE (single ISP)
# =============================================================================

def rev_imb_be(ecart, isp):
    """
    ecart = nomination - production (MW), isp = EUR/MWh.
    SHORT + ISP<0 → gain   | LONG  + ISP>0 → gain
    SHORT + ISP>0 → penalty | LONG  + ISP<0 → penalty
    """
    rev = np.zeros(len(ecart))
    m = (ecart > 0) & (isp < 0);  rev[m] =  ecart[m] * np.abs(isp[m]) / 4
    m = (ecart < 0) & (isp > 0);  rev[m] =  np.abs(ecart[m]) * isp[m] / 4
    m = (ecart > 0) & (isp > 0);  rev[m] = -ecart[m] * isp[m] / 4
    m = (ecart < 0) & (isp < 0);  rev[m] = -np.abs(ecart[m]) * np.abs(isp[m]) / 4
    return rev

# =============================================================================
# STRATEGIES BE
# =============================================================================

def compute_strategies_be(df):
    """S1 / S2 / S3 avec revenus BE (single ISP)."""
    fp   = df["forecast_parc"].values
    prod = df["production_mw"].values
    da   = df["price_eur_mwh"].values
    isp  = df["isp"].fillna(0).values

    # Forecast from actual data (perfect foresight)
    vol   = np.abs(df["forecast_volume"].fillna(0).values)
    fdir  = df["forecast_direction"].fillna("UP").values

    mfrr_neg  = df["mfrr_ratio_negative"].fillna(0).values
    afrr_neg  = df["afrr_ratio_negative"].fillna(0).values
    afrr_vneg = df["afrr_ratio_very_negative"].fillna(0).values

    solar_on = prod > 0.01

    # Signal DOWN tres selectif pour BE — vol>300 = quasi-inexistant (21 events/an)
    # Le curtailment est contre-productif en BE car la position SHORT (nom<prod) est
    # deja rentable quand ISP>0 (dominant en BE). On garde le signal pour compat.
    sd = (solar_on & (fdir == "DOWN")
          & (afrr_neg > 50) & (mfrr_neg > 50) & (vol > 300))
    df["signal_down_150"] = sd   # colonne conservee pour compatibilite

    # ── S1 Baseline — nomination 100% forecast, pas d'ajustement ─────────── #
    df["s1_nomination"]  = fp
    df["s1_production"]  = prod
    df["s1_ecart"]       = fp - prod
    df["s1_revenue_imb"] = rev_imb_be(df["s1_ecart"].values, isp)
    df["s1_revenue_da"]  = fp * da / 4
    df["s1_total"]       = df["s1_revenue_da"] + df["s1_revenue_imb"]

    # ── S2 Active — DA-adapt 0%/100% (seuil_da=30) + curtail signal optimise #
    curtail_s2 = sd & solar_on
    nom_s2 = np.where(da < 0, 0.0, np.where(da < SEUIL_DA_MID, fp * 0.5, fp))
    df["s2_curtail"]     = curtail_s2
    df["s2_nomination"]  = nom_s2
    df["s2_production"]  = np.where(curtail_s2, 0.0, prod)
    df["s2_ecart"]       = nom_s2 - df["s2_production"].values
    df["s2_revenue_imb"] = rev_imb_be(df["s2_ecart"].values, isp)
    df["s2_revenue_da"]  = nom_s2 * da / 4
    df["s2_total"]       = df["s2_revenue_da"] + df["s2_revenue_imb"]

    # ── S3 Active — nomination 10% fixe + curtail signal quasi-inexistant ───── #
    # 10% maximise la position SHORT (nom<<prod) qui profite de ISP>0 dominant en BE
    curtail_s3 = sd & solar_on
    nom_s3 = fp * 0.1
    df["s3_curtail"]     = curtail_s3
    df["s3_nomination"]  = nom_s3
    df["s3_production"]  = np.where(curtail_s3, 0.0, prod)
    df["s3_ecart"]       = nom_s3 - df["s3_production"].values
    df["s3_revenue_imb"] = rev_imb_be(df["s3_ecart"].values, isp)
    df["s3_revenue_da"]  = nom_s3 * da / 4
    df["s3_total"]       = df["s3_revenue_da"] + df["s3_revenue_imb"]

    return df

# =============================================================================
# SIMULATION UN JOUR
# =============================================================================

def simulate_day_be(date_str, df_pvgis, df_hist):
    result_path = RESULTS_DIR / f"day_{date_str}.csv"
    if file_ok(result_path):
        df = pd.read_csv(result_path, parse_dates=["timestamp"])
        print(f"  {date_str}  -> cache ({len(df)} QH)")
        return df

    print(f"\n  {date_str}")
    df_imb    = dl_imbalance_be(date_str)
    df_da     = dl_da_be(date_str)
    df_pv_day = pvgis_day(date_str, df_pvgis)
    df_solar  = dl_solar_be(date_str)   # forecast_national + actual_national BE

    if df_imb.empty or df_da.empty or df_pv_day.empty:
        print(f"    -> SKIP (imb={df_imb.empty} da={df_da.empty} pv={df_pv_day.empty})")
        return None

    metrics = build_merit_metrics_be(date_str)

    df_fc = forecast_be(date_str, df_hist)
    if df_fc is not None:
        print(f"    forecast_be   : {len(df_fc)} QH")
    else:
        print(f"    forecast_be   : historique insuffisant -> volume=0")

    full = pd.DataFrame({"timestamp": qh_range(date_str)})
    full = full.merge(df_imb[["timestamp", "volume", "isp"]],
                      on="timestamp", how="left")
    full = full.merge(df_da[["timestamp", "price_eur_mwh"]],
                      on="timestamp", how="left")
    full = full.merge(df_pv_day[["timestamp", "production_mw"]],
                      on="timestamp", how="left")
    full = full.merge(metrics, on="timestamp", how="left")

    if df_fc is not None and not df_fc.empty:
        full = full.merge(df_fc[["timestamp", "forecast_volume", "forecast_direction"]],
                          on="timestamp", how="left")
    else:
        full["forecast_volume"]    = 0.0
        full["forecast_direction"] = "UP"

    if not df_solar.empty:
        full = full.merge(df_solar[["timestamp", "forecast_mw", "actual_mw"]],
                          on="timestamp", how="left")
    else:
        full["forecast_mw"] = 0.0
        full["actual_mw"]   = 0.0

    full["production_mw"]      = full["production_mw"].fillna(0)
    full["price_eur_mwh"]      = full["price_eur_mwh"].ffill().fillna(0)
    full["isp"]                = full["isp"].ffill().bfill().fillna(0)
    full["forecast_volume"]    = full["forecast_volume"].fillna(0)
    full["forecast_direction"] = full["forecast_direction"].fillna("UP")
    full["forecast_mw"]        = full["forecast_mw"].fillna(0)
    full["actual_mw"]          = full["actual_mw"].fillna(1).replace(0, 1)

    # forecast_parc = PVGIS × (forecast_national / actual_national)
    full["forecast_parc"] = build_forecast_parc(
        full["production_mw"], full[["timestamp", "forecast_mw", "actual_mw"]])
    full["date"] = date_str
    full = compute_strategies_be(full)

    full.to_csv(result_path, index=False)
    return full

# =============================================================================
# SUMMARY
# =============================================================================

def save_summary(results, errors):
    df_all = pd.concat(results, ignore_index=True)
    rows = []
    for date_str, grp in df_all.groupby("date"):
        row = {"date": date_str}
        for s in ["s1", "s2", "s3"]:
            row[f"{s}_total"] = grp[f"{s}_total"].sum()
            row[f"{s}_da"]    = grp[f"{s}_revenue_da"].sum()
            row[f"{s}_imb"]   = grp[f"{s}_revenue_imb"].sum()
            if s != "s1":
                row[f"{s}_curt"] = int(grp[f"{s}_curtail"].sum())
        row["n_solar_qh"]   = int((grp["production_mw"] > 0.01).sum())
        row["prod_mwh_day"] = grp["production_mw"].sum() / 4.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(SUMMARY_FILE, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(ERRORS_FILE, index=False)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print(SEP)
    print("  SIMULATION SOLAIRE BE 2025  --  S1 / S2 / S3")
    print(f"  Periode : {START_DATE} -> {END_DATE}")
    print(f"  Data    : {DATA_2025}")
    print(f"  Outputs : {RESULTS_DIR}")
    print(SEP)

    print("\n  PVGIS BE...")
    df_pvgis = dl_pvgis_be()

    # df_hist vide au depart : walk-forward strict, aucune donnee future n'est disponible
    # Le modele peut prevoir a partir du jour 2 (apres 16 QH de Jan1 en hist)
    df_hist = pd.DataFrame(columns=["timestamp", "volume"])
    print("\n  Historique imbalance : walk-forward (depart vide)")

    all_dates = [d.strftime("%Y-%m-%d")
                 for d in pd.date_range(START_DATE, END_DATE, freq="D")]
    print(f"\n  {len(all_dates)} jours a simuler\n")

    results, errors = [], []

    for i, date_str in enumerate(all_dates):
        try:
            df_day = simulate_day_be(date_str, df_pvgis, df_hist)
        except Exception as e:
            print(f"  {date_str}  -> ERREUR: {e}")
            errors.append({"date": date_str, "error": str(e)})
            continue

        if df_day is None:
            errors.append({"date": date_str, "error": "donnees manquantes"})
            continue

        # Walk-forward : on ajoute les volumes reels du jour simule
        new_rows = df_day[["timestamp", "volume"]].dropna()
        if not df_hist.empty:
            df_hist = (pd.concat([df_hist, new_rows], ignore_index=True)
                       .sort_values("timestamp").drop_duplicates("timestamp"))
        else:
            df_hist = new_rows.copy()

        results.append(df_day)

        n_sol = int((df_day["production_mw"] > 0.01).sum())
        s1 = df_day["s1_total"].sum()
        s2 = df_day["s2_total"].sum()
        s3 = df_day["s3_total"].sum()
        best = ["S1", "S2", "S3"][[s1, s2, s3].index(max(s1, s2, s3))]
        print(f"    {date_str}: {n_sol} QH sol  S1={s1:+.0f}  S2={s2:+.0f}  S3={s3:+.0f}  best={best}")

        if (i + 1) % 14 == 0 or i == len(all_dates) - 1:
            save_summary(results, errors)
            print(f"  [SAVE] {i+1}/{len(all_dates)} jours")

    # Resume final
    print(f"\n{SEP}")
    print("  RESUME FINAL")
    print(SEP)
    if not results:
        print("  Aucun resultat.")
        return

    df_all = pd.concat(results, ignore_index=True)
    save_summary(results, errors)

    total_mwh = df_all["production_mw"].sum() / 4.0

    df_all["month"] = pd.to_datetime(df_all["timestamp"]).dt.to_period("M").astype(str)
    print()
    for mois in sorted(df_all["month"].unique()):
        dm   = df_all[df_all["month"] == mois]
        vals = [dm[f"s{k}_total"].sum() for k in [1, 2, 3]]
        best = ["S1", "S2", "S3"][vals.index(max(vals))]
        print(f"  {mois}  S1={vals[0]:+7.0f}  S2={vals[1]:+7.0f}  S3={vals[2]:+7.0f}  best={best}")

    print()
    print(f"  Production totale PVGIS: {total_mwh:,.1f} MWh/GWc")
    print()
    for s, name in [("s1", "S1 Baseline          "),
                    ("s2", "S2 DA-adapt+curt150MW"),
                    ("s3", "S3 50%fix+curt150MW  ")]:
        tot   = df_all[f"{s}_total"].sum()
        da    = df_all[f"{s}_revenue_da"].sum()
        imb   = df_all[f"{s}_revenue_imb"].sum()
        nc    = int(df_all[f"{s}_curtail"].sum()) if s != "s1" else 0
        eur_mwh = tot / total_mwh if total_mwh > 0 else 0
        print(f"  {name}  total={tot:+9.2f}  DA={da:+9.2f}  imb={imb:+9.2f}  "
              f"curt={nc:4d}  EUR/MWh={eur_mwh:+.2f}")

    if errors:
        print(f"\n  {len(errors)} jours en erreur / manquants")

    print(f"\n  Resultats dans : {RESULTS_DIR}")
    print(SEP)


if __name__ == "__main__":
    main()
