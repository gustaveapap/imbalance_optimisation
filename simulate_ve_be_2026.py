#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VE BE -- simulation en continu, un QH a la fois.
  ISP          : Elia ODS162 (temps reel, fallback ODS134)
  Merit ratios : fetch live BE+DE+NL via solar_be_scheduler.compute_day_eu_metrics
                 (cache local merit_cache/merit_{date}.csv en fallback)
  Forecast vol : forecasters/elia_forecaster/forecastV3.csv
  Prix DA      : ENTSO-E A44 (domaine BE)
Au demarrage reprend depuis le dernier QH dans outputs/ve_be/simulation_ve_be_2026.csv.
"""

import sys, io, time
import xml.etree.ElementTree as ET
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import requests

# =============================================================================
# PATHS & CONFIG
# =============================================================================
REPO              = Path(__file__).resolve().parent
DATA_2026         = REPO / "data_ve_2026"
MERIT_CACHE       = DATA_2026 / "merit_cache"
FORECAST_LOG_FULL = REPO / "forecasters" / "elia_forecaster" / "forecast_log_full.csv"
FORECAST_V3       = REPO / "forecasters" / "elia_forecaster" / "forecastV3.csv"
ISP_BE_LOCAL_DIR  = REPO / "data" / "raw" / "solar_be"
OUT_QH            = REPO / "outputs" / "ve_be" / "simulation_ve_be_2026.csv"
OUT_DAILY         = REPO / "outputs" / "ve_be" / "summary_daily_ve_be_2026.csv"

ELIA_API     = "https://external-elia.opendatasoft.com/api/records/1.0/search/"
ENTSOE_API   = "https://web-api.tp.entsoe.eu/api"
ENTSOE_TOKEN = "dcc0c513-dab6-4add-82be-7b693f2b7fab"
BE_DOMAIN    = "10YBE----------2"
BRUSSELS     = "Europe/Brussels"

BAT_MAX    = 66.0
CHARGER_QH = 2.75
SOC_INIT   = 33.0
KWH_PER_KM = 0.15

TRIP_WD_H_M = 6;  TRIP_WD_H_E = 18;  TRIP_WD_KM = 50.0
TRIP_WE_H_S = 10; TRIP_WE_H_E = 20;  TRIP_WE_KM = 200.0
FORCED_WD_H = {4, 5, 16, 17};  FORCED_WD_MIN = 22.0
FORCED_WE_H = {7, 8, 9};       FORCED_WE_MIN = 33.0

STRATEGIES = ["S1", "S2", "S_BE_OPT"]
START_DATE  = pd.Timestamp("2026-01-01")

# =============================================================================
# TRIGGERS (alignes sur metriques 2025)
# =============================================================================
def s1_trigger(vol, mfrr, afrr, da=np.nan):
    return ((vol > 300 and mfrr > 75 and afrr > 75) or
            (not pd.isna(da) and da < 0 and vol > 200 and afrr > 75) or
            (mfrr > 75 and afrr > 75 and vol > 100))

def s2_trigger(vol, mfrr, afrr, da=np.nan):
    return ((not pd.isna(da) and da < 0 and vol > 150) or
            (vol > 450 and afrr > 75))

def s_be_opt(vol, mfrr, afrr, da=np.nan):
    return afrr > 50 and mfrr > 50 and vol > 200

TRIGGER_FN = {"S1": s1_trigger, "S2": s2_trigger, "S_BE_OPT": s_be_opt}

def smart_window(h, is_weekend):
    if not is_weekend:
        return (8 <= h < 18) or (h >= 20) or (h < 6)
    return (h >= 20) or (h < 10)

# =============================================================================
# ISP LOCAL CACHE (data/raw/solar_be/isp_2026-*.csv, chargé une fois au démarrage)
# =============================================================================
_isp_be_local: dict = {}  # pd.Timestamp -> float

def _load_isp_be_local():
    global _isp_be_local
    if _isp_be_local:
        return
    frames = []
    for f in sorted(ISP_BE_LOCAL_DIR.glob("isp_2026-*.csv")):
        try:
            frames.append(pd.read_csv(f, parse_dates=["datetime"]))
        except Exception:
            pass
    if frames:
        df = pd.concat(frames, ignore_index=True).dropna(subset=["imbalanceprice"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        _isp_be_local = dict(zip(df["datetime"], df["imbalanceprice"].astype(float)))
        mn, mx = min(_isp_be_local), max(_isp_be_local)
        print(f"  [ISP-BE] {len(_isp_be_local)} QH locaux charges "
              f"({pd.Timestamp(mn).date()} -> {pd.Timestamp(mx).date()})", flush=True)
    else:
        print("  [ISP-BE] Aucun fichier ISP local trouve", flush=True)


# =============================================================================
# ISP FETCH (local d'abord, ODS162/ODS134 uniquement pour QH recents hors cache)
# =============================================================================
def _query_isp(dataset, ws, we):
    r = requests.get(ELIA_API, params={
        "dataset": dataset,
        "q":       f"datetime:[{ws} TO {we}]",
        "rows":    10, "sort": "datetime",
    }, timeout=30)
    r.raise_for_status()
    for rec in r.json().get("records", []):
        v = rec.get("fields", {}).get("imbalanceprice")
        if v is not None:
            return float(v)
    return None

def fetch_isp_be(ts, retries=5, delay=30):
    """Retourne ISP (EUR/MWh) pour le QH ts.
    Cherche d'abord dans le cache local (isp_2026-*.csv).
    Appel API (ODS162 puis ODS134) uniquement pour les QH hors cache."""
    ts = pd.Timestamp(ts)
    v = _isp_be_local.get(ts)
    if v is not None:
        return v
    ws = (ts.tz_localize(BRUSSELS, nonexistent="shift_forward")
          .tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
    we = ((ts + pd.Timedelta(minutes=14))
          .tz_localize(BRUSSELS, nonexistent="shift_forward")
          .tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
    for attempt in range(retries):
        try:
            v = _query_isp("ods162", ws, we)
            if v is not None:
                return v
            v = _query_isp("ods134", ws, we)
            if v is not None:
                return v
        except Exception as e:
            print(f"  [ISP-BE] tentative {attempt+1}/{retries}: {e}")
        if attempt < retries - 1:
            print(f"  [ISP-BE] {ts.strftime('%H:%M')} indisponible, attente {delay}s...", flush=True)
            time.sleep(delay)
    return None

# =============================================================================
# MERIT METRICS (fetch live BE+DE+NL via API, cache local)
# =============================================================================
_merit_day_cache = {}

def _fetch_merit_day(date_str):
    """Retourne DataFrame 96 QH avec afrr/mfrr_ratio_negative.
    Lit le cache local si present, sinon fetche live BE+DE+NL via API."""
    if date_str in _merit_day_cache:
        return _merit_day_cache[date_str]
    cache_file = MERIT_CACHE / f"merit_{date_str}.csv"
    MERIT_CACHE.mkdir(exist_ok=True)
    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=["timestamp"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            _merit_day_cache[date_str] = df
            return df
        except Exception:
            pass
    # Cache absent -> fetch live (BE Elia ODS163/164 + DE regelleistung + NL TenNET)
    try:
        import sys as _sys
        _p = str(REPO / "optimizers" / "solar_be")
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
        from solar_be_scheduler import compute_day_eu_metrics
        print(f"  [merit] fetch live {date_str} (BE+DE+NL)...", flush=True)
        df = compute_day_eu_metrics(date_str)
        df.to_csv(cache_file, index=False)
        _merit_day_cache[date_str] = df
        return df
    except Exception as e:
        print(f"  WARNING merit live {date_str}: {e}", flush=True)
        _merit_day_cache[date_str] = pd.DataFrame()
        return pd.DataFrame()


def get_merit_for_qh_be(ts):
    """Retourne (mfrr_ratio_neg, afrr_ratio_neg) — fetch live si cache absent.
    Pour les QH live (jour en cours incomplet), utilise le même créneau de J-1."""
    ts = pd.Timestamp(ts)
    for ds in [ts.strftime("%Y-%m-%d"),
               (ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")]:
        df = _fetch_merit_day(ds)
        if df.empty:
            continue
        # Exact match d'abord (données historiques complètes)
        row = df[df["timestamp"] == ts]
        if row.empty:
            # Fallback : même heure du jour disponible (proxy J-1 pour QH live)
            row = df[df["timestamp"].dt.time == ts.time()]
        if not row.empty:
            r = row.iloc[0]
            return (float(r.get("mfrr_ratio_negative", 0) or 0),
                    float(r.get("afrr_ratio_negative",  0) or 0))
    return 0.0, 0.0

# =============================================================================
# PRIX DA (ENTSO-E A44, domaine BE, cache local)
# =============================================================================
_da_day_cache_be = {}

def _fetch_da_day_be(date_str):
    if date_str in _da_day_cache_be:
        return _da_day_cache_be[date_str]
    cache_file = DATA_2026 / f"da_be_{date_str}.csv"
    DATA_2026.mkdir(exist_ok=True)
    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=["timestamp"])
            _da_day_cache_be[date_str] = df
            return df
        except Exception:
            pass
    start = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d%H%M")
    end   = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y%m%d%H%M")
    try:
        r = requests.get(ENTSOE_API, params={
            "securityToken": ENTSOE_TOKEN, "documentType": "A44",
            "in_Domain": BE_DOMAIN, "out_Domain": BE_DOMAIN,
            "periodStart": start, "periodEnd": end,
        }, timeout=60)
        if r.status_code != 200:
            _da_day_cache_be[date_str] = pd.DataFrame()
            return pd.DataFrame()
        rows = []
        for period in ET.fromstring(r.content).findall(".//{*}Period"):
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
                ts_bxl = (s_dt + off).tz_convert(BRUSSELS).tz_localize(None)
                rows.append({"timestamp": ts_bxl, "price_eur_mwh": px})
        if not rows:
            _da_day_cache_be[date_str] = pd.DataFrame()
            return pd.DataFrame()
        base = pd.DataFrame({"timestamp": pd.date_range(
            f"{date_str} 00:00", periods=96, freq="15min")})
        df = (base.merge(
                pd.DataFrame(rows).drop_duplicates("timestamp"),
                on="timestamp", how="left")
              .ffill().bfill())
        df.to_csv(cache_file, index=False)
        _da_day_cache_be[date_str] = df
        return df
    except Exception as e:
        print(f"  WARNING DA-BE {date_str}: {e}", flush=True)
        _da_day_cache_be[date_str] = pd.DataFrame()
        return pd.DataFrame()


def get_da_price_be(ts):
    """Retourne prix DA (EUR/MWh) pour le QH ts. np.nan si indisponible."""
    ts = pd.Timestamp(ts)
    df = _fetch_da_day_be(ts.strftime("%Y-%m-%d"))
    if df.empty:
        return np.nan
    row = df[df["timestamp"] == ts]
    if not row.empty:
        return float(row.iloc[0]["price_eur_mwh"])
    return np.nan


# =============================================================================
# FORECAST VOLUME (forecast_log_full.csv principal + forecastV3.csv pour mai+)
# Chargé une seule fois en mémoire. Signe préservé — pas de abs().
# forecast_value > 0 = surplus réseau → ISP tend négatif → bon moment pour charger.
# =============================================================================
_forecast_be_cache = None

def _load_forecast_be():
    global _forecast_be_cache
    if _forecast_be_cache is not None:
        return _forecast_be_cache
    frames = []
    # Source principale : forecast_log_full.csv (jan 2025 → avr 2026)
    if FORECAST_LOG_FULL.exists():
        try:
            df = pd.read_csv(FORECAST_LOG_FULL,
                             usecols=["forecast_time", "forecast_value"],
                             parse_dates=["forecast_time"])
            df = df.rename(columns={"forecast_time": "timestamp"})
            frames.append(df)
        except Exception as e:
            print(f"  [VOL-BE] WARN forecast_log_full: {e}", flush=True)
    # Complément : forecastV3.csv (couvre mai 2026 et données live récentes)
    if FORECAST_V3.exists():
        try:
            df = pd.read_csv(FORECAST_V3,
                             usecols=["forecast_time", "forecast_value"],
                             parse_dates=["forecast_time"])
            df = df.rename(columns={"forecast_time": "timestamp"})
            frames.append(df)
        except Exception as e:
            print(f"  [VOL-BE] WARN forecastV3: {e}", flush=True)
    if not frames:
        _forecast_be_cache = pd.DataFrame(columns=["timestamp", "forecast_value"])
        return _forecast_be_cache
    combined = (pd.concat(frames, ignore_index=True)
                .sort_values("timestamp")
                .drop_duplicates("timestamp", keep="first")
                .reset_index(drop=True))
    _forecast_be_cache = combined
    print(f"  [VOL-BE] {len(combined)} QH de forecast charges "
          f"({combined['timestamp'].min().date()} -> {combined['timestamp'].max().date()})", flush=True)
    return _forecast_be_cache


def get_forecast_vol_be(ts):
    """Retourne forecast_volume pour le QH ts. 0 si absent. Signe préservé (pas de abs()).
    Positif = surplus réseau = ISP tend négatif = intérêt de charger."""
    df = _load_forecast_be()
    if df.empty:
        return 0.0
    row = df[df["timestamp"] == pd.Timestamp(ts)]
    if not row.empty:
        v = row.iloc[0]["forecast_value"]
        if pd.notna(v):
            return float(v)
    return 0.0

# =============================================================================
# UN PAS DE QH (3 strategies)
# =============================================================================
def step_qh(ts, isp, mfrr, afrr, vol, da, soc):
    """Run une iteration QH pour les 3 strategies. Modifie soc en place. Retourne les rows."""
    ts = pd.Timestamp(ts)
    h  = ts.hour
    is_weekend = ts.dayofweek >= 5

    # Decharge (identique pour toutes les strategies)
    discharge = 0.0
    if not is_weekend:
        if h in (TRIP_WD_H_M, TRIP_WD_H_E):
            discharge = TRIP_WD_KM * KWH_PER_KM / 4
    else:
        if TRIP_WE_H_S <= h < TRIP_WE_H_E:
            discharge = TRIP_WE_KM * KWH_PER_KM / 40

    rows = []
    for strat in STRATEGIES:
        s = max(0.0, soc[strat] - discharge)

        forced_kwh = 0.0
        if not is_weekend and h in FORCED_WD_H:
            if s < FORCED_WD_MIN:
                forced_kwh = min(FORCED_WD_MIN - s, CHARGER_QH)
        elif is_weekend and h in FORCED_WE_H:
            if s < FORCED_WE_MIN:
                forced_kwh = min(FORCED_WE_MIN - s, CHARGER_QH)
        s += forced_kwh

        smart_kwh = 0.0
        remaining = CHARGER_QH - forced_kwh
        if remaining > 0 and smart_window(h, is_weekend) and s < BAT_MAX and vol > 50:
            if TRIGGER_FN[strat](vol, mfrr, afrr, da):
                smart_kwh = min(remaining, BAT_MAX - s)
        s += smart_kwh
        soc[strat] = s

        total = forced_kwh + smart_kwh
        cost  = total * isp / 1000.0

        rows.append({
            "timestamp":        ts,
            "date":             str(ts.date()),
            "hour":             h,
            "is_weekend":       is_weekend,
            "isp":              round(isp,   4),
            "forecast_volume":  round(vol,   2),
            "mfrr_ratio_neg":   round(mfrr,  2),
            "afrr_ratio_neg":   round(afrr,  2),
            "discharge_kwh":    round(discharge,  4),
            "forced_kwh":       round(forced_kwh, 4),
            "smart_kwh":        round(smart_kwh,  4),
            "total_charge_kwh": round(total,      4),
            "soc_kwh":          round(s,           4),
            "cost_eur":         round(cost,         6),
            "is_smart_neg_isp": bool(smart_kwh > 0 and isp < 0),
            "strategy":         strat,
        })
    return rows

# =============================================================================
# OUTPUT
# =============================================================================
def append_rows(rows):
    df = pd.DataFrame(rows)
    write_header = not OUT_QH.exists() or OUT_QH.stat().st_size == 0
    df.to_csv(OUT_QH, mode="a", header=write_header, index=False)

def recompute_daily_summary():
    try:
        df = pd.read_csv(OUT_QH, parse_dates=["timestamp"])
        daily_list = []
        for strat, grp in df.groupby("strategy"):
            d = grp.groupby("date").agg(
                total_charge_kwh =("total_charge_kwh", "sum"),
                forced_kwh       =("forced_kwh",        "sum"),
                smart_kwh        =("smart_kwh",         "sum"),
                total_cost_eur   =("cost_eur",           "sum"),
                n_smart_events   =("smart_kwh",          lambda x: (x > 0).sum()),
                n_smart_neg_isp  =("is_smart_neg_isp",   "sum"),
                soc_min          =("soc_kwh",            "min"),
                soc_max          =("soc_kwh",            "max"),
                discharge_kwh    =("discharge_kwh",      "sum"),
            ).reset_index()
            d["strategy"] = strat
            d["avg_isp_smart"] = (
                grp[grp["smart_kwh"] > 0].groupby("date")["isp"].mean()
                .reindex(d["date"]).values
            )
            daily_list.append(d)
        pd.concat(daily_list, ignore_index=True).to_csv(OUT_DAILY, index=False)
        print("  [SUMMARY] Mis a jour.", flush=True)
    except Exception as e:
        print(f"  [SUMMARY] WARN: {e}")

# =============================================================================
# ETAT (reprise)
# =============================================================================
def restore_state():
    """Retourne (last_ts, soc_dict). last_ts=None si fichier vide/absent."""
    soc = {s: SOC_INIT for s in STRATEGIES}
    if not OUT_QH.exists() or OUT_QH.stat().st_size == 0:
        return None, soc
    try:
        df = pd.read_csv(OUT_QH, parse_dates=["timestamp"])
        if df.empty:
            return None, soc
        last_ts = df["timestamp"].max()
        for strat in STRATEGIES:
            sub = df[df["strategy"] == strat]
            if not sub.empty:
                soc[strat] = float(sub.loc[sub["timestamp"].idxmax(), "soc_kwh"])
        return last_ts, soc
    except Exception as e:
        print(f"  [RESTORE] WARN: {e}")
        return None, soc

# =============================================================================
# HELPERS TIMING
# =============================================================================
def iter_qh(from_ts, to_ts):
    """Genere les QH de from_ts+15min a to_ts inclus."""
    ts  = pd.Timestamp(from_ts).floor("15min") + pd.Timedelta(minutes=15)
    end = pd.Timestamp(to_ts).floor("15min")
    while ts <= end:
        yield ts
        ts += pd.Timedelta(minutes=15)

def wait_until(target):
    delta = (pd.Timestamp(target) - pd.Timestamp(datetime.now())).total_seconds()
    if delta > 0:
        print(f"  Attente {delta/60:.1f} min jusqu'a {pd.Timestamp(target).strftime('%H:%M:%S')}...",
              flush=True)
        time.sleep(delta)

def last_completed_qh():
    """Dernier QH entierement ecoule (avec 3 min de grace pour ISP)."""
    now = pd.Timestamp(datetime.now())
    return now.floor("15min") - pd.Timedelta(minutes=15)

# =============================================================================
# MAIN
# =============================================================================
def main():
    OUT_QH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SIMULATION VE BE -- CONTINU")
    print("=" * 70)

    # Chargement en mémoire au démarrage (une seule fois)
    _load_isp_be_local()
    _load_forecast_be()

    last_ts, soc = restore_state()
    if last_ts is None:
        last_ts = START_DATE - pd.Timedelta(minutes=15)
        print(f"  Demarrage zero depuis {START_DATE.date()}")
    else:
        print(f"  Reprise depuis {last_ts}  |  SOC S1={soc['S1']:.1f}  S_BE_OPT={soc['S_BE_OPT']:.1f} kWh")

    last_summary_date = None

    while True:
        ceiling = last_completed_qh()

        if last_ts >= ceiling:
            # A jour -- attendre le prochain QH + 3 min
            next_fire = ceiling + pd.Timedelta(minutes=18)
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] A jour jusqu'a {last_ts.strftime('%H:%M')}. "
                  f"Prochaine execution : {next_fire.strftime('%H:%M')}", flush=True)
            wait_until(next_fire)
            continue

        todo = list(iter_qh(last_ts, ceiling))
        is_catchup = len(todo) > 1
        if is_catchup:
            print(f"  Rattrapage : {len(todo)} QH  "
                  f"({todo[0].strftime('%Y-%m-%d %H:%M')} -> {todo[-1].strftime('%Y-%m-%d %H:%M')})",
                  flush=True)

        for qh in todo:
            # Pour les QH recents (<1h), retries longs ; historique = rapide
            is_recent = (pd.Timestamp(datetime.now()) - qh) < pd.Timedelta(hours=1)
            isp = fetch_isp_be(qh,
                               retries=6 if is_recent else 2,
                               delay =30 if is_recent else 5)

            if isp is None:
                print(f"  [SKIP] {qh.strftime('%Y-%m-%d %H:%M')} ISP indisponible -- nouvel essai au prochain cycle",
                      flush=True)
                break  # on retentera ce QH au prochain tour de boucle

            mfrr, afrr = get_merit_for_qh_be(qh)
            vol         = get_forecast_vol_be(qh)
            da          = get_da_price_be(qh)

            rows = step_qh(qh, isp, mfrr, afrr, vol, da, soc)
            append_rows(rows)
            last_ts = qh

            opt_cost = rows[STRATEGIES.index("S_BE_OPT")]["cost_eur"]
            da_str   = f"{da:+6.1f}" if not pd.isna(da) else "   N/A"
            print(f"  {qh.strftime('%Y-%m-%d %H:%M')}  ISP={isp:+7.1f}  vol={vol:5.0f}  "
                  f"mfrr={mfrr:4.0f}%  afrr={afrr:4.0f}%  DA={da_str}  "
                  f"cout OPT={opt_cost:+.4f} EUR", flush=True)

            # Recalcul summary journalier au passage de minuit
            if last_summary_date != qh.date() and qh.hour == 23 and qh.minute == 45:
                recompute_daily_summary()
                last_summary_date = qh.date()

        # Recalcul summary si on vient de finir un rattrapage
        if is_catchup and last_summary_date != last_ts.date():
            recompute_daily_summary()
            last_summary_date = last_ts.date()

        if last_ts >= ceiling:
            next_fire = ceiling + pd.Timedelta(minutes=18)
            wait_until(next_fire)

if __name__ == "__main__":
    main()
