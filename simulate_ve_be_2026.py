#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VE BE -- simulation en continu, un QH a la fois.
  ISP          : Elia ODS134 (API temps reel, retries)
  Merit ratios : fetch live BE+DE+NL via solar_be_scheduler.compute_day_eu_metrics
                 (cache local merit_cache/merit_{date}.csv en fallback)
  Forecast vol : forecasters/elia_forecaster/forecastV3.csv
Au demarrage reprend depuis le dernier QH dans outputs/ve_be/simulation_ve_be_2026.csv.
"""

import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import requests

# =============================================================================
# PATHS & CONFIG
# =============================================================================
REPO        = Path(__file__).resolve().parent
DATA_2026   = REPO / "data_ve_2026"
MERIT_CACHE = DATA_2026 / "merit_cache"
FORECAST_V3 = REPO / "forecasters" / "elia_forecaster" / "forecastV3.csv"
OUT_QH      = REPO / "outputs" / "ve_be" / "simulation_ve_be_2026.csv"
OUT_DAILY   = REPO / "outputs" / "ve_be" / "summary_daily_ve_be_2026.csv"

ODS134_URL  = "https://external-elia.opendatasoft.com/api/records/1.0/search/"
BRUSSELS    = "Europe/Brussels"

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
# TRIGGERS (identiques a l'ancien script)
# =============================================================================
def s1_trigger(vol, mfrr, afrr):
    return ((vol > 300 and mfrr > 75 and afrr > 65) or
            ((mfrr > 95 or afrr > 95) and vol > 50) or
            (mfrr > 75 and afrr > 75 and vol > 100))

def s2_trigger(vol, mfrr, afrr):
    return mfrr > 80 or (mfrr > 60 and vol > 250)

def s_be_opt(vol, mfrr, afrr):
    return afrr > 50 and mfrr > 50 and vol > 200

TRIGGER_FN = {"S1": s1_trigger, "S2": s2_trigger, "S_BE_OPT": s_be_opt}

def smart_window(h, is_weekend):
    if not is_weekend:
        return (8 <= h < 18) or (h >= 20) or (h < 6)
    return (h >= 20) or (h < 10)

# =============================================================================
# ISP FETCH (Elia ODS134)
# =============================================================================
def fetch_isp_be(ts, retries=5, delay=30):
    """Retourne ISP (EUR/MWh) pour le QH ts. None si indisponible apres retries."""
    ts = pd.Timestamp(ts)
    ws = (ts.tz_localize(BRUSSELS, nonexistent="shift_forward")
          .tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
    we = ((ts + pd.Timedelta(minutes=14))
          .tz_localize(BRUSSELS, nonexistent="shift_forward")
          .tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
    for attempt in range(retries):
        try:
            r = requests.get(ODS134_URL, params={
                "dataset": "ods134",
                "q":       f"datetime:[{ws} TO {we}]",
                "rows":    10, "sort": "datetime",
            }, timeout=30)
            r.raise_for_status()
            for rec in r.json().get("records", []):
                v = rec.get("fields", {}).get("imbalanceprice")
                if v is not None:
                    return float(v)
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
    """Retourne (mfrr_ratio_neg, afrr_ratio_neg) — fetch live si cache absent."""
    ts = pd.Timestamp(ts)
    for ds in [ts.strftime("%Y-%m-%d"),
               (ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")]:
        df = _fetch_merit_day(ds)
        if df.empty:
            continue
        row = df[df["timestamp"] == ts]
        if not row.empty:
            r = row.iloc[0]
            return (float(r.get("mfrr_ratio_negative", 0) or 0),
                    float(r.get("afrr_ratio_negative",  0) or 0))
    return 0.0, 0.0

# =============================================================================
# FORECAST VOLUME (forecastV3.csv)
# =============================================================================
def get_forecast_vol_be(ts):
    """Retourne |forecast| pour le QH ts depuis forecastV3.csv. 0 si absent."""
    try:
        df = pd.read_csv(FORECAST_V3, header=None, usecols=[0, 1],
                         names=["timestamp", "forecast"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        row = df[df["timestamp"] == pd.Timestamp(ts)]
        if not row.empty:
            return abs(float(row.iloc[0]["forecast"]))
    except Exception:
        pass
    return 0.0

# =============================================================================
# UN PAS DE QH (3 strategies)
# =============================================================================
def step_qh(ts, isp, mfrr, afrr, vol, soc):
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
        if remaining > 0 and smart_window(h, is_weekend) and s < BAT_MAX and vol > 150:
            if TRIGGER_FN[strat](vol, mfrr, afrr):
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
        # Lire seulement les dernieres lignes pour perf
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

            rows = step_qh(qh, isp, mfrr, afrr, vol, soc)
            append_rows(rows)
            last_ts = qh

            opt_cost = rows[STRATEGIES.index("S_BE_OPT")]["cost_eur"]
            print(f"  {qh.strftime('%Y-%m-%d %H:%M')}  ISP={isp:+7.1f}  vol={vol:5.0f}  "
                  f"mfrr={mfrr:4.0f}%  afrr={afrr:4.0f}%  "
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
