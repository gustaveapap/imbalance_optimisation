#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VE FR -- simulation en continu, un QH a la fois.
  ISP          : RTE API OAuth2 (imbalance_data v5)
  Merit ratios : fetch live BE+DE+NL via solar_be_scheduler.compute_day_eu_metrics
                 (cache local merit_cache/merit_{date}.csv en fallback)
  Forecast vol : forecasters/rte_forecaster/forecast_log.csv
Au demarrage reprend depuis le dernier QH dans outputs/ve_fr/simulation_ve_fr_2026.csv.
"""

import sys, io, time, base64
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
FORECAST_LOG = REPO / "forecasters" / "rte_forecaster" / "forecast_log.csv"
OUT_QH      = REPO / "outputs" / "ve_fr" / "simulation_ve_fr_2026.csv"
OUT_DAILY   = REPO / "outputs" / "ve_fr" / "summary_daily_ve_fr_2026.csv"

CLIENT_ID     = "bdc03388-6c93-46f6-adbf-1a77d5b89684"
CLIENT_SECRET = "da352352-63f8-42ce-9a34-0624f7560a72"
RTE_TOKEN_URL = "https://digital.iservices.rte-france.com/token/oauth/"
RTE_IMB_URL   = "https://digital.iservices.rte-france.com/open_api/balancing_energy/v5/imbalance_data"
PARIS         = "Europe/Paris"

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
# TRIGGERS (identiques VE BE)
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
# AUTH RTE
# =============================================================================
_rte = {"token": None, "ts": 0}

def ensure_rte_token():
    if _rte["token"] is None or time.time() - _rte["ts"] > 3000:
        creds = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
        r = requests.post(RTE_TOKEN_URL,
                          headers={"Authorization": f"Basic {creds}",
                                   "Content-Type": "application/x-www-form-urlencoded"},
                          data={"grant_type": "client_credentials"}, timeout=30)
        r.raise_for_status()
        _rte["token"] = r.json()["access_token"]
        _rte["ts"]    = time.time()
    return _rte["token"]

# =============================================================================
# ISP FETCH (RTE API)
# =============================================================================
def fetch_isp_fr(ts, retries=5, delay=30):
    """Retourne (volume, prix_positif, prix_negatif) pour le QH ts. None si indisponible."""
    ts = pd.Timestamp(ts)
    s  = ts.strftime("%Y-%m-%dT%H:%M:%S+01:00")
    e  = (ts + pd.Timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%S+01:00")
    for attempt in range(retries):
        try:
            token = ensure_rte_token()
            r = requests.get(RTE_IMB_URL,
                             headers={"Authorization": f"Bearer {token}",
                                      "Accept": "application/json"},
                             params={"start_date": s, "end_date": e},
                             timeout=30)
            if r.status_code == 401:
                _rte["token"] = None
                continue
            r.raise_for_status()
            for item in r.json().get("imbalance_data", []):
                for v in item.get("values", []):
                    vts = pd.to_datetime(v.get("start_date"), utc=True, errors="coerce")
                    if pd.isna(vts):
                        continue
                    vts = vts.tz_convert(PARIS).tz_localize(None).floor("15min")
                    if vts == ts:
                        return (float(v.get("imbalance") or 0),
                                float(v.get("positive_imbalance_settlement_price") or 0),
                                float(v.get("negative_imbalance_settlement_price") or 0))
        except Exception as e:
            print(f"  [ISP-FR] tentative {attempt+1}/{retries}: {e}")
        if attempt < retries - 1:
            print(f"  [ISP-FR] {ts.strftime('%H:%M')} indisponible, attente {delay}s...", flush=True)
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


def get_merit_for_qh_fr(ts):
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
# FORECAST VOLUME (forecast_log.csv)
# =============================================================================
def get_forecast_vol_fr(ts):
    """Retourne |forecast| pour le QH ts depuis forecast_log.csv. 0 si absent."""
    try:
        df = pd.read_csv(FORECAST_LOG, header=None, usecols=[0, 1],
                         names=["timestamp", "forecast"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        row = df[df["timestamp"] == pd.Timestamp(ts)]
        if not row.empty:
            return abs(float(row.iloc[0]["forecast"]))
    except Exception:
        pass
    return 0.0

# =============================================================================
# UN PAS DE QH
# =============================================================================
def step_qh(ts, volume, prix_pos, prix_neg, mfrr, afrr, vol, soc):
    """ISP FR = prix_positif si volume>0 sinon prix_negatif."""
    ts  = pd.Timestamp(ts)
    h   = ts.hour
    is_weekend = ts.dayofweek >= 5
    isp = prix_pos if volume > 0 else prix_neg

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
            "volume":           round(volume,   2),
            "isp":              round(isp,      4),
            "forecast_volume":  round(vol,      2),
            "mfrr_ratio_neg":   round(mfrr,     2),
            "afrr_ratio_neg":   round(afrr,     2),
            "discharge_kwh":    round(discharge,   4),
            "forced_kwh":       round(forced_kwh,  4),
            "smart_kwh":        round(smart_kwh,   4),
            "total_charge_kwh": round(total,       4),
            "soc_kwh":          round(s,            4),
            "cost_eur":         round(cost,          6),
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
    now = pd.Timestamp(datetime.now())
    return now.floor("15min") - pd.Timedelta(minutes=15)

# =============================================================================
# MAIN
# =============================================================================
def main():
    OUT_QH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SIMULATION VE FR -- CONTINU")
    print("=" * 70)

    # Test auth RTE
    try:
        ensure_rte_token()
        print("  Auth RTE : OK")
    except Exception as e:
        print(f"  Auth RTE : ECHEC ({e}) -- on continue, reessai au premier QH")

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
            is_recent = (pd.Timestamp(datetime.now()) - qh) < pd.Timedelta(hours=1)
            result = fetch_isp_fr(qh,
                                  retries=6 if is_recent else 2,
                                  delay =30 if is_recent else 5)

            if result is None:
                print(f"  [SKIP] {qh.strftime('%Y-%m-%d %H:%M')} ISP indisponible -- nouvel essai au prochain cycle",
                      flush=True)
                break

            volume, prix_pos, prix_neg = result
            mfrr, afrr = get_merit_for_qh_fr(qh)
            vol         = get_forecast_vol_fr(qh)

            rows = step_qh(qh, volume, prix_pos, prix_neg, mfrr, afrr, vol, soc)
            append_rows(rows)
            last_ts = qh

            isp = prix_pos if volume > 0 else prix_neg
            opt_cost = rows[STRATEGIES.index("S_BE_OPT")]["cost_eur"]
            print(f"  {qh.strftime('%Y-%m-%d %H:%M')}  ISP={isp:+7.1f}  vol={vol:5.0f}  "
                  f"mfrr={mfrr:4.0f}%  afrr={afrr:4.0f}%  "
                  f"cout OPT={opt_cost:+.4f} EUR", flush=True)

            if last_summary_date != qh.date() and qh.hour == 23 and qh.minute == 45:
                recompute_daily_summary()
                last_summary_date = qh.date()

        if is_catchup and last_summary_date != last_ts.date():
            recompute_daily_summary()
            last_summary_date = last_ts.date()

        if last_ts >= ceiling:
            next_fire = ceiling + pd.Timedelta(minutes=18)
            wait_until(next_fire)

if __name__ == "__main__":
    main()
