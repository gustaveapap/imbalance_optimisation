#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation VE (véhicule électrique) -- Belgique 2026
Input  : C:/Users/gusta/imbalance_optimisation/outputs/solar_be/simulation_be_2026.csv
Outputs: simulation_ve_be_2026.csv, summary_daily_ve_be_2026.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================
INPUT_FILE  = Path("C:/Users/gusta/imbalance_optimisation/outputs/solar_be/simulation_be_2026.csv")
OUT_QH      = Path("C:/Users/gusta/simulation_ve_be_2026.csv")
OUT_DAILY   = Path("C:/Users/gusta/summary_daily_ve_be_2026.csv")

BAT_MAX     = 66.0      # kWh
CHARGER_QH  = 2.75      # kWh/QH  (11 kW * 0.25 h)
SOC_INIT    = 33.0      # kWh
KWH_PER_KM  = 0.15

# Trajets semaine
TRIP_WD_MORNING_H = 6
TRIP_WD_EVENING_H = 18
TRIP_WD_KM        = 50.0   # par trajet => 7.5 kWh sur 4 QH => 1.875 kWh/QH

# Trajets weekend
TRIP_WE_KM        = 200.0  # sur 10h-20h => 40 QH => 0.75 kWh/QH
TRIP_WE_H_START   = 10
TRIP_WE_H_END     = 20     # exclusif

# Forced charge
FORCED_WD_HOURS   = {4, 5, 16, 17}
FORCED_WD_MIN     = 22.0
FORCED_WE_HOURS   = {7, 8, 9}
FORCED_WE_MIN     = 33.0

SEP  = "=" * 100
SEP2 = "-" * 100

# =============================================================================
# CHARGEMENT
# =============================================================================
print(SEP)
print("  SIMULATION VE -- BELGIQUE 2026")
print(SEP)
print(f"\n  Chargement {INPUT_FILE.name}...")
df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
# isp est directement disponible en BE
df["isp"] = pd.to_numeric(df["isp"], errors="coerce").fillna(0.0)
df["forecast_volume"]    = pd.to_numeric(df["forecast_volume"],    errors="coerce").fillna(0.0)
df["mfrr_ratio_negative"]= pd.to_numeric(df["mfrr_ratio_negative"],errors="coerce").fillna(0.0)
df["afrr_ratio_negative"]= pd.to_numeric(df["afrr_ratio_negative"],errors="coerce").fillna(0.0)
print(f"  {len(df)} QH chargés  ({df['timestamp'].min().date()} -> {df['timestamp'].max().date()})")

# =============================================================================
# SIMULATION
# =============================================================================
def smart_window(h, is_weekend):
    """True si l'heure h est dans la fenêtre smart charge autorisée."""
    if not is_weekend:
        return (8 <= h < 18) or (h >= 20) or (h < 6)
    else:
        return (h >= 20) or (h < 10)

def s1_trigger(vol, mfrr, afrr):
    return (
        (vol > 300 and mfrr > 75 and afrr > 65) or
        ((mfrr > 95 or afrr > 95) and vol > 50) or
        (mfrr > 75 and afrr > 75 and vol > 100)
    )

def s2_trigger(vol, mfrr, afrr):
    return mfrr > 80 or (mfrr > 60 and vol > 250)

def s_be_opt(vol, mfrr, afrr):
    return afrr > 50 and mfrr > 50 and vol > 200

def run_simulation(df, strategy):
    soc  = SOC_INIT
    rows = []

    for row in df.itertuples(index=False):
        ts  = row.timestamp
        h   = ts.hour
        dow = ts.dayofweek   # 0=Mon, 6=Sun
        is_weekend = dow >= 5

        isp  = row.isp
        vol  = row.forecast_volume
        mfrr = row.mfrr_ratio_negative
        afrr = row.afrr_ratio_negative

        # ── DISCHARGE ────────────────────────────────────────────────────────
        discharge = 0.0
        if not is_weekend:
            if h == TRIP_WD_MORNING_H:
                discharge = TRIP_WD_KM * KWH_PER_KM / 4   # 1.875 kWh/QH
            elif h == TRIP_WD_EVENING_H:
                discharge = TRIP_WD_KM * KWH_PER_KM / 4
        else:
            if TRIP_WE_H_START <= h < TRIP_WE_H_END:
                discharge = TRIP_WE_KM * KWH_PER_KM / 40  # 0.75 kWh/QH

        soc = max(0.0, soc - discharge)

        # ── FORCED CHARGE ────────────────────────────────────────────────────
        forced_kwh = 0.0
        if not is_weekend and h in FORCED_WD_HOURS:
            if soc < FORCED_WD_MIN:
                forced_kwh = min(FORCED_WD_MIN - soc, CHARGER_QH)
        elif is_weekend and h in FORCED_WE_HOURS:
            if soc < FORCED_WE_MIN:
                forced_kwh = min(FORCED_WE_MIN - soc, CHARGER_QH)
        soc += forced_kwh

        # ── SMART CHARGE ─────────────────────────────────────────────────────
        smart_kwh = 0.0
        remaining = CHARGER_QH - forced_kwh
        if (remaining > 0
                and smart_window(h, is_weekend)
                and soc < BAT_MAX
                and vol > 150):
            if strategy == "S1":
                trigger = s1_trigger(vol, mfrr, afrr)
            elif strategy == "S2":
                trigger = s2_trigger(vol, mfrr, afrr)
            else:
                trigger = s_be_opt(vol, mfrr, afrr)
            if trigger:
                smart_kwh = min(remaining, BAT_MAX - soc)
        soc += smart_kwh

        total_charge = forced_kwh + smart_kwh
        cost = total_charge * isp / 1000.0   # kWh * EUR/MWh / 1000 = EUR

        rows.append({
            "timestamp":       ts,
            "date":            str(ts.date()),
            "hour":            h,
            "is_weekend":      is_weekend,
            "isp":             isp,
            "forecast_volume": row.forecast_volume,
            "mfrr_ratio_neg":  mfrr,
            "afrr_ratio_neg":  afrr,
            "discharge_kwh":   round(discharge,   4),
            "forced_kwh":      round(forced_kwh,  4),
            "smart_kwh":       round(smart_kwh,   4),
            "total_charge_kwh":round(total_charge,4),
            "soc_kwh":         round(soc,          4),
            "cost_eur":        round(cost,          6),
            "is_smart_neg_isp":smart_kwh > 0 and isp < 0,
            "strategy":        strategy,
        })

    return pd.DataFrame(rows)

print("\n  Simulation S1 PRUDENT...")
res_s1 = run_simulation(df, "S1")
print(f"  -> {res_s1['smart_kwh'].gt(0).sum()} QH smart, {res_s1['is_smart_neg_isp'].sum()} à ISP<0")

print("  Simulation S2 ULTRA...")
res_s2 = run_simulation(df, "S2")
print(f"  -> {res_s2['smart_kwh'].gt(0).sum()} QH smart, {res_s2['is_smart_neg_isp'].sum()} à ISP<0")

print("  Simulation S_BE_OPT...")
res_opt = run_simulation(df, "S_BE_OPT")
print(f"  -> {res_opt['smart_kwh'].gt(0).sum()} QH smart, {res_opt['is_smart_neg_isp'].sum()} à ISP<0")

# =============================================================================
# SAUVEGARDE QH
# =============================================================================
res_all = pd.concat([res_s1, res_s2, res_opt], ignore_index=True)
res_all.to_csv(OUT_QH, index=False)
print(f"\n  Sauvegardé : {OUT_QH}")

# =============================================================================
# RÉSUMÉ JOURNALIER
# =============================================================================
daily_list = []
for strat, res in [("S1", res_s1), ("S2", res_s2), ("S_BE_OPT", res_opt)]:
    d = res.groupby("date").agg(
        total_charge_kwh  =("total_charge_kwh",  "sum"),
        forced_kwh        =("forced_kwh",         "sum"),
        smart_kwh         =("smart_kwh",          "sum"),
        total_cost_eur    =("cost_eur",            "sum"),
        n_smart_events    =("smart_kwh",           lambda x: (x > 0).sum()),
        n_smart_neg_isp   =("is_smart_neg_isp",    "sum"),
        soc_min           =("soc_kwh",             "min"),
        soc_max           =("soc_kwh",             "max"),
        discharge_kwh     =("discharge_kwh",       "sum"),
    ).reset_index()
    d["strategy"] = strat
    d["avg_isp_smart"] = (
        res[res["smart_kwh"] > 0].groupby("date")["isp"].mean()
        .reindex(d["date"]).values
    )
    daily_list.append(d)
daily = pd.concat(daily_list, ignore_index=True)
daily.to_csv(OUT_DAILY, index=False)
print(f"  Sauvegardé : {OUT_DAILY}")

# =============================================================================
# AFFICHAGE FINAL
# =============================================================================
print(f"\n{SEP}")
print("  RÉSULTATS VE -- BELGIQUE 2026")
print(SEP)

for strat, res in [("S1 PRUDENT", res_s1), ("S2 ULTRA", res_s2), ("S_BE_OPT", res_opt)]:
    res["month"] = pd.to_datetime(res["timestamp"]).dt.to_period("M").astype(str)
    total_kwh = res["total_charge_kwh"].sum()
    total_cost = res["cost_eur"].sum()
    avg_price = total_cost / total_kwh * 1000 if total_kwh > 0 else 0
    print(f"\n  -- {strat} --")
    print(f"  {'Mois':<10}  {'Coût EUR':>10}  {'Energie kWh':>12}  {'Prix moy EUR/MWh':>18}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*18}")
    for m in sorted(res["month"].unique()):
        rm = res[res["month"] == m]
        mc = rm["cost_eur"].sum()
        mk = rm["total_charge_kwh"].sum()
        mp = mc / mk * 1000 if mk > 0 else 0
        print(f"  {m:<10}  {mc:>+10.2f}  {mk:>12.2f}  {mp:>+18.2f}")
    print(f"  {'TOTAL':<10}  {total_cost:>+10.2f}  {total_kwh:>12.2f}  {avg_price:>+18.2f}")
    print(f"\n  SOC min={res['soc_kwh'].min():.2f} kWh  max={res['soc_kwh'].max():.2f} kWh")
    print(f"  Smart events : {res['smart_kwh'].gt(0).sum()} QH  dont ISP<0 : {res['is_smart_neg_isp'].sum()}")

# Comparaison des stratégies
c1   = res_s1["cost_eur"].sum()
c2   = res_s2["cost_eur"].sum()
copt = res_opt["cost_eur"].sum()
print(f"\n{SEP2}")
print(f"  GAIN S2 vs S1      : {c1-c2:+.2f} EUR  ({(c1-c2)/abs(c1)*100 if c1!=0 else 0:+.2f}%)  (négatif = S2 moins cher)")
print(f"  GAIN S_BE_OPT vs S1: {c1-copt:+.2f} EUR  ({(c1-copt)/abs(c1)*100 if c1!=0 else 0:+.2f}%)  (négatif = OPT moins cher)")
print(SEP)
