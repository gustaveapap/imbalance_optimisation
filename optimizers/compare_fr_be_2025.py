#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparaison des simulations solaires FR vs BE
Revenue total + par MWh pour S1/S2/S3
Usage : python compare_fr_be_2025.py [2025|2026]
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

YEAR = sys.argv[1] if len(sys.argv) > 1 else "2025"
FR_DIR = REPO / "outputs" / "solar_fr" / f"simulation_{YEAR}"
BE_DIR = REPO / "outputs" / "solar_be" / f"simulation_{YEAR}"

SEP  = "=" * 90
SEP2 = "-" * 90


def load_all_days(sim_dir, country):
    parts = []
    for f in sorted(sim_dir.glob("day_*.csv")):
        try:
            parts.append(pd.read_csv(f, parse_dates=["timestamp"], on_bad_lines="skip"))
        except Exception as e:
            print(f"  WARN {country} {f.name}: {e}")
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    return df


def strategy_stats(df, s, country):
    """Returns dict with total/da/imb/curt/mwh/eur_mwh for strategy s."""
    prod_mwh = df["production_mw"].sum() / 4.0  # same denominator for all strategies
    tot  = df[f"{s}_total"].sum()
    da   = df[f"{s}_revenue_da"].sum()
    imb  = df[f"{s}_revenue_imb"].sum()
    curt = int(df[f"{s}_curtail"].sum()) if f"{s}_curtail" in df.columns else 0
    return {
        "total_eur":  tot,
        "da_eur":     da,
        "imb_eur":    imb,
        "curt_qh":    curt,
        "prod_mwh":   prod_mwh,
        "eur_per_mwh": tot / prod_mwh if prod_mwh > 0 else 0,
    }


def print_country_table(df, country):
    print(f"\n{'-'*90}")
    print(f"  {country} -- detail mensuel (EUR / GW installé)")
    print(f"{'-'*90}")
    print(f"  {'Mois':<8}  {'S1 Total':>10}  {'S2 Total':>10}  {'S3 Total':>10}"
          f"  {'Best':>6}  {'Prod MWh':>10}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}  {'-'*10}")
    for mois in sorted(df["month"].unique()):
        dm   = df[df["month"] == mois]
        vals = [dm[f"s{k}_total"].sum() for k in [1, 2, 3]]
        best = ["S1", "S2", "S3"][vals.index(max(vals))]
        mwh  = dm["production_mw"].sum() / 4.0
        print(f"  {mois:<8}  {vals[0]:>+10.0f}  {vals[1]:>+10.0f}  {vals[2]:>+10.0f}"
              f"  {best:>6}  {mwh:>10.1f}")
    tot_vals = [df[f"s{k}_total"].sum() for k in [1, 2, 3]]
    best = ["S1", "S2", "S3"][tot_vals.index(max(tot_vals))]
    mwh  = df["production_mw"].sum() / 4.0
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}  {'-'*10}")
    print(f"  {'TOTAL':<8}  {tot_vals[0]:>+10.0f}  {tot_vals[1]:>+10.0f}  {tot_vals[2]:>+10.0f}"
          f"  {best:>6}  {mwh:>10.1f}")


def main():
    print(SEP)
    print(f"  COMPARAISON SIMULATION SOLAIRE FR vs BE -- {YEAR}")
    print(SEP)

    print("\n  Chargement FR...")
    df_fr = load_all_days(FR_DIR, "FR")
    print(f"  {len(df_fr)} QH FR ({df_fr['timestamp'].min().date()} -> {df_fr['timestamp'].max().date()})")

    print("\n  Chargement BE...")
    df_be = load_all_days(BE_DIR, "BE")
    print(f"  {len(df_be)} QH BE ({df_be['timestamp'].min().date()} -> {df_be['timestamp'].max().date()})")

    # -- Monthly breakdown ---------------------------------------------------- #
    print_country_table(df_fr, "FRANCE")
    print_country_table(df_be, "BELGIQUE")

    # -- Summary table ------------------------------------------------------- #
    print(f"\n{SEP}")
    print(f"  RESUME STRATEGIES -- {YEAR}")
    print(SEP)

    print(f"\n  {'Strategie':<26}  {'Total EUR':>11}  {'DA EUR':>11}  {'Imb EUR':>11}"
          f"  {'Curt QH':>8}  {'Prod MWh':>10}  {'EUR/MWh':>9}")
    print(f"  {'-'*26}  {'-'*11}  {'-'*11}  {'-'*11}  {'-'*8}  {'-'*10}  {'-'*9}")

    for country, df in [("FR", df_fr), ("BE", df_be)]:
        for s, name in [("s1", "S1 Baseline"),
                        ("s2", "S2 DA-adapt+curt"),
                        ("s3", "S3 50%fix+curt")]:
            st   = strategy_stats(df, s, country)
            label = f"{country} {name}"
            curt = st["curt_qh"] if s != "s1" else 0
            print(f"  {label:<26}  {st['total_eur']:>+11.0f}  {st['da_eur']:>+11.0f}"
                  f"  {st['imb_eur']:>+11.0f}  {curt:>8}  {st['prod_mwh']:>10.1f}"
                  f"  {st['eur_per_mwh']:>+9.2f}")
        print(f"  {'-'*26}  {'-'*11}  {'-'*11}  {'-'*11}  {'-'*8}  {'-'*10}  {'-'*9}")

    # -- Gain comparison ----------------------------------------------------- #
    print(f"\n{SEP2}")
    print("  GAINS vs S1 Baseline")
    print(SEP2)
    for country, df in [("FR", df_fr), ("BE", df_be)]:
        s1_tot = df["s1_total"].sum()
        s1_mwh = df["production_mw"].sum() / 4.0
        for s, name in [("s2", "S2 DA-adapt+curt"), ("s3", "S3 50%fix+curt")]:
            st = strategy_stats(df, s, country)
            gain_eur    = st["total_eur"] - s1_tot
            gain_mwh    = st["eur_per_mwh"] - (s1_tot / s1_mwh if s1_mwh > 0 else 0)
            gain_pct    = gain_eur / abs(s1_tot) * 100 if s1_tot != 0 else 0
            print(f"  {country} {name:<20}  gain={gain_eur:>+9.0f} EUR  "
                  f"({gain_pct:>+6.2f}%)  +{gain_mwh:>+.2f} EUR/MWh")

    print(SEP)


if __name__ == "__main__":
    main()
