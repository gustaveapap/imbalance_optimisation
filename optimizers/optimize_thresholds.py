#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid search sur les parametres des strategies S2/S3 pour maximiser le revenu.
Exploite les day_*.csv deja calcules (pas de re-telechargement).
Optimise FR et BE separement.
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from itertools import product

REPO = Path(__file__).resolve().parents[1]
SEP  = "=" * 90

# =============================================================================
# CHARGEMENT
# =============================================================================

def load_days(sim_dir):
    parts = []
    for f in sorted(glob.glob(str(sim_dir / "day_*.csv"))):
        try:
            df = pd.read_csv(f, parse_dates=["timestamp"], on_bad_lines="skip")
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            parts.append(df.dropna(subset=["timestamp"]))
        except Exception:
            pass
    return pd.concat(parts, ignore_index=True)

# =============================================================================
# RECOMPUTE STRATEGIES FR
# =============================================================================

SEUIL_DA_MID_DEFAULT = 30

def rev_imb_fr(ecart, ppos, pneg):
    rev = np.zeros(len(ecart))
    m = (ecart > 0) & (pneg < 0); rev[m] =  ecart[m] * np.abs(pneg[m]) / 4
    m = (ecart < 0) & (ppos > 0); rev[m] =  np.abs(ecart[m]) * ppos[m] / 4
    m = (ecart > 0) & (ppos > 0); rev[m] = -ecart[m] * ppos[m] / 4
    m = (ecart < 0) & (pneg < 0); rev[m] = -np.abs(ecart[m]) * np.abs(pneg[m]) / 4
    return rev

def rev_imb_be(ecart, isp):
    rev = np.zeros(len(ecart))
    m = (ecart > 0) & (isp < 0); rev[m] =  ecart[m] * np.abs(isp[m]) / 4
    m = (ecart < 0) & (isp > 0); rev[m] =  np.abs(ecart[m]) * isp[m] / 4
    m = (ecart > 0) & (isp > 0); rev[m] = -ecart[m] * isp[m] / 4
    m = (ecart < 0) & (isp < 0); rev[m] = -np.abs(ecart[m]) * np.abs(isp[m]) / 4
    return rev

def compute_fr(df, vol_thr, mfrr_thr, afrr_thr, afrr_vneg_thr,
               seuil_da_mid, nom_s3_frac):
    fp   = df["forecast_parc"].values
    prod = df["production_mw"].values
    da   = df["price_eur_mwh"].values
    ppos = df["prix_positif"].values
    pneg = df["prix_negatif"].values
    vol  = df["forecast_volume"].fillna(0).values
    fdir = df["forecast_direction"].fillna("UP").values
    mn   = df["mfrr_ratio_negative"].fillna(0).values
    an   = df["afrr_ratio_negative"].fillna(0).values
    avn  = df["afrr_ratio_very_negative"].fillna(0).values

    solar_on = prod > 0.01
    sd = (solar_on & (fdir == "DOWN")
          & (((vol > vol_thr) & (mn > mfrr_thr) & (an > afrr_thr))
             | ((vol > vol_thr) & (avn > afrr_vneg_thr))))

    # S1
    s1_ecart = fp - prod
    s1_imb   = rev_imb_fr(s1_ecart, ppos, pneg)
    s1_da    = fp * da / 4
    s1_total = (s1_da + s1_imb).sum()

    # S2
    curtail_s2 = sd & solar_on
    nom_s2     = np.where(da < 0, 0.0, np.where(da < seuil_da_mid, fp * 0.5, fp))
    prod_s2    = np.where(curtail_s2, 0.0, prod)
    s2_ecart   = nom_s2 - prod_s2
    s2_imb     = rev_imb_fr(s2_ecart, ppos, pneg)
    s2_da      = nom_s2 * da / 4
    s2_total   = (s2_da + s2_imb).sum()

    # S3
    curtail_s3 = sd & solar_on
    nom_s3     = fp * nom_s3_frac
    prod_s3    = np.where(curtail_s3, 0.0, prod)
    s3_ecart   = nom_s3 - prod_s3
    s3_imb     = rev_imb_fr(s3_ecart, ppos, pneg)
    s3_da      = nom_s3 * da / 4
    s3_total   = (s3_da + s3_imb).sum()

    n_curt = int(curtail_s2.sum())
    return s1_total, s2_total, s3_total, n_curt

def compute_be(df, vol_thr, mfrr_thr, afrr_thr, afrr_vneg_thr,
               seuil_da_mid, nom_s3_frac):
    fp   = df["forecast_parc"].values
    prod = df["production_mw"].values
    da   = df["price_eur_mwh"].values
    isp  = df["isp"].fillna(0).values
    vol  = df["forecast_volume"].fillna(0).values
    fdir = df["forecast_direction"].fillna("UP").values
    mn   = df["mfrr_ratio_negative"].fillna(0).values
    an   = df["afrr_ratio_negative"].fillna(0).values
    avn  = df["afrr_ratio_very_negative"].fillna(0).values

    solar_on = prod > 0.01
    sd = (solar_on & (fdir == "DOWN")
          & (((vol > vol_thr) & (mn > mfrr_thr) & (an > afrr_thr))
             | ((vol > vol_thr) & (avn > afrr_vneg_thr))))

    # S1
    s1_ecart = fp - prod
    s1_imb   = rev_imb_be(s1_ecart, isp)
    s1_da    = fp * da / 4
    s1_total = (s1_da + s1_imb).sum()

    # S2
    curtail_s2 = sd & solar_on
    nom_s2     = np.where(da < 0, 0.0, np.where(da < seuil_da_mid, fp * 0.5, fp))
    prod_s2    = np.where(curtail_s2, 0.0, prod)
    s2_ecart   = nom_s2 - prod_s2
    s2_imb     = rev_imb_be(s2_ecart, isp)
    s2_da      = nom_s2 * da / 4
    s2_total   = (s2_da + s2_imb).sum()

    # S3
    curtail_s3 = sd & solar_on
    nom_s3     = fp * nom_s3_frac
    prod_s3    = np.where(curtail_s3, 0.0, prod)
    s3_ecart   = nom_s3 - prod_s3
    s3_imb     = rev_imb_be(s3_ecart, isp)
    s3_da      = nom_s3 * da / 4
    s3_total   = (s3_da + s3_imb).sum()

    n_curt = int(curtail_s2.sum())
    return s1_total, s2_total, s3_total, n_curt

# =============================================================================
# GRID SEARCH
# =============================================================================

# Parametres communs curtailment signal
VOL_THRS       = [50, 100, 150, 200, 300]
MFRR_THRS      = [70, 75, 80, 85, 90, 95]
AFRR_THRS      = [55, 60, 65, 70, 75, 80, 85]
AFRR_VNEG_THRS = [55, 60, 65, 70, 75, 80, 85]

# Parametres nomination
SEUIL_DA_MIDS  = [0, 15, 30, 45, 60]
NOM_S3_FRACS   = [0.3, 0.4, 0.5, 0.6, 0.7]

def grid_search(df, compute_fn, label):
    print(f"\n  Grid search {label}...")
    print(f"  Combinaisons curtailment : {len(VOL_THRS)*len(MFRR_THRS)*len(AFRR_THRS)*len(AFRR_VNEG_THRS):,}")

    best_s2, best_s3, best_both = -np.inf, -np.inf, -np.inf
    best_p_s2 = best_p_s3 = best_p_both = None

    # Stage 1 : optimise le signal curtailment (avec params DA par defaut)
    for vt, mt, at, avt in product(VOL_THRS, MFRR_THRS, AFRR_THRS, AFRR_VNEG_THRS):
        s1, s2, s3, nc = compute_fn(df, vt, mt, at, avt, SEUIL_DA_MID_DEFAULT, 0.5)
        if s2 > best_s2:
            best_s2 = s2
            best_p_s2 = (vt, mt, at, avt)
        if s3 > best_s3:
            best_s3 = s3
            best_p_s3 = (vt, mt, at, avt)
        best_strat = max(s2, s3)
        if best_strat > best_both:
            best_both = best_strat
            best_p_both = (vt, mt, at, avt, "S2" if s2 > s3 else "S3")

    print(f"  Meilleur signal (S2) : vol>{best_p_s2[0]} mfrr>{best_p_s2[1]} afrr>{best_p_s2[2]} afrr_vneg>{best_p_s2[3]}")
    print(f"  Meilleur signal (S3) : vol>{best_p_s3[0]} mfrr>{best_p_s3[1]} afrr>{best_p_s3[2]} afrr_vneg>{best_p_s3[3]}")

    # Stage 2 : optimise params DA avec meilleur signal S2
    print(f"\n  Optimisation params DA/nomination...")
    best_s2_final = -np.inf
    best_s3_final = -np.inf
    best_p2_final = best_p3_final = None

    for sdm, nf in product(SEUIL_DA_MIDS, NOM_S3_FRACS):
        vt, mt, at, avt = best_p_s2
        s1, s2, s3, nc  = compute_fn(df, vt, mt, at, avt, sdm, nf)
        if s2 > best_s2_final:
            best_s2_final = s2
            best_p2_final = (vt, mt, at, avt, sdm, nf, nc)
        if s3 > best_s3_final:
            best_s3_final = s3
            best_p3_final = (best_p_s3[0], best_p_s3[1], best_p_s3[2], best_p_s3[3], sdm, nf, nc)

    # Recalc baseline S1
    s1_base, _, _, _ = compute_fn(df, 150, 85, 70, 75, SEUIL_DA_MID_DEFAULT, 0.5)
    s2_base, _, _, _ = compute_fn(df, 150, 85, 70, 75, SEUIL_DA_MID_DEFAULT, 0.5)
    _, s2_base, s3_base, nc_base = compute_fn(df, 150, 85, 70, 75, SEUIL_DA_MID_DEFAULT, 0.5)

    return {
        "s1_base":       s1_base,
        "s2_base":       s2_base,
        "s3_base":       s3_base,
        "nc_base":       nc_base,
        "s2_opt":        best_s2_final,
        "s2_opt_params": best_p2_final,
        "s3_opt":        best_s3_final,
        "s3_opt_params": best_p3_final,
    }

def print_results(r, label, prod_mwh):
    print(f"\n{SEP}")
    print(f"  {label}")
    print(SEP)
    s1 = r["s1_base"]
    print(f"  {'Strategie':<30}  {'Total EUR':>10}  {'Gain vs S1':>10}  {'Gain %':>7}  {'EUR/MWh':>8}  {'Curt':>6}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*6}")
    print(f"  {'S1 Baseline':<30}  {s1:>+10.0f}  {'':>10}  {'':>7}  {s1/prod_mwh:>+8.2f}  {'0':>6}")

    for label_s, val, params in [
        ("S2 base (vol>150 mfrr>85 afrr>70)", r["s2_base"], None),
        ("S3 base (vol>150 mfrr>85 afrr>70)", r["s3_base"], None),
        ("S2 OPTIMISE", r["s2_opt"], r["s2_opt_params"]),
        ("S3 OPTIMISE", r["s3_opt"], r["s3_opt_params"]),
    ]:
        gain    = val - s1
        gain_pct = gain / abs(s1) * 100 if s1 != 0 else 0
        nc      = r["nc_base"] if params is None else params[6]
        print(f"  {label_s:<30}  {val:>+10.0f}  {gain:>+10.0f}  {gain_pct:>+7.1f}%  {val/prod_mwh:>+8.2f}  {nc:>6}")
        if params is not None:
            vt, mt, at, avt, sdm, nf, nc = params
            print(f"    -> signal: vol>{vt} mfrr>{mt} afrr>{at} afrr_vneg>{avt}  seuil_da={sdm} nom_frac={nf:.1f}")

# =============================================================================
# MAIN
# =============================================================================

print(SEP)
print("  OPTIMISATION SEUILS STRATEGIES -- FR et BE 2025")
print(SEP)

print("\nChargement FR...")
df_fr = load_days(REPO / "outputs" / "solar_fr" / "simulation_2025")
prod_fr = df_fr["production_mw"].sum() / 4.0
print(f"  {len(df_fr)} QH  prod={prod_fr:.0f} MWh")

print("\nChargement BE...")
df_be = load_days(REPO / "outputs" / "solar_be" / "simulation_2025")
prod_be = df_be["production_mw"].sum() / 4.0
print(f"  {len(df_be)} QH  prod={prod_be:.0f} MWh")

r_fr = grid_search(df_fr, compute_fr, "FRANCE")
r_be = grid_search(df_be, compute_be, "BELGIQUE")

print_results(r_fr, "FRANCE -- Resultats optimisation", prod_fr)
print_results(r_be, "BELGIQUE -- Resultats optimisation", prod_be)
