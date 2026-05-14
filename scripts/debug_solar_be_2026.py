#!/usr/bin/env python3
"""
DEBUG SOLAR BE 2026 — DIAGNOSTIC + CORRECTIONS
===============================================
Lancer depuis : C:/Users/gusta/imbalance_optimisation/
Usage         : python debug_solar_be_2026.py

Ce script :
  1. Diagnostique les bugs dans simulation_be_2026.csv
  2. Teste l'API ODS162 (ISP) pour une date 2026 recente
  3. Applique les corrections dans solar_be_scheduler.py
  4. Reporte chaque bug avec statistiques

BUGS ATTENDUS (d'apres l'analyse) :
  BUG-A : ISP = 0 partout  -> imbalance revenue = 0 pour S1/S2/S3/S4
  BUG-B : Curtailment la nuit (production ≈ 0) -> ecart ≈ 0 meme avec curtail
  BUG-C : aFRR/mFRR tronques a 9000 rows -> metriques fausses
  BUG-D : DA ENTSO-E timeout -> price = 0 sur jours entiers
"""

import re
import sys
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# CHEMINS
# ─────────────────────────────────────────────────────────────────────────────

ROOT        = Path(".")
CSV_2026    = ROOT / "outputs" / "solar_be" / "simulation_be_2026.csv"
SCHEDULER   = ROOT / "optimizers" / "solar_be" / "solar_be_scheduler.py"
REPORT_FILE = ROOT / "outputs" / "solar_be" / "debug_report_2026.txt"

SEP  = "=" * 70
SEP2 = "-" * 70

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

lines_report = []

def log(msg=""):
    print(msg)
    lines_report.append(msg)

def section(title):
    log()
    log(SEP)
    log(f"  {title}")
    log(SEP)

def subsection(title):
    log()
    log(f"  --- {title} ---")
    log(SEP2)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT DU CSV 2026
# ─────────────────────────────────────────────────────────────────────────────

section("CHARGEMENT simulation_be_2026.csv")

if not CSV_2026.exists():
    log(f"  ERREUR : fichier introuvable -> {CSV_2026}")
    log("  Verifie que le batch 2026 a termine et que le fichier existe.")
    sys.exit(1)

df = pd.read_csv(CSV_2026, parse_dates=["timestamp"])
log(f"  {len(df):,} lignes  |  colonnes : {list(df.columns)}")
log(f"  Periode   : {df['timestamp'].min()} -> {df['timestamp'].max()}")
log(f"  Jours     : {df['timestamp'].dt.date.nunique()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. BUG-A : ISP = 0 ?
# ─────────────────────────────────────────────────────────────────────────────

section("BUG-A : ISP (imbalance settlement price)")

if "isp" not in df.columns:
    log("  ERREUR : colonne 'isp' absente du CSV.")
else:
    isp = df["isp"]
    n_total   = len(isp)
    n_zero    = (isp == 0).sum()
    n_nan     = isp.isna().sum()
    n_nonzero = (isp != 0).sum()
    pct_zero  = n_zero / n_total * 100

    log(f"  Total QH       : {n_total:,}")
    log(f"  ISP = 0        : {n_zero:,}  ({pct_zero:.1f}%)")
    log(f"  ISP NaN        : {n_nan:,}")
    log(f"  ISP != 0       : {n_nonzero:,}  ({n_nonzero/n_total*100:.1f}%)")
    log(f"  ISP min/max    : {isp.min():.2f} / {isp.max():.2f}")
    log(f"  ISP mean (!=0) : {isp[isp != 0].mean():.2f} EUR/MWh"
        if n_nonzero > 0 else "  ISP mean (!= 0) : N/A")

    if pct_zero > 80:
        log()
        log("  >>> BUG-A CONFIRME : ISP = 0 dans plus de 80% des QH")
        log("      Cause probable : ODS162 renvoie des donnees vides pour les dates 2026")
        log("      ou le fallback ISP n'est pas correctement applique.")
        BUG_A = True
    elif pct_zero > 20:
        log()
        log(f"  >>> BUG-A PARTIEL : ISP = 0 dans {pct_zero:.0f}% des QH")
        BUG_A = True
    else:
        log(f"  >>> BUG-A OK : ISP semble correct ({pct_zero:.1f}% zeros)")
        BUG_A = False

    # Distribution ISP par mois
    log()
    log("  ISP moyen par mois :")
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    monthly_isp = df.groupby("month")["isp"].agg(
        mean="mean", n_zero=lambda x: (x == 0).sum(),
        n_total="count"
    )
    for m, row in monthly_isp.iterrows():
        pct = row["n_zero"] / row["n_total"] * 100
        log(f"    {m}  mean={row['mean']:+8.2f}  zeros={int(row['n_zero']):4d}/{int(row['n_total']):4d} ({pct:.0f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TEST API ODS162 EN DIRECT
# ─────────────────────────────────────────────────────────────────────────────

subsection("TEST API ODS162 (ISP live)")

# Test sur une date 2026 recente (J-2)
test_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
log(f"  Test ODS162 pour {test_date} ...")

try:
    import pytz
    brussels_tz = pytz.timezone("Europe/Brussels")
    utc_tz      = pytz.UTC
    chunk_start = pd.Timestamp(test_date)
    chunk_end   = chunk_start + pd.Timedelta(days=1)
    s_utc = chunk_start.tz_localize(brussels_tz).tz_convert(utc_tz).strftime("%Y-%m-%dT%H:%M:%SZ")
    e_utc = chunk_end.tz_localize(brussels_tz).tz_convert(utc_tz).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ODS162 sur external-elia
    r = requests.get(
        "https://external-elia.opendatasoft.com/api/records/1.0/search/",
        params={
            "dataset": "ods162",
            "q": f"datetime:[{s_utc} TO {e_utc}]",
            "rows": 200, "start": 0, "sort": "datetime",
        }, timeout=30
    )
    data = r.json().get("records", [])
    log(f"  ODS162 records : {len(data)}")
    if data:
        fields = [rec["fields"] for rec in data[:3]]
        for f in fields:
            log(f"    {f}")
        # Cherche la colonne ISP
        sample_keys = list(data[0]["fields"].keys())
        log(f"  Colonnes disponibles : {sample_keys}")
        BUG_A_API = len(data) < 10
    else:
        log("  >>> ODS162 renvoie 0 records pour cette date -> confirme BUG-A")
        BUG_A_API = True

    # Test ODS134 (source alternative)
    log(f"\n  Test ODS134 (backup) pour {test_date} ...")
    r2 = requests.get(
        "https://opendata.elia.be/api/records/1.0/search/",
        params={
            "dataset": "ods134",
            "q": f"datetime:[{s_utc} TO {e_utc}]",
            "rows": 200, "start": 0, "sort": "datetime",
        }, timeout=30
    )
    data2 = r2.json().get("records", [])
    log(f"  ODS134 records : {len(data2)}")
    if data2:
        sample_keys2 = list(data2[0]["fields"].keys())
        log(f"  Colonnes disponibles ODS134 : {sample_keys2}")

except Exception as e:
    log(f"  EXCEPTION API : {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. BUG-B : CURTAILMENT LA NUIT
# ─────────────────────────────────────────────────────────────────────────────

section("BUG-B : Curtailment la nuit (production = 0)")

curtail_col = None
for c in ["curtail_v8_300", "curtail_300", "curtail_s2"]:
    if c in df.columns:
        curtail_col = c
        break

if curtail_col is None:
    log("  Colonne curtailment non trouvee dans le CSV.")
    log(f"  Colonnes disponibles : {[c for c in df.columns if 'curtail' in c.lower()]}")
    BUG_B = None
else:
    prod_col = "production_mw" if "production_mw" in df.columns else None
    if prod_col is None:
        log("  Colonne production_mw non trouvee.")
        BUG_B = None
    else:
        df_curtail = df[df[curtail_col] == True].copy()
        n_curtail  = len(df_curtail)
        log(f"  Total curtailments S2 : {n_curtail}")

        if n_curtail == 0:
            log("  Aucun curtailment dans le CSV.")
            BUG_B = False
        else:
            n_night = (df_curtail[prod_col] < 0.001).sum()
            n_low   = (df_curtail[prod_col] < 0.01).sum()
            n_solar = (df_curtail[prod_col] >= 0.01).sum()
            pct_night = n_night / n_curtail * 100

            log(f"  Curtailments avec prod < 0.001 MW (nuit) : {n_night:4d}  ({pct_night:.1f}%)")
            log(f"  Curtailments avec prod < 0.01  MW        : {n_low:4d}  ({n_low/n_curtail*100:.1f}%)")
            log(f"  Curtailments avec prod >= 0.01 MW        : {n_solar:4d}  ({n_solar/n_curtail*100:.1f}%)")

            # Distribution par heure
            log()
            log("  Curtailments par heure solaire :")
            df_curtail["hour"] = df_curtail["timestamp"].dt.hour
            hourly = df_curtail.groupby("hour").size()
            for h in sorted(hourly.index):
                bar = "#" * min(hourly[h], 40)
                log(f"    {h:02d}h  {hourly[h]:4d}  {bar}")

            # Revenu imbalance sur curtailments de nuit vs jour
            if "s2_revenue_imb" in df.columns:
                rev_night = df_curtail[df_curtail[prod_col] < 0.001]["s2_revenue_imb"].sum()
                rev_solar = df_curtail[df_curtail[prod_col] >= 0.01]["s2_revenue_imb"].sum()
                log(f"\n  Rev imb S2 sur curtailments nuit  : {rev_night:+.2f} EUR")
                log(f"  Rev imb S2 sur curtailments solaire: {rev_solar:+.2f} EUR")

            if pct_night > 30:
                log()
                log(f"  >>> BUG-B CONFIRME : {pct_night:.0f}% des curtailments la nuit")
                log("      FIX : conditionner curtailment sur production_mw > 0.01")
                BUG_B = True
            else:
                log(f"  >>> BUG-B LIMITE : seulement {pct_night:.0f}% la nuit")
                BUG_B = False

# ─────────────────────────────────────────────────────────────────────────────
# 5. BUG-C : aFRR/mFRR TRONQUES A 9000 ROWS
# ─────────────────────────────────────────────────────────────────────────────

section("BUG-C : aFRR/mFRR tronques (cap 9000 rows)")

for col in ["mfrr_ratio_negative", "afrr_ratio_negative"]:
    if col not in df.columns:
        log(f"  Colonne {col} absente.")
        continue

    vals = df[col]
    n_zero   = (vals == 0).sum()
    n_100    = (vals == 100).sum()
    n_nan    = vals.isna().sum()
    log(f"  {col}")
    log(f"    = 0    : {n_zero:,}  ({n_zero/len(vals)*100:.1f}%)")
    log(f"    = 100  : {n_100:,}  ({n_100/len(vals)*100:.1f}%)")
    log(f"    NaN    : {n_nan:,}")
    log(f"    mean   : {vals.mean():.1f}%")
    log(f"    median : {vals.median():.1f}%")

# Cherche MAX_START dans le scheduler
log()
if SCHEDULER.exists():
    src = SCHEDULER.read_text(encoding="utf-8", errors="replace")
    max_start_match = re.findall(r"MAX_START\s*=\s*(\d+)", src)
    paginator_matches = re.findall(r"start\s*[<>=]+\s*(\d+)", src)
    log(f"  MAX_START dans scheduler  : {max_start_match}")
    log(f"  Plafonds start trouves    : {paginator_matches[:10]}")
    if max_start_match and int(max_start_match[0]) == 9000:
        log("  >>> BUG-C CONFIRME : MAX_START = 9000 detecte dans le code")
        BUG_C = True
    elif max_start_match:
        log(f"  MAX_START = {max_start_match[0]} (a verifier)")
        BUG_C = bool(max_start_match)
    else:
        log("  MAX_START non trouve explicitement -> verifier la logique de pagination")
        BUG_C = False
else:
    log(f"  Fichier scheduler non trouve : {SCHEDULER}")
    BUG_C = None

# ─────────────────────────────────────────────────────────────────────────────
# 6. BUG-D : DA ENTSO-E TIMEOUTS (price = 0)
# ─────────────────────────────────────────────────────────────────────────────

section("BUG-D : DA ENTSO-E price = 0 (timeouts)")

if "price_eur_mwh" not in df.columns:
    log("  Colonne price_eur_mwh absente.")
    BUG_D = None
else:
    price = df["price_eur_mwh"]
    n_zero  = (price == 0).sum()
    n_neg   = (price < 0).sum()
    n_pos   = (price > 0).sum()
    log(f"  price = 0   : {n_zero:,}  ({n_zero/len(price)*100:.1f}%)")
    log(f"  price < 0   : {n_neg:,}  ({n_neg/len(price)*100:.1f}%)")
    log(f"  price > 0   : {n_pos:,}  ({n_pos/len(price)*100:.1f}%)")
    log(f"  mean / min / max : {price.mean():.1f} / {price.min():.1f} / {price.max():.1f}")

    # Jours avec tous les prix = 0
    df["date"] = df["timestamp"].dt.date
    daily_price = df.groupby("date")["price_eur_mwh"].agg(
        mean="mean", n_zero=lambda x: (x == 0).sum(), n_total="count"
    )
    days_all_zero = daily_price[daily_price["n_zero"] == daily_price["n_total"]]
    log(f"\n  Jours avec 100% des prix = 0 (timeout) : {len(days_all_zero)}")
    for d, row in days_all_zero.iterrows():
        log(f"    {d}  ({row['n_total']} QH)")

    if len(days_all_zero) > 0:
        log("  >>> BUG-D CONFIRME : timeouts ENTSO-E sans retry")
        BUG_D = True
    else:
        log("  >>> BUG-D OK : aucun jour entier a zero")
        BUG_D = False

# ─────────────────────────────────────────────────────────────────────────────
# 7. ANALYSE REVENUE S1 vs S2
# ─────────────────────────────────────────────────────────────────────────────

section("ANALYSE REVENUE S1 / S2 / S3 / S4")

for key in ["s1", "s2", "s3", "s4"]:
    if f"{key}_total" not in df.columns:
        continue
    tot   = df[f"{key}_total"].sum()
    da    = df[f"{key}_revenue_da"].sum() if f"{key}_revenue_da" in df.columns else 0
    imb   = df[f"{key}_revenue_imb"].sum() if f"{key}_revenue_imb" in df.columns else 0
    log(f"  {key.upper()}  total={tot:+9,.0f}  DA={da:+9,.0f}  imb={imb:+9,.0f} EUR")

log()
log("  Revenue imb mensuel S1/S2 :")
if "s1_revenue_imb" in df.columns and "s2_revenue_imb" in df.columns:
    monthly_rev = df.groupby("month").agg(
        s1_imb=("s1_revenue_imb", "sum"),
        s2_imb=("s2_revenue_imb", "sum"),
        s1_da=("s1_revenue_da", "sum"),
        s2_da=("s2_revenue_da", "sum"),
    )
    for m, row in monthly_rev.iterrows():
        log(f"    {m}  S1_imb={row['s1_imb']:+8.0f}  S2_imb={row['s2_imb']:+8.0f}  "
            f"S1_da={row['s1_da']:+8.0f}  S2_da={row['s2_da']:+8.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. DIAGNOSTIC DU CODE SOURCE (scheduler)
# ─────────────────────────────────────────────────────────────────────────────

section("ANALYSE CODE SOURCE solar_be_scheduler.py")

if not SCHEDULER.exists():
    log(f"  Fichier non trouve : {SCHEDULER}")
else:
    src = SCHEDULER.read_text(encoding="utf-8", errors="replace")
    log(f"  Fichier : {SCHEDULER}  ({len(src.splitlines())} lignes)")

    # Cherche logique ISP
    log()
    log("  [ISP] Lignes pertinentes :")
    isp_lines = [(i+1, l.rstrip()) for i, l in enumerate(src.splitlines())
                 if "isp" in l.lower() and ("ods" in l.lower() or "fillna" in l.lower()
                 or "load_isp" in l.lower() or "fallback" in l.lower())]
    for ln, l in isp_lines[:20]:
        log(f"    L{ln:4d}: {l}")

    # Cherche la logique curtailment
    log()
    log("  [CURTAILMENT] Logique de decision :")
    curtail_lines = [(i+1, l.rstrip()) for i, l in enumerate(src.splitlines())
                     if "curtail" in l.lower() or "_curtail" in l.lower()]
    for ln, l in curtail_lines[:20]:
        log(f"    L{ln:4d}: {l}")

    # Cherche la condition sur production dans curtailment
    has_prod_check = "production_mw" in "\n".join(
        l for l in src.splitlines()
        if "curtail" in l.lower()
    )
    log()
    if has_prod_check:
        log("  [CURTAILMENT] Condition production_mw presente dans la logique curtailment -> OK")
    else:
        log("  [CURTAILMENT] Pas de condition production_mw dans la logique curtailment -> BUG-B confirme")

    # Cherche logique pagination
    log()
    log("  [PAGINATION] Lignes pertinentes :")
    pag_lines = [(i+1, l.rstrip()) for i, l in enumerate(src.splitlines())
                 if any(x in l for x in ["MAX_START", "max_start", "9000", "9_000", "10000",
                                          "start >=", "start>", "offset", "paginator"])]
    for ln, l in pag_lines[:15]:
        log(f"    L{ln:4d}: {l}")

    # Cherche logique ENTSO-E retry
    log()
    log("  [ENTSO-E] Retry / fallback :")
    retry_lines = [(i+1, l.rstrip()) for i, l in enumerate(src.splitlines())
                   if any(x in l for x in ["retry", "ENTSO", "entsoe", "timeout",
                                            "for _ in range", "sleep"])]
    for ln, l in retry_lines[:15]:
        log(f"    L{ln:4d}: {l}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. RESUME DES BUGS ET CORRECTIONS
# ─────────────────────────────────────────────────────────────────────────────

section("RESUME BUGS ET PRIORITES")

bugs_found = []

isp_zero_pct = 0
if "isp" in df.columns:
    isp_zero_pct = (df["isp"] == 0).sum() / len(df) * 100

if isp_zero_pct > 50:
    bugs_found.append({
        "id": "BUG-A",
        "priorite": 1,
        "titre": "ISP = 0 partout",
        "impact": f"Imbalance revenue = 0 (ISP nul dans {isp_zero_pct:.0f}% des QH)",
        "cause": "ODS162 renvoie vide pour dates historiques 2026, fillna(0) ecrase tout",
        "fix": (
            "Dans load_isp_day() :\n"
            "  1. Tester ODS162 (external-elia) -> si < 10 records, fallback ODS134\n"
            "  2. Si ODS134 aussi vide, fallback ISP depuis forecast log Elia\n"
            "  3. Ajouter assertion : si df_isp vide ou tout a zero -> skip le jour\n"
            "     (ne pas imputer 0 -> biaise tous les revenus)\n"
            "  Code :\n"
            "    df_isp = load_isp_ods162(date_str)\n"
            "    if df_isp.empty or (df_isp['isp'] == 0).all():\n"
            "        df_isp = load_isp_ods134(date_str)  # fallback\n"
            "    if df_isp.empty:\n"
            "        skipped.append((date_str, 'ISP vide')); continue"
        )
    })

if BUG_B:
    bugs_found.append({
        "id": "BUG-B",
        "priorite": 2,
        "titre": "Curtailment la nuit",
        "impact": "Curtailments ne generent pas de revenu imbalance (production = 0 la nuit)",
        "cause": "Signal DOWN peut se declencher 24h/24, PVGIS = 0 la nuit -> ecart ≈ 0",
        "fix": (
            "Dans _curtail_v8() ou avant l'appel :\n"
            "  Ajouter condition : production_mw > 0.01 MW (environ 1% du pic)\n"
            "  Code :\n"
            "    curtail_300 = _curtail_v8(...) AND (production_mw > 0.01)\n"
            "  OU dans _curtail_v8 :\n"
            "    def _curtail_v8(fc_dir, fc_vol, mfrr, afrr, prod_mw, seuil=300):\n"
            "        if prod_mw < 0.01: return False  # <- ajouter cette ligne\n"
            "        if fc_dir != 'DOWN': return False\n"
            "        ..."
        )
    })

if BUG_C:
    bugs_found.append({
        "id": "BUG-C",
        "priorite": 3,
        "titre": "aFRR/mFRR tronques",
        "impact": "Metriques calculees sur stack incomplete -> signaux curtailment incorrects",
        "cause": "MAX_START=9000 dans le paginator (limite 10 000 rows API)",
        "fix": (
            "Chunking journalier dans le download aFRR/mFRR :\n"
            "  Ne plus charger toute la periode en une requete.\n"
            "  Boucler par jour ou par demi-journee.\n"
            "  Supprimer MAX_START (ou augmenter a None / sans limite).\n"
            "  Code :\n"
            "    all_rows = []\n"
            "    idx = 0\n"
            "    while True:\n"
            "        r = requests.get(..., params={..., 'start': idx})\n"
            "        data = r.json()['records']\n"
            "        if not data: break\n"
            "        all_rows.extend(data)\n"
            "        idx += len(data)\n"
            "        # SANS cap sur idx"
        )
    })

if BUG_D:
    bugs_found.append({
        "id": "BUG-D",
        "priorite": 4,
        "titre": "DA ENTSO-E timeouts",
        "impact": f"Prix DA = 0 sur {len(days_all_zero)} jours entiers -> S1=S2 ces jours",
        "cause": "Pas de retry adequat sur l'API ENTSO-E",
        "fix": (
            "Dans load_da_day() :\n"
            "  Augmenter retries a 5, timeout a 60s, sleep exponentiel.\n"
            "  Ajouter source backup : ENTSOE REST API v2 ou cache EPEX spot.\n"
            "  Code :\n"
            "    for attempt in range(5):\n"
            "        try:\n"
            "            r = requests.get(..., timeout=60)\n"
            "            if r.status_code == 200 and len(rows) >= 24: break\n"
            "        except:\n"
            "            pass\n"
            "        time.sleep(2 ** attempt)  # backoff exponentiel"
        )
    })

for bug in bugs_found:
    log()
    log(f"  [{bug['id']}] PRIORITE {bug['priorite']} : {bug['titre']}")
    log(f"    Impact  : {bug['impact']}")
    log(f"    Cause   : {bug['cause']}")
    log("    Fix     :")
    for line in bug['fix'].split("\n"):
        log(f"      {line}")

if not bugs_found:
    log("  Aucun bug majeur detecte depuis les donnees CSV.")
    log("  Verifier les logs d'execution du batch pour d'autres problemes.")

# ─────────────────────────────────────────────────────────────────────────────
# 10. PATCH AUTOMATIQUE DU SCHEDULER (BUG-B)
# ─────────────────────────────────────────────────────────────────────────────

section("PATCH AUTOMATIQUE BUG-B (curtailment + condition production)")

if not SCHEDULER.exists():
    log("  Scheduler non trouve, patch impossible.")
elif not BUG_B:
    log("  BUG-B non confirme, patch non applique.")
else:
    src = SCHEDULER.read_text(encoding="utf-8", errors="replace")
    backup = SCHEDULER.with_suffix(".py.bak_bugB")
    backup.write_text(src, encoding="utf-8")
    log(f"  Backup cree : {backup}")

    # Cherche la signature _curtail_v8 et ajoute la condition prod
    old_sig = 'def _curtail_v8(fc_direction, fc_volume, mfrr, afrr, vol_seuil=300):'
    new_sig = 'def _curtail_v8(fc_direction, fc_volume, mfrr, afrr, vol_seuil=300, production_mw=0.0):'

    old_guard = '    if fc_direction != "DOWN": return False'
    new_guard  = ('    if production_mw < 0.01: return False  # BUG-B fix: pas de curtailment la nuit\n'
                  '    if fc_direction != "DOWN": return False')

    if old_sig in src:
        src = src.replace(old_sig, new_sig)
        log(f"  Signature _curtail_v8 mise a jour.")
    else:
        log("  ATTENTION : signature _curtail_v8 non trouvee exactement.")
        log("  Cherche une variante...")
        # Cherche la fonction avec regex
        if "_curtail_v8" in src:
            log("  _curtail_v8 presente mais signature differente -> patch manuel requis")

    if old_guard in src:
        src = src.replace(old_guard, new_guard)
        log("  Condition production_mw < 0.01 ajoutee dans _curtail_v8.")
    else:
        log("  Guard 'if fc_direction != DOWN' non trouve exactement.")
        log("  Patch BUG-B a appliquer manuellement (voir instructions ci-dessus).")

    # Met a jour les appels a _curtail_v8 pour passer production_mw
    # Cherche les patterns d'appel courants
    old_call_300 = ('df["curtail_v8_300"] = df.apply(lambda r: _curtail_v8(\n'
                    '        r["forecast_direction"], r["forecast_volume"],\n'
                    '        r["mfrr_ratio_negative"], r["afrr_ratio_negative"],\n'
                    '        vol_seuil=300), axis=1)')
    new_call_300 = ('df["curtail_v8_300"] = df.apply(lambda r: _curtail_v8(\n'
                    '        r["forecast_direction"], r["forecast_volume"],\n'
                    '        r["mfrr_ratio_negative"], r["afrr_ratio_negative"],\n'
                    '        vol_seuil=300, production_mw=r.get("production_mw", 0.0)), axis=1)')

    old_call_150 = ('df["curtail_v8_150"] = df.apply(lambda r: _curtail_v8(\n'
                    '        r["forecast_direction"], r["forecast_volume"],\n'
                    '        r["mfrr_ratio_negative"], r["afrr_ratio_negative"],\n'
                    '        vol_seuil=150), axis=1)')
    new_call_150 = ('df["curtail_v8_150"] = df.apply(lambda r: _curtail_v8(\n'
                    '        r["forecast_direction"], r["forecast_volume"],\n'
                    '        r["mfrr_ratio_negative"], r["afrr_ratio_negative"],\n'
                    '        vol_seuil=150, production_mw=r.get("production_mw", 0.0)), axis=1)')

    if old_call_300 in src:
        src = src.replace(old_call_300, new_call_300)
        log("  Appel curtail_v8_300 mis a jour avec production_mw.")
    else:
        log("  Appel curtail_v8_300 non trouve exactement -> cherche variante single-line...")
        # Single-line version
        m300 = re.search(
            r'df\["curtail_v8_300"\]\s*=\s*df\.apply\([^)]+vol_seuil=300\)[^)]*,\s*axis=1\)',
            src
        )
        if m300:
            old_m = m300.group(0)
            new_m = old_m.replace(
                "vol_seuil=300)",
                "vol_seuil=300, production_mw=r.get('production_mw', 0.0))"
            )
            src = src.replace(old_m, new_m)
            log("  Appel curtail_v8_300 (variante) mis a jour.")

    if old_call_150 in src:
        src = src.replace(old_call_150, new_call_150)
        log("  Appel curtail_v8_150 mis a jour avec production_mw.")

    SCHEDULER.write_text(src, encoding="utf-8")
    log(f"  Fichier {SCHEDULER.name} mis a jour.")

# ─────────────────────────────────────────────────────────────────────────────
# 11. INSTRUCTIONS PATCH BUG-A (ISP) — A APPLIQUER MANUELLEMENT
# ─────────────────────────────────────────────────────────────────────────────

section("INSTRUCTIONS PATCH BUG-A (ISP) — MANUEL")

log("""
  Dans solar_be_scheduler.py, trouve la fonction load_isp_day() et remplace
  la logique de fallback par ceci :

  def load_isp_day(date_str):
      # 1. Essaie ODS162 sur external-elia
      df = _ods_download_day_isp("ods162", date_str, "imbalanceprice", "isp",
                                  host="external-elia.opendatasoft.com")
      if not df.empty and not (df["isp"] == 0).all() and len(df) >= 10:
          return reindex_day(df.rename(columns={"datetime": "timestamp"}),
                             date_str, "ffill")

      # 2. Fallback ODS134 sur opendata.elia.be
      df = _ods_download_day_isp("ods134", date_str, "imbalanceprice", "isp",
                                  host="opendata.elia.be")
      if not df.empty and not (df["isp"] == 0).all() and len(df) >= 10:
          return reindex_day(df.rename(columns={"datetime": "timestamp"}),
                             date_str, "ffill")

      # 3. Fallback ODS162 avec colonne differente (verifier les noms de colonnes)
      # Parfois la colonne s'appelle 'marginalincrementalprice' ou 'price'
      for col_name in ["marginalincrementalprice", "price", "systemimbalanceprice"]:
          df = _ods_download_day_isp("ods162", date_str, col_name, "isp",
                                      host="external-elia.opendatasoft.com")
          if not df.empty and not (df["isp"] == 0).all():
              return reindex_day(df.rename(columns={"datetime": "timestamp"}),
                                 date_str, "ffill")

      print(f"  WARNING ISP vide pour {date_str} -> jour skippe")
      return pd.DataFrame()  # <- ne pas retourner des zeros

  IMPORTANT : le script retourne pd.DataFrame() vide si ISP absent.
  Dans run_batch(), verifier :
      df_isp = load_isp_day(ds)
      if df_isp.empty:
          skipped.append((ds, "ISP_vide")); continue  # <- sauter le jour
""")

# ─────────────────────────────────────────────────────────────────────────────
# 12. INSTRUCTIONS PATCH BUG-C (aFRR/mFRR pagination) — A APPLIQUER
# ─────────────────────────────────────────────────────────────────────────────

section("INSTRUCTIONS PATCH BUG-C (pagination aFRR/mFRR) — MANUEL")

log("""
  Dans la fonction de download aFRR/mFRR (cherche MAX_START ou la boucle
  qui telecharge les bids BE) :

  AVANT (bugge) :
      MAX_START = 9000
      while idx < MAX_START:
          ...
          idx += 1000

  APRES (corrige) :
      # Pas de cap : boucle jusqu'a ce que l'API retourne 0 records
      while True:
          r = requests.get(ELIA_URL, params={
              "dataset": dataset,
              "q": f"datetime:[{s_utc} TO {e_utc}]",
              "rows": 1000,
              "start": idx,
              "sort": "datetime",
          }, timeout=60)
          r.raise_for_status()
          data = r.json().get("records", [])
          if not data:
              break
          all_records.extend(data)
          if len(data) < 1000:
              break
          idx += 1000
          # Pas de cap sur idx

  Note : pour les bids BE aFRR/mFRR, la requete en journalier (1 jour a la fois)
  est plus fiable que sur une longue periode.
  Si le batch telecharge plusieurs jours en une seule requete, decouper par jour.
""")

# ─────────────────────────────────────────────────────────────────────────────
# 13. SAVE REPORT
# ─────────────────────────────────────────────────────────────────────────────

section("RAPPORT SAUVEGARDE")

REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.write_text("\n".join(lines_report), encoding="utf-8")
log(f"  Rapport complet -> {REPORT_FILE}")
log()
log("  PROCHAINES ETAPES :")
log("  1. Lire le rapport : debug_report_2026.txt")
log("  2. Corriger BUG-A (ISP) manuellement dans load_isp_day()")
log("  3. Verifier patch BUG-B applique (curtailment + production_mw)")
log("  4. Corriger BUG-C (pagination) si MAX_START detecte")
log("  5. Relancer le batch 2026 : python solar_be_scheduler.py --batch ...")
log("  6. Comparer les nouveaux totaux avec les resultats 2025")
