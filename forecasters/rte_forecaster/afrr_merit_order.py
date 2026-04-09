# -*- coding: utf-8 -*-
# EU aFRR merit-order – DE / BE / NL
# Téléchargement (DE/BE/NL) + Construction de 2 tableaux (ALL_DAY et UNTIL_DE)
# + CLI: displayall / displayde
#
# Modifs clés :
# - ALL_DAY = BE + NL (sans DE)
# - UNTIL_DE = BE + DE + NL
# - BE (Elia) : pagination (rows/start) + sort=datetime pour couvrir 24h
# - DE : sélection du dernier RESULT_LIST*.xlsx
# - NL : 00:00→24:00
# - Sorties en .xlsx
# - display* ne télécharge pas

import os, time, glob, shutil, tempfile, argparse, sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from io import StringIO

# ========================
# Paramètres généraux
# ========================
TIMEZONE = "Europe/Brussels"
TODAY = datetime.now()
DATE_STR = TODAY.strftime("%Y-%m-%d")
START_DATE = pd.Timestamp(DATE_STR)

# Dossiers & fichiers (Téléchargements UNIQUEMENT)
DOWNLOAD_DIR = os.path.join(os.path.expanduser('~'), 'Downloads')
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

PATH_BE_INC = os.path.join(DOWNLOAD_DIR, f"afrr_incremental_bids_BE_{DATE_STR}.csv")
PATH_BE_DEC = os.path.join(DOWNLOAD_DIR, f"afrr_decremental_bids_BE_{DATE_STR}.csv")

# DE : on ne cherche QUE RESULT_LIST*.xlsx
PATH_DE_XLSX_GLOB = os.path.join(DOWNLOAD_DIR, "RESULT_LIST*.xlsx")
PATH_DE_FINAL = os.path.join(DOWNLOAD_DIR, f"afrr_anonymousresults_DE_{DATE_STR}.xlsx")

PATH_NL = os.path.join(DOWNLOAD_DIR, f"tennet_merit_order_full_{DATE_STR.replace('-', '')}.csv")

# Sorties en EXCEL
OUT_ALL_XLSX = os.path.join(DOWNLOAD_DIR, f"eu_merit_order_{DATE_STR}_ALL_DAY.xlsx")
OUT_DE_XLSX  = os.path.join(DOWNLOAD_DIR, f"eu_merit_order_{DATE_STR}_UNTIL_DE.xlsx")

THRESHOLDS = [25, 200, 400, 600, 800, 1000]

# ========================
# 1) Téléchargement DONNÉES
# ========================

# --- DE via Selenium (xlsx dans Téléchargements)
def download_de_xlsx(download_dir: str, date_str: str, wait_s: int = 10):
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except Exception as e:
        raise RuntimeError(f"Selenium non disponible : {e}")

    chrome_profile_dir = tempfile.mkdtemp(prefix="selenium_chrome_profile_")
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--no-first-run")
    opts.add_argument("--no-default-browser-check")
    opts.add_argument(f"--user-data-dir={chrome_profile_dir}")
    opts.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    driver = webdriver.Chrome(options=opts)
    try:
        target_url = f"https://www.regelleistung.net/apps/datacenter/tenders/?productTypes=SRL&markets=BALANCING_ENERGY&date={date_str}"
        driver.get(target_url)
        time.sleep(5)
        links = driver.find_elements("xpath", '//a[contains(@href, "anonymousresults") and contains(@href, "xlsx")]')
        if not links:
            print("❌ [DE] Aucun lien de téléchargement trouvé.")
        else:
            file_url = links[0].get_attribute('href')
            print(f"✅ [DE] Téléchargement depuis : {file_url}")
            driver.get(file_url)
            time.sleep(wait_s)
    finally:
        driver.quit()

    # attente simple de fin de téléchargement
    for _ in range(30):
        if glob.glob(os.path.join(download_dir, "*.crdownload")):
            time.sleep(1)
        else:
            break

    # Utiliser uniquement des fichiers commençant par RESULT_LIST
    candidates = glob.glob(PATH_DE_XLSX_GLOB)
    if not candidates:
        raise RuntimeError("❌ [DE] Aucun fichier RESULT_LIST*.xlsx dans Téléchargements.")
    latest = sorted(candidates, key=os.path.getmtime)[-1]

    # copie vers un nom fixe du jour (pour la suite du script)
    if os.path.abspath(latest) != os.path.abspath(PATH_DE_FINAL):
        shutil.copy2(latest, PATH_DE_FINAL)
    print(f"📄 [DE] Fichier prêt (RESULT_LIST* détecté) : {PATH_DE_FINAL}")
    return PATH_DE_FINAL

# --- BE via Elia API (2 CSV) — PAGINATION + TRI → 24h complètes
def download_be_csvs(date_str: str, out_inc: str, out_dec: str):
    """
    Récupère 24h complètes pour la date `date_str` via pagination (rows/start)
    sur les datasets Elia:
      - incremental: ods163
      - decremental: ods164
    Tri explicite: sort=datetime (croissant).
    Filtre: refine.date_start=YYYY-MM-DD ; refine.balancingproduct=aFRR
    """
    base_url = 'https://opendata.elia.be/api/records/1.0/search/'
    DATASETS = [
        ('incremental', 'ods163', out_inc),
        ('decremental', 'ods164', out_dec),
    ]
    ROWS = 1000
    SLEEP = 0.8  # petit délai pour respecter l'API

    for direction, dataset, out_path in DATASETS:
        all_frames = []
        start = 0
        page = 1
        while True:
            params = {
                'dataset': dataset,
                'rows': ROWS,
                'start': start,
                'refine.date_start': date_str,
                'refine.balancingproduct': 'aFRR',
                'sort': 'datetime',  # tri croissant
            }
            r = requests.get(base_url, params=params)
            if r.status_code == 429:
                time.sleep(3)
                r = requests.get(base_url, params=params)

            if r.status_code != 200:
                raise RuntimeError(f"❌ [BE] {direction}: Erreur {r.status_code} – {r.url} – {r.text[:200]}")

            payload = r.json()
            records = payload.get('records', [])
            if not records:
                break  # plus de pages

            data = [rec.get('fields', {}) for rec in records]
            df_page = pd.DataFrame(data)

            if 'datetime' in df_page.columns:
                df_page['datetime'] = pd.to_datetime(df_page['datetime'], errors='coerce', utc=True)

            all_frames.append(df_page)
            print(f"✅ [BE] {direction} page {page}: {len(df_page)} lignes (start={start})")
            page += 1
            start += ROWS
            time.sleep(SLEEP)

        if not all_frames:
            pd.DataFrame().to_csv(out_path, index=False)
            print(f"⚠️ [BE] {direction}: 0 ligne pour {date_str} → {out_path}")
            continue

        final_df = pd.concat(all_frames, ignore_index=True)
        if 'datetime' in final_df.columns:
            final_df = final_df.sort_values('datetime').reset_index(drop=True)

        final_df.to_csv(out_path, index=False)
        print(f"📄 [BE] {direction}: {len(final_df)} lignes (toutes pages) -> {out_path}")

# --- NL via TenneT API (CSV fusionné 24h) — 00:00→24:00 du jour courant
def download_nl_csv_for_day(date_str: str, out_path: str, api_key: str = "18fe140e-12a2-446e-ab89-455d33709fac"):
    url = "https://api.tennet.eu/publications/v1/merit-order-list"
    headers = {"apikey": api_key, "Accept": "text/csv"}
    MAX_RETRIES, SLEEP_SECONDS = 3, 6
    base = datetime.strptime(date_str + " 00:00:00", "%Y-%m-%d %H:%M:%S")

    all_dfs = []
    for hour in range(24):
        from_time = (base + timedelta(hours=hour)).strftime("%d-%m-%Y %H:%M:%S")
        to_time   = (base + timedelta(hours=hour+1)).strftime("%d-%m-%Y %H:%M:%S")
        params = {"date_from": from_time, "date_to": to_time}
        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(url, headers=headers, params=params)
                if resp.status_code == 200 and resp.content.strip():
                    df = pd.read_csv(StringIO(resp.content.decode("utf-8")))
                    all_dfs.append(df)
                    print(f"✅ [NL] {from_time} → {to_time} (tentative {attempt})")
                    success = True
                    break
                elif resp.status_code == 429:
                    print(f"⏳ [NL] Rate limit {from_time} (tentative {attempt})")
                    time.sleep(SLEEP_SECONDS * 2)
                else:
                    print(f"❌ [NL] Erreur {resp.status_code}: {resp.text[:120]}")
                    break
            except Exception as e:
                print(f"⚠️ [NL] Exception @ {from_time}: {e}")
            time.sleep(SLEEP_SECONDS)
        if not success:
            print(f"❌ [NL] Échec définitif @ {from_time}")
    if not all_dfs:
        raise RuntimeError("❌ [NL] Aucune donnée récupérée.")
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(out_path, index=False)
    print(f"📄 [NL] Fichier fusionné : {out_path}")
    return out_path

# ========================
# 2) Construction des tableaux
# ========================

def load_and_build():
    print("📥 Téléchargements…")
    path_de = download_de_xlsx(DOWNLOAD_DIR, DATE_STR, wait_s=10)
    download_be_csvs(DATE_STR, PATH_BE_INC, PATH_BE_DEC)
    download_nl_csv_for_day(DATE_STR, PATH_NL)
    print("✅ Téléchargements terminés.\n")

    print("📊 Prétraitement & construction des piles…")
    # --- Chargement
    inc_be = pd.read_csv(PATH_BE_INC)
    dec_be = pd.read_csv(PATH_BE_DEC)
    xde = pd.read_excel(path_de, sheet_name=0)
    mo = pd.read_csv(PATH_NL)

    # --- DE (robuste)
    xde['QH_IDX'] = xde['PRODUCT'].astype(str).str.extract(r'(\d{3})').astype(float).astype('Int64')
    xde = xde.dropna(subset=['QH_IDX']).copy()
    xde['QH_IDX'] = xde['QH_IDX'].astype(int)

    # détecte la colonne date et fallback si besoin
    date_candidates = [c for c in xde.columns if 'DELIVERY' in c.upper() and 'DATE' in c.upper()]
    if 'DELIVERY_DATE' not in date_candidates and 'DELIVERY_DATE' in xde.columns:
        date_candidates.append('DELIVERY_DATE')
    if 'DATE' in xde.columns and 'DATE' not in date_candidates:
        date_candidates.append('DATE')

    delivery = None
    date_col_used = None
    for c in date_candidates:
        try:
            col = pd.to_datetime(xde[c], errors='coerce')
            if col.notna().sum() > 0:
                delivery = col
                date_col_used = c
                break
        except Exception:
            pass
    if delivery is None:
        delivery = pd.Series(pd.to_datetime(DATE_STR), index=xde.index)
        date_col_used = '(fallback DATE_STR)'

    xde['DeliveryDate'] = delivery.dt.normalize()
    xde['Timestamp'] = xde['DeliveryDate'] + pd.to_timedelta(xde['QH_IDX'] * 15, unit='m')

    day = pd.Timestamp(DATE_STR).date()
    mask_today = xde['DeliveryDate'].dt.date == day
    xde_day = xde[mask_today].copy()
    if xde_day.empty:
        vc = xde['DeliveryDate'].dt.date.value_counts()
        if not vc.empty:
            dominant_day = vc.index[0]
            xde_day = xde[xde['DeliveryDate'].dt.date == dominant_day].copy()
            print(f"⚠️ [DE] Aucune ligne exactement à {DATE_STR}. Utilisation de la date dominante du fichier: {dominant_day}")
        else:
            raise RuntimeError("❌ Aucun enregistrement DE dans le fichier (pas de dates valides).")

    latest_ts_de = xde_day['Timestamp'].max()
    qh_min, qh_max = int(xde_day['QH_IDX'].min()), int(xde_day['QH_IDX'].max())
    print(f"🕒 Dernier quart d’heure DE disponible : {latest_ts_de} (QH {qh_min}→{qh_max}, source='{date_col_used}')")

    # --- NL
    time_col = 'Timeinterval Start Loc' if 'Timeinterval Start Loc' in mo.columns else ('Timeinterval Start (Local Time)' if 'Timeinterval Start (Local Time)' in mo.columns else None)
    if time_col is None:
        raise RuntimeError("❌ [NL] Colonne de temps introuvable dans le CSV.")
    mo['Start_dt'] = pd.to_datetime(mo[time_col], errors='coerce').dt.tz_localize(None)

    # --- BE
    for df in (inc_be, dec_be):
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
        df['Datetime_parsed'] = df['datetime'].dt.tz_convert(TIMEZONE).dt.tz_localize(None)

    inc_be_day = inc_be[inc_be['balancingproduct'] == 'aFRR'].copy()
    dec_be_day = dec_be[dec_be['balancingproduct'] == 'aFRR'].copy()

    # --- fonction de calcul des prix (paramètre include_de)
    def compute_prices_eu(ts: pd.Timestamp, include_de: bool):
        # BE (QH exact)
        inc_b = inc_be_day[inc_be_day['Datetime_parsed'].dt.floor('15min') == ts][['energybidmarginalprice', 'energybidvolume']].copy()
        inc_b.columns = ['Bid Price', 'Bid Volume']
        dec_b = dec_be_day[dec_be_day['Datetime_parsed'].dt.floor('15min') == ts][['energybidmarginalprice', 'energybidvolume']].copy()
        dec_b.columns = ['Bid Price', 'Bid Volume']

        # NL (déjà quart-horaire)
        sel = mo[mo['Start_dt'] == ts.replace(second=0, microsecond=0)].copy()
        inc_nl = pd.DataFrame()
        dec_nl = pd.DataFrame()
        if not sel.empty and 'Capacity Threshold' in sel.columns:
            sel = sel.sort_values('Capacity Threshold').copy()
            sel['PrevCap'] = sel['Capacity Threshold'].shift(1).fillna(0)
            sel['Bid Volume'] = sel['Capacity Threshold'] - sel['PrevCap']
            if 'Price Up' in sel.columns:
                inc_nl = sel[['Price Up', 'Bid Volume']].rename(columns={'Price Up': 'Bid Price'})
            if 'Price Down' in sel.columns:
                dec_nl = sel[['Price Down', 'Bid Volume']].rename(columns={'Price Down': 'Bid Price'})

        # DE (QH exact) — uniquement si include_de=True
        inc_ge = pd.DataFrame()
        dec_ge = pd.DataFrame()
        if include_de:
            inc_ge = xde_day[(xde_day['Timestamp'] == ts) & xde_day['PRODUCT'].astype(str).str.startswith('POS')][[
                'ENERGY_PRICE_[EUR/MWh]', 'ENERGY_PRICE_PAYMENT_DIRECTION', 'ALLOCATED_CAPACITY_[MW]'
            ]].copy()
            if not inc_ge.empty:
                inc_ge.loc[inc_ge['ENERGY_PRICE_PAYMENT_DIRECTION'] == 'PROVIDER_TO_GRID', 'ENERGY_PRICE_[EUR/MWh]'] *= -1
                inc_ge = inc_ge.rename(columns={'ENERGY_PRICE_[EUR/MWh]': 'Bid Price', 'ALLOCATED_CAPACITY_[MW]': 'Bid Volume'})[['Bid Price', 'Bid Volume']]

            dec_ge = xde_day[(xde_day['Timestamp'] == ts) & xde_day['PRODUCT'].astype(str).str.startswith('NEG')][[
                'ENERGY_PRICE_[EUR/MWh]', 'ENERGY_PRICE_PAYMENT_DIRECTION', 'ALLOCATED_CAPACITY_[MW]'
            ]].copy()
            if not dec_ge.empty:
                dec_ge.loc[dec_ge['ENERGY_PRICE_PAYMENT_DIRECTION'] == 'GRID_TO_PROVIDER', 'ENERGY_PRICE_[EUR/MWh]'] *= -1
                dec_ge = dec_ge.rename(columns={'ENERGY_PRICE_[EUR/MWh]': 'Bid Price', 'ALLOCATED_CAPACITY_[MW]': 'Bid Volume'})[['Bid Price', 'Bid Volume']]

        # Piles EU — inc/dec indépendants
        if include_de:
            inc_stack = pd.concat([inc_b, inc_ge, inc_nl], ignore_index=True).dropna()
            dec_stack = pd.concat([dec_b, dec_ge, dec_nl], ignore_index=True).dropna()
        else:
            inc_stack = pd.concat([inc_b, inc_nl], ignore_index=True).dropna()
            dec_stack = pd.concat([dec_b, dec_nl], ignore_index=True).dropna()

        # Ordonner & cumsum si non vides
        if not inc_stack.empty:
            inc_stack = inc_stack.sort_values('Bid Price').reset_index(drop=True)
            inc_stack['Bid Volume'] = inc_stack['Bid Volume'].astype(float)
            inc_stack['CumVol'] = inc_stack['Bid Volume'].cumsum()
        if not dec_stack.empty:
            dec_stack = dec_stack.sort_values('Bid Price', ascending=False).reset_index(drop=True)
            dec_stack['Bid Volume'] = dec_stack['Bid Volume'].astype(float)
            dec_stack['CumVol'] = dec_stack['Bid Volume'].cumsum()

        def price_at(df, thresh):
            if df is None or df.empty or 'CumVol' not in df:
                return np.nan
            sub = df[df['CumVol'] <= float(thresh)]
            if sub.empty:
                return np.nan
            return float(sub['Bid Price'].iloc[-1])

        out = []
        for th in THRESHOLDS:
            out.append(price_at(inc_stack, th))  # Inc
            out.append(price_at(dec_stack, th))  # Dec
        return out

    # --- construction ALL_DAY et UNTIL_DE
    def build_table(ts_end: pd.Timestamp, include_de: bool):
        stamps = pd.date_range(start=START_DATE, end=ts_end, freq='15T')
        recs = []
        for ts in stamps:
            row = {'Timestamp': ts}
            prices = compute_prices_eu(ts, include_de=include_de)
            for i, th in enumerate(THRESHOLDS):
                row[f'Inc_{th}'] = prices[2*i]
                row[f'Dec_{th}'] = prices[2*i+1]
            recs.append(row)
        return pd.DataFrame(recs)

    # tableau 1 : ALL_DAY (BE+NL uniquement)
    all_day_end = START_DATE + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    eu_merit_order_ALL_DAY = build_table(all_day_end, include_de=False)

    # tableau 2 : UNTIL_DE (BE+DE+NL jusqu'au dernier QH DE)
    eu_merit_order_UNTIL_DE = build_table(latest_ts_de, include_de=True)

    # Sauvegardes en EXCEL
    eu_merit_order_ALL_DAY.to_excel(OUT_ALL_XLSX, index=False)
    eu_merit_order_UNTIL_DE.to_excel(OUT_DE_XLSX, index=False)

    print("\n✅ Tableaux générés (Excel).")
    print(f"🕒 Dernier QH DE : {latest_ts_de}")
    print(f"📄 ALL_DAY  -> {OUT_ALL_XLSX}")
    print(f"📄 UNTIL_DE -> {OUT_DE_XLSX}")

    return OUT_ALL_XLSX, OUT_DE_XLSX

# ========================
# 3) CLI : displayall / displayde
# ========================

def display_xlsx(path: str, rows: int = 20):
    if not os.path.exists(path):
        print(f"❌ Fichier introuvable : {path}")
        return 1
    try:
        df = pd.read_excel(path)
        print(f"\n=== {os.path.basename(path)} (rows={len(df)}, cols={len(df.columns)}) ===")
        print("\n--- HEAD ---")
        print(df.head(rows).to_string(index=False))
        print("\n--- TAIL ---")
        print(df.tail(rows).to_string(index=False))
        print("\nColonnes :", ", ".join(df.columns))
        return 0
    except Exception as e:
        print(f"⚠️ Erreur lecture Excel: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description="EU aFRR merit-order (DE/BE/NL): téléchargement + construction + affichage"
    )
    parser.add_argument("command", nargs="?", default="run",
                        choices=["run", "displayall", "displayde"],
                        help="run = télécharger & (re)générer; displayall = afficher le tableau ALL_DAY; displayde = afficher le tableau UNTIL_DE")
    parser.add_argument("--rows", type=int, default=20)
    args = parser.parse_args()

    if args.command == "run":
        load_and_build()
        return 0
    elif args.command == "displayall":
        return display_xlsx(OUT_ALL_XLSX, rows=args.rows)
    elif args.command == "displayde":
        return display_xlsx(OUT_DE_XLSX, rows=args.rows)
    else:
        print("Commande inconnue.")
        return 2

if __name__ == "__main__":
    sys.exit(main())
