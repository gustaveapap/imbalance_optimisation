# -*- coding: utf-8 -*-
# EU aFRR merit-order – DE / BE / NL
# VERSION DEBUG CONDITIONNEL : Logs détaillés UNIQUEMENT si erreur

import os, time, glob, shutil, tempfile, argparse, sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import requests
from io import StringIO
from pathlib import Path

# ========================
# Paramètres généraux
# ========================
TIMEZONE = "Europe/Brussels"

HOME = Path(os.path.expanduser('~'))
DOWNLOAD_DIR = HOME / 'Downloads'
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [25, 200, 400, 600, 800, 1000]

def acquire_lock():
    """Acquiert un lock pour éviter les exécutions parallèles"""
    DATE_STR = datetime.now().strftime("%Y-%m-%d")
    LOCKFILE = DOWNLOAD_DIR / f".afrr_merit_order_{DATE_STR}.lock"
    
    if LOCKFILE.exists():
        try:
            age = time.time() - LOCKFILE.stat().st_mtime
            if age > 3600:
                LOCKFILE.unlink(missing_ok=True)
        except Exception:
            pass
    try:
        fd = os.open(str(LOCKFILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False

def release_lock():
    """Libère le lock"""
    DATE_STR = datetime.now().strftime("%Y-%m-%d")
    LOCKFILE = DOWNLOAD_DIR / f".afrr_merit_order_{DATE_STR}.lock"
    try:
        LOCKFILE.unlink(missing_ok=True)
    except Exception:
        pass

def file_is_recent(path: Path, max_age_minutes: int = 45, min_size_bytes: int = 100) -> bool:
    """
    Vérifie qu'un fichier est récent ET a une taille minimale valide.
    
    Args:
        path: Chemin du fichier
        max_age_minutes: Age maximum en minutes (défaut 45)
        min_size_bytes: Taille minimale en octets (défaut 100)
    
    Returns:
        True si fichier existe, récent ET taille >= min_size_bytes
    """
    if not path.exists():
        return False
    
    # Vérifier l'âge
    age_ok = (time.time() - path.stat().st_mtime) < (max_age_minutes * 60)
    
    # Vérifier la taille
    size_ok = path.stat().st_size >= min_size_bytes
    
    return age_ok and size_ok

# ============ CLASSE POUR CAPTURER LES LOGS DEBUG ============
class DebugLogger:
    """Capture les logs DEBUG et ne les affiche qu'en cas d'erreur"""
    def __init__(self):
        self.logs = []
    
    def log(self, message):
        self.logs.append(message)
    
    def dump_on_error(self):
        """Affiche tous les logs capturés (appelé en cas d'erreur)"""
        for log in self.logs:
            print(log)
    
    def clear(self):
        self.logs = []

# ============ 1) TELECHARGEMENTS ============

def download_de_xlsx(download_dir: str, date_str: str, wait_s: int = 10):
    import requests, time, os, glob, shutil, tempfile
    api_url = "https://www.regelleistung.net/apps/cpp-publisher/api/v2/tenders/results/anonymous"
    params = {
        "productType": "aFRR",
        "market": "ENERGY",
        "exportFormat": "xlsx",
        "deliveryDate": date_str
    }

    try:
        print(f"[DE] Tentative API directe pour {date_str} ...")
        r = requests.get(api_url, params=params, timeout=60)
        if r.status_code == 200 and b'PK' in r.content[:4]:
            out_path = os.path.join(download_dir, f"afrr_anonymousresults_DE_{date_str}.xlsx")
            with open(out_path, "wb") as f:
                f.write(r.content)
            print(f"[DE] Fichier téléchargé via API : {out_path}")
            return out_path
        else:
            print(f"[DE] API directe a échoué ({r.status_code}), bascule vers Selenium.")
    except Exception as e:
        print(f"[DE] Erreur API directe : {e}. Bascule Selenium.")

    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except Exception as e:
        raise RuntimeError(f"Selenium non disponible : {e}")

    chrome_profile_dir = tempfile.mkdtemp(prefix="selenium_chrome_profile_")
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-first-run")
    opts.add_argument("--no-default-browser-check")
    opts.add_argument(f"--user-data-dir={chrome_profile_dir}")
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    opts.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(options=opts)

    try:
        target_url = f"https://www.regelleistung.net/apps/datacenter/tenders/?productTypes=SRL&markets=BALANCING_ENERGY&date={date_str}"
        driver.get(target_url)
        time.sleep(6)
        links = driver.find_elements("xpath", '//a[contains(@href, "anonymousresults") and contains(@href, "xlsx")]')
        if not links:
            raise RuntimeError("[DE] Aucun lien de téléchargement trouvé avec Selenium.")
        file_url = links[0].get_attribute('href')
        print(f"[DE] Téléchargement via Selenium : {file_url}")
        driver.get(file_url)
        time.sleep(wait_s)
    finally:
        driver.quit()

    for _ in range(60):
        if glob.glob(os.path.join(download_dir, "*.crdownload")):
            time.sleep(1)
        else:
            break

    candidates = glob.glob(os.path.join(download_dir, "RESULT_LIST*.xlsx"))
    if not candidates:
        raise RuntimeError("[DE] Aucun fichier RESULT_LIST*.xlsx trouvé après Selenium.")
    latest = sorted(candidates, key=os.path.getmtime)[-1]
    out_path = os.path.join(download_dir, f"afrr_anonymousresults_DE_{date_str}.xlsx")
    shutil.copy2(latest, out_path)
    print(f"[DE] Fichier prêt via Selenium : {out_path}")
    return out_path

def download_be_csvs(date_str: str, out_inc: Path, out_dec: Path):
    base_url = 'https://opendata.elia.be/api/records/1.0/search/'
    DATASETS = [
        ('incremental', 'ods163', out_inc),
        ('decremental', 'ods164', out_dec),
    ]
    ROWS = 1000
    SLEEP = 0.8
    MAX_START = 9000  # ✅ CORRECTION : Arrêter à 9000 pour éviter limite 10000

    for direction, dataset, out_path in DATASETS:
        all_frames = []
        start = 0
        page = 1
        
        while True:
            # ✅ CORRECTION : Arrêt préventif AVANT d'atteindre la limite API
            if start >= MAX_START:
                print(f"[BE] {direction}: Limite API proche ({start} records), arrêt pagination")
                break
            
            params = {
                'dataset': dataset,
                'rows': ROWS,
                'start': start,
                'refine.date_start': date_str,
                'refine.balancingproduct': 'aFRR',
                'sort': 'datetime',
            }
            
            r = requests.get(base_url, params=params)
            
            if r.status_code == 429:
                time.sleep(3)
                r = requests.get(base_url, params=params)
            
            if r.status_code != 200:
                # ✅ CORRECTION : Gestion gracieuse de l'erreur limite 10000
                if r.status_code == 400 and "10000" in r.text:
                    print(f"[BE] {direction}: Limite 10000 atteinte, utilisation des {len(all_frames)} pages récupérées")
                    break
                raise RuntimeError(f"[BE] {direction}: Erreur {r.status_code} - {r.url} - {r.text[:200]}")
            
            payload = r.json()
            records = payload.get('records', [])
            
            if not records:
                break
            
            data = [rec.get('fields', {}) for rec in records]
            df_page = pd.DataFrame(data)
            
            if 'datetime' in df_page.columns:
                df_page['datetime'] = pd.to_datetime(df_page['datetime'], errors='coerce', utc=True)
            
            all_frames.append(df_page)
            print(f"[BE] {direction} page {page}: {len(df_page)} lignes (start={start})")
            
            page += 1
            start += ROWS
            time.sleep(SLEEP)

        if not all_frames:
            pd.DataFrame().to_csv(out_path, index=False)
            print(f"[BE] {direction}: 0 ligne pour {date_str} -> {out_path}")
            continue

        final_df = pd.concat(all_frames, ignore_index=True)
        
        if 'datetime' in final_df.columns:
            final_df = final_df.sort_values('datetime').reset_index(drop=True)
        
        final_df.to_csv(out_path, index=False)
        print(f"[BE] {direction}: {len(final_df)} lignes -> {out_path}")

def download_nl_csv_for_day(date_str: str, out_path: Path, api_key: str = "18fe140e-12a2-446e-ab89-455d33709fac"):
    url = "https://api.tennet.eu/publications/v1/merit-order-list"
    headers = {"apikey": api_key, "Accept": "text/csv"}
    MAX_RETRIES, SLEEP_SECONDS = 3, 6
    # TenNET API parameters use CET/CEST local time (naive datetime is correct for their API).
    # We use Europe/Brussels to determine the current local hour so we don't fetch future hours
    # — this handles CEST (UTC+2 in summer) correctly, avoiding an off-by-2 error vs UTC.
    base = datetime.strptime(date_str + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    now_local = datetime.now(ZoneInfo("Europe/Brussels"))
    today_local = now_local.strftime("%Y-%m-%d")
    max_hour = now_local.hour if date_str == today_local else 23

    all_dfs = []
    for hour in range(max_hour + 1):
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
                    print(f"[NL] {from_time} -> {to_time} (tentative {attempt})")
                    success = True
                    break
                elif resp.status_code == 429:
                    print(f"[NL] Rate limit {from_time} (tentative {attempt})")
                    time.sleep(SLEEP_SECONDS * 2)
                else:
                    print(f"[NL] Erreur {resp.status_code}: {resp.text[:120]}")
                    break
            except Exception as e:
                print(f"[NL] Exception @ {from_time}: {e}")
            time.sleep(SLEEP_SECONDS)
        if not success:
            print(f"[NL] Échec définitif @ {from_time}")
    if not all_dfs:
        raise RuntimeError("[NL] Aucune donnée récupérée.")
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(out_path, index=False)
    print(f"[NL] Fichier fusionné : {out_path}")
    return str(out_path)

# ============ 2) CONSTRUCTION TABLEAUX AVEC DEBUG CONDITIONNEL ============

def load_and_build():
    # ✅ CORRECTION : Recalculer la date à chaque appel (passage minuit)
    TODAY = datetime.now()
    DATE_STR = TODAY.strftime("%Y-%m-%d")
    START_DATE = pd.Timestamp(DATE_STR)
    
    # Définir les paths avec la date actuelle
    PATH_BE_INC = DOWNLOAD_DIR / f"afrr_incremental_bids_BE_{DATE_STR}.csv"
    PATH_BE_DEC = DOWNLOAD_DIR / f"afrr_decremental_bids_BE_{DATE_STR}.csv"
    PATH_DE_FINAL = DOWNLOAD_DIR / f"afrr_anonymousresults_DE_{DATE_STR}.xlsx"
    PATH_NL = DOWNLOAD_DIR / f"tennet_merit_order_full_{DATE_STR.replace('-', '')}.csv"
    OUT_ALL_XLSX = DOWNLOAD_DIR / f"eu_merit_order_{DATE_STR}_ALL_DAY.xlsx"
    OUT_DE_XLSX = DOWNLOAD_DIR / f"eu_merit_order_{DATE_STR}_UNTIL_DE.xlsx"
    
    debug = DebugLogger()
    
    try:
        print(f"Date de traitement: {DATE_STR}")
        print("Téléchargements en cours...")

        be_recent = file_is_recent(PATH_BE_INC) and file_is_recent(PATH_BE_DEC)
        nl_recent = file_is_recent(PATH_NL)

        path_de = download_de_xlsx(str(DOWNLOAD_DIR), DATE_STR, wait_s=10)

        if not be_recent:
            download_be_csvs(DATE_STR, PATH_BE_INC, PATH_BE_DEC)
        else:
            print("[BE] Fichiers déjà récents, téléchargement sauté.")

        if not nl_recent:
            download_nl_csv_for_day(DATE_STR, PATH_NL)
        else:
            print("[NL] Fichier déjà récent, téléchargement sauté.")

        print("Téléchargements terminés.\n")
        print("Prétraitement et construction des piles...")
        
        # ===== DEBUG : Vérification fichiers =====
        debug.log("[DEBUG] Verification des fichiers sources...")
        debug.log(f"[DEBUG]   PATH_BE_INC: existe={PATH_BE_INC.exists()}, taille={PATH_BE_INC.stat().st_size if PATH_BE_INC.exists() else 0} octets")
        debug.log(f"[DEBUG]   PATH_BE_DEC: existe={PATH_BE_DEC.exists()}, taille={PATH_BE_DEC.stat().st_size if PATH_BE_DEC.exists() else 0} octets")
        debug.log(f"[DEBUG]   PATH_NL: existe={PATH_NL.exists()}, taille={PATH_NL.stat().st_size if PATH_NL.exists() else 0} octets")
        debug.log(f"[DEBUG]   path_de: existe={Path(path_de).exists()}, taille={Path(path_de).stat().st_size if Path(path_de).exists() else 0} octets")
        
        # ===== Lecture fichiers =====
        debug.log("[DEBUG] Lecture inc_be...")
        inc_be = pd.read_csv(PATH_BE_INC)
        debug.log(f"[DEBUG] inc_be: {len(inc_be)} lignes, {len(inc_be.columns)} colonnes")
        
        debug.log("[DEBUG] Lecture dec_be...")
        dec_be = pd.read_csv(PATH_BE_DEC)
        debug.log(f"[DEBUG] dec_be: {len(dec_be)} lignes, {len(dec_be.columns)} colonnes")
        
        debug.log("[DEBUG] Lecture xde (Excel)...")
        xde = pd.read_excel(path_de, sheet_name=0)
        debug.log(f"[DEBUG] xde: {len(xde)} lignes, {len(xde.columns)} colonnes")
        
        debug.log("[DEBUG] Lecture mo (NL)...")
        mo = pd.read_csv(PATH_NL)
        debug.log(f"[DEBUG] mo: {len(mo)} lignes, {len(mo.columns)} colonnes")

        # ===== Traitement DE =====
        debug.log("[DEBUG] Traitement données DE...")
        xde['QH_IDX'] = xde['PRODUCT'].astype(str).str.extract(r'(\d{3})').astype(float).astype('Int64')
        xde = xde.dropna(subset=['QH_IDX']).copy()
        xde['QH_IDX'] = xde['QH_IDX'].astype(int)

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
                print(f"[DE] Aucune ligne exactement à {DATE_STR}. Utilisation de la date dominante: {dominant_day}")
            else:
                raise RuntimeError("Aucun enregistrement DE valide.")

        latest_ts_de = xde_day['Timestamp'].max()
        qh_min, qh_max = int(xde_day['QH_IDX'].min()), int(xde_day['QH_IDX'].max())
        print(f"Dernier quart d'heure DE: {latest_ts_de} (QH {qh_min}->{qh_max}, source='{date_col_used}')")

        # ===== Traitement NL =====
        debug.log("[DEBUG] Traitement données NL...")
        time_col = 'Timeinterval Start Loc' if 'Timeinterval Start Loc' in mo.columns else ('Timeinterval Start (Local Time)' if 'Timeinterval Start (Local Time)' in mo.columns else None)
        if time_col is None:
            raise RuntimeError("[NL] Colonne de temps introuvable dans le CSV.")
        mo['Start_dt'] = pd.to_datetime(mo[time_col], errors='coerce').dt.tz_localize(None)
        debug.log(f"[DEBUG] NL: colonne temps '{time_col}' parsée")

        # ===== Traitement BE =====
        debug.log("[DEBUG] Traitement données BE...")
        for df in (inc_be, dec_be):
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
            df['Datetime_parsed'] = df['datetime'].dt.tz_convert(TIMEZONE).dt.tz_localize(None)

        inc_be_day = inc_be[inc_be['balancingproduct'] == 'aFRR'].copy()
        dec_be_day = dec_be[dec_be['balancingproduct'] == 'aFRR'].copy()
        debug.log(f"[DEBUG] BE: inc={len(inc_be_day)} lignes, dec={len(dec_be_day)} lignes")

        # ===== Fonction compute_prices_eu =====
        def compute_prices_eu(ts: pd.Timestamp, include_de: bool):
            inc_b = inc_be_day[inc_be_day['Datetime_parsed'].dt.floor('15min') == ts][['energybidmarginalprice', 'energybidvolume']].copy()
            inc_b.columns = ['Bid Price', 'Bid Volume']
            dec_b = dec_be_day[dec_be_day['Datetime_parsed'].dt.floor('15min') == ts][['energybidmarginalprice', 'energybidvolume']].copy()
            dec_b.columns = ['Bid Price', 'Bid Volume']

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

            if include_de:
                inc_stack = pd.concat([inc_b, inc_ge, inc_nl], ignore_index=True).dropna()
                dec_stack = pd.concat([dec_b, dec_ge, dec_nl], ignore_index=True).dropna()
            else:
                inc_stack = pd.concat([inc_b, inc_nl], ignore_index=True).dropna()
                dec_stack = pd.concat([dec_b, dec_nl], ignore_index=True).dropna()

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
                out.append(price_at(inc_stack, th))
                out.append(price_at(dec_stack, th))
            return out

        # ===== Construction tableaux =====
        def build_table(ts_end: pd.Timestamp, include_de: bool):
            stamps = pd.date_range(start=START_DATE, end=ts_end, freq='15min')
            recs = []
            for ts in stamps:
                row = {'Timestamp': ts}
                prices = compute_prices_eu(ts, include_de=include_de)
                for i, th in enumerate(THRESHOLDS):
                    row[f'Inc_{th}'] = prices[2*i]
                    row[f'Dec_{th}'] = prices[2*i+1]
                recs.append(row)
            return pd.DataFrame(recs)

        debug.log("[DEBUG] Construction ALL_DAY...")
        all_day_end = START_DATE + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
        eu_merit_order_ALL_DAY = build_table(all_day_end, include_de=False)
        debug.log(f"[DEBUG] ALL_DAY: {len(eu_merit_order_ALL_DAY)} lignes")

        debug.log("[DEBUG] Construction UNTIL_DE...")
        eu_merit_order_UNTIL_DE = build_table(latest_ts_de, include_de=True)
        debug.log(f"[DEBUG] UNTIL_DE: {len(eu_merit_order_UNTIL_DE)} lignes")

        # ===== Écriture fichiers Excel =====
        def write_excel_atomic(df: pd.DataFrame, out_path: Path):
            tmp_fd, tmp_name = tempfile.mkstemp(suffix=".xlsx", prefix="tmp_", dir=str(out_path.parent))
            os.close(tmp_fd)
            try:
                df.to_excel(tmp_name, index=False)
                try:
                    os.replace(tmp_name, out_path)
                except PermissionError:
                    alt = out_path.with_name(out_path.stem + f"_{int(time.time())}" + out_path.suffix)
                    shutil.move(tmp_name, alt)
                    print(f"[WARN] Cible verrouillée. Écrit dans {alt}")
            except Exception:
                try:
                    os.remove(tmp_name)
                except Exception:
                    pass
                raise

        debug.log(f"[DEBUG] Ecriture ALL_DAY -> {OUT_ALL_XLSX}")
        write_excel_atomic(eu_merit_order_ALL_DAY, OUT_ALL_XLSX)
        debug.log(f"[DEBUG] ALL_DAY ecrit: {OUT_ALL_XLSX.stat().st_size} octets")

        debug.log(f"[DEBUG] Ecriture UNTIL_DE -> {OUT_DE_XLSX}")
        write_excel_atomic(eu_merit_order_UNTIL_DE, OUT_DE_XLSX)
        debug.log(f"[DEBUG] UNTIL_DE ecrit: {OUT_DE_XLSX.stat().st_size} octets")

        print("Tableaux générés.")
        print(f"Dernier QH DE : {latest_ts_de}")
        print(f"ALL_DAY  -> {OUT_ALL_XLSX}")
        print(f"UNTIL_DE -> {OUT_DE_XLSX}")

        return str(OUT_ALL_XLSX), str(OUT_DE_XLSX)

    except Exception as e:
        # ===== EN CAS D'ERREUR : AFFICHER TOUS LES LOGS DEBUG =====
        print("\n" + "="*80)
        print("ERREUR DETECTEE - Logs DEBUG :")
        print("="*80)
        debug.dump_on_error()
        print("="*80)
        print(f"Exception: {e}")
        print("="*80 + "\n")
        raise

# ============ 3) CLI ============

def display_xlsx(path: Path, rows: int = 20):
    if not path.exists():
        print(f"Fichier introuvable : {path}")
        return 1
    try:
        df = pd.read_excel(path)
        print(f"\n=== {path.name} (rows={len(df)}, cols={len(df.columns)}) ===")
        print("\n--- HEAD ---")
        print(df.head(rows).to_string(index=False))
        print("\n--- TAIL ---")
        print(df.tail(rows).to_string(index=False))
        print("\nColonnes :", ", ".join(df.columns))
        return 0
    except Exception as e:
        print(f"Erreur lecture Excel: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description="EU aFRR merit-order (DE/BE/NL): téléchargement + construction + affichage"
    )
    parser.add_argument("command", nargs="?", default="run",
                        choices=["run", "displayall", "displayde"],
                        help="run = télécharger & (re)générer; displayall = afficher ALL_DAY; displayde = afficher UNTIL_DE")
    parser.add_argument("--rows", type=int, default=20)
    args = parser.parse_args()

    if args.command == "run":
        if not acquire_lock():
            print("Un autre run afrr_merit_order est en cours. Sortie.")
            return 0
        try:
            load_and_build()
            return 0
        finally:
            release_lock()
    elif args.command == "displayall":
        DATE_STR = datetime.now().strftime("%Y-%m-%d")
        OUT_ALL_XLSX = DOWNLOAD_DIR / f"eu_merit_order_{DATE_STR}_ALL_DAY.xlsx"
        return display_xlsx(OUT_ALL_XLSX, rows=args.rows)
    elif args.command == "displayde":
        DATE_STR = datetime.now().strftime("%Y-%m-%d")
        OUT_DE_XLSX = DOWNLOAD_DIR / f"eu_merit_order_{DATE_STR}_UNTIL_DE.xlsx"
        return display_xlsx(OUT_DE_XLSX, rows=args.rows)
    else:
        print("Commande inconnue.")
        return 2

if __name__ == "__main__":
    sys.exit(main())