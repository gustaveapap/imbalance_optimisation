# ====================================
# ✅ SCRIPT COMBINÉ : aFRR Allemagne (DE), Belgique (BE), Pays-Bas (NL)
# Date cible : aujourd'hui (dynamique)
# ====================================

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from io import StringIO
import os
import glob

# ================================
# 📂 Dossier Téléchargements (forcé en chemin absolu)
download_dir = r"C:\Users\gusta\Downloads"
os.makedirs(download_dir, exist_ok=True)

# 📅 Date dynamique = aujourd'hui
date_str = datetime.now().strftime("%Y-%m-%d")
target_date = datetime.now()

# ================================
# 🇩🇪 ALLEMAGNE – Téléchargement via Selenium
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    chrome_options = Options()
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,   # ✅ fichiers vont dans Téléchargements
        "download.prompt_for_download": False,
        "directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    chrome_options.add_argument("--headless")  # Optionnel
    driver = webdriver.Chrome(options=chrome_options)

    target_url = f"https://www.regelleistung.net/apps/datacenter/tenders/?productTypes=SRL&markets=BALANCING_ENERGY&date={date_str}"
    driver.get(target_url)
    time.sleep(5)

    download_links = driver.find_elements("xpath", '//a[contains(@href, "anonymousresults") and contains(@href, "xlsx")]')
    if download_links:
        file_url = download_links[0].get_attribute('href')
        print(f"✅ [DE] Téléchargement depuis : {file_url}")
        driver.get(file_url)
        time.sleep(5)
    else:
        print("❌ [DE] Aucun lien de téléchargement trouvé.")
    driver.quit()

    # Vérifier que le fichier est bien arrivé
    candidates = glob.glob(os.path.join(download_dir, "*anonymousresults*.xlsx"))
    if candidates:
        latest = sorted(candidates, key=os.path.getmtime)[-1]
        print(f"📄 [DE] Fichier téléchargé : {latest}")
    else:
        print("❌ [DE] Aucun fichier téléchargé trouvé dans Téléchargements.")

except Exception as e:
    print(f"❌ [DE] Erreur Selenium : {e}")

# ================================
# 🇧🇪 BELGIQUE – API Elia (incremental + decremental)
datasets = {'incremental': 'ods163', 'decremental': 'ods164'}

for direction, dataset in datasets.items():
    url = 'https://opendata.elia.be/api/records/1.0/search/'
    params = {
        'dataset': dataset,
        'rows': 1000,
        'refine.date_start': date_str
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        records = response.json().get('records', [])
        data = [r['fields'] for r in records]
        df = pd.DataFrame(data)
        filename = f'afrr_{direction}_bids_BE_{date_str}.csv'
        filepath = os.path.join(download_dir, filename)   # ✅ fichiers BE dans Téléchargements
        df.to_csv(filepath, index=False)
        print(f"✅ [BE] {len(df)} lignes dans : {filepath}")
    else:
        print(f"❌ [BE] Erreur {response.status_code} – {response.url}")

# ================================
# 🇳🇱 PAYS-BAS – API TenneT (Merit Order List)
api_key = "18fe140e-12a2-446e-ab89-455d33709fac"
url = "https://api.tennet.eu/publications/v1/merit-order-list"
headers = {
    "apikey": api_key,
    "Accept": "text/csv"
}
MAX_RETRIES = 3
SLEEP_SECONDS = 6
all_dfs = []

for hour in range(24):
    from_time = (target_date + timedelta(hours=hour)).strftime("%d-%m-%Y %H:%M:%S")
    to_time = (target_date + timedelta(hours=hour+1)).strftime("%d-%m-%Y %H:%M:%S")
    params = {"date_from": from_time, "date_to": to_time}
    success = False

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200 and response.content.strip():
                df = pd.read_csv(StringIO(response.content.decode("utf-8")))
                all_dfs.append(df)
                print(f"✅ [NL] {from_time} → {to_time} (tentative {attempt})")
                success = True
                break
            elif response.status_code == 429:
                print(f"⏳ [NL] Rate limit pour {from_time} (tentative {attempt})")
                time.sleep(SLEEP_SECONDS * 2)
            else:
                print(f"❌ [NL] Erreur {response.status_code} : {response.text}")
                break
        except Exception as e:
            print(f"⚠️ [NL] Exception à {from_time} : {e}")
        time.sleep(SLEEP_SECONDS)

    if not success:
        print(f"❌ [NL] Échec définitif à {from_time}")

if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)
    nl_filename = os.path.join(download_dir, f"tennet_merit_order_full_{target_date.strftime('%Y%m%d')}.csv")  # ✅ NL dans Téléchargements
    final_df.to_csv(nl_filename, index=False)
    print(f"\n📄 [NL] Fichier fusionné : {nl_filename}")
    print(f"🔢 PTU uniques : {final_df[['Timeinterval Start Loc', 'Timeinterval End Loc']].drop_duplicates().shape[0]}")
else:
    print("❌ [NL] Aucune donnée récupérée.")
