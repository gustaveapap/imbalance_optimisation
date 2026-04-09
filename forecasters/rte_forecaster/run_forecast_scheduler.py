# -*- coding: utf-8 -*-
import sys, base64, datetime as dt, io, logging, time
from pathlib import Path
import requests, pandas as pd, numpy as np, joblib
from apscheduler.schedulers.blocking import BlockingScheduler

# Ensure stdout/stderr can handle UTF-8 emoji on Windows (cp1252 terminals)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

CLIENT_ID = "bdc03388-6c93-46f6-adbf-1a77d5b89684"
CLIENT_SECRET = "da352352-63f8-42ce-9a34-0624f7560a72"
MODEL_PATH = "artifacts/fr_imbalance_full_model.pkl"
FORECAST_LOG = "forecast_log.csv"
MAX_LAG = 96

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def get_token(cid, secret):
    basic = base64.b64encode((cid + ':' + secret).encode()).decode()
    r = requests.post(
        'https://digital.iservices.rte-france.com/token/oauth/',
        headers={'Content-Type': 'application/x-www-form-urlencoded', 'Authorization': 'Basic ' + basic},
        data={'grant_type': 'client_credentials'}, timeout=30)
    r.raise_for_status()
    return r.json()['access_token']

def download_imbalance(tok, start, end):
    url = 'https://digital.iservices.rte-france.com/open_api/balancing_energy/v4/imbalance_data'
    params = {
        'start_date': start.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'end_date': end.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'resolution': 'PT15M'
    }
    headers = {'Authorization': 'Bearer ' + tok, 'Accept': 'text/csv'}
    r = requests.get(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    lines = r.text.splitlines()
    idx = next(i for i, l in enumerate(lines) if l.startswith("Heure de"))
    df = pd.read_csv(io.StringIO('\n'.join(lines[idx:])), sep=';')
    df.columns = ['start_time', 'end_time', 'imbalance_mwh', 'trend', 'price_pos', 'price_neg']
    df['start_time'] = pd.to_datetime(df['start_time'], format='%d/%m/%Y %H:%M')
    df['imbalance_mwh'] = pd.to_numeric(df['imbalance_mwh'], errors='coerce')
    return df.set_index('start_time')[['imbalance_mwh']]

def build_features(raw):
    raw = raw.sort_index()
    full = raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), freq='15min'))
    full['imbalance_mwh'] = full['imbalance_mwh'].interpolate(limit=2).ffill().bfill()
    lag_cols = {f'lag_{l}': full['imbalance_mwh'].shift(l) for l in range(1, MAX_LAG + 1)}
    full = pd.concat([full, pd.DataFrame(lag_cols, index=full.index)], axis=1)
    slots = (full.index.hour * 4) + (full.index.minute // 15)
    full['sin_time'] = np.sin(2 * np.pi * slots / 96)
    full['cos_time'] = np.cos(2 * np.pi * slots / 96)
    full['sin_dow'] = np.sin(2 * np.pi * full.index.dayofweek / 7)
    full['cos_dow'] = np.cos(2 * np.pi * full.index.dayofweek / 7)
    full['sin_month'] = np.sin(2 * np.pi * full.index.month / 12)
    full['cos_month'] = np.cos(2 * np.pi * full.index.month / 12)
    return full

def forecast_next(feat, model):
    latest_ts = feat.index[-1]
    next_ts = latest_ts + pd.Timedelta(minutes=15)
    row = feat.iloc[-1].copy()
    for l in range(1, MAX_LAG + 1):
        src = next_ts - pd.Timedelta(minutes=15 * l)
        row[f'lag_{l}'] = feat.at[src, 'imbalance_mwh']
    slot = (next_ts.hour * 4) + (next_ts.minute // 15)
    row['sin_time'] = np.sin(2 * np.pi * slot / 96)
    row['cos_time'] = np.cos(2 * np.pi * slot / 96)
    row['sin_dow'] = np.sin(2 * np.pi * next_ts.dayofweek / 7)
    row['cos_dow'] = np.cos(2 * np.pi * next_ts.dayofweek / 7)
    row['sin_month'] = np.sin(2 * np.pi * next_ts.month / 12)
    row['cos_month'] = np.cos(2 * np.pi * next_ts.month / 12)
    X = pd.DataFrame([row.drop('imbalance_mwh')])
    y = float(model.predict(X)[0])
    return next_ts, y, feat.loc[latest_ts, 'imbalance_mwh']

def run_cycle():
    try:
        now = dt.datetime.now().replace(second=0, microsecond=0)
        forecast_target = now if now.minute % 15 == 0 else now - dt.timedelta(minutes=now.minute % 15)
        latest_needed = forecast_target - dt.timedelta(minutes=15)
        start = forecast_target - dt.timedelta(hours=27)

        tok = get_token(CLIENT_ID, CLIENT_SECRET)

        for attempt in range(30):
            raw = download_imbalance(tok, start, forecast_target)
            print("🧪 Données brutes récupérées :", raw.index.min(), "→", raw.index.max())
            print("🧮 Quart-heures disponibles :", len(raw))
            missing = raw.asfreq('15min').isna().sum()['imbalance_mwh']
            print("📉 Quart-heures manquants ?", missing)
            
            # DEBUG AJOUTÉ
            print(f"🔎 DEBUG - forecast_target : {forecast_target} (type: {type(forecast_target)})")
            print(f"🔎 DEBUG - latest_needed recherché : {latest_needed} (type: {type(latest_needed)})")
            print(f"🔎 DEBUG - Timezone latest_needed : {latest_needed.tzinfo}")
            print(f"🔎 DEBUG - raw.index[-5:] : {raw.index[-5:].tolist()}")
            print(f"🔎 DEBUG - Timezone raw.index : {raw.index[-1].tzinfo if hasattr(raw.index[-1], 'tzinfo') else 'None'}")
            print(f"🔎 DEBUG - latest_needed in raw.index ? {latest_needed in raw.index}")
            
            if latest_needed in raw.index:
                val = raw.at[latest_needed, 'imbalance_mwh']
                print(f"🔎 DEBUG - Valeur à latest_needed : {val}")
                print(f"🔎 DEBUG - Est non-NaN ? {pd.notna(val)}")

            if latest_needed in raw.index and pd.notna(raw.at[latest_needed, 'imbalance_mwh']):
                logging.info("✅ Donnée %s disponible, on prédit %s.",
                             latest_needed.strftime('%H:%M'),
                             forecast_target.strftime('%H:%M'))
                break
            logging.info("🔁 Tentative %d/30 : donnée %s indisponible, attente 30s...",
                         attempt + 1, latest_needed.strftime('%H:%M'))
            time.sleep(30)
        else:
            logging.warning("⚠️ Donnée %s toujours indisponible après 15 min. Prévision annulée.",
                            latest_needed.strftime('%H:%M'))
            return

        feat = build_features(raw)
        model = joblib.load(MODEL_PATH)
        next_ts, forecast, latest_val = forecast_next(feat, model)

        Path(FORECAST_LOG).parent.mkdir(exist_ok=True, parents=True)
        new_row = pd.DataFrame([[next_ts, forecast, None]], columns=['timestamp', 'forecast_mwh', 'actual_mwh'])

        if Path(FORECAST_LOG).exists():
            history = pd.read_csv(FORECAST_LOG, parse_dates=['timestamp'])
            history = pd.concat([history, new_row], ignore_index=True)
        else:
            history = new_row

        missing_actuals = history[history['actual_mwh'].isna()]
        if not missing_actuals.empty:
            min_ts = missing_actuals['timestamp'].min()
            max_ts = missing_actuals['timestamp'].max() + pd.Timedelta(minutes=15)

            for attempt in range(30):
                try:
                    raw_patch = download_imbalance(tok, min_ts, max_ts)
                    for idx, row in history.iterrows():
                        ts = row['timestamp']
                        if pd.isna(row['actual_mwh']) and ts in raw_patch.index:
                            history.at[idx, 'actual_mwh'] = raw_patch.at[ts, 'imbalance_mwh']
                    break
                except Exception as e:
                    logging.warning("⏳ Tentative %d récupération des réels : %s", attempt + 1, e)
                    time.sleep(10)
            else:
                logging.warning("⚠️ Échec de complétion des réels. On réessaiera plus tard.")

        history.to_csv(FORECAST_LOG, index=False)

        display = history.dropna(subset=['actual_mwh']).tail(4)
        for _, row in display.iterrows():
            ts, f, a = row['timestamp'], row['forecast_mwh'], row['actual_mwh']
            e = round(f - a, 2)
            logging.info("⏱️ %s | ⚡Prévu: %.2f MWh | 📊 Réel: %.2f MWh | 🔁 Écart: %.2f MWh", ts, f, a, e)

        logging.info("✅ Prochaine prévision prévue pour %s : %.2f MWh", next_ts.strftime('%H:%M'), forecast)

    except Exception as e:
        logging.exception("❌ Échec du cycle")

if __name__ == '__main__':
    logging.info("🚀 DÉMARRAGE DU FORECAST SCHEDULER")
    logging.info(f"📁 Chemin du modèle : {MODEL_PATH}")
    
    # Vérifie que le modèle existe
    if not Path(MODEL_PATH).exists():
        logging.error(f"❌ MODÈLE INTROUVABLE : {MODEL_PATH}")
        logging.error(f"📂 Répertoire actuel : {Path.cwd()}")
        exit(1)
    
    scheduler = BlockingScheduler()
    scheduler.add_job(run_cycle, 'cron', minute='*/15', second='45')
    logging.info("⏰ Scheduler en place : lancement toutes les 15 min à xx:45s (heure locale)")

    now = dt.datetime.now()
    if now.minute % 15 != 0:
        logging.info("⏳ Trop tôt (%s), prévision immédiate du quart d'heure en cours, puis attente %d sec.",
                     now.strftime('%H:%M'), (15 - now.minute % 15) * 60 - now.second)
        run_cycle()
        time.sleep((15 - now.minute % 15) * 60 - now.second)
    run_cycle()
    scheduler.start()