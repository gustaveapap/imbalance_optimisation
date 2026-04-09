"""
Script de diagnostic pour comprendre pourquoi extract_features retourne None
"""
import pandas as pd
import pytz
from datetime import datetime, timedelta

BRUSSELS = pytz.timezone("Europe/Brussels")

# Charger le CSV
print("=" * 60)
print("DIAGNOSTIC DES FEATURES")
print("=" * 60)

df = pd.read_csv("historical_imbalance_data.csv")
print(f"\n1. CSV chargé : {len(df)} lignes")
print(f"   Colonnes : {df.columns.tolist()}")

# Convertir datetime
df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
df = df.set_index("datetime").sort_index()

print(f"\n2. Index datetime configuré")
print(f"   Type de l'index : {type(df.index)}")
print(f"   Timezone de l'index : {df.index.tz}")
print(f"   Premier timestamp : {df.index[0]} (type: {type(df.index[0])})")
print(f"   Dernier timestamp : {df.index[-1]} (type: {type(df.index[-1])})")

# Simuler la prochaine prédiction
now_bxl = datetime.now(BRUSSELS)
current_qh = now_bxl.replace(minute=(now_bxl.minute // 15) * 15, second=0, microsecond=0)
next_qh = current_qh + timedelta(minutes=15)

# Convertir en naive
if next_qh.tzinfo is not None:
    next_qh_naive = next_qh.astimezone(BRUSSELS).replace(tzinfo=None)
else:
    next_qh_naive = next_qh

print(f"\n3. Target quarter hour")
print(f"   next_qh original : {next_qh} (type: {type(next_qh)})")
print(f"   next_qh_naive : {next_qh_naive} (type: {type(next_qh_naive)})")
print(f"   Timezone : {getattr(next_qh_naive, 'tz', None) or getattr(next_qh_naive, 'tzinfo', None)}")

# Test extract_features
print(f"\n4. Simulation extract_features")
feature_end = next_qh_naive - pd.Timedelta(minutes=5)
feature_start = feature_end - pd.Timedelta(minutes=60)

print(f"   feature_start : {feature_start}")
print(f"   feature_end : {feature_end}")

# Vérifier les comparaisons
print(f"\n5. Test des comparaisons")
try:
    mask_start = df.index >= feature_start
    print(f"   ✓ df.index >= feature_start fonctionne")
    print(f"     {mask_start.sum()} valeurs >= feature_start")
except Exception as e:
    print(f"   ✗ Erreur : {e}")

try:
    mask_end = df.index < feature_end
    print(f"   ✓ df.index < feature_end fonctionne")
    print(f"     {mask_end.sum()} valeurs < feature_end")
except Exception as e:
    print(f"   ✗ Erreur : {e}")

try:
    feature_window = df[
        (df.index >= feature_start) & 
        (df.index < feature_end)
    ]['actual_system_imbalance'].values
    
    print(f"   ✓ Extraction réussie")
    print(f"     {len(feature_window)} valeurs extraites (besoin: 60)")
    
    if len(feature_window) != 60:
        print(f"\n   ⚠️  PROBLÈME : {len(feature_window)} valeurs au lieu de 60")
        print(f"      Période demandée : {feature_start} → {feature_end}")
        
        # Afficher quelques timestamps autour de la fenêtre
        around_start = df[(df.index >= feature_start - pd.Timedelta(minutes=5)) & 
                          (df.index <= feature_start + pd.Timedelta(minutes=5))]
        around_end = df[(df.index >= feature_end - pd.Timedelta(minutes=5)) & 
                        (df.index <= feature_end + pd.Timedelta(minutes=5))]
        
        print(f"\n   Timestamps autour de feature_start ({feature_start}):")
        for ts in around_start.index[:10]:
            print(f"      {ts}")
        
        print(f"\n   Timestamps autour de feature_end ({feature_end}):")
        for ts in around_end.index[:10]:
            print(f"      {ts}")
    else:
        print(f"   ✓ PARFAIT : 60 valeurs extraites")
        
except Exception as e:
    print(f"   ✗ Erreur lors de l'extraction : {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)