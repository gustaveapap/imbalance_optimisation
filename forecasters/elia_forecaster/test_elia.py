import os
import pandas as pd

TIMEZONE = "Europe/Brussels"
DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
DATE_STR = pd.Timestamp.now(tz=TIMEZONE).strftime("%Y-%m-%d")

# Fichiers produits par afrr_merit_order.py
PATH_BE_INC = os.path.join(DOWNLOAD_DIR, f"afrr_incremental_bids_BE_{DATE_STR}.csv")
PATH_BE_DEC = os.path.join(DOWNLOAD_DIR, f"afrr_decremental_bids_BE_{DATE_STR}.csv")
OUT_DE_XLSX = os.path.join(DOWNLOAD_DIR, f"eu_merit_order_{DATE_STR}_UNTIL_DE.xlsx")

def must_have(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}\n→ Lance d'abord: python afrr_merit_order.py run")

def get_cap_price(df, ts_floor_15):
    """Cap price BE = premier bid dispo (rank=1 & energybidactivationorder=1) au QH ts."""
    if not {"rank", "energybidactivationorder", "energybidmarginalprice", "datetime"} <= set(df.columns):
        return None

    d = df.copy()
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce", utc=True)
    d["dt_loc"] = d["datetime"].dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    d["qh"] = d["dt_loc"].dt.floor("15min")

    row = d[(d["rank"] == 1) & (d["energybidactivationorder"] == 1) & (d["qh"] == ts_floor_15)]
    if row.empty:
        return None
    return float(row["energybidmarginalprice"].iloc[0])

def main():
    # Vérifs
    must_have(PATH_BE_INC)
    must_have(PATH_BE_DEC)
    must_have(OUT_DE_XLSX)

    print("Chargement EU UNTIL_DE (BE+DE+NL)…")
    eu = pd.read_excel(OUT_DE_XLSX)
    eu["Timestamp"] = pd.to_datetime(eu["Timestamp"], errors="coerce")

    # Dernier QH dispo
    target_ts = eu["Timestamp"].max()
    row_eu = eu[eu["Timestamp"] == target_ts]

    col_inc_1000, col_dec_1000 = "Inc_1000", "Dec_1000"
    inc1000 = float(row_eu[col_inc_1000].iloc[0]) if pd.notna(row_eu[col_inc_1000].iloc[0]) else None
    dec1000 = float(row_eu[col_dec_1000].iloc[0]) if pd.notna(row_eu[col_dec_1000].iloc[0]) else None

    # Cap price BE
    print("Chargement BE incremental/decremental…")
    inc_be = pd.read_csv(PATH_BE_INC)
    dec_be = pd.read_csv(PATH_BE_DEC)

    cap_inc = get_cap_price(inc_be, target_ts)
    cap_dec = get_cap_price(dec_be, target_ts)

    print("\n=== Résultat test (QH) ===")
    print("Timestamp QH (EU UNTIL_DE):", target_ts)
    print("\n-- BE Cap Price --")
    print("INC cap (rank=1, activationorder=1):", cap_inc)
    print("DEC cap (rank=1, activationorder=1):", cap_dec)
    print("\n-- EU Max Price (DE inclus) --")
    print("INC_1000:", inc1000)
    print("DEC_1000:", dec1000)

if __name__ == "__main__":
    main()
