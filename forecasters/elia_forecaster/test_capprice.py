import pandas as pd
import os

tz = "Europe/Brussels"
DATE_STR = pd.Timestamp.now(tz=tz).strftime("%Y-%m-%d")
DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), "Downloads")

# fichiers BE
inc = pd.read_csv(os.path.join(DOWNLOAD_DIR, f"afrr_incremental_bids_BE_{DATE_STR}.csv"))
dec = pd.read_csv(os.path.join(DOWNLOAD_DIR, f"afrr_decremental_bids_BE_{DATE_STR}.csv"))

def cap_price(df, ts):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["qh"] = df["datetime"].dt.tz_convert(tz).dt.tz_localize(None).dt.floor("15min")
    row = df[(df.get("rank") == 1) & (df.get("energybidactivationorder") == 1) & (df["qh"] == ts)]
    if row.empty:
        return None
    return float(row["energybidmarginalprice"].iloc[0])

# QH cible = 11:15
ts_target = pd.Timestamp(f"{DATE_STR} 11:15:00")

print("=== BE Cap Price @11:15 ===")
print("INC cap:", cap_price(inc, ts_target))
print("DEC cap:", cap_price(dec, ts_target))
