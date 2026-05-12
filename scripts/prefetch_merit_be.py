#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
"""
Telecharge et met en cache les offres aFRR + mFRR pour BE, DE et NL.

Sources :
  BE   : ODS156 (incremental hist) / ODS157 (decremental hist)
         ODS163 (incremental NRT)  / ODS164 (decremental NRT)
  DE   : regelleistung.net  crds/api/v2  (xlsx -> parsed in memory)
  NL   : TenneT merit-order-list API

Usage :
  python scripts/prefetch_merit_be.py                   # 2025-01-01 -> aujourd'hui
  python scripts/prefetch_merit_be.py 2026-01-01 2026-05-11
  python scripts/prefetch_merit_be.py 2026-04-06 2026-04-06   # un seul jour
"""

import re
import time
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests

# =============================================================================
# CONFIG
# =============================================================================
REPO      = Path(__file__).resolve().parents[1]
DATA_2025 = REPO / "data_ve_2025"
DATA_2026 = REPO / "data_ve_2026"

ELIA_API        = "https://opendata.elia.be/api/records/1.0/search/"
ODS_HIST        = {"incremental": "ods156", "decremental": "ods157"}
ODS_NRT         = {"incremental": "ods163", "decremental": "ods164"}
BRUSSELS        = "Europe/Brussels"

API_KEY_TENNET  = "18fe140e-12a2-446e-ab89-455d33709fac"
TENNET_API      = "https://api.tennet.eu/publications/v1/merit-order-list"

REGELLEISTUNG   = "https://www.regelleistung.net/apps/crds/api/v2/tenders/results/anonymous"

RESERVES = ["aFRR", "mFRR"]


# =============================================================================
# DATA DIRECTORY
# =============================================================================

def _data_dir(date_str: str) -> Path:
    year = date_str[:4]
    d = REPO / f"data_ve_{year}"
    d.mkdir(exist_ok=True)
    return d


# =============================================================================
# BE — Elia ODS156/157 (hist) + ODS163/164 (NRT)
# =============================================================================

def _fetch_elia_dir(date_str, product, direction, dataset_id):
    date_obj        = datetime.strptime(date_str, "%Y-%m-%d")
    date_end        = date_obj + timedelta(days=1)
    direction_label = "UP" if direction == "incremental" else "DOWN"
    params = {
        "dataset":                 dataset_id,
        "rows":                    1000, "start": 0,
        "refine.balancingproduct": product,
        "sort":                    "datetime",
        "q": (f"datetime:[{date_str}T00:00:00 "
              f"TO {date_end.strftime('%Y-%m-%d')}T00:00:00]"),
    }
    try:
        r = requests.get(ELIA_API, params=params, timeout=45)
        if r.status_code == 429:
            time.sleep(15)
            r = requests.get(ELIA_API, params=params, timeout=45)
        if r.status_code != 200:
            return pd.DataFrame()
        data  = r.json()
        nhits = data.get("nhits", 0)
        if nhits == 0:
            return pd.DataFrame()
        recs = []
        start = 0
        while start < nhits and start < 10000:
            params["start"] = start
            rp = requests.get(ELIA_API, params=params, timeout=45)
            if rp.status_code == 429:
                time.sleep(15)
                rp = requests.get(ELIA_API, params=params, timeout=45)
            if rp.status_code != 200:
                break
            recs.extend(rp.json().get("records", []))
            start += 1000
            time.sleep(0.3)
    except Exception:
        return pd.DataFrame()

    offers = []
    for rec in recs:
        f  = rec.get("fields", {})
        dt = pd.to_datetime(f.get("datetime", ""), errors="coerce", utc=True)
        if pd.isna(dt):
            continue
        ts = dt.tz_convert(BRUSSELS).tz_localize(None).floor("15min")
        if ts.date() != date_obj.date():
            continue
        try:
            price    = float(f.get("energybidmarginalprice"))
            quantity = float(f.get("energybidvolume"))
        except (TypeError, ValueError):
            continue
        if quantity <= 0:
            continue
        offers.append({"timestamp": ts, "direction": direction_label,
                       "price": price, "quantity": quantity,
                       "country": "BE", "reserve": product})
    if not offers:
        return pd.DataFrame()
    df = pd.DataFrame(offers)
    return df.drop_duplicates(subset=["timestamp", "direction", "price", "quantity", "reserve"])


def fetch_be(date_str, product):
    """Combine UP+DOWN for one BE product/day. Tries hist first, falls back to NRT."""
    parts = []
    for direction in ["incremental", "decremental"]:
        df = _fetch_elia_dir(date_str, product, direction, ODS_HIST[direction])
        if df.empty:
            df = _fetch_elia_dir(date_str, product, direction, ODS_NRT[direction])
        if not df.empty:
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# =============================================================================
# DE — Regelleistung.net (xlsx parsed in-memory, same as simulation_ve_2025_fr.py)
# =============================================================================

def fetch_de(date_str, product):
    try:
        r = requests.get(
            REGELLEISTUNG,
            params={"productType": product, "market": "ENERGY",
                    "exportFormat": "xlsx", "deliveryDate": date_str},
            timeout=60)
        if r.status_code != 200 or r.content[:2] != b"PK":
            return pd.DataFrame()

        with ZipFile(BytesIO(r.content)) as z:
            xml = z.read("xl/worksheets/sheet1.xml").decode("utf-8", errors="ignore")

        rows = []
        for row_match in re.finditer(r"<row r=\"\d+\">(.*?)</row>", xml, re.DOTALL):
            cells = []
            for cell in re.finditer(r"<c r=\"[A-Z]+\d+\"[^>]*>(.*?)</c>",
                                    row_match.group(1), re.DOTALL):
                t = re.search(r"<t>(.*?)</t>", cell.group(1))
                v = re.search(r"<v>(.*?)</v>", cell.group(1))
                cells.append(t.group(1) if t else (v.group(1) if v else ""))
            rows.append(cells)

        if len(rows) < 2:
            return pd.DataFrame()
        headers = rows[0]
        max_len = max(len(r) for r in rows)
        data    = [r + [""] * (max_len - len(r)) for r in rows[1:]]
        if max_len > len(headers):
            headers = headers + [f"col_{i}" for i in range(max_len - len(headers))]
        xde = pd.DataFrame(data, columns=headers)
        if "PRODUCT" not in xde.columns:
            return pd.DataFrame()

        xde["QH_IDX"] = (xde["PRODUCT"].astype(str)
                         .str.extract(r"(\d{3})").astype(float).astype("Int64"))
        xde = xde.dropna(subset=["QH_IDX"])
        xde["DeliveryDate"] = pd.to_datetime(
            pd.to_numeric(xde["DELIVERY_DATE"], errors="coerce"),
            unit="D", origin="1899-12-30").dt.normalize()
        xde["timestamp"] = (xde["DeliveryDate"] +
                            pd.to_timedelta((xde["QH_IDX"] - 1) * 15, unit="m"))

        qty_col = next((c for c in xde.columns if "CAPACITY" in c.upper()), None)
        if qty_col is None:
            return pd.DataFrame()

        offers = []
        for pfx, direction in [("POS", "UP"), ("NEG", "DOWN")]:
            sub = xde[xde["PRODUCT"].astype(str).str.startswith(pfx)].copy()
            if sub.empty or "ENERGY_PRICE_[EUR/MWh]" not in sub.columns:
                continue
            if "ENERGY_PRICE_PAYMENT_DIRECTION" in sub.columns:
                mask = sub["ENERGY_PRICE_PAYMENT_DIRECTION"] == "GRID_TO_PROVIDER"
                sub.loc[mask, "ENERGY_PRICE_[EUR/MWh]"] = (
                    pd.to_numeric(sub.loc[mask, "ENERGY_PRICE_[EUR/MWh]"], errors="coerce") * -1)
            of = pd.DataFrame({
                "timestamp": pd.to_datetime(sub["timestamp"]).dt.floor("15min"),
                "direction": direction,
                "price":     pd.to_numeric(sub["ENERGY_PRICE_[EUR/MWh]"], errors="coerce"),
                "quantity":  pd.to_numeric(sub[qty_col], errors="coerce"),
                "country":   "DE", "reserve": product,
            }).dropna()
            of = of[of["quantity"] > 0]
            if not of.empty:
                offers.append(of)

        if not offers:
            return pd.DataFrame()
        return pd.concat(offers, ignore_index=True).drop_duplicates(
            subset=["timestamp", "direction", "price", "quantity", "reserve"])
    except Exception:
        return pd.DataFrame()


# =============================================================================
# NL — TenneT merit-order-list (same as simulation_ve_2025_be.py)
# =============================================================================

def fetch_nl(date_str):
    """Returns (df_afrr, df_mfrr) for NL."""
    try:
        base = datetime.strptime(date_str + " 00:00:00", "%Y-%m-%d %H:%M:%S")
        resp = requests.get(
            TENNET_API,
            headers={"apikey": API_KEY_TENNET, "Accept": "text/csv"},
            params={"date_from": base.strftime("%d-%m-%Y %H:%M:%S"),
                    "date_to":   (base + timedelta(days=1)).strftime("%d-%m-%Y %H:%M:%S")},
            timeout=60)
        if resp.status_code != 200 or not resp.content.strip():
            return pd.DataFrame(), pd.DataFrame()

        mo = pd.read_csv(StringIO(resp.content.decode("utf-8")))
        tc = ("Timeinterval Start Loc" if "Timeinterval Start Loc" in mo.columns
              else "Timeinterval Start (Local Time)")
        if tc not in mo.columns:
            return pd.DataFrame(), pd.DataFrame()

        mo["timestamp"] = pd.to_datetime(mo[tc], errors="coerce").dt.tz_localize(None)
        mo = mo.dropna(subset=["timestamp", "Capacity Threshold"])
        mo = mo[mo["timestamp"].dt.date == base.date()]
        mo = mo.sort_values(["timestamp", "Capacity Threshold"]).reset_index(drop=True)
        mo["volume"] = mo.groupby("timestamp")["Capacity Threshold"].diff().fillna(
            mo["Capacity Threshold"])
        mo = mo[mo["volume"] > 0]

        all_a, all_m = [], []
        for ts in mo["timestamp"].unique():
            mt = mo[mo["timestamp"] == ts]
            for pc, d in [("Price Up", "UP"), ("Price Down", "DOWN")]:
                if pc not in mt.columns:
                    continue
                ots = mt[[pc, "volume"]].dropna().copy()
                ots.columns = ["price", "quantity"]
                ots["timestamp"] = ts
                ots = ots.sort_values("price", ascending=(d == "UP"))
                ots["cumul"] = ots["quantity"].cumsum()
                af = ots[ots["cumul"] <= 110].copy()
                mf = ots[ots["cumul"] >  110].copy()
                cross = ots[(ots["cumul"] > 110) & ((ots["cumul"] - ots["quantity"]) < 110)]
                if not cross.empty:
                    rc = cross.iloc[0]
                    qa = 110 - (rc["cumul"] - rc["quantity"])
                    qm = rc["quantity"] - qa
                    if qa > 0:
                        af = pd.concat([af, pd.DataFrame([{
                            "timestamp": ts, "price": rc["price"], "quantity": qa}])],
                            ignore_index=True)
                    if qm > 0 and not mf.empty:
                        mf.iloc[0, mf.columns.get_loc("quantity")] = qm
                for dft, rt in [(af, "aFRR"), (mf, "mFRR")]:
                    if dft.empty:
                        continue
                    dft = dft[["timestamp", "price", "quantity"]].copy()
                    dft["direction"] = d
                    dft["country"]   = "NL"
                    dft["reserve"]   = rt
                    (all_a if rt == "aFRR" else all_m).append(dft)

        def cat(lst):
            if not lst:
                return pd.DataFrame()
            return pd.concat(lst, ignore_index=True).drop_duplicates(
                subset=["timestamp", "direction", "price", "quantity", "reserve"])

        return cat(all_a), cat(all_m)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


# =============================================================================
# MAIN
# =============================================================================

def prefetch(start_date: str, end_date: str):
    all_dates = [d.strftime("%Y-%m-%d")
                 for d in pd.date_range(start_date, end_date, freq="D")]

    # Audit missing files
    missing_be = {r: [] for r in RESERVES}
    missing_de = {r: [] for r in RESERVES}
    missing_nl = {r: [] for r in RESERVES}

    for d in all_dates:
        dd = _data_dir(d)
        for reserve in RESERVES:
            if not (dd / f"{reserve.lower()}_be_{d}.csv").exists():
                missing_be[reserve].append(d)
            if not (dd / f"{reserve.lower()}_de_{d}.csv").exists():
                missing_de[reserve].append(d)
        af = dd / f"afrr_nl_{d}.csv"
        mf = dd / f"mfrr_nl_{d}.csv"
        if not af.exists(): missing_nl["aFRR"].append(d)
        if not mf.exists(): missing_nl["mFRR"].append(d)

    total = (sum(len(v) for v in missing_be.values()) +
             sum(len(v) for v in missing_de.values()) +
             len(missing_nl["aFRR"]))   # NL gives aFRR+mFRR in one call

    print(f"\nPeriode : {start_date} -> {end_date}  ({len(all_dates)} jours)")
    print(f"  BE aFRR manquants : {len(missing_be['aFRR'])}")
    print(f"  BE mFRR manquants : {len(missing_be['mFRR'])}")
    print(f"  DE aFRR manquants : {len(missing_de['aFRR'])}")
    print(f"  DE mFRR manquants : {len(missing_de['mFRR'])}")
    print(f"  NL manquants      : {len(missing_nl['aFRR'])} jours")

    if total == 0:
        print("\nTout est a jour.")
        return

    print(f"\nTelechargement...\n")

    done = ok = fail = 0
    t0   = time.time()
    bar_width = 40

    def _bar():
        elapsed = time.time() - t0
        eta = (elapsed / done * (total - done)) if done > 0 else 0
        filled = int(bar_width * done / total)
        return (f"[{'#'*filled}{'-'*(bar_width-filled)}]"
                f" {done}/{total} ({100*done/total:.1f}%)"
                f"  ETA {int(eta//60):02d}:{int(eta%60):02d}")

    def _dl(label, df, out):
        nonlocal done, ok, fail
        done += 1
        if df.empty:
            fail += 1
            print(f"  {label}  FAIL")
        else:
            df.to_csv(out, index=False)
            ok += 1
            n_up   = (df["direction"] == "UP").sum()
            n_down = (df["direction"] == "DOWN").sum()
            print(f"  {label}  OK  UP={n_up:<5} DOWN={n_down}")
        print(f"  {_bar()}", flush=True)

    # --- BE ---
    for reserve in RESERVES:
        for date_str in missing_be[reserve]:
            dd  = _data_dir(date_str)
            out = dd / f"{reserve.lower()}_be_{date_str}.csv"
            for attempt in range(3):
                df = fetch_be(date_str, reserve)
                if not df.empty: break
                time.sleep(5 * (attempt + 1))
            _dl(f"BE {reserve} {date_str}", df, out)

    # --- DE ---
    for reserve in RESERVES:
        for date_str in missing_de[reserve]:
            dd  = _data_dir(date_str)
            out = dd / f"{reserve.lower()}_de_{date_str}.csv"
            for attempt in range(3):
                df = fetch_de(date_str, reserve)
                if not df.empty: break
                time.sleep(5 * (attempt + 1))
            _dl(f"DE {reserve} {date_str}", df, out)

    # --- NL (one API call gives both aFRR + mFRR) ---
    nl_dates = sorted(set(missing_nl["aFRR"]) | set(missing_nl["mFRR"]))
    for date_str in nl_dates:
        dd = _data_dir(date_str)
        for attempt in range(3):
            df_a, df_m = fetch_nl(date_str)
            if not df_a.empty or not df_m.empty: break
            time.sleep(5 * (attempt + 1))
        if not df_a.empty and not (dd / f"afrr_nl_{date_str}.csv").exists():
            df_a.to_csv(dd / f"afrr_nl_{date_str}.csv", index=False)
        if not df_m.empty and not (dd / f"mfrr_nl_{date_str}.csv").exists():
            df_m.to_csv(dd / f"mfrr_nl_{date_str}.csv", index=False)
        _dl(f"NL      {date_str}", df_a if not df_a.empty else df_m, dd / f"afrr_nl_{date_str}.csv")

    print(f"\nResultat : {ok} OK  {fail} FAIL")

    # Final audit
    print("\nAudit final :")
    for tmpl in ['afrr_be_{d}.csv','mfrr_be_{d}.csv',
                 'afrr_de_{d}.csv','mfrr_de_{d}.csv',
                 'afrr_nl_{d}.csv','mfrr_nl_{d}.csv']:
        have = sum(1 for d in all_dates if (_data_dir(d) / tmpl.format(d=d)).exists())
        bar  = '#'*int(40*have/len(all_dates)) + '-'*(40-int(40*have/len(all_dates)))
        print(f"  {tmpl.split('{d}')[0]:<12} [{bar}] {have}/{len(all_dates)}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        start, end = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 2:
        start = end = sys.argv[1]
    else:
        start = "2025-01-01"
        end   = datetime.today().strftime("%Y-%m-%d")

    prefetch(start, end)
