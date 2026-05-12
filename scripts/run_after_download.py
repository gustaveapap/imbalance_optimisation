#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io, time, subprocess, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import pandas as pd

REPO      = Path(__file__).resolve().parents[1]
DATA_2025 = REPO / "data_ve_2025"
DATA_2026 = REPO / "data_ve_2026"
VENV_PY   = REPO / "venv" / "Scripts" / "python.exe"

ALL_25 = [d.strftime("%Y-%m-%d") for d in pd.date_range("2025-01-01", "2025-12-31")]
ALL_26 = [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-01-01", "2026-05-11")]

def count(data_dir, tmpl, dates):
    return sum(1 for d in dates if (data_dir / tmpl.format(d=d)).exists())

def audit():
    print("\n--- Audit ---")
    all_ok = True
    for tmpl, data_dir, dates, label in [
        ("afrr_be_{d}.csv", DATA_2025, ALL_25, "2025"),
        ("mfrr_be_{d}.csv", DATA_2025, ALL_25, "2025"),
        ("afrr_de_{d}.csv", DATA_2025, ALL_25, "2025"),
        ("mfrr_de_{d}.csv", DATA_2025, ALL_25, "2025"),
        ("afrr_nl_{d}.csv", DATA_2025, ALL_25, "2025"),
        ("mfrr_nl_{d}.csv", DATA_2025, ALL_25, "2025"),
        ("afrr_be_{d}.csv", DATA_2026, ALL_26, "2026"),
        ("mfrr_be_{d}.csv", DATA_2026, ALL_26, "2026"),
        ("afrr_de_{d}.csv", DATA_2026, ALL_26, "2026"),
        ("mfrr_de_{d}.csv", DATA_2026, ALL_26, "2026"),
        ("afrr_nl_{d}.csv", DATA_2026, ALL_26, "2026"),
        ("mfrr_nl_{d}.csv", DATA_2026, ALL_26, "2026"),
    ]:
        n = count(data_dir, tmpl, dates)
        total = len(dates)
        ok = (n >= total - 2)  # tolerate up to 2 missing (e.g. API gaps)
        sym = "OK" if ok else "!!"
        print(f"  [{sym}] {label} {tmpl.split('{d}')[0]:<12}: {n}/{total}")
        if not ok:
            all_ok = False
    return all_ok

# =============================================================================
# STEP 1 — Wait for downloads to complete
# =============================================================================
print("Waiting for downloads to complete...")
while True:
    afrr_de_26 = count(DATA_2026, "afrr_de_{d}.csv", ALL_26)
    mfrr_de_26 = count(DATA_2026, "mfrr_de_{d}.csv", ALL_26)
    nl_26      = count(DATA_2026, "afrr_nl_{d}.csv", ALL_26)
    de_25      = count(DATA_2025, "mfrr_de_{d}.csv", ALL_25)
    print(f"  aFRR_DE26: {afrr_de_26}/131  mFRR_DE26: {mfrr_de_26}/131  NL26: {nl_26}/131  mFRR_DE25: {de_25}/365", flush=True)
    if afrr_de_26 >= 129 and mfrr_de_26 >= 129 and nl_26 >= 116 and de_25 >= 363:
        print("Downloads look complete.")
        break
    time.sleep(30)

audit()

# =============================================================================
# STEP 2 — Clear stale merit cache and simulation outputs
# =============================================================================
print("\nClearing stale merit cache and simulation outputs...")

merit_cache = DATA_2026 / "merit_cache"
if merit_cache.exists():
    shutil.rmtree(merit_cache)
    merit_cache.mkdir()
    print(f"  Cleared {merit_cache}")

for sim_dir in [
    REPO / "outputs" / "solar_fr" / "simulation_2026",
    REPO / "outputs" / "solar_be" / "simulation_2026",
]:
    if sim_dir.exists():
        for f in sim_dir.glob("*.csv"):
            f.unlink()
        print(f"  Cleared {sim_dir}")

# =============================================================================
# STEP 3 — Run FR 2026 simulation
# =============================================================================
print("\n" + "="*60)
print("Running FR 2026 simulation...")
print("="*60, flush=True)
r = subprocess.run(
    [str(VENV_PY), str(REPO / "optimizers" / "solar_fr" / "sim_fr_2026.py")],
    cwd=str(REPO)
)
print(f"FR sim exit code: {r.returncode}")

# =============================================================================
# STEP 4 — Run BE 2026 simulation
# =============================================================================
print("\n" + "="*60)
print("Running BE 2026 simulation...")
print("="*60, flush=True)
r = subprocess.run(
    [str(VENV_PY), str(REPO / "optimizers" / "solar_be" / "sim_be_2026.py")],
    cwd=str(REPO)
)
print(f"BE sim exit code: {r.returncode}")

# =============================================================================
# STEP 5 — Compare
# =============================================================================
print("\n" + "="*60)
print("Running comparison FR vs BE 2026...")
print("="*60, flush=True)
r = subprocess.run(
    [str(VENV_PY), str(REPO / "optimizers" / "compare_fr_be_2025.py"), "2026"],
    cwd=str(REPO), capture_output=False
)

print("\nAll done.")
