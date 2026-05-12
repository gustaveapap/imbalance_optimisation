# CLAUDE CODE — verify_all.py
# Exécuter depuis la racine du repo : python3 verify_all.py
# Audit complet : données, simulations, cohérence logique, résultats

import os, sys, io, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from glob import glob
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "pandas", "-q"])
    import pandas as pd

RESET = "\033[0m"; GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; BOLD = "\033[1m"
ok  = lambda s: print(f"  {GREEN}✓{RESET} {s}")
err = lambda s: print(f"  {RED}✗{RESET} {s}")
warn= lambda s: print(f"  {YELLOW}!{RESET} {s}")
hdr = lambda s: print(f"\n{BOLD}{'='*70}\n{s}\n{'='*70}{RESET}")

report = []  # Tableau final

# ─────────────────────────────────────────────────────────────────────────────
# 1. GIT STATUS
# ─────────────────────────────────────────────────────────────────────────────
hdr("1. GIT STATUS")

result = subprocess.run(["git", "log", "--oneline", "-5"], capture_output=True, text=True)
print(result.stdout)

result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
untracked = [l for l in result.stdout.splitlines() if l.startswith("??")]
modified  = [l for l in result.stdout.splitlines() if not l.startswith("??")]

if untracked:
    warn(f"{len(untracked)} fichiers non-trackés :")
    for f in untracked: print(f"    {f}")
else:
    ok("Aucun fichier non-tracké")

if modified:
    warn(f"{len(modified)} fichiers modifiés non commités :")
    for f in modified: print(f"    {f}")
else:
    ok("Working tree propre")

# Scripts .py à la racine (désordre)
root_py = glob("*.py")
if root_py:
    warn(f"{len(root_py)} scripts .py à la racine (devraient être dans scripts/) :")
    for f in sorted(root_py): print(f"    {f}")
else:
    ok("Aucun script .py à la racine")

# ─────────────────────────────────────────────────────────────────────────────
# 2. INVENTAIRE SCRIPTS DE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
hdr("2. INVENTAIRE SCRIPTS DE SIMULATION")

sim_patterns = ["**/sim*ve*.py", "**/simulate*ve*.py", "**/sim*solar*.py",
                "**/*solar*sched*.py", "**/sim*be*.py", "**/sim*fr*.py"]
sim_scripts = sorted(set(
    f for pat in sim_patterns
    for f in glob(pat, recursive=True)
    if ".git" not in f and "venv" not in f and "__pycache__" not in f
))

if sim_scripts:
    ok(f"{len(sim_scripts)} scripts de simulation trouvés :")
    for f in sim_scripts: print(f"    {f}")
else:
    err("Aucun script de simulation trouvé !")

# ─────────────────────────────────────────────────────────────────────────────
# 3. VÉRIFICATION DONNÉES D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────
hdr("3. DONNÉES D'ENTRÉE — COUVERTURE ET COMPLÉTUDE")

def check_csv(label, paths_or_pattern, is_pattern=False):
    paths = glob(paths_or_pattern, recursive=True) if is_pattern else \
            ([paths_or_pattern] if isinstance(paths_or_pattern, str) else paths_or_pattern)
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        err(f"{label} — AUCUN FICHIER TROUVÉ")
        report.append({"sim": label, "data_ok": False, "period": "?", "rows": 0,
                        "source": "?", "best": "?"})
        return None
    path = paths[0]
    try:
        df = pd.read_csv(path, low_memory=False)
        date_col = next((c for c in df.columns
                         if any(k in c.lower() for k in ["date","time","dt","day","quarter"])), None)
        nulls = df.isnull().sum().sum()
        period = "?"
        if date_col:
            try:
                dates = pd.to_datetime(df[date_col], utc=True, errors="coerce").dropna()
                period = f"{dates.min().date()} → {dates.max().date()}"
            except Exception:
                period = "parse error"
        status = "OK" if nulls == 0 else f"⚠ {nulls} NaN"
        ok(f"{label} | {len(df)} lignes | {period} | {status}")
        print(f"      fichier : {path}")
        return df, path, period, len(df)
    except Exception as e:
        err(f"{label} — ERREUR LECTURE : {e}")
        return None

data_ve_be_26 = check_csv("Données VE BE 2026", "data_ve_2026/**/*.csv", is_pattern=True)
data_ve_fr_26 = check_csv("Données VE FR 2026", "data_ve_2026/**/*.csv", is_pattern=True)
data_solar_be = check_csv("Données Solar BE",   "data/raw/solar_be/**/*.csv", is_pattern=True)
data_merit_be = check_csv("Merit Order BE",     "data/raw/**/*merit*be*.csv", is_pattern=True)
data_merit_fr = check_csv("Merit Order FR",     "data/raw/**/*merit*fr*.csv", is_pattern=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4. VÉRIFICATION RÉSULTATS EXISTANTS
# ─────────────────────────────────────────────────────────────────────────────
hdr("4. RÉSULTATS — COHÉRENCE ET COMPLÉTUDE")

result_files = {
    "Solar BE 2026" : "outputs/solar_be/simulation_2026/summary_2026.csv",
    "Solar FR 2026" : "outputs/solar_fr/simulation_2026/summary_2026.csv",
    "Solar FR 2025" : "outputs/solar_fr/simulation_2025/summary_2025.csv",
    "Solar BE 2025" : "outputs/solar_be/simulation_2025/summary_2025.csv",
}

# VE results — chercher dynamiquement
ve_patterns = {
    "VE FR 2025" : ["**/simulation_ve_2025*fr*/**/*.csv", "**/simulation_ve_2025_fr*/*.csv"],
    "VE BE 2025" : ["**/simulation_ve_2025*be*/**/*.csv", "**/simulation_ve_2025_be*/*.csv"],
    "VE BE 2026" : ["outputs/ve_be/summary_daily_ve_be_2026.csv", "**/summary_daily_ve_be_2026.csv"],
    "VE FR 2026" : ["outputs/ve_fr/summary_daily_ve_fr_2026.csv", "**/summary_daily_ve_fr_2026.csv"],
}
for label, patterns in ve_patterns.items():
    found = []
    for p in patterns:
        found.extend(glob(p, recursive=True))
    found = [f for f in found if ".git" not in f and "__pycache__" not in f]
    result_files[label] = found[0] if found else None

summary_rows = []
for label, path in result_files.items():
    if path is None or not os.path.exists(path):
        err(f"{label} — MANQUANT ({path})")
        summary_rows.append({"Simulation": label, "Fichier": "MANQUANT",
                             "Lignes": "-", "Période": "-", "Meilleure stratégie": "-"})
        continue
    try:
        df = pd.read_csv(path)
        profit_col = next((c for c in df.columns
                           if any(k in c.lower() for k in ["profit","revenue","gain","eur","total"])), None)
        strat_col  = next((c for c in df.columns
                           if any(k in c.lower() for k in ["strat","scenario","name","strategy"])), None)
        date_col   = next((c for c in df.columns
                           if any(k in c.lower() for k in ["date","day","period"])), None)
        period = "?"
        if date_col:
            try:
                dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
                period = f"{dates.min().date()} → {dates.max().date()}"
            except Exception:
                period = str(df[date_col].iloc[[0,-1]].values)
        best = "-"
        if profit_col and strat_col and len(df) > 0:
            b = df.loc[df[profit_col].idxmax()]
            best = f"{b[strat_col]} = {b[profit_col]:.0f}€"
        elif profit_col and len(df) > 0:
            best = f"{df[profit_col].max():.0f}€"
        ok(f"{label} | {len(df)} lignes | best: {best}")
        print(f"      fichier : {path}")
        summary_rows.append({"Simulation": label, "Fichier": path,
                             "Lignes": len(df), "Période": period,
                             "Meilleure stratégie": best})
    except Exception as e:
        err(f"{label} — ERREUR : {e}")
        summary_rows.append({"Simulation": label, "Fichier": path,
                             "Lignes": "ERR", "Période": "-",
                             "Meilleure stratégie": str(e)})

# ─────────────────────────────────────────────────────────────────────────────
# 5. ANALYSE COHÉRENCE DES SCRIPTS (logique, source données, bug abs)
# ─────────────────────────────────────────────────────────────────────────────
hdr("5. COHÉRENCE DES SCRIPTS DE SIMULATION")

issues = []
for script in sim_scripts:
    try:
        content = open(script).read()
    except Exception:
        continue

    print(f"\n  [{script}]")

    # Source des données : API ou CSV local ?
    reads_csv = "read_csv" in content or "pd.read" in content
    calls_api = any(k in content for k in ["requests.", "http", "API", "fetch", "OAuth"])
    if reads_csv and not calls_api:
        warn(f"Lit des CSV locaux — ne fait PAS d'appel API")
        issues.append(f"{script}: lit CSV local, pas d'appel API")
    elif calls_api:
        ok("Appelle une API")
    else:
        warn("Source de données non identifiée")

    # Bug abs() corrigé ?
    # Chercher pattern problématique : abs() autour d'une somme de valeurs
    import re
    abs_patterns = re.findall(r"abs\([^)]{0,60}\)", content)
    if abs_patterns:
        ok(f"abs() présent ({len(abs_patterns)}x) — vérifier correction bug")
        for p in abs_patterns[:3]: print(f"      {p}")
    else:
        ok("Pas de abs() — bug non applicable")

    # Période hardcodée ?
    dates_hard = re.findall(r"202[0-9]-\d{2}-\d{2}", content)
    if dates_hard:
        warn(f"Dates hardcodées : {sorted(set(dates_hard))}")
        issues.append(f"{script}: dates hardcodées {sorted(set(dates_hard))}")
    else:
        ok("Pas de dates hardcodées")

    # Stratégies définies
    strats = re.findall(r"['\"]?(S\d+[v]?\d*)['\"]?", content)
    if strats:
        ok(f"Stratégies : {sorted(set(strats))}")
    else:
        warn("Aucune stratégie S1/S2/... détectée")

# ─────────────────────────────────────────────────────────────────────────────
# 6. FICHIERS REDONDANTS
# ─────────────────────────────────────────────────────────────────────────────
hdr("6. FICHIERS REDONDANTS OU DOUBLONS")

redundant_candidates = [
    "outputs/solar_be/simulation_be_2026.csv",
    "outputs/solar_fr/summary_2026.csv",
]
for f in redundant_candidates:
    if os.path.exists(f):
        df = pd.read_csv(f)
        warn(f"Redondant probable : {f} ({len(df)} lignes)")
    else:
        ok(f"Déjà supprimé : {f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. RAPPORT FINAL
# ─────────────────────────────────────────────────────────────────────────────
hdr("7. RAPPORT FINAL")

print(f"\n{'Simulation':<18} {'Fichier résultat':<10} {'Lignes':<8} {'Période':<30} {'Best'}")
print("-" * 90)
for row in summary_rows:
    exists = "✓" if row["Lignes"] not in ["-", "ERR"] else "✗"
    fname  = os.path.basename(str(row["Fichier"])) if row["Fichier"] != "MANQUANT" else "MANQUANT"
    print(f"{row['Simulation']:<18} {exists} {fname:<22} {str(row['Lignes']):<8} "
          f"{str(row['Période']):<30} {row['Meilleure stratégie']}")

if issues:
    print(f"\n{YELLOW}Problèmes à corriger avant réorganisation :{RESET}")
    for i in issues:
        print(f"  - {i}")
else:
    print(f"\n{GREEN}Aucun problème critique détecté.{RESET}")

print(f"\n{BOLD}Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}\n")
