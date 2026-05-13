#!/usr/bin/env python3
"""
Visualisation Solaire FR 2025 - 2 panneaux
  Haut : revenus cumulatifs jour par jour par strategie (S1, S2, S3)
  Bas  : nuage de points des QH de production (price_eur_mwh vs timestamp)
Sortie : outputs/reports/solar_fr_2025.png (300 DPI)
"""

import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

REPO    = Path(__file__).resolve().parent
CSV_DIR = REPO / "outputs" / "solar_fr" / "simulation_2025"
OUT_DIR = REPO / "outputs" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / "solar_fr_2025.png"

BG       = "#1a1a2e"
GRID_COL = "#ffffff"
TEXT_COL = "#e0e0e0"

STRAT_STYLES = {
    "S3": {"color": "#00d4ff", "lw": 2.5, "zorder": 5},
    "S2": {"color": "#ffd700", "lw": 1.5, "zorder": 3},
    "S1": {"color": "#a8e6cf", "lw": 1.5, "zorder": 2},
}

files = sorted(glob.glob(str(CSV_DIR / "day_*.csv")))
df = pd.concat(
    [pd.read_csv(f, parse_dates=["timestamp", "date"]) for f in files],
    ignore_index=True
)

strat_cols = {"S1": "s1_total", "S2": "s2_total", "S3": "s3_total"}
rows = []
for s, col in strat_cols.items():
    if col not in df.columns:
        continue
    tmp = df.groupby("date")[col].sum().reset_index()
    tmp["strategy"] = s
    tmp.rename(columns={col: "daily_rev"}, inplace=True)
    rows.append(tmp[["date", "strategy", "daily_rev"]])
daily = pd.concat(rows).sort_values("date")
daily["cumul"] = daily.groupby("strategy")["daily_rev"].cumsum()

totals = daily.groupby("strategy")["daily_rev"].sum()
best = totals.idxmax()
best_total = totals[best]
date_min = df["date"].min()
date_max = df["date"].max()

events = df[df["s1_production"] > 0].copy()
events = events.dropna(subset=["price_eur_mwh"])
events["size_pt"] = (events["s1_production"].clip(lower=0.1) * 6).clip(upper=80)
events["color"]   = np.where(events["price_eur_mwh"] > 0, "#4ade80", "#f87171")

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 10), facecolor=BG,
    gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.35}
)
for ax in (ax1, ax2):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.spines[:].set_color("#444466")
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)

for strat in ["S3", "S2", "S1"]:
    sub = daily[daily["strategy"] == strat]
    if sub.empty:
        continue
    st = STRAT_STYLES.get(strat, {"color": "#888888", "lw": 1.2})
    ax1.plot(sub["date"], sub["cumul"],
             color=st["color"], lw=st.get("lw", 1.5),
             ls=st.get("ls", "-"), zorder=st.get("zorder", 1),
             label=f"{strat}  ({sub['daily_rev'].sum():+.0f}EUR)")

ax1.axhline(0, color="#666688", lw=0.8, ls="--")
ax1.set_ylabel("Revenu cumulatif (EUR)", color=TEXT_COL, fontsize=10)
ax1.set_title(
    f"Solaire FR 2025 - Strategies d'optimisation\n"
    f"{best} (meilleure FR) total : {best_total:+.2f} EUR  "
    f"({date_min} -> {date_max})",
    color=TEXT_COL, fontsize=13, pad=12
)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.grid(True, color=GRID_COL, alpha=0.12, lw=0.6)
ax1.legend(loc="upper left", fontsize=9,
           facecolor="#12122a", edgecolor="#444466",
           labelcolor=TEXT_COL, framealpha=0.85)

ax2.scatter(events["timestamp"], events["price_eur_mwh"],
            c=events["color"], s=events["size_pt"],
            alpha=0.6, linewidths=0, zorder=3)
ax2.axhline(0, color="#aaaacc", lw=1.0, ls="--", alpha=0.7)
ax2.set_ylabel("Prix ISP (EUR/MWh)", color=TEXT_COL, fontsize=10)
ax2.set_title("QH de production S1 - Prix ISP au moment de la production",
              color=TEXT_COL, fontsize=11, pad=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.grid(True, color=GRID_COL, alpha=0.12, lw=0.6)

n_pos = int((events["price_eur_mwh"] > 0).sum())
n_neg = int((events["price_eur_mwh"] <= 0).sum())
ax2.legend(handles=[
    Line2D([0],[0], marker="o", color="none", markerfacecolor="#4ade80",
           markersize=8, label=f"ISP > 0 -> revenu  ({n_pos} QH)"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor="#f87171",
           markersize=8, label=f"ISP < 0 -> S3 ecrете  ({n_neg} QH)"),
], loc="upper right", fontsize=8.5,
   facecolor="#12122a", edgecolor="#444466",
   labelcolor=TEXT_COL, framealpha=0.85)

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=BG)
print(f"Saved: {OUT_PNG}")
plt.close(fig)
