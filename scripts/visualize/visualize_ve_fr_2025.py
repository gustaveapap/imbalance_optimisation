#!/usr/bin/env python3
"""
Visualisation VE FR 2025 — 2 panneaux
  Haut  : coûts cumulatifs jour par jour par stratégie
  Bas   : nuage de points des événements de charge (ISP, date, volume)
Sortie  : outputs/reports/ve_fr_2025.png (300 DPI)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

REPO    = Path(__file__).resolve().parents[2]
CSV     = REPO / "simulation_ve_2025_fr" / "simulation_complete.csv"
OUT_DIR = REPO / "outputs" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / "ve_fr_2025.png"

BG       = "#1a1a2e"
GRID_COL = "#ffffff"
TEXT_COL = "#e0e0e0"

STRAT_STYLES = {
    "S_BE_opt":  {"color": "#00d4ff", "lw": 2.5, "zorder": 5},
    "S1_Prudent":{"color": "#a8e6cf", "lw": 1.5, "zorder": 4},
    "S2_Ultra":  {"color": "#ffd700", "lw": 1.5, "zorder": 3},
    "S8v4":      {"color": "#ff6b6b", "lw": 1.5, "zorder": 2},
    "S8v2":      {"color": "#c084fc", "lw": 1.2, "zorder": 1, "ls": "--"},
    "S9_Hybrid": {"color": "#fb923c", "lw": 1.2, "zorder": 1, "ls": "--"},
}

df = pd.read_csv(CSV, parse_dates=["timestamp", "date"])

daily = (df.groupby(["strategy", "date"])["cost_eur"]
           .sum().reset_index().sort_values("date"))
daily["cumul"] = daily.groupby("strategy")["cost_eur"].cumsum()

best_total = daily[daily["strategy"] == "S_BE_opt"]["cost_eur"].sum()

events = df[(df["smart_triggered"] == True) & (df["strategy"] == "S_BE_opt")].copy()
events = events.dropna(subset=["isp"])
events["size_pt"] = (events["smart_kwh"].clip(lower=0.1) * 8).clip(upper=80)
events["color"]   = np.where(events["isp"] < 0, "#4ade80", "#f87171")

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

for strat in ["S_BE_opt", "S1_Prudent", "S2_Ultra", "S8v4", "S8v2", "S9_Hybrid"]:
    sub = daily[daily["strategy"] == strat]
    if sub.empty:
        continue
    st = STRAT_STYLES.get(strat, {"color": "#888888", "lw": 1.2})
    ax1.plot(sub["date"], sub["cumul"],
             color=st["color"], lw=st.get("lw", 1.5),
             ls=st.get("ls", "-"), zorder=st.get("zorder", 1),
             label=f"{strat}  ({sub['cost_eur'].sum():+.0f}EUR)")

ax1.axhline(0, color="#666688", lw=0.8, ls="--")
ax1.set_ylabel("Cout cumulatif (EUR)", color=TEXT_COL, fontsize=10)
ax1.set_title(
    f"VE FR 2025 - Strategies de charge intelligente\n"
    f"S_BE_opt total : {best_total:+.2f} EUR (364 jours)",
    color=TEXT_COL, fontsize=13, pad=12
)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.grid(True, color=GRID_COL, alpha=0.12, lw=0.6)
ax1.legend(loc="upper left", fontsize=8.5,
           facecolor="#12122a", edgecolor="#444466",
           labelcolor=TEXT_COL, framealpha=0.85)

ax2.scatter(events["timestamp"], events["isp"],
            c=events["color"], s=events["size_pt"],
            alpha=0.6, linewidths=0, zorder=3)
ax2.axhline(0, color="#aaaacc", lw=1.0, ls="--", alpha=0.7)
ax2.set_ylabel("ISP (EUR/MWh)", color=TEXT_COL, fontsize=10)
ax2.set_title("Evenements de charge S_BE_opt - ISP au declenchement",
              color=TEXT_COL, fontsize=11, pad=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.grid(True, color=GRID_COL, alpha=0.12, lw=0.6)

n_gain = int((events["isp"] < 0).sum())
n_cost = int((events["isp"] >= 0).sum())
ax2.legend(handles=[
    Line2D([0],[0], marker="o", color="none", markerfacecolor="#4ade80",
           markersize=8, label=f"ISP < 0 -> gain  ({n_gain} QH)"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor="#f87171",
           markersize=8, label=f"ISP > 0 -> cout  ({n_cost} QH)"),
], loc="upper right", fontsize=8.5,
   facecolor="#12122a", edgecolor="#444466",
   labelcolor=TEXT_COL, framealpha=0.85)

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=BG)
print(f"Saved: {OUT_PNG}")
plt.close(fig)
