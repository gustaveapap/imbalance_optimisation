#!/usr/bin/env python3
"""
Visualisation VE FR 2026 — 2 panneaux
  Haut  : couts cumulatifs jour par jour par strategie (S1, S2, S_BE_OPT)
  Bas   : nuage de points des evenements de charge S_BE_OPT
Sortie  : outputs/reports/ve_fr_2026.png (300 DPI)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

REPO    = Path(__file__).resolve().parent
CSV     = REPO / "outputs" / "ve_fr" / "simulation_ve_fr_2026.csv"
OUT_DIR = REPO / "outputs" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / "ve_fr_2026.png"

BG       = "#1a1a2e"
GRID_COL = "#ffffff"
TEXT_COL = "#e0e0e0"

STRATS = ["S_BE_OPT", "S1", "S2"]
STRAT_STYLES = {
    "S_BE_OPT": {"color": "#00d4ff", "lw": 2.5, "zorder": 5},
    "S1":       {"color": "#a8e6cf", "lw": 1.5, "zorder": 3},
    "S2":       {"color": "#ffd700", "lw": 1.5, "zorder": 2},
}

df = pd.read_csv(CSV, parse_dates=["timestamp", "date"])

daily = (df.groupby(["strategy", "date"])["cost_eur"]
           .sum().reset_index().sort_values("date"))
daily["cumul"] = daily.groupby("strategy")["cost_eur"].cumsum()

opt_total = daily[daily["strategy"] == "S_BE_OPT"]["cost_eur"].sum()
date_min  = df["date"].min().strftime("%Y-%m-%d")
date_max  = df["date"].max().strftime("%Y-%m-%d")

events = df[(df["strategy"] == "S_BE_OPT") & (df["smart_kwh"] > 0)].copy()
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

for strat in STRATS:
    sub = daily[daily["strategy"] == strat]
    if sub.empty:
        continue
    st = STRAT_STYLES[strat]
    ax1.plot(sub["date"], sub["cumul"],
             color=st["color"], lw=st["lw"], zorder=st["zorder"],
             label=f"{strat}  ({sub['cost_eur'].sum():+.0f} EUR)")

sign = "+" if opt_total >= 0 else ""
ax1.axhline(0, color="#666688", lw=0.8, ls="--")
ax1.set_ylabel("Cout cumulatif (EUR)", color=TEXT_COL, fontsize=10)
ax1.set_title(
    f"VE France 2026 — Strategies de charge intelligente\n"
    f"S_BE_opt total : {sign}{opt_total:.2f} EUR  ({date_min} -> {date_max})",
    color=TEXT_COL, fontsize=13, pad=12
)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.grid(True, color=GRID_COL, alpha=0.12, lw=0.6)
ax1.legend(loc="upper left", fontsize=9,
           facecolor="#12122a", edgecolor="#444466",
           labelcolor=TEXT_COL, framealpha=0.85)

ax2.scatter(events["timestamp"], events["isp"],
            c=events["color"], s=events["size_pt"],
            alpha=0.6, linewidths=0, zorder=3)
ax2.axhline(0, color="#aaaacc", lw=1.0, ls="--", alpha=0.7)
ax2.set_ylabel("ISP (EUR/MWh)", color=TEXT_COL, fontsize=10)

n_gain = int((events["isp"] < 0).sum())
n_cost = int((events["isp"] >= 0).sum())
ax2.set_title(
    f"Evenements de charge S_BE_opt — {len(events)} total  ({n_gain} a prix negatif)",
    color=TEXT_COL, fontsize=11, pad=8
)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.grid(True, color=GRID_COL, alpha=0.12, lw=0.6)
ax2.legend(handles=[
    Line2D([0],[0], marker="o", color="none", markerfacecolor="#4ade80",
           markersize=8, label=f"ISP < 0  gain  ({n_gain} QH)"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor="#f87171",
           markersize=8, label=f"ISP >= 0  cout  ({n_cost} QH)"),
], loc="upper right", fontsize=8.5,
   facecolor="#12122a", edgecolor="#444466",
   labelcolor=TEXT_COL, framealpha=0.85)

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=BG)
print(f"Saved: {OUT_PNG}  (S_BE_opt = {sign}{opt_total:.2f} EUR)")
plt.close(fig)
