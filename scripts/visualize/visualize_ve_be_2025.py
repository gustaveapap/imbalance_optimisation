#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

CSV_IN  = Path("simulation_ve_2025_be_v2/simulation_complete.csv")
PNG_OUT = Path("outputs/reports/ve_be_2025.png")
PNG_OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_IN, parse_dates=["timestamp"])
df["date"] = pd.to_datetime(df["date"])

STRATS   = ["S_BE_opt", "S1_Prudent", "S2_Ultra"]
COLORS   = {"S_BE_opt": "#00e5ff", "S1_Prudent": "#ff9800", "S2_Ultra": "#ab47bc"}
LABELS   = {"S_BE_opt": "S_BE_opt", "S1_Prudent": "S1 Prudent", "S2_Ultra": "S2 Ultra"}
BG       = "#0d1117"
GRID_COL = "#1f2937"

daily = (df.groupby(["strategy", "date"])["cost_eur"]
           .sum()
           .reset_index()
           .sort_values("date"))

cum = {}
for s in STRATS:
    sub = daily[daily["strategy"] == s].set_index("date")["cost_eur"]
    cum[s] = sub.cumsum()

total_opt = round(df[df["strategy"] == "S_BE_opt"]["cost_eur"].sum(), 2)

smart = df[(df["smart_kwh"] > 0) & df["strategy"].isin(STRATS)].copy()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                facecolor=BG,
                                gridspec_kw={"height_ratios": [1.4, 1]})
fig.subplots_adjust(hspace=0.35)

sign = "+" if total_opt >= 0 else ""
fig.suptitle(
    f"VE Belgique 2025 — Coût recharge  |  S_BE_opt total : {sign}{total_opt:.2f} €",
    color="white", fontsize=14, fontweight="bold", y=0.98
)

# panneau haut : cumulatif
ax1.set_facecolor(BG)
ax1.grid(True, color=GRID_COL, linewidth=0.6, linestyle="--")
ax1.axhline(0, color="#444", linewidth=0.8)

for s in STRATS:
    if s in cum and not cum[s].empty:
        ax1.plot(cum[s].index, cum[s].values,
                 color=COLORS[s], linewidth=1.8, label=LABELS[s])

ax1.set_ylabel("Coût cumulatif (€)", color="white", fontsize=11)
ax1.tick_params(colors="white")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
for spine in ax1.spines.values():
    spine.set_edgecolor(GRID_COL)
ax1.legend(facecolor="#1a1a2e", edgecolor=GRID_COL, labelcolor="white", fontsize=10)
ax1.set_title("Coûts cumulatifs par stratégie", color="#aaa", fontsize=11, pad=6)

# panneau bas : scatter ISP
ax2.set_facecolor(BG)
ax2.grid(True, color=GRID_COL, linewidth=0.6, linestyle="--")
ax2.axhline(0, color="#555", linewidth=1.0)

if not smart.empty:
    kwh_min, kwh_max = smart["smart_kwh"].min(), smart["smart_kwh"].max()
    sizes = 20 + 100 * (smart["smart_kwh"] - kwh_min) / (kwh_max - kwh_min + 1e-9)

    mask_neg = smart["isp"] < 0
    mask_pos = ~mask_neg

    if mask_neg.any():
        ax2.scatter(smart.loc[mask_neg, "date"], smart.loc[mask_neg, "isp"],
                    s=sizes[mask_neg], color="#00c853", alpha=0.75,
                    linewidths=0, label="ISP < 0 (gain)")
    if mask_pos.any():
        ax2.scatter(smart.loc[mask_pos, "date"], smart.loc[mask_pos, "isp"],
                    s=sizes[mask_pos], color="#ff1744", alpha=0.75,
                    linewidths=0, label="ISP ≥ 0 (coût)")

    n_strats = len(STRATS)
    n_events = len(smart) // n_strats
    n_neg    = int(mask_neg.sum()) // n_strats
    ax2.set_title(
        f"Événements de charge smart — {n_events} total  ({n_neg} à prix négatif)",
        color="#aaa", fontsize=11, pad=6
    )

ax2.set_ylabel("ISP (€/MWh)", color="white", fontsize=11)
ax2.tick_params(colors="white")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax2.xaxis.set_major_locator(mdates.MonthLocator())
for spine in ax2.spines.values():
    spine.set_edgecolor(GRID_COL)
ax2.legend(facecolor="#1a1a2e", edgecolor=GRID_COL, labelcolor="white", fontsize=10)

fig.savefig(PNG_OUT, dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Sauvegarde : {PNG_OUT}  (cout S_BE_opt = {sign}{total_opt:.2f} EUR)")
