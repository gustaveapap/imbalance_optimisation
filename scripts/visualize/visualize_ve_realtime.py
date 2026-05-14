#!/usr/bin/env python3
"""
Visualisation VE temps reel - dernieres 24H
  Haut  : ISP BE (cyan) + ISP FR (jaune) + evenements de charge marques
  Milieu: puissance de charge smart (kWh/QH) BE et FR
  Bas   : SOC (etat de charge batterie) BE et FR
Sortie : outputs/reports/ve_realtime_24h.png (300 DPI)
Usage  : python visualize_ve_realtime.py   (relancer pour actualiser)
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO     = Path(__file__).resolve().parents[2]
CSV_BE   = REPO / "outputs" / "ve_be" / "simulation_ve_be_2026.csv"
CSV_FR   = REPO / "outputs" / "ve_fr" / "simulation_ve_fr_2026.csv"
OUT_DIR  = REPO / "outputs" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG  = OUT_DIR / "ve_realtime_24h.png"

BG       = "#1a1a2e"
GRID_COL = "#ffffff"
TEXT_COL = "#e0e0e0"
COL_BE   = "#00d4ff"
COL_FR   = "#ffd700"

BEST_BE = "S_BE_OPT"
BEST_FR = "S1"

df_be_all = pd.read_csv(CSV_BE, parse_dates=["timestamp"])
df_fr_all = pd.read_csv(CSV_FR, parse_dates=["timestamp"])

latest = max(df_be_all["timestamp"].max(), df_fr_all["timestamp"].max())
cutoff = latest - pd.Timedelta(hours=24)

df_be = df_be_all[
    (df_be_all["strategy"] == BEST_BE) & (df_be_all["timestamp"] >= cutoff)
].copy().sort_values("timestamp")

df_fr = df_fr_all[
    (df_fr_all["strategy"] == BEST_FR) & (df_fr_all["timestamp"] >= cutoff)
].copy().sort_values("timestamp")

gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")
data_up_to = latest.strftime("%Y-%m-%d %H:%M")

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(16, 12), facecolor=BG,
    gridspec_kw={"height_ratios": [1.4, 1, 1], "hspace": 0.40}
)
for ax in (ax1, ax2, ax3):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8.5)
    ax.spines[:].set_color("#444466")
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)

# ── Panel 1 : ISP + evenements de charge ──────────────────────────────────────
ax1.plot(df_be["timestamp"], df_be["isp"],
         color=COL_BE, lw=1.5, alpha=0.9, zorder=3, label=f"ISP BE ({BEST_BE})")
ax1.plot(df_fr["timestamp"], df_fr["isp"],
         color=COL_FR, lw=1.5, alpha=0.9, zorder=3, label=f"ISP FR ({BEST_FR})")

ev_be = df_be[df_be["smart_kwh"] > 0]
ax1.scatter(ev_be["timestamp"], ev_be["isp"],
            c=np.where(ev_be["isp"] < 0, "#4ade80", "#f87171"),
            s=(ev_be["smart_kwh"].clip(lower=0.1) * 10).clip(upper=120),
            zorder=5, linewidths=0, alpha=0.9)

ev_fr = df_fr[df_fr["smart_kwh"] > 0]
ax1.scatter(ev_fr["timestamp"], ev_fr["isp"],
            c=np.where(ev_fr["isp"] < 0, "#4ade80", "#f87171"),
            s=(ev_fr["smart_kwh"].clip(lower=0.1) * 10).clip(upper=120),
            marker="D", zorder=5, linewidths=0, alpha=0.9)

ax1.axhline(0, color="#aaaacc", lw=1.0, ls="--", alpha=0.7)
ax1.set_ylabel("ISP (EUR/MWh)", color=TEXT_COL, fontsize=10)
ax1.set_title(
    f"VE Temps Reel - Dernieres 24H  |  Donnees jusqu'a : {data_up_to}\n"
    f"Generated : {gen_time}",
    color=TEXT_COL, fontsize=13, pad=10
)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d %b"))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax1.grid(True, color=GRID_COL, alpha=0.10, lw=0.6)
ax1.legend(handles=[
    Line2D([0],[0], color=COL_BE, lw=2, label=f"ISP BE ({BEST_BE})"),
    Line2D([0],[0], color=COL_FR, lw=2, label=f"ISP FR ({BEST_FR})"),
    Line2D([0],[0], marker="o",  color="none", markerfacecolor="#4ade80",
           markersize=8, label="Charge ISP<0 (gain) BE"),
    Line2D([0],[0], marker="D",  color="none", markerfacecolor="#4ade80",
           markersize=7, label="Charge ISP<0 (gain) FR"),
    Line2D([0],[0], marker="o",  color="none", markerfacecolor="#f87171",
           markersize=8, label="Charge ISP>0 (cout) BE"),
], loc="upper left", fontsize=8, facecolor="#12122a", edgecolor="#444466",
   labelcolor=TEXT_COL, framealpha=0.85, ncol=2)

# ── Panel 2 : Puissance de charge smart ───────────────────────────────────────
bar_w = pd.Timedelta(minutes=12)

if not ev_be.empty:
    ax2.bar(ev_be["timestamp"], ev_be["smart_kwh"],
            width=bar_w, color=COL_BE, alpha=0.8, label=f"BE {BEST_BE}", zorder=3)
if not ev_fr.empty:
    ax2.bar(ev_fr["timestamp"] + pd.Timedelta(minutes=2), ev_fr["smart_kwh"],
            width=bar_w, color=COL_FR, alpha=0.8, label=f"FR {BEST_FR}", zorder=3)

ax2.set_ylabel("Charge smart (kWh/QH)", color=TEXT_COL, fontsize=10)
ax2.set_title("Puissance de charge intelligente par QH",
              color=TEXT_COL, fontsize=10, pad=6)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d %b"))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax2.grid(True, color=GRID_COL, alpha=0.10, lw=0.6)

be_kwh = ev_be["smart_kwh"].sum() if not ev_be.empty else 0
fr_kwh = ev_fr["smart_kwh"].sum() if not ev_fr.empty else 0
be_cost = ev_be["cost_eur"].sum() if not ev_be.empty else 0
fr_cost = ev_fr["cost_eur"].sum() if not ev_fr.empty else 0
ax2.legend(handles=[
    Patch(facecolor=COL_BE, alpha=0.8,
          label=f"BE {BEST_BE}  {be_kwh:.1f} kWh  {be_cost:+.2f} EUR"),
    Patch(facecolor=COL_FR, alpha=0.8,
          label=f"FR {BEST_FR}  {fr_kwh:.1f} kWh  {fr_cost:+.2f} EUR"),
], loc="upper left", fontsize=8.5, facecolor="#12122a", edgecolor="#444466",
   labelcolor=TEXT_COL, framealpha=0.85)

# ── Panel 3 : SOC ─────────────────────────────────────────────────────────────
ax3.plot(df_be["timestamp"], df_be["soc_kwh"],
         color=COL_BE, lw=2, label=f"SOC BE ({BEST_BE})", zorder=3)
ax3.plot(df_fr["timestamp"], df_fr["soc_kwh"],
         color=COL_FR, lw=2, label=f"SOC FR ({BEST_FR})", zorder=3)

ax3.fill_between(df_be["timestamp"], df_be["soc_kwh"],
                 alpha=0.12, color=COL_BE, zorder=2)
ax3.fill_between(df_fr["timestamp"], df_fr["soc_kwh"],
                 alpha=0.12, color=COL_FR, zorder=2)

ax3.set_ylabel("SOC (kWh)", color=TEXT_COL, fontsize=10)
ax3.set_title("Etat de charge batterie VE (SOC)",
              color=TEXT_COL, fontsize=10, pad=6)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d %b"))
ax3.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax3.grid(True, color=GRID_COL, alpha=0.10, lw=0.6)
ax3.legend(loc="upper left", fontsize=8.5, facecolor="#12122a", edgecolor="#444466",
           labelcolor=TEXT_COL, framealpha=0.85)

for ax in (ax1, ax2, ax3):
    ax.set_xlim(cutoff, latest + pd.Timedelta(minutes=30))

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=BG)
print(f"Saved: {OUT_PNG}")
plt.close(fig)
