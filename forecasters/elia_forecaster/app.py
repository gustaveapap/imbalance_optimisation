from flask import Flask, Response, jsonify, request
import io, os, time, threading, subprocess, glob, requests, warnings
from datetime import datetime, timedelta
import numpy as np, pandas as pd, pytz, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.graph_objects as go
import plotly.subplots as sp

# Supprimer les warnings sklearn
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

from system_imbalance_forecaster.forecaster import SystemImbalanceForecaster
from system_imbalance_forecaster.utils.feature_utils import extract_features

# ✅ Import wakepy pour empêcher mise en veille
try:
    from wakepy import keep
    WAKEPY_AVAILABLE = True
except ImportError:
    WAKEPY_AVAILABLE = False

# --------------------------------------------------------------------------------------
# CONFIGURATION GÉNÉRALE
# --------------------------------------------------------------------------------------

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "imbalance_forecaster_v1.joblib"
forecaster = SystemImbalanceForecaster(str(MODEL_PATH))
BRUSSELS = pytz.timezone("Europe/Brussels")
DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")

def today_str_bxl():
    return datetime.now(BRUSSELS).strftime("%Y-%m-%d")

# --------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------

def as_naive_bxl(ts: pd.Timestamp) -> pd.Timestamp:
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is not None:
            return ts.tz_convert(BRUSSELS).tz_localize(None)
        return ts
    if ts.tzinfo is not None:
        return ts.astimezone(BRUSSELS).replace(tzinfo=None)
    return ts

def floor_qh_naive_bxl(ts: pd.Timestamp) -> pd.Timestamp:
    return as_naive_bxl(pd.Timestamp(ts)).floor("15min")

def wait_until(target_dt_bxl: datetime):
    now = datetime.now(BRUSSELS)
    wait_seconds = (target_dt_bxl - now).total_seconds()
    if wait_seconds > 0:
        time.sleep(wait_seconds)

def print_separator():
    print("\n" + "="*80)

def print_section(title):
    print(f"\n{'-'*80}")
    print(f"  {title}")
    print(f"{'-'*80}")

# --------------------------------------------------------------------------------------
# PAGE D'ACCUEIL
# --------------------------------------------------------------------------------------
@app.route("/")
def home():
    today = datetime.now(BRUSSELS).date()
    start_default = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    end_default = today.strftime("%Y-%m-%d")
    last_update = datetime.now(BRUSSELS).strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
    <html>
    <head>
        <title>⚡ Elia Forecast Dashboard</title>
        <style>
            body {{
                background-color: #0b0e16;
                color: white;
                text-align: center;
                font-family: Arial, sans-serif;
                margin-top: 80px;
            }}
            input[type=date] {{
                padding: 8px;
                border-radius: 6px;
                border: 1px solid #00bfff;
                background-color: #11182b;
                color: white;
                margin: 5px;
            }}
            button {{
                background-color: deepskyblue;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                color: black;
                font-weight: bold;
                cursor: pointer;
            }}
            button:hover {{
                background-color: #00a0e0;
            }}
            .update {{
                margin-top: 30px;
                color: #999;
            }}
        </style>
    </head>
    <body>
        <h1>⚡ Elia Forecast Dashboard</h1>
        <form action="/dashboard" method="get" target="_blank">
            <label>Du :</label>
            <input type="date" name="start" value="{start_default}">
            <label>au :</label>
            <input type="date" name="end" value="{end_default}">
            <button type="submit">Afficher le dashboard</button>
        </form>
        <div class='update'>Dernière mise à jour : {last_update}</div>
        <div style="margin-top: 30px;">
            <a href="/evaluation" target="_blank" style="text-decoration: none;">
                <button type="button">Evaluation Dashboard</button>
            </a>
        </div>
        <div style="margin-top: 16px;">
            <a href="/dashboard_rte" target="_blank" style="text-decoration: none;">
                <button type="button" style="background-color:#e8543a;">🇫🇷 RTE FR Dashboard</button>
            </a>
        </div>
        <div style="margin-top: 16px;">
            <a href="/reports" target="_blank" style="text-decoration: none;">
                <button type="button" style="background-color:#7c3aed;">📊 Reports 2026</button>
            </a>
        </div>
    </body>
    </html>
    """
    return html

# --------------------------------------------------------------------------------------
# MODULE-LEVEL HELPERS (used by /dashboard and /evaluation)
# --------------------------------------------------------------------------------------
def fetch_opendatasoft(dataset, start_dt, end_dt, page_size=1000):
    url = "https://external-elia.opendatasoft.com/api/records/1.0/search/"
    ws_utc = pd.Timestamp(start_dt).tz_localize(BRUSSELS).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    we_utc = pd.Timestamp(end_dt).tz_localize(BRUSSELS).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    all_records, start_i = [], 0
    while True:
        params = {"dataset": dataset, "q": f"datetime:[{ws_utc} TO {we_utc}]",
                  "rows": page_size, "start": start_i, "sort": "datetime"}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get("records", [])
        if not data:
            break
        all_records.extend(data)
        if len(data) < page_size:
            break
        start_i += page_size
    if not all_records:
        return pd.DataFrame(columns=["Datetime", "System imbalance", "imbalanceprice"])
    df = pd.DataFrame([{
        "Datetime":         rec["fields"].get("datetime"),
        "System imbalance": rec["fields"].get("systemimbalance"),
        "imbalanceprice":   rec["fields"].get("imbalanceprice"),
    } for rec in all_records])
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True).dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    df["System imbalance"] = pd.to_numeric(df["System imbalance"], errors="coerce")
    df["imbalanceprice"]   = pd.to_numeric(df["imbalanceprice"],   errors="coerce")
    return df


def fetch_actuals_combined(start_dt, end_dt, chunk_days=60):
    # ODS v1 API hard-limits start+rows <= 10 000 (~10 400 QHs max per call).
    # Chunk into 60-day windows (5 760 QHs each) to stay well under that limit.
    frames = []
    cursor = pd.Timestamp(start_dt)
    end_ts = pd.Timestamp(end_dt)
    while cursor < end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days), end_ts)
        df134 = fetch_opendatasoft("ods134", cursor, chunk_end)
        df162 = fetch_opendatasoft("ods162", cursor, chunk_end)
        frames.append(pd.concat([df134, df162], ignore_index=True))
        cursor = chunk_end
    if not frames:
        return pd.DataFrame(columns=["Datetime", "System imbalance", "imbalanceprice"])
    df = pd.concat(frames, ignore_index=True)
    return (df.drop_duplicates(subset=["Datetime"], keep="last")
              .sort_values("Datetime")
              .reset_index(drop=True))


def merge_and_flags(forecasts_df, actuals_df):
    forecasts_df = forecasts_df.copy()
    actuals_df   = actuals_df.copy()
    forecasts_df["Datetime"] = forecasts_df["Datetime"].dt.floor("15min")
    actuals_df["Datetime"]   = actuals_df["Datetime"].dt.floor("15min")
    merged = pd.merge(forecasts_df, actuals_df, on="Datetime", how="inner")
    merged["cap_min"] = merged[["cap_BE", "cap_EU"]].min(axis=1, skipna=True)
    merged["cap_max"] = merged[["cap_BE", "cap_EU"]].max(axis=1, skipna=True)
    merged["has_range"]      = merged["cap_min"].notna() & merged["cap_max"].notna()
    merged["CorrectForecast"] = (merged["Forecast"] * merged["System imbalance"] > 0)
    merged["InRange"] = merged["has_range"] & (
        (merged["imbalanceprice"] >= merged["cap_min"]) &
        (merged["imbalanceprice"] <= merged["cap_max"])
    )
    return merged


def compute_all_metrics(merged):
    n_total = len(merged)
    acc_forecast = round(merged["CorrectForecast"].mean() * 100, 1) if n_total else None
    subset_range = merged[merged["has_range"]]
    acc_range_global = round(subset_range["InRange"].mean() * 100, 1) if len(subset_range) else None
    subset_cf = subset_range[subset_range["CorrectForecast"]]
    acc_range_when_cf = round(subset_cf["InRange"].mean() * 100, 1) if len(subset_cf) else None
    corr = round(merged["Forecast"].corr(merged["System imbalance"]), 3)
    return {
        "Forecast accuracy":                  acc_forecast,
        "Range accuracy":                     acc_range_global,
        "Range accuracy – when forecast correct": acc_range_when_cf,
        "Correlation":                        corr,
    }


def plot_bloomberg_dashboard_points(merged, metrics):
    fig = sp.make_subplots(rows=3, cols=1, row_heights=[0.6, 0.25, 0.6], specs=[[{}],[{}],[{}]],
                           vertical_spacing=0.15,
                           subplot_titles=["Price Range Accuracy (all QH)", None, "System Imbalance vs Forecast"])
    color_bg = "#0b0e16"; color_fill = "rgba(0,150,255,0.15)"
    color_inrange = "deepskyblue"; color_outrange = "orange"
    color_bar_system = "rgba(0,212,255,0.55)"; color_bar_forecast = "rgba(255,204,0,0.55)"
    color_grid = "rgba(255,255,255,0.08)"
    rng = merged[merged["has_range"]].sort_values("Datetime")
    fig.add_trace(go.Scatter(
        x=pd.Series(list(rng["Datetime"]) + list(rng["Datetime"][::-1])),
        y=pd.Series(list(rng["cap_min"])  + list(rng["cap_max"][::-1])),
        fill="toself", fillcolor=color_fill, line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip", name="Price Range"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=merged["Datetime"][merged["InRange"]], y=merged["imbalanceprice"][merged["InRange"]],
        mode="markers", marker=dict(color=color_inrange, size=7), name="In range"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=merged["Datetime"][~merged["InRange"]], y=merged["imbalanceprice"][~merged["InRange"]],
        mode="markers", marker=dict(color=color_outrange, size=7, symbol="x"), name="Out of range"), row=1, col=1)
    kpi_titles = list(metrics.keys()); kpi_values = list(metrics.values())
    positions = [0.15, 0.42, 0.68, 0.90]
    for i, (title, val) in enumerate(zip(kpi_titles, kpi_values)):
        title_wrapped = title.replace(" – ", "<br>")
        val_txt = f"{val}%" if isinstance(val, (float, int)) else str(val)
        fig.add_annotation(xref="paper", yref="paper", x=positions[i], y=0.42,
                           text=(f"<b style='font-size:22px;color:white'>{val_txt}</b>"
                                 f"<br><span style='font-size:14px;color:#cccccc'>{title_wrapped}</span>"),
                           showarrow=False, xanchor="center", align="center")
    ms = merged.sort_values("Datetime")
    fig.add_trace(go.Bar(x=ms["Datetime"], y=ms["System imbalance"],
                         marker_color=color_bar_system, opacity=0.6, name="System"), row=3, col=1)
    fig.add_trace(go.Bar(x=ms["Datetime"], y=ms["Forecast"],
                         marker_color=color_bar_forecast, opacity=0.5, name="Forecast"), row=3, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor=color_bg, plot_bgcolor=color_bg,
                      font=dict(color="white", family="Segoe UI"), height=950,
                      hovermode="x unified", title="⚡ aFRR Forecast Dashboard — Bloomberg Style")
    fig.update_xaxes(showgrid=True, gridcolor=color_grid)
    fig.update_yaxes(showgrid=True, gridcolor=color_grid)
    return fig


# --------------------------------------------------------------------------------------
# DASHBOARD BLOOMBERG
# --------------------------------------------------------------------------------------
@app.route("/dashboard")
def dashboard_bloomberg():
    try:
        start = request.args.get("start")
        end = request.args.get("end")
        if not start or not end:
            end_dt = datetime.now(BRUSSELS)
            start_dt = end_dt - timedelta(days=7)
            start, end = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

        CSV_PATH = BASE_DIR / "forecast_log_full.csv"

        def load_forecasts(csv_path, start=None, end=None):
            df = pd.read_csv(csv_path, parse_dates=["forecast_time"])
            df = df.rename(columns={
                "forecast_value": "Forecast",
                "forecast_time":  "Datetime",
                "price_min":      "cap_BE",
                "price_max":      "cap_EU",
            })
            df["cap_BE"] = pd.to_numeric(df["cap_BE"], errors="coerce")
            df["cap_EU"] = pd.to_numeric(df["cap_EU"], errors="coerce")
            df = df.sort_values("forecast_generated_at").drop_duplicates(subset=["Datetime"], keep="last")
            if start:
                df = df[df["Datetime"] >= pd.to_datetime(start)]
            if end:
                df = df[df["Datetime"] < pd.Timestamp(end) + pd.Timedelta(days=1)]
            return df.reset_index(drop=True)

        end_ts_db = pd.Timestamp(end) + pd.Timedelta(days=1)
        forecasts_df=load_forecasts(CSV_PATH,start,end)
        actuals_df=fetch_actuals_combined(start, end_ts_db)
        merged=merge_and_flags(forecasts_df,actuals_df)
        metrics=compute_all_metrics(merged)
        fig=plot_bloomberg_dashboard_points(merged,metrics)
        return fig.to_html(full_html=True,include_plotlyjs="cdn")

    except Exception as e:
        return f"<h3>Erreur Dashboard Bloomberg : {e}</h3>"

# --------------------------------------------------------------------------------------
# EVALUATION DASHBOARD
# --------------------------------------------------------------------------------------
def _kpi_cards(kpis):
    cards = "".join(f"""
    <div style="background:#11182b;border:1px solid #1e2a3a;border-radius:10px;
                padding:16px 28px;text-align:center;min-width:150px">
      <div style="font-size:26px;font-weight:bold;color:{c}">{v}</div>
      <div style="font-size:12px;color:#aaa;margin-top:6px">{l}</div>
    </div>""" for l, v, c in kpis)
    return f'<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;margin:20px 0">{cards}</div>'


def _eval_page(start, end, tab, elia_kpi, elia_fig, rte_kpi, rte_fig):
    elia_disp  = "block" if tab == "elia" else "none"
    rte_disp   = "block" if tab == "rte"  else "none"
    elia_cls   = "tab-active" if tab == "elia" else ""
    rte_cls    = "tab-active" if tab == "rte"  else ""
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Evaluation Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body{{margin:0;padding:0;background:#0b0e16;color:white;font-family:'Segoe UI',Arial,sans-serif}}
  .hdr{{background:#0d1120;border-bottom:1px solid #1e2a3a;padding:14px 28px;
        display:flex;align-items:center;gap:20px;flex-wrap:wrap}}
  .hdr h1{{margin:0;font-size:18px;color:deepskyblue}}
  .tabs{{display:flex;gap:8px;padding:16px 28px 0}}
  .tab-btn{{background:#11182b;border:1px solid #1e2a3a;border-bottom:none;
            border-radius:8px 8px 0 0;padding:9px 26px;color:#aaa;
            cursor:pointer;font-size:13px;font-weight:bold;transition:all .15s}}
  .tab-btn.tab-active,.tab-btn:hover{{background:#0b0e16;color:white;border-color:#00bfff}}
  .tab-content{{padding:16px 28px 32px}}
  input[type=date]{{padding:6px 10px;border-radius:6px;border:1px solid #00bfff;
                   background:#11182b;color:white;font-size:13px}}
  .btn{{background:deepskyblue;border:none;border-radius:6px;padding:7px 16px;
        color:#000;font-weight:bold;cursor:pointer;font-size:13px}}
  .btn:hover{{background:#00a0e0}}
  hr{{border:none;border-top:1px solid #1e2a3a;margin:20px 0}}
</style></head><body>
<div class="hdr">
  <h1>⚡ Evaluation Dashboard</h1>
  <form method="get" action="/evaluation" style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
    <input type="hidden" name="tab" value="{tab}">
    <label style="font-size:13px;color:#aaa">Du</label>
    <input type="date" name="start" value="{start}">
    <label style="font-size:13px;color:#aaa">au</label>
    <input type="date" name="end" value="{end}">
    <button class="btn" type="submit">Appliquer</button>
  </form>
  <a href="/" style="color:#aaa;font-size:13px;margin-left:auto;text-decoration:none">← Accueil</a>
</div>
<div class="tabs">
  <button class="tab-btn {elia_cls}" onclick="switchTab('elia')">⚡ Elia BE</button>
  <button class="tab-btn {rte_cls}"  onclick="switchTab('rte')">🇫🇷 RTE FR</button>
</div>
<div id="tab-elia" class="tab-content" style="display:{elia_disp}">{elia_kpi}<hr>{elia_fig}</div>
<div id="tab-rte"  class="tab-content" style="display:{rte_disp}">{rte_kpi}<hr>{rte_fig}</div>
<script>
function switchTab(n){{
  document.getElementById('tab-elia').style.display=n==='elia'?'block':'none';
  document.getElementById('tab-rte').style.display=n==='rte'?'block':'none';
  document.querySelectorAll('.tab-btn').forEach(function(b,i){{
    b.classList.toggle('tab-active',(i===0&&n==='elia')||(i===1&&n==='rte'));
  }});
  var u=new URL(window.location);u.searchParams.set('tab',n);
  window.history.replaceState(null,'',u);
}}
</script></body></html>"""


@app.route("/evaluation")
def evaluation_dashboard():
    try:
        tab   = request.args.get("tab", "elia")
        start = request.args.get("start", "")
        end   = request.args.get("end", "")
        if not start or not end:
            now = datetime.now(BRUSSELS)
            end   = now.strftime("%Y-%m-%d")
            start = (now - timedelta(days=30)).strftime("%Y-%m-%d")

        BG   = "#0b0e16"
        GRID = "rgba(255,255,255,0.08)"
        LAY  = dict(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
                    font=dict(color="white", family="Segoe UI"), hovermode="x unified")
        start_ts = pd.Timestamp(start)
        end_ts   = pd.Timestamp(end) + pd.Timedelta(days=1)

        # ── ELIA ────────────────────────────────────────────────────────────
        elia_kpi = elia_fig = ""
        elia_path = BASE_DIR / "forecast_log_full.csv"
        if elia_path.exists():
            de = pd.read_csv(elia_path, parse_dates=["forecast_time"])
            de = de[(de["forecast_time"] >= start_ts) & (de["forecast_time"] < end_ts)]
            de = de.rename(columns={
                "forecast_value": "Forecast",
                "forecast_time":  "Datetime",
                "price_min":      "cap_BE",
                "price_max":      "cap_EU",
            })
            actuals_e = fetch_actuals_combined(start_ts, end_ts)
            merged_e  = merge_and_flags(de, actuals_e)
            metrics_e = compute_all_metrics(merged_e)

            def _fmt_pct(v):
                return f"{v:.1f}%" if v is not None else "N/A"

            elia_kpi = _kpi_cards([
                ("Forecast Accuracy",         _fmt_pct(metrics_e["Forecast accuracy"]),                      "deepskyblue"),
                ("Range Accuracy",            _fmt_pct(metrics_e["Range accuracy"]),                         "orange"),
                ("Range Acc. (fcst correct)", _fmt_pct(metrics_e["Range accuracy – when forecast correct"]), "lime"),
                ("Correlation",               str(metrics_e["Correlation"]) if metrics_e["Correlation"] is not None else "N/A", "#aaa"),
            ])

            fig_e = plot_bloomberg_dashboard_points(merged_e, metrics_e)
            elia_fig = fig_e.to_html(full_html=False, include_plotlyjs=False)
        else:
            elia_fig = "<p style='color:#999'>forecast_log_full.csv introuvable.</p>"

        # ── RTE ─────────────────────────────────────────────────────────────
        rte_kpi = rte_fig = ""
        rte_path = BASE_DIR.parent / "rte_forecaster" / "forecast_log_full.csv"
        if rte_path.exists():
            dr = pd.read_csv(rte_path, parse_dates=["timestamp"])
            dr = dr[(dr["timestamp"] >= start_ts) & (dr["timestamp"] < end_ts)].sort_values("timestamp")

            valid_r = dr.dropna(subset=["actual_mwh"])
            if len(valid_r):
                dir_acc_r = (np.sign(valid_r["forecast_mwh"]) == np.sign(valid_r["actual_mwh"])).mean() * 100
                mae_r     = (valid_r["forecast_mwh"] - valid_r["actual_mwh"]).abs().mean()
                cut30_r   = valid_r["timestamp"].max() - pd.Timedelta(days=30)
                rec30_r   = valid_r[valid_r["timestamp"] >= cut30_r]
                da30_r    = (np.sign(rec30_r["forecast_mwh"]) == np.sign(rec30_r["actual_mwh"])).mean() * 100 if len(rec30_r) else float("nan")
            else:
                dir_acc_r = mae_r = da30_r = float("nan")

            rte_kpi = _kpi_cards([
                ("Direction Accuracy", f"{dir_acc_r:.1f}%" if not np.isnan(dir_acc_r) else "N/A", "deepskyblue"),
                ("MAE",                f"{mae_r:.1f} MWh"  if not np.isnan(mae_r)     else "N/A", "orange"),
                ("Dir. Acc. 30j",      f"{da30_r:.1f}%"    if not np.isnan(da30_r)    else "N/A", "lime"),
                ("QHs valides",        f"{len(valid_r):,}", "#aaa"),
            ])

            fr = go.Figure()
            fr.add_trace(go.Scatter(x=dr["timestamp"], y=dr["forecast_mwh"],
                                     mode="lines", name="Forecast",
                                     line=dict(color="deepskyblue", width=1)))
            if not valid_r.empty:
                fr.add_trace(go.Scatter(x=valid_r["timestamp"], y=valid_r["actual_mwh"],
                                         mode="lines", name="Réel",
                                         line=dict(color="white", width=1)))
            fr.update_layout(**LAY, height=500,
                              title="🇫🇷 RTE FR — Évaluation du modèle")
            fr.update_xaxes(showgrid=True, gridcolor=GRID)
            fr.update_yaxes(showgrid=True, gridcolor=GRID)
            rte_fig = fr.to_html(full_html=False, include_plotlyjs=False)
        else:
            rte_fig = "<p style='color:#999'>forecast_log_full.csv introuvable.</p>"

        return _eval_page(start=start, end=end, tab=tab,
                          elia_kpi=elia_kpi, elia_fig=elia_fig,
                          rte_kpi=rte_kpi,   rte_fig=rte_fig)

    except Exception as e:
        import traceback
        return (f"<pre style='color:red;background:#0b0e16;padding:20px'>"
                f"{traceback.format_exc()}</pre>")

# --------------------------------------------------------------------------------------
# API ELIA
# --------------------------------------------------------------------------------------
ACTUAL_API_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods169/records"
data_lock = threading.Lock()

try:
    historical_data = pd.read_csv("historical_imbalance_data.csv")
    historical_data["datetime"] = pd.to_datetime(historical_data["datetime"]).dt.tz_localize(None)
    historical_data = historical_data.set_index("datetime")
except Exception:
    historical_data = pd.DataFrame(columns=["actual_system_imbalance"])
    historical_data.index.name = "datetime"

def fetch_and_process_data(ws_bxl, we_bxl):
    ws_utc = ws_bxl.astimezone(pytz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    we_utc = we_bxl.astimezone(pytz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {"where": f"datetime >= '{ws_utc}' and datetime <= '{we_utc}'", "limit": 100, "order_by": "datetime asc"}
    try:
        r = requests.get(ACTUAL_API_URL, params=params, timeout=30)
        r.raise_for_status()
        res = r.json().get("results", [])
        if not res:
            return None
        df = pd.DataFrame(res)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(BRUSSELS).dt.tz_localize(None)
        df["actual_system_imbalance"] = pd.to_numeric(df["systemimbalance"], errors="coerce")
        return df[["datetime", "actual_system_imbalance"]].set_index("datetime").sort_index()
    except Exception as e:
        print(f"[ELIA][ERROR] {e}")
        return None

# --------------------------------------------------------------------------------------
# MERIT ORDER
# --------------------------------------------------------------------------------------
def build_merit_orders_silent():
    import concurrent.futures
    try:
        from afrr_merit_order import load_and_build
    except Exception as e:
        print(f"  [WARN] Erreur import afrr_merit_order: {e}", flush=True)
        return None, None

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(load_and_build)
            try:
                path_all, path_de = future.result(timeout=120)
                return path_all, path_de
            except concurrent.futures.TimeoutError:
                print("  [WARN] build_merit_orders timeout (120s), skip.", flush=True)
                return None, None
            except Exception as e:
                print(f"  [WARN] Erreur construction merit orders: {e}", flush=True)
                return None, None
    except Exception as e:
        print(f"  [WARN] Erreur build_merit_orders: {e}", flush=True)
        return None, None

# --------------------------------------------------------------------------------------
# VoAA - ✅ DST CORRIGÉ
# --------------------------------------------------------------------------------------
def fetch_voaa_value(dataset: str, ts_qh_naive_bxl: pd.Timestamp, direction: str) -> float | None:
    BASE_URL = "https://external-elia.opendatasoft.com/api/records/1.0/search/"
    
    # ✅ CORRECTION DST : Gestion des heures ambiguës
    try:
        ts_utc = ts_qh_naive_bxl.tz_localize(BRUSSELS).tz_convert(pytz.UTC)
    except pytz.exceptions.AmbiguousTimeError:
        ts_utc = ts_qh_naive_bxl.tz_localize(BRUSSELS, ambiguous=False).tz_convert(pytz.UTC)
    except pytz.exceptions.NonExistentTimeError:
        ts_utc = (ts_qh_naive_bxl + pd.Timedelta(hours=1)).tz_localize(BRUSSELS).tz_convert(pytz.UTC)
    
    print(f"  [INFO] VoAA {direction} (dataset {dataset}):")
    print(f"      Input Brussels: {ts_qh_naive_bxl.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"      Converti UTC:   {ts_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    
    ws = (ts_utc - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    we = (ts_utc + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    print(f"      Fenêtre API:    [{ws} TO {we}]")

    params = {
        "dataset": dataset,
        "q": f"datetime:[{ws} TO {we}]",
        "rows": 10000,
        "sort": "datetime",
        "timezone": "UTC",
    }

    try:
        r = requests.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get("records", [])
        
        print(f"      Records trouvés: {len(data)}")
        
        if not data:
            print(f"      [WARN] Aucune donnée API")
            return None
        
        df = pd.json_normalize([d["fields"] for d in data])
        if "energybidmarginalprice" not in df.columns:
            print(f"      [WARN] Colonne 'energybidmarginalprice' manquante")
            return None
        
        df["energybidmarginalprice"] = pd.to_numeric(df["energybidmarginalprice"], errors="coerce")
        df = df.dropna(subset=["energybidmarginalprice"])
        
        if "available" in df.columns:
            df = df[df["available"] == True]
        if "producttype" in df.columns:
            df = df[df["producttype"].isin(["aFRR", "mFRR"])]
        
        if df.empty:
            print(f"      [WARN] Aucun bid valide après filtrage")
            return None
        
        value = df["energybidmarginalprice"].min() if direction == "INC" else df["energybidmarginalprice"].max()
        
        print(f"      [OK] Valeur calculee: {value:.2f} EUR/MWh ({len(df)} bids)")
        
        return float(value)
        
    except Exception as e:
        print(f"      [WARN] Erreur: {e}")
        return None

# --------------------------------------------------------------------------------------
# mFRR - ✅ DST CORRIGÉ
# --------------------------------------------------------------------------------------
def fetch_mfrr_be_price_at_800mw(ts_qh_naive_bxl: pd.Timestamp, direction: str) -> float | None:
    BASE_URL = "https://external-elia.opendatasoft.com/api/records/1.0/search/"
    dataset = "ods163" if direction == "INC" else "ods164"
    
    # ✅ CORRECTION DST : Gestion des heures ambiguës
    try:
        ts_utc = ts_qh_naive_bxl.tz_localize(BRUSSELS).tz_convert(pytz.UTC)
    except pytz.exceptions.AmbiguousTimeError:
        ts_utc = ts_qh_naive_bxl.tz_localize(BRUSSELS, ambiguous=False).tz_convert(pytz.UTC)
    except pytz.exceptions.NonExistentTimeError:
        ts_utc = (ts_qh_naive_bxl + pd.Timedelta(hours=1)).tz_localize(BRUSSELS).tz_convert(pytz.UTC)
    
    ws = (ts_utc - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    we = (ts_utc + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    params = {
        "dataset": dataset,
        "q": f"datetime:[{ws} TO {we}]",
        "rows": 10000,
        "sort": "datetime",
        "timezone": "UTC",
    }
    
    try:
        r = requests.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get("records", [])
        
        if not data:
            print(f"  [INFO] mFRR {direction}: Aucune donnee API")
            return None
        
        df = pd.json_normalize([d["fields"] for d in data])
        print(f"  [INFO] mFRR {direction}: {len(df)} lignes recuperees")
        
        product_col = None
        for col_name in ["product", "Product", "balancingproduct"]:
            if col_name in df.columns:
                product_col = col_name
                break
        
        if product_col is None:
            print(f"  [WARN] Colonnes disponibles: {list(df.columns)}")
            return None
        
        product_types = df[product_col].unique()
        print(f"  [INFO] Types produits ({product_col}): {product_types}")
        
        df_mfrr = df[df[product_col] == "mFRR"]
        print(f"  [INFO] mFRR {direction}: {len(df_mfrr)} lignes apres filtre")
        
        if df_mfrr.empty:
            print(f"  [WARN] Aucun bid mFRR trouve")
            return None
        
        price_col = None
        volume_col = None
        
        for col in ["energybidmarginalprice", "bid_price", "bidprice"]:
            if col in df_mfrr.columns:
                price_col = col
                break
        
        for col in ["energybidvolume", "bid_volume", "bidvolume"]:
            if col in df_mfrr.columns:
                volume_col = col
                break
        
        if price_col is None or volume_col is None:
            print(f"  [WARN] Colonnes prix/volume introuvables")
            print(f"  [INFO] Colonnes: {list(df_mfrr.columns)}")
            return None
        
        df_mfrr[price_col] = pd.to_numeric(df_mfrr[price_col], errors="coerce")
        df_mfrr[volume_col] = pd.to_numeric(df_mfrr[volume_col], errors="coerce")
        df_mfrr = df_mfrr.dropna(subset=[price_col, volume_col])
        
        if df_mfrr.empty:
            print(f"  [WARN] Aucun bid valide")
            return None
        
        if direction == "INC":
            df_mfrr = df_mfrr.sort_values(price_col, ascending=True)
        else:
            df_mfrr = df_mfrr.sort_values(price_col, ascending=False)
        
        df_mfrr["cumvol"] = df_mfrr[volume_col].cumsum()
        max_vol = df_mfrr["cumvol"].max()
        print(f"  [INFO] Volume total mFRR: {max_vol:.1f} MW")
        
        subset = df_mfrr[df_mfrr["cumvol"] <= 800]
        if subset.empty:
            print(f"  [WARN] Volume < 800 MW, fallback max")
            price = float(df_mfrr[price_col].iloc[-1])
            return price
        
        price = float(subset[price_col].iloc[-1])
        print(f"  [OK] Prix mFRR @ 800 MW: {price:.2f} EUR/MWh")
        return price
        
    except Exception as e:
        print(f"  [WARN] Erreur mFRR: {e}")
        return None

# --------------------------------------------------------------------------------------
# DÉTECTION BALANCED+
# --------------------------------------------------------------------------------------
def detect_balanced_plus(df_hist: pd.DataFrame, forecast: float):
    try:
        si = df_hist["actual_system_imbalance"].dropna().copy()
        
        if len(si) < 60:
            return 0, False
        
        if not isinstance(si.index, pd.DatetimeIndex):
            si.index = pd.to_datetime(si.index)
        
        si_qh = si.resample('15min').mean().dropna()
        
        if len(si_qh) < 2:
            return 0, False
        
        current_qh = si_qh.iloc[-1]
        prev_qh = si_qh.iloc[-2] if len(si_qh) >= 2 else current_qh
        
        if len(si_qh) >= 96 * 7:
            avg_7d = si_qh.rolling(window=96*7, min_periods=96).mean().iloc[-1]
        else:
            avg_7d = si_qh.mean()
        
        delta_avg = abs(current_qh - avg_7d)
        cond1_delta_avg = delta_avg > 180
        
        if len(si_qh) >= 2:
            last_2qh = si_qh.iloc[-2:]
            direction_current = np.sign(current_qh)
            same_direction = all(np.sign(last_2qh) == direction_current)
            cond2_persistence = same_direction
        else:
            cond2_persistence = False
        
        delta_qh = abs(current_qh - prev_qh)
        cond3_brutal_change = delta_qh > 200
        
        cond4_high_ace = abs(current_qh) > 100
        
        cond_amplitude = abs(current_qh) > 240
        
        conditions = [cond1_delta_avg, cond2_persistence, cond3_brutal_change, cond4_high_ace]
        score = sum(conditions)
        
        triggered = (score >= 3) or (cond_amplitude and score >= 2)
        
        return score, triggered
        
    except Exception as e:
        print(f"  [WARN] Erreur Balanced+: {e}")
        return 0, False

# --------------------------------------------------------------------------------------
# PRICE RANGE - ✅ MODIFIÉ : INVERSION VoAA/DIRECTION
# --------------------------------------------------------------------------------------
def compute_price_range(ts_qh_bxl, forecast_value, pred_balanced_plus=0):
    ts_key = floor_qh_naive_bxl(ts_qh_bxl)

    voaa_inc = fetch_voaa_value("ods163", ts_key, "INC")
    voaa_dec = fetch_voaa_value("ods164", ts_key, "DEC")

    if voaa_inc is None and voaa_dec is None:
        print(f"  [WARN] VoAA indisponible pour {ts_key}")
        return None, None, None, None, None, None

    if forecast_value < 0:
        direction = "INC"
    else:
        direction = "DEC"

    merit_price = None
    merit_source = None
    
    if pred_balanced_plus:
        merit_price = fetch_mfrr_be_price_at_800mw(ts_key, direction)
        merit_source = f"mFRR 800 {direction}"
    else:
        try:
            from afrr_merit_order import load_and_build
            _, path_de = load_and_build()
            df_merit = pd.read_excel(path_de)
            
            row = df_merit.loc[df_merit["Timestamp"] == ts_key]
            
            if not row.empty:
                if direction == "INC":
                    cols_inc = [c for c in df_merit.columns if "Inc_" in c]
                    if cols_inc:
                        merit_price = float(row[cols_inc[-1]].values[0])
                else:
                    cols_dec = [c for c in df_merit.columns if "Dec_" in c]
                    if cols_dec:
                        merit_price = float(row[cols_dec[-1]].values[0])
                merit_source = f"Merit Order EU {direction}"
        except Exception as e:
            print(f"  [WARN] Merit Order EU indisponible: {e}")

    # ✅ MODIFICATION : Inversion VoAA selon direction
    if direction == "INC":
        # Direction INC → Range = [VoAA Dec, Merit Order EU INC]
        price_min = voaa_dec
        price_max = merit_price
    else:
        # Direction DEC → Range = [VoAA Inc, Merit Order EU DEC]
        price_min = voaa_inc
        price_max = merit_price

    return price_min, price_max, direction, voaa_inc, voaa_dec, merit_source

# --------------------------------------------------------------------------------------
# NETTOYAGE CSV
# --------------------------------------------------------------------------------------
def fix_existing_csv_timezone():
    try:
        if os.path.exists("historical_imbalance_data.csv"):
            df = pd.read_csv("historical_imbalance_data.csv")
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
            df.to_csv("historical_imbalance_data.csv", index=False)
    except Exception:
        pass

# --------------------------------------------------------------------------------------
# BOUCLE CONTINUE
# --------------------------------------------------------------------------------------
def continuous_data_collection_loop():
    global historical_data
    
    while True:
        now_bxl = datetime.now(BRUSSELS)
        current_qh = now_bxl.replace(minute=(now_bxl.minute // 15) * 15, second=0, microsecond=0)
        next_qh = current_qh + timedelta(minutes=15)
        
        merit_order_time = next_qh - timedelta(minutes=5)
        wait_until(merit_order_time)
        
        print_separator()
        print(f"[TIME] {datetime.now(BRUSSELS).strftime('%H:%M:%S')} | Target QH: {next_qh.strftime('%H:%M')}", flush=True)
        print_section("CONSTRUCTION MERIT ORDERS")

        build_merit_orders_silent()
        print("  [OK] Piles EU construites", flush=True)
        
        forecast_time = next_qh - timedelta(seconds=30)
        wait_until(forecast_time)
        
        print_section("FORECAST & PRICE RANGE")
        
        window_end = datetime.now(BRUSSELS)
        window_start = window_end - timedelta(minutes=75)
        
        print(f"  [INFO] Fenetre de recuperation: [{window_start.strftime('%H:%M:%S')} -> {window_end.strftime('%H:%M:%S')}]")
        
        new_data = fetch_and_process_data(window_start, window_end)
        forecast = np.nan

        with data_lock:
            if new_data is not None and not new_data.empty:
                try:
                    historical_data = pd.read_csv("historical_imbalance_data.csv")
                    historical_data["datetime"] = pd.to_datetime(historical_data["datetime"]).dt.tz_localize(None)
                    historical_data = historical_data.set_index("datetime")
                except Exception:
                    historical_data = pd.DataFrame(columns=["actual_system_imbalance"])
                    historical_data.index.name = "datetime"

                historical_data = pd.concat([historical_data, new_data])
                historical_data = historical_data[~historical_data.index.duplicated(keep="last")].sort_index()
                historical_data.to_csv("historical_imbalance_data.csv")

                try:
                    next_qh_naive = as_naive_bxl(next_qh)
                    features = extract_features(historical_data, next_qh_naive)
                    if features is not None:
                        features = features.reshape(1, -1)
                        forecast = forecaster.predict(features)[0]
                    else:
                        forecast = np.nan
                        print("  [WARN] Features indisponibles")
                except Exception as e:
                    print(f"  [WARN] Erreur forecast: {e}")
                    forecast = np.nan

        forecast_generated_at = datetime.now(BRUSSELS)
        
        last_imbalance = None
        last_imbalance_time = None
        if not historical_data.empty:
            last_imbalance = historical_data["actual_system_imbalance"].iloc[-1]
            last_imbalance_time = historical_data.index[-1]
        
        score, triggered = detect_balanced_plus(historical_data, forecast)
        
        price_min, price_max, direction, voaa_inc, voaa_dec, merit_source = compute_price_range(
            next_qh, forecast, 1 if triggered else 0
        )
        
        print(f"\n  [TIME] Forecast genere a:  {forecast_generated_at.strftime('%H:%M:%S')}")
        if last_imbalance is not None and last_imbalance_time is not None:
            print(f"  [INFO] Derniere imbalance: {last_imbalance:>7.1f} MW (a {last_imbalance_time.strftime('%H:%M:%S')})")
        print(f"\n  Forecast:       {forecast:>8.1f} MW")
        print(f"  Balanced+:      {'TRIGGERED' if triggered else 'Normal'} (score: {score}/4)")
        print(f"\n  VoAA Inc:       {f'{voaa_inc:.2f}' if voaa_inc else 'N/A':>8} EUR/MWh")
        print(f"  VoAA Dec:       {f'{voaa_dec:.2f}' if voaa_dec else 'N/A':>8} EUR/MWh")
        
        if merit_source and price_min is not None and price_max is not None:
            merit_display_price = price_max if direction == 'INC' else price_min
            print(f"  {merit_source}:  {merit_display_price:>8.2f} EUR/MWh")
        
        print(f"\n  Direction:      {direction}")
        
        if price_min is not None and price_max is not None:
            print(f"  Price Range:    [{price_min:.2f} - {price_max:.2f}] EUR/MWh")
        else:
            print("  Price Range:    N/A")
        
        pd.DataFrame({
            "forecast_time": [next_qh.replace(tzinfo=None)],
            "forecast_value": [forecast],
            "forecast_generated_at": [forecast_generated_at.replace(tzinfo=None)],
            "price_direction": [direction],
            "price_min": [price_min],
            "price_max": [price_max],
            "voaa_inc": [voaa_inc],
            "voaa_dec": [voaa_dec],
            "balanced_plus": [triggered]
        }).to_csv("forecastV3.csv", mode="a", header=not os.path.exists("forecastV3.csv"), index=False)
        
        print(f"\n  [OK] Forecast enregistre dans forecastV3.csv")
        print("="*80 + "\n", flush=True)

        wait_until(next_qh + timedelta(minutes=1))

# --------------------------------------------------------------------------------------
# RTE FR DASHBOARD
# --------------------------------------------------------------------------------------
RTE_LOG_CSV  = BASE_DIR.parent / "rte_forecaster" / "forecast_log_full.csv"
RTE_DIR_PATH = BASE_DIR.parent / "rte_forecaster"

_rte_token      = None
_rte_token_time = None

def _rte_fresh_token():
    global _rte_token, _rte_token_time
    import base64
    RTE_CLIENT_ID     = "bdc03388-6c93-46f6-adbf-1a77d5b89684"
    RTE_CLIENT_SECRET = "da352352-63f8-42ce-9a34-0624f7560a72"
    if (_rte_token is None or _rte_token_time is None or
            (datetime.now() - _rte_token_time).total_seconds() > 3000):
        basic = base64.b64encode(
            (RTE_CLIENT_ID + ":" + RTE_CLIENT_SECRET).encode()).decode()
        r = requests.post(
            "https://digital.iservices.rte-france.com/token/oauth/",
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "Authorization": "Basic " + basic},
            data={"grant_type": "client_credentials"}, timeout=30)
        r.raise_for_status()
        _rte_token      = r.json()["access_token"]
        _rte_token_time = datetime.now()
    return _rte_token


def _rte_download_actuals(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    import io as _io
    tok = _rte_fresh_token()
    url = "https://digital.iservices.rte-france.com/open_api/balancing_energy/v4/imbalance_data"
    params = {
        "start_date": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_date":   end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "resolution": "PT15M",
    }
    r = requests.get(url, params=params,
                     headers={"Authorization": "Bearer " + tok, "Accept": "text/csv"},
                     timeout=60)
    r.raise_for_status()
    lines = r.text.splitlines()
    idx = next((i for i, l in enumerate(lines) if l.startswith("Heure de")), None)
    if idx is None:
        return pd.DataFrame()
    df = pd.read_csv(_io.StringIO("\n".join(lines[idx:])), sep=";")
    df.columns = ["start_time", "end_time", "imbalance_mwh", "trend", "price_pos", "price_neg"]
    df["start_time"]    = pd.to_datetime(df["start_time"], format="%d/%m/%Y %H:%M")
    df["imbalance_mwh"] = pd.to_numeric(df["imbalance_mwh"], errors="coerce")
    return df.set_index("start_time")[["imbalance_mwh"]]


@app.route("/dashboard_rte")
def dashboard_rte():
    try:
        start = request.args.get("start")
        end   = request.args.get("end")
        if not start or not end:
            now   = datetime.now(BRUSSELS)
            end   = now.strftime("%Y-%m-%d")
            start = (now - timedelta(days=30)).strftime("%Y-%m-%d")

        start_ts = pd.Timestamp(start)
        end_ts   = pd.Timestamp(end) + pd.Timedelta(days=1)

        df = pd.read_csv(RTE_LOG_CSV, parse_dates=["timestamp"])
        df = (df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)]
                .sort_values("timestamp")
                .reset_index(drop=True))

        # Fill recent missing actual_mwh from RTE API (last 7 days only to stay fast)
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=7)
        need_actual = df[(df["actual_mwh"].isna()) & (df["timestamp"] >= cutoff)]
        if not need_actual.empty:
            try:
                api_start = need_actual["timestamp"].min().to_pydatetime()
                api_end   = (need_actual["timestamp"].max() + pd.Timedelta(minutes=15)).to_pydatetime()
                fetched   = _rte_download_actuals(api_start, api_end)
                if not fetched.empty:
                    df = df.set_index("timestamp")
                    df.loc[df["actual_mwh"].isna() & df.index.isin(fetched.index),
                           "actual_mwh"] = fetched["imbalance_mwh"]
                    df = df.reset_index()
            except Exception as e:
                print(f"[RTE][WARN] actuals fetch failed: {e}")

        valid = df.dropna(subset=["actual_mwh"])

        # KPIs
        if len(valid):
            dir_acc = (np.sign(valid["forecast_mwh"]) == np.sign(valid["actual_mwh"])).mean() * 100
            mae     = (valid["forecast_mwh"] - valid["actual_mwh"]).abs().mean()
            corr    = round(float(valid["forecast_mwh"].corr(valid["actual_mwh"])), 3)
            cut30   = valid["timestamp"].max() - pd.Timedelta(days=30)
            rec30   = valid[valid["timestamp"] >= cut30]
            da30    = (np.sign(rec30["forecast_mwh"]) == np.sign(rec30["actual_mwh"])).mean() * 100 if len(rec30) else float("nan")
        else:
            dir_acc = mae = corr = da30 = float("nan")

        kpis = {
            "Direction Accuracy":        f"{dir_acc:.1f}%" if not np.isnan(dir_acc) else "N/A",
            "MAE":                        f"{mae:.1f} MWh"  if not np.isnan(mae)     else "N/A",
            "Rolling 30d Dir. Accuracy":  f"{da30:.1f}%"    if not np.isnan(da30)    else "N/A",
            "Correlation":                str(corr)          if not np.isnan(corr)    else "N/A",
        }

        color_bg   = "#0b0e16"
        color_grid = "rgba(255,255,255,0.08)"

        fig = sp.make_subplots(
            rows=2, cols=1,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.12,
            subplot_titles=["🇫🇷 RTE FR — Forecast vs Actual", None],
        )

        # Row 1: forecast bars + actual line
        fig.add_trace(go.Bar(
            x=df["timestamp"], y=df["forecast_mwh"],
            marker_color="deepskyblue", opacity=0.6, name="Forecast MWh",
        ), row=1, col=1)
        if not valid.empty:
            fig.add_trace(go.Scatter(
                x=valid["timestamp"], y=valid["actual_mwh"],
                mode="lines", line=dict(color="white", width=1.5), name="Actual MWh",
            ), row=1, col=1)

        # Row 2: KPI annotations
        positions = [0.12, 0.38, 0.65, 0.88]
        for i, (title, val) in enumerate(kpis.items()):
            fig.add_annotation(
                xref="paper", yref="paper",
                x=positions[i], y=0.08,
                text=(f"<b style='font-size:22px;color:white'>{val}</b>"
                      f"<br><span style='font-size:13px;color:#cccccc'>{title}</span>"),
                showarrow=False, xanchor="center", align="center",
            )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=color_bg, plot_bgcolor=color_bg,
            font=dict(color="white", family="Segoe UI"),
            height=800,
            hovermode="x unified",
            title=f"⚡ RTE FR Forecast Dashboard — {start} → {end}",
            bargap=0.1,
        )
        fig.update_xaxes(showgrid=True, gridcolor=color_grid)
        fig.update_yaxes(showgrid=True, gridcolor=color_grid, title_text="MWh", row=1, col=1)

        return fig.to_html(full_html=True, include_plotlyjs="cdn")

    except Exception as e:
        import traceback
        return (f"<pre style='color:red;background:#0b0e16;padding:20px'>"
                f"{traceback.format_exc()}</pre>")


# --------------------------------------------------------------------------------------
# VE 2026 DASHBOARD (BE + FR)
# --------------------------------------------------------------------------------------
REPO_ROOT   = BASE_DIR.parent.parent
REPORTS_DIR = REPO_ROOT / "outputs" / "reports"

@app.route("/ve_2026")
def ve_2026():
    import base64

    def _img_b64(filename):
        p = REPORTS_DIR / filename
        if not p.exists():
            return None
        return base64.b64encode(p.read_bytes()).decode()

    be_b64 = _img_b64("ve_be_2026.png")
    fr_b64 = _img_b64("ve_fr_2026.png")

    cards = []
    for title, cost, b64 in [
        ("VE Belgique 2026", "+86.86 EUR", be_b64),
        ("VE France 2026",  "-11.18 EUR", fr_b64),
    ]:
        color = "#f87171" if cost.startswith("+") else "#4ade80"
        img_tag = (f'<img src="data:image/png;base64,{b64}" '
                   f'style="width:100%;border-radius:8px">'
                   if b64 else '<p style="color:#aaa">Image non trouvee</p>')
        cards.append(f"""
        <div style="background:#12122a;border:1px solid #2a2a4a;
                    border-radius:12px;padding:20px;margin-bottom:32px">
          <div style="display:flex;justify-content:space-between;
                      align-items:center;margin-bottom:14px">
            <h2 style="margin:0;color:#e0e0e0;font-size:1.2rem">{title}</h2>
            <span style="color:{color};font-size:1.6rem;
                         font-weight:700;letter-spacing:1px">{cost}</span>
          </div>
          {img_tag}
        </div>""")

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>VE 2026 Dashboard</title>
  <style>
    body{{margin:0;padding:24px;background:#0b0e16;
         font-family:'Segoe UI',sans-serif;color:#e0e0e0}}
    h1{{text-align:center;color:#00d4ff;font-size:1.5rem;margin-bottom:8px}}
    .sub{{text-align:center;color:#666;font-size:.85rem;margin-bottom:32px}}
    .grid{{max-width:1400px;margin:0 auto}}
    a{{color:#00d4ff;font-size:.85rem}}
  </style>
</head>
<body>
  <h1>VE 2026 — Dashboard Charge Intelligente</h1>
  <p class="sub">
    S_BE_opt &nbsp;|&nbsp; forecast_volume &gt; 200 AND afrr &gt; 50% AND mfrr &gt; 50%
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="/">Elia Dashboard</a> &nbsp;|&nbsp;
    <a href="/dashboard_rte">RTE Dashboard</a>
  </p>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>"""
    return html


# --------------------------------------------------------------------------------------
# REPORTS DASHBOARD (VE + Solar 2026 — Plotly dark Bloomberg)
# --------------------------------------------------------------------------------------
@app.route("/reports")
def reports_dashboard():
    try:
        BG   = "#0b0e16"
        GRID = "rgba(255,255,255,0.08)"
        _LAY = dict(
            template="plotly_dark",
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(color="white", family="Segoe UI"),
            hovermode="x unified",
        )
        C_VE  = {"S_BE_OPT": "#00e5ff", "S1": "#ff9800", "S2": "#ab47bc"}
        L_VE  = {"S_BE_OPT": "S_BE_opt", "S1": "S1 Prudent", "S2": "S2 DA"}
        C_SOL = {"s1": "#ff9800", "s2": "#ab47bc", "s3": "#00e5ff", "s4": "#4ade80"}

        # ── VE ─────────────────────────────────────────────────────────────────
        def _ve_fig(csv_path, label):
            df = pd.read_csv(csv_path, parse_dates=["timestamp", "date"])
            df["date"] = pd.to_datetime(df["date"])
            strats = [s for s in ["S_BE_OPT", "S1", "S2"] if s in df["strategy"].unique()]
            daily  = (df.groupby(["strategy", "date"])["cost_eur"]
                        .sum().reset_index().sort_values("date"))
            smart  = df[(df["smart_kwh"] > 0) & (df["strategy"] == "S_BE_OPT")].copy()
            total  = df[df["strategy"] == "S_BE_OPT"]["cost_eur"].sum()
            sign   = "+" if total >= 0 else ""
            kpi    = f"{sign}{total:.2f} €"

            fig = sp.make_subplots(
                rows=2, cols=1,
                subplot_titles=["Coûts cumulatifs par stratégie",
                                "Charge smart S_BE_opt — ISP (€/MWh)"],
                vertical_spacing=0.13, row_heights=[0.55, 0.45],
            )
            for s in strats:
                sub = daily[daily["strategy"] == s].set_index("date")["cost_eur"]
                cum = sub.cumsum().reset_index()
                fig.add_trace(go.Scatter(
                    x=cum["date"], y=cum["cost_eur"], mode="lines",
                    line=dict(color=C_VE.get(s, "#fff"), width=2.2),
                    name=L_VE.get(s, s),
                ), row=1, col=1)

            if not smart.empty:
                kwh_rng = smart["smart_kwh"].max() - smart["smart_kwh"].min()
                sizes   = (6 + 16 * (smart["smart_kwh"] - smart["smart_kwh"].min())
                           / (kwh_rng + 1e-9))
                neg_m = smart["isp"] < 0
                for mask, color, sym, name in [
                    (neg_m,  "#00c853", "circle",      "ISP < 0 (gain)"),
                    (~neg_m, "#ff1744", "x-thin-open", "ISP ≥ 0 (coût)"),
                ]:
                    sub2 = smart[mask]
                    if not sub2.empty:
                        fig.add_trace(go.Scatter(
                            x=sub2["date"], y=sub2["isp"], mode="markers",
                            marker=dict(color=color, size=sizes[mask].values,
                                        symbol=sym, opacity=0.8),
                            name=name,
                        ), row=2, col=1)

            fig.update_layout(**_LAY, height=720, title=f"{label}  —  S_BE_opt : {kpi}")
            fig.add_hline(y=0, line_color="#333", line_width=0.8, row=1, col=1)
            fig.add_hline(y=0, line_color="#444", line_width=0.8, row=2, col=1)
            fig.update_xaxes(showgrid=True, gridcolor=GRID, tickformat="%b")
            fig.update_yaxes(showgrid=True, gridcolor=GRID)
            fig.update_yaxes(title_text="Coût cumulatif (€)", row=1, col=1)
            fig.update_yaxes(title_text="ISP (€/MWh)", row=2, col=1)
            return fig, kpi, total >= 0

        # ── Solar BE ───────────────────────────────────────────────────────────
        def _solar_be_fig(csv_path, label):
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            df["date"] = df["timestamp"].dt.floor("D")
            daily = df.groupby("date")[
                ["s1_total", "s2_total", "s3_total", "s4_total"]
            ].sum().reset_index()
            total = daily["s3_total"].sum()
            sign  = "+" if total >= 0 else ""
            kpi   = f"{sign}{total:,.0f} €"

            fig = sp.make_subplots(
                rows=2, cols=1,
                subplot_titles=["Revenus cumulatifs par stratégie",
                                "ISP (€/MWh) — QH avec production"],
                vertical_spacing=0.13, row_heights=[0.55, 0.45],
            )
            for s, lbl in [("s1", "S1"), ("s2", "S2"), ("s3", "S3"), ("s4", "S4")]:
                cum = daily[f"{s}_total"].cumsum()
                fig.add_trace(go.Scatter(
                    x=daily["date"], y=cum, mode="lines",
                    line=dict(color=C_SOL[s], width=2.2), name=lbl,
                ), row=1, col=1)

            if "production_mw" in df.columns and "isp" in df.columns:
                prod = df[(df["production_mw"] > 0) & df["isp"].notna()].copy()
                if not prod.empty:
                    p_rng = prod["production_mw"].max() - prod["production_mw"].min()
                    sizes = (4 + 12 * (prod["production_mw"] - prod["production_mw"].min())
                             / (p_rng + 1e-9))
                    neg_m = prod["isp"] < 0
                    for mask, color, name in [
                        (neg_m,  "#00c853", "ISP < 0"),
                        (~neg_m, "#ff9800", "ISP ≥ 0"),
                    ]:
                        sub = prod[mask]
                        if not sub.empty:
                            fig.add_trace(go.Scatter(
                                x=sub["date"], y=sub["isp"], mode="markers",
                                marker=dict(color=color, size=sizes[mask].values, opacity=0.5),
                                name=name,
                            ), row=2, col=1)

            fig.update_layout(**_LAY, height=720, title=f"{label}  —  S3 : {kpi}")
            fig.add_hline(y=0, line_color="#333", line_width=0.8, row=1, col=1)
            fig.add_hline(y=0, line_color="#444", line_width=0.8, row=2, col=1)
            fig.update_xaxes(showgrid=True, gridcolor=GRID, tickformat="%b")
            fig.update_yaxes(showgrid=True, gridcolor=GRID)
            fig.update_yaxes(title_text="Revenu cumulatif (€)", row=1, col=1)
            fig.update_yaxes(title_text="ISP (€/MWh)", row=2, col=1)
            return fig, kpi, total >= 0

        # ── Solar FR ───────────────────────────────────────────────────────────
        def _solar_fr_fig(csv_path, label):
            df = pd.read_csv(csv_path, parse_dates=["date"])
            df = df.sort_values("date")
            total = df["s3_total"].sum()
            sign  = "+" if total >= 0 else ""
            kpi   = f"{sign}{total:,.0f} €"

            fig = sp.make_subplots(
                rows=2, cols=1,
                subplot_titles=["Revenus cumulatifs par stratégie",
                                "S3 — DA vs Déséquilibre (journalier)"],
                vertical_spacing=0.13, row_heights=[0.55, 0.45],
            )
            for s, lbl in [("s1", "S1"), ("s2", "S2"), ("s3", "S3")]:
                cum = df[f"{s}_total"].cumsum()
                fig.add_trace(go.Scatter(
                    x=df["date"], y=cum, mode="lines",
                    line=dict(color=C_SOL[s], width=2.2), name=lbl,
                ), row=1, col=1)

            fig.add_trace(go.Bar(
                x=df["date"], y=df["s3_da"],
                name="S3 DA", marker_color="deepskyblue", opacity=0.7,
            ), row=2, col=1)
            fig.add_trace(go.Bar(
                x=df["date"], y=df["s3_imb"],
                name="S3 Imbalance", marker_color="#ff9800", opacity=0.7,
            ), row=2, col=1)

            fig.update_layout(**_LAY, height=720, barmode="relative",
                              title=f"{label}  —  S3 : {kpi}")
            fig.add_hline(y=0, line_color="#333", line_width=0.8, row=1, col=1)
            fig.add_hline(y=0, line_color="#444", line_width=0.8, row=2, col=1)
            fig.update_xaxes(showgrid=True, gridcolor=GRID, tickformat="%b")
            fig.update_yaxes(showgrid=True, gridcolor=GRID)
            fig.update_yaxes(title_text="Revenu cumulatif (€)", row=1, col=1)
            fig.update_yaxes(title_text="Revenu journalier (€)", row=2, col=1)
            return fig, kpi, total >= 0

        # ── Assemble ───────────────────────────────────────────────────────────
        sims = [
            ("VE Belgique 2026",
             REPO_ROOT / "outputs" / "ve_be" / "simulation_ve_be_2026.csv", "ve"),
            ("VE France 2026",
             REPO_ROOT / "outputs" / "ve_fr" / "simulation_ve_fr_2026.csv", "ve"),
            ("Solaire Belgique 2026",
             REPO_ROOT / "outputs" / "solar_be" / "simulation_be_2026.csv", "solar_be"),
            ("Solaire France 2026",
             REPO_ROOT / "outputs" / "solar_fr" / "simulation_2026" / "summary_2026.csv",
             "solar_fr"),
        ]

        sections = []
        for sim_label, csv_path, sim_type in sims:
            try:
                if sim_type == "ve":
                    fig, kpi, is_cost = _ve_fig(csv_path, sim_label)
                    kpi_color = "#f87171" if is_cost else "#4ade80"
                elif sim_type == "solar_be":
                    fig, kpi, is_pos = _solar_be_fig(csv_path, sim_label)
                    kpi_color = "#4ade80" if is_pos else "#f87171"
                else:
                    fig, kpi, is_pos = _solar_fr_fig(csv_path, sim_label)
                    kpi_color = "#4ade80" if is_pos else "#f87171"

                sections.append(
                    f'<div class="section">'
                    f'<div class="sec-hdr">'
                    f'<span class="sec-title">{sim_label}</span>'
                    f'<span style="color:{kpi_color};font-size:1.4rem;font-weight:700">{kpi}</span>'
                    f'</div>'
                    f'{fig.to_html(full_html=False, include_plotlyjs=False)}'
                    f'</div>'
                )
            except Exception as _e:
                import traceback as _tb
                sections.append(
                    f'<div class="section">'
                    f'<p style="color:#f87171">Erreur {sim_label} : {_e}</p>'
                    f'<pre style="color:#aaa;font-size:11px">{_tb.format_exc()}</pre>'
                    f'</div>'
                )

        return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Reports 2026</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    *{{box-sizing:border-box}}
    body{{margin:0;padding:0;background:{BG};color:#e0e0e0;font-family:'Segoe UI',Arial,sans-serif}}
    .hdr{{background:#0d1120;border-bottom:1px solid #1e2a3a;padding:14px 28px;
          display:flex;align-items:center;gap:20px;flex-wrap:wrap;
          position:sticky;top:0;z-index:10}}
    .hdr h1{{margin:0;font-size:18px;color:deepskyblue}}
    .nav{{display:flex;gap:16px;margin-left:auto;flex-wrap:wrap}}
    .nav a{{color:#aaa;text-decoration:none;font-size:13px}}
    .nav a:hover{{color:white}}
    .section{{max-width:1500px;margin:28px auto 48px auto;padding:0 24px}}
    .sec-hdr{{display:flex;justify-content:space-between;align-items:center;
              margin-bottom:10px;border-bottom:1px solid #1e2a3a;padding-bottom:8px}}
    .sec-title{{font-size:1.05rem;font-weight:600;color:#e0e0e0}}
  </style>
</head>
<body>
  <div class="hdr">
    <h1>📊 Reports 2026 — Simulations</h1>
    <nav class="nav">
      <a href="/">⚡ Elia BE</a>
      <a href="/dashboard_rte">🇫🇷 RTE FR</a>
      <a href="/evaluation">Évaluation</a>
    </nav>
  </div>
  {''.join(sections)}
</body>
</html>"""

    except Exception as e:
        import traceback
        return (f"<pre style='color:red;background:#0b0e16;padding:20px'>"
                f"{traceback.format_exc()}</pre>")


# --------------------------------------------------------------------------------------
# STARTUP
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*80)
    print("  ELIA FORECASTER - Demarrage")
    print("="*80)
    
    if not WAKEPY_AVAILABLE:
        print("\n[WARN] wakepy non installe - L'ordinateur peut se mettre en veille")
        print("    Pour installer: pip install wakepy")
    
    fix_existing_csv_timezone()
    
    print("\n[INFO] Lancement thread de forecast...")
    threading.Thread(target=continuous_data_collection_loop, daemon=True).start()
    
    print("[INFO] Demarrage Flask sur http://0.0.0.0:8080")
    
    if WAKEPY_AVAILABLE:
        print("[INFO] Mode keep-awake active (pas de mise en veille)")
        print("="*80 + "\n")
        with keep.running():
            app.run(host="0.0.0.0", port=8080, debug=False)
    else:
        print("="*80 + "\n")
        app.run(host="0.0.0.0", port=8080, debug=False)