# imbalance_optimisation

Monorepo for Belgian and French electricity grid imbalance forecasting and solar asset optimisation. All sub-projects share a single Python 3.10 virtual environment.

## Shared Virtual Environment

**Location:** `venv/` (repo root)

```
venv/Scripts/activate          # Windows (cmd / PowerShell)
source venv/Scripts/activate   # Git Bash / WSL
```

Python version: 3.10.11
Key packages: pandas 1.5.3, numpy 1.23.5, scikit-learn 1.3.2, scipy 1.10.1, matplotlib 3.10.5, Flask 3.1.3, plotly 6.6.0, selenium 4.35.0, APScheduler 3.11.0

---

## Project Structure

```
imbalance_optimisation/
├── forecasters/
│   ├── elia_forecaster/       # Belgian TSO (Elia) imbalance forecaster
│   └── rte_forecaster/        # French TSO (RTE) imbalance forecaster
├── optimizers/
│   ├── solar_be/              # Solar optimiser — Belgium
│   └── solar_fr/              # Solar optimiser — France
├── data/
│   ├── raw/                   # Unprocessed source data
│   └── processed/             # Cleaned / feature-engineered data
├── outputs/
│   ├── reports/               # Generated reports
│   └── plots/                 # Generated charts
├── models/                    # Trained model artefacts (shared)
├── notebooks/                 # Exploratory Jupyter notebooks
└── venv/                      # Shared Python 3.10 virtual environment
```

---

## Sub-projects

### 1. `forecasters/elia_forecaster/`

Real-time system imbalance forecaster for the Belgian grid (Elia).

- **Entry point:** `app.py` — Flask web server exposing forecast endpoints and live Plotly charts
- **Core logic:** `system_imbalance_forecaster/` package — `SystemImbalanceForecaster` class and feature engineering utilities
- **Key files:**
  - `afrr_merit_order.py` — aFRR merit order data fetching and processing
  - `feature_utils.py` — shared feature extraction helpers
  - `test_elia.py`, `test_capprice.py` — unit/integration tests
  - `run_no_sleep.ps1` — PowerShell launcher that prevents Windows sleep
  - `drivers/`, `chrome-win64/` — Selenium ChromeDriver binaries for web scraping
  - `models/` — serialised sklearn model artefacts
  - `logs/` — runtime forecast logs

### 2. `forecasters/rte_forecaster/`

Scheduled imbalance forecaster for the French grid (RTE), using the RTE Open API.

- **Entry point:** `run_forecast_scheduler.py` — `BlockingScheduler` (APScheduler) that fetches imbalance data from the RTE API every 15 minutes, runs inference, and appends results to `forecast_log.csv`
- **API auth:** OAuth2 client-credentials flow against `digital.iservices.rte-france.com`
- **Key files:**
  - `afrr_merit_order.py` — French aFRR merit order scraping (Selenium)
  - `de_only_selenium.py` — standalone Selenium scraper
  - `artifacts/` — serialised model (`fr_imbalance_full_model.pkl`) and scalers
  - `forecast_log.csv` — rolling inference log

### 3. `optimizers/solar_be/`

Solar generation optimiser for Belgian assets. (In development.)

### 4. `optimizers/solar_fr/`

Solar generation optimiser for French assets. (In development.)

---

## Data

| Path | Contents |
|------|----------|
| `data/raw/` | Source CSVs, API snapshots, scraped data before any transformation |
| `data/processed/` | Cleaned datasets and feature matrices ready for training or inference |

Each forecaster also keeps its own local data cache (e.g. `elia_forecaster/data/`, `rte_forecaster/afrr_offers_test.csv`).

---

## Outputs

| Path | Contents |
|------|----------|
| `outputs/reports/` | Performance reports, backtests, evaluation summaries |
| `outputs/plots/` | Saved figures (PNG/HTML) from analysis and forecasts |

---

## Notes

- The `rte_forecaster/venv/` sub-directory is the old, project-local venv — it is superseded by the root `venv/` and can be removed.
- Selenium scrapers require the ChromeDriver binaries in `elia_forecaster/drivers/` to match the installed Chrome version.
