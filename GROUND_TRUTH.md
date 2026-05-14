# Résultats validés — référence unique

## VE (coût recharge)

| Simulation | Résultat | Stratégie | Fichier source |
|---|---|---|---|
| VE BE 2025 | **64€** | S_BE_opt | `simulation_ve_2025_be_v2/simulation_complete.csv` |
| VE FR 2025 | **-5.19€** | S_BE_opt | `simulation_ve_2025_fr/simulation_complete.csv` |
| VE BE 2026 | **86.86€** | S_BE_opt | `outputs/ve_be/simulation_ve_be_2026.csv` |
| VE FR 2026 | **-11.18€** | S_BE_opt | `outputs/ve_fr/simulation_ve_fr_2026.csv` |

> Simulations 2026 backfill terminé le 2026-05-14 (jan→mai, 99.9% des QH).
> Corrections appliquées : source forecast\_log\_full.csv, abs() supprimé, triggers alignés 2025,
> ISP depuis cache local (BE : isp\_2026-\*.csv / FR : bulk-fetch RTE par jour),
> fix DST 2026-03-29 02:00 (FR). Services VeBeSimulator2026 + VeFrSimulator2026 en prod.

## Solaire (revenus)

| Simulation | Résultat | Stratégie | Fichier source |
|---|---|---|---|
| Solar BE 2025 | **68 514€** | S3 | `outputs/solar_be/simulation_2025/` |
| Solar FR 2025 | **63 545€** | S3 | `outputs/solar_fr/simulation_2025/` |
| Solar BE 2026 | **19 399€** | S3 | `outputs/solar_be/simulation_be_2026.csv` |
| Solar FR 2026 | **14 919€** | S4 | `simulation_solar_intelligent/summary_2026.csv` |

## Trigger S_BE_opt

```
forecast_volume > 200 AND afrr_ratio_neg > 50% AND mfrr_ratio_neg > 50%
```

> **JAMAIS `abs()` sur `forecast_volume`**

## Services en prod

| Service | Fichier |
|---|---|
| EliaImbalanceForecaster | `forecasters/elia_forecaster/app.py` |
| RteImbalanceForecaster | `forecasters/rte_forecaster/run_forecast_scheduler.py` |
| SolarBeScheduler | `optimizers/solar_be/solar_be_scheduler.py` |
| SolarFrScheduler | `optimizers/solar_fr/solar_fr_scheduler.py` |
