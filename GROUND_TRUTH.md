# Résultats validés — référence unique

## VE (coût recharge)

| Simulation | Résultat | Stratégie | Fichier source |
|---|---|---|---|
| VE BE 2025 | **64€** | S_BE_opt | `simulation_ve_2025_be_v2/simulation_complete.csv` |
| VE FR 2025 | **-5.19€** | S_BE_opt | `simulation_ve_2025_fr/simulation_complete.csv` |
| VE BE 2026 | **EN COURS** | S_BE_opt | `outputs/ve_be/simulation_ve_be_2026.csv` |
| VE FR 2026 | **EN COURS** | S_BE_opt | `outputs/ve_fr/simulation_ve_fr_2026.csv` |

> Simulations 2026 relancées le 2026-05-14 avec métriques corrigées (triggers alignés 2025,
> source forecast switchée vers forecast\_log\_full.csv + abs() supprimé).
> Résultats à valider et à reporter ici une fois le backfill terminé.

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
