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

## Notes Solar

> Solar FR 2025 : source autoritaire = `day_*.csv` (364 fichiers, somme = 63 545.19 €).
> Le `summary_2025.csv` est inexact (55 jours divergent, écart −146 €) — ne pas utiliser.
>
> Solar BE 2026 : 26 391 € = simulation complète jan→mai 12 (132 j) avec métriques correctes.
> Valeur précédente 19 399 € était un snapshot jan→avr (101 j) avant continuation du service.
>
> Solar FR 2026 : 22 531 € = S2 DA-adapt, jan→mai 13 (133 j), logique vérifiée (DOWN bids,
> forecast\_volume sans abs() effectif, triggers VE-éprouvés).
> Archive legacy S4 (14 919 €, 102 j jan→avr) : `outputs/solar_fr/simulation_2026/summary_2026_s4_legacy.csv`.

## Solaire (revenus)

| Simulation | Résultat | Stratégie | Fichier source |
|---|---|---|---|
| Solar BE 2025 | **68 514€** | S3 | `outputs/solar_be/simulation_2025/` |
| Solar FR 2025 | **63 545€** | S3 | `outputs/solar_fr/simulation_2025/day_*.csv` |
| Solar BE 2026 | **26 391€** | S3 | `outputs/solar_be/simulation_be_2026.csv` |
| Solar FR 2026 | **22 531€** | S2 | `outputs/solar_fr/simulation_2026/summary_2026.csv` |

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
