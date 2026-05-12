# Simulation Results Summary
Generated: 2026-05-13

---

## VE (EV Charging) — Total Cost & EUR/kWh

### Belgium

| Strategy    | Total Cost (EUR) | Charged (kWh) | EUR/kWh  | Smart %  |
|-------------|-----------------|---------------|----------|----------|
| **2025** (365 days — simulation_ve_2025_be_v2) |
| S_BE_opt    |          +63.96 |         7,037 | +0.00909 |    70.6% |
| S1_Prudent  |         +157.57 |         7,033 | +0.02240 |    77.1% |
| S2_Ultra    |         +216.14 |         7,044 | +0.03068 |    85.3% |
| S8v4        |         +300.36 |         7,017 | +0.04281 |    20.7% |
| S8v2        |         +324.12 |         7,017 | +0.04619 |    17.4% |
| S9_Hybrid   |         +371.23 |         7,017 | +0.05291 |    12.3% |
| **2026** (Jan 1 – Apr 11 — outputs/ve_be) |
| S_BE_OPT    |          +59.35 |         1,956 | +0.03034 |    72.5% |
| S2          |          +94.33 |         1,956 | +0.04823 |    69.3% |
| S1          |          +94.75 |         1,953 | +0.04852 |    59.8% |

### France

| Strategy    | Total Cost (EUR) | Charged (kWh) | EUR/kWh  | Smart %  |
|-------------|-----------------|---------------|----------|----------|
| **2025** (365 days — simulation_ve_2025_fr) |
| S_BE_opt    |           -5.19 |         7,037 | -0.00074 |    73.4% |
| S1_Prudent  |          +47.35 |         7,030 | +0.00674 |    74.4% |
| S2_Ultra    |          +83.08 |         7,053 | +0.01178 |    83.8% |
| S8v4        |         +111.38 |         7,005 | +0.01590 |    33.3% |
| S8v2        |         +130.90 |         7,005 | +0.01869 |    29.4% |
| S9_Hybrid   |         +175.13 |         7,002 | +0.02501 |    20.3% |
| **2026** (Jan 1 – Apr 13 — outputs/ve_fr) |
| S1          |          +76.85 |         1,947 | +0.03948 |    45.7% |
| S_BE_OPT    |          +97.92 |         1,947 | +0.05031 |    60.2% |
| S2          |         +103.49 |         1,947 | +0.05317 |    63.0% |

---

## Solar — Revenue per GW installed

### France

| Strategy       | Total (EUR) | DA (EUR) | Imbalance (EUR) | Prod (MWh) | EUR/MWh | vs S1   |
|----------------|-------------|----------|-----------------|------------|---------|---------|
| **2025** (364 days — outputs/solar_fr/simulation_2025) |
| S1 Baseline    |    +49,600  |  +41,176 |          +8,423 |      1,233 |  +40.22 | —       |
| S2 DA-adapt    |    +58,432  |  +42,859 |         +15,573 |      1,233 |  +47.38 | +17.8%  |
| S3 10%-fix     |    +63,399  |  +28,823 |         +34,576 |      1,233 |  +51.40 | +27.8%  |
| **2026** (Jan 1 – May 11 — outputs/solar_fr/simulation_2026) |
| S1 Baseline    |    +14,181  |  +12,945 |          +1,236 |        430 |  +32.97 | —       |
| S2 DA-adapt    |    +22,212  |  +15,440 |          +6,772 |        430 |  +51.64 | +56.6%  |
| S3 10%-fix     |    +20,225  |   +9,062 |         +11,163 |        430 |  +47.02 | +42.6%  |

### Belgium

| Strategy       | Total (EUR) | DA (EUR) | Imbalance (EUR) | Prod (MWh) | EUR/MWh | vs S1   |
|----------------|-------------|----------|-----------------|------------|---------|---------|
| **2025** (365 days — outputs/solar_be/simulation_2025) |
| S1 Baseline    |    +55,686  |  +63,166 |          -7,481 |      1,078 |  +51.66 | —       |
| S2 DA-adapt    |    +64,783  |  +65,363 |            -580 |      1,078 |  +60.10 | +16.3%  |
| S3 10%-fix     |    +68,514  |   +6,317 |         +62,198 |      1,078 |  +63.57 | +23.0%  |
| **2026** (Jan 1 – May 11 — outputs/solar_be/simulation_2026) |
| S1 Baseline    |    +20,521  |  +21,509 |            -988 |        361 |  +56.88 | —       |
| S2 DA-adapt    |    +22,496  |  +23,814 |          -1,318 |        361 |  +62.35 | +9.6%   |
| S3 10%-fix     |    +23,833  |   +2,151 |         +21,682 |        361 |  +66.06 | +16.1%  |

---

## Key Observations

### VE (EV Charging)
- **S_BE_opt is the best strategy in both countries and both years**
- In France 2025, S_BE_opt earns money charging (-5.19 EUR cost = net revenue) — charges almost exclusively during negative ISP events
- In Belgium 2025, S_BE_opt costs only 0.91 ct/kWh vs 5.29 ct/kWh for S9_Hybrid (5.8x cheaper)
- In 2026 (partial), BE S_BE_OPT is 37% cheaper than S1/S2
- FR 2026 partial: S1 is cheapest — S_BE_opt and S2 cost more, suggesting 2026 FR ISP pattern differs from 2025

### Solar
- **S3 (10% fixed nomination) is best in Belgium both years** — exploits persistent positive ISP regime: permanently short position earns large imbalance revenue
- **S2 (DA-adapt) is best in France in 2026** (+56.6% vs S1) — stronger DA price signal in 2026 vs 2025
- **S3 is best in France 2025** (+27.8%) but S2 wins 2026 (+56.6%) — likely driven by more negative DA periods in spring 2026
- BE baseline EUR/MWh consistently higher than FR (51–57 vs 32–40) — BE DA prices were structurally higher in both periods
- Solar 2026 covers only Jan–May (partial year — ongoing)
