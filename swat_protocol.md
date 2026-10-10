# SWaT Validation Protocol (planned external validation)

Goal: run the CADT trust pipeline from `cadt_proper_simulation.py` against
real ICS attack data, replacing simulated telemetry with historian logs
while keeping the detector byte-identical.

## Dataset

SWaT (Secure Water Treatment testbed, iTrust, SUTD): 11 days of historian
logs (7 normal + 4 attack) from a six-stage water-treatment plant, with
labeled attack intervals. Access requires a data-request agreement via the
iTrust lab; this repository cannot redistribute the files.

- Normal: `SWaT_Dataset_Normal_v0.xlsx/.csv` (7 days, 1 s sampling)
- Attack: `SWaT_Dataset_Attack_v0.xlsx/.csv` (4 days, 36 labeled attacks)

## Mapping plan (sensor → context vector)

| Context group | Simulator feature | SWaT surrogate |
|---|---|---|
| env (3) | RSSI / temp / EMI | Flow/pressure readings of *neighboring* stages (ambient process context) |
| ops (3) | V_batt / inter-arrival / seq drift | Historian timestamp gaps + pump/valve actuator states (discretized) |
| phy (1) | \|obs − physics(dev,t)\| | Residual of a disclosed predictor (below) normalized by rolling std |

## Physics surrogate (disclosed, not hidden)

The simulator's analytic process model has no SWaT equivalent, so the
residual uses a persistence-plus-drift predictor fit on normal-operation
data only: `ŷ(t) = y(t−1) + median daily drift`. This is deliberately
weaker than a plant model; any detection it yields understates what a
calibrated twin would achieve.

## Calibration discipline (anti-poisoning)

Baselines (μ, Σ, D50, D99) are fit on normal-operation days only.
Attack days are never touched during calibration — the same rule the
paper's cold-start discussion demands.

## Evaluation (mirrors the paper)

- Per-attack detection / false-alarm rate / response latency at the
  paper's fixed thresholds (α=0.8, λ=1.5, Γ_lo=0.40).
- Same window-minimum ROC construction, disclosed as optimistic.
- Comparison targets: published SWaT IDS results (detection rate + FAR),
  not our own simulator — this is the external baseline the paper lacks.

## Status

`swat_validation.py` implements the loader → context builder → scoring
loop against the real `trust()`/`mahalanobis` machinery. It is untested
against data pending access approval; run `python swat_validation.py
--data-dir <dir>` once files are in place.
