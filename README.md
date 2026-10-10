# CADT Paper — Simulation Artifacts

Simulation source code and output data for the paper:

**"Context-Aware Digital Twin as a Security Proxy: An AI-Driven Zero-Trust
Framework for Intelligent Infrastructure"**

## Files

- `cadt_proper_simulation.py` — simulation implementing the CADT trust model,
  energy model (TI CC2650), and Monte Carlo detection evaluation.
- `energy_data.csv` — per-minute energy consumption (DTLS vs. CADT) over a
  100-minute window. Regenerated on every run.
- `trust_data.csv` — replay-scenario trust score, status, and divergence over
  250 s with attack injected at t=100 s. Regenerated on every run.
- `run_manifest.json` — seed, parameters, package versions, git SHA, result
  summary (including gateway-cost benchmark and ablation, when run), and
  SHA-256 hashes of the output CSVs. Regenerated on every run.
- `hs_sweep.csv` / `figure_hs_sweep.png` — savings-vs-re-key-window curve.
  Generated only by `--sweep-hs`.

## Requirements

See `requirements.txt` (NumPy, SciPy, Matplotlib; scikit-learn optional —
the script uses a dependency-free ROC fallback when it is absent).

```bash
pip install -r requirements.txt
```

## Reproducibility

```bash
python cadt_proper_simulation.py
```

Defaults match the paper (`--seed 42 --trials 500 --hs-seconds 43.0`).
The 43 s value is the *effective* DTLS re-key window (worst case: crypto +
radio-on time + retransmissions), not pure ECDH compute time.

Useful overrides:

```bash
python cadt_proper_simulation.py --trials 50 --seed 1 --no-plots
python cadt_proper_simulation.py --hs-seconds 2.0
python cadt_proper_simulation.py --ablate          # context-group ablation +
                                                   # Euclidean/IsolationForest baselines
python cadt_proper_simulation.py --noise 0.01      # 10x sensor-noise robustness check
python cadt_proper_simulation.py --sweep-hs        # energy-only savings curve
```

All stochastic elements derive from the `--seed` argument (default 42).
`run_manifest.json` records the exact configuration and output hashes.
Verified cross-seed stability: replay detection 97.6% / 96.8% / 96.8% /
97.6% for seeds 42 / 1 / 2 / 3 (500 trials each, FAR 0.0% throughout).

## Baselines

`--ablate` additionally reports two dependency-free baselines on identical
features: a Euclidean-distance variant (same thresholds and recursion) and
a NumPy-only isolation forest (50 trees, 1% score threshold). Where
scikit-learn installs cleanly, its `IsolationForest` runs side by side;
otherwise the run records the skip and the NumPy figures stand.

## External validation (planned)

`swat_protocol.md` + `swat_validation.py` scaffold validation against SWaT
public testbed data (pending dataset access). The script imports the
paper's trust machinery unmodified; only the telemetry source changes.

## Continuous integration

`.github/workflows/sim.yml` runs a reduced verification
(`--seed 42 --trials 50 --ablate --no-plots`, with sklearn) on every push
and uploads the resulting CSVs and manifest as artifacts.
