# CADT Paper — Simulation Artifacts

This repository contains the simulation source code and output data for the paper:

**"Context-Aware Digital Twin Framework for Zero-Trust Authentication in Resource-Constrained IIoT"**

## Files

- \cadt_proper_simulation.py\ — Standalone Python 3.9 simulation implementing the CADT trust model, energy model (TI CC2650), and Monte Carlo detection evaluation.
- \energy_data.csv\ — Per-minute energy consumption (DTLS vs. CADT) over a 100-minute window.
- \	rust_data.csv\ — Trust score, status, and divergence values over 250 seconds under a replay attack scenario injected at t=100s.

## Requirements

NumPy, SciPy, Matplotlib.

## Reproducibility

Run: \python cadt_proper_simulation.py\

All stochastic elements use \
umpy.random.seed(42)\ for bit-identical outputs.
