"""SWaT external-validation scaffold (requires dataset access, see swat_protocol.md).

Runs the paper's trust machinery -- imported unmodified from
cadt_proper_simulation -- against real SWaT historian logs instead of
simulated telemetry. Untested against data pending iTrust approval.

Usage:
    python swat_validation.py --data-dir <dir-with-SWaT-CSVs>
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cadt_proper_simulation import (
    ALP, LAM, G_LO, G_HI, ATK_S, trust,
)
from scipy.spatial.distance import mahalanobis

# SWaT v0 historian columns used as process context (documented in protocol).
# Adjust to the exact header row of the released files.
PROCESS_COLS = ["FIT101", "LIT101", "MV101", "P101", "P102",
                "AIT201", "AIT202", "FIT201"]
LABEL_COL_CANDIDATES = ["Normal/Attack", "Label", "Attack"]


def load_series(data_dir):
    """Return (normal_rows, attack_rows, attack_intervals).

    attack_intervals: list of (start_idx, end_idx) into attack_rows.
    Falls back to whole-file treatment when no label column is found.
    """
    def read(name):
        for ext in (".csv", ".xlsx"):
            p = os.path.join(data_dir, name + ext)
            if os.path.exists(p):
                if ext == ".csv":
                    with open(p, newline="") as f:
                        return list(csv.DictReader(f))
                raise SystemExit(f"convert {p} to CSV first (xl -> csv)")
        raise SystemExit(f"missing {name}.csv in {data_dir} "
                         f"(see swat_protocol.md for access)")

    normal = read("SWaT_Dataset_Normal_v0")
    attack = read("SWaT_Dataset_Attack_v0")
    header = set(attack[0].keys()) if attack else set()
    label_col = next((c for c in LABEL_COL_CANDIDATES if c in header), None)
    intervals = []
    if label_col:
        in_atk, start = False, 0
        for i, row in enumerate(attack):
            bad = str(row[label_col]).strip().lower() not in ("normal", "0", "")
            if bad and not in_atk:
                in_atk, start = True, i
            elif not bad and in_atk:
                in_atk = False
                intervals.append((start, i))
        if in_atk:
            intervals.append((start, len(attack)))
    print(f"normal rows: {len(normal)}, attack rows: {len(attack)}, "
          f"labeled intervals: {len(intervals)}")
    return normal, attack, intervals


def to_float(row, cols):
    vals = []
    for c in cols:
        try:
            vals.append(float(row.get(c, "nan")))
        except (TypeError, ValueError):
            vals.append(np.nan)
    return np.array(vals)


def build_context_matrix(rows, cols, ops_dim=3):
    """Context rows: [env(process neighbors) | ops(gaps+actuators) | phy].

    phy residual is filled by the caller (needs the predictor); here it is
    initialized to NaN and must be replaced before scoring.
    """
    M = np.array([to_float(r, cols) for r in rows])
    n = len(M)
    gaps = np.zeros(n)  # historian is 1 s; kept for schema compatibility
    phy = np.full(n, np.nan)
    d = len(cols)
    C = np.zeros((n, d + ops_dim + 1))
    C[:, :d] = M
    C[:, d] = gaps
    return C


def persistence_residual(series):
    """Disclosed weak predictor: y(t-1) + median drift (fit on normal only).
    Returns residuals normalized by rolling std (window 3600)."""
    y = np.asarray(series, dtype=float)
    drift = float(np.nanmedian(np.diff(y[np.isfinite(y)])))
    pred = np.roll(y, 1) + drift
    pred[0] = y[0]
    resid = np.abs(y - pred)
    out = np.full_like(resid, np.nan)
    for i in range(len(y)):
        w = resid[max(0, i - 3600):i + 1]
        w = w[np.isfinite(w)]
        out[i] = resid[i] / (np.std(w) + 1e-9) if len(w) > 10 else 0.0
    return out


def evaluate(C_norm, mu, inv_cov, d50, d99, intervals):
    """Score every row; an attack interval counts detected if trust crosses
    G_LO within it. Returns dict mirroring the paper's MC metrics."""
    tp = fp = 0
    rts = []
    attacked = np.zeros(len(C_norm), dtype=bool)
    for s, e in intervals:
        attacked[s:e] = True
    tr = 1.0
    det_in = {}
    for i, v in enumerate(C_norm):
        if np.isnan(v).any():
            continue
        dk = float(mahalanobis(v, mu, inv_cov))
        dn = max(0.0, (dk - d50) / max(d99 - d50, 1e-6))
        tr = max(0.0, min(1.0, ALP * tr + (1 - ALP) * np.exp(-LAM * dn)))
        if tr <= G_LO:
            for k, (s, e) in enumerate(intervals):
                if s <= i < e and k not in det_in:
                    det_in[k] = i - s
    tp = len(det_in)
    fn = len(intervals) - tp
    # FAR: fraction of normal rows below threshold (proxy; see protocol)
    far_rows = 0
    n_norm = 0
    tr = 1.0
    for i, v in enumerate(C_norm):
        if attacked[i] or np.isnan(v).any():
            continue
        dk = float(mahalanobis(v, mu, inv_cov))
        dn = max(0.0, (dk - d50) / max(d99 - d50, 1e-6))
        tr = max(0.0, min(1.0, ALP * tr + (1 - ALP) * np.exp(-LAM * dn)))
        n_norm += 1
        if tr <= G_LO:
            far_rows += 1
    return {"attacks": len(intervals), "detected": tp, "missed": fn,
            "det_rate": round(100 * tp / max(len(intervals), 1), 1),
            "far_row_pct": round(100 * far_rows / max(n_norm, 1), 3),
            "mean_resp_s": round(float(np.mean(list(det_in.values())))
                                 if det_in else 0.0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    normal, attack, intervals = load_series(args.data_dir)
    cols = [c for c in PROCESS_COLS if c in normal[0]]
    print("using columns:", cols)
    Cn = build_context_matrix(normal, cols)
    Ca = build_context_matrix(attack, cols)
    # phy residual from normal-fit predictor (anti-poisoning: predictor
    # form fixed here, statistics drawn from normal days only in evaluate).
    # Primary process variable (first column) carries the residual.
    for C, rows in ((Cn, normal), (Ca, attack)):
        col = np.array([to_float(r, cols)[0] for r in rows])
        C[:, len(cols) + 3] = persistence_residual(col)
    ok = ~(np.isnan(Cn).any(axis=1))
    mu = np.mean(Cn[ok], axis=0)
    cov = np.cov(Cn[ok], rowvar=False) + np.eye(Cn.shape[1]) * 1e-4
    inv_cov = np.linalg.inv(cov)
    dists = [mahalanobis(c, mu, inv_cov) for c in Cn[ok][:5000]]
    d50, d99 = float(np.percentile(dists, 50)), float(np.percentile(dists, 99))
    res = evaluate(Ca, mu, inv_cov, d50, d99, intervals)
    print(json.dumps(res, indent=1))
    with open("swat_results.json", "w") as f:
        json.dump({"config": {"alpha": ALP, "lambda": LAM,
                              "g_lo": G_LO, "g_hi": G_HI,
                              "columns": cols},
                   "results": res}, f, indent=2)


if __name__ == "__main__":
    main()
