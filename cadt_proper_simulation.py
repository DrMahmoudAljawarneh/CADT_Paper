"""CADT Framework — Physics-Based Simulation.

Implements 7-dimensional Mahalanobis context fusion (3 groups: env/ops/phy),
3 attack scenarios (replay, masquerade, FDI), energy model (TI CC2650), and
Monte Carlo detection evaluation with sensitivity/ROC analysis.

Outputs (written to CWD):
  energy_data.csv, trust_data.csv (replay scenario), run_manifest.json,
  figure1_energy.png, figure2_trust.png, figure_divergence.png,
  figure_sensitivity.png, figure_roc.png

Run:  python cadt_proper_simulation.py [--seed 42] [--trials 500]
      [--hs-seconds 43.0] [--no-plots]
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

try:
    from sklearn.metrics import roc_curve, auc
    _HAS_SKLEARN = True
except Exception:  # reproducibility: ROC still works without sklearn
    _HAS_SKLEARN = False


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CADT trust-model simulation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trials", type=int, default=500,
                   help="Monte Carlo trials per attack type (balanced atk/normal)")
    p.add_argument("--hs-seconds", type=float, default=43.0,
                   help="Effective DTLS re-key window in seconds (worst case: "
                        "crypto + radio-on + retransmissions, NOT pure ECDH time)")
    p.add_argument("--ablate", action="store_true",
                   help="Run context-group ablation (env/ops/phy/full) plus "
                        "IsolationForest and Euclidean baselines")
    p.add_argument("--noise", type=float, default=0.001,
                   help="Sensor noise as fraction of reading (default 0.001 = "
                        "paper's 0.1%%; try 0.01 for a 10x robustness check)")
    p.add_argument("--sweep-hs", action="store_true",
                   help="Energy-only sweep of savings vs re-key window; writes "
                        "figure_hs_sweep.png and prints the table, then exits")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


# ===== Config (defaults match paper Table sim_params) =====
N_DEV = 500
SIM_MIN = 100
TOTAL_S = 250
ATK_S = 100

V = 3.0
I_TX = 9.1e-3
I_HSPIKE = 45e-3
TX_S = 25.0

ALP = 0.8
LAM = 1.5
G_HI = 0.85
G_LO = 0.40
BATT_J = 2000 * 3.0 * 3.6  # 2000 mAh @ 3.0 V
SENSOR_PRECISION = 0.005
SIGMA_NOISE = 0.001  # default 0.1% of reading; overridden by --noise
HS_SWEEP = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 43.0]  # re-key windows (s)

# ===== Context Generation =====
# C_env = [RSSI (dBm), Temp (C), EMI]
# C_ops = [V_batt (V), inter_arrival (s), seq_drift]
# C_phy = [Physics residual |obs-pred| / (base*precision)]
# Full vector = 7 features in 3 groups


def make_devs(n):
    devs = []
    nv, np_ = int(n * 0.4), int(n * 0.3)
    for i in range(n):
        if i < nv:
            t = 'vib'
            b = np.random.uniform(5, 15)
        elif i < nv + np_:
            t = 'proc'
            b = np.random.uniform(50, 150)
        else:
            t = 'act'
            b = np.random.uniform(0, 100)
        devs.append(dict(id=i, type=t, base=b))
    return devs


def physics(dev, t):
    if dev['type'] == 'vib':
        return dev['base'] + np.sin(t * 0.02) * 0.5
    elif dev['type'] == 'proc':
        return dev['base'] + np.sin(t * 0.005) * 2.0
    return dev['base']


def sensor_noise(val, sigma=None):
    sig = SIGMA_NOISE if sigma is None else sigma
    return val + np.random.normal(0, abs(val) * sig)


def status_of(trust):
    if trust <= G_LO:
        return "Isolated"
    if trust < G_HI:
        return "Challenge"
    return "Normal"


def generate_context(dev, t, atk_type=None):
    """Return [RSSI, Temp, EMI, V_batt, IA, SeqDrift, PhyRes]."""
    pred = physics(dev, t)

    # Benign behavior
    rssi = np.random.normal(-60, 2)
    temp = np.random.normal(25, 0.5)
    emi = np.random.normal(0.1, 0.05)
    vbatt = 3.0 - (t / 10000) + np.random.normal(0, 0.01)
    ia = np.random.normal(1.0, 0.05)
    # NOTE: benign seq drift is exactly 0 (matches paper calibration data).
    # The seq/ia dimensions are exercised only by the masquerade attack below;
    # see README/paper limitations discussion.
    seq = 0.0
    obs = sensor_noise(pred)

    if atk_type == 'replay':
        # Stale payload, valid HMAC; env/ops stay nominal
        obs = sensor_noise(pred * 0.99795)
    elif atk_type == 'masquerade':
        # Impersonates another class: radio jump AND timing/sequence anomaly,
        # matching the paper's "sequence number + transmission frequency" claim
        rssi = np.random.normal(-75, 2)
        ia = np.random.normal(0.35, 0.05)
        seq = np.random.normal(3.0, 0.3)
    elif atk_type == 'fdi':
        drift = 1.0 - 0.0005 * (t - ATK_S)
        obs = sensor_noise(pred * drift)

    phy_res = abs(obs - pred) / (max(abs(dev['base']), 1e-6) * SENSOR_PRECISION)
    return np.array([rssi, temp, emi, vbatt, ia, seq, phy_res])


GROUPS = {"env": [0, 1, 2], "ops": [3, 4, 5], "phy": [6],
          "full": [0, 1, 2, 3, 4, 5, 6]}


def fit_gaussian(C_matrix, idx):
    """Baseline (mu, inv_cov, d50, d99) restricted to feature subset idx."""
    from scipy.spatial.distance import mahalanobis as _mh
    X = np.asarray(C_matrix)[:, idx]
    mu = np.mean(X, axis=0)
    k = len(idx)
    cov = np.cov(X, rowvar=False).reshape(k, k) + np.eye(k) * 1e-4
    inv_cov = np.linalg.inv(cov)
    dists = [_mh(c, mu, inv_cov) for c in X]
    return mu, inv_cov, float(np.percentile(dists, 50)), float(np.percentile(dists, 99))


def compute_baselines(devs):
    import time
    bl = {}
    cal = {}
    by = {}
    for d in devs:
        by.setdefault(d['type'], []).append(d)

    t0 = time.perf_counter()
    for tn, grp in by.items():
        C_matrix = []
        for d in grp[:10]:  # subset for speed; documented limitation
            for t in range(500):
                C_matrix.append(generate_context(d, t, None))
        C_matrix = np.array(C_matrix)
        cal[tn] = C_matrix
        mu, inv_cov, d_50, d_99 = fit_gaussian(C_matrix, GROUPS["full"])
        bl[tn] = (mu, inv_cov, d_50, d_99)
        print(f"  {tn}: D_50={d_50:.4f}, D_99={d_99:.4f}")
    calib_s = time.perf_counter() - t0
    n_calib = sum(len(g[:10]) * 500 for g in by.values())
    print(f"  calibration wall time: {calib_s:.1f}s for {n_calib} samples "
          f"({calib_s / n_calib * 1e6:.0f}us/sample)")
    return bl, {"calib_wall_s": calib_s, "calib_samples": n_calib}, cal


class NumpyIsolationForest:
    """Minimal isolation forest, NumPy-only (isolation principle: anomalies
    isolate in fewer random splits).

    Score follows the standard definition s(x) = 2**(-E[h(x)] / c(psi))
    with c(psi) = 2*H(psi-1) - 2*(psi-1)/psi. Uses the ambient np.random
    stream, so callers control determinism. Exists because scikit-learn is
    not installable in every environment; where sklearn is present both
    implementations run and are reported side by side.
    """

    def __init__(self, n_trees=50, subsample=256, max_depth=None):
        self.n_trees = n_trees
        self.subsample = subsample
        self.max_depth = max_depth or int(np.ceil(np.log2(subsample)))
        hn = np.zeros(subsample + 1)
        hn[1:] = np.cumsum(1.0 / np.arange(1, subsample + 1))
        self._c = np.zeros(subsample + 1)
        n = np.arange(2, subsample + 1)
        self._c[n] = 2 * hn[n - 1] - 2 * (n - 1) / n
        self.trees = []

    def _build(self, X, idx, depth):
        feat, thr, left, right, leaf, size = self._cur
        node = len(feat)
        feat.append(0)
        thr.append(0.0)
        left.append(-1)
        right.append(-1)
        leaf.append(False)
        size.append(len(idx))
        xs = X[idx]
        lo, hi = xs.min(axis=0), xs.max(axis=0)
        movable = np.where(hi > lo)[0]
        if depth >= self.max_depth or len(idx) <= 1 or len(movable) == 0:
            leaf[node] = True
            return node
        f = movable[np.random.randint(len(movable))]
        t = float(np.random.uniform(lo[f], hi[f]))
        l_idx = idx[xs[:, f] < t]
        r_idx = idx[xs[:, f] >= t]
        if len(l_idx) == 0 or len(r_idx) == 0:  # degenerate split
            leaf[node] = True
            return node
        feat[node] = int(f)
        thr[node] = t
        left[node] = self._build(X, l_idx, depth + 1)
        right[node] = self._build(X, r_idx, depth + 1)
        return node

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        self.trees = []
        for _ in range(self.n_trees):
            idx = np.random.choice(n, min(self.subsample, n), replace=False)
            self._cur = ([], [], [], [], [], [])
            self._build(X, idx, 0)
            self.trees.append(tuple(np.asarray(a) for a in self._cur))
        return self

    def _score_tree(self, X, tree):
        feat, thr, left, right, leaf, size = tree
        N = len(X)
        node = np.zeros(N, dtype=int)
        depth = np.zeros(N)
        active = np.ones(N, dtype=bool)
        for _ in range(self.max_depth + 1):
            if not active.any():
                break
            f = feat[node]
            go_left = X[np.arange(N), f] < thr[node]
            nxt = np.where(go_left, left[node], right[node])
            depth += active & ~leaf[node]
            node = np.where(active, nxt, node)
            active = active & ~leaf[node]
        adj = np.where(size[node] > 1, self._c[np.minimum(size[node],
                                                          self.subsample)],
                       0.0)
        return depth + adj

    def anomaly_score(self, X):
        X = np.asarray(X, dtype=float)
        h = np.mean([self._score_tree(X, t) for t in self.trees], axis=0)
        return np.power(2.0, -h / self._c[self.subsample])


def ablation_report(bl, cal, devs, seed, n_paths=100):
    """Context-group ablation + IsolationForest baseline (replay pool).

    Answers: is the fusion doing work, or would one group suffice? And how
    does the Mahalanobis+recursion pipeline compare to a generic detector
    on identical features? Returns a dict for the manifest.

    Uses a dedicated RNG stream derived from seed so results are identical
    regardless of --trials; global state is restored on exit.
    """
    from scipy.spatial.distance import mahalanobis as _mh
    st = np.random.get_state()
    np.random.seed((seed + 1) * 7919 % (2 ** 31))
    ref = devs[0]
    tn = ref['type']
    C_full = cal[tn]

    out = {}
    for atk_name in ['replay', 'masquerade', 'fdi']:
        pool = []
        for i in range(n_paths):
            atk = (i % 2 == 0)
            vecs = []
            for s in range(TOTAL_S):
                a_type = atk_name if (atk and s >= ATK_S) else None
                vecs.append(generate_context(ref, s, a_type))
            pool.append((atk, np.array(vecs)))

        out[atk_name] = {}
        for gname, idx in GROUPS.items():
            mu, inv_cov, d50, d99 = fit_gaussian(C_full, idx)
            tp = tn_ = fp = fn = 0
            for atk, vecs in pool:
                tr = 1.0
                det = False
                fiso = False
                for s, v in enumerate(vecs[:, idx].reshape(len(vecs), len(idx))):
                    dk = float(_mh(v, mu, inv_cov))
                    tr = trust(dk, tr, d50, d99)
                    if tr <= G_LO and not det:
                        det = True
                        if s < ATK_S:
                            fiso = True
                if atk:
                    if det and not fiso:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if det:
                        fp += 1
                    else:
                        tn_ += 1
            dr = tp / (tp + fn) * 100 if (tp + fn) else 0
            far = fp / (fp + tn_) * 100 if (fp + tn_) else 0
            out[atk_name][gname] = {"det_rate": round(dr, 1), "far": round(far, 1)}
            print(f"  ablate[{atk_name}/{gname}]: Det {dr:.1f}%, FAR {far:.1f}%")

        # Naive baseline: Euclidean distance + identical trust recursion.
        # Same features, same thresholds, same decision rule -- only the
        # covariance-aware metric is removed. Dependency-free by construction.
        Xe = np.asarray(C_full)
        mu_e = np.mean(Xe, axis=0)
        euc = [float(np.linalg.norm(c - mu_e)) for c in Xe]
        e50, e99 = float(np.percentile(euc, 50)), float(np.percentile(euc, 99))
        tp = tn_ = fp = fn = 0
        for atk, vecs in pool:
            tr = 1.0
            det = False
            fiso = False
            for s, v in enumerate(vecs):
                dn = max(0.0, (float(np.linalg.norm(v - mu_e)) - e50)
                         / max(e99 - e50, 1e-6))
                tr = max(0.0, min(1.0, ALP * tr + (1 - ALP) * np.exp(-LAM * dn)))
                if tr <= G_LO and not det:
                    det = True
                    if s < ATK_S:
                        fiso = True
            if atk:
                if det and not fiso:
                    tp += 1
                else:
                    fn += 1
            else:
                if det:
                    fp += 1
                else:
                    tn_ += 1
        dr = tp / (tp + fn) * 100 if (tp + fn) else 0
        far = fp / (fp + tn_) * 100 if (fp + tn_) else 0
        out[atk_name]["euclidean_full"] = {"det_rate": round(dr, 1),
                                           "far": round(far, 1)}
        print(f"  baseline[{atk_name}/euclidean]: Det {dr:.1f}%, FAR {far:.1f}%")

        # Dependency-free isolation-forest baseline on identical features.
        # Threshold = 99th percentile of training scores (matches the 1%
        # contamination convention); decision rule mirrors the sklearn path.
        nif = NumpyIsolationForest().fit(C_full)
        thr_if = float(np.percentile(nif.anomaly_score(C_full), 99))
        tp = tn_ = fp = fn = 0
        for atk, vecs in pool:
            pred = (nif.anomaly_score(np.asarray(vecs)) >= thr_if)
            det = bool(pred[ATK_S:].sum() >= 5)
            pre_fp = bool(pred[:ATK_S].sum() >= 5)
            if atk:
                if det and not pre_fp:
                    tp += 1
                else:
                    fn += 1
            else:
                if pre_fp or det:
                    fp += 1
                else:
                    tn_ += 1
        dr = tp / (tp + fn) * 100 if (tp + fn) else 0
        far = fp / (fp + tn_) * 100 if (fp + tn_) else 0
        out[atk_name]["numpy_isoforest_full"] = {"det_rate": round(dr, 1),
                                                 "far": round(far, 1)}
        print(f"  baseline[{atk_name}/numpy-isoforest]: Det {dr:.1f}%, "
              f"FAR {far:.1f}%")

        # Generic baseline: IsolationForest on identical full features
        try:
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(contamination=0.01, random_state=0).fit(C_full)
            tp = tn_ = fp = fn = 0
            for atk, vecs in pool:
                pred = clf.predict(vecs)  # 1 normal, -1 anomaly
                post = pred[ATK_S:]
                det = bool((post == -1).sum() >= 5)  # >=5 anomalous seconds
                pre_fp = bool((pred[:ATK_S] == -1).sum() >= 5)
                if atk:
                    if det and not pre_fp:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if pre_fp or det:
                        fp += 1
                    else:
                        tn_ += 1
            dr = tp / (tp + fn) * 100 if (tp + fn) else 0
            far = fp / (fp + tn_) * 100 if (fp + tn_) else 0
            out[atk_name]["isoforest_full"] = {"det_rate": round(dr, 1),
                                               "far": round(far, 1)}
            print(f"  baseline[{atk_name}/isoforest]: Det {dr:.1f}%, FAR {far:.1f}%")
        except Exception as e:
            out[atk_name]["isoforest_full"] = {"error": f"sklearn unavailable: {e}"}
            print(f"  baseline[{atk_name}/isoforest]: skipped ({str(e)[:80]})")
    np.random.set_state(st)
    return out


def benchmark_step(bl, devs, steps=2000):
    """Time one twin-side update: context gen + Mahalanobis + trust.

    Preserves the global np.random state so timed draws never alter
    simulation results.
    """
    import time
    st = np.random.get_state()
    t0 = time.perf_counter()
    tr = 1.0
    for s in range(steps):
        dev = devs[s % len(devs)]
        mu, inv_cov, d50, d99 = bl[dev['type']]
        ctx = generate_context(dev, s % TOTAL_S, None)
        dk = float(mahalanobis(ctx, mu, inv_cov))
        tr = trust(dk, tr, d50, d99)
    wall = time.perf_counter() - t0
    np.random.set_state(st)
    per_step_ms = wall / steps * 1e3
    # Projection: one update per twin per second across the fleet
    fleet_ms_per_s = per_step_ms * len(devs)
    print(f"  twin update: {per_step_ms:.3f}ms/step -> "
          f"{fleet_ms_per_s:.0f}ms CPU per 1s fleet sweep ({len(devs)} twins)")
    return {"per_step_ms": per_step_ms, "fleet_ms_per_s": fleet_ms_per_s,
            "n_twins": len(devs), "steps": steps}


def trust(dk, prev, d_50, d_99, alp=ALP, lam=LAM):
    dn = max(0.0, (dk - d_50) / max(d_99 - d_50, 1e-6))
    return max(0.0, min(1.0, alp * prev + (1 - alp) * np.exp(-lam * dn)))


def _roc_fpr_tpr(scores, labels):
    """Binary ROC (higher score = more anomalous). sklearn if present."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if _HAS_SKLEARN:
        fpr, tpr, _ = roc_curve(labels, scores)
        return np.asarray(fpr), np.asarray(tpr), float(auc(fpr, tpr))
    # Fallback: threshold sweep (no dependency)
    thrs = np.unique(np.concatenate([[scores.max() + 1e-9], scores]))
    P = max(int(labels.sum()), 1)
    N = max(int((1 - labels).sum()), 1)
    fpr, tpr = [0.0], [0.0]
    for th in sorted(thrs, reverse=True):
        pred = scores >= th
        tpr.append(float(((pred == 1) & (labels == 1)).sum()) / P)
        fpr.append(float(((pred == 1) & (labels == 0)).sum()) / N)
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    order = np.argsort(fpr)
    return fpr[order], tpr[order], float(np.trapezoid(tpr[order], fpr[order]))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return "unknown"


def energy_sweep(hs_list=HS_SWEEP):
    """Deterministic savings-vs-window curve (no RNG). Returns rows."""
    rows = []
    for hs in hs_list:
        de = [V * I_HSPIKE * hs + V * I_TX * max(0.0, TX_S - hs)
              if (t % 10 == 0 and t > 0) else V * I_TX * TX_S
              for t in range(SIM_MIN)]
        ce = [V * I_TX * TX_S] * SIM_MIN
        ad, ac = float(np.mean(de)), float(np.mean(ce))
        rows.append({"hs_seconds": hs,
                     "dtls_j_min": round(ad, 4), "cadt_j_min": round(ac, 4),
                     "saving_pct": round((1 - ac / ad) * 100, 1),
                     "dtls_days": round(BATT_J / (ad * 60 * 24), 1),
                     "cadt_days": round(BATT_J / (ac * 60 * 24), 1)})
    return rows


def run(args):
    global SIGMA_NOISE
    SIGMA_NOISE = args.noise
    if args.sweep_hs:
        rows = energy_sweep()
        print("hs_s  dtls_J  cadt_J  save%  dtls_d  cadt_d")
        for r in rows:
            print(f"{r['hs_seconds']:>5}  {r['dtls_j_min']:.4f}  "
                  f"{r['cadt_j_min']:.4f}  {r['saving_pct']:>5}%  "
                  f"{r['dtls_days']:>6}d  {r['cadt_days']:>6}d")
        with open("hs_sweep.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        if not args.no_plots:
            plt.figure(figsize=(8, 5.5))
            xs = [r["hs_seconds"] for r in rows]
            plt.semilogx(xs, [r["saving_pct"] for r in rows], "bo-", lw=2)
            plt.axvline(x=43.0, color="r", ls="--",
                        label="Paper worst-case bound (43 s)")
            plt.xlabel("Effective DTLS re-key window (s, log scale)")
            plt.ylabel("Mean edge-energy saving (%)")
            plt.title("CADT savings vs re-key window")
            plt.grid(True, ls="--", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig("figure_hs_sweep.png", dpi=300)
            print("  -> figure_hs_sweep.png + hs_sweep.csv written")
        return
    np.random.seed(args.seed)
    print("=" * 60)
    print("  CADT PHYSICS-BASED SIMULATION")
    print(f"  seed={args.seed} trials={args.trials} hs_seconds={args.hs_seconds} "
          f"sklearn={'yes' if _HAS_SKLEARN else 'no (fallback ROC)'}")
    print("=" * 60)
    devs = make_devs(N_DEV)
    print(f"Devices: {N_DEV}")

    print("\n[1/5] Baselines ...")
    bl, calib_cost, cal_data = compute_baselines(devs)

    print("\n[1b/5] Gateway cost benchmark ...")
    step_cost = benchmark_step(bl, devs)

    print("\n[2/5] Energy ...")
    HS_S = args.hs_seconds
    re = []
    for t in range(SIM_MIN):
        hs = (t % 10 == 0 and t > 0)  # 10-min re-key interval
        de = V * I_HSPIKE * HS_S + V * I_TX * max(0.0, TX_S - HS_S) if hs else V * I_TX * TX_S
        ce = V * I_TX * TX_S
        re.append(dict(Time_Minute=t, DTLS_Energy_J=round(de, 4), CADT_Energy_J=round(ce, 4)))

    ad = np.mean([r['DTLS_Energy_J'] for r in re])
    ac = np.mean([r['CADT_Energy_J'] for r in re])
    bd = BATT_J / (ad * 60 * 24)
    bc = BATT_J / (ac * 60 * 24)
    print(f"  Avg DTLS: {ad:.4f} J/min, CADT: {ac:.4f} J/min, "
          f"Reduction: {(1 - ac / ad) * 100:.1f}%")
    print(f"  Battery: DTLS {bd:.1f}d vs CADT {bc:.1f}d (at {TX_S:.0f}s TX/min duty)")

    with open("energy_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Time_Minute", "DTLS_Energy_J", "CADT_Energy_J"])
        w.writeheader()
        w.writerows(re)

    print("\n[3/5] Single Scenarios (Replay, Masq, FDI) ...")
    dev0 = devs[0]
    mu, inv_cov, d50, d99 = bl[dev0['type']]

    def _norm(dk, a, b):
        return max(0.0, (dk - a) / max(b - a, 1e-6))

    scen_data = {}
    for atk_name in ['replay', 'masquerade', 'fdi']:
        tr = 1.0
        rt = []
        for s in range(TOTAL_S):
            a_type = atk_name if s >= ATK_S else None
            ctx = generate_context(dev0, s, a_type)
            dk = float(mahalanobis(ctx, mu, inv_cov))
            tr = trust(dk, tr, d50, d99)
            # Paper plots/discusses the *normalized* divergence Dn
            rt.append(dict(Time=s, Trust=tr, Div=_norm(dk, d50, d99), Status=status_of(tr)))
        scen_data[atk_name] = rt
        iso = [r['Time'] for r in rt if r['Trust'] <= G_LO]
        print(f"  {atk_name.upper()} Isolated at: {iso[0] if iso else 'Never'}s")

    # trust_data.csv: replay scenario (paper §Results), exact paper schema
    with open("trust_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Time_Second", "Trust_Score", "Status", "Divergence"])
        w.writeheader()
        for r in scen_data['replay']:
            w.writerow({"Time_Second": r["Time"],
                        "Trust_Score": round(r["Trust"], 4),
                        "Status": r["Status"],
                        "Divergence": round(r["Div"], 4)})

    print(f"\n[4/5] Monte Carlo ({args.trials} trials) ...")
    MC = args.trials
    results = {}
    for atk_name in ['replay', 'masquerade', 'fdi']:
        tp = tn = fp = fn = 0
        rts = []
        scores_for_roc = []
        labels_for_roc = []
        for tl in range(MC):
            dev = devs[np.random.randint(len(devs))]
            mu, inv_cov, d50, d99 = bl[dev['type']]
            tr = 1.0
            atk = (tl % 2 == 0)
            det = False
            fiso = False
            min_tr_after_atk = 1.0
            min_tr_normal = 1.0

            for s in range(TOTAL_S):
                a_type = atk_name if (atk and s >= ATK_S) else None
                ctx = generate_context(dev, s, a_type)
                dk = mahalanobis(ctx, mu, inv_cov)
                tr = trust(dk, tr, d50, d99)

                if s >= ATK_S:
                    min_tr_after_atk = min(min_tr_after_atk, tr)
                else:
                    min_tr_normal = min(min_tr_normal, tr)

                if tr <= G_LO and not det:
                    det = True
                    if s < ATK_S:
                        fiso = True
                    elif atk:
                        rts.append(s - ATK_S)

            if atk:
                if det and not fiso:
                    tp += 1
                else:
                    fn += 1
                scores_for_roc.append(min_tr_after_atk)
                labels_for_roc.append(1)
            else:
                if det:
                    fp += 1
                else:
                    tn += 1
                scores_for_roc.append(min_tr_normal)
                labels_for_roc.append(0)

        dr = tp / (tp + fn) * 100 if (tp + fn) else 0
        ci = 1.96 * np.sqrt((dr / 100) * (1 - dr / 100) / (tp + fn)) * 100 if (tp + fn) else 0
        far = fp / (fp + tn) * 100 if (fp + tn) else 0
        ar = float(np.mean(rts)) if rts else 0.0
        print(f"  {atk_name.upper()}: Det {dr:.1f}% (±{ci:.1f}%), "
              f"FAR {far:.1f}%, Resp {ar:.1f}s")
        results[atk_name] = {'dr': dr, 'ci': ci, 'far': far, 'ar': ar,
                             'scores': scores_for_roc, 'labels': labels_for_roc,
                             'rts': rts}

    if args.ablate:
        print("\n[4b/5] Ablation + generic baseline ...")
        ablation = ablation_report(bl, cal_data, devs, args.seed)
    else:
        ablation = {"note": "run with --ablate for context-group ablation"}

    manifest = {
        "seed": args.seed, "trials": MC, "hs_seconds": HS_S,
        "N_DEV": N_DEV, "SIM_MIN": SIM_MIN, "TOTAL_S": TOTAL_S, "ATK_S": ATK_S,
        "ALP": ALP, "LAM": LAM, "G_HI": G_HI, "G_LO": G_LO,
        "V": V, "I_TX": I_TX, "I_HSPIKE": I_HSPIKE, "TX_S": TX_S,
        "SENSOR_PRECISION": SENSOR_PRECISION, "SIGMA_NOISE": SIGMA_NOISE,
        "noise_sigma": SIGMA_NOISE,
        "git_sha": git_sha(), "python": sys.version.split()[0],
        "sklearn": _HAS_SKLEARN,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "gateway_cost": {"calibration": calib_cost, "per_step": step_cost},
        "ablation": ablation,
        "note": "ROC scores use window-minimum trust (optimistic full-window "
                "statistic); online performance = detection/FAR at fixed thresholds.",
        "results": {k: {"det_rate": v["dr"], "det_ci95": v["ci"],
                        "far": v["far"], "mean_resp_s": v["ar"]}
                    for k, v in results.items()},
    }
    with open("run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    manifest["files"] = {}
    for fn in ["energy_data.csv", "trust_data.csv", "run_manifest.json"]:
        try:
            manifest["files"][fn] = sha256_of(fn)
        except OSError:
            pass
    with open("run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n[5/5] Sensitivity Analysis & Plots ...")
    lam_vals = [0.5, 1.0, 1.5, 2.0, 3.0]
    glo_vals = [0.3, 0.4, 0.5]
    f1_grid = np.zeros((len(lam_vals), len(glo_vals)))

    paths = []
    for _ in range(100):
        dev = devs[0]
        mu, inv_cov, d50, d99 = bl[dev['type']]
        atk = (_ % 2 == 0)
        dks = []
        for s in range(TOTAL_S):
            a_type = 'replay' if (atk and s >= ATK_S) else None
            ctx = generate_context(dev, s, a_type)
            dks.append(mahalanobis(ctx, mu, inv_cov))
        paths.append((atk, dks, d50, d99))

    for i, l in enumerate(lam_vals):
        for j, g in enumerate(glo_vals):
            tp = tn = fp = fn = 0
            for atk, dks, d50, d99 in paths:
                tr = 1.0
                det = False
                fiso = False
                for s, dk in enumerate(dks):
                    tr = trust(dk, tr, d50, d99, lam=l)
                    if tr <= g and not det:
                        det = True
                        if s < ATK_S:
                            fiso = True
                if atk:
                    if det and not fiso:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if det:
                        fp += 1
                    else:
                        tn += 1
            dr = tp / (tp + fn) if (tp + fn) else 0
            pr = tp / (tp + fp) if (tp + fp) else 0
            f1 = 2 * pr * dr / (pr + dr) if (pr + dr) else 0
            f1_grid[i, j] = f1

    if not args.no_plots:
        plt.figure(figsize=(8, 6))
        plt.imshow(f1_grid, cmap='viridis', aspect='auto', origin='lower')
        plt.colorbar(label='F1-Score')
        plt.xticks(range(len(glo_vals)), glo_vals)
        plt.yticks(range(len(lam_vals)), lam_vals)
        plt.xlabel('Isolation Threshold (G_lo)')
        plt.ylabel('Sensitivity (lambda)')
        plt.title('Sensitivity Analysis: F1-Score')
        for i in range(len(lam_vals)):
            for j in range(len(glo_vals)):
                plt.text(j, i, f"{f1_grid[i, j]:.2f}", ha='center', va='center',
                         color='w' if f1_grid[i, j] < 0.8 else 'k')
        plt.tight_layout()
        plt.savefig('figure_sensitivity.png', dpi=300)

        plt.figure(figsize=(8, 6))
        for atk_name in ['replay', 'masquerade', 'fdi']:
            scores = 1.0 - np.array(results[atk_name]['scores'])
            labels = results[atk_name]['labels']
            fpr, tpr, roc_auc = _roc_fpr_tpr(scores, labels)
            results[atk_name]['auc'] = roc_auc
            print(f"  {atk_name.upper()} AUC: {roc_auc:.3f}")
            plt.plot(fpr, tpr, lw=2,
                     label=f'{atk_name.capitalize()} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.grid(True, ls='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('figure_roc.png', dpi=300)

        plt.figure(figsize=(10, 6))
        ts = [r['Time'] for r in scen_data['replay']]
        plt.plot(ts, [r['Trust'] for r in scen_data['replay']],
                 'g-', label='Replay', lw=2)
        plt.plot(ts, [r['Trust'] for r in scen_data['masquerade']],
                 'm--', label='Masquerade', lw=2)
        plt.plot(ts, [r['Trust'] for r in scen_data['fdi']],
                 'b:', label='FDI Slow-Drift', lw=2)
        plt.axvline(x=ATK_S, color='r', ls='--', label='Attack Start')
        plt.axhline(y=G_LO, color='orange', ls=':', label=f'Threshold ({G_LO})')
        plt.title('Trust Score Evolution Under Various Attacks')
        plt.xlabel('Time (s)')
        plt.ylabel('Theta(t)')
        plt.grid(True, ls='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('figure2_trust.png', dpi=300)

        # Divergence companion plot (paper Figure divergence)
        plt.figure(figsize=(10, 6))
        plt.plot(ts, [r['Div'] for r in scen_data['replay']], 'r-', lw=1.5)
        plt.axvline(x=ATK_S, color='k', ls='--', label='Attack Start')
        plt.title('Mahalanobis Divergence Under Replay Attack')
        plt.xlabel('Time (s)')
        plt.ylabel('D_k (normalized)')
        plt.grid(True, ls='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('figure_divergence.png', dpi=300)

        tm = [r['Time_Minute'] for r in re]
        de = [r['DTLS_Energy_J'] for r in re]
        ce = [r['CADT_Energy_J'] for r in re]
        plt.figure(figsize=(10, 6))
        plt.plot(tm, de, 'r-', label='Baseline (DTLS)', lw=1.5)
        plt.plot(tm, ce, 'b-', label='Proposed (CADT)', lw=2)
        plt.title('Energy Consumption Profile', fontsize=14)
        plt.xlabel('Time (Minutes)')
        plt.ylabel('Energy (Joules)')
        plt.grid(True, ls='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('figure1_energy.png', dpi=300)
        print("\n  -> Plots generated")

    print("\n" + "=" * 60 + "\n  DONE\n" + "=" * 60)


if __name__ == '__main__':
    run(parse_args())
