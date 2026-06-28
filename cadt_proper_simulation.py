"""
CADT Framework — Proper Physics-Based Simulation 
Implements full 3D Mahalanobis distance context fusion, 3 attack scenarios,
and comprehensive Monte Carlo evaluation with Sensitivity/ROC analysis.
"""
import numpy as np, csv, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_curve, auc
import scipy.stats as stats
np.random.seed(42)

# ===== Config =====
N_DEV = 500
SIM_MIN = 100
TOTAL_S = 250
ATK_S = 100
MC = 500

V = 3.0; I_TX = 9.1e-3; I_HSPIKE = 45e-3
TX_S = 25.0; HS_S = 43.0

ALP = 0.8; LAM = 1.5; G_HI = 0.85; G_LO = 0.40
BATT_J = 2000 * 3.0 * 3.6
SENSOR_PRECISION = 0.005

# ===== Context Generation =====
# C_env = [RSSI (dBm), Temp (C), EMI]
# C_ops = [V_batt (V), inter_arrival (s), seq_drift]
# C_phy = [Physics residual |obs-pred| / (base*precision)]
# True 3D vector length = 7 features

def make_devs(n):
    devs = []; nv, np_ = int(n*0.4), int(n*0.3)
    for i in range(n):
        if i < nv: t='vib'; b=np.random.uniform(5,15)
        elif i < nv+np_: t='proc'; b=np.random.uniform(50,150)
        else: t='act'; b=np.random.uniform(0,100)
        devs.append(dict(id=i,type=t,base=b))
    return devs

def physics(dev, t):
    if dev['type'] == 'vib': return dev['base'] + np.sin(t*0.02)*0.5
    elif dev['type'] == 'proc': return dev['base'] + np.sin(t*0.005)*2.0
    return dev['base']

def sensor_noise(val): return val + np.random.normal(0, abs(val)*0.001)

def generate_context(dev, t, atk_type=None):
    """
    Returns [RSSI, Temp, EMI, V_batt, IA, SeqDrift, PhyRes]
    atk_type: None (normal), 'replay', 'masquerade', 'fdi'
    """
    pred = physics(dev, t)
    
    # Normal behavior
    rssi = np.random.normal(-60, 2)
    temp = np.random.normal(25, 0.5)
    emi = np.random.normal(0.1, 0.05)
    vbatt = 3.0 - (t / 10000) + np.random.normal(0, 0.01)
    ia = np.random.normal(1.0, 0.05)
    seq = 0 # 0 drift
    obs = sensor_noise(pred)
    
    if atk_type == 'replay':
        # Replays old physical value (shifted by small amount from current), env/ops normal
        obs = sensor_noise(pred * 0.99795)
    elif atk_type == 'masquerade':
        # Spoofs physical value perfectly, but from different location/radio (RSSI jump)
        rssi = np.random.normal(-75, 2)
    elif atk_type == 'fdi':
        # Slow drift injection: drift depends on time since attack
        drift = 1.0 - 0.0005 * (t - ATK_S)
        obs = sensor_noise(pred * drift)
        
    phy_res = abs(obs - pred) / (max(abs(dev['base']), 1e-6) * SENSOR_PRECISION)
    return np.array([rssi, temp, emi, vbatt, ia, seq, phy_res])

def compute_baselines(devs):
    bl = {}
    by = {}
    for d in devs: by.setdefault(d['type'], []).append(d)
    
    for tn, grp in by.items():
        C_matrix = []
        # Sample context for the group
        for d in grp[:10]: # Use subset for speed
            for t in range(500):
                C_matrix.append(generate_context(d, t, None))
        C_matrix = np.array(C_matrix)
        mu = np.mean(C_matrix, axis=0)
        # Add small ridge to covariance to avoid singularity
        cov = np.cov(C_matrix, rowvar=False) + np.eye(7)*1e-4
        inv_cov = np.linalg.inv(cov)
        
        # Calculate D_99
        dists = [mahalanobis(c, mu, inv_cov) for c in C_matrix]
        d_50 = np.percentile(dists, 50)
        d_99 = np.percentile(dists, 99)
        bl[tn] = (mu, inv_cov, d_50, d_99)
        print(f"  {tn}: D_50={d_50:.4f}, D_99={d_99:.4f}")
    return bl

# Trust Eq
def trust(dk, prev, d_50, d_99, alp=ALP, lam=LAM):
    dn = max(0.0, (dk - d_50) / max(d_99 - d_50, 1e-6))
    return max(0.0, min(1.0, alp*prev + (1-alp)*np.exp(-lam*dn)))

def run():
    print("="*60)
    print("  CADT PROPER PHYSICS-BASED SIMULATION (Q1 VERSION)")
    print("="*60)
    devs = make_devs(N_DEV)
    print(f"Devices: {N_DEV}")
    
    print("\n[1/5] Baselines ...")
    bl = compute_baselines(devs)
    
    print("\n[2/5] Energy ...")
    re = []
    for t in range(SIM_MIN):
        hs = (t%10==0 and t>0)
        de = V*I_HSPIKE*HS_S + V*I_TX*max(0, TX_S-HS_S) if hs else V*I_TX*TX_S
        ce = V*I_TX*TX_S
        re.append(dict(Time_Minute=t, DTLS_Energy_J=round(de,4), CADT_Energy_J=round(ce,4)))
    
    ad = np.mean([r['DTLS_Energy_J'] for r in re])
    ac = np.mean([r['CADT_Energy_J'] for r in re])
    bd = BATT_J/(ad*60*24); bc = BATT_J/(ac*60*24)
    print(f"  Avg DTLS: {ad:.4f} J/min, CADT: {ac:.4f} J/min, Reduction: {(1-ac/ad)*100:.1f}%")
    
    print("\n[3/5] Single Scenarios (Replay, Masq, FDI) ...")
    dev = devs[0]
    mu, inv_cov, d50, d99 = bl[dev['type']]
    
    scen_data = {}
    for atk_name in ['replay', 'masquerade', 'fdi']:
        tr = 1.0; rt = []
        for s in range(TOTAL_S):
            a_type = atk_name if s >= ATK_S else None
            ctx = generate_context(dev, s, a_type)
            dk = mahalanobis(ctx, mu, inv_cov)
            tr = trust(dk, tr, d50, d99)
            rt.append(dict(Time=s, Trust=tr, Div=dk))
        scen_data[atk_name] = rt
        iso = [r['Time'] for r in rt if r['Trust'] <= G_LO]
        print(f"  {atk_name.upper()} Isolated at: {iso[0] if iso else 'Never'}s")

    print(f"\n[4/5] Monte Carlo ({MC} trials) ...")
    results = {}
    for atk_name in ['replay', 'masquerade', 'fdi']:
        tp=tn=fp=fn=0; rts=[]
        scores_for_roc = []
        labels_for_roc = []
        for tl in range(MC):
            dev=np.random.choice(devs); mu, inv_cov, d50, d99 = bl[dev['type']]
            tr=1.0; atk=(tl%2==0); det=False; fiso=False
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
                    if s < ATK_S: fiso = True
                    elif atk: rts.append(s - ATK_S)
                    
            if atk:
                if det and not fiso: tp+=1
                else: fn+=1
                scores_for_roc.append(min_tr_after_atk)
                labels_for_roc.append(1)
            else:
                if det: fp+=1
                else: tn+=1
                scores_for_roc.append(min_tr_normal) # take min trust as score for normal
                labels_for_roc.append(0)
                
        dr = tp/(tp+fn)*100 if (tp+fn) else 0
        ci = 1.96 * np.sqrt( (dr/100)*(1-dr/100) / (tp+fn) ) * 100 if (tp+fn) else 0
        far = fp/(fp+tn)*100 if (fp+tn) else 0
        ar = np.mean(rts) if rts else 0
        print(f"  {atk_name.upper()}: Det {dr:.1f}% (±{ci:.1f}%), FAR {far:.1f}%, Resp {ar:.1f}s")
        results[atk_name] = {'dr':dr, 'ci':ci, 'far':far, 'ar':ar, 'scores':scores_for_roc, 'labels':labels_for_roc}

    print("\n[5/5] Sensitivity Analysis & Plots ...")
    # Sweep LAM and G_LO for Replay
    lam_vals = [0.5, 1.0, 1.5, 2.0, 3.0]
    glo_vals = [0.3, 0.4, 0.5]
    f1_grid = np.zeros((len(lam_vals), len(glo_vals)))
    
    # We can approximate sensitivity by reusing one trial's Mahalanobis distances
    # Generate 100 paths
    paths = []
    for _ in range(100):
        dev = devs[0]; mu, inv_cov, d50, d99 = bl[dev['type']]
        atk = (_ % 2 == 0)
        dks = []
        for s in range(TOTAL_S):
            a_type = 'replay' if (atk and s >= ATK_S) else None
            ctx = generate_context(dev, s, a_type)
            dks.append( mahalanobis(ctx, mu, inv_cov) )
        paths.append((atk, dks, d50, d99))
        
    for i, l in enumerate(lam_vals):
        for j, g in enumerate(glo_vals):
            tp=tn=fp=fn=0
            for atk, dks, d50, d99 in paths:
                tr = 1.0; det = False; fiso = False
                for s, dk in enumerate(dks):
                    tr = trust(dk, tr, d50, d99, lam=l)
                    if tr <= g:
                        det = True
                        if s < ATK_S: fiso = True
                        break
                if atk:
                    if det and not fiso: tp+=1
                    else: fn+=1
                else:
                    if det: fp+=1
                    else: tn+=1
            dr = tp/(tp+fn) if (tp+fn) else 0
            pr = tp/(tp+fp) if (tp+fp) else 0
            f1 = 2*pr*dr/(pr+dr) if (pr+dr) else 0
            f1_grid[i, j] = f1

    plt.figure(figsize=(8,6))
    plt.imshow(f1_grid, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label='F1-Score')
    plt.xticks(range(len(glo_vals)), glo_vals)
    plt.yticks(range(len(lam_vals)), lam_vals)
    plt.xlabel('Isolation Threshold (Γ_low)')
    plt.ylabel('Sensitivity (λ)')
    plt.title('Sensitivity Analysis: F1-Score')
    for i in range(len(lam_vals)):
        for j in range(len(glo_vals)):
            plt.text(j, i, f"{f1_grid[i,j]:.2f}", ha='center', va='center', color='w' if f1_grid[i,j]<0.8 else 'k')
    plt.tight_layout(); plt.savefig('figure_sensitivity.png', dpi=300)

    # ROC Curve
    plt.figure(figsize=(8,6))
    # For ROC, model outputs trust score. Lower score = more likely attack.
    # So we use 1 - min_trust as the anomaly score.
    for atk_name in ['replay', 'masquerade', 'fdi']:
        scores = 1.0 - np.array(results[atk_name]['scores'])
        labels = results[atk_name]['labels']
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{atk_name.capitalize()} (AUC = {roc_auc:.3f})')
    plt.plot([0,1],[0,1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout(); plt.savefig('figure_roc.png', dpi=300)

    # Trust plot (3 scenarios)
    plt.figure(figsize=(10,6))
    ts = [r['Time'] for r in scen_data['replay']]
    plt.plot(ts, [r['Trust'] for r in scen_data['replay']], 'g-', label='Replay', lw=2)
    plt.plot(ts, [r['Trust'] for r in scen_data['masquerade']], 'm--', label='Masquerade', lw=2)
    plt.plot(ts, [r['Trust'] for r in scen_data['fdi']], 'b:', label='FDI Slow-Drift', lw=2)
    plt.axvline(x=ATK_S, color='r', ls='--', label='Attack Start')
    plt.axhline(y=G_LO, color='orange', ls=':', label=f'Threshold ({G_LO})')
    plt.title('Trust Score Evolution Under Various Attacks')
    plt.xlabel('Time (s)'); plt.ylabel('Θ(t)'); plt.grid(True, ls='--', alpha=0.7)
    plt.legend(); plt.tight_layout(); plt.savefig('figure2_trust.png', dpi=300)

    # Other plots
    tm = [r['Time_Minute'] for r in re]; de = [r['DTLS_Energy_J'] for r in re]; ce = [r['CADT_Energy_J'] for r in re]
    plt.figure(figsize=(10,6))
    plt.plot(tm, de, 'r-', label='Baseline (DTLS)', lw=1.5)
    plt.plot(tm, ce, 'b-', label='Proposed (CADT)', lw=2)
    plt.title('Energy Consumption Profile', fontsize=14)
    plt.xlabel('Time (Minutes)'); plt.ylabel('Energy (Joules)')
    plt.grid(True, ls='--', alpha=0.7); plt.legend(); plt.tight_layout(); plt.savefig('figure1_energy.png', dpi=300)

    print("\n  -> Plots generated")
    print("\n"+"="*60+"\n  DONE\n"+"="*60)
    
if __name__=='__main__':
    run()
