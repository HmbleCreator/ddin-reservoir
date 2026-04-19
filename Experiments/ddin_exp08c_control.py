"""
DDIN v8c — Control Run: Zero-Training Baseline
==============================================

Scientific Control for Experiment 08b.

Goal:
  Determine how much of the ARI=0.906 in v8b came from 
  homeostatic plasticity vs. pure random initialization.

Setup:
  - Exact same architecture and initialization as v8b.
  - Weights set to 0 (mimicking the v8b collapse).
  - TRAINING DISABLED: No BCM, no Homeostasis.
  - Metrics measured on the raw, random-init leaky integrators.

Hypothesis:
  If ARI is significantly lower than 0.906, homeostatic balance
  was the key driver. If ARI is similar, the grounding was 
  already latent in the random heterogeneity of the alphas.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from scipy.signal import find_peaks

# SET FIXED SEED FOR REPRODUCIBILITY
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print("="*60)
print("DDIN v8c — CONTROL RUN (Zero training)")
print("="*60)

class HebbianLiquidSystem(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        # Initialized same as v8b but we will zero W
        self.W     = nn.Parameter(torch.randn(dim, dim) * 0.05, requires_grad=False)
        self.alpha = nn.Parameter(torch.rand(dim) * 0.3 + 0.2,  requires_grad=False)
        self.beta  = nn.Parameter(torch.rand(dim) * 0.3 + 0.05, requires_grad=False)
        self.dim   = dim

    def forward(self, x, u, dt=0.1):
        y  = torch.tanh(x @ self.W)
        dx = -self.alpha * x + y + self.beta * u
        return x + dt * dx, y

# Instantiate and ZERO THE WEIGHTS to match v8b final state
dim = 64
model = HebbianLiquidSystem(dim=dim).to(device)
model.W.data.zero_()
print("Recurrent weights W set to ZERO (mimicking v8b collapse).")
print(f"Mean Alpha: {model.alpha.mean().item():.3f}")

# ────────────────────────────────────────────────────────────────
# 2.  INPUT GENERATOR (unchanged)
# ────────────────────────────────────────────────────────────────
def generate_input(seq_len, dim, mode="sine"):
    t = torch.linspace(0, 10, seq_len).to(device)
    if mode == "sine":
        return torch.sin(t.unsqueeze(1) * torch.linspace(0.8, 1.2, dim).to(device))
    elif mode == "cosine":
        return torch.cos(t.unsqueeze(1) * torch.linspace(0.8, 1.2, dim).to(device))
    elif mode == "constant":
        return torch.linspace(0.2, 0.8, dim).to(device).unsqueeze(0).repeat(seq_len, 1)
    elif mode == "impulse":
        u = torch.zeros(seq_len, dim).to(device)
        for i in range(dim):
            s = 5 + (i % 10) * 2
            u[s:s+3, i] = 1.0
        return u
    elif mode == "noise":
        return torch.randn(seq_len, dim).to(device) * 0.3
    elif mode == "ramp":
        return (t/t.max()).unsqueeze(1) * torch.linspace(0.1, 1.0, dim).to(device)
    elif mode == "anti_sine":
        return -torch.sin(t.unsqueeze(1) * torch.linspace(0.8, 1.2, dim).to(device))
    elif mode == "burst":
        u = torch.zeros(seq_len, dim).to(device)
        for b in range(0, seq_len, 30):
            u[b:b+5, :] = 1.0
        return u
    elif mode == "slow_sine":
        return torch.sin(t.unsqueeze(1) * torch.linspace(0.2, 0.4, dim).to(device))
    elif mode == "fast_sine":
        return torch.sin(t.unsqueeze(1) * torch.linspace(2.0, 3.0, dim).to(device))
    raise ValueError(f"Unknown mode: {mode}")

# ────────────────────────────────────────────────────────────────
# 3.  TRAINING LOOP — COMPLETELY DISABLED
# ────────────────────────────────────────────────────────────────
print("\n[ZERO TRAINING] Skipping 600 epochs...")
# No epochs run in control.

# ────────────────────────────────────────────────────────────────
# 4.  TRAJECTORY COLLECTION + DHĀTU CLUSTERING
# ────────────────────────────────────────────────────────────────
CLUSTER_MODES = ["sine", "cosine", "constant", "impulse", "noise", "ramp"]
N_DHATU = 5

def collect_traj(mode, steps=300):
    x = torch.zeros(1, dim).to(device)
    u = generate_input(steps, dim, mode=mode)
    states = []
    with torch.no_grad():
        for t in range(steps):
            x_next, _ = model(x, u[t])
            states.append(x_next.cpu().numpy())
            x = x_next
    return np.array(states).squeeze()

print("\nCollecting trajectories (Control)...")
trajs_base = {m: collect_traj(m) for m in CLUSTER_MODES}
mean_acts  = {m: np.mean(np.abs(trajs_base[m]), axis=0) for m in CLUSTER_MODES}
var_acts   = {m: np.var(trajs_base[m],          axis=0) for m in CLUSTER_MODES}

pairs = [("sine","cosine"),("sine","constant"),("sine","impulse"),
         ("cosine","impulse"),("constant","noise")]
mean_diff = np.mean(np.stack(
    [np.abs(mean_acts[m1]-mean_acts[m2]) for m1,m2 in pairs], axis=0
), axis=0)

W = model.W.detach().cpu().numpy()
hub_score = np.sum(np.abs(W), axis=1) + np.sum(np.abs(W), axis=0)

def autocorr_peak(sig):
    sig = sig - np.mean(sig)
    if np.std(sig) < 1e-6: return 0.0
    ac = np.correlate(sig, sig, mode='full')[len(sig)-1:]
    ac /= (ac[0] + 1e-8)
    segs = ac[5:60]
    pks, _ = find_peaks(segs, height=0.2)
    return float(ac[pks[0]+5]) if len(pks) > 0 else 0.0

periodicity = np.array([autocorr_peak(trajs_base["sine"][:,i]) for i in range(dim)])

F = np.column_stack([
    mean_acts["sine"], mean_acts["cosine"], mean_acts["impulse"],
    var_acts["sine"], var_acts["impulse"], mean_diff, hub_score, periodicity,
])
F_norm = StandardScaler().fit_transform(F)
cluster_labels = KMeans(n_clusters=N_DHATU, random_state=42,
                        n_init=30, max_iter=500).fit_predict(F_norm)
print(f"Dhatu cluster sizes: {[int(np.sum(cluster_labels==k)) for k in range(N_DHATU)]}")

# ────────────────────────────────────────────────────────────────
# 5.  EXTENDED DHĀTU CODE
# ────────────────────────────────────────────────────────────────
def extended_code(traj, labels, n=N_DHATU):
    feats = []
    for k in range(n):
        m = labels == k
        if not m.any():
            feats.extend([0., 0., 0.])
            continue
        nt = traj[:, m]
        feats.append(float(np.mean(np.abs(nt))))
        feats.append(float(np.mean(nt)))
        feats.append(float(np.var(np.abs(nt))))
    return np.array(feats)

# ────────────────────────────────────────────────────────────────
# 6.  GROUNDING TEST
# ────────────────────────────────────────────────────────────────
ALL_MODES = ["sine", "cosine", "constant", "impulse", "noise", "ramp",
             "anti_sine", "burst", "slow_sine", "fast_sine"]
MODE_CATEGORIES = {
    "sine": "OSCILLATION", "cosine": "OSCILLATION",
    "slow_sine": "SLOW_WAVE", "fast_sine": "FAST_WAVE",
    "anti_sine": "INVERSION", "constant": "STEADY_STATE",
    "ramp": "ACCUMULATION", "impulse": "EVENT",
    "burst": "BURST", "noise": "STOCHASTIC",
}

print("\nComputing extended Dhatu codes (Control)...")
all_codes, all_labels = [], []
for mode in ALL_MODES:
    for r in range(12):
        torch.manual_seed(r * 137)
        traj = collect_traj(mode)
        all_codes.append(extended_code(traj, cluster_labels))
        all_labels.append(MODE_CATEGORIES[mode])

all_codes  = np.array(all_codes)
all_labels = np.array(all_labels)
le = LabelEncoder()
label_ids  = le.fit_transform(all_labels)
n_cats     = len(le.classes_)

scaler     = StandardScaler()
codes_norm = scaler.fit_transform(all_codes)

# Grounding check
best_ari, best_k, best_pred = -1, 7, None
for k in range(7, 13):
    pred = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(codes_norm)
    ari  = adjusted_rand_score(label_ids, pred)
    if ari > best_ari:
        best_ari, best_k, best_pred = ari, k, pred

nmi  = normalized_mutual_info_score(label_ids, best_pred)
def cluster_purity(lt, lp):
    total = 0
    for k in np.unique(lp):
        m = lp == k
        total += np.bincount(lt[m]).max()
    return total / len(lt)
pur  = cluster_purity(label_ids, best_pred)

print(f"\nGrounding test (Control, NO training):")
print(f"  ARI    : {best_ari:.3f}   (v8b BCM: 0.906)")
print(f"  NMI    : {nmi:.3f}   (v8b BCM: 0.967)")
print(f"  Purity : {pur:.3f}   (v8b BCM: 0.900)")

print("\n" + "="*60)
print("FINAL VERDICT: CONTRIBUTION OF LEARNING")
print("="*60)
diff_ari = 0.906 - best_ari
print(f"  ARI Delta: {diff_ari:+.3f}")
if diff_ari > 0.05:
    print("  RESULT: Training genuinely improved grounding structure.")
else:
    print("  RESULT: Grounding is mostly latent in random initialization.")
print("="*60)
