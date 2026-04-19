"""
DDIN v12b — BCM-Fixed Benchmark-Grounded Embedding
====================================================

Regression analysis: v12 (and v13) collapsed to ARI=0.026 because:

  1. BCM decay=2e-3 is TOO STRONG for 150-root diversity.
     With 150 roots cycled in 600 epochs, each root is visited only 4x.
     W decays at 2e-3 per step × 200 steps per root × 600 epochs
     = cumulative exponential decay driving W → 0.
     Evidence: active_syn=0 by epoch 60 (only 60/150 unique roots seen).

  2. theta_init=0.02 is too high relative to BCM activity scale.
     With weak W (decaying fast), y ≈ tanh(0) is near zero, so
     y*(y - theta) is always negative → further suppresses W updates.

  Fixes in v12b:
    - decay:      2e-3  → 2e-4   (10x reduction, matches 150-root scale)
    - theta_init: 0.02  → 0.005  (lower sliding threshold for sparse input)
    - eta_theta:  1e-4  → 5e-5   (slower sliding threshold adaptation)
    - Add W-norm floor: if ||W|| < 0.1, re-initialize sparse W component
    - eta_hom:    2e-3  → 5e-4   (gentler homeostatic damping)
    - prune_start: 300  → 450    (allow BCM to stabilize before pruning)
    - prune_thresh: 0.015 → 0.008 (finer pruning granularity)

Architecture: identical 21D input, 128D reservoir (v12 baseline).
Purpose: establish corrected v12 baseline for v13/v14 comparison.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from itertools import combinations

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print("="*65)
print("DDIN v12b -- BCM FIXED (decay=2e-4, theta_init=0.005)")
print("="*65)

IN_DIM = 21   # 16D acoustic + 5D locus one-hot (same as v12)

# ────────────────────────────────────────────────────────────────
# 1.  PHONEME VECTORS (16D from exp11/v12)
# ────────────────────────────────────────────────────────────────

PHONEME_VECTORS_16 = {
    'a'  : [0.50, 0.50, 0.0, 1.0, 0.0, 0.0, 0.50, 0.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 1.0, 0.7],
    'A'  : [0.50, 0.50, 0.0, 1.0, 0.0, 0.0, 0.50, 1.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 1.0, 0.7],
    'i'  : [0.25, 0.80, 0.0, 1.0, 0.0, 1.0, 0.0,  0.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.3, 0.7],
    'I'  : [0.25, 0.80, 0.0, 1.0, 0.0, 1.0, 0.0,  1.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.3, 0.7],
    'u'  : [0.75, 0.80, 0.0, 1.0, 0.0, 1.0, 1.0,  0.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.2, 0.7],
    'U'  : [0.75, 0.80, 0.0, 1.0, 0.0, 1.0, 1.0,  1.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.2, 0.7],
    'R'  : [0.50, 0.60, 0.0, 1.0, 0.0, 0.5, 0.0,  0.0, 1.0, 0.9,  1.0, 1.0, 0.0, 0.0, 0.5, 0.7],
    'e'  : [0.25, 0.70, 0.0, 1.0, 0.0, 0.7, 0.0,  0.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.7, 0.7],
    'o'  : [0.75, 0.70, 0.0, 1.0, 0.0, 0.7, 1.0,  0.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.7, 0.7],
    # Velar
    'k'  : [0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    'K'  : [0.0,  0.0,  1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 0.0, 0.0, 0.0],
    'g'  : [0.0,  0.0,  0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 0.0, 0.0, 0.0],
    'G'  : [0.0,  0.0,  1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 0.0, 0.0, 0.0],
    'N'  : [0.0,  0.0,  0.0, 1.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.5,  0.5, 0.5,  0.0, 0.0, 0.0, 1.0],
    'h'  : [0.0,  0.0,  0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.3,  1.0, 0.65, 0.0, 0.0, 0.0, 0.0],
    # Palatal
    'c'  : [0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.0, 0.0],
    'C'  : [0.25, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],
    'j'  : [0.25, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 1.0, 0.0, 0.0],
    'J'  : [0.25, 0.25, 1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],
    'y'  : [0.25, 0.25, 0.0, 1.0, 0.0, 0.8, 0.0,  0.0, 1.0, 0.8,  1.0, 0.85, 0.0, 1.0, 0.0, 0.4],
    'z'  : [0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.2,  1.0, 0.55, 1.0, 1.0, 0.0, 0.0],
    # Cerebral
    'T'  : [0.5,  0.5,  0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.0, 0.0],
    'Q'  : [0.5,  0.5,  1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],
    'D'  : [0.5,  0.5,  0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 1.0, 0.0, 0.0],
    'X'  : [0.5,  0.5,  1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],
    'r'  : [0.5,  0.5,  0.0, 1.0, 0.0, 0.3, 0.0,  0.0, 1.0, 0.8,  1.0, 0.75, 0.0, 1.0, 0.0, 0.4],
    'x'  : [0.5,  0.5,  0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.2,  1.0, 0.55, 1.0, 1.0, 0.0, 0.0],
    # Dental
    't'  : [0.75, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.0, 0.0],
    'H'  : [0.75, 0.75, 1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],
    'd'  : [0.75, 0.75, 0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 1.0, 0.0, 0.0],
    'W'  : [0.75, 0.75, 1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],
    'n'  : [0.75, 0.75, 0.0, 1.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.5,  0.5, 0.5,  0.0, 1.0, 0.0, 1.0],
    'l'  : [0.75, 0.7,  0.0, 1.0, 0.0, 0.3, 0.0,  0.0, 1.0, 0.8,  1.0, 0.75, 0.0, 1.0, 0.0, 0.4],
    's'  : [0.75, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.2,  1.0, 0.55, 1.0, 1.0, 0.0, 0.0],
    # Labial
    'p'  : [1.0,  1.0,  0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    'P'  : [1.0,  1.0,  1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 0.0, 0.0, 0.0],
    'b'  : [1.0,  1.0,  0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 0.0, 0.0, 0.0],
    'B'  : [1.0,  1.0,  1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 0.0, 0.0, 0.0],
    'm'  : [1.0,  1.0,  0.0, 1.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.5,  0.5, 0.5,  0.0, 0.0, 0.0, 1.0],
    'v'  : [1.0,  1.0,  0.0, 1.0, 0.0, 0.2, 1.0,  0.0, 1.0, 0.7,  1.0, 0.85, 0.0, 0.0, 0.0, 0.4],
}

LOCUS_ONEHOT = {
    'THROAT'   : [1, 0, 0, 0, 0],
    'PALATE'   : [0, 1, 0, 0, 0],
    'CEREBRAL' : [0, 0, 1, 0, 0],
    'DENTAL'   : [0, 0, 0, 1, 0],
    'LABIAL'   : [0, 0, 0, 0, 1],
}

TRANSLIT = {
    'A': 'A', 'I': 'I', 'U': 'U', 'R': 'R',
    'kh': 'K', 'gh': 'G', 'ch': 'C', 'jh': 'J',
    'Th': 'Q', 'Dh': 'X', 'th': 'H', 'dh': 'W',
    'ph': 'P', 'bh': 'B', 'sh': 'z', 'sh2': 'x',
    'ng': 'N',
}

def root_to_phon_chars(root_str):
    chars = []
    i = 0
    s = root_str
    while i < len(s):
        if i+1 < len(s) and s[i:i+2] in ('kh','gh','ch','jh','Th','Dh','th','dh','ph','bh','sh','ng'):
            dg = s[i:i+2]
            chars.append(TRANSLIT.get(dg, dg))
            i += 2
        else:
            c = s[i]
            chars.append(TRANSLIT.get(c, c))
            i += 1
    return chars

def phoneme_vec_21(ph_char, locus):
    base = list(PHONEME_VECTORS_16.get(ph_char, [0.5]*16))
    loh  = LOCUS_ONEHOT.get(locus, [0,0,0,0,0])
    return np.array(base + loh, dtype=np.float32)

# ────────────────────────────────────────────────────────────────
# 2.  LOAD BENCHMARK DATA
# ────────────────────────────────────────────────────────────────

DATA_PATH = r'c:\Users\amiku\Downloads\AI Research New Paradigm\SampleData\task1_axis_prediction.csv'
df = pd.read_csv(DATA_PATH)
print(f"\nLoaded {len(df)} roots from benchmark dataset.")
print(f"Axes: {dict(df['actual_axis'].value_counts())}")
print(f"Loci: {dict(df['locus'].value_counts())}")

VERBAL_ROOTS = []
for _, row in df.iterrows():
    root  = row['root']
    locus = row['locus']
    axis  = row['actual_axis']
    gloss = str(row['mw_gloss'])[:40]
    chars = root_to_phon_chars(root)
    VERBAL_ROOTS.append((root, chars, axis, gloss, locus))

# ────────────────────────────────────────────────────────────────
# 3.  INPUT GENERATOR
# ────────────────────────────────────────────────────────────────

def root_to_input(chars, locus, seq_len=300, noise_std=0.01):
    PHONEME_DUR    = 40
    TRANSITION_DUR = 10
    vecs = [phoneme_vec_21(c, locus) for c in chars]
    signal_parts = []
    for i, vec in enumerate(vecs):
        plateau = np.tile(vec, (PHONEME_DUR, 1))
        signal_parts.append(plateau)
        if i < len(vecs) - 1:
            alphas = np.linspace(0, 1, TRANSITION_DUR)
            trans  = np.outer(alphas, vecs[i+1]) + np.outer(1-alphas, vec)
            signal_parts.append(trans)
    signal_np = np.concatenate(signal_parts, axis=0)
    T = signal_np.shape[0]
    if T < seq_len:
        pad = np.tile(signal_np[-1:], (seq_len - T, 1))
        signal_np = np.concatenate([signal_np, pad], axis=0)
    else:
        signal_np = signal_np[:seq_len]
    signal_t = torch.FloatTensor(signal_np).to(device)
    if noise_std > 0:
        signal_t += torch.randn_like(signal_t) * noise_std
    return signal_t.clamp(-2, 2)

# ────────────────────────────────────────────────────────────────
# 4.  PCA BETA
# ────────────────────────────────────────────────────────────────

def build_pca_beta(verbal_roots, dim=128, in_dim=IN_DIM):
    all_vecs = []
    for (_, chars, _, _, locus) in verbal_roots:
        vecs = [phoneme_vec_21(c, locus) for c in chars]
        all_vecs.append(np.mean(vecs, axis=0))
    X = np.array(all_vecs)
    n_comp = min(in_dim, X.shape[0]-1, X.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    components = pca.components_
    repeats  = dim // n_comp + 1
    beta_np  = np.tile(components, (repeats, 1))[:dim]
    beta_np += np.random.randn(*beta_np.shape) * 0.05
    norms    = np.linalg.norm(beta_np, axis=1, keepdims=True) + 1e-8
    beta_np /= norms
    return torch.FloatTensor(beta_np)

# ────────────────────────────────────────────────────────────────
# 5.  DDIN MODEL — with W-norm floor guard
# ────────────────────────────────────────────────────────────────

class HebbianConvergenceSystem(nn.Module):
    def __init__(self, dim=128, beta_init=None):
        super().__init__()
        self.W     = nn.Parameter(torch.randn(dim, dim) * 0.02, requires_grad=False)
        self.alpha = nn.Parameter(torch.rand(dim) * 0.3 + 0.2,  requires_grad=False)
        if beta_init is not None:
            self.beta = nn.Parameter(beta_init.to(device), requires_grad=False)
        else:
            beta_r = torch.rand(dim, IN_DIM)
            beta_r /= beta_r.norm(dim=1, keepdim=True) + 1e-8
            self.beta = nn.Parameter(beta_r.to(device), requires_grad=False)
        self.dim = dim
        self._w_init = self.W.data.clone()   # save initial W for norm floor

    def forward(self, x, u, dt=0.1):
        y  = torch.tanh(x @ self.W)
        dx = -self.alpha * x + y + (u @ self.beta.T)
        return x + dt * dx, y

    def bcm_update(self, x, y, theta, eta=1e-4, decay=2e-4):
        with torch.no_grad():
            bcm_gate = y * (y - theta.unsqueeze(0))
            dW = (bcm_gate.T @ x) / x.shape[0]
            self.W.data += eta * dW - decay * self.W.data

    def w_norm_floor(self, floor=0.05):
        """Re-inject small noise if W collapses below floor norm."""
        with torch.no_grad():
            w_norm = self.W.data.norm()
            if w_norm < floor:
                noise = torch.randn_like(self.W.data) * 0.01
                self.W.data += noise

    def activity_prune(self, threshold=0.008):
        with torch.no_grad():
            self.W.data[torch.abs(self.W.data) < threshold] = 0.0

    def homeostatic_update(self, mean_act, eta_hom=5e-4, alpha_target=0.35):
        with torch.no_grad():
            d_alpha = eta_hom * (mean_act - alpha_target)
            self.alpha.data += d_alpha
            self.alpha.data.clamp_(0.08, 0.85)

# ────────────────────────────────────────────────────────────────
# 6.  TRAINING — v12b hyperparameters
# ────────────────────────────────────────────────────────────────

dim          = 128
EPOCHS       = 800        # +200 over v12 to allow longer stabilization
PRUNE_START  = 450        # later pruning start (BCM needs time with 150 roots)
PRUNE_EVERY  = 30
PRUNE_THRESH = 0.008      # finer threshold

# Fixed hyperparameters
BCM_DECAY    = 2e-4       # 10x lower than v12's 2e-3
ETA_BCM      = 1e-4
ETA_THETA    = 5e-5       # slower theta adaptation
THETA_INIT   = 0.005      # lower init threshold
ETA_HOM      = 5e-4       # gentler homeostatic damping
W_FLOOR      = 0.08       # re-inject noise if W norm drops below this

print(f"\nHyperparameters (v12b):")
print(f"  BCM decay  : {BCM_DECAY}  (was 2e-3)")
print(f"  theta_init : {THETA_INIT}  (was 0.02)")
print(f"  eta_theta  : {ETA_THETA}   (was 1e-4)")
print(f"  eta_hom    : {ETA_HOM}   (was 2e-3)")
print(f"  prune_start: {PRUNE_START}     (was 300)")
print(f"  prune_thresh: {PRUNE_THRESH}   (was 0.015)")
print(f"  W_floor    : {W_FLOOR}      (NEW)")

beta_pca = build_pca_beta(VERBAL_ROOTS, dim=dim, in_dim=IN_DIM)
model    = HebbianConvergenceSystem(dim=dim, beta_init=beta_pca).to(device)
theta    = torch.full((dim,), THETA_INIT).to(device)

print(f"\nTraining v12b (dim={dim}, IN_DIM={IN_DIM}, n_roots={len(VERBAL_ROOTS)}, epochs={EPOCHS})...")
for epoch in range(EPOCHS):
    ri = epoch % len(VERBAL_ROOTS)
    _, chars, _, _, locus = VERBAL_ROOTS[ri]
    u = root_to_input(chars, locus, seq_len=200)
    x = torch.zeros(1, dim).to(device)
    act_acc = torch.zeros(dim).to(device)

    with torch.no_grad():
        for t in range(200):
            x_next, y = model(x, u[t])
            model.bcm_update(x, y, theta, eta=ETA_BCM, decay=BCM_DECAY)
            theta += ETA_THETA * (y.squeeze()**2 - theta)
            act_acc += torch.abs(x_next.squeeze())
            x = x_next

    model.homeostatic_update(act_acc / 200, eta_hom=ETA_HOM, alpha_target=0.35)
    model.w_norm_floor(floor=W_FLOOR)

    if epoch >= PRUNE_START and epoch % PRUNE_EVERY == 0:
        model.activity_prune(threshold=PRUNE_THRESH)

    if epoch % 80 == 0:
        active = torch.count_nonzero(torch.abs(model.W) > 0.005).item()
        w_norm = model.W.data.norm().item()
        phase  = "BCM    " if epoch < PRUNE_START else "Pruning"
        print(f"  Epoch {epoch:3d} ({phase}) | active_syn={active:5d} | "
              f"mean_theta={theta.mean().item():.5f} | W_norm={w_norm:.4f}")

# ────────────────────────────────────────────────────────────────
# 7.  COLLECT 128D CODES
# ────────────────────────────────────────────────────────────────

def collect_mean_traj(ri, steps=200):
    _, chars, _, _, locus = VERBAL_ROOTS[ri]
    x = torch.zeros(1, dim).to(device)
    u = root_to_input(chars, locus, seq_len=steps)
    acc = torch.zeros(dim).to(device)
    with torch.no_grad():
        for t in range(steps):
            x = model(x, u[t])[0]
            acc += x.squeeze()
    return (acc / steps).cpu().numpy()

print("\nCollecting 128D mean trajectories for all 150 roots...")
all_codes, all_labels, all_roots_str, all_loci = [], [], [], []
for i, (rname, chars, axis, _, locus) in enumerate(VERBAL_ROOTS):
    code = collect_mean_traj(i)
    all_codes.append(code)
    all_labels.append(axis)
    all_roots_str.append(rname)
    all_loci.append(locus)

all_codes  = np.array(all_codes)
all_labels = np.array(all_labels)
all_loci   = np.array(all_loci)

le  = LabelEncoder()
ids = le.fit_transform(all_labels)

pca_code = PCA(n_components=50, random_state=42)
reduced  = pca_code.fit_transform(all_codes)
scaler   = StandardScaler()
norm     = scaler.fit_transform(reduced)

# ────────────────────────────────────────────────────────────────
# 8.  ARI SWEEP
# ────────────────────────────────────────────────────────────────

def best_ari_for(labels_arr, norm_codes):
    le_tmp = LabelEncoder()
    ids_tmp = le_tmp.fit_transform(labels_arr)
    n_uniq   = len(le_tmp.classes_)
    best, bk = -1, n_uniq
    for k in range(n_uniq, n_uniq+4):
        pred = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(norm_codes)
        a    = adjusted_rand_score(ids_tmp, pred)
        if a > best:
            best, bk = a, k
    return best, bk

best_ari_axis, best_k = best_ari_for(all_labels, norm)
best_pred = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(norm)
nmi       = normalized_mutual_info_score(ids, best_pred)
best_ari_locus, _ = best_ari_for(all_loci, norm)

def sep_ratio(codes, labels):
    intra, inter = [], []
    cats = np.unique(labels)
    for cat in cats:
        sub = codes[labels==cat]
        for i, j in combinations(range(len(sub)), 2):
            intra.append(np.linalg.norm(sub[i] - sub[j]))
    for c1, c2 in combinations(cats, 2):
        for a in codes[labels==c1]:
            for b in codes[labels==c2]:
                inter.append(np.linalg.norm(a - b))
    return np.mean(inter) / (np.mean(intra) + 1e-8)

ratio = sep_ratio(norm, all_labels)

print(f"\n{'='*60}")
print(f"GROUNDING TEST — v12b (21D: 16D acoustic + 5D locus, BCM-Fixed)")
print(f"{'='*60}")
print(f"  ARI (axis)  : {best_ari_axis:.3f}   (v12 collapse: 0.026, v10b: 0.182)")
print(f"  ARI (locus) : {best_ari_locus:.3f}   (diagnostic)")
print(f"  NMI         : {nmi:.3f}")
print(f"  Sep. ratio  : {ratio:.2f}x")
print(f"  Chance      : {1/len(le.classes_):.3f}  ({len(le.classes_)} categories)")

print(f"\nPer-axis cluster purity:")
for ax in le.classes_:
    m = all_labels == ax
    pred_for_ax = best_pred[m]
    dominant = np.bincount(pred_for_ax).argmax()
    purity = np.sum(pred_for_ax == dominant) / m.sum()
    print(f"  {ax}: n={m.sum():3d}  purity={purity:.2f}")

# ────────────────────────────────────────────────────────────────
# 9.  SAVE EMBEDDING FOR v14 MULTI-TASK EVAL
# ────────────────────────────────────────────────────────────────

np.save('ddin_v12b_norm_codes.npy',  norm)
np.save('ddin_v12b_raw_codes.npy',   all_codes)
np.save('ddin_v12b_labels_axis.npy', all_labels.astype(str))
np.save('ddin_v12b_labels_locus.npy',all_loci.astype(str))
np.save('ddin_v12b_roots.npy',       np.array(all_roots_str))
print("\nSaved v12b embedding arrays for v14 multi-task evaluator.")

# ────────────────────────────────────────────────────────────────
# 10.  PLOT
# ────────────────────────────────────────────────────────────────

pca2 = PCA(n_components=2)
low  = pca2.fit_transform(norm)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f"DDIN v12b: BCM-Fixed Benchmark (150 roots, 21D)\n"
    f"ARI_axis={best_ari_axis:.3f} | ARI_locus={best_ari_locus:.3f} | NMI={nmi:.3f} | Sep={ratio:.2f}x\n"
    f"Fix: decay=2e-4, theta_init=0.005, W-floor guard",
    fontsize=10
)

ax = axes[0]
ax.set_title("By Phenomenological Axis")
colors = {'EXP':'#ff6b6b','TRN':'#ffd93d','MOT':'#6bcb77','SEP':'#4d96ff','CNT':'#c77dff'}
for axis in np.unique(all_labels):
    m = all_labels == axis
    ax.scatter(low[m,0], low[m,1], label=axis, alpha=0.65, s=30,
               color=colors.get(axis,'#888888'))
ax.legend(fontsize=9)

ax2 = axes[1]
ax2.set_title("By Locus / Sthana")
locus_colors = {'THROAT':'#ff4444','PALATE':'#ffaa00','CEREBRAL':'#44cc44',
                'DENTAL':'#4444ff','LABIAL':'#ff44ff'}
for loc in ['THROAT','PALATE','CEREBRAL','DENTAL','LABIAL']:
    m = all_loci == loc
    ax2.scatter(low[m,0], low[m,1], label=loc, alpha=0.65, s=30,
                color=locus_colors[loc])
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig('ddin_v12b_bcm_fixed.png', dpi=150, bbox_inches='tight')
print("Saved: ddin_v12b_bcm_fixed.png")

print(f"\n{'='*60}")
print(f"ARI Progression:")
print(f"  v10b (10D, 30 roots) : 0.182")
print(f"  v12  (21D, BCM collapse) : 0.026  [BCM decay too aggressive]")
print(f"  v12b (21D, BCM fixed)    : {best_ari_axis:.3f}  [this run]")
print(f"{'='*60}")
