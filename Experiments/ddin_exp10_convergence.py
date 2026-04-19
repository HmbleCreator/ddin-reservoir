"""
DDIN v10b — Convergence (Fixed BCM + PCA-beta)
===============================================

Fixes applied over v10a:
  Fix 1 — BCM theta saturation:
    theta_init  : 0.1   → 0.02  (start below natural activity level)
    eta_theta   : 5e-4  → 1e-4  (slow the sliding so it doesn't saturate)

  Fix 2 — No pruning / weak decay:
    decay       : 5e-4  → 2e-3  (match v8b which achieved 100% sparsity)
    Explicit pruning pass every 30 epochs after epoch 300

  Fix 3 — Random beta washes out articulatory geometry:
    beta        : random normalised rows → PCA-projection from real
                  phoneme vectors, so the 10D manifold is expanded into
                  128D while preserving inter-phoneme distances.
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
from itertools import combinations

# SET FIXED SEED
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print("="*60)
print("DDIN v10 — CONVERGENCE CONSOLIDATION")
print("="*60)

# ────────────────────────────────────────────────────────────────
# 1.  PHONEME & ROOT DATA (Same as v9/v9b)
# ────────────────────────────────────────────────────────────────

PHONEME_VECTORS = {
    'a'  : [0.50, 0.50, 0.0, 1.0, 0.0,  0.0, 0.50, 0.0, 1.0, 1.0],
    'ā'  : [0.50, 0.50, 0.0, 1.0, 0.0,  0.0, 0.50, 1.0, 1.0, 1.0],
    'i'  : [0.25, 0.80, 0.0, 1.0, 0.0,  1.0, 0.0,  0.0, 1.0, 1.0],
    'ī'  : [0.25, 0.80, 0.0, 1.0, 0.0,  1.0, 0.0,  1.0, 1.0, 1.0],
    'u'  : [0.75, 0.80, 0.0, 1.0, 0.0,  1.0, 1.0,  0.0, 1.0, 1.0],
    'ū'  : [0.75, 0.80, 0.0, 1.0, 0.0,  1.0, 1.0,  1.0, 1.0, 1.0],
    'ṛ'  : [0.50, 0.60, 0.0, 1.0, 0.0,  0.5, 0.0,  0.0, 1.0, 0.9],
    'e'  : [0.25, 0.70, 0.0, 1.0, 0.0,  0.7, 0.0,  0.0, 1.0, 1.0],
    'ai' : [0.25, 0.70, 0.0, 1.0, 0.0,  0.5, 0.0,  0.5, 1.0, 1.0],
    'o'  : [0.75, 0.70, 0.0, 1.0, 0.0,  0.7, 1.0,  0.0, 1.0, 1.0],
    'au' : [0.75, 0.70, 0.0, 1.0, 0.0,  0.5, 1.0,  0.5, 1.0, 1.0],
    'k'  : [0.0,  0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0],
    'kh' : [0.0,  0.0,  1.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0],
    'g'  : [0.0,  0.0,  0.0, 1.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.1],
    'gh' : [0.0,  0.0,  1.0, 1.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.1],
    'ṅ'  : [0.0,  0.0,  0.0, 1.0, 1.0,  0.0, 0.0,  0.0, 0.0, 0.5],
    'c'  : [0.25, 0.25, 0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0],
    'ch' : [0.25, 0.25, 1.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0],
    'j'  : [0.25, 0.25, 0.0, 1.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.1],
    'jh' : [0.25, 0.25, 1.0, 1.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.1],
    'ñ'  : [0.25, 0.25, 0.0, 1.0, 1.0,  0.0, 0.0,  0.0, 0.0, 0.5],
    'ṭ'  : [0.5,  0.5,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0],
    'ṭh' : [0.5,  0.5,  1.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0],
    'ḍ'  : [0.5,  0.5,  0.0, 1.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.1],
    'ḍh' : [0.5,  0.5,  1.0, 1.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.1],
    'ṇ'  : [0.5,  0.5,  0.0, 1.0, 1.0,  0.0, 0.0,  0.0, 0.0, 0.5],
    't'  : [0.75, 0.75, 0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0],
    'th' : [0.75, 0.75, 1.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0],
    'd'  : [0.75, 0.75, 0.0, 1.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.1],
    'dh' : [0.75, 0.75, 1.0, 1.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.1],
    'n'  : [0.75, 0.75, 0.0, 1.0, 1.0,  0.0, 0.0,  0.0, 0.0, 0.5],
    'p'  : [1.0,  1.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0],
    'ph' : [1.0,  1.0,  1.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0],
    'b'  : [1.0,  1.0,  0.0, 1.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.1],
    'bh' : [1.0,  1.0,  1.0, 1.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.1],
    'm'  : [1.0,  1.0,  0.0, 1.0, 1.0,  0.0, 0.0,  0.0, 0.0, 0.5],
    'y'  : [0.25, 0.25, 0.0, 1.0, 0.0,  0.8, 0.0,  0.0, 1.0, 0.8],
    'r'  : [0.5,  0.5,  0.0, 1.0, 0.0,  0.3, 0.0,  0.0, 1.0, 0.8],
    'l'  : [0.75, 0.7,  0.0, 1.0, 0.0,  0.3, 0.0,  0.0, 1.0, 0.8],
    'v'  : [1.0,  1.0,  0.0, 1.0, 0.0,  0.2, 1.0,  0.0, 1.0, 0.7],
    'ś'  : [0.25, 0.25, 0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.2],
    'ṣ'  : [0.5,  0.5,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.2],
    's'  : [0.75, 0.75, 0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.2],
    'h'  : [0.0,  0.0,  0.0, 1.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.3],
}

VERBAL_ROOTS = [
    ("gam",   ['g', 'a', 'm'],          "MOTION",     "to go"),
    ("dhāv",  ['dh', 'ā', 'v'],         "MOTION",     "to flow/run"),
    ("car",   ['c', 'a', 'r'],          "MOTION",     "to move/wander"),
    ("yā",    ['y', 'ā'],               "MOTION",     "to go away"),
    ("pat",   ['p', 'a', 't'],          "MOTION",     "to fly/fall"),
    ("vah",   ['v', 'a', 'h'],          "MOTION",     "to carry/flow"),
    ("sthā",  ['s', 'th', 'ā'],         "STABILITY",  "to stand"),
    ("vas",   ['v', 'a', 's'],          "STABILITY",  "to dwell"),
    ("ram",   ['r', 'a', 'm'],          "STABILITY",  "to rest/delight"),
    ("śī",    ['ś', 'ī'],               "STABILITY",  "to lie down"),
    ("dṛś",   ['d', 'ṛ', 'ś'],          "PERCEPTION", "to see"),
    ("śru",   ['ś', 'r', 'u'],          "PERCEPTION", "to hear"),
    ("vid",   ['v', 'i', 'd'],          "PERCEPTION", "to know/perceive"),
    ("spṛś",  ['s', 'p', 'ṛ', 'ś'],     "PERCEPTION", "to touch"),
    ("vac",   ['v', 'a', 'c'],          "SPEECH",     "to speak"),
    ("brū",   ['b', 'r', 'ū'],          "SPEECH",     "to say"),
    ("śaṃs",  ['ś', 'a', 'm', 's'],     "SPEECH",     "to praise"),
    ("jap",   ['j', 'a', 'p'],          "SPEECH",     "to whisper/chant"),
    ("bhū",   ['bh', 'ū'],              "EXISTENCE",  "to become/be"),
    ("jan",   ['j', 'a', 'n'],          "EXISTENCE",  "to be born"),
    ("mṛ",    ['m', 'ṛ'],               "EXISTENCE",  "to die"),
    ("as",    ['a', 's'],               "EXISTENCE",  "to be/exist"),
    ("dā",    ['d', 'ā'],               "EXCHANGE",   "to give"),
    ("krī",   ['k', 'r', 'ī'],          "EXCHANGE",   "to buy"),
    ("ji",    ['j', 'i'],               "EXCHANGE",   "to win/conquer"),
    ("nī",    ['n', 'ī'],               "EXCHANGE",   "to lead"),
    ("kṛ",    ['k', 'ṛ'],               "ACTION",     "to do/make"),
    ("han",   ['h', 'a', 'n'],          "ACTION",     "to strike"),
    ("tap",   ['t', 'a', 'p'],          "ACTION",     "to heat/ascetic practice"),
    ("yuj",   ['y', 'u', 'j'],          "ACTION",     "to yoke/join"),
]

# ────────────────────────────────────────────────────────────────
# 2.  INPUT GENERATOR (Direct 10D Drive)
# ────────────────────────────────────────────────────────────────

def phoneme_vec(phoneme):
    if phoneme in PHONEME_VECTORS:
        return np.array(PHONEME_VECTORS[phoneme], dtype=np.float32)
    return np.array([0.5]*10, dtype=np.float32)

def root_to_input(phoneme_list, seq_len=300, noise_std=0.01):
    PHONEME_DUR = 40
    TRANSITION_DUR = 10
    vecs = [phoneme_vec(p) for p in phoneme_list]
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
# 3.  DDIN MODEL (128 neurons, BCM-enabled)
# ────────────────────────────────────────────────────────────────

def build_pca_beta(verbal_roots, phoneme_vectors_dict, dim=128, in_dim=10):
    """
    FIX 3: PCA-initialised input projection.
    Compute the mean phoneme vector for every root, stack them,
    run PCA to get 'in_dim' principal directions, then tile/expand
    to fill 'dim' rows so every reservoir neuron gets a unique
    linear combination of the articulatory axes.
    This preserves inter-root distances (unlike a random matrix).
    """
    all_vecs = []
    for _, phonemes, _, _ in verbal_roots:
        vecs = [np.array(phoneme_vectors_dict.get(p, [0.5]*10), dtype=np.float32) for p in phonemes]
        all_vecs.append(np.mean(vecs, axis=0))
    X = np.array(all_vecs)          # (30, 10)

    # PCA: gives us up to 10 orthogonal directions in phoneme space
    pca = PCA(n_components=in_dim)
    pca.fit(X)
    components = pca.components_    # (10, 10) — rows are principal axes

    # Expand to dim rows by cycling + adding small noise for heterogeneity
    repeats   = dim // in_dim + 1
    beta_np   = np.tile(components, (repeats, 1))[:dim]  # (dim, 10)
    beta_np  += np.random.randn(*beta_np.shape) * 0.05   # small noise
    # Normalise rows so each neuron has unit input gain
    norms     = np.linalg.norm(beta_np, axis=1, keepdims=True) + 1e-8
    beta_np  /= norms
    return torch.FloatTensor(beta_np)


class HebbianConvergenceSystem(nn.Module):
    def __init__(self, dim=128, beta_init=None):
        super().__init__()
        self.W     = nn.Parameter(torch.randn(dim, dim) * 0.02, requires_grad=False)
        self.alpha = nn.Parameter(torch.rand(dim) * 0.3 + 0.2,  requires_grad=False)
        # FIX 3: use PCA-initialised beta if provided
        if beta_init is not None:
            self.beta = nn.Parameter(beta_init.to(device), requires_grad=False)
        else:
            beta_rand = torch.rand(dim, 10)
            beta_rand /= beta_rand.norm(dim=1, keepdim=True) + 1e-8
            self.beta = nn.Parameter(beta_rand.to(device), requires_grad=False)
        self.dim = dim

    def forward(self, x, u, dt=0.1):
        y  = torch.tanh(x @ self.W)
        dx = -self.alpha * x + y + (u @ self.beta.T)
        return x + dt * dx, y

    def bcm_update(self, x, y, theta, eta=1e-4, decay=2e-3):
        """FIX 2: decay restored to 2e-3 (same as working v8b)."""
        with torch.no_grad():
            bcm_gate = y * (y - theta.unsqueeze(0))
            dW = (bcm_gate.T @ x) / x.shape[0]
            self.W.data += eta * dW - decay * self.W.data

    def activity_prune(self, threshold=0.015):
        """FIX 2: explicit pruning brought back from v8b."""
        with torch.no_grad():
            self.W.data[torch.abs(self.W.data) < threshold] = 0.0

    def homeostatic_update(self, mean_act, eta_hom=1e-3, alpha_target=0.35):
        with torch.no_grad():
            d_alpha = eta_hom * (mean_act - alpha_target)
            self.alpha.data += d_alpha
            self.alpha.data.clamp_(0.08, 0.85)

# ────────────────────────────────────────────────────────────────
# 4.  TRAINING (BCM + Homeostasis, no backprop)
# ────────────────────────────────────────────────────────────────

dim          = 128
EPOCHS       = 600
PRUNE_START  = 300
PRUNE_EVERY  = 30
PRUNE_THRESH = 0.015

# FIX 3: build PCA beta before model init
beta_pca = build_pca_beta(VERBAL_ROOTS, PHONEME_VECTORS, dim=dim, in_dim=10)
model    = HebbianConvergenceSystem(dim=dim, beta_init=beta_pca).to(device)

# FIX 1: theta starts LOW (0.02) and slides SLOWLY (eta=1e-4)
theta    = torch.full((dim,), 0.02).to(device)

print(f"Training v10b (dim={dim}) — Fixed BCM + PCA-beta...")
print(f"  theta_init=0.02  eta_theta=1e-4  decay=2e-3  prune_start={PRUNE_START}")
for epoch in range(EPOCHS):
    root_idx = epoch % len(VERBAL_ROOTS)
    _, phonemes, _, _ = VERBAL_ROOTS[root_idx]
    u = root_to_input(phonemes, seq_len=200)
    x = torch.zeros(1, dim).to(device)
    act_acc = torch.zeros(dim).to(device)

    with torch.no_grad():
        for t in range(200):
            x_next, y = model(x, u[t])
            # FIX 1+2: lower theta eta, stronger decay
            model.bcm_update(x, y, theta, eta=1e-4, decay=2e-3)
            # FIX 1: slow theta update
            theta += 1e-4 * (y.squeeze()**2 - theta)
            act_acc += torch.abs(x_next.squeeze())
            x = x_next

    # Homeostasis
    model.homeostatic_update(act_acc / 200, eta_hom=2e-3, alpha_target=0.35)

    # FIX 2: explicit pruning (same as v8b)
    if epoch >= PRUNE_START and epoch % PRUNE_EVERY == 0:
        model.activity_prune(threshold=PRUNE_THRESH)

    if epoch % 60 == 0:
        active = torch.count_nonzero(torch.abs(model.W) > 0.005).item()
        phase  = "BCM    " if epoch < PRUNE_START else "Pruning"
        print(f"  Epoch {epoch:3d} ({phase}) | active_syn={active:5d} | mean_theta={theta.mean().item():.4f}")

# ────────────────────────────────────────────────────────────────
# 5.  VERIFICATION (ARI & Distance)
# ────────────────────────────────────────────────────────────────

def collect_traj(root_name, steps=300):
    root_data = {r[0]: r[1] for r in VERBAL_ROOTS}
    phonemes  = root_data.get(root_name, ['a'])
    x = torch.zeros(1, dim).to(device)
    u = root_to_input(phonemes, seq_len=steps)
    states = []
    with torch.no_grad():
        for t in range(steps):
            x = model(x, u[t])[0]
            states.append(x.cpu().numpy())
    return np.array(states).squeeze()

# Dhatu Clustering
CLUSTER_ROOTS = ["gam", "sthā", "dṛś", "vac", "bhū", "kṛ"]
trajs = {r: collect_traj(r) for r in CLUSTER_ROOTS}
F = np.column_stack([np.mean(trajs[r], axis=0) for r in CLUSTER_ROOTS])
cluster_labels = KMeans(n_clusters=5, random_state=42, n_init=20).fit_predict(F)

# Code extraction
all_codes, all_labels, all_roots = [], [], []
for root_orig, ph, cat, gloss in VERBAL_ROOTS:
    for r in range(10): # 10 samples
        traj  = collect_traj(root_orig)
        # Simplify code to mean per-cluster activation
        code = [float(np.mean(np.abs(traj[:, cluster_labels==k]))) if np.any(cluster_labels==k) else 0.0 for k in range(5)]
        all_codes.append(code)
        all_labels.append(cat)
        all_roots.append(root_orig)

all_codes = np.array(all_codes)
le = LabelEncoder()
ids = le.fit_transform(all_labels)
scaler = StandardScaler()
norm = scaler.fit_transform(all_codes)

# ARI
best_ari = -1
for k in range(len(le.classes_), len(le.classes_)+3):
    pred = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(norm)
    ari = adjusted_rand_score(ids, pred)
    if ari > best_ari: best_ari = ari

print(f"\nFinal Convergence Grounding ARI: {best_ari:.3f}")

# Phonetic vs Dhatu Distance
def phonetic_distance(r1, r2):
    v1 = np.mean([phoneme_vec(p) for p in [ph for n,ph,_,_ in VERBAL_ROOTS if n==r1][0]], axis=0)
    v2 = np.mean([phoneme_vec(p) for p in [ph for n,ph,_,_ in VERBAL_ROOTS if n==r2][0]], axis=0)
    return float(np.linalg.norm(v1 - v2))

def dhatu_distance(r1, r2):
    c1 = np.mean(all_codes[np.array(all_roots)==r1], axis=0)
    c2 = np.mean(all_codes[np.array(all_roots)==r2], axis=0)
    n1 = scaler.transform(c1.reshape(1,-1))[0]
    n2 = scaler.transform(c2.reshape(1,-1))[0]
    return float(np.linalg.norm(n1 - n2))

print(f"\nDistance Consistency Check (Victory Run):")
print(f"  {'Pair':15s} | SAME? | PhonDist | DhatuDist")
for r1, r2 in [("gam","dhāv"), ("gam","car"), ("gam","sthā"), ("śī","śru"), ("śī","vid")]:
    r1_a = r1.replace("ā", "a").replace("ś", "sh").replace("ī", "i")
    r2_a = r2.replace("ā", "a").replace("ś", "sh").replace("ī", "i")
    c1 = [c for n,_,c,_ in VERBAL_ROOTS if n==r1][0]
    c2 = [c for n,_,c,_ in VERBAL_ROOTS if n==r2][0]
    pd = phonetic_distance(r1, r2)
    dd = dhatu_distance(r1, r2)
    print(f"  {r1_a:5s}-{r2_a:5s}       | {'YES' if c1==c2 else 'no':5s} | {pd:8.3f} | {dd:.3f}")

# Final Plot
pca = PCA(n_components=2)
low = pca.fit_transform(norm)
plt.figure(figsize=(10, 8))
plt.title(f"DDIN v10: Final Convergence\nARI={best_ari:.3f} | BCM-Sparse 128D Reservoir")
for cat in np.unique(all_labels):
    mask = all_labels == cat
    plt.scatter(low[mask,0], low[mask,1], label=cat, alpha=0.6)
plt.legend()
plt.savefig('ddin_v10_victory.png')
print("\nSaved: ddin_v10_victory.png")
