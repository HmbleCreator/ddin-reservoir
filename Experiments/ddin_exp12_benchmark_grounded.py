"""
DDIN v12 — Benchmark-Grounded Phonosemantic Embedding
=======================================================

Two upgrades over exp11:

  1. Real ground truth (150 roots, 5 phenomenological axes):
     Instead of our hand-crafted 30-root / 7-category set, we use the
     task1_axis_prediction.csv from the PhonosemanticMeta benchmark:
       - 150 Sanskrit verbal roots (from Panini's Dhatupatha)
       - 5 categories: EXP / TRN / MOT / SEP / CNT
         (Expansion, Transformation, Motion, Separation, Containment)
       - Labelled with Monier-Williams glosses (real semantic grounding)
       - Locus assigned (THROAT / PALATE / CEREBRAL / DENTAL / LABIAL)

  2. Locus one-hot as embedding prior (16D → 21D):
     The Paninian Sthana (place of articulation) is added as a structured
     categorical prior. This gives the DDIN a direct "address" in the
     phonosemantic space without overriding the acoustic features.

     Locus encoding:
       [THROAT, PALATE, CEREBRAL, DENTAL, LABIAL] → dims 16-20

     Cross-tab from task1 shows real correlations:
       THROAT  → EXP (70% accuracy)
       DENTAL  → SEP (40%)
       LABIAL  → CNT (40%)
     These structured priors should directly lift the ARI ceiling.
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
print("="*60)
print("DDIN v12 -- BENCHMARK-GROUNDED (150 roots, 5 axes, 21D)")
print("="*60)

IN_DIM = 21   # 16D acoustic + 5D locus one-hot

# ────────────────────────────────────────────────────────────────
# 1.  PHONEME VECTORS (16D from exp11)
# ────────────────────────────────────────────────────────────────

PHONEME_VECTORS_16 = {
    'a'  : [0.50, 0.50, 0.0, 1.0, 0.0, 0.0, 0.50, 0.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 1.0, 0.7],
    'A'  : [0.50, 0.50, 0.0, 1.0, 0.0, 0.0, 0.50, 1.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 1.0, 0.7],  # ā
    'i'  : [0.25, 0.80, 0.0, 1.0, 0.0, 1.0, 0.0,  0.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.3, 0.7],
    'I'  : [0.25, 0.80, 0.0, 1.0, 0.0, 1.0, 0.0,  1.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.3, 0.7],  # ī
    'u'  : [0.75, 0.80, 0.0, 1.0, 0.0, 1.0, 1.0,  0.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.2, 0.7],
    'U'  : [0.75, 0.80, 0.0, 1.0, 0.0, 1.0, 1.0,  1.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.2, 0.7],  # ū
    'R'  : [0.50, 0.60, 0.0, 1.0, 0.0, 0.5, 0.0,  0.0, 1.0, 0.9,  1.0, 1.0, 0.0, 0.0, 0.5, 0.7],  # ṛ
    'e'  : [0.25, 0.70, 0.0, 1.0, 0.0, 0.7, 0.0,  0.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.7, 0.7],
    'o'  : [0.75, 0.70, 0.0, 1.0, 0.0, 0.7, 1.0,  0.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.7, 0.7],
    # Velar (THROAT)
    'k'  : [0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    'K'  : [0.0,  0.0,  1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 0.0, 0.0, 0.0],  # kh
    'g'  : [0.0,  0.0,  0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 0.0, 0.0, 0.0],
    'G'  : [0.0,  0.0,  1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 0.0, 0.0, 0.0],  # gh
    'N'  : [0.0,  0.0,  0.0, 1.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.5,  0.5, 0.5,  0.0, 0.0, 0.0, 1.0],  # ṅ
    'h'  : [0.0,  0.0,  0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.3,  1.0, 0.65, 0.0, 0.0, 0.0, 0.0],
    # Palatal
    'c'  : [0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.0, 0.0],
    'C'  : [0.25, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],  # ch
    'j'  : [0.25, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 1.0, 0.0, 0.0],
    'J'  : [0.25, 0.25, 1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],  # jh
    'y'  : [0.25, 0.25, 0.0, 1.0, 0.0, 0.8, 0.0,  0.0, 1.0, 0.8,  1.0, 0.85, 0.0, 1.0, 0.0, 0.4],
    'z'  : [0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.2,  1.0, 0.55, 1.0, 1.0, 0.0, 0.0],  # ś
    # Cerebral (retroflex)
    'T'  : [0.5,  0.5,  0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.0, 0.0],  # ṭ
    'Q'  : [0.5,  0.5,  1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],  # ṭh
    'D'  : [0.5,  0.5,  0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 1.0, 0.0, 0.0],  # ḍ
    'X'  : [0.5,  0.5,  1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],  # ḍh
    'r'  : [0.5,  0.5,  0.0, 1.0, 0.0, 0.3, 0.0,  0.0, 1.0, 0.8,  1.0, 0.75, 0.0, 1.0, 0.0, 0.4],
    'x'  : [0.5,  0.5,  0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.2,  1.0, 0.55, 1.0, 1.0, 0.0, 0.0],  # ṣ
    # Dental
    't'  : [0.75, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.0, 0.0],
    'H'  : [0.75, 0.75, 1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],  # th
    'd'  : [0.75, 0.75, 0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 1.0, 0.0, 0.0],
    'W'  : [0.75, 0.75, 1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],  # dh
    'n'  : [0.75, 0.75, 0.0, 1.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.5,  0.5, 0.5,  0.0, 1.0, 0.0, 1.0],
    'l'  : [0.75, 0.7,  0.0, 1.0, 0.0, 0.3, 0.0,  0.0, 1.0, 0.8,  1.0, 0.75, 0.0, 1.0, 0.0, 0.4],
    's'  : [0.75, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.2,  1.0, 0.55, 1.0, 1.0, 0.0, 0.0],
    # Labial
    'p'  : [1.0,  1.0,  0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    'P'  : [1.0,  1.0,  1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 0.0, 0.0, 0.0],  # ph
    'b'  : [1.0,  1.0,  0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 0.0, 0.0, 0.0],
    'B'  : [1.0,  1.0,  1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 0.0, 0.0, 0.0],  # bh
    'm'  : [1.0,  1.0,  0.0, 1.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.5,  0.5, 0.5,  0.0, 0.0, 0.0, 1.0],
    'v'  : [1.0,  1.0,  0.0, 1.0, 0.0, 0.2, 1.0,  0.0, 1.0, 0.7,  1.0, 0.85, 0.0, 0.0, 0.0, 0.4],
}

# Locus one-hot mapping
LOCUS_ONEHOT = {
    'THROAT'   : [1, 0, 0, 0, 0],
    'PALATE'   : [0, 1, 0, 0, 0],
    'CEREBRAL' : [0, 0, 1, 0, 0],
    'DENTAL'   : [0, 0, 0, 1, 0],
    'LABIAL'   : [0, 0, 0, 0, 1],
}

# Transliteration: map common romanization to our key scheme
TRANSLIT = {
    'A': 'A', 'I': 'I', 'U': 'U', 'R': 'R',  # long vowels
    'kh': 'K', 'gh': 'G', 'ch': 'C', 'jh': 'J',
    'Th': 'Q', 'Dh': 'X', 'th': 'H', 'dh': 'W',
    'ph': 'P', 'bh': 'B', 'sh': 'z', 'sh2': 'x',
    'ng': 'N',
}

def root_to_phon_chars(root_str):
    """Convert a root string like 'kram' to list of phoneme keys."""
    chars = []
    i = 0
    s = root_str
    while i < len(s):
        # Two-char digraphs first
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
    """Return 21D vector: 16D acoustic + 5D locus one-hot."""
    base = list(PHONEME_VECTORS_16.get(ph_char, [0.5]*16))
    loh  = LOCUS_ONEHOT.get(locus, [0,0,0,0,0])
    return np.array(base + loh, dtype=np.float32)

# ────────────────────────────────────────────────────────────────
# 2.  LOAD BENCHMARK DATA (task1_axis_prediction.csv)
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


print(f"\nFirst 6 roots (with char decomp):")
for name, chars, ax, gl, loc in VERBAL_ROOTS[:6]:
    covered = [c for c in chars if c in PHONEME_VECTORS_16]
    miss    = [c for c in chars if c not in PHONEME_VECTORS_16]
    print(f"  {name:8s} [{loc:10s}] {ax} chars={chars} miss={miss}")

# ────────────────────────────────────────────────────────────────
# 3.  INPUT GENERATOR (Direct 21D Drive)
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
# 4.  PCA BETA (21D → 128D, geometry-preserving)
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
    components = pca.components_  # (n_comp, in_dim)
    repeats  = dim // n_comp + 1
    beta_np  = np.tile(components, (repeats, 1))[:dim]
    beta_np += np.random.randn(*beta_np.shape) * 0.05
    norms    = np.linalg.norm(beta_np, axis=1, keepdims=True) + 1e-8
    beta_np /= norms
    return torch.FloatTensor(beta_np)

# ────────────────────────────────────────────────────────────────
# 5.  DDIN MODEL (identical to exp11)
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

    def forward(self, x, u, dt=0.1):
        y  = torch.tanh(x @ self.W)
        dx = -self.alpha * x + y + (u @ self.beta.T)
        return x + dt * dx, y

    def bcm_update(self, x, y, theta, eta=1e-4, decay=2e-3):
        with torch.no_grad():
            bcm_gate = y * (y - theta.unsqueeze(0))
            dW = (bcm_gate.T @ x) / x.shape[0]
            self.W.data += eta * dW - decay * self.W.data

    def activity_prune(self, threshold=0.015):
        with torch.no_grad():
            self.W.data[torch.abs(self.W.data) < threshold] = 0.0

    def homeostatic_update(self, mean_act, eta_hom=1e-3, alpha_target=0.35):
        with torch.no_grad():
            d_alpha = eta_hom * (mean_act - alpha_target)
            self.alpha.data += d_alpha
            self.alpha.data.clamp_(0.08, 0.85)

# ────────────────────────────────────────────────────────────────
# 6.  TRAINING
# ────────────────────────────────────────────────────────────────

dim          = 128
EPOCHS       = 600
PRUNE_START  = 300
PRUNE_EVERY  = 30
PRUNE_THRESH = 0.015

beta_pca = build_pca_beta(VERBAL_ROOTS, dim=dim, in_dim=IN_DIM)
model    = HebbianConvergenceSystem(dim=dim, beta_init=beta_pca).to(device)
theta    = torch.full((dim,), 0.02).to(device)

print(f"\nTraining v12 (dim={dim}, IN_DIM={IN_DIM}, n_roots={len(VERBAL_ROOTS)})...")
print(f"  theta_init=0.02  eta_theta=1e-4  decay=2e-3  prune_start={PRUNE_START}")
for epoch in range(EPOCHS):
    ri = epoch % len(VERBAL_ROOTS)
    _, chars, _, _, locus = VERBAL_ROOTS[ri]
    u = root_to_input(chars, locus, seq_len=200)
    x = torch.zeros(1, dim).to(device)
    act_acc = torch.zeros(dim).to(device)

    with torch.no_grad():
        for t in range(200):
            x_next, y = model(x, u[t])
            model.bcm_update(x, y, theta, eta=1e-4, decay=2e-3)
            theta += 1e-4 * (y.squeeze()**2 - theta)
            act_acc += torch.abs(x_next.squeeze())
            x = x_next

    model.homeostatic_update(act_acc / 200, eta_hom=2e-3, alpha_target=0.35)
    if epoch >= PRUNE_START and epoch % PRUNE_EVERY == 0:
        model.activity_prune(threshold=PRUNE_THRESH)

    if epoch % 60 == 0:
        active = torch.count_nonzero(torch.abs(model.W) > 0.005).item()
        phase  = "BCM    " if epoch < PRUNE_START else "Pruning"
        print(f"  Epoch {epoch:3d} ({phase}) | active_syn={active:5d} | mean_theta={theta.mean().item():.4f}")

# ────────────────────────────────────────────────────────────────
# 7.  VERIFICATION
# ────────────────────────────────────────────────────────────────

def collect_mean_traj(ri, steps=200):
    """Return mean reservoir state across the full trajectory — 128D code."""
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
all_codes, all_labels, all_roots_str = [], [], []
for i, (rname, chars, axis, _, locus) in enumerate(VERBAL_ROOTS):
    code = collect_mean_traj(i)
    all_codes.append(code)
    all_labels.append(axis)
    all_roots_str.append(rname)

all_codes  = np.array(all_codes)   # (150, 128)
all_labels = np.array(all_labels)
le  = LabelEncoder()
ids = le.fit_transform(all_labels)

# PCA(50) to reduce 128D → 50D before clustering
pca_code = PCA(n_components=50, random_state=42)
reduced  = pca_code.fit_transform(all_codes)  # (150, 50)
scaler   = StandardScaler()
norm     = scaler.fit_transform(reduced)

# ARI sweep
best_ari, best_k = -1, len(le.classes_)
for k in range(len(le.classes_), len(le.classes_)+4):
    pred = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(norm)
    ari  = adjusted_rand_score(ids, pred)
    if ari > best_ari:
        best_ari, best_k = ari, k
best_pred = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(norm)
nmi = normalized_mutual_info_score(ids, best_pred)

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

print(f"\nGrounding test (150 roots, 5 axes):")
print(f"  ARI    : {best_ari:.3f}   (exp11 baseline: 0.228)")
print(f"  NMI    : {nmi:.3f}")
print(f"  Sep.   : {ratio:.2f}x")
print(f"  Chance : {1/len(le.classes_):.3f} ({len(le.classes_)} categories)")

# Per-axis breakdown
print(f"\nPer-axis accuracy:")
for ax in le.classes_:
    m = all_labels == ax
    pred_for_ax = best_pred[m]
    # Most common cluster for this axis
    dominant = np.bincount(pred_for_ax).argmax()
    purity = np.sum(pred_for_ax == dominant) / m.sum()
    print(f"  {ax}: n={m.sum():3d}  cluster_purity={purity:.2f}")

# ── additional diagnostic: ARI against locus (should be higher)
locus_ids = LabelEncoder().fit_transform([VERBAL_ROOTS[i][4] for i in range(len(VERBAL_ROOTS))])
best_ari_locus = -1
for k in range(5, 8):
    pred_l = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(norm)
    ari_l  = adjusted_rand_score(locus_ids, pred_l)
    if ari_l > best_ari_locus: best_ari_locus = ari_l

print(f"\nDiagnostic ARI vs LOCUS (should be > ARI vs axis): {best_ari_locus:.3f}")
print(f"  (If locus ARI >> axis ARI: embedding encodes Sthana, not semantic axis)")
print(f"  (If both low: phoneme map coverage is the bottleneck)")

# Fix: loci_labels now has same length as all_codes (150)
loci_labels_arr = np.array([VERBAL_ROOTS[i][4] for i in range(len(VERBAL_ROOTS))])

# Plot
pca2 = PCA(n_components=2)
low  = pca2.fit_transform(norm)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"DDIN v12: Benchmark-Grounded (150 roots, 5 axes)\n"
             f"ARI_axis={best_ari:.3f} | ARI_locus={best_ari_locus:.3f} | NMI={nmi:.3f} | IN={IN_DIM}D")

ax = axes[0]
ax.set_title("By Phenomenological Axis (5 categories)")
colors = {'EXP':'#ff6b6b','TRN':'#ffd93d','MOT':'#6bcb77','SEP':'#4d96ff','CNT':'#c77dff'}
for axis in np.unique(all_labels):
    m = all_labels == axis
    ax.scatter(low[m,0], low[m,1], label=axis, alpha=0.6, s=25,
               color=colors.get(axis, '#888888'))
ax.legend(fontsize=9)

ax2 = axes[1]
ax2.set_title("By Locus / Sthana (5 loci)")
locus_colors = {'THROAT':'#ff4444','PALATE':'#ffaa00','CEREBRAL':'#44ff44',
                'DENTAL':'#4444ff','LABIAL':'#ff44ff'}
for loc in ['THROAT','PALATE','CEREBRAL','DENTAL','LABIAL']:
    m = loci_labels_arr == loc
    ax2.scatter(low[m,0], low[m,1], label=loc, alpha=0.6, s=25,
                color=locus_colors[loc])
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig('ddin_v12_benchmark.png', dpi=150, bbox_inches='tight')
print("\nSaved: ddin_v12_benchmark.png")
