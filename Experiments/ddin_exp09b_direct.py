"""
DDIN v9b — Phonosemantic Grounding (Direct Drive)
================================================

Fixing the "Distance Inversion" problem in exp09.

Goal:
  Bypass the fixed random projection (10D -> 64D) which scrambled
  the articulatory geometry. Feed the 10D Paninian phoneme
  embeddings directly into a smaller DDIN.

Hypothesis:
  Using the raw 10D articulatory space as the DDIN's inherent 
  state space will preserve semantic topography, leading to 
  higher ARI and fixing the distance inversion.
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print("="*60)
print("DDIN v9b — PHONOSEMANTIC GROUNDING (Direct Drive)")
print("="*60)

# ────────────────────────────────────────────────────────────────
# 1.  PANINIAN PHONEME SYSTEM (10-dim)
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

def phoneme_vec(phoneme):
    if phoneme in PHONEME_VECTORS:
        return np.array(PHONEME_VECTORS[phoneme], dtype=np.float32)
    return np.array([0.5]*10, dtype=np.float32)

# ────────────────────────────────────────────────────────────────
# 2.  SANSKRIT VERBAL ROOTS
# ────────────────────────────────────────────────────────────────
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
# 3.  ROOT → TIME-VARYING INPUT SIGNAL (DIRECT DRIVE)
# ────────────────────────────────────────────────────────────────

INPUT_DIM      = 10      # MATCHING PHONEME VECTOR DIM
PHONEME_DUR    = 40      
TRANSITION_DUR = 10      

def root_to_input(phoneme_list, seq_len=300, noise_std=0.01):
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
        signal_t = signal_t + torch.randn_like(signal_t) * noise_std
    return signal_t.clamp(-2, 2)

# ────────────────────────────────────────────────────────────────
# 4.  DDIN MODEL (Small footprint: 10 neurons)
# ────────────────────────────────────────────────────────────────

class HeterogeneousLiquidSystem(nn.Module):
    def __init__(self, dim=10):
        super().__init__()
        self.W     = nn.Parameter(torch.randn(dim, dim) * 0.05)
        self.alpha = nn.Parameter(torch.rand(dim) * 0.6 + 0.1)
        self.beta  = nn.Parameter(torch.rand(dim) * 0.4 + 0.05)

    def forward(self, x, u, dt=0.1):
        # Direct drive: beta * u where u is already the articulatory vector
        dx = -self.alpha * x + torch.tanh(x @ self.W) + self.beta * u
        return x + dt * dx

# ────────────────────────────────────────────────────────────────
# 5.  TRAIN DDIN
# ────────────────────────────────────────────────────────────────

dim     = 10
EPOCHS  = 400
model   = HeterogeneousLiquidSystem(dim=dim).to(device)
opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_history    = []

print("\nTraining small DDIN (10 neurons) on direct articulatory drive...")
for epoch in range(EPOCHS):
    opt.zero_grad()
    root_idx = epoch % len(VERBAL_ROOTS)
    _, phonemes, _, _ = VERBAL_ROOTS[root_idx]
    u = root_to_input(phonemes, seq_len=200)
    x = torch.zeros(1, dim).to(device)
    pred_loss = smooth_loss = energy_loss = 0.0
    for t_step in range(199):
        x_next   = model(x,      u[t_step])
        x_future = model(x_next, u[t_step + 1])
        pred_loss   += torch.mean((x_future - x_next.detach()) ** 2)
        energy_loss += torch.mean(torch.abs(x))
        smooth_loss += torch.mean((x_next - x) ** 2)
        x = x_next
    
    if epoch < 200:
        loss = pred_loss + 0.1 * smooth_loss
    else:
        loss = pred_loss + 0.1*smooth_loss + 0.05*energy_loss + 0.05*torch.sum(torch.abs(model.W))
    loss.backward()
    opt.step()
    if epoch >= 200 and epoch % 20 == 0:
        with torch.no_grad():
            model.W[torch.abs(model.W) < 0.02] = 0.0
    if epoch % 100 == 0:
        print(f"  Epoch {epoch:3d} | loss={loss.item():.4f}")

# ────────────────────────────────────────────────────────────────
# 6.  DHĀTU CLUSTERING (on 10D states)
# ────────────────────────────────────────────────────────────────

CLUSTER_ROOTS = ["gam", "sthā", "dṛś", "vac", "bhū", "kṛ"]
N_DHATU = 5

def collect_root_traj(root_name, steps=300):
    root_data = {r[0]: r[1] for r in VERBAL_ROOTS}
    phonemes  = root_data.get(root_name, ['a'])
    x = torch.zeros(1, dim).to(device)
    u = root_to_input(phonemes, seq_len=steps)
    states = []
    with torch.no_grad():
        for t in range(steps):
            x = model(x, u[t])
            states.append(x.cpu().numpy())
    return np.array(states).squeeze()

print("\nCollecting trajectories...")
trajs_phono = {r: collect_root_traj(r) for r in CLUSTER_ROOTS}
mean_acts_p = {r: np.mean(np.abs(trajs_phono[r]), axis=0) for r in CLUSTER_ROOTS}
var_acts_p  = {r: np.var(trajs_phono[r],          axis=0) for r in CLUSTER_ROOTS}
mean_diff_p = np.mean(np.stack([np.abs(mean_acts_p["gam"]-mean_acts_p["sthā"])], axis=0), axis=0)
W_np = model.W.detach().cpu().numpy()
hub_score = np.sum(np.abs(W_np), axis=1) + np.sum(np.abs(W_np), axis=0)

F = np.column_stack([mean_acts_p["gam"], var_acts_p["gam"], hub_score])
F_norm = StandardScaler().fit_transform(F)
cluster_labels = KMeans(n_clusters=N_DHATU, random_state=42, n_init=30).fit_predict(F_norm)
print(f"Dhatu cluster sizes: {[int(np.sum(cluster_labels==k)) for k in range(N_DHATU)]}")

# ────────────────────────────────────────────────────────────────
# 7.  EXTENDED DHĀTU CODES
# ────────────────────────────────────────────────────────────────

def extended_code(traj, labels, n=N_DHATU):
    feats = []
    for k in range(n):
        m = labels == k
        nt = traj[:, m] if m.any() else np.zeros((traj.shape[0], 1))
        feats.append(float(np.mean(np.abs(nt))))
        feats.append(float(np.mean(nt)))
        feats.append(float(np.var(np.abs(nt))))
    return np.array(feats)

all_codes, all_labels, all_roots_arr = [], [], []
for root_name, phonemes, category, gloss in VERBAL_ROOTS:
    for r in range(8):
        traj = collect_root_traj(root_name)
        all_codes.append(extended_code(traj, cluster_labels))
        all_labels.append(category)
        all_roots_arr.append(root_name)

all_codes  = np.array(all_codes)
all_labels = np.array(all_labels)
all_roots_arr = np.array(all_roots_arr)
le = LabelEncoder()
label_ids = le.fit_transform(all_labels)
n_cats = len(le.classes_)
scaler = StandardScaler()
codes_norm = scaler.fit_transform(all_codes)

# Separability
def sep_ratio(codes, labels):
    cats = np.unique(labels)
    intra, inter = [], []
    for cat in cats:
        sub = codes[labels == cat]
        for i, j in combinations(range(len(sub)), 2):
            intra.append(np.linalg.norm(sub[i]-sub[j]))
    for c1, c2 in combinations(cats, 2):
        s1, s2 = codes[labels==c1], codes[labels==c2]
        for a in s1:
            for b in s2:
                inter.append(np.linalg.norm(a-b))
    return np.mean(inter)/(np.mean(intra)+1e-8), np.mean(intra), np.mean(inter)

ratio, intra_d, inter_d = sep_ratio(codes_norm, all_labels)
print(f"\nSeparability (Direct Drive): Ratio={ratio:.2f}x")

# Grounding test
best_ari, best_k, best_pred = -1, n_cats, None
for k in range(n_cats, n_cats+3):
    pred = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(codes_norm)
    ari = adjusted_rand_score(label_ids, pred)
    if ari > best_ari:
        best_ari, best_k, best_pred = ari, k, pred

print(f"Grounding ARI (Direct Drive): {best_ari:.3f} (v9 was 0.195)")

# ────────────────────────────────────────────────────────────────
# 8.  DISTANCE INVERSION CHECK
# ────────────────────────────────────────────────────────────────

def phonetic_distance(root1, root2):
    ph1 = [phoneme_vec(p) for p in root1]
    ph2 = [phoneme_vec(p) for p in root2]
    v1, v2 = np.mean(ph1, axis=0), np.mean(ph2, axis=0)
    return float(np.linalg.norm(v1 - v2))

def dhatu_distance(r1, r2):
    m1 = all_roots_arr == r1
    m2 = all_roots_arr == r2
    if not m1.any() or not m2.any(): return None
    p1 = np.mean(all_codes[m1], axis=0)
    p2 = np.mean(all_codes[m2], axis=0)
    n1 = scaler.transform(p1.reshape(1,-1))[0]
    n2 = scaler.transform(p2.reshape(1,-1))[0]
    return float(np.linalg.norm(n1 - n2))

print(f"\nDistance Check (Direct Drive):")
print(f"  {'Pair':15s} | SAME? | PhonDist | DhatuDist")
for r_orig1, r_orig2 in [("gam","dhāv"), ("gam","car"), ("gam","sthā"), ("śī","śru"), ("śī","vid")]:
    r1_ascii = r_orig1.replace("ā", "a").replace("ḍ", "d").replace("ṇ", "n").replace("ś", "sh").replace("ī", "i").replace("ṛ", "r")
    r2_ascii = r_orig2.replace("ā", "a").replace("ḍ", "d").replace("ṇ", "n").replace("ś", "sh").replace("ī", "i").replace("ṛ", "r")
    c1 = [c for n,_,c,_ in VERBAL_ROOTS if n==r_orig1][0]
    c2 = [c for n,_,c,_ in VERBAL_ROOTS if n==r_orig2][0]
    pd = phonetic_distance([ph for n,ph,_,_ in VERBAL_ROOTS if n==r_orig1][0], [ph for n,ph,_,_ in VERBAL_ROOTS if n==r_orig2][0])
    dd = dhatu_distance(r_orig1, r_orig2)
    if dd is not None:
        print(f"  {r1_ascii:5s}-{r2_ascii:5s}       | {'YES' if c1==c2 else 'no':5s} | {pd:8.3f} | {dd:.3f}")

# Visualisation (Simplified)
pca2 = PCA(n_components=2)
codes_2d = pca2.fit_transform(codes_norm)
plt.figure(figsize=(10, 8))
plt.title(f"Phonosemantic Grounding (Direct 10D Drive)\nARI={best_ari:.3f} | Ratio={ratio:.2f}x")
for cat in np.unique(all_labels):
    m = all_labels == cat
    plt.scatter(codes_2d[m,0], codes_2d[m,1], label=cat, alpha=0.6)
plt.legend()
plt.savefig('ddin_v9b_direct.png')
print("\nSaved: ddin_v9b_direct.png")
