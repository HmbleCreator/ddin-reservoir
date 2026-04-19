"""
DDIN v11 — Enriched Phoneme Embedding (16D)
============================================

Core hypothesis being tested:
  The ~0.18 ARI ceiling in exp10b is caused by insufficient articulatory
  discriminability in the 10D phoneme vectors, not by the DDIN architecture.

Enrichment strategy (10D → 16D):
  The current 10 features encode basic place/manner/voicing.
  Six new features are added that specifically target the discriminative
  dimensions needed to separate the 7 semantic categories:

  Dim 10 — Continuancy
    0 = stop (k, p, t, c, ṭ, ...)  → quick burst energy (ACTION, STABILITY)
    1 = fricative/liquid/vowel      → sustained flow (MOTION, SPEECH)
    Rationale: MOTION roots (gam, dhāv, car) are stop+sonorant dominant.
               SPEECH roots (vac, śaṃs) are fricative+approximant dominant.

  Dim 11 — Manner Energy (fine-grained sonority rank)
    0.0 = voiceless stop     (k, p, t)
    0.2 = voiced stop        (g, b, d)
    0.35= aspirated stop     (kh, ph, th, bh, dh, gh) — "projective force"
    0.5 = nasal              (m, n, ṅ)
    0.55= voiceless fricative(ś, ṣ, s)
    0.65= voiced fricative   (h)
    0.75= liquid/trill       (r, l)
    0.85= glide              (y, v)
    1.0 = vowel
    Rationale: Distinguishes the "energetic profile" of each phoneme more
               finely than dim 9, which is binary sonorant.

  Dim 12 — Stridence (sibilant hiss)
    1 = ś, ṣ, s (high-frequency fricative)
    0 = all others
    Rationale: SPEECH (śaṃs, jap) and PERCEPTION (śru, dṛś) roots
               are uniquely rich in sibilants.

  Dim 13 — Coronal (tongue-tip articulation)
    1 = dental/retroflex/palatal stops and fricatives
        (t, th, d, dh, n, ṭ, ṭh, ḍ, ḍh, ṇ, c, ch, j, jh, ñ, ś, ṣ, s)
    0 = velar/labial/laryngeal
    Rationale: ACTION roots (tap, han) and PERCEPTION roots (spṛś) cluster
               in coronal stop space; helps separate them from MOTION (guttural).

  Dim 14 — Vowel Openness (only meaningful for sonorants; 0 for consonants)
    1.0 = a, ā  (maximally open — associated with outward/motion)
    0.7 = e, o  (mid-vowels)
    0.5 = ṛ     (vocalic r — mid-central)
    0.3 = i, ī  (front close — EXCHANGE roots: krī, ji, nī, all use high-front)
    0.2 = u, ū  (back close — EXISTENCE: bhū)
    0.0 = consonants
    Rationale: Vowel quality tracks the "energetic openness" hypothesis from
               Sanskrit aesthetics (a=open, u=closed/contained).

  Dim 15 — Nasality + Sonorant Convergence
    Combines nasal resonance with sonorant flow for root-level profiling.
    1.0 = nasal (m, n, ṅ, ñ, ṇ)        — EXISTENCE (jan, mṛ) and MOTION (gam, ram)
    0.7 = vowel sonorant                — general
    0.4 = liquid sonorant (r, l)
    0.0 = non-sonorant
    Rationale: EXISTENCE roots (jan, mṛ, gam) have high nasal content.
               This allows the network to detect that nasality ≠ random.

All new features are grounded in the Paninian Siksha (phonetics) tradition,
extending rather than replacing the existing Sthana-Karana (place-manner) system.
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
from itertools import combinations

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print("="*60)
print("DDIN v11 -- ENRICHED PHONEME EMBEDDING (10D->16D)")
print("="*60)

IN_DIM = 16   # expanded from 10

# ────────────────────────────────────────────────────────────────
# 1.  ENRICHED PHONEME VECTORS (16D)
#     Format: [place_back, place_ht, aspir, voic, nasal, approx,
#              round, length, sonorant, sonority,
#              continuancy, manner_energy, strident, coronal,
#              vowel_openness, nasal_sonorant_converge]
# ────────────────────────────────────────────────────────────────

PHONEME_VECTORS = {
    # Vowels
    'a'  : [0.50, 0.50, 0.0, 1.0, 0.0, 0.0, 0.50, 0.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 1.0, 0.7],
    'ā'  : [0.50, 0.50, 0.0, 1.0, 0.0, 0.0, 0.50, 1.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 1.0, 0.7],
    'i'  : [0.25, 0.80, 0.0, 1.0, 0.0, 1.0, 0.0,  0.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.3, 0.7],
    'ī'  : [0.25, 0.80, 0.0, 1.0, 0.0, 1.0, 0.0,  1.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.3, 0.7],
    'u'  : [0.75, 0.80, 0.0, 1.0, 0.0, 1.0, 1.0,  0.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.2, 0.7],
    'ū'  : [0.75, 0.80, 0.0, 1.0, 0.0, 1.0, 1.0,  1.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.2, 0.7],
    'ṛ'  : [0.50, 0.60, 0.0, 1.0, 0.0, 0.5, 0.0,  0.0, 1.0, 0.9,  1.0, 1.0, 0.0, 0.0, 0.5, 0.7],
    'e'  : [0.25, 0.70, 0.0, 1.0, 0.0, 0.7, 0.0,  0.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.7, 0.7],
    'ai' : [0.25, 0.70, 0.0, 1.0, 0.0, 0.5, 0.0,  0.5, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.7, 0.7],
    'o'  : [0.75, 0.70, 0.0, 1.0, 0.0, 0.7, 1.0,  0.0, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.7, 0.7],
    'au' : [0.75, 0.70, 0.0, 1.0, 0.0, 0.5, 1.0,  0.5, 1.0, 1.0,  1.0, 1.0, 0.0, 0.0, 0.7, 0.7],

    # Velar stops
    'k'  : [0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    'kh' : [0.0,  0.0,  1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 0.0, 0.0, 0.0],
    'g'  : [0.0,  0.0,  0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 0.0, 0.0, 0.0],
    'gh' : [0.0,  0.0,  1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 0.0, 0.0, 0.0],
    'ṅ'  : [0.0,  0.0,  0.0, 1.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.5,  0.5, 0.5,  0.0, 0.0, 0.0, 1.0],

    # Palatal stops/affricates
    'c'  : [0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.0, 0.0],
    'ch' : [0.25, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],
    'j'  : [0.25, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 1.0, 0.0, 0.0],
    'jh' : [0.25, 0.25, 1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],
    'ñ'  : [0.25, 0.25, 0.0, 1.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.5,  0.5, 0.5,  0.0, 1.0, 0.0, 1.0],

    # Retroflex stops
    'ṭ'  : [0.5,  0.5,  0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.0, 0.0],
    'ṭh' : [0.5,  0.5,  1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],
    'ḍ'  : [0.5,  0.5,  0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 1.0, 0.0, 0.0],
    'ḍh' : [0.5,  0.5,  1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],
    'ṇ'  : [0.5,  0.5,  0.0, 1.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.5,  0.5, 0.5,  0.0, 1.0, 0.0, 1.0],

    # Dental stops
    't'  : [0.75, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.0, 0.0],
    'th' : [0.75, 0.75, 1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],
    'd'  : [0.75, 0.75, 0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 1.0, 0.0, 0.0],
    'dh' : [0.75, 0.75, 1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],
    'n'  : [0.75, 0.75, 0.0, 1.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.5,  0.5, 0.5,  0.0, 1.0, 0.0, 1.0],

    # Labial stops
    'p'  : [1.0,  1.0,  0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    'ph' : [1.0,  1.0,  1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 0.0, 0.0, 0.0],
    'b'  : [1.0,  1.0,  0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 0.0, 0.0, 0.0],
    'bh' : [1.0,  1.0,  1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 0.0, 0.0, 0.0],
    'm'  : [1.0,  1.0,  0.0, 1.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.5,  0.5, 0.5,  0.0, 0.0, 0.0, 1.0],

    # Approximants / semivowels
    'y'  : [0.25, 0.25, 0.0, 1.0, 0.0, 0.8, 0.0,  0.0, 1.0, 0.8,  1.0, 0.85, 0.0, 1.0, 0.0, 0.4],
    'r'  : [0.5,  0.5,  0.0, 1.0, 0.0, 0.3, 0.0,  0.0, 1.0, 0.8,  1.0, 0.75, 0.0, 1.0, 0.0, 0.4],
    'l'  : [0.75, 0.7,  0.0, 1.0, 0.0, 0.3, 0.0,  0.0, 1.0, 0.8,  1.0, 0.75, 0.0, 1.0, 0.0, 0.4],
    'v'  : [1.0,  1.0,  0.0, 1.0, 0.0, 0.2, 1.0,  0.0, 1.0, 0.7,  1.0, 0.85, 0.0, 0.0, 0.0, 0.4],

    # Fricatives / sibilants
    'ś'  : [0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.2,  1.0, 0.55, 1.0, 1.0, 0.0, 0.0],
    'ṣ'  : [0.5,  0.5,  0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.2,  1.0, 0.55, 1.0, 1.0, 0.0, 0.0],
    's'  : [0.75, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.2,  1.0, 0.55, 1.0, 1.0, 0.0, 0.0],

    # Laryngeal
    'h'  : [0.0,  0.0,  0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.3,  1.0, 0.65, 0.0, 0.0, 0.0, 0.0],
}

# ────────────────────────────────────────────────────────────────
# 2.  VERBAL ROOTS (unchanged, same 30 roots as v9-v10)
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
# 3.  INPUT GENERATOR (Direct 16D Drive)
# ────────────────────────────────────────────────────────────────

def phoneme_vec(phoneme):
    if phoneme in PHONEME_VECTORS:
        return np.array(PHONEME_VECTORS[phoneme], dtype=np.float32)
    return np.array([0.5]*IN_DIM, dtype=np.float32)

def root_to_input(phoneme_list, seq_len=300, noise_std=0.01):
    PHONEME_DUR    = 40
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
# 4.  PCA BETA (preserves 16D articulatory manifold in 128D)
# ────────────────────────────────────────────────────────────────

def build_pca_beta(verbal_roots, phoneme_vectors_dict, dim=128, in_dim=IN_DIM):
    all_vecs = []
    for _, phonemes, _, _ in verbal_roots:
        vecs = [np.array(phoneme_vectors_dict.get(p, [0.5]*in_dim), dtype=np.float32)
                for p in phonemes]
        all_vecs.append(np.mean(vecs, axis=0))
    X = np.array(all_vecs)
    pca = PCA(n_components=in_dim)
    pca.fit(X)
    components = pca.components_
    repeats   = dim // in_dim + 1
    beta_np   = np.tile(components, (repeats, 1))[:dim]
    beta_np  += np.random.randn(*beta_np.shape) * 0.05
    norms     = np.linalg.norm(beta_np, axis=1, keepdims=True) + 1e-8
    beta_np  /= norms
    return torch.FloatTensor(beta_np)

# ────────────────────────────────────────────────────────────────
# 5.  DDIN MODEL (128 neurons, fixed BCM from exp10b)
# ────────────────────────────────────────────────────────────────

class HebbianConvergenceSystem(nn.Module):
    def __init__(self, dim=128, beta_init=None):
        super().__init__()
        self.W     = nn.Parameter(torch.randn(dim, dim) * 0.02, requires_grad=False)
        self.alpha = nn.Parameter(torch.rand(dim) * 0.3 + 0.2,  requires_grad=False)
        if beta_init is not None:
            self.beta = nn.Parameter(beta_init.to(device), requires_grad=False)
        else:
            beta_rand = torch.rand(dim, IN_DIM)
            beta_rand /= beta_rand.norm(dim=1, keepdim=True) + 1e-8
            self.beta = nn.Parameter(beta_rand.to(device), requires_grad=False)
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
# 6.  TRAINING (same fixed BCM params as v10b)
# ────────────────────────────────────────────────────────────────

dim          = 128
EPOCHS       = 600
PRUNE_START  = 300
PRUNE_EVERY  = 30
PRUNE_THRESH = 0.015

beta_pca = build_pca_beta(VERBAL_ROOTS, PHONEME_VECTORS, dim=dim, in_dim=IN_DIM)
model    = HebbianConvergenceSystem(dim=dim, beta_init=beta_pca).to(device)
theta    = torch.full((dim,), 0.02).to(device)

print(f"Training v11 (dim={dim}, IN_DIM={IN_DIM}) -- Fixed BCM + PCA-beta...")
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

# Dhatu clustering
CLUSTER_ROOTS = ["gam", "sthā", "dṛś", "vac", "bhū", "kṛ"]
trajs = {r: collect_traj(r) for r in CLUSTER_ROOTS}
F = np.column_stack([np.mean(trajs[r], axis=0) for r in CLUSTER_ROOTS])
cluster_labels = KMeans(n_clusters=5, random_state=42, n_init=20).fit_predict(F)
print(f"\nDhatu cluster sizes: {[int(np.sum(cluster_labels==k)) for k in range(5)]}")

# Code extraction
all_codes, all_labels, all_roots = [], [], []
for root_orig, ph, cat, _ in VERBAL_ROOTS:
    for r in range(10):
        traj = collect_traj(root_orig)
        code = [float(np.mean(np.abs(traj[:, cluster_labels==k])))
                if np.any(cluster_labels==k) else 0.0 for k in range(5)]
        all_codes.append(code)
        all_labels.append(cat)
        all_roots.append(root_orig)

all_codes  = np.array(all_codes)
all_labels = np.array(all_labels)
all_roots  = np.array(all_roots)
le = LabelEncoder()
ids = le.fit_transform(all_labels)
scaler = StandardScaler()
norm = scaler.fit_transform(all_codes)

# ARI sweep
best_ari, best_k = -1, len(le.classes_)
for k in range(len(le.classes_), len(le.classes_)+4):
    pred = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(norm)
    ari  = adjusted_rand_score(ids, pred)
    if ari > best_ari:
        best_ari, best_k = ari, k

nmi = normalized_mutual_info_score(ids,
      KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(norm))

def sep_ratio(codes, labels):
    intra, inter = [], []
    cats = np.unique(labels)
    for cat in cats:
        sub = codes[labels==cat]
        for i, j in combinations(range(len(sub)), 2):
            intra.append(np.linalg.norm(sub[i]-sub[j]))
    for c1, c2 in combinations(cats, 2):
        for a in codes[labels==c1]:
            for b in codes[labels==c2]:
                inter.append(np.linalg.norm(a-b))
    return np.mean(inter)/(np.mean(intra)+1e-8)

ratio = sep_ratio(norm, all_labels)

print(f"\nGrounding test (16D enriched phonemes):")
print(f"  ARI  : {best_ari:.3f}   (v10b baseline: 0.182)")
print(f"  NMI  : {nmi:.3f}")
print(f"  Sep. : {ratio:.2f}x  (v10b baseline: ~1.2x)")

# Distance check
def phonetic_distance(r1, r2):
    v1 = np.mean([phoneme_vec(p) for p in [ph for n,ph,_,_ in VERBAL_ROOTS if n==r1][0]], axis=0)
    v2 = np.mean([phoneme_vec(p) for p in [ph for n,ph,_,_ in VERBAL_ROOTS if n==r2][0]], axis=0)
    return float(np.linalg.norm(v1 - v2))

def dhatu_distance(r1, r2):
    c1 = np.mean(all_codes[all_roots==r1], axis=0)
    c2 = np.mean(all_codes[all_roots==r2], axis=0)
    return float(np.linalg.norm(scaler.transform(c1.reshape(1,-1))[0] -
                                scaler.transform(c2.reshape(1,-1))[0]))

print(f"\nDistance Consistency Check:")
print(f"  {'Pair':15s} | SAME? | PhonDist | DhatuDist | OK?")
for r1, r2 in [("gam","dhāv"), ("gam","car"), ("gam","sthā"), ("śī","śru"), ("śī","vid")]:
    r1_a = r1.replace("ā","a").replace("ś","sh").replace("ī","i")
    r2_a = r2.replace("ā","a").replace("ś","sh").replace("ī","i")
    c1 = [c for n,_,c,_ in VERBAL_ROOTS if n==r1][0]
    c2 = [c for n,_,c,_ in VERBAL_ROOTS if n==r2][0]
    pd = phonetic_distance(r1, r2)
    dd = dhatu_distance(r1, r2)
    same = c1 == c2
    # For SAME pairs: want dd to be SMALL; for DIFF pairs: want dd to be LARGE
    ok = (dd < 6 if same else dd > 4)
    print(f"  {r1_a:5s}-{r2_a:5s}       | {'YES' if same else 'no':5s} | {pd:8.3f} | {dd:9.3f} | {'OK' if ok else '!'}")

# Plot
pca2   = PCA(n_components=2)
low    = pca2.fit_transform(norm)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"DDIN v11: Enriched 16D Phoneme Embedding\nARI={best_ari:.3f} | Sep={ratio:.2f}x | NMI={nmi:.3f}")

# Scatter by semantic category
ax = axes[0]
ax.set_title("By Semantic Category")
for cat in np.unique(all_labels):
    m = all_labels == cat
    ax.scatter(low[m,0], low[m,1], label=cat, alpha=0.65, s=40)
ax.legend(fontsize=7)

# Scatter by root (first letter silhouette)
ax2 = axes[1]
ax2.set_title("By Root (phoneme overlap visible)")
cat_colors = {c: plt.cm.Set1(i/6) for i, c in enumerate(np.unique(all_labels))}
unique_roots = list(dict.fromkeys(all_roots))
for root in unique_roots:
    m  = all_roots == root
    c  = [c for n,_,c,_ in VERBAL_ROOTS if n==root][0]
    ax2.scatter(low[m,0], low[m,1], color=cat_colors[c], alpha=0.5, s=30)
    cx, cy = low[m,0].mean(), low[m,1].mean()
    label  = root.replace("ā","a").replace("ś","sh").replace("ī","i").replace("ṛ","r")
    ax2.text(cx, cy, label, fontsize=5.5, ha='center')

plt.tight_layout()
plt.savefig('ddin_v11_enriched.png', dpi=150, bbox_inches='tight')
print("\nSaved: ddin_v11_enriched.png")
