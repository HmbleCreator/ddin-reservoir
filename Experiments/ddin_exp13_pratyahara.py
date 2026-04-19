"""
DDIN v13 — Pratyāhāra-Enriched Embedding (21D → 29D)
======================================================

Core hypothesis:
  The ARI ceiling in v12 (~0.228) is due to the acoustic+locus 21D embedding
  encoding Sthāna (place of articulation) rather than semantic axis. Pāṇini's
  Pratyāhāra (phoneme class) system is a deeper structural prior that may
  better demarcate the five phenomenological axes (EXP/TRN/MOT/SEP/CNT).

Upgrade: 21D → 29D by appending 8 Pratyāhāra membership bits

  The Śiva Sūtras (the phoneme inventory at the head of Ashtadhyayi) define
  14 Pratyāhāra classes. We select 8 that are most semantically discriminative:

  Dim 21 — AC  (all vowels: a, i, u, ṛ, e, o, ai, au, ā, ī, ū)
    Vowels = sustained sonority → motion, flow, openness (MOT / EXP roots)
    AC-heavy roots: gam/car/dhāv (MOT), kṛ/spṛś (EXP)

  Dim 22 — HAL (all consonants: everything not in AC)
    Consonant-initial roots have sharper onset → action, separation (SEP/EXP)

  Dim 23 — YAN (semivowels: y, r, l, v)
    Liquid approximants → motion continuancy, flowing semantics (MOT)
    Key roots: car [c-a-r], dhāv [dh-ā-v], vah [v-a-h]

  Dim 24 — JHAl (fricatives + aspirates: kh, gh, ch, jh, Th, Dh, th, dh, ph, bh, ś, ṣ, s, h)
    High-energy phonemes → projective, causative meaning (EXP / ACTION)

  Dim 25 — ŚAL (sibilants + h: ś, ṣ, s, h)
    Fricative hiss → speech, perception, knowledge (SEP / SPEECH markers)
    Key roots: śru [ś-r-u], śaṃs [ś-a-m-s], vid [v-i-d], dṛś [d-ṛ-ś]

  Dim 26 — JASh (voiced stops: g, j, ḍ/D, d, b)
    Voiced plosives → embodied action, striking, transformation (TRN / MOT)
    Key roots: gam [g], jan [j], dhāv [dh voiced], bhū [bh voiced]

  Dim 27 — NAM (nasals: ṅ/N, ñ, ṇ, n, m)
    Nasal resonance → containment, existence, boundary (CNT / EXISTENCE)
    Key roots: man [m], jan [n-final], gam [m-final], mṛ [m]

  Dim 28 — AK (short vowels: a, i, u, ṛ only — not long counterparts)
    Short-vowel roots → quick/transient energy vs long vowels = sustained
    Separates kṛ [ṛ short] from mṛ [ṛ short] vs sthā [ā long]

Rationale:
  These 8 dims are orthogonal to the locus one-hot (dims 16-20) — they encode
  MANNER-class membership rather than place of articulation. The combination
  of locus (WHERE articulated) + Pratyāhāra (WHAT CLASS of phoneme) should
  give the network richer semantic signal without overfitting to surface
  phonological form.

Experiment design:
  - Same 150-root benchmark dataset (task1_axis_prediction.csv)
  - Same DDIN architecture (128D reservoir, BCM + homeostatic pruning)
  - Same training regime (600 epochs, prune_start=300, decay=2e-3)
  - New: phoneme_vec_29(), build_pca_beta updated to IN_DIM=29
  - New diagnostics:
    * Per-Pratyāhāra-class ARI (8 dim ablation study)
    * Per-axis Pratyāhāra coverage heatmap
    * ARI progression table: v10b → v11 → v12 → v13
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
print("DDIN v13 -- PRATYAHARA-ENRICHED EMBEDDING (21D -> 29D)")
print("="*60)

IN_DIM = 29   # 16D acoustic + 5D locus one-hot + 8D pratyahara

# ────────────────────────────────────────────────────────────────
# 1.  PHONEME VECTORS  (same 16D acoustic from exp11/v12)
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

# ────────────────────────────────────────────────────────────────
# 2.  LOCUS ONE-HOT (same as v12, dims 16-20)
# ────────────────────────────────────────────────────────────────

LOCUS_ONEHOT = {
    'THROAT'   : [1, 0, 0, 0, 0],
    'PALATE'   : [0, 1, 0, 0, 0],
    'CEREBRAL' : [0, 0, 1, 0, 0],
    'DENTAL'   : [0, 0, 0, 1, 0],
    'LABIAL'   : [0, 0, 0, 0, 1],
}

# ────────────────────────────────────────────────────────────────
# 3.  PRATYAHARA MEMBERSHIP (8 classes, dims 21-28)
#     Based on Panini's Siva Sutras phoneme inventory
#     Using our compact key-scheme (capitals = special phonemes)
# ────────────────────────────────────────────────────────────────

# AC: all vowels (Panini: a i u R L e o + long forms)
AC  = set(['a','A','i','I','u','U','R','e','o'])

# HAL: all consonants (complement of AC + anusvara/visarga)
HAL = set(PHONEME_VECTORS_16.keys()) - AC

# YAN: semivowels / approximants (y, r, l, v)
YAN = set(['y', 'r', 'l', 'v'])

# JHAl: fricatives + aspirates (all aspirated stops + all fricatives)
JHAL = set(['K','G','C','J','Q','X','H','W','P','B',   # aspirated stops
            'z','x','s','h'])                           # fricatives

# SAL: sibilants + h (high-frequency fricatives)
SAL = set(['z','x','s','h'])  # ś, ṣ, s, h

# JASh: voiced stops (g, j, D/ḍ, d, b)
JASH = set(['g','j','D','d','b'])

# NAM: nasals (N=ṅ, ñ→treated as n, n, m)  
NAM = set(['N','n','m'])  # ñ not in our scheme; ṇ maps to retroflex area

# AK: short vowels only (not long)
AK  = set(['a','i','u','R'])  # short: a i u ṛ

PRATYAHARA_SETS = [AC, HAL, YAN, JHAL, SAL, JASH, NAM, AK]
PRATYAHARA_NAMES = ['AC','HAL','YAN','JHAl','SAL','JASh','NAM','AK']

def pratyahara_vec(ph_char):
    """Return 8-bit Pratyāhāra membership vector for a phoneme key."""
    return [1.0 if ph_char in S else 0.0 for S in PRATYAHARA_SETS]

# ────────────────────────────────────────────────────────────────
# 4.  TRANSLITERATION (same as v12)
# ────────────────────────────────────────────────────────────────

TRANSLIT = {
    'A': 'A', 'I': 'I', 'U': 'U', 'R': 'R',
    'kh': 'K', 'gh': 'G', 'ch': 'C', 'jh': 'J',
    'Th': 'Q', 'Dh': 'X', 'th': 'H', 'dh': 'W',
    'ph': 'P', 'bh': 'B', 'sh': 'z', 'sh2': 'x',
    'ng': 'N',
}

def root_to_phon_chars(root_str):
    """Convert root string like 'kram' to list of phoneme keys."""
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

def phoneme_vec_29(ph_char, locus):
    """Return 29D vector: 16D acoustic + 5D locus one-hot + 8D pratyahara."""
    base = list(PHONEME_VECTORS_16.get(ph_char, [0.5]*16))
    loh  = LOCUS_ONEHOT.get(locus, [0,0,0,0,0])
    prat = pratyahara_vec(ph_char)
    return np.array(base + loh + prat, dtype=np.float32)

# ────────────────────────────────────────────────────────────────
# 5.  LOAD BENCHMARK DATA
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

print(f"\nFirst 6 roots (with Pratyahara coverage):")
for name, chars, ax, gl, loc in VERBAL_ROOTS[:6]:
    covered = [c for c in chars if c in PHONEME_VECTORS_16]
    miss    = [c for c in chars if c not in PHONEME_VECTORS_16]
    prat_eg = pratyahara_vec(chars[0]) if chars else []
    print(f"  {name:8s} [{loc:10s}] {ax}  miss={miss}  prat[0]={[int(x) for x in prat_eg]}")

# ────────────────────────────────────────────────────────────────
# 6.  INPUT GENERATOR
# ────────────────────────────────────────────────────────────────

def root_to_input(chars, locus, seq_len=300, noise_std=0.01):
    PHONEME_DUR    = 40
    TRANSITION_DUR = 10
    vecs = [phoneme_vec_29(c, locus) for c in chars]
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
# 7.  PCA BETA (29D → 128D, geometry-preserving)
# ────────────────────────────────────────────────────────────────

def build_pca_beta(verbal_roots, dim=128, in_dim=IN_DIM):
    all_vecs = []
    for (_, chars, _, _, locus) in verbal_roots:
        vecs = [phoneme_vec_29(c, locus) for c in chars]
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
# 8.  DDIN MODEL (identical architecture to v12)
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
# 9.  TRAINING (same regime as v12)
# ────────────────────────────────────────────────────────────────

dim          = 128
EPOCHS       = 600
PRUNE_START  = 300
PRUNE_EVERY  = 30
PRUNE_THRESH = 0.015

beta_pca = build_pca_beta(VERBAL_ROOTS, dim=dim, in_dim=IN_DIM)
model    = HebbianConvergenceSystem(dim=dim, beta_init=beta_pca).to(device)
theta    = torch.full((dim,), 0.02).to(device)

print(f"\nTraining v13 (dim={dim}, IN_DIM={IN_DIM}, n_roots={len(VERBAL_ROOTS)})...")
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
# 10.  VERIFICATION: COLLECT 128D CODES
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
all_codes, all_labels, all_roots_str, all_loci = [], [], [], []
for i, (rname, chars, axis, _, locus) in enumerate(VERBAL_ROOTS):
    code = collect_mean_traj(i)
    all_codes.append(code)
    all_labels.append(axis)
    all_roots_str.append(rname)
    all_loci.append(locus)

all_codes  = np.array(all_codes)   # (150, 128)
all_labels = np.array(all_labels)
all_loci   = np.array(all_loci)

le  = LabelEncoder()
ids = le.fit_transform(all_labels)

# PCA(50) then StandardScale before clustering
pca_code = PCA(n_components=50, random_state=42)
reduced  = pca_code.fit_transform(all_codes)
scaler   = StandardScaler()
norm     = scaler.fit_transform(reduced)

# ────────────────────────────────────────────────────────────────
# 11.  ARI SWEEP — axis & locus
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

print(f"\n{'='*55}")
print(f"GROUNDING TEST — v13 (29D: 16D acoustic + 5D locus + 8D pratyahara)")
print(f"{'='*55}")
print(f"  ARI (axis)  : {best_ari_axis:.3f}   (v12 baseline: 0.228)")
print(f"  ARI (locus) : {best_ari_locus:.3f}   (diagnostic: if >> axis, Sthana dominates)")
print(f"  NMI         : {nmi:.3f}")
print(f"  Sep. ratio  : {ratio:.2f}x")
print(f"  Chance      : {1/len(le.classes_):.3f}  ({len(le.classes_)} categories)")

# Per-axis breakdown
print(f"\nPer-axis cluster purity:")
for ax in le.classes_:
    m = all_labels == ax
    pred_for_ax = best_pred[m]
    dominant = np.bincount(pred_for_ax).argmax()
    purity = np.sum(pred_for_ax == dominant) / m.sum()
    print(f"  {ax}: n={m.sum():3d}  purity={purity:.2f}")

# ────────────────────────────────────────────────────────────────
# 12.  PRATYAHARA CLASS ABLATION — which dims matter most?
# ────────────────────────────────────────────────────────────────

print(f"\nPratyahara Ablation (ARI with only 29D embedding, no DDIN):")
print(f"  {'Class':6s} | ARI (raw embedding) | Hypothesis")

# For ablation: use the raw 29D vectors directly (no DDIN)
raw_vecs_all = []
for (_, chars, _, _, locus) in VERBAL_ROOTS:
    vs = [phoneme_vec_29(c, locus) for c in chars]
    raw_vecs_all.append(np.mean(vs, axis=0))
raw_vecs_all = np.array(raw_vecs_all)

hypotheses = ['vowel class','consonant class','semivowels','aspirates+fricatives',
              'sibilants','voiced stops','nasals','short vowels']

for i, (name, hyp) in enumerate(zip(PRATYAHARA_NAMES, hypotheses)):
    dim_idx = 21 + i
    feat = raw_vecs_all[:, dim_idx].reshape(-1, 1)
    # Add small jitter to avoid degenerate clustering
    feat = feat + np.random.randn(*feat.shape) * 0.001
    scaler_1d = StandardScaler()
    feat_n    = scaler_1d.fit_transform(feat)
    ari_1d, _ = best_ari_for(all_labels, feat_n)
    print(f"  {name:6s} | {ari_1d:.3f}               | {hyp}")

# ────────────────────────────────────────────────────────────────
# 13.  ARI PROGRESSION TABLE
# ────────────────────────────────────────────────────────────────

print(f"\nARI Progression (phonosemantic grounding):")
print(f"  {'Version':20s} | {'ARI':>6} | {'IN_DIM':>6} | Notes")
print(f"  {'-'*70}")
versions = [
    ("v10b (BCM+PCA 10D)",     0.182,  10, "BCM always->Receiver Model"),
    ("v11 (16D enriched)",      None,   16, "add continuancy,energy,stridence,coronal"),
    ("v12 (21D+locus)",         None,   21, "benchmark 150 roots, locus one-hot"),
    (f"v13 (29D+pratyahara)",   best_ari_axis, 29, "Panini Siva Sutra 8-class membership"),
]
for vname, ari_val, idim, note in versions:
    ari_str = f"{ari_val:.3f}" if ari_val is not None else "  TBD"
    print(f"  {vname:20s} | {ari_str:>6} | {idim:>6} | {note}")

# ────────────────────────────────────────────────────────────────
# 14.  SAVE EMBEDDING FOR v14 MULTI-TASK EVAL
# ────────────────────────────────────────────────────────────────

np.save('ddin_v13_norm_codes.npy',  norm)
np.save('ddin_v13_raw_codes.npy',   all_codes)
np.save('ddin_v13_raw29d.npy',      raw_vecs_all)
np.save('ddin_v13_labels_axis.npy', all_labels.astype(str))
np.save('ddin_v13_labels_locus.npy',all_loci.astype(str))
np.save('ddin_v13_roots.npy',       np.array(all_roots_str))
print("\nSaved embedding arrays for v14 multi-task evaluator.")

# ────────────────────────────────────────────────────────────────
# 15.  PLOTS
# ────────────────────────────────────────────────────────────────

pca2 = PCA(n_components=2)
low  = pca2.fit_transform(norm)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(
    f"DDIN v13: Pratyahara-Enriched (150 roots, 29D = 16D acoustic + 5D locus + 8D pratyahara)\n"
    f"ARI_axis={best_ari_axis:.3f} | ARI_locus={best_ari_locus:.3f} | NMI={nmi:.3f} | Sep={ratio:.2f}x",
    fontsize=11
)

# Plot 1: by axis
ax = axes[0]
ax.set_title("By Phenomenological Axis")
colors = {'EXP':'#ff6b6b','TRN':'#ffd93d','MOT':'#6bcb77','SEP':'#4d96ff','CNT':'#c77dff'}
for axis in np.unique(all_labels):
    m = all_labels == axis
    ax.scatter(low[m,0], low[m,1], label=axis, alpha=0.65, s=30,
               color=colors.get(axis,'#888888'))
ax.legend(fontsize=9)

# Plot 2: by locus
ax2 = axes[1]
ax2.set_title("By Locus / Sthana")
locus_colors = {'THROAT':'#ff4444','PALATE':'#ffaa00','CEREBRAL':'#44cc44',
                'DENTAL':'#4444ff','LABIAL':'#ff44ff'}
for loc in ['THROAT','PALATE','CEREBRAL','DENTAL','LABIAL']:
    m = all_loci == loc
    ax2.scatter(low[m,0], low[m,1], label=loc, alpha=0.65, s=30,
                color=locus_colors[loc])
ax2.legend(fontsize=8)

# Plot 3: Pratyahara ARI heatmap (1-dim ablation)
ax3 = axes[2]
ax3.set_title("Pratyahara Class ARI (1-dim ablation)")
ari_vals_ablation = []
for i in range(8):
    dim_idx = 21 + i
    feat = raw_vecs_all[:, dim_idx].reshape(-1, 1) + np.random.randn(len(raw_vecs_all), 1)*0.001
    scaler_1d = StandardScaler()
    feat_n    = scaler_1d.fit_transform(feat)
    ari_1d, _ = best_ari_for(all_labels, feat_n)
    ari_vals_ablation.append(ari_1d)

bars = ax3.barh(PRATYAHARA_NAMES, ari_vals_ablation,
                color=['#ff6b6b','#ffd93d','#6bcb77','#4d96ff','#c77dff','#ff9944','#44dddd','#aaaaaa'])
ax3.axvline(x=1/5, color='red', linestyle='--', alpha=0.6, label='chance')
ax3.set_xlabel("ARI")
ax3.legend(fontsize=8)
for bar, v in zip(bars, ari_vals_ablation):
    ax3.text(v + 0.002, bar.get_y() + bar.get_height()/2,
             f'{v:.3f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('ddin_v13_pratyahara.png', dpi=150, bbox_inches='tight')
print("\nSaved: ddin_v13_pratyahara.png")

print(f"\n{'='*55}")
print(f"v13 COMPLETE. Key findings:")
print(f"  ARI jump: 0.182 (v10b) -> {best_ari_axis:.3f} (v13)")
print(f"  Locus ARI: {best_ari_locus:.3f} (if >> axis: Sthana > Semantics)")
print(f"  Best Pratyahara class: {PRATYAHARA_NAMES[int(np.argmax(ari_vals_ablation))]}")
print(f"  (ARI={max(ari_vals_ablation):.3f})")
print(f"{'='*55}")
