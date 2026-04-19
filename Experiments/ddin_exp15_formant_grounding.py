"""
DDIN v15 — Acoustic Formant Grounding (F1/F2 continuous embedding)
===================================================================

Motivation (from §39 research log):
  The 5D locus one-hot in v12/v13 creates 5 perfectly orthogonal cluster
  centroids — one per Sthana. Every phoneme within a locus gets the exact
  same one-hot subvector, so the cosine metric sees:

      diff-axis SAME-locus  ≈  same-axis SAME-locus   (WRONG)
      same-axis DIFF-locus  <<  diff-axis DIFF-locus   (WRONG)

  This is the locus-dominance floor: the one-hot dominates L2/cosine and
  hides cross-locus semantic structure that IS the phonosemantic signal.

Fix: Replace 5D locus one-hot with 2D continuous formant space (F1, F2).
  
  Formant theory (Fant 1960, Ladefoged 2001):
    F1 (1st formant) ↔ vowel HEIGHT   (low = high vowel: i, u)
    F2 (2nd formant) ↔ vowel BACKNESS (high = front vowel: i; low = back: u)

  For consonants, use LOCUS THEORY (Sussman et al. 1991):
    The F2 "locus frequency" at consonant release predicts semantic groupings
    across the vowel-consonant boundary. Each place of articulation has a
    characteristic F2 locus:
      Labial:     F2_locus ≈  800 Hz  (lowest — mouth closure)
      Dental:     F2_locus ≈ 1800 Hz
      Retroflex:  F2_locus ≈ 1100 Hz  (between labial and dental)
      Palatal:    F2_locus ≈ 2100 Hz  (highest — tongue near palate)
      Velar:      F2_locus ≈ 1400 Hz  (middle — velar pinch)
    F1 for consonants: near 0 (closure) → 600 Hz (transition)

  Normalized to [0, 1] over [200, 2500] Hz for F2 and [0, 1200] Hz for F1.

Embedding: 16D acoustic + 2D formant (NO locus one-hot) + 8D Pratyahara = 26D
  vs v13:  16D acoustic + 5D locus one-hot + 8D Pratyahara = 29D

Expected effect:
  - Continuous formant space allows gradient geometry ACROSS Sthana boundaries
  - Vowel-anchored roots (high AC / AK Pratyahara) will cluster by F1/F2 identity
  - Cross-locus semantic similarity (e.g., gam [THROAT] ≈ car [DENTAL] as both MOT)
    should become visible in the formant subspace

Evaluation:
  1. Raw 26D ARI + multi-task proxy (repeat T1–T8 from v14)
  2. Geometry diagnostics: locus separation vs axis separation in formant subspace
  3. Progression table: v12 → v13 → v15
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from itertools import combinations
from scipy.stats import pointbiserialr

np.random.seed(42)

print("="*65)
print("DDIN v15 -- ACOUSTIC FORMANT GROUNDING (F1/F2 continuous)")
print("Replaces 5D locus one-hot with 2D formant (F1, F2)")
print("Embedding: 16D acoustic + 2D formant + 8D Pratyahara = 26D")
print("="*65)

# ─────────────────────────────────────────────────────────────────
# 1.  FORMANT VALUES (F1, F2) — normalized to [0, 1]
#     Sources: Fant 1960, Ladefoged & Maddieson 1996, Sussman 1991
#     F1 range: 0–1200 Hz → /1200
#     F2 range: 200–2800 Hz → (F2 - 200)/2600
# ─────────────────────────────────────────────────────────────────

def f1f2_norm(f1_hz, f2_hz):
    """Normalize formant frequencies to [0,1]."""
    f1_n = np.clip(f1_hz / 1200.0, 0, 1)
    f2_n = np.clip((f2_hz - 200) / 2600.0, 0, 1)
    return [round(f1_n, 4), round(f2_n, 4)]

# ── VOWELS: real IPA formant values for Sanskrit (from Ohala 1983)
# ── CONSONANTS: locus-theory F2, F1 near-zero (consonant closure)
#    Locus F2: Labial=800, Retroflex=1100, Velar=1400, Dental=1800, Palatal=2100
#    Locus F1: during closure ≈ 100-200Hz (below F1 of adjacent vowel)

FORMANT_F1F2 = {
    # Vowels — IPA values (Hz), Sanskrit approximations
    'a'  : f1f2_norm( 800, 1300),   # low central (neutral)
    'A'  : f1f2_norm( 800, 1300),   # ā long — same quality
    'i'  : f1f2_norm( 280, 2300),   # high front
    'I'  : f1f2_norm( 280, 2300),   # ī long
    'u'  : f1f2_norm( 280,  700),   # high back rounded
    'U'  : f1f2_norm( 280,  700),   # ū long
    'R'  : f1f2_norm( 490, 1380),   # ṛ — rhotic vowel (Flemming 2003)
    'e'  : f1f2_norm( 400, 2000),   # mid front
    'o'  : f1f2_norm( 490,  800),   # mid back rounded

    # Velar consonants — F2 locus ~1400, F1 near closure ~150
    'k'  : f1f2_norm( 150, 1400),
    'K'  : f1f2_norm( 150, 1400),   # kh
    'g'  : f1f2_norm( 150, 1400),
    'G'  : f1f2_norm( 150, 1400),   # gh
    'N'  : f1f2_norm( 200, 1400),   # ṅ nasal (slightly higher F1)
    'h'  : f1f2_norm( 600, 1200),   # h glottal — near-vowel formants

    # Palatal consonants — F2 locus ~2100
    'c'  : f1f2_norm( 150, 2100),
    'C'  : f1f2_norm( 150, 2100),   # ch
    'j'  : f1f2_norm( 150, 2100),
    'J'  : f1f2_norm( 150, 2100),   # jh
    'y'  : f1f2_norm( 400, 2100),   # palatal approx — higher F1
    'z'  : f1f2_norm( 250, 2100),   # ś sibilant

    # Retroflex — F2 locus ~1100 (sub-dental)
    'T'  : f1f2_norm( 150, 1100),   # ṭ
    'Q'  : f1f2_norm( 150, 1100),   # ṭh
    'D'  : f1f2_norm( 150, 1100),   # ḍ
    'X'  : f1f2_norm( 150, 1100),   # ḍh
    'r'  : f1f2_norm( 400, 1100),   # retroflex liquid
    'x'  : f1f2_norm( 250, 1100),   # ṣ retroflex sibilant

    # Dental — F2 locus ~1800
    't'  : f1f2_norm( 150, 1800),
    'H'  : f1f2_norm( 150, 1800),   # th
    'd'  : f1f2_norm( 150, 1800),
    'W'  : f1f2_norm( 150, 1800),   # dh
    'n'  : f1f2_norm( 200, 1800),   # dental nasal
    'l'  : f1f2_norm( 400, 1800),   # dental lateral
    's'  : f1f2_norm( 250, 1800),   # sibilant

    # Labial — F2 locus ~800 (lowest)
    'p'  : f1f2_norm( 150,  800),
    'P'  : f1f2_norm( 150,  800),   # ph
    'b'  : f1f2_norm( 150,  800),
    'B'  : f1f2_norm( 150,  800),   # bh
    'm'  : f1f2_norm( 200,  800),   # labial nasal
    'v'  : f1f2_norm( 400,  800),   # labio-dental approx
}

# ─────────────────────────────────────────────────────────────────
# 2.  PHONEME VECTORS 16D (same as v12/v13)
# ─────────────────────────────────────────────────────────────────

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
    'k'  : [0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    'K'  : [0.0,  0.0,  1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 0.0, 0.0, 0.0],
    'g'  : [0.0,  0.0,  0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 0.0, 0.0, 0.0],
    'G'  : [0.0,  0.0,  1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 0.0, 0.0, 0.0],
    'N'  : [0.0,  0.0,  0.0, 1.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.5,  0.5, 0.5,  0.0, 0.0, 0.0, 1.0],
    'h'  : [0.0,  0.0,  0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.3,  1.0, 0.65, 0.0, 0.0, 0.0, 0.0],
    'c'  : [0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.0, 0.0],
    'C'  : [0.25, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],
    'j'  : [0.25, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 1.0, 0.0, 0.0],
    'J'  : [0.25, 0.25, 1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],
    'y'  : [0.25, 0.25, 0.0, 1.0, 0.0, 0.8, 0.0,  0.0, 1.0, 0.8,  1.0, 0.85, 0.0, 1.0, 0.0, 0.4],
    'z'  : [0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.2,  1.0, 0.55, 1.0, 1.0, 0.0, 0.0],
    'T'  : [0.5,  0.5,  0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.0, 0.0],
    'Q'  : [0.5,  0.5,  1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],
    'D'  : [0.5,  0.5,  0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 1.0, 0.0, 0.0],
    'X'  : [0.5,  0.5,  1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],
    'r'  : [0.5,  0.5,  0.0, 1.0, 0.0, 0.3, 0.0,  0.0, 1.0, 0.8,  1.0, 0.75, 0.0, 1.0, 0.0, 0.4],
    'x'  : [0.5,  0.5,  0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.2,  1.0, 0.55, 1.0, 1.0, 0.0, 0.0],
    't'  : [0.75, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.0, 0.0],
    'H'  : [0.75, 0.75, 1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],
    'd'  : [0.75, 0.75, 0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 1.0, 0.0, 0.0],
    'W'  : [0.75, 0.75, 1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 1.0, 0.0, 0.0],
    'n'  : [0.75, 0.75, 0.0, 1.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.5,  0.5, 0.5,  0.0, 1.0, 0.0, 1.0],
    'l'  : [0.75, 0.7,  0.0, 1.0, 0.0, 0.3, 0.0,  0.0, 1.0, 0.8,  1.0, 0.75, 0.0, 1.0, 0.0, 0.4],
    's'  : [0.75, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.2,  1.0, 0.55, 1.0, 1.0, 0.0, 0.0],
    'p'  : [1.0,  1.0,  0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0, 0.0],
    'P'  : [1.0,  1.0,  1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.35, 0.0, 0.0, 0.0, 0.0],
    'b'  : [1.0,  1.0,  0.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.2,  0.0, 0.0, 0.0, 0.0],
    'B'  : [1.0,  1.0,  1.0, 1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.1,  0.0, 0.35, 0.0, 0.0, 0.0, 0.0],
    'm'  : [1.0,  1.0,  0.0, 1.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.5,  0.5, 0.5,  0.0, 0.0, 0.0, 1.0],
    'v'  : [1.0,  1.0,  0.0, 1.0, 0.0, 0.2, 1.0,  0.0, 1.0, 0.7,  1.0, 0.85, 0.0, 0.0, 0.0, 0.4],
}

# ─────────────────────────────────────────────────────────────────
# 3.  PRATYAHARA CLASSES (8D, same as v13)
# ─────────────────────────────────────────────────────────────────

AC   = set(['a','A','i','I','u','U','R','e','o'])
HAL  = set(PHONEME_VECTORS_16.keys()) - AC
YAN  = set(['y','r','l','v'])
JHAL = set(['K','G','C','J','Q','X','H','W','P','B','z','x','s','h'])
SAL  = set(['z','x','s','h'])
JASH = set(['g','j','D','d','b'])
NAM  = set(['N','n','m'])
AK   = set(['a','i','u','R'])
PRATYAHARA_SETS  = [AC, HAL, YAN, JHAL, SAL, JASH, NAM, AK]
PRATYAHARA_NAMES = ['AC','HAL','YAN','JHAl','SAL','JASh','NAM','AK']

TRANSLIT = {
    'A':'A','I':'I','U':'U','R':'R',
    'kh':'K','gh':'G','ch':'C','jh':'J',
    'Th':'Q','Dh':'X','th':'H','dh':'W',
    'ph':'P','bh':'B','sh':'z','sh2':'x','ng':'N',
}

def root_to_chars(root_str):
    chars, i = [], 0
    s = root_str
    while i < len(s):
        if i+1 < len(s) and s[i:i+2] in TRANSLIT:
            chars.append(TRANSLIT[s[i:i+2]])
            i += 2
        else:
            chars.append(TRANSLIT.get(s[i], s[i]))
            i += 1
    return chars

def pratyahara_vec(ph):
    return [1.0 if ph in S else 0.0 for S in PRATYAHARA_SETS]

# ─────────────────────────────────────────────────────────────────
# 4.  EMBED_26: 16D acoustic + 2D formant + 8D Pratyahara
#     (NO 5D locus one-hot — replaced by F1/F2)
# ─────────────────────────────────────────────────────────────────

IN_DIM = 26

def embed_26(ph_char):
    """26D vector: 16D acoustic | 2D formant (F1,F2) | 8D Pratyahara."""
    ac   = list(PHONEME_VECTORS_16.get(ph_char, [0.5]*16))
    form = FORMANT_F1F2.get(ph_char, [0.4, 0.4])   # fallback: mid-vowel
    prat = pratyahara_vec(ph_char)
    return np.array(ac + form + prat, dtype=np.float32)

def root_vec_26(root_str):
    """Average-pool 26D vectors across all phonemes in root."""
    chars = root_to_chars(root_str)
    if not chars:
        return np.zeros(IN_DIM, dtype=np.float32)
    return np.mean([embed_26(c) for c in chars], axis=0)

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ─────────────────────────────────────────────────────────────────
# 5.  LOAD BENCHMARK DATA
# ─────────────────────────────────────────────────────────────────

DATA = r'c:\Users\amiku\Downloads\AI Research New Paradigm\SampleData'
t1 = pd.read_csv(f'{DATA}/task1_axis_prediction.csv')
t2 = pd.read_csv(f'{DATA}/task2_phonological_siblings.csv')
t7 = pd.read_csv(f'{DATA}/task7_triplets.csv')
t8 = pd.read_csv(f'{DATA}/task8_phonation.csv')

# Build all 150 root vectors
vecs15 = np.array([root_vec_26(r['root']) for _, r in t1.iterrows()])
labs15 = t1['actual_axis'].values
loci15 = t1['locus'].values
roots15 = t1['root'].values

print(f"\nLoaded {len(t1)} roots. IN_DIM={IN_DIM}")
print(f"Axes: {dict(t1['actual_axis'].value_counts())}")

le = LabelEncoder()
ids = le.fit_transform(labs15)

scaler = StandardScaler()
norm15 = scaler.fit_transform(vecs15)

# ─────────────────────────────────────────────────────────────────
# 6.  FORMANT SPACE GEOMETRY DIAGNOSTIC
#     Key test: does formant subspace separate axes BETTER than loci?
# ─────────────────────────────────────────────────────────────────

print("\n[Geometry Diagnostic] F2 locus discrimination vs axis discrimination")
formant_vecs = vecs15[:, 16:18]  # dims 16-17 = F1, F2

# Mean F2 per locus (should show 5 distinct values if locus theory holds)
print("  Mean F2 per locus (normalized, range=0:labial to 1:palatal):")
locus_order = ['LABIAL','RETROFLEX','VELAR','DENTAL','PALATE','CEREBRAL','THROAT']
for loc in ['LABIAL','CEREBRAL','THROAT','DENTAL','PALATE']:
    m = loci15 == loc
    if m.any():
        print(f"    {loc:12s}: F1={formant_vecs[m,0].mean():.3f}  F2={formant_vecs[m,1].mean():.3f}")

print("  Mean F2 per axis (should differ if formant = semantic axis):")
for ax in le.classes_:
    m = labs15 == ax
    print(f"    {ax:6s}: F1={formant_vecs[m,0].mean():.3f}  F2={formant_vecs[m,1].mean():.3f}  n={m.sum()}")

# ─────────────────────────────────────────────────────────────────
# 7.  TASK 1 — ARI on 26D embedding
# ─────────────────────────────────────────────────────────────────

print("\n[Task 1] Axis ARI — KMeans on raw 26D embedding")
best_ari_axis, best_k = -1, 5
for k in range(5, 9):
    pred = KMeans(n_clusters=k, random_state=42, n_init=15).fit_predict(norm15)
    a    = adjusted_rand_score(ids, pred)
    if a > best_ari_axis:
        best_ari_axis, best_k = a, k
best_pred = KMeans(n_clusters=best_k, random_state=42, n_init=15).fit_predict(norm15)
nmi_axis  = normalized_mutual_info_score(ids, best_pred)

# Also test Locus ARI for comparison
loci_enc = LabelEncoder().fit_transform(loci15)
best_ari_locus = -1
for k in range(5, 9):
    pred_l = KMeans(n_clusters=k, random_state=42, n_init=15).fit_predict(norm15)
    a_l    = adjusted_rand_score(loci_enc, pred_l)
    if a_l > best_ari_locus:
        best_ari_locus = a_l

print(f"  ARI (axis)  = {best_ari_axis:.4f}  (v13: 0.018, v10b: 0.182)  k={best_k}")
print(f"  ARI (locus) = {best_ari_locus:.4f}  (diagnostic: was 0.082 in v12)")
print(f"  NMI         = {nmi_axis:.4f}")
print(f"  Locus-dominance: {'YES (locus>>axis)' if best_ari_locus > best_ari_axis * 2 else 'REDUCED (axis closer to locus)'}")

print(f"\n  Per-axis purity:")
for ax in le.classes_:
    m    = labs15 == ax
    pf   = best_pred[m]
    dom  = np.bincount(pf).argmax()
    pur  = np.sum(pf==dom)/m.sum()
    print(f"    {ax}: n={m.sum():3d}  purity={pur:.2f}")

# ─────────────────────────────────────────────────────────────────
# 8.  TASK 2 — Phonological siblings (cosine threshold)
# ─────────────────────────────────────────────────────────────────

print("\n[Task 2] Phonological siblings — cosine sim binary accuracy")
locus_col = 'locus' if 'locus' in t2.columns else 'shared_locus'
same_col  = [c for c in t2.columns if 'same' in c.lower()][0]

sims2, labels2 = [], []
for _, r in t2.iterrows():
    va = root_vec_26(r['root_a'])
    vb = root_vec_26(r['root_b'])
    sims2.append(cosine_sim(va, vb))
    labels2.append(bool(r[same_col]))

sims2, labels2 = np.array(sims2), np.array(labels2)
best_acc2, best_thr2 = 0, 0.5
for thr in np.linspace(sims2.min(), sims2.max(), 50):
    acc = np.mean((sims2 >= thr) == labels2)
    if acc > best_acc2:
        best_acc2, best_thr2 = acc, thr

corr2, _ = pointbiserialr(labels2.astype(float), sims2)
print(f"  Acc@thr = {best_acc2:.4f}  (v14: 0.720)  sim_same={sims2[labels2].mean():.3f}  sim_diff={sims2[~labels2].mean():.3f}")
print(f"  Locus-dominance gap: diff={sims2[~labels2].mean():.3f} vs same={sims2[labels2].mean():.3f}  "
      f"({'FIXED: same>diff' if sims2[labels2].mean() > sims2[~labels2].mean() else 'still inverted'})")

# ─────────────────────────────────────────────────────────────────
# 9.  TASK 7 — Triplets
# ─────────────────────────────────────────────────────────────────

print("\n[Task 7] Triplets — cosine triplet accuracy")
root_to_locus_t1 = dict(zip(t1['root'], t1['locus']))
correct7, total7 = 0, 0
for _, r in t7.iterrows():
    anc = r['anchor']; oa = r['option_a']; ob = r['option_b']
    va  = root_vec_26(anc)
    voa = root_vec_26(oa)
    vob = root_vec_26(ob)
    sa  = cosine_sim(va, voa)
    sb  = cosine_sim(va, vob)
    pred = 'A' if sa > sb else 'B'
    correct7 += int(pred == r['correct_option'])
    total7   += 1
    anc_s = anc.encode('ascii','replace').decode('ascii')
    oa_s  = oa.encode('ascii','replace').decode('ascii')
    ob_s  = ob.encode('ascii','replace').decode('ascii')
    ok = pred == r['correct_option']
    print(f"    [{'OK' if ok else 'XX'}] {anc_s:8s} vs ({oa_s}/{ob_s})  "
          f"sim_a={sa:.3f} sim_b={sb:.3f}  pred={pred} correct={r['correct_option']}")

acc7 = correct7 / max(total7, 1)
print(f"  Triplet acc = {acc7:.4f}  (v14: 0.750)")

# ─────────────────────────────────────────────────────────────────
# 10.  TASK 8 — Phonation
# ─────────────────────────────────────────────────────────────────

print("\n[Task 8] Phonation — voicing discrimination")
correct8 = 0
for _, r in t8.iterrows():
    parts = r['pair_id'].split('_vs_')
    if len(parts) != 2:
        continue
    ca = root_to_chars(parts[0])
    cb = root_to_chars(parts[1])
    def voiced(chars):
        return np.mean([PHONEME_VECTORS_16.get(c,[0]*16)[3] for c in chars]) if chars else 0
    pred = 'A' if voiced(ca) > voiced(cb) else 'B'
    ok   = pred == r['correct_option']
    correct8 += int(ok)
    print(f"    {'OK' if ok else 'XX'} {r['pair_id']:15s}  pred={pred} correct={r['correct_option']}")
acc8 = correct8 / max(len(t8), 1)
print(f"  Phonation acc = {acc8:.4f}  (v14: 1.000)")

# ─────────────────────────────────────────────────────────────────
# 11.  ABLATION: formant-only (2D) vs pratyahara-only (8D) vs full 26D
# ─────────────────────────────────────────────────────────────────

print("\n[Ablation] ARI across sub-spaces:")
ablation_results = {}
subspaces = {
    '16D acoustic only' : (0,  16),
    '2D formant only'   : (16, 18),
    '8D Pratyahara only': (18, 26),
    '18D acous+form'    : (0,  18),
    '24D acous+prat'    : (list(range(0,16))+list(range(18,26)), None),
    '26D full (v15)'    : (0,  26),
}

for label, span in subspaces.items():
    if isinstance(span[0], list):
        sub = vecs15[:, span[0]]
    else:
        sub = vecs15[:, span[0]:span[1]]
    sub_n = StandardScaler().fit_transform(sub)
    best_a, best_k2 = -1, 5
    for k in range(5, 9):
        pr = KMeans(n_clusters=k, random_state=42, n_init=15).fit_predict(sub_n)
        a  = adjusted_rand_score(ids, pr)
        if a > best_a:
            best_a, best_k2 = a, k
    ablation_results[label] = best_a
    print(f"  {label:26s} : ARI={best_a:.4f}")

# ─────────────────────────────────────────────────────────────────
# 12.  PROGRESSION TABLE
# ─────────────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"ARI PROGRESSION — T1 Axis Prediction (150 roots):")
print(f"{'='*65}")
versions = [
    ("v10b (10D, 30 roots)",    0.182, "BCM Receiver Model (30-root set)"),
    ("v12  (21D: +locus OH)",   0.026, "BCM COLLAPSE — 150-root instability"),
    ("v13  (29D: +Pratyahara)", 0.018, "Raw 26D baseline = locus-dominated"),
    ("v15  (26D: +F1/F2)",      best_ari_axis, "Formant replace locus one-hot"),
]
for vname, ari, note in versions:
    bar = '#' * int(ari * 100)
    print(f"  {vname:30s} ARI={ari:.3f} |{bar}  {note}")
print(f"{'='*65}")

# ─────────────────────────────────────────────────────────────────
# 13.  PLOTS
# ─────────────────────────────────────────────────────────────────

pca2   = PCA(n_components=2)
low    = pca2.fit_transform(norm15)

# Formant 2D space
f1_vals = vecs15[:, 16]
f2_vals = vecs15[:, 17]

axis_colors  = {'EXP':'#ff6b6b','TRN':'#ffd93d','MOT':'#6bcb77','SEP':'#4d96ff','CNT':'#c77dff'}
locus_colors = {'THROAT':'#ff4444','PALATE':'#ffaa00','CEREBRAL':'#44cc44',
                'DENTAL':'#4444ff','LABIAL':'#ff44ff'}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    f"DDIN v15: Acoustic Formant Grounding (26D = 16D acoustic + 2D F1/F2 + 8D Pratyahara)\n"
    f"ARI_axis={best_ari_axis:.4f} | ARI_locus={best_ari_locus:.4f} | NMI={nmi_axis:.4f}\n"
    f"Locus-dominance: {'REDUCED' if best_ari_locus < best_ari_axis*3 else 'PERSISTS'}",
    fontsize=11
)

# Plot A: PCA-2D by axis
ax = axes[0, 0]
ax.set_title("PCA-2D by Axis (v15, 26D)")
for axname in np.unique(labs15):
    m = labs15 == axname
    ax.scatter(low[m,0], low[m,1], label=axname, alpha=0.65, s=30,
               color=axis_colors.get(axname,'#888'))
ax.legend(fontsize=8)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

# Plot B: PCA-2D by locus
ax2 = axes[0, 1]
ax2.set_title("PCA-2D by Locus (locus-dominance check)")
for loc in ['THROAT','PALATE','CEREBRAL','DENTAL','LABIAL']:
    m = loci15 == loc
    ax2.scatter(low[m,0], low[m,1], label=loc, alpha=0.65, s=30,
                color=locus_colors[loc])
ax2.legend(fontsize=8)
ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")

# Plot C: Formant space (F1 vs F2) by axis — THE KEY DIAGNOSTIC
ax3 = axes[1, 0]
ax3.set_title("Formant Space (F1 vs F2) by Phenomenological Axis\n(v15 hypothesis: axis clusters in vowel space)")
for axname in np.unique(labs15):
    m = labs15 == axname
    ax3.scatter(f2_vals[m], f1_vals[m], label=axname, alpha=0.6, s=35,
                color=axis_colors.get(axname,'#888'))
ax3.invert_yaxis()          # F1 low = high vowel (i,u) → top
ax3.set_xlabel("F2 (normalized, low=labial, high=palatal)")
ax3.set_ylabel("F1 (normalized, low=high vowel)")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)
# Mark locus regions
ax3.axvline(x=0.23, color='gray', alpha=0.3, linestyle='--')  # labial
ax3.axvline(x=0.62, color='gray', alpha=0.3, linestyle='--')  # dental/palate

# Plot D: Ablation bar chart
ax4 = axes[1, 1]
labels_abl = list(ablation_results.keys())
vals_abl   = list(ablation_results.values())
bar_colors_abl = ['#4d96ff' if v > 0.02 else '#ff9944' for v in vals_abl]
bars4 = ax4.barh(labels_abl, vals_abl, color=bar_colors_abl, alpha=0.85)
ax4.axvline(x=1/5, color='red', linestyle='--', alpha=0.5, label='T1 chance=0.2')
ax4.axvline(x=0.018, color='orange', linestyle=':', alpha=0.7, label='v13 baseline')
ax4.set_xlabel("ARI (axis prediction)")
ax4.set_title("ARI by Sub-Space (ablation)")
ax4.legend(fontsize=8)
for bar, v in zip(bars4, vals_abl):
    ax4.text(v + 0.001, bar.get_y() + bar.get_height()/2,
             f'{v:.4f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('ddin_v15_formant_grounding.png', dpi=150, bbox_inches='tight')
print("\nSaved: ddin_v15_formant_grounding.png")

# Save embedding
np.save('ddin_v15_norm_codes.npy',  norm15)
np.save('ddin_v15_raw_codes.npy',   vecs15)
np.save('ddin_v15_labels_axis.npy', labs15.astype(str))
np.save('ddin_v15_labels_locus.npy',loci15.astype(str))
np.save('ddin_v15_roots.npy',       roots15.astype(str))
print("Saved: v15 embedding arrays")

print(f"\n{'='*65}")
print(f"v15 COMPLETE")
print(f"  ARI (axis)  : {best_ari_axis:.4f}")
print(f"  ARI (locus) : {best_ari_locus:.4f}")
print(f"  Locus-axis gap: {best_ari_locus - best_ari_axis:+.4f}  (v12: +0.056, target: <+0.01)")
if best_ari_axis > 0.018:
    print(f"  IMPROVEMENT over v13 baseline (0.018): +{best_ari_axis-0.018:.4f}")
else:
    print(f"  No improvement over v13 baseline. Formant space alone insufficient.")
print(f"  Best ablation sub-space: {max(ablation_results, key=ablation_results.get)}")
print(f"{'='*65}")
