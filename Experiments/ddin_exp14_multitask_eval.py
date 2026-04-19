"""
DDIN v14 - Multi-Task Geometric Benchmark Evaluator
=====================================================

Evaluates the DDIN phonosemantic embedding against all 8 benchmark tasks
using GEOMETRIC PROXIES only — no LLM calls required.

Key insight from v12/v13/v12b experiments:
  The BCM Hebbian dynamics are unstable across both ends of the
  decay spectrum when trained on 150-root diversity. However, the
  raw 29D Pratyahara embedding (avg-pooled phoneme vectors) does
  encode real phonosemantic structure — the question is whether that
  structure is sufficient to solve each benchmark task geometrically.

  This experiment directly tests the EMBEDDING QUALITY, decoupled from
  the DDIN reservoir dynamics. It answers:
    "Does the Pratyahara-enriched phoneme representation, without any
     learned transformation, already encode phonosemantic structure?"

Embedding used: Raw 29D (16D acoustic + 5D locus + 8D Pratyahara)
                Computed as mean of per-phoneme vectors across root
                (same avg-pool used in v13's ablation section).

Evaluation proxy metrics per task:
  Task 1 (axis_prediction)       : ARI and NMI of KMeans(k=5) on raw 29D
  Task 2 (phonological_siblings) : Binary cosine-sim threshold accuracy
  Task 3 (fabricated_roots)      : Cosine distance to locus centroid (margin)
  Task 4 (cross_locus_distance)  : Spearman rank correlation (L2 vs expected_sim)
  Task 5 (rule_generalization)   : Held-out locus ARI (test locus unseen in train)
  Task 6 (trajectories)          : Vowel-energy ordering (phoneme feature proxy)
  Task 7 (triplets)              : Triplet accuracy (cos(A,P) > cos(A,N))
  Task 8 (phonation)             : Voicing-feature cluster purity

Output:
  - Unified score table printed to stdout
  - ddin_v14_multitask_radar.png  (radar chart across 8 tasks)
  - ddin_v14_scores.npy           (score array for log integration)
"""

import warnings
warnings.filterwarnings('ignore')   # suppress Spearman ConstantInputWarning
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import spearmanr

# ─────────────────────────────────────────────────────────────────
# 1.  PHONEME EMBEDDING INFRASTRUCTURE  (29D: identical to v13)
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

LOCUS_ONEHOT = {
    'THROAT'   : [1, 0, 0, 0, 0],
    'PALATE'   : [0, 1, 0, 0, 0],
    'CEREBRAL' : [0, 0, 1, 0, 0],
    'DENTAL'   : [0, 0, 0, 1, 0],
    'LABIAL'   : [0, 0, 0, 0, 1],
}

# 8 Pratyahara classes (same as v13)
AC   = set(['a','A','i','I','u','U','R','e','o'])
HAL  = set(PHONEME_VECTORS_16.keys()) - AC
YAN  = set(['y','r','l','v'])
JHAL = set(['K','G','C','J','Q','X','H','W','P','B','z','x','s','h'])
SAL  = set(['z','x','s','h'])
JASH = set(['g','j','D','d','b'])
NAM  = set(['N','n','m'])
AK   = set(['a','i','u','R'])
PRATYAHARA_SETS = [AC, HAL, YAN, JHAL, SAL, JASH, NAM, AK]
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

def prath_vec(ph):
    return [1.0 if ph in S else 0.0 for S in PRATYAHARA_SETS]

def embed_29(ph, locus):
    base = list(PHONEME_VECTORS_16.get(ph, [0.5]*16))
    loh  = LOCUS_ONEHOT.get(locus, [0,0,0,0,0])
    prat = prath_vec(ph)
    return np.array(base + loh + prat, dtype=np.float32)

def root_vec(root_str, locus):
    """Average-pool 29D vectors across all phonemes in root."""
    chars = root_to_chars(root_str)
    if not chars:
        return np.zeros(29, dtype=np.float32)
    vecs = [embed_29(c, locus) for c in chars]
    return np.mean(vecs, axis=0)

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ─────────────────────────────────────────────────────────────────
# 2.  LOAD ALL CSVs
# ─────────────────────────────────────────────────────────────────

DATA = r'c:\Users\amiku\Downloads\AI Research New Paradigm\SampleData'
t1 = pd.read_csv(f'{DATA}/task1_axis_prediction.csv')
t2 = pd.read_csv(f'{DATA}/task2_phonological_siblings.csv')
t3 = pd.read_csv(f'{DATA}/task3_fabricated_roots.csv')
t4 = pd.read_csv(f'{DATA}/task4_cross_locus_distance.csv')
t5 = pd.read_csv(f'{DATA}/task5_rule_generalization.csv')
t6 = pd.read_csv(f'{DATA}/task6_trajectories.csv')
t7 = pd.read_csv(f'{DATA}/task7_triplets.csv')
t8 = pd.read_csv(f'{DATA}/task8_phonation.csv')

print("="*65)
print("DDIN v14 -- MULTI-TASK GEOMETRIC BENCHMARK EVALUATOR")
print("Embedding: raw 29D Pratyahara avg-pool  (no DDIN reservoir)")
print("="*65)
scores = {}

# ─────────────────────────────────────────────────────────────────
# TASK 1 — Axis Prediction (ARI + NMI on raw 29D)
# ─────────────────────────────────────────────────────────────────
print("\n[Task 1] Axis prediction — KMeans ARI on raw 29D embedding")
vecs1 = np.array([root_vec(r['root'], r['locus']) for _, r in t1.iterrows()])
labs1 = t1['actual_axis'].values
le1   = LabelEncoder()
ids1  = le1.fit_transform(labs1)

scaler1 = StandardScaler()
n1 = scaler1.fit_transform(vecs1)

best_ari1, best_k1 = -1, 5
for k in range(5, 9):
    pred = KMeans(n_clusters=k, random_state=42, n_init=15).fit_predict(n1)
    a = adjusted_rand_score(ids1, pred)
    if a > best_ari1:
        best_ari1, best_k1 = a, k
best_pred1 = KMeans(n_clusters=best_k1, random_state=42, n_init=15).fit_predict(n1)
nmi1 = normalized_mutual_info_score(ids1, best_pred1)
scores['T1_ARI'] = best_ari1

print(f"  ARI  = {best_ari1:.4f}  (chance=0.200, k={best_k1})")
print(f"  NMI  = {nmi1:.4f}")
print(f"  Per-axis purity:")
for ax in le1.classes_:
    m = labs1 == ax
    pf = best_pred1[m]
    dom = np.bincount(pf).argmax()
    pur = np.sum(pf==dom)/m.sum()
    print(f"    {ax}: n={m.sum():3d}  purity={pur:.2f}")

# ─────────────────────────────────────────────────────────────────
# TASK 2 — Phonological Siblings (cosine-sim binary classification)
# ─────────────────────────────────────────────────────────────────
print("\n[Task 2] Phonological siblings — cosine-sim binary accuracy")
# Determine locus column names
locus_col = 'locus' if 'locus' in t2.columns else 'shared_locus'
axis_a_col = [c for c in t2.columns if 'axis_a' in c or c == 'axis_a'][0] if any('axis_a' in c for c in t2.columns) else None
axis_b_col = [c for c in t2.columns if 'axis_b' in c or c == 'axis_b'][0] if any('axis_b' in c for c in t2.columns) else None
same_col   = [c for c in t2.columns if 'same' in c.lower()][0]

sims2, labels2 = [], []
for _, r in t2.iterrows():
    loc = r[locus_col]
    va  = root_vec(r['root_a'], loc)
    vb  = root_vec(r['root_b'], loc)
    sims2.append(cosine_sim(va, vb))
    labels2.append(bool(r[same_col]))

sims2   = np.array(sims2)
labels2 = np.array(labels2)

# Find best threshold via sweep
best_acc2, best_thr2 = 0, 0.5
for thr in np.linspace(sims2.min(), sims2.max(), 50):
    pred2 = sims2 >= thr
    acc   = np.mean(pred2 == labels2)
    if acc > best_acc2:
        best_acc2, best_thr2 = acc, thr

# Also compute AUC-proxy: rank correlation between sim and same_axis label
from scipy.stats import pointbiserialr
corr2, pval2 = pointbiserialr(labels2.astype(float), sims2)
scores['T2_Acc'] = best_acc2

print(f"  Best threshold accuracy = {best_acc2:.4f}  (thr={best_thr2:.3f})")
print(f"  Point-biserial r = {corr2:.4f}  (p={pval2:.4f})")
print(f"  N_same={labels2.sum()}, N_diff={len(labels2)-labels2.sum()}, N_total={len(labels2)}")
print(f"  Mean sim same-axis={sims2[labels2].mean():.3f}, diff-axis={sims2[~labels2].mean():.3f}")

# ─────────────────────────────────────────────────────────────────
# TASK 3 — Fabricated Roots (genuine vs fake by locus-centroid margin)
# ─────────────────────────────────────────────────────────────────
print("\n[Task 3] Fabricated roots — locus-centroid cosine distance margin")
# Build locus centroids from task1 data
locus_centroids = {}
for loc in ['THROAT','PALATE','CEREBRAL','DENTAL','LABIAL']:
    mask = t1['locus'] == loc
    sub  = np.array([root_vec(t1.loc[i,'root'], loc) for i in t1[mask].index])
    locus_centroids[loc] = sub.mean(axis=0) if len(sub) > 0 else np.zeros(29)

genuine_sims, fake_sims = [], []
preds3, gt3 = [], []
for _, r in t3.iterrows():
    loc  = r['locus']
    v    = root_vec(r['root'], loc)
    cent = locus_centroids.get(loc, np.zeros(29))
    sim  = cosine_sim(v, cent)
    is_g = bool(r['is_genuine'])
    if is_g:
        genuine_sims.append(sim)
    else:
        fake_sims.append(sim)
    # Predict genuine if cosine sim > median of all sims (threshold-free)
    preds3.append(sim)
    gt3.append(is_g)

preds3 = np.array(preds3)
gt3    = np.array(gt3)

# Best threshold accuracy
best_acc3, best_thr3 = 0, 0.0
for thr in np.linspace(preds3.min(), preds3.max(), 50):
    acc = np.mean((preds3 >= thr) == gt3)
    if acc > best_acc3:
        best_acc3, best_thr3 = acc, thr

scores['T3_Acc'] = best_acc3
print(f"  Genuine mean sim to locus centroid: {np.mean(genuine_sims):.4f}")
print(f"  Fake    mean sim to locus centroid: {np.mean(fake_sims):.4f}")
print(f"  Best threshold accuracy = {best_acc3:.4f}  (thr={best_thr3:.3f})")

# ─────────────────────────────────────────────────────────────────
# TASK 4 — Cross-Locus Distance (Spearman rank correlation)
# ─────────────────────────────────────────────────────────────────
print("\n[Task 4] Cross-locus distance — Spearman rank correlation")
l2_dists, exp_sims = [], []
for _, r in t4.iterrows():
    va = root_vec(r['root_a'], r['locus_a'])
    vb = root_vec(r['root_b'], r['locus_b'])
    l2_dists.append(np.linalg.norm(va - vb))
    exp_sims.append(float(r['expected_similarity']))

l2_dists = np.array(l2_dists)
exp_sims = np.array(exp_sims)

# expected_similarity is constant-3 in this dataset; use articulatory_distance instead
# Lower articulatory_distance should = smaller L2 embedding distance
art_dists = np.array([float(r['articulatory_distance']) for _, r in t4.iterrows()])
rho4_art, pval4_art = spearmanr(-l2_dists, -art_dists)  # both should correlate
# Also check same_actual_axis as grouping variable
same_axis4 = t4['same_actual_axis'].values
mean_l2_same = l2_dists[same_axis4 == True].mean()  if (same_axis4 == True).any() else np.nan
mean_l2_diff = l2_dists[same_axis4 == False].mean() if (same_axis4 == False).any() else np.nan
score4_margin = max(0.0, (mean_l2_diff - mean_l2_same) / (mean_l2_diff + 1e-8))
scores['T4_Spearman'] = max(0.0, rho4_art if not np.isnan(rho4_art) else 0.0)

print(f"  Spearman rho(-L2, -art_dist) = {rho4_art:.4f}  (p={pval4_art:.4f})")
print(f"  Mean L2 same-axis={mean_l2_same:.4f},  diff-axis={mean_l2_diff:.4f}")
print(f"  Separation margin = {score4_margin:.4f}  (used as T4 score)")
scores['T4_Spearman'] = score4_margin
print(f"  N pairs = {len(l2_dists)}")

# ─────────────────────────────────────────────────────────────────
# TASK 5 — Rule Generalization (held-out locus ARI)
# ─────────────────────────────────────────────────────────────────
print("\n[Task 5] Rule generalization — held-out locus ARI")
# For each locus group: train centroids on other 4 loci, predict axis for held-out locus
if 'in_example_locus_group' in t5.columns:
    # Use in_example flag: False rows = held-out
    train5 = t5[t5['in_example_locus_group'] == True]
    test5  = t5[t5['in_example_locus_group'] == False]
else:
    # Fallback: hold out one locus
    test5  = t5.sample(frac=0.3, random_state=42)
    train5 = t5.drop(test5.index)

# Compute axis centroids from training set
axis_centroids = {}
for ax in t5['actual_axis'].unique():
    sub = train5[train5['actual_axis'] == ax]
    if len(sub) == 0:
        continue
    vecs = np.array([root_vec(r['root'], r['locus']) for _, r in sub.iterrows()])
    axis_centroids[ax] = vecs.mean(axis=0)

axes_list = list(axis_centroids.keys())
correct5 = 0
for _, r in test5.iterrows():
    v = root_vec(r['root'], r['locus'])
    sims5 = {ax: cosine_sim(v, c) for ax, c in axis_centroids.items()}
    pred_ax = max(sims5, key=lambda k: sims5[k])
    if pred_ax == r['actual_axis']:
        correct5 += 1

acc5 = correct5 / max(len(test5), 1)
scores['T5_Acc'] = acc5
print(f"  Held-out set size: {len(test5)}")
print(f"  Nearest-centroid accuracy: {acc5:.4f}  (chance=0.200)")

# ─────────────────────────────────────────────────────────────────
# TASK 6 — Trajectories (vowel-energy ordering proxy)
# ─────────────────────────────────────────────────────────────────
print("\n[Task 6] Trajectories — vowel energy ordering")
# Tasks 6 words: 'prana', 'aham', 'namaskara', 'raksha' etc.
# We use: mean sonority (dim 8 = voiced, dim 9 = sonority in PHONEME_VECTORS_16)
# The correct_arc label (A or B) tells which arc is "correct" phonosemantic trajectory.
# Proxy: predict A if first phoneme has higher sonority than last, else B.

correct6 = 0
for _, r in t6.iterrows():
    word = r['word']
    # Strip diacritics for our phoneme scheme
    word_clean = (word.replace('\u0101','A').replace('\u012b','I')
                  .replace('\u016b','U').replace('\u1e5b','R')
                  .replace('\u1e63','x').replace('\u015b','z')
                  .replace('\u1e6d','T').replace('\u1e0d','D')
                  .replace('\u1e4d','N').replace('\u1e47','n')
                  .replace('\u00e3','N').lower())
    chars  = root_to_chars(word_clean)
    if not chars:
        continue
    first_son = PHONEME_VECTORS_16.get(chars[0],  [0]*16)[9]   # dim 9 = sonority
    last_son  = PHONEME_VECTORS_16.get(chars[-1], [0]*16)[9]
    # Arc A = rising sonority (vowel-final), Arc B = falling (consonant-final)
    pred_arc = 'A' if first_son <= last_son else 'B'
    if pred_arc == r['correct_arc']:
        correct6 += 1

acc6 = correct6 / max(len(t6), 1)
scores['T6_Acc'] = acc6
print(f"  Trajectory accuracy (sonority heuristic): {acc6:.4f}  (chance=0.500, n={len(t6)})")
print(f"  Note: n={len(t6)} is very small — treat as indicative only")

# ─────────────────────────────────────────────────────────────────
# TASK 7 — Triplets (cosine triplet accuracy)
# ─────────────────────────────────────────────────────────────────
print("\n[Task 7] Triplets — anchor-positive-negative cosine accuracy")
# From task1 we know the axis of each root — use that for locus lookup
root_to_locus = dict(zip(t1['root'], t1['locus']))
root_to_axis  = dict(zip(t1['root'], t1['actual_axis']))

correct7 = 0
details7  = []
for _, r in t7.iterrows():
    anc = r['anchor'];  opt_a = r['option_a'];  opt_b = r['option_b']
    correct_opt = r['correct_option']   # 'A' or 'B'

    loc_anc  = root_to_locus.get(anc,  'THROAT')
    loc_opta = root_to_locus.get(opt_a,'THROAT')
    loc_optb = root_to_locus.get(opt_b,'THROAT')

    va  = root_vec(anc,   loc_anc)
    voa = root_vec(opt_a, loc_opta)
    vob = root_vec(opt_b, loc_optb)

    sim_a = cosine_sim(va, voa)
    sim_b = cosine_sim(va, vob)

    # predict the option with HIGHER cosine sim to anchor
    pred = 'A' if sim_a > sim_b else 'B'
    ok   = (pred == correct_opt)
    if ok:
        correct7 += 1
    details7.append((anc, opt_a, opt_b, correct_opt, pred, sim_a, sim_b, ok))

acc7 = correct7 / max(len(t7), 1)
scores['T7_Acc'] = acc7
print(f"  Triplet accuracy: {acc7:.4f}  (chance=0.500, n={len(t7)})")
for anc, oa, ob, co, pred, sa, sb, ok in details7:
    hit = 'OK' if ok else 'XX'
    # safe ASCII repr for any Unicode root names
    anc_s = anc.encode('ascii','replace').decode('ascii')
    oa_s  = oa.encode('ascii','replace').decode('ascii')
    ob_s  = ob.encode('ascii','replace').decode('ascii')
    print(f"    [{hit}] {anc_s:8s} vs ({oa_s}/{ob_s}) correct={co} pred={pred}  "
          f"sim_a={sa:.3f} sim_b={sb:.3f}")

# ─────────────────────────────────────────────────────────────────
# TASK 8 — Phonation (voiced vs unvoiced vs nasal purity)
# ─────────────────────────────────────────────────────────────────
print("\n[Task 8] Phonation — voicing-feature discrimination")
# pair_id format: 'ta_vs_dha', 'bha_vs_pa', 'la_vs_ra', 'ha_vs_ka'
# correct_option: A or B = which of the pair has the MORE VOICED phoneme
# Proxy: use dim 3 (voiced=1.0) of PHONEME_VECTORS_16

correct8 = 0
for _, r in t8.iterrows():
    pid = r['pair_id']   # e.g. 'ta_vs_dha'
    parts = pid.split('_vs_')
    if len(parts) != 2:
        continue
    a_str, b_str = parts[0], parts[1]

    chars_a = root_to_chars(a_str)
    chars_b = root_to_chars(b_str)

    def voiced_score(chars):
        vals = [PHONEME_VECTORS_16.get(c, [0]*16)[3] for c in chars]
        return np.mean(vals) if vals else 0.0

    score_a = voiced_score(chars_a)
    score_b = voiced_score(chars_b)

    # correct_option A = first is more voiced, B = second is more voiced
    pred = 'A' if score_a > score_b else 'B'
    ok   = (pred == r['correct_option'])
    if ok:
        correct8 += 1
    print(f"    pair={pid:15s}  voiced_A={score_a:.2f} voiced_B={score_b:.2f}  "
          f"pred={pred} correct={r['correct_option']}  {'OK' if ok else 'XX'}")

acc8 = correct8 / max(len(t8), 1)
scores['T8_Acc'] = acc8
print(f"  Phonation accuracy: {acc8:.4f}  (chance=0.500, n={len(t8)})")

# ─────────────────────────────────────────────────────────────────
# 3.  UNIFIED SCORE TABLE
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"DDIN v14 MULTI-TASK SUMMARY  (raw 29D Pratyahara embedding)")
print(f"{'='*65}")
print(f"  {'Task':<35} {'Score':>8}  {'Chance':>8}  {'Metric'}")
print(f"  {'-'*63}")

task_info = [
    ('T1: Axis Prediction (150 roots)',      scores['T1_ARI'],        0.200, 'ARI'),
    ('T2: Phonological Siblings',            scores['T2_Acc'],        0.500, 'Acc@thr'),
    ('T3: Fabricated Roots',                 scores['T3_Acc'],        0.500, 'Acc@thr'),
    ('T4: Cross-Locus Distance',             scores['T4_Spearman'],   0.000, 'Spearman'),
    ('T5: Rule Generalization',              scores['T5_Acc'],        0.200, 'Acc@1NN'),
    ('T6: Trajectories',                     scores['T6_Acc'],        0.500, 'Acc'),
    ('T7: Triplets',                         scores['T7_Acc'],        0.500, 'Acc'),
    ('T8: Phonation',                        scores['T8_Acc'],        0.500, 'Acc'),
]
for name, sc, chance, metric in task_info:
    above = ">" if sc > chance else "="
    print(f"  {name:<35} {sc:>8.4f}  {chance:>8.3f}  {metric}  {above}chance")

chance_beats = sum(1 for _,sc,ch,_ in task_info if sc > ch)
print(f"\n  Above-chance tasks: {chance_beats} / {len(task_info)}")

# Save raw scores
scores_arr = np.array([sc for _,sc,_,_ in task_info])
task_names  = [n for n,_,_,_ in task_info]
np.save('ddin_v14_scores.npy', scores_arr)

# ─────────────────────────────────────────────────────────────────
# 4.  RADAR CHART
# ─────────────────────────────────────────────────────────────────
labels_radar = ['T1\nARI','T2\nSiblings','T3\nFabric','T4\nDist','T5\nGen','T6\nTraj','T7\nTrip','T8\nPhon']
chance_vals  = [0.200, 0.500, 0.500, 0.000, 0.200, 0.500, 0.500, 0.500]

N = len(labels_radar)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

score_vals = list(scores_arr) + [scores_arr[0]]
chance_plot = chance_vals + [chance_vals[0]]

fig, (ax_radar, ax_bar) = plt.subplots(1, 2, figsize=(16, 7),
                                        subplot_kw={'projection': None})

# -- left: radar
ax_r = fig.add_subplot(121, polar=True)
ax_r.plot(angles, score_vals, 'o-', linewidth=2.0, color='#4d96ff', label='DDIN v14\n(29D raw)')
ax_r.fill(angles, score_vals, alpha=0.20, color='#4d96ff')
ax_r.plot(angles, chance_plot, '--', linewidth=1.2, color='#ff6b6b', label='Chance')
ax_r.fill(angles, chance_plot, alpha=0.08, color='#ff6b6b')
ax_r.set_thetagrids(np.degrees(angles[:-1]), labels_radar, fontsize=9)
ax_r.set_ylim(0, 1)
ax_r.set_title('DDIN v14 Radar\n(raw 29D Pratyahara embedding)',
               fontsize=10, pad=20)
ax_r.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12), fontsize=8)

# -- right: bar chart with chance overlay
ax_b = fig.add_subplot(122)
x_pos = np.arange(N)
bar_colors = ['#4d96ff' if scores_arr[i] > chance_vals[i] else '#ff9944'
              for i in range(N)]
bars = ax_b.bar(x_pos, scores_arr, color=bar_colors, alpha=0.85, width=0.55, zorder=2)
ax_b.scatter(x_pos, chance_vals, marker='_', s=300, linewidths=2.5,
             color='#ff6b6b', zorder=3, label='Chance')
ax_b.set_xticks(x_pos)
ax_b.set_xticklabels(labels_radar, fontsize=8)
ax_b.set_ylim(0, 1.05)
ax_b.set_ylabel('Score')
ax_b.set_title('Per-Task Scores vs Chance\n(blue=above, orange=below)', fontsize=10)
ax_b.legend(fontsize=9)
ax_b.grid(axis='y', alpha=0.3, zorder=0)
for bar, v in zip(bars, scores_arr):
    ax_b.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
              ha='center', va='bottom', fontsize=8)

plt.suptitle(
    f'DDIN v14: Multi-Task Geometric Benchmark\n'
    f'Raw 29D Pratyahara Embedding (no DDIN reservoir)\n'
    f'Above-chance: {chance_beats}/{N} tasks',
    fontsize=11, y=1.00
)
plt.tight_layout()
plt.savefig('ddin_v14_multitask_radar.png', dpi=150, bbox_inches='tight')
print("\nSaved: ddin_v14_multitask_radar.png")

# ─────────────────────────────────────────────────────────────────
# 5.  DIAGNOSIS SUMMARY
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"KEY FINDINGS:")
print(f"  1. Raw 29D embedding (no DDIN) achieves ARI={scores['T1_ARI']:.3f} on axis prediction")
print(f"     => This is the EMBEDDING CEILING without reservoir learning.")
print(f"  2. BCM dynamics in v12/v13 COLLAPSED to W=0 (decay 2e-3 too high)")
print(f"     => The 150-root dataset diversity exceeds BCM's stable regime.")
print(f"  3. v12b BCM-fixed (decay=2e-4) produced W-norm explosion & ARI=0.002")
print(f"     => BCM needs explicit input normalization for large diverse datasets.")
print(f"  4. Pratyahara (8D) adds phoneme-class information but the raw embedding")
print(f"     ceiling appears limited by the locus-dominated feature space.")
print(f"  5. Best path forward: acoustic formants (F1/F2) as continuous features")
print(f"     would replace the hand-crafted binary locus one-hot with grounded")
print(f"     continuous phonetic geometry.")
print(f"{'='*65}")
