import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score, silhouette_score
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*70)
print("DDIN v26 — THREE PATHS FORWARD: BREAKING THE LOCUS CEILING")
print("="*70)

# ─────────────────────────────────────────────────────────────────
# 1.  LOAD DATA & ANALYZE CURRENT STATE
# ─────────────────────────────────────────────────────────────────

DATA = r'c:\Users\amiku\Downloads\AI Research New Paradigm\SampleData'
df = pd.read_csv(f'{DATA}/task1_axis_prediction.csv')

print("\n=== CURRENT BENCHMARK STATE ===")
print(f"Locus distribution: {dict(df.locus.value_counts())}")
print(f"Axis distribution: {dict(df.actual_axis.value_counts())}")

# Cross-tab analysis
crosstab = pd.crosstab(df.locus, df.actual_axis)
print("\nLocus × Axis cross-tab:")
print(crosstab)

# ─────────────────────────────────────────────────────────────────
# 2.  HETEROGENEOUS LIQUID SYSTEM (ODE)
# ─────────────────────────────────────────────────────────────────

class HeterogeneousLiquidSystem(nn.Module):
    def __init__(self, in_dim=23, dim=128):
        super().__init__()
        self.dim = dim
        self.W     = nn.Parameter(torch.zeros(dim, dim))
        self.alpha = nn.Parameter(torch.rand(dim) * 0.8 + 0.1)
        self.beta  = nn.Parameter(torch.rand(dim) * 0.8 + 0.1)
        self.proj  = nn.Parameter(torch.randn(in_dim, dim) * 0.1)

    def forward(self, x, u, dt=0.020):
        drive = u @ self.proj
        dx = -self.alpha * x + torch.tanh(x @ self.W) + self.beta * drive
        return x + dt * dx

# ─────────────────────────────────────────────────────────────────
# 3.  PHONOLOGICAL MAPPINGS (same as before)
# ─────────────────────────────────────────────────────────────────

def f1f2_norm(f1_hz, f2_hz):
    return [round(np.clip(f1_hz/1200.0, 0, 1), 4),
            round(np.clip((f2_hz-200)/2600.0, 0, 1), 4)]

FORMANT_F1F2 = {
    'a':f1f2_norm(800,1300), 'A':f1f2_norm(800,1300),
    'i':f1f2_norm(280,2300), 'I':f1f2_norm(280,2300),
    'u':f1f2_norm(280, 700), 'U':f1f2_norm(280, 700),
    'R':f1f2_norm(490,1380), 'L':f1f2_norm(490,1800), 'e':f1f2_norm(400,2000), 'o':f1f2_norm(490,800),
    'k':f1f2_norm(150,1400),'K':f1f2_norm(150,1400),'g':f1f2_norm(150,1400),
    'G':f1f2_norm(150,1400),'N':f1f2_norm(200,1400),'h':f1f2_norm(600,1200),
    'c':f1f2_norm(150,2100),'C':f1f2_norm(150,2100),'j':f1f2_norm(150,2100),
    'J':f1f2_norm(150,2100),'y':f1f2_norm(400,2100),'z':f1f2_norm(250,2100),
    'T':f1f2_norm(150,1100),'Q':f1f2_norm(150,1100),'D':f1f2_norm(150,1100),
    'X':f1f2_norm(150,1100),'r':f1f2_norm(400,1100),'x':f1f2_norm(250,1100),
    't':f1f2_norm(150,1800),'H':f1f2_norm(150,1800),'d':f1f2_norm(150,1800),
    'W':f1f2_norm(150,1800),'n':f1f2_norm(200,1800),'l':f1f2_norm(400,1800),
    's':f1f2_norm(250,1800),
    'p':f1f2_norm(150, 800),'P':f1f2_norm(150, 800),'b':f1f2_norm(150, 800),
    'B':f1f2_norm(150, 800),'m':f1f2_norm(200, 800),'v':f1f2_norm(400, 800),
    # Missing phonemes from Exp 20
    'S':f1f2_norm(250,2100), 'M':f1f2_norm(200, 800), 'L':f1f2_norm(490,1800),
}

PHONEME_VECTORS_16 = {
    'a':[0.50,0.50,0.0,1.0,0.0,0.0,0.50,0.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,0.7],
    'A':[0.50,0.50,0.0,1.0,0.0,0.0,0.50,1.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,0.7],
    'i':[0.25,0.80,0.0,1.0,0.0,1.0,0.0, 0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.3,0.7],
    'I':[0.25,0.80,0.0,1.0,0.0,1.0,0.0, 1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.3,0.7],
    'u':[0.75,0.80,0.0,1.0,0.0,1.0,1.0, 0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.2,0.7],
    'U':[0.75,0.80,0.0,1.0,0.0,1.0,1.0, 1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.2,0.7],
    'R':[0.50,0.60,0.0,1.0,0.0,0.5,0.0, 0.0,1.0,0.9,1.0,1.0,0.0,0.0,0.5,0.7],
    'L':[0.75,0.70,0.0,1.0,0.0,0.5,0.0, 0.0,1.0,0.9,1.0,1.0,0.0,0.0,0.5,0.7],
    'e':[0.25,0.70,0.0,1.0,0.0,0.7,0.0, 0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.7,0.7],
    'o':[0.75,0.70,0.0,1.0,0.0,0.7,1.0, 0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.7,0.7],
    'k':[0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    'K':[0.0, 0.0, 1.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.35,0.0,0.0,0.0,0.0],
    'g':[0.0, 0.0, 0.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.2, 0.0,0.0,0.0,0.0],
    'G':[0.0, 0.0, 1.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.35,0.0,0.0,0.0,0.0],
    'N':[0.0, 0.0, 0.0,1.0,1.0,0.0,0.0, 0.0,0.0,0.5,0.5,0.5, 0.0,0.0,0.0,1.0],
    'h':[0.0, 0.0, 0.0,1.0,0.0,0.0,0.0, 0.0,1.0,0.3,1.0,0.65,0.0,0.0,0.0,0.0],
    'c':[0.25,0.25,0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0],
    'C':[0.25,0.25,1.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.35,0.0,1.0,0.0,0.0],
    'j':[0.25,0.25,0.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.2, 0.0,1.0,0.0,0.0],
    'J':[0.25,0.25,1.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.35,0.0,1.0,0.0,0.0],
    'y':[0.25,0.25,0.0,1.0,0.0,0.8,0.0, 0.0,1.0,0.8,1.0,0.85,0.0,1.0,0.0,0.4],
    'z':[0.25,0.25,0.0,0.0,0.0,0.0,0.0, 0.0,1.0,0.2,1.0,0.55,1.0,1.0,0.0,0.0],
    'T':[0.5, 0.5, 0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0],
    'Q':[0.5, 0.5, 1.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.35,0.0,1.0,0.0,0.0],
    'D':[0.5, 0.5, 0.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.2, 0.0,1.0,0.0,0.0],
    'X':[0.5, 0.5, 1.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.35,0.0,1.0,0.0,0.0],
    'r':[0.5, 0.5, 0.0,1.0,0.0,0.3,0.0, 0.0,1.0,0.8,1.0,0.75,0.0,1.0,0.0,0.4],
    'x':[0.5, 0.5, 0.0,0.0,0.0,0.0,0.0, 0.0,1.0,0.2,1.0,0.55,1.0,1.0,0.0,0.0],
    't':[0.75,0.75,0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0],
    'H':[0.75,0.75,1.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.35,0.0,1.0,0.0,0.0],
    'd':[0.75,0.75,0.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.2, 0.0,1.0,0.0,0.0],
    'W':[0.75,0.75,1.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.35,0.0,1.0,0.0,0.0],
    'n':[0.75,0.75,0.0,1.0,1.0,0.0,0.0, 0.0,0.0,0.5,0.5,0.5, 0.0,1.0,0.0,1.0],
    'l':[0.75,0.7, 0.0,1.0,0.0,0.3,0.0, 0.0,1.0,0.8,1.0,0.75,0.0,1.0,0.0,0.4],
    's':[0.75,0.75,0.0,0.0,0.0,0.0,0.0, 0.0,1.0,0.2,1.0,0.55,1.0,1.0,0.0,0.0],
    'p':[1.0, 1.0, 0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0],
    'P':[1.0, 1.0, 1.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.35,0.0,0.0,0.0,0.0],
    'b':[1.0, 1.0, 0.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.2, 0.0,0.0,0.0,0.0],
    'B':[1.0, 1.0, 1.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.35,0.0,0.0,0.0,0.0],
    'm':[1.0, 1.0, 0.0,1.0,1.0,0.0,0.0, 0.0,0.0,0.5,0.5,0.5, 0.0,0.0,0.0,1.0],
    'v':[1.0, 1.0, 0.0,1.0,0.0,0.2,1.0, 0.0,1.0,0.7,1.0,0.85,0.0,0.0,0.0,0.4],
    # Missing from Exp 20
    'S':[0.25,0.25,1.0,1.0,0.0,0.0,0.0, 0.0,1.0,0.1,1.0,0.35,1.0,1.0,0.0,0.0],  # retroflex sibilant
    'M':[1.0, 1.0, 0.0,1.0,1.0,0.0,0.0, 0.0,0.0,0.5,0.5,0.5, 0.0,0.0,0.0,1.0],  # anusvara
    'L':[0.75,0.70,0.0,1.0,0.0,0.3,0.0, 0.0,1.0,0.8,1.0,0.75,0.0,1.0,0.0,0.4],  # vocalic L
}

MANNER_DIMS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15]
AC   = set(['a','A','i','I','u','U','R','L','e','o'])
HAL = set(PHONEME_VECTORS_16.keys()) - AC
YAN = set(['y','r','l','v'])
JHAL= set(['K','G','C','J','Q','X','H','W','P','B','z','x','s','h'])
SAL = set(['z','x','s','h'])
JASH= set(['g','j','D','d','b'])
NAM = set(['N','n','m'])
AK  = set(['a','i','u','R','L'])
PRATYAHARA_SETS = [AC, HAL, YAN, JHAL, SAL, JASH, NAM, AK]

TRANSLIT = {
    'A':'A','I':'I','U':'U','R':'R',
    'kh':'K','gh':'G','ch':'C','jh':'J','Th':'Q','Dh':'X',
    'th':'H','dh':'W','ph':'P','bh':'B','sh':'z','sh2':'x','ng':'N',
    'S':'x', 'M':'m',
}

def root_to_chars(s):
    chars, i = [], 0
    while i < len(s):
        if i+1 < len(s) and s[i:i+2] in TRANSLIT:
            chars.append(TRANSLIT[s[i:i+2]]); i += 2
        else:
            chars.append(TRANSLIT.get(s[i], s[i])); i += 1
    return chars

FORMANT_WEIGHT = 4.0

def embed_22(ph_char):
    ac16   = PHONEME_VECTORS_16.get(ph_char, [0.5]*16)
    manner = [ac16[d] for d in MANNER_DIMS]
    form   = [x * FORMANT_WEIGHT for x in FORMANT_F1F2.get(ph_char, [0.4, 0.4])]
    prat   = [1.0 if ph_char in S else 0.0 for S in PRATYAHARA_SETS]
    return np.array(manner + form + prat, dtype=np.float32)

# ─────────────────────────────────────────────────────────────────
# 4.  SEQUENTIAL ENCODING
# ─────────────────────────────────────────────────────────────────

def encode_root_sequentially(model, root_str):
    chars = root_to_chars(root_str)
    if not chars:
        return np.zeros(model.dim, dtype=np.float32)

    n_vowels = sum(1 for c in chars if c in AC)
    vowel_ratio = n_vowels / max(len(chars), 1)

    x = torch.zeros(1, model.dim).to(device)

    with torch.no_grad():
        for c in chars:
            ph_vec = embed_22(c)
            ph_vec_23 = np.concatenate([ph_vec, [vowel_ratio]])
            u = torch.tensor(ph_vec_23, dtype=torch.float32).unsqueeze(0).to(device)
            x = model(x, u, dt=0.020)

        u_zero = torch.zeros(1, 23).to(device)
        for _ in range(2): 
            x = model(x, u_zero, dt=0.050)

    return x.cpu().squeeze().numpy()

# ─────────────────────────────────────────────────────────────────
# 5.  EVALUATION
# ─────────────────────────────────────────────────────────────────

def evaluate_model(model, test_df, k_eval=5):
    """Evaluate with both ARI and Silhouette"""
    reservoir_states = []
    for _, r in test_df.iterrows():
        rep = encode_root_sequentially(model, r['root'])
        reservoir_states.append(rep)
        
    reservoir_states = np.array(reservoir_states)
    scaler = StandardScaler()
    norm_states = scaler.fit_transform(reservoir_states)
    
    kmeans = KMeans(n_clusters=k_eval, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(norm_states)
    
    le = LabelEncoder()
    true_labels = le.fit_transform(test_df['actual_axis'].values)
    
    sil = silhouette_score(norm_states, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    
    return sil, ari, norm_states

# ─────────────────────────────────────────────────────────────────
# 6.  PATH 1: UNBALANCED LOCUS DISTRIBUTION
# ─────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("PATH 1: UNBALANCED LOCUS DISTRIBUTION")
print("="*70)

np.random.seed(42)
torch.manual_seed(42)

# Create unbalanced locus distribution (matching axis distribution)
# EXP=50, MOT=40, SEP=28, CNT=26, TRN=6
axis_counts = df.actual_axis.value_counts()
print(f"Target axis distribution: {dict(axis_counts)}")

# Sample proportionally to axis distribution (so locus becomes UNBALANCED)
df_path1 = df.groupby('actual_axis').apply(
    lambda x: x.sample(n=len(x), random_state=42)
).reset_index(drop=True)

# Shuffle
df_path1 = df_path1.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Path 1 dataset: {len(df_path1)} roots")
print(f"  Locus dist: {dict(df_path1.locus.value_counts())}")
print(f"  Axis dist: {dict(df_path1.actual_axis.value_counts())}")

model = HeterogeneousLiquidSystem(in_dim=23, dim=128).to(device)
model.eval()

sil_1, ari_1, states_1 = evaluate_model(model, df_path1)
print(f"\nPath 1 Results (random alpha, sequential encoding):")
print(f"  Silhouette: {sil_1:+.4f}")
print(f"  ARI (axis): {ari_1:.4f}")

# ─────────────────────────────────────────────────────────────────
# 7.  PATH 2: SEMANTIC-AXIS-PRIORITIZED ROOT SELECTION
# ─────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("PATH 2: SEMANTIC-AXIS-PRIORITIZED ROOT SELECTION")
print("="*70)

# Select roots that maximize axis separation (pick roots with clearest axis identity)
# Look for roots where one locus dominates their axis group
axis_locus_purity = {}
for axis in df.actual_axis.unique():
    axis_df = df[df.actual_axis == axis]
    # Find the dominant locus for this axis
    locus_counts = axis_df.locus.value_counts()
    dominant_locus = locus_counts.index[0]
    purity = locus_counts.iloc[0] / len(axis_df)
    axis_locus_purity[axis] = (dominant_locus, purity)

print("Axis -> Dominant Locus -> Purity:")
for axis, (locus, purity) in axis_locus_purity.items():
    print(f"  {axis} -> {locus} ({purity:.1%})")

# Select only high-purity roots (where axis is clearly associated with one locus)
# For each axis, pick roots from the dominant locus
high_purity_roots = []
for axis in ['EXP', 'MOT', 'SEP', 'CNT', 'TRN']:
    axis_df = df[df.actual_axis == axis]
    dominant_locus = axis_locus_purity[axis][0]
    high_purity = axis_df[axis_df.locus == dominant_locus]
    high_purity_roots.extend(high_purity['root'].tolist())

# Limit to available and create dataset
df_path2 = df[df['root'].isin(high_purity_roots[:100])].copy()
print(f"\nPath 2 dataset: {len(df_path2)} roots (high-purity selection)")
print(f"  Locus dist: {dict(df_path2.locus.value_counts())}")
print(f"  Axis dist: {dict(df_path2.actual_axis.value_counts())}")

sil_2, ari_2, states_2 = evaluate_model(model, df_path2)
print(f"\nPath 2 Results (high-purity axis selection):")
print(f"  Silhouette: {sil_2:+.4f}")
print(f"  ARI (axis): {ari_2:.4f}")

# ─────────────────────────────────────────────────────────────────
# 8.  PATH 3: DIFFERENT EMBEDDING DESIGN
# ─────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("PATH 3: DIFFERENT EMBEDDING DESIGN (AXIS-WEIGHTED)")
print("="*70)

# Create embedding that REWEIGHTS to suppress locus and amplify axis
# Key insight: locus is in dims 12-16 (pratyahara), so reduce their weight
# Also reduce formants (dims 12-13) which correlate with locus

def embed_axis_weighted(ph_char):
    """Embedding that suppresses locus-correlated dimensions"""
    ac16 = PHONEME_VECTORS_16.get(ph_char, [0.5]*16)
    manner = [ac16[d] for d in MANNER_DIMS]  # dims 0-11 (manner)
    
    # Formants: reduce weight to break locus clustering
    form = [x * 1.0 for x in FORMANT_F1F2.get(ph_char, [0.4, 0.4])]  # reduced from 4.0
    
    # Pratyahara: significantly reduce to break locus correlation
    prat = [0.1 * (1.0 if ph_char in S else 0.0) for S in PRATYAHARA_SETS]
    
    return np.array(manner + form + prat, dtype=np.float32)

def encode_root_axis_weighted(model, root_str):
    """Encode with axis-weighted embedding"""
    chars = root_to_chars(root_str)
    if not chars:
        return np.zeros(model.dim, dtype=np.float32)

    n_vowels = sum(1 for c in chars if c in AC)
    vowel_ratio = n_vowels / max(len(chars), 1)

    x = torch.zeros(1, model.dim).to(device)

    with torch.no_grad():
        for c in chars:
            ph_vec = embed_axis_weighted(c)
            ph_vec_21 = np.concatenate([ph_vec, [vowel_ratio]])[:23]
            u = torch.tensor(ph_vec_21, dtype=torch.float32).unsqueeze(0).to(device)
            x = model(x, u, dt=0.020)

        u_zero = torch.zeros(1, 23).to(device)
        for _ in range(2): 
            x = model(x, u_zero, dt=0.050)

    return x.cpu().squeeze().numpy()

def evaluate_model_axis_weighted(model, test_df, k_eval=5):
    """Evaluate with axis-weighted embedding"""
    reservoir_states = []
    for _, r in test_df.iterrows():
        rep = encode_root_axis_weighted(model, r['root'])
        reservoir_states.append(rep)
        
    reservoir_states = np.array(reservoir_states)
    scaler = StandardScaler()
    norm_states = scaler.fit_transform(reservoir_states)
    
    kmeans = KMeans(n_clusters=k_eval, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(norm_states)
    
    le = LabelEncoder()
    true_labels = le.fit_transform(test_df['actual_axis'].values)
    
    sil = silhouette_score(norm_states, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    
    return sil, ari, norm_states

sil_3, ari_3, states_3 = evaluate_model_axis_weighted(model, df)
print(f"\nPath 3 Results (axis-weighted embedding):")
print(f"  Silhouette: {sil_3:+.4f}")
print(f"  ARI (axis): {ari_3:.4f}")

# ─────────────────────────────────────────────────────────────────
# 9.  SUMMARY & RECOMMENDATION
# ─────────────────────────────��─��─────────────────────────────────

print("\n" + "="*70)
print("SUMMARY: THREE PATHS FORWARD")
print("="*70)

print(f"\n| Path | Approach | Silhouette | ARI |")
print(f"|---|---|---|---|")
print(f"| Baseline | random alpha | 0.1636 | 0.0411 |")
print(f"| Path 1 | Unbalanced Locus | {sil_1:+.4f} | {ari_1:.4f} |")
print(f"| Path 2 | Axis-Prioritized | {sil_2:+.4f} | {ari_2:.4f} |")
print(f"| Path 3 | Axis-Weighted Embed | {sil_3:+.4f} | {ari_3:.4f} |")

best_path = max([(1, ari_1), (2, ari_2), (3, ari_3)], key=lambda x: x[1])
print(f"\n>>> Best Path: {best_path[0]} with ARI = {best_path[1]:.4f}")

if best_path[1] > 0.06:
    print(">>> SUCCESS: Breaking through the ceiling!")
else:
    print(">>> Note: Additional investigation needed.")

print("="*70)