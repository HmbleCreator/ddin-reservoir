import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score, silhouette_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*70)
print("DDIN v28B — STRUCTURED W INITIALIZATION (Architecture Ceiling Test)")
print("="*70)

# ─────────────────────────────────────────────────────────────────
# 1.  DATA
# ─────────────────────────────────────────────────────────────────

DATA = r'c:\Users\amiku\Downloads\AI Research New Paradigm\SampleData'
df = pd.read_csv(f'{DATA}/task1_axis_prediction.csv')

print(f"\nDataset: {len(df)} roots, {df.actual_axis.nunique()} axes")

# Map axis to integer
axis_to_idx = {'EXP': 0, 'TRN': 1, 'MOT': 2, 'SEP': 3, 'CNT': 4}

# ─────────────────────────────────────────────────────────────────
# 2.  MODEL WITH STRUCTURED W
# ─────────────────────────────────────────────────────────────────

class HeterogeneousLiquidSystemStructured(nn.Module):
    def __init__(self, in_dim=23, dim=128, W_init=None):
        super().__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.rand(dim) * 0.8 + 0.1)
        self.beta = nn.Parameter(torch.rand(dim) * 0.8 + 0.1)
        self.proj = nn.Parameter(torch.randn(in_dim, dim) * 0.1)
        
        # Structured W initialization
        if W_init is not None:
            self.W = nn.Parameter(W_init)
        else:
            self.W = nn.Parameter(torch.zeros(dim, dim))

    def forward(self, x, u, dt=0.020):
        drive = u @ self.proj
        dx = -self.alpha * x + torch.tanh(x @ self.W) + self.beta * drive
        return x + dt * dx

# ─────────────────────────────────────────────────────────────────
# 3.  PHONOLOGICAL MAPPINGS
# ─────────────────────────────────────────────────────────────────

def f1f2_norm(f1_hz, f2_hz):
    return [round(np.clip(f1_hz/1200.0, 0, 1), 4),
            round(np.clip((f2_hz-200)/2600.0, 0, 1), 4)]

FORMANT_F1F2 = {
    'a':f1f2_norm(800,1300), 'A':f1f2_norm(800,1300),
    'i':f1f2_norm(280,2300), 'I':f1f2_norm(280,2300),
    'u':f1f2_norm(280,700), 'U':f1f2_norm(280,700),
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
    'p':f1f2_norm(150,800),'P':f1f2_norm(150,800),'b':f1f2_norm(150,800),
    'B':f1f2_norm(150,800),'m':f1f2_norm(200,800),'v':f1f2_norm(400,800),
    'S':f1f2_norm(250,2100), 'M':f1f2_norm(200,800), 'L':f1f2_norm(490,1800),
}

PHONEME_VECTORS_16 = {
    'a':[0.50,0.50,0.0,1.0,0.0,0.0,0.50,0.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,0.7],
    'A':[0.50,0.50,0.0,1.0,0.0,0.0,0.50,1.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,0.7],
    'i':[0.25,0.80,0.0,1.0,0.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.3,0.7],
    'I':[0.25,0.80,0.0,1.0,0.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.3,0.7],
    'u':[0.75,0.80,0.0,1.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.2,0.7],
    'U':[0.75,0.80,0.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.2,0.7],
    'R':[0.50,0.60,0.0,1.0,0.0,0.5,0.0,0.0,1.0,0.9,1.0,1.0,0.0,0.0,0.5,0.7],
    'L':[0.75,0.70,0.0,1.0,0.0,0.5,0.0,0.0,1.0,0.9,1.0,1.0,0.0,0.0,0.5,0.7],
    'e':[0.25,0.70,0.0,1.0,0.0,0.7,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.7,0.7],
    'o':[0.75,0.70,0.0,1.0,0.0,0.7,1.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.7,0.7],
    'k':[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    'K':[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.35,0.0,0.0,0.0,0.0],
    'g':[0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.2,0.0,0.0,0.0,0.0],
    'G':[0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.35,0.0,0.0,0.0,0.0],
    'N':[0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.5,0.5,0.5,0.0,0.0,0.0,1.0],
    'h':[0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.3,1.0,0.65,0.0,0.0,0.0,0.0],
    'c':[0.25,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],
    'C':[0.25,0.25,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.35,0.0,1.0,0.0,0.0],
    'j':[0.25,0.25,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.2,0.0,1.0,0.0,0.0],
    'J':[0.25,0.25,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.35,0.0,1.0,0.0,0.0],
    'y':[0.25,0.25,0.0,1.0,0.0,0.8,0.0,0.0,1.0,0.8,1.0,0.85,0.0,1.0,0.0,0.4],
    'z':[0.25,0.25,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.2,1.0,0.55,1.0,1.0,0.0,0.0],
    'T':[0.5,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],
    'Q':[0.5,0.5,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.35,0.0,1.0,0.0,0.0],
    'D':[0.5,0.5,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.2,0.0,1.0,0.0,0.0],
    'X':[0.5,0.5,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.35,0.0,1.0,0.0,0.0],
    'r':[0.5,0.5,0.0,1.0,0.0,0.3,0.0,0.0,1.0,0.8,1.0,0.75,0.0,1.0,0.0,0.4],
    'x':[0.5,0.5,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.2,1.0,0.55,1.0,1.0,0.0,0.0],
    't':[0.75,0.75,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],
    'H':[0.75,0.75,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.35,0.0,1.0,0.0,0.0],
    'd':[0.75,0.75,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.2,0.0,1.0,0.0,0.0],
    'W':[0.75,0.75,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.35,0.0,1.0,0.0,0.0],
    'n':[0.75,0.75,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.5,0.5,0.5,0.0,1.0,0.0,1.0],
    'l':[0.75,0.7,0.0,1.0,0.0,0.3,0.0,0.0,1.0,0.8,1.0,0.75,0.0,1.0,0.0,0.4],
    's':[0.75,0.75,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.2,1.0,0.55,1.0,1.0,0.0,0.0],
    'p':[1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    'P':[1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.35,0.0,0.0,0.0,0.0],
    'b':[1.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.2,0.0,0.0,0.0,0.0],
    'B':[1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.35,0.0,0.0,0.0,0.0],
    'm':[1.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.5,0.5,0.5,0.0,0.0,0.0,1.0],
    'v':[1.0,1.0,0.0,1.0,0.0,0.2,1.0,0.0,1.0,0.7,1.0,0.85,0.0,0.0,0.0,0.4],
    'S':[0.25,0.25,1.0,1.0,0.0,0.0,0.0,0.0,1.0,0.1,1.0,0.35,1.0,1.0,0.0,0.0],
    'M':[1.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.5,0.5,0.5,0.0,0.0,0.0,1.0],
    'L':[0.75,0.70,0.0,1.0,0.0,0.3,0.0,0.0,1.0,0.8,1.0,0.75,0.0,1.0,0.0,0.4],
}

MANNER_DIMS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15]
AC = set(['a','A','i','I','u','U','R','L','e','o'])
PRATYAHARA_SETS = [AC, set(PHONEME_VECTORS_16.keys()) - AC, set(['y','r','l','v']),
                  set(['K','G','C','J','Q','X','H','W','P','B','z','x','s','h']),
                  set(['z','x','s','h']), set(['g','j','D','d','b']), set(['N','n','m']), set(['a','i','u','R','L'])]

TRANSLIT = {'A':'A','I':'I','U':'U','R':'R','kh':'K','gh':'G','ch':'C','jh':'J','Th':'Q','Dh':'X',
           'th':'H','dh':'W','ph':'P','bh':'B','sh':'z','sh2':'x','ng':'N','S':'x', 'M':'m'}

def root_to_chars(s):
    chars = []
    i = 0
    while i < len(s):
        if i+1 < len(s) and s[i:i+2] in TRANSLIT:
            chars.append(TRANSLIT[s[i:i+2]])
            i += 2
        else:
            chars.append(TRANSLIT.get(s[i], s[i]))
            i += 1
    return chars

FORMANT_WEIGHT = 4.0

def embed_22(ph_char):
    ac16 = PHONEME_VECTORS_16.get(ph_char, [0.5]*16)
    manner = [ac16[d] for d in MANNER_DIMS]
    form = [x * FORMANT_WEIGHT for x in FORMANT_F1F2.get(ph_char, [0.4, 0.4])]
    prat = [1.0 if ph_char in S else 0.0 for S in PRATYAHARA_SETS]
    return np.array(manner + form + prat, dtype=np.float32)

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
# 4.  CREATE STRUCTURED W
# ─────────────────────────────────────────────────────────────────

def create_structured_W(dim=128, n_clusters=5, sparsity=0.1, cross_inhibit=0.01):
    """
    Create a structured W matrix with:
    - Block-diagonal positive connections within semantic clusters
    - Weak cross-cluster inhibition
    - Sparsity within blocks
    """
    np.random.seed(42)
    
    # Divide neurons into clusters
    neurons_per_cluster = dim // n_clusters
    W = np.zeros((dim, dim))
    
    for c in range(n_clusters):
        start = c * neurons_per_cluster
        end = min((c + 1) * neurons_per_cluster, dim)
        
        # Within-cluster: sparse positive connections (promote same-cluster activation)
        for i in range(start, end):
            for j in range(start, end):
                if i != j and np.random.rand() < sparsity:
                    W[i, j] = np.random.rand() * 0.15 + 0.05  # [0.05, 0.20]
        
        # Cross-cluster: weak inhibition (prevent different-cluster activation)
        for c2 in range(n_clusters):
            if c2 != c:
                start2 = c2 * neurons_per_cluster
                end2 = min((c2 + 1) * neurons_per_cluster, dim)
                for i in range(start, end):
                    for j in range(start2, end2):
                        if np.random.rand() < cross_inhibit:
                            W[i, j] = -np.random.rand() * 0.02 - 0.01  # [-0.03, -0.01]
    
    return torch.tensor(W, dtype=torch.float32)

# ─────────────────────────────────────────────────────────────────
# 5.  EVALUATION
# ─────────────────────────────────────────────────────────────────

def encode_all_roots(model, test_df):
    states = []
    for _, r in test_df.iterrows():
        rep = encode_root_sequentially(model, r['root'])
        states.append(rep)
    return np.array(states)

def evaluate_model(model, test_df, k_eval=5):
    states = encode_all_roots(model, test_df)
    scaler = StandardScaler()
    states = scaler.fit_transform(states)
    
    kmeans = KMeans(n_clusters=k_eval, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(states)
    
    le = LabelEncoder()
    true_labels = le.fit_transform(test_df['actual_axis'].values)
    
    ari = adjusted_rand_score(true_labels, pred_labels)
    sil = silhouette_score(states, pred_labels)
    return ari, sil

# ─────────────────────────────────────────────────────────────────
# 6.  EXPERIMENT 28B: COMPARE W=0 vs STRUCTURED W
# ─────────────────────────────────────────────────────────────────

np.random.seed(42)
torch.manual_seed(42)

k_clusters = df['actual_axis'].nunique()

print("\n" + "="*70)
print("Testing different W structures...")
print("="*70)

results = []

# Test 1: W=0 (baseline)
print("\n[1] W = 0 (baseline)")
model_w0 = HeterogeneousLiquidSystemStructured(in_dim=23, dim=128, W_init=torch.zeros(128, 128)).to(device)
model_w0.eval()
ari_w0, sil_w0 = evaluate_model(model_w0, df, k_eval=k_clusters)
print(f"    ARI = {ari_w0:.4f}, Silhouette = {sil_w0:+.4f}")
results.append(("W=0", ari_w0, sil_w0))

# Test 2: Random W (control)
print("\n[2] W = random (control)")
W_rand = torch.randn(128, 128) * 0.05
model_rand = HeterogeneousLiquidSystemStructured(in_dim=23, dim=128, W_init=W_rand).to(device)
model_rand.eval()
ari_rand, sil_rand = evaluate_model(model_rand, df, k_eval=k_clusters)
print(f"    ARI = {ari_rand:.4f}, Silhouette = {sil_rand:+.4f}")
results.append(("Random W", ari_rand, sil_rand))

# Test 3: Structured W (block diagonal + cross inhibition)
print("\n[3] W = structured (block-diagonal + cross inhibition)")
W_structured = create_structured_W(dim=128, n_clusters=5, sparsity=0.15, cross_inhibit=0.02)
model_struct = HeterogeneousLiquidSystemStructured(in_dim=23, dim=128, W_init=W_structured).to(device)
model_struct.eval()
ari_struct, sil_struct = evaluate_model(model_struct, df, k_eval=k_clusters)
print(f"    ARI = {ari_struct:.4f}, Silhouette = {sil_struct:+.4f}")
results.append(("Structured W", ari_struct, sil_struct))

# Test 4: Higher structured connectivity
print("\n[4] W = structured (higher connectivity)")
W_struct_hi = create_structured_W(dim=128, n_clusters=5, sparsity=0.30, cross_inhibit=0.05)
model_struct_hi = HeterogeneousLiquidSystemStructured(in_dim=23, dim=128, W_init=W_struct_hi).to(device)
model_struct_hi.eval()
ari_struct_hi, sil_struct_hi = evaluate_model(model_struct_hi, df, k_eval=k_clusters)
print(f"    ARI = {ari_struct_hi:.4f}, Silhouette = {sil_struct_hi:+.4f}")
results.append(("Structured W (hi)", ari_struct_hi, sil_struct_hi))

# Test 5: 10 clusters (finer granularity)
print("\n[5] W = structured (10 clusters)")
W_struct_10 = create_structured_W(dim=128, n_clusters=10, sparsity=0.20, cross_inhibit=0.02)
model_struct_10 = HeterogeneousLiquidSystemStructured(in_dim=23, dim=128, W_init=W_struct_10).to(device)
model_struct_10.eval()
ari_struct_10, sil_struct_10 = evaluate_model(model_struct_10, df, k_eval=k_clusters)
print(f"    ARI = {ari_struct_10:.4f}, Silhouette = {sil_struct_10:+.4f}")
results.append(("Structured W (10cl)", ari_struct_10, sil_struct_10))

# ─────────────────────────────────────────────────────────────────
# 7.  SUMMARY
# ─────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("SUMMARY: W STRUCTURE COMPARISON")
print("="*70)

print(f"\n| W Structure | ARI | Silhouette |")
print(f"|---|---|---|")
for name, ari, sil in results:
    print(f"| {name} | {ari:.4f} | {sil:+.4f} |")

best = max(results, key=lambda x: x[1])
print(f"\n>>> Best: {best[0]} with ARI = {best[1]:.4f}")

if best[1] > 0.15:
    print("\n>>> SUCCESS! Architecture ceiling broken with structured W!")
elif best[1] > 0.08:
    print(f"\n>>> PARTIAL: ARI improved to {best[1]:.4f}")
else:
    print(f"\n>>> Below target. Ceiling confirmed at ARI ~ 0.06")

print("="*70)