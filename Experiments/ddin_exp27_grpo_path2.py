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
print("DDIN v27 — GRPO ON AXIS-PRIORITIZED ROOT SET (Path 2 + Optimization)")
print("="*70)

# ─────────────────────────────────────────────────────────────────
# 1.  DATA SETUP (PATH 2: AXIS-PRIORITIZED)
# ─────────────────────────────────────────────────────────────────

DATA = r'c:\Users\amiku\Downloads\AI Research New Paradigm\SampleData'
df_full = pd.read_csv(f'{DATA}/task1_axis_prediction.csv')

# Analyze axis-locus purity
axis_locus_purity = {}
for axis in df_full.actual_axis.unique():
    axis_df = df_full[df_full.actual_axis == axis]
    locus_counts = axis_df.locus.value_counts()
    dominant_locus = locus_counts.index[0]
    purity = locus_counts.iloc[0] / len(axis_df)
    axis_locus_purity[axis] = (dominant_locus, purity)

print("Axis -> Dominant Locus -> Purity:")
for axis, (locus, purity) in axis_locus_purity.items():
    print(f"  {axis} -> {locus} ({purity:.1%})")

# Select high-purity roots
high_purity_roots = []
for axis in ['EXP', 'MOT', 'SEP', 'CNT', 'TRN']:
    axis_df = df_full[df_full.actual_axis == axis]
    dominant_locus = axis_locus_purity[axis][0]
    high_purity = axis_df[axis_df.locus == dominant_locus]
    high_purity_roots.extend(high_purity['root'].tolist())

df = df_full[df_full['root'].isin(high_purity_roots[:100])].copy()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nPath 2 dataset: {len(df)} roots (high-purity)")
print(f"  Locus dist: {dict(df.locus.value_counts())}")
print(f"  Axis dist: {dict(df.actual_axis.value_counts())}")

# ─────────────────────────────────────────────────────────────────
# 2.  MODEL & MAPPINGS
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

# Phonological mappings (same as before)
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
AC   = set(['a','A','i','I','u','U','R','L','e','o'])
PRATYAHARA_SETS = [AC, set(PHONEME_VECTORS_16.keys()) - AC, set(['y','r','l','v']),
                set(['K','G','C','J','Q','X','H','W','P','B','z','x','s','h']),
                set(['z','x','s','h']), set(['g','j','D','d','b']), set(['N','n','m']), set(['a','i','u','R','L'])]

TRANSLIT = {'A':'A','I':'I','U':'U','R':'R','kh':'K','gh':'G','ch':'C','jh':'J','Th':'Q','Dh':'X',
           'th':'H','dh':'W','ph':'P','bh':'B','sh':'z','sh2':'x','ng':'N','S':'x', 'M':'m'}

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
    ac16 = PHONEME_VECTORS_16.get(ph_char, [0.5]*16)
    manner = [ac16[d] for d in MANNER_DIMS]
    form = [x * FORMANT_WEIGHT for x in FORMANT_F1F2.get(ph_char, [0.4, 0.4])]
    prat = [1.0 if ph_char in S else 0.0 for S in PRATYAHARA_SETS]
    return np.array(manner + form + prat, dtype=np.float32)

# Sequential encoding
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
# 3.  EVALUATION
# ─────────────────────────────────────────────────────────────────

def evaluate_model(model, test_df, k_eval=5):
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
    
    return sil, ari

# ─────────────────────────────────────────────────────────────────
# 4.  GRPO OPTIMIZATION
# ─────────────────────────────────────────────────────────────────

G = 8
ROUNDS = 50
SIGMA = 0.50
ETA = 0.05

np.random.seed(42)
torch.manual_seed(42)

model = HeterogeneousLiquidSystem(in_dim=23, dim=128).to(device)
model.eval()

le = LabelEncoder()
le.fit(df['actual_axis'].values)
k_clusters = df['actual_axis'].nunique()

print(f"\nStarting GRPO on Axis-Prioritized dataset ({len(df)} roots, {k_clusters} clusters)...")

base_alpha = torch.rand(128).to(device) * 0.8 + 0.1
base_beta = torch.rand(128).to(device) * 0.8 + 0.1

for round_i in range(ROUNDS + 1):
    if round_i in [0, 10, 25, 40, 50]:
        model.alpha.data = base_alpha
        model.beta.data = base_beta
        sil, ari = evaluate_model(model, df, k_eval=k_clusters)
        print(f"Round {round_i:3d} | Silhouette={sil:+.4f} | ARI={ari:.4f} | mean(a)={base_alpha.mean():.3f}", flush=True)

    if round_i == ROUNDS:
        break

    epsilons_a = [torch.randn(128).to(device) * SIGMA for _ in range(G)]
    epsilons_b = [torch.randn(128).to(device) * SIGMA for _ in range(G)]
    
    rewards = []
    for eps_a, eps_b in zip(epsilons_a, epsilons_b):
        perturbed_alpha = torch.clamp(base_alpha + eps_a, 0.05, 0.99)
        perturbed_beta = torch.clamp(base_beta + eps_b, 0.05, 0.99)
        model.alpha.data = perturbed_alpha
        model.beta.data = perturbed_beta
        sil_r, _ = evaluate_model(model, df, k_eval=k_clusters)
        rewards.append(sil_r)
        
    mu_r = np.mean(rewards)
    std_r = np.std(rewards) + 1e-8
    advantages = [(r - mu_r) / std_r for r in rewards]
    
    delta_alpha = torch.zeros(128).to(device)
    delta_beta = torch.zeros(128).to(device)
    for a, eps_a, eps_b in zip(advantages, epsilons_a, epsilons_b):
        delta_alpha += a * eps_a
        delta_beta += a * eps_b
        
    base_alpha = torch.clamp(base_alpha + ETA * delta_alpha, 0.05, 0.99)
    base_beta = torch.clamp(base_beta + ETA * delta_beta, 0.05, 0.99)

print("\n" + "="*70)
print("EXP 27 FINAL RESULT")
print("="*70)
model.alpha.data = base_alpha
model.beta.data = base_beta
final_sil, final_ari = evaluate_model(model, df, k_eval=k_clusters)

print(f"Target ARI > 0.15")
print(f"Final Silhouette: {final_sil:+.4f}")
print(f"Final ARI: {final_ari:.4f}")

if final_ari > 0.15:
    print("\n>>> MISSION SUCCESS! Path 2 + GRPO broke the ceiling!")
else:
    print(f"\n>>> ARI = {final_ari:.4f} (Path 2 baseline was 0.0564)")
    print(">>> Additional optimization paths needed.")