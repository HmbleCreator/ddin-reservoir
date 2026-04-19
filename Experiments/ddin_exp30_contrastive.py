import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score, silhouette_score
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*70)
print("DDIN v30 — CONTRASTIVE GRPO ON TWO-LAYER (Phase 6)")
print("="*70)

# ─────────────────────────────────────────────────────────────────
# 1.  DATA
# ─────────────────────────────────────────────────────────────────

DATA = r'c:\Users\amiku\Downloads\AI Research New Paradigm\SampleData'
df = pd.read_csv(f'{DATA}/task1_axis_prediction.csv')

print(f"\nDataset: {len(df)} roots, {df.actual_axis.nunique()} axes")

# Map axis and locus
axis_to_idx = {'EXP': 0, 'TRN': 1, 'MOT': 2, 'SEP': 3, 'CNT': 4}
locus_list = ['THROAT', 'PALATE', 'CEREBRAL', 'DENTAL', 'LABIAL']

# Create positive and negative pairs
def create_pairs(df, n_pairs=50):
    """Create contrastive pairs"""
    positive_pairs = []  # same axis, different locus
    negative_pairs = []  # same locus, different axis
    
    roots = df['root'].tolist()
    axes = df['actual_axis'].tolist()
    loci = df['locus'].tolist()
    
    for _ in range(n_pairs):
        # Positive: same axis, different locus
        for i in range(len(df)):
            for j in range(len(df)):
                if i != j and axes[i] == axes[j] and loci[i] != loci[j]:
                    if random.random() < 0.1:  # sample some
                        positive_pairs.append((i, j))
                        break
        
        # Negative: same locus, different axis
        for i in range(len(df)):
            for j in range(len(df)):
                if i != j and loci[i] == loci[j] and axes[i] != axes[j]:
                    if random.random() < 0.1:
                        negative_pairs.append((i, j))
                        break
    
    return positive_pairs[:n_pairs], negative_pairs[:n_pairs]

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

positive_pairs, negative_pairs = create_pairs(df, n_pairs=30)
print(f"Positive pairs (same axis, diff locus): {len(positive_pairs)}")
print(f"Negative pairs (same locus, diff axis): {len(negative_pairs)}")

# ─────────────────────────────────────────────────────────────────
# 2.  TWO-LAYER MODEL
# ─────────────────────────────────────────────────────────────────

class TwoLayerDDIN(nn.Module):
    def __init__(self, in_dim=23, dim1=128, dim2=64):
        super().__init__()
        
        self.layer1_dim = dim1
        self.layer1_alpha = nn.Parameter(torch.rand(dim1) * 0.8 + 0.1)
        self.layer1_beta = nn.Parameter(torch.rand(dim1) * 0.8 + 0.1)
        self.layer1_proj = nn.Parameter(torch.randn(in_dim, dim1) * 0.1)
        self.layer1_W = nn.Parameter(torch.zeros(dim1, dim1))
        
        self.layer2_dim = dim2
        self.layer2_alpha = nn.Parameter(torch.rand(dim2) * 0.6 + 0.3)
        self.layer2_beta = nn.Parameter(torch.rand(dim2) * 0.8 + 0.1)
        self.layer2_proj = nn.Parameter(torch.randn(dim1, dim2) * 0.1)
        self.layer2_W = nn.Parameter(torch.randn(dim2, dim2) * 0.02)

    def forward_layer1(self, x, u, dt=0.020):
        drive = u @ self.layer1_proj
        dx = -self.layer1_alpha * x + torch.tanh(x @ self.layer1_W) + self.layer1_beta * drive
        return x + dt * dx

    def forward_layer2(self, h1_output, h2, dt=0.050):
        drive = h1_output @ self.layer2_proj
        dx = -self.layer2_alpha * h2 + torch.tanh(h2 @ self.layer2_W) + self.layer2_beta * drive
        return h2 + dt * dx

def encode_root(model, root_str):
    chars = root_to_chars(root_str)
    if not chars:
        return np.zeros(model.layer2_dim, dtype=np.float32)
    
    n_vowels = sum(1 for c in chars if c in AC)
    vowel_ratio = n_vowels / max(len(chars), 1)
    
    x = torch.zeros(1, model.layer1_dim).to(device)
    with torch.no_grad():
        for c in chars:
            ph_vec = embed_22(c)
            ph_vec_23 = np.concatenate([ph_vec, [vowel_ratio]])
            u = torch.tensor(ph_vec_23, dtype=torch.float32).unsqueeze(0).to(device)
            x = model.forward_layer1(x, u, dt=0.020)
        
        u_zero = torch.zeros(1, 23).to(device)
        for _ in range(2):
            x = model.forward_layer1(x, u_zero, dt=0.050)
        
        layer1_out = x.squeeze()
        
        h2 = torch.zeros(1, model.layer2_dim).to(device)
        for _ in range(3):
            h2 = model.forward_layer2(layer1_out, h2, dt=0.050)
    
    return h2.squeeze().cpu().numpy()

# Phonological mappings (same as before)
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

# ─────────────────────────────────────────────────────────────────
# 3.  EVALUATION
# ─────────────────────────────────────────────────────────────────

def evaluate_model(model, test_df, k_eval=5):
    states = np.array([encode_root(model, r['root']) for _, r in test_df.iterrows()])
    scaler = StandardScaler()
    states = scaler.fit_transform(states)
    kmeans = KMeans(n_clusters=k_eval, random_state=42, n_init=10)
    pred = kmeans.fit_predict(states)
    le = LabelEncoder()
    true = le.fit_transform(test_df['actual_axis'].values)
    return adjusted_rand_score(true, pred)

# ─────────────────────────────────────────────────────────────────
# 4.  CONTRASTIVE GRPO TRAINING
# ─────────────────────────────────────────────────────────────────

G = 8
ROUNDS = 30
SIGMA = 0.10
ETA = 0.02
TEMPERATURE = 0.5

print(f"\nStarting Contrastive GRPO...")
print(f"Parameters: G={G}, sigma={SIGMA}, eta={ETA}, rounds={ROUNDS}, T={TEMPERATURE}")

model = TwoLayerDDIN(in_dim=23, dim1=128, dim2=64).to(device)
model.eval()

le = LabelEncoder()
le.fit(df['actual_axis'].values)
k_clusters = df['actual_axis'].nunique()

roots_list = df['root'].tolist()
states_cache = {}

def get_state(model, idx):
    if idx not in states_cache:
        states_cache[idx] = encode_root(model, roots_list[idx])
    return states_cache[idx]

def compute_contrastive_reward(model):
    """Contrastive reward: push positive pairs close, negative pairs far"""
    # Encode all needed roots first
    states_cache.clear()
    for idx in range(len(roots_list)):
        get_state(model, idx)
    
    # Compute rewards
    pos_rewards = []
    neg_rewards = []
    
    for i, j in positive_pairs[:20]:
        s1 = get_state(model, i)
        s2 = get_state(model, j)
        dist = np.linalg.norm(s1 - s2) ** 2
        pos_rewards.append(np.exp(-dist / TEMPERATURE))
    
    for i, j in negative_pairs[:20]:
        s1 = get_state(model, i)
        s2 = get_state(model, j)
        dist = np.linalg.norm(s1 - s2) ** 2
        neg_rewards.append(np.exp(-dist / TEMPERATURE))
    
    # Contrastive: maximize positive, minimize negative
    reward = np.mean(pos_rewards) - np.mean(neg_rewards)
    return reward

# Initialize parameters
base_layer2_alpha = torch.rand(64) * 0.6 + 0.3  # slower dynamics
base_layer2_beta = torch.rand(64) * 0.8 + 0.1
with torch.no_grad():
    model.layer2_alpha.data = base_layer2_alpha.clone()
    model.layer2_beta.data = base_layer2_beta.clone()

best_ari = 0
best_alpha = base_layer2_alpha.clone()
best_beta = base_layer2_beta.clone()

for round_i in range(ROUNDS + 1):
    if round_i in [0, 5, 10, 20, 30]:
        model.layer2_alpha.data = base_layer2_alpha
        model.layer2_beta.data = base_layer2_beta
        ari = evaluate_model(model, df, k_eval=k_clusters)
        contrastive = compute_contrastive_reward(model)
        
        if ari > best_ari:
            best_ari = ari
            best_alpha = base_layer2_alpha.clone()
            best_beta = base_layer2_beta.clone()
        
        print(f"Round {round_i:3d} | ARI={ari:.4f} | Contrastive={contrastive:+.4f}", flush=True)

    if round_i == ROUNDS:
        break
    
    # GRPO on Layer 2 parameters
    epsilons_a = [torch.randn(64) * SIGMA for _ in range(G)]
    epsilons_b = [torch.randn(64) * SIGMA for _ in range(G)]
    
    rewards = []
    for eps_a, eps_b in zip(epsilons_a, epsilons_b):
        perturbed_alpha = torch.clamp(base_layer2_alpha + eps_a, 0.1, 0.99)
        perturbed_beta = torch.clamp(base_layer2_beta + eps_b, 0.05, 0.99)
        
        model.layer2_alpha.data = perturbed_alpha
        model.layer2_beta.data = perturbed_beta
        
        # Clear cache for new parameters
        states_cache.clear()
        
        r = compute_contrastive_reward(model)
        rewards.append(r)
    
    mu_r = np.mean(rewards)
    std_r = np.std(rewards) + 1e-8
    advantages = [(r - mu_r) / std_r for r in rewards]
    
    delta_alpha = torch.zeros(64)
    delta_beta = torch.zeros(64)
    for a, eps_a, eps_b in zip(advantages, epsilons_a, epsilons_b):
        delta_alpha += a * eps_a
        delta_beta += a * eps_b
    
    base_layer2_alpha = torch.clamp(base_layer2_alpha + ETA * delta_alpha, 0.1, 0.99)
    base_layer2_beta = torch.clamp(base_layer2_beta + ETA * delta_beta, 0.05, 0.99)

# Restore best
model.layer2_alpha.data = best_alpha
model.layer2_beta.data = best_beta
final_ari = evaluate_model(model, df, k_eval=k_clusters)

print("\n" + "="*70)
print("EXP 30 RESULT: CONTRASTIVE GRPO")
print("="*70)
print(f"Target: ARI > 0.15")
print(f"Best ARI: {best_ari:.4f}")
print(f"Final ARI: {final_ari:.4f}")

if best_ari > 0.15:
    print("\n>>> SUCCESS! Phase 5 milestone achieved!")
elif best_ari > 0.08:
    print(f"\n>>> PARTIAL: Improved to {best_ari:.4f}")
else:
    print(f"\n>>> BELOW TARGET: ARI = {best_ari:.4f}")

print("="*70)