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
print("DDIN v28 — SUPERVISED CENTROID REWARD (Architecture Ceiling Test)")
print("="*70)

# ─────────────────────────────────────────────────────────────────
# 1.  DATA
# ─────────────────────────────────────────────────────────────────

DATA = r'c:\Users\amiku\Downloads\AI Research New Paradigm\SampleData'
df = pd.read_csv(f'{DATA}/task1_axis_prediction.csv')

print(f"\nDataset: {len(df)} roots, {df.actual_axis.nunique()} axes")
print(f"Axis distribution: {dict(df.actual_axis.value_counts())}")

# Map axis to integer
axis_to_idx = {'EXP': 0, 'TRN': 1, 'MOT': 2, 'SEP': 3, 'CNT': 4}
idx_to_axis = {v: k for k, v in axis_to_idx.items()}

# ─────────────────────────────────────────────────────────────────
# 2.  MODEL
# ─────────────────────────────────────────────────────────────────

class HeterogeneousLiquidSystem(nn.Module):
    def __init__(self, in_dim=23, dim=128):
        super().__init__()
        self.dim = dim
        self.W = nn.Parameter(torch.zeros(dim, dim))
        self.alpha = nn.Parameter(torch.rand(dim) * 0.8 + 0.1)
        self.beta = nn.Parameter(torch.rand(dim) * 0.8 + 0.1)
        self.proj = nn.Parameter(torch.randn(in_dim, dim) * 0.1)

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
# 4.  EVALUATION & REWARD FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def encode_all_roots(model, test_df):
    """Encode all roots to reservoir states"""
    states = []
    for _, r in test_df.iterrows():
        rep = encode_root_sequentially(model, r['root'])
        states.append(rep)
    return np.array(states)

def compute_centroid_reward(model, test_df):
    """
    Supervised centroid reward:
    Reward = -mean(||state - centroid(axis)||²)
    """
    states = encode_all_roots(model, test_df)
    scaler = StandardScaler()
    states = scaler.fit_transform(states)
    
    # Compute axis centroids from ground truth
    axis_labels = [axis_to_idx[r] for r in test_df['actual_axis'].values]
    centroids = {}
    for axis_idx in range(5):
        mask = [l == axis_idx for l in axis_labels]
        if sum(mask) > 0:
            centroids[axis_idx] = np.mean(states[mask], axis=0)
        else:
            centroids[axis_idx] = np.zeros(states.shape[1])
    
    # Reward: negative distance to own centroid
    distances = []
    for i, axis_idx in enumerate(axis_labels):
        dist = np.linalg.norm(states[i] - centroids[axis_idx])
        distances.append(dist)
    
    reward = -np.mean(distances)
    return reward, states

def evaluate_model(model, test_df, k_eval=5):
    """Evaluate with ARI"""
    states = encode_all_roots(model, test_df)
    scaler = StandardScaler()
    states = scaler.fit_transform(states)
    
    kmeans = KMeans(n_clusters=k_eval, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(states)
    
    le = LabelEncoder()
    true_labels = le.fit_transform(test_df['actual_axis'].values)
    
    ari = adjusted_rand_score(true_labels, pred_labels)
    return ari

def track_beta_features(beta):
    """Track which phonological features each neuron amplifies"""
    # β is 128D, one per neuron - map to top features
    # This is a placeholder - the real tracking happens during analysis
    return {"mean": beta.mean().item(), "std": beta.std().item()}

# ─────────────────────────────────────────────────────────────────
# 5.  GRPO WITH SUPERVISED CENTROID REWARD
# ─────────────────────────────────────────────────────────────────

G = 8
ROUNDS = 100
SIGMA = 0.05  # Smaller than before - we have a real gradient now
ETA = 0.01

np.random.seed(42)
torch.manual_seed(42)

model = HeterogeneousLiquidSystem(in_dim=23, dim=128).to(device)
model.eval()

k_clusters = df['actual_axis'].nunique()

print(f"\nStarting GRPO with Supervised Centroid Reward...")
print(f"Dataset: {len(df)} roots, {k_clusters} clusters")
print(f"Parameters: G={G}, sigma={SIGMA}, eta={ETA}, rounds={ROUNDS}")

base_alpha = torch.rand(128).to(device) * 0.8 + 0.1
base_beta = torch.rand(128).to(device) * 0.8 + 0.1

best_ari = 0
best_alpha = base_alpha.clone()
best_beta = base_beta.clone()

for round_i in range(ROUNDS + 1):
    # Evaluate every 10 rounds
    if round_i in [0, 10, 25, 50, 75, 100]:
        model.alpha.data = base_alpha
        model.beta.data = base_beta
        ari = evaluate_model(model, df, k_eval=k_clusters)
        
        # Track β features
        beta_stats = track_beta_features(base_beta)
        
        print(f"Round {round_i:3d} | ARI={ari:.4f} | mean(alpha)={base_alpha.mean():.3f} | mean(beta)={beta_stats['mean']:.3f}", flush=True)
        
        if ari > best_ari:
            best_ari = ari
            best_alpha = base_alpha.clone()
            best_beta = base_beta.clone()

    if round_i == ROUNDS:
        break

    # GRPO step with supervised centroid reward
    epsilons_a = [torch.randn(128).to(device) * SIGMA for _ in range(G)]
    epsilons_b = [torch.randn(128).to(device) * SIGMA for _ in range(G)]
    
    rewards = []
    for eps_a, eps_b in zip(epsilons_a, epsilons_b):
        perturbed_alpha = torch.clamp(base_alpha + eps_a, 0.05, 0.99)
        perturbed_beta = torch.clamp(base_beta + eps_b, 0.05, 0.99)
        model.alpha.data = perturbed_alpha
        model.beta.data = perturbed_beta
        reward, _ = compute_centroid_reward(model, df)
        rewards.append(reward)
        
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
print("EXP 28 FINAL RESULT")
print("="*70)

# Restore best model
model.alpha.data = best_alpha
model.beta.data = best_beta
final_ari = evaluate_model(model, df, k_eval=k_clusters)
final_reward, _ = compute_centroid_reward(model, df)

print(f"Target ARI > 0.15")
print(f"Best ARI: {best_ari:.4f}")
print(f"Final Centroid Reward: {final_reward:+.4f}")

if best_ari > 0.15:
    print("\n>>> MISSION SUCCESS! Architecture ceiling broken!")
    print(">>> The problem was always the reward signal, not the architecture.")
elif best_ari > 0.08:
    print(f"\n>>> PARTIAL: ARI = {best_ari:.4f} (above random baseline)")
else:
    print(f"\n>>> Below target. ARI = {best_ari:.4f}")

print("="*70)