import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*65)
print("DDIN Exp 32 -- THE 'ALL-IN' TRAINABLE TOPOGRAPHY")
print("Phase 7A: Global Dense W2 Optimization via Contrastive GRPO")
print("="*65)

# ─────────────────────────────────────────────────────────────────
# 1. HETEROGENEOUS LIQUID SYSTEMS 
# ─────────────────────────────────────────────────────────────────

class HeterogeneousLiquidSystem(nn.Module):
    def __init__(self, in_dim, dim, alpha_min, alpha_max, w_matrix=None, freeze_w=False):
        super().__init__()
        self.dim = dim
        if w_matrix is not None:
             self.W = nn.Parameter(w_matrix, requires_grad=(not freeze_w))
        else:
             self.W = nn.Parameter(torch.zeros(dim, dim), requires_grad=(not freeze_w))
             
        self.alpha = nn.Parameter(torch.rand(dim) * (alpha_max - alpha_min) + alpha_min)
        self.beta  = nn.Parameter(torch.rand(dim) * 0.8 + 0.1)
        self.proj  = nn.Parameter(torch.randn(in_dim, dim) * 0.1)

    def forward(self, x, u, dt=0.020):
        drive = u @ self.proj
        dx = -self.alpha * x + torch.tanh(x @ self.W) + self.beta * drive
        return x + dt * dx

def build_structured_W(dim=64, num_clusters=5):
    """
    Builds the topographical seed map.
    This time, it acts as a structured PRIOR, not an immutable cage.
    """
    W = torch.randn(dim, dim) * 0.01 - 0.05
    for i in range(num_clusters):
        start = int(i * (dim/num_clusters))
        end = int((i+1) * (dim/num_clusters)) if i < num_clusters - 1 else dim
        block = torch.randn(end-start, end-start) * 0.01 + 0.15
        W[start:end, start:end] = block
    return W

class TwoLayerDDIN(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: Zero-weight encoding pipeline (Immutable)
        self.layer1 = HeterogeneousLiquidSystem(in_dim=23, dim=128, alpha_min=0.1, alpha_max=0.9, w_matrix=torch.zeros(128,128), freeze_w=True)
        
        # Layer 2: The Mutable Block-Diagonal topological map
        W2_struct = build_structured_W(dim=64, num_clusters=5)
        self.layer2 = HeterogeneousLiquidSystem(in_dim=128, dim=64, alpha_min=0.3, alpha_max=0.95, w_matrix=W2_struct, freeze_w=False)

# ─────────────────────────────────────────────────────────────────
# 2. PHONOLOGICAL MAPPINGS & ENCODING
# ─────────────────────────────────────────────────────────────────
def f1f2_norm(f1_hz, f2_hz): return [round(np.clip(f1_hz/1200.0, 0, 1), 4), round(np.clip((f2_hz-200)/2600.0, 0, 1), 4)]

FORMANT_F1F2 = {'a':f1f2_norm(800,1300), 'A':f1f2_norm(800,1300), 'i':f1f2_norm(280,2300), 'I':f1f2_norm(280,2300), 'u':f1f2_norm(280, 700), 'U':f1f2_norm(280, 700), 'R':f1f2_norm(490,1380), 'L':f1f2_norm(490,1800), 'e':f1f2_norm(400,2000), 'o':f1f2_norm(490,800), 'k':f1f2_norm(150,1400),'K':f1f2_norm(150,1400),'g':f1f2_norm(150,1400), 'G':f1f2_norm(150,1400),'N':f1f2_norm(200,1400),'h':f1f2_norm(600,1200), 'c':f1f2_norm(150,2100),'C':f1f2_norm(150,2100),'j':f1f2_norm(150,2100), 'J':f1f2_norm(150,2100),'y':f1f2_norm(400,2100),'z':f1f2_norm(250,2100), 'T':f1f2_norm(150,1100),'Q':f1f2_norm(150,1100),'D':f1f2_norm(150,1100), 'X':f1f2_norm(150,1100),'r':f1f2_norm(400,1100),'x':f1f2_norm(250,1100), 't':f1f2_norm(150,1800),'H':f1f2_norm(150,1800),'d':f1f2_norm(150,1800), 'W':f1f2_norm(150,1800),'n':f1f2_norm(200,1800),'l':f1f2_norm(400,1800), 's':f1f2_norm(250,1800), 'p':f1f2_norm(150, 800),'P':f1f2_norm(150, 800),'b':f1f2_norm(150, 800), 'B':f1f2_norm(150, 800),'m':f1f2_norm(200, 800),'v':f1f2_norm(400, 800)}
PHONEME_VECTORS_16 = {'a':[0.50,0.50,0.0,1.0,0.0,0.0,0.50,0.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,0.7], 'A':[0.50,0.50,0.0,1.0,0.0,0.0,0.50,1.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,0.7], 'i':[0.25,0.80,0.0,1.0,0.0,1.0,0.0, 0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.3,0.7], 'I':[0.25,0.80,0.0,1.0,0.0,1.0,0.0, 1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.3,0.7], 'u':[0.75,0.80,0.0,1.0,0.0,1.0,1.0, 0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.2,0.7], 'U':[0.75,0.80,0.0,1.0,0.0,1.0,1.0, 1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.2,0.7], 'R':[0.50,0.60,0.0,1.0,0.0,0.5,0.0, 0.0,1.0,0.9,1.0,1.0,0.0,0.0,0.5,0.7], 'L':[0.75,0.70,0.0,1.0,0.0,0.5,0.0, 0.0,1.0,0.9,1.0,1.0,0.0,0.0,0.5,0.7], 'e':[0.25,0.70,0.0,1.0,0.0,0.7,0.0, 0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.7,0.7], 'o':[0.75,0.70,0.0,1.0,0.0,0.7,1.0, 0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.7,0.7], 'k':[0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 'K':[0.0, 0.0, 1.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.35,0.0,0.0,0.0,0.0], 'g':[0.0, 0.0, 0.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.2, 0.0,0.0,0.0,0.0], 'G':[0.0, 0.0, 1.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.35,0.0,0.0,0.0,0.0], 'N':[0.0, 0.0, 0.0,1.0,1.0,0.0,0.0, 0.0,0.0,0.5,0.5,0.5, 0.0,0.0,0.0,1.0], 'h':[0.0, 0.0, 0.0,1.0,0.0,0.0,0.0, 0.0,1.0,0.3,1.0,0.65,0.0,0.0,0.0,0.0], 'c':[0.25,0.25,0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0], 'C':[0.25,0.25,1.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.35,0.0,1.0,0.0,0.0], 'j':[0.25,0.25,0.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.2, 0.0,1.0,0.0,0.0], 'J':[0.25,0.25,1.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.35,0.0,1.0,0.0,0.0], 'y':[0.25,0.25,0.0,1.0,0.0,0.8,0.0, 0.0,1.0,0.8,1.0,0.85,0.0,1.0,0.0,0.4], 'z':[0.25,0.25,0.0,0.0,0.0,0.0,0.0, 0.0,1.0,0.2,1.0,0.55,1.0,1.0,0.0,0.0], 'T':[0.5, 0.5, 0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0], 'Q':[0.5, 0.5, 1.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.35,0.0,1.0,0.0,0.0], 'D':[0.5, 0.5, 0.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.2, 0.0,1.0,0.0,0.0], 'X':[0.5, 0.5, 1.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.35,0.0,1.0,0.0,0.0], 'r':[0.5, 0.5, 0.0,1.0,0.0,0.3,0.0, 0.0,1.0,0.8,1.0,0.75,0.0,1.0,0.0,0.4], 'x':[0.5, 0.5, 0.0,0.0,0.0,0.0,0.0, 0.0,1.0,0.2,1.0,0.55,1.0,1.0,0.0,0.0], 't':[0.75,0.75,0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0], 'H':[0.75,0.75,1.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.35,0.0,1.0,0.0,0.0], 'd':[0.75,0.75,0.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.2, 0.0,1.0,0.0,0.0], 'W':[0.75,0.75,1.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.35,0.0,1.0,0.0,0.0], 'n':[0.75,0.75,0.0,1.0,1.0,0.0,0.0, 0.0,0.0,0.5,0.5,0.5, 0.0,1.0,0.0,1.0], 'l':[0.75,0.7, 0.0,1.0,0.0,0.3,0.0, 0.0,1.0,0.8,1.0,0.75,0.0,1.0,0.0,0.4], 's':[0.75,0.75,0.0,0.0,0.0,0.0,0.0, 0.0,1.0,0.2,1.0,0.55,1.0,1.0,0.0,0.0], 'p':[1.0, 1.0, 0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0], 'P':[1.0, 1.0, 1.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.35,0.0,0.0,0.0,0.0], 'b':[1.0, 1.0, 0.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.2, 0.0,0.0,0.0,0.0], 'B':[1.0, 1.0, 1.0,1.0,0.0,0.0,0.0, 0.0,0.0,0.1,0.0,0.35,0.0,0.0,0.0,0.0], 'm':[1.0, 1.0, 0.0,1.0,1.0,0.0,0.0, 0.0,0.0,0.5,0.5,0.5, 0.0,0.0,0.0,1.0], 'v':[1.0, 1.0, 0.0,1.0,0.0,0.2,1.0, 0.0,1.0,0.7,1.0,0.85,0.0,0.0,0.0,0.4]}

MANNER_DIMS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15]
AC, HAL, YAN, JHAL, SAL, JASH, NAM, AK  = set(['a','A','i','I','u','U','R','L','e','o']), set(PHONEME_VECTORS_16.keys()) - set(['a','A','i','I','u','U','R','L','e','o']), set(['y','r','l','v']), set(['K','G','C','J','Q','X','H','W','P','B','z','x','s','h']), set(['z','x','s','h']), set(['g','j','D','d','b']), set(['N','n','m']), set(['a','i','u','R','L'])
PRATYAHARA_SETS = [AC, HAL, YAN, JHAL, SAL, JASH, NAM, AK]

TRANSLIT = {'A':'A','I':'I','U':'U','R':'R', 'kh':'K','gh':'G','ch':'C','jh':'J','Th':'Q','Dh':'X', 'th':'H','dh':'W','ph':'P','bh':'B','sh':'z','sh2':'x','ng':'N', 'S':'x', 'M':'m'}
def root_to_chars(s):
    chars, i = [], 0
    while i < len(s):
        if i+1 < len(s) and s[i:i+2] in TRANSLIT: chars.append(TRANSLIT[s[i:i+2]]); i += 2
        else: chars.append(TRANSLIT.get(s[i], s[i])); i += 1
    return chars

def embed_22(ph_char):
    ac16   = PHONEME_VECTORS_16.get(ph_char, [0.5]*16)
    manner = [ac16[d] for d in MANNER_DIMS]
    form   = [x * 4.0 for x in FORMANT_F1F2.get(ph_char, [0.4, 0.4])]
    prat   = [1.0 if ph_char in S else 0.0 for S in PRATYAHARA_SETS]
    return np.array(manner + form + prat, dtype=np.float32)

def encode_root_hierarchical(model, root_str):
    chars = root_to_chars(root_str)
    if not chars: return np.zeros(64, dtype=np.float32)
    n_vowels = sum(1 for c in chars if c in AC)
    v_ratio = n_vowels / max(len(chars), 1)

    x1 = torch.zeros(1, 128).to(device)
    with torch.no_grad():
        for c in chars:
            pv = np.concatenate([embed_22(c), [v_ratio]])
            u = torch.tensor(pv, dtype=torch.float32).unsqueeze(0).to(device)
            x1 = model.layer1(x1, u, dt=0.020)
            
        u_zero = torch.zeros(1, 23).to(device)
        for _ in range(2): x1 = model.layer1(x1, u_zero, dt=0.050)
            
        x2 = torch.zeros(1, 64).to(device)
        layer1_steady = x1.clone()
        
        for _ in range(5):
            x2 = model.layer2(x2, layer1_steady, dt=0.050)
    return x2.cpu().squeeze().numpy()

# ─────────────────────────────────────────────────────────────────
# 3. BENCHMARK & REWARD 
# ─────────────────────────────────────────────────────────────────

DATA = r'c:\Users\amiku\Downloads\AI Research New Paradigm\SampleData'
t1 = pd.read_csv(f'{DATA}/task1_axis_prediction.csv')

axes = t1['actual_axis'].values
loci = t1['locus'].values if 'locus' in t1.columns else np.random.choice(['THROAT','PALATE','CEREBRAL','DENTAL','LABIAL'], len(t1))

pos_pairs, neg_pairs = [], []
for i in range(len(t1)):
    for j in range(i+1, len(t1)):
        if axes[i] == axes[j]: pos_pairs.append((i, j))
        elif loci[i] == loci[j]: neg_pairs.append((i, j))

def evaluate_metrics_on_model(test_model, k_eval=5, n_init=1):
    reservoir_states = []
    for _, r in t1.iterrows():
        reservoir_states.append(encode_root_hierarchical(test_model, r['root']))
    reservoir_states = np.array(reservoir_states)
    
    pos_dists = [np.linalg.norm(reservoir_states[i] - reservoir_states[j])**2 for (i,j) in pos_pairs[:500]]
    neg_dists = [np.linalg.norm(reservoir_states[i] - reservoir_states[j])**2 for (i,j) in neg_pairs[:500]]
    
    mean_pos = np.mean(pos_dists) if pos_dists else 1.0
    mean_neg = np.mean(neg_dists) if neg_dists else 1.0
    
    contrastive_reward = mean_neg - (1.5 * mean_pos)

    scaler = StandardScaler()
    norm_states = scaler.fit_transform(reservoir_states)    
    pred = KMeans(n_clusters=k_eval, random_state=42, n_init=n_init).fit_predict(norm_states)
    le = LabelEncoder()
    axis_ids = le.fit_transform(t1['actual_axis'].values)
    ari_score = adjusted_rand_score(axis_ids, pred)
    return contrastive_reward, ari_score

# ─────────────────────────────────────────────────────────────────
# 4. GRPO TRAINING (ALL-IN: W, Alpha, Beta)
# ─────────────────────────────────────────────────────────────────

torch.manual_seed(42)
np.random.seed(42)

model = TwoLayerDDIN().to(device)
model.eval()

G = 8          # Num perturbations per round
ROUNDS = 40    # More rounds for the massive Dense W parameter space
SIGMA = 0.2    # Noise standard deviation
ETA = 0.05     # Learning Rate
ETA_W = 0.01   # Smaller step size for Dense W

print("Initializing fully unconstrained GRPO over W2, Alpha, and Beta...", flush=True)

base_alpha = model.layer2.alpha.data.clone()
base_beta = model.layer2.beta.data.clone()
base_w = model.layer2.W.data.clone()

best_reward = -float('inf')
best_ari = 0.0

for round_i in range(ROUNDS + 1):
    if round_i % 4 == 0:
        model.layer2.alpha.data = base_alpha
        model.layer2.beta.data = base_beta
        model.layer2.W.data = base_w
        reward, ari_score = evaluate_metrics_on_model(model, k_eval=5, n_init=5)
        
        if ari_score > best_ari:
            best_ari = ari_score
            
        print(f"Round {round_i:3d} | Reward = {reward:+.3f} | Eval(ARI) = {ari_score:.4f} | W_norm = {torch.norm(base_w):.2f}", flush=True)

    if round_i == ROUNDS: break

    # Massive parallel perturbations
    alpha_eps = [torch.randn(64).to(device) * SIGMA for _ in range(G)]
    beta_eps = [torch.randn(64).to(device) * SIGMA for _ in range(G)]
    w_eps = [torch.randn(64, 64).to(device) * SIGMA for _ in range(G)]
    
    rewards = []
    for eps_a, eps_b, eps_w in zip(alpha_eps, beta_eps, w_eps):
        model.layer2.alpha.data = torch.clamp(base_alpha + eps_a, 0.1, 0.99)
        model.layer2.beta.data = torch.clamp(base_beta + eps_b, 0.05, 0.99)
        
        # We perturb the massive geometry space (and clamp it slightly to prevent explosion)
        model.layer2.W.data = torch.clamp(base_w + eps_w, -2.0, 2.0)
        
        r, _ = evaluate_metrics_on_model(model, k_eval=5, n_init=1)
        rewards.append(r)
        
    mu_r = np.mean(rewards)
    std_r = np.std(rewards) + 1e-8
    advantages = [(r - mu_r) / std_r for r in rewards]
    
    delta_alpha = torch.zeros(64).to(device)
    delta_beta = torch.zeros(64).to(device)
    delta_w = torch.zeros(64, 64).to(device)
    
    for a, e_a, e_b, e_w in zip(advantages, alpha_eps, beta_eps, w_eps):
        delta_alpha += a * e_a
        delta_beta  += a * e_b
        delta_w     += a * e_w
        
    base_alpha = torch.clamp(base_alpha + ETA * delta_alpha, 0.1, 0.99)
    base_beta  = torch.clamp(base_beta + ETA * delta_beta, 0.05, 0.99)
    # Bound the magnitude of W to prevent limit cycle explosion
    base_w     = torch.clamp(base_w + ETA_W * delta_w, -2.0, 2.0)

print("\n=================================================================", flush=True)
print("EXP 32 DENSE OPTIMIZATION FINAL RESULT", flush=True)
model.layer2.alpha.data = base_alpha
model.layer2.beta.data = base_beta
model.layer2.W.data = base_w
final_r, final_ari = evaluate_metrics_on_model(model, k_eval=5, n_init=15)
print(f"  Best ARI Observed : {best_ari:.4f}", flush=True)
print(f"  Final Round ARI   : {final_ari:.4f}", flush=True)
if best_ari > 0.0690:
    print("  --> MISSION SUCCESS! The dense structural tuning successfully broke the 0.069 Single-Layer Bound.", flush=True)
else:
    print("  --> The Dense W space was too chaotic for GRPO to find a better configuration than 0.0690.", flush=True)
print("=================================================================", flush=True)
