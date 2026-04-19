import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*65)
print("DDIN v24 -- GRPO WITH SUPERVISED CENTROID REWARD")
print("Track C (Candidate 2): Bypassing self-supervised limits to prove the architectural ceiling")
print("="*65)

# ─────────────────────────────────────────────────────────────────
# 1.  HETEROGENEOUS LIQUID SYSTEM (The Receiver ODE)
# ─────────────────────────────────────────────────────────────────

class HeterogeneousLiquidSystem(nn.Module):
    """128-neuron Liquid Neural Network ODE with W=0"""
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
# 2.  PHONOLOGICAL MAPPINGS
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
}

MANNER_DIMS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15]
AC  = set(['a','A','i','I','u','U','R','L','e','o'])
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
# 3.  SEQUENTIAL ENCODING LOOP
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
# 4.  LOAD DATA & EVALUATION FRAMEWORK
# ─────────────────────────────────────────────────────────────────

DATA = r'c:\Users\amiku\Downloads\AI Research New Paradigm\SampleData'
t1 = pd.read_csv(f'{DATA}/task1_axis_prediction.csv')

torch.manual_seed(42)
np.random.seed(42)

model = HeterogeneousLiquidSystem(in_dim=23, dim=128).to(device)
model.eval()

le = LabelEncoder()
axis_ids = le.fit_transform(t1['actual_axis'].values)

def evaluate_metrics_on_model(test_model, k_eval=5, n_init=1):
    """
    Returns TWO metrics:
    1. Supervised Negative Centroid MSE (The Continuous GRPO Training Reward)
    2. Semantic Axis ARI (The Discrete External Evaluation Metric)
    """
    reservoir_states = []
    for _, r in t1.iterrows():
        rep = encode_root_sequentially(test_model, r['root'])
        reservoir_states.append(rep)
        
    reservoir_states = np.array(reservoir_states)
    
    # Scale to prevent trivial collapse to zero
    scaler = StandardScaler()
    norm_states = scaler.fit_transform(reservoir_states)
    
    # Calculate Candidate 2 Reward: Negative Centroid MSE
    mse_sum = 0
    unique_labels = np.unique(axis_ids)
    for label in unique_labels:
        idx = (axis_ids == label)
        class_states = norm_states[idx]
        centroid = np.mean(class_states, axis=0)
        mse_sum += np.sum((class_states - centroid)**2)
    
    # Reward is negative MSE (minimizing intra-cluster variance, maximizing inter-cluster variance via norm)
    neg_mse_reward = - (mse_sum / len(norm_states))
    
    # Evaluation ARI (discrete)
    pred = KMeans(n_clusters=k_eval, random_state=42, n_init=n_init).fit_predict(norm_states)
    ari_score = adjusted_rand_score(axis_ids, pred)
    
    return neg_mse_reward, ari_score

# ─────────────────────────────────────────────────────────────────
# 5.  GRPO TRAINING LOOP (SUPERVISED CENTROID REWARD)
# ─────────────────────────────────────────────────────────────────

G = 8          # Num perturbations per round
ROUNDS = 50    # Total optimization rounds
SIGMA = 0.50   # Variance of the Gaussian perturbation
ETA = 0.05     # Learning rate

print("\nInitializing GRPO Alpha Optimization (Supervised Continuous Reward)...", flush=True)
base_alpha = torch.rand(128).to(device) * 0.8 + 0.1  # U(0.1, 0.9)

for round_i in range(ROUNDS + 1):
    if round_i in [0, 10, 25, 40, 50]:
        model.alpha.data = base_alpha
        neg_mse, ari_score = evaluate_metrics_on_model(model, k_eval=5, n_init=5)
        print(f"Round {round_i:3d} | Reward(Neg_MSE) = {neg_mse:+.4f} | Eval(ARI) = {ari_score:.4f} | mean(alpha) = {base_alpha.mean():.3f}", flush=True)

    if round_i == ROUNDS:
        break

    epsilons = [torch.randn(128).to(device) * SIGMA for _ in range(G)]
    
    rewards = []
    for eps in epsilons:
        perturbed_alpha = torch.clamp(base_alpha + eps, 0.05, 0.99)
        model.alpha.data = perturbed_alpha
        # n_init=1 for speed in inner loop (doesn't affect reward, only ARI eval if used, but we only use reward)
        neg_mse_r, _ = evaluate_metrics_on_model(model, k_eval=5, n_init=1)
        rewards.append(neg_mse_r)  # Candidate 2 supervised distance
        
    mu_r = np.mean(rewards)
    std_r = np.std(rewards) + 1e-8
    advantages = [(r - mu_r) / std_r for r in rewards]
    
    delta_alpha = torch.zeros(128).to(device)
    for a, eps in zip(advantages, epsilons):
        delta_alpha += a * eps
        
    base_alpha = base_alpha + ETA * delta_alpha
    base_alpha = torch.clamp(base_alpha, 0.05, 0.99)

print("\n=================================================================", flush=True)
print("EXP 24 FINAL RESULT (CEILING TEST)", flush=True)
model.alpha.data = base_alpha
final_mse, final_ari = evaluate_metrics_on_model(model, k_eval=5, n_init=15)
print(f"  Target ARI > 0.15", flush=True)
print(f"  Final Negative MSE : {final_mse:+.4f}", flush=True)
print(f"  Final ARI          : {final_ari:.4f}", flush=True)

if final_ari > 0.15:
    print("  --> MISSION SUCCESS! Architecture ceiling theoretically validated! The temporal sequence physics successfully extract the latent semantic structure.", flush=True)
else:
    print("  --> FAILED. Even with perfect supervised ground truth gradients, Alpha optimization couldn't break the ceiling. Beta tuning required.", flush=True)
print("=================================================================", flush=True)
