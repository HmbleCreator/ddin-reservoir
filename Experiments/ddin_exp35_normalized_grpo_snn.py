import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score
import time
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*65)
print("DDIN Exp 35 -- NORMALIZED GRPO-SNN")
print("Phase 8B: Resolving Magnitude Bias via Geometric Normalization")
print("="*65)

# ─────────────────────────────────────────────────────────────────
# 1. ADEX DIFFERENTIAL ENGINE
# ─────────────────────────────────────────────────────────────────

class AdExPopulation(nn.Module):
    def __init__(self, size, dt=1.0, 
                 C=200.0, gL=10.0, EL=-70.0, VT=-50.0, DeltaT=2.0, 
                 Vpeak=0.0, Vreset=-58.0, 
                 a=2.0, b=80.0, tau_w=30.0):
        super().__init__()
        self.size = size
        self.dt = dt
        self.C, self.gL, self.EL, self.DeltaT, self.Vpeak, self.Vreset = C, gL, EL, DeltaT, Vpeak, Vreset
        
        # Optimization Targets (Per-neuron dynamics)
        self.VT = nn.Parameter(torch.ones(1, size).to(device) * VT)
        self.a = nn.Parameter(torch.ones(1, size).to(device) * a)
        self.b = nn.Parameter(torch.ones(1, size).to(device) * b)
        self.tau_w = nn.Parameter(torch.ones(1, size).to(device) * tau_w)
        
        self.V = torch.ones(1, size).to(device) * EL
        self.w = torch.zeros(1, size).to(device)
        self.spike_counts = torch.zeros(1, size).to(device)

    def reset_states(self):
        self.V = torch.ones(1, self.size).to(device) * self.EL
        self.w = torch.zeros(1, self.size).to(device)
        self.spike_counts = torch.zeros(1, self.size).to(device)

    def step(self, I_ext):
        # AdEx Update
        exp_term = self.gL * self.DeltaT * torch.exp((self.V - self.VT) / self.DeltaT)
        dV = (-self.gL * (self.V - self.EL) + exp_term - self.w + I_ext) / self.C
        self.V += self.dt * dV
        
        dw = (self.a * (self.V - self.EL) - self.w) / self.tau_w
        self.w += self.dt * dw
        
        spikes = self.V >= self.Vpeak
        self.spike_counts += spikes.float()
        
        self.V = torch.where(spikes, torch.tensor(self.Vreset).to(device), self.V)
        self.w = torch.where(spikes, self.w + self.b, self.w)
        return spikes

class TwoLayerSNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: Fast Driver (128 neurons, Fixed)
        self.L1 = AdExPopulation(size=128, a=0.0, b=0.0, tau_w=5.0, gL=15.0)
        self.proj_in = nn.Parameter(torch.randn(23, 128) * 900.0) 
        
        # Layer 2: Slow Organizer (64 neurons, Target)
        self.L2 = AdExPopulation(size=64, a=2.0, b=80.0, tau_w=150.0, gL=2.0)
        self.W12 = nn.Parameter(torch.randn(128, 64) * 350.0)

    def reset(self):
        self.L1.reset_states()
        self.L2.reset_states()

    def step(self, I_input):
        I1 = I_input @ self.proj_in
        sp1 = self.L1.step(I1)
        I2 = sp1.float() @ self.W12
        sp2 = self.L2.step(I2)
        return sp2

# ─────────────────────────────────────────────────────────────────
# 2. PHONETIC MAPPING (Exp 16 - 23D Acoustic/Formant)
# ─────────────────────────────────────────────────────────────────

def f1f2_norm(f1, f2): return [np.clip(f1/1200.0, 0, 1), np.clip((f2-200)/2600.0, 0, 1)]
FORMANT_DATA = {'a':f1f2_norm(800,1300), 'i':f1f2_norm(280,2300), 'u':f1f2_norm(280,700), 'e':f1f2_norm(400,2000), 'o':f1f2_norm(490,800)}
TRANSLIT = {'A':'a','I':'i','U':'u','R':'a','L':'a','kh':'k','gh':'g','ch':'c','jh':'j','Th':'t','Dh':'d','th':'t','dh':'d','ph':'p','bh':'b','sh':'s','sh2':'s','ng':'n','S':'s','M':'m'}

def embed_23(char, vr):
    c = char.lower()
    c = TRANSLIT.get(char, c)[0]
    f = FORMANT_DATA.get(c, [0.4, 0.4])
    return np.concatenate([np.zeros(12), f, np.zeros(8), [vr]])

def encode_root_snn(model, root_str):
    chars = list(root_str)
    v_ratio = sum(1 for c in chars if c.lower() in 'aiueo') / max(len(chars), 1)
    model.reset()
    with torch.no_grad():
        for c in chars:
            pv = embed_23(c, v_ratio)
            I = torch.tensor(pv, dtype=torch.float32).unsqueeze(0).to(device)
            for _ in range(20): model.step(I)
        for _ in range(20): model.step(torch.zeros(1, 23).to(device))
    return model.L2.spike_counts.cpu().numpy().squeeze()

# ─────────────────────────────────────────────────────────────────
# 3. NORMALIZED REWARD CALCULATION
# ─────────────────────────────────────────────────────────────────

DATA_PATH = r'c:\Users\amiku\Downloads\AI Research New Paradigm\SampleData\task1_axis_prediction.csv'
t1 = pd.read_csv(DATA_PATH)
axes = t1['actual_axis'].values
ids = LabelEncoder().fit_transform(axes)

# Sample pairs for contrastive reward (balanced subset)
pos_pairs, neg_pairs = [], []
for i in range(len(t1)):
    for j in range(i+1, len(t1)):
        if axes[i] == axes[j]: pos_pairs.append((i, j))
        else: neg_pairs.append((i, j))
random.shuffle(pos_pairs)
random.shuffle(neg_pairs)
pos_pairs, neg_pairs = pos_pairs[:500], neg_pairs[:500]

def get_reward_and_ari(test_model):
    states = []
    for _, r in t1.iterrows():
        states.append(encode_root_snn(test_model, r['root']))
    states = np.array(states)
    mean_fire = states.mean()
    
    if states.sum() == 0: return -10.0, 0.0, 0.0
    
    # 1. L2 Normalization (Crucial Fix)
    norms = np.linalg.norm(states, axis=1, keepdims=True) + 1e-8
    states_norm = states / norms
    
    # 2. Cosine Distance Contrastive Reward (Range: -1 to 1 per pair)
    # distance = 1 - cosine_similarity. Pull same-axis together, push different apart.
    reward = 0.0
    for i, j in pos_pairs:
        dist = 1.0 - np.dot(states_norm[i], states_norm[j])
        reward -= dist # Minimize distance for positive pairs
    for i, j in neg_pairs:
        dist = 1.0 - np.dot(states_norm[i], states_norm[j])
        reward += dist # Maximize distance for negative pairs
        
    reward /= (len(pos_pairs) + len(neg_pairs))
    
    # 3. ARI Calculation
    scaler = StandardScaler()
    norm_states = scaler.fit_transform(states)
    pred = KMeans(n_clusters=5, random_state=42, n_init=1).fit_predict(norm_states)
    ari = adjusted_rand_score(ids, pred)
    
    return reward, ari, mean_fire

# ─────────────────────────────────────────────────────────────────
# 4. GRPO OPTIMIZATION LOOP
# ─────────────────────────────────────────────────────────────────

torch.manual_seed(42)
np.random.seed(42)
model = TwoLayerSNN().to(device)

G = 8
SIGMA = 0.02
ETA = 0.005
ROUNDS = 20  # Reduced for speed

params = ['VT', 'a', 'b', 'tau_w']
base_vals = {p: getattr(model.L2, p).data.clone() for p in params}

print(f"Starting Exp 35: G={G}, Sigma={SIGMA}, Eta={ETA}, Rounds={ROUNDS}", flush=True)

for round_i in range(ROUNDS + 1):
    if round_i % 10 == 0:
        for p in params: getattr(model.L2, p).data = base_vals[p]
        reward, ari, rate = get_reward_and_ari(model)
        print(f"Round {round_i:2d} | Reward: {reward:+.4f} | ARI: {ari:.4f} | Rate: {rate:.2f}", flush=True)

    if round_i == ROUNDS: break

    epsilons = {p: [torch.randn(1, 64).to(device) * SIGMA * base_vals[p].abs().mean() for _ in range(G)] for p in params}
    rewards = []
    
    for g_idx in range(G):
        for p in params:
            val = base_vals[p] + epsilons[p][g_idx]
            # Constraints
            if p == 'VT': val = torch.clamp(val, -65.0, -35.0)
            if p == 'tau_w': val = torch.clamp(val, 10.0, 500.0)
            if p in ['a', 'b']: val = torch.clamp(val, 0.0, 200.0)
            getattr(model.L2, p).data = val
        r, _, _ = get_reward_and_ari(model)
        rewards.append(r)
        
    mu_r, std_r = np.mean(rewards), np.std(rewards) + 1e-8
    adv = [(r - mu_r) / std_r for r in rewards]
    
    for p in params:
        delta = torch.sum(torch.stack([a * e for a, e in zip(adv, epsilons[p])]), dim=0)
        base_vals[p] = base_vals[p] + ETA * delta

print("\n=================================================================", flush=True)
print("EXP 35 FINAL RESULTS", flush=True)
for p in params: getattr(model.L2, p).data = base_vals[p]
final_reward, final_ari, final_rate = get_reward_and_ari(model)
print(f"  Final ARI  : {final_ari:.4f}", flush=True)
print(f"  Final Rate : {final_rate:.2f} spikes/root", flush=True)
print(f"  Baseline   : 0.0591", flush=True)
print(f"  Target     : 0.0690", flush=True)
print("=================================================================", flush=True)
