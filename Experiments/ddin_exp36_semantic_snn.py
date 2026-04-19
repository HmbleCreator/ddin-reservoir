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
print("DDIN Exp 36 -- SEMANTIC PRIOR INJECTION")
print("Phase 9A: Breaking the 0.06 Ceiling with Paninian Ganas")
print("="*65)

# ─────────────────────────────────────────────────────────────────
# 1. DATA ENRICHMENT (GANAS)
# ─────────────────────────────────────────────────────────────────

# Representative Paninian Gana Mapping for the 150 Benchmark Roots
GANA_MAPPING = {
    'kR': 8, 'kram': 1, 'krand': 1, 'kRp': 1, 'kruS': 1, 'klp': 1, 'kSip': 6, 'kath': 10, 'kuj': 1, 'kan': 1,
    'kAz': 1, 'kIrt': 10, 'kup': 4, 'kSar': 1, 'gai': 1, 'gam': 1, 'garj': 1, 'gRh': 9, 'guh': 1, 'han': 2,
    'hR': 1, 'hu': 3, 'hve': 1, 'kI': 1, 'kep': 1, 'gup': 1, 'kzip': 6, 'hims': 7, 'kLp': 1, 'ci': 5,
    'cit': 1, 'cint': 10, 'cur': 10, 'cud': 10, 'jan': 4, 'jap': 1, 'jalp': 1, 'ji': 1, 'jval': 1, 'jash': 1,
    'yaj': 1, 'yam': 1, 'yu': 2, 'yuj': 7, 'car': 1, 'chad': 10, 'chid': 7, 'jIv': 1, 'jak': 2, 'cup': 1,
    # (Defaulting unspecified roots to Class 1 as per Dhatupatha statistics)
}

def get_gana_vec(root_str):
    g = GANA_MAPPING.get(root_str, 1) # Default to Bhvadi (Class 1)
    vec = np.zeros(10)
    vec[g-1] = 1.0
    return vec

# ─────────────────────────────────────────────────────────────────
# 2. ADEX DIFFERENTIAL ENGINE
# ─────────────────────────────────────────────────────────────────

class AdExPopulation(nn.Module):
    def __init__(self, size, dt=1.0, C=200.0, gL=10.0, EL=-70.0, VT=-50.0, DeltaT=2.0, Vpeak=0.0, Vreset=-58.0, a=2.0, b=80.0, tau_w=30.0):
        super().__init__()
        self.size = size
        self.dt = dt
        self.C, self.gL, self.EL, self.DeltaT, self.Vpeak, self.Vreset = C, gL, EL, DeltaT, Vpeak, Vreset
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
    def __init__(self, input_dim=33): # 23 (acoustic) + 10 (Gana)
        super().__init__()
        self.L1 = AdExPopulation(size=128, a=0.0, b=0.0, tau_w=5.0, gL=15.0)
        self.proj_in = nn.Parameter(torch.randn(input_dim, 128) * 900.0) 
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
# 3. ENRICHED ENCODING PIPELINE (33D)
# ─────────────────────────────────────────────────────────────────

def f1f2_norm(f1, f2): return [np.clip(f1/1200.0, 0, 1), np.clip((f2-200)/2600.0, 0, 1)]
FORMANT_DATA = {'a':f1f2_norm(800,1300), 'i':f1f2_norm(280,2300), 'u':f1f2_norm(280,700), 'e':f1f2_norm(400,2000), 'o':f1f2_norm(490,800)}
TRANSLIT = {'A':'a','I':'i','U':'u','R':'a','L':'a','kh':'k','gh':'g','ch':'c','jh':'j','Th':'t','Dh':'d','th':'t','dh':'d','ph':'p','bh':'b','sh':'s','sh2':'s','ng':'n','S':'s','M':'m'}

def embed_acoustic_23(char, vr):
    c = char.lower()
    c = TRANSLIT.get(char, c)[0]
    f = FORMANT_DATA.get(c, [0.4, 0.4])
    return np.concatenate([np.zeros(12), f, np.zeros(8), [vr]])

def encode_root_semantic_snn(model, root_str):
    chars = list(root_str)
    v_ratio = sum(1 for c in chars if c.lower() in 'aiueo') / max(len(chars), 1)
    gana_vec = get_gana_vec(root_str)
    
    model.reset()
    with torch.no_grad():
        for c in chars:
            pv = embed_acoustic_23(c, v_ratio)
            # Inject Gaṇa as a static bias during the root presentation
            combined_33 = np.concatenate([pv, gana_vec])
            I = torch.tensor(combined_33, dtype=torch.float32).unsqueeze(0).to(device)
            for _ in range(20): model.step(I)
        # Decay phase
        for _ in range(20): 
            decay_33 = np.concatenate([np.zeros(23), gana_vec]) # Keep semantic bias during decay
            model.step(torch.tensor(decay_33, dtype=torch.float32).unsqueeze(0).to(device))
    return model.L2.spike_counts.cpu().numpy().squeeze()

# ─────────────────────────────────────────────────────────────────
# 4. EVALUATION & GRPO (Normalized)
# ─────────────────────────────────────────────────────────────────

DATA_PATH = r'c:\Users\amiku\Downloads\AI Research New Paradigm\SampleData\task1_axis_prediction.csv'
t1 = pd.read_csv(DATA_PATH)
axes = t1['actual_axis'].values
ids = LabelEncoder().fit_transform(axes)

def get_reward_and_ari(test_model):
    states = []
    for _, r in t1.iterrows():
        states.append(encode_root_semantic_snn(test_model, r['root']))
    states = np.array(states)
    if states.sum() == 0: return -10.0, 0.0, 0.0
    
    # L2 Norm + Cosine Distance
    norms = np.linalg.norm(states, axis=1, keepdims=True) + 1e-8
    states_norm = states / norms
    
    # We sample 1000 random pairs for reward calculation
    idx_i = np.random.randint(0, len(states), 1000)
    idx_j = np.random.randint(0, len(states), 1000)
    
    reward = 0.0
    for i, j in zip(idx_i, idx_j):
        dist = 1.0 - np.dot(states_norm[i], states_norm[j])
        if axes[i] == axes[j]: reward -= dist
        else: reward += dist
    reward /= 1000.0
    
    pred = KMeans(n_clusters=5, random_state=42, n_init=1).fit_predict(StandardScaler().fit_transform(states))
    ari = adjusted_rand_score(ids, pred)
    return reward, ari, states.mean()

# ─────────────────────────────────────────────────────────────────
# 5. RUN
# ─────────────────────────────────────────────────────────────────

model = TwoLayerSNN(input_dim=33).to(device)
G, SIGMA, ETA, ROUNDS = 8, 0.02, 0.005, 50
params = ['VT', 'a', 'b', 'tau_w']
base_vals = {p: getattr(model.L2, p).data.clone() for p in params}

print("Running Exp 36 (Gana Enrichment)...", flush=True)

for round_i in range(ROUNDS + 1):
    if round_i % 10 == 0:
        for p in params: getattr(model.L2, p).data = base_vals[p]
        reward, ari, rate = get_reward_and_ari(model)
        print(f"Round {round_i:2d} | Reward: {reward:+.4f} | ARI: {ari:.4f} | Rate: {rate:.2f}", flush=True)

    if round_i == ROUNDS: break

    epsilons = {p: [torch.randn(1, 64).to(device) * SIGMA * base_vals[p].abs().mean() for _ in range(G)] for p in params}
    rewards = []
    for g_idx in range(G):
        for p in params: getattr(model.L2, p).data = base_vals[p] + epsilons[p][g_idx]
        r, _, _ = get_reward_and_ari(model)
        rewards.append(r)
        
    mu_r, std_r = np.mean(rewards), np.std(rewards) + 1e-8
    adv = [(r - mu_r) / std_r for r in rewards]
    for p in params:
        delta = torch.sum(torch.stack([a * e for a, e in zip(adv, epsilons[p])]), dim=0)
        base_vals[p] = base_vals[p] + ETA * delta

print("\nEXP 36 FINAL (GANA-ENRICHED) ARI:", get_reward_and_ari(model)[1])
