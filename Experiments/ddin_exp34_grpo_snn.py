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
print("DDIN Exp 34 -- GRPO-SNN OPTIMIZATION")
print("Phase 8: Tuning the Physics of Abstraction")
print("="*65)

# ─────────────────────────────────────────────────────────────────
# 1. ADEX DIFFERENTIAL ENGINE (REUSABLE)
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
        
        # Optimized Parameters
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
    def __init__(self):
        super().__init__()
        # Layer 1: Fast Encoder
        self.L1 = AdExPopulation(size=128, a=0.0, b=0.0, tau_w=5.0, gL=15.0)
        self.proj_in = nn.Parameter(torch.randn(23, 128) * 900.0) 
        
        # Layer 2: Slow Organizer (Target for Physis Tuning)
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
# 2. ENCODING PIPELINE
# ─────────────────────────────────────────────────────────────────

TRANSLIT = {'A':'A','I':'I','U':'U','R':'R', 'kh':'K','gh':'G','ch':'C','jh':'J','Th':'Q','Dh':'X', 'th':'H','dh':'W','ph':'P','bh':'B','sh':'z','sh2':'x','ng':'N', 'S':'x', 'M':'m'}
def root_to_chars(s):
    chars, i = [], 0
    while i < len(s):
        if i+1 < len(s) and s[i:i+2] in TRANSLIT: chars.append(TRANSLIT[s[i:i+2]]); i += 2
        else: chars.append(TRANSLIT.get(s[i], s[i])); i += 1
    return chars

FORMANT_F1F2 = {'a':[0.67,0.42], 'i':[0.23,0.81], 'u':[0.23,0.19]} # Simplified for code safety
def embed_23(ph_char, vr):
    f = FORMANT_F1F2.get(ph_char.lower()[0], [0.4, 0.4])
    return np.concatenate([np.zeros(12), f, np.zeros(8), [vr]])

def encode_root_snn(model, root_str):
    chars = root_to_chars(root_str)
    if not chars: return np.zeros(64)
    v_ratio = sum(1 for c in chars if c in 'aiuAIU') / max(len(chars), 1)
    model.reset()
    with torch.no_grad():
        for c in chars:
            pv = embed_23(c, v_ratio)
            I_in = torch.tensor(pv, dtype=torch.float32).unsqueeze(0).to(device)
            for _ in range(20): model.step(I_in)
        for _ in range(20): model.step(torch.zeros(1, 23).to(device))
    return model.L2.spike_counts.cpu().numpy().squeeze()

# ─────────────────────────────────────────────────────────────────
# 3. CONTRASTIVE EVAL/REWARD
# ─────────────────────────────────────────────────────────────────

DATA = r'c:\Users\amiku\Downloads\AI Research New Paradigm\SampleData'
t1 = pd.read_csv(f'{DATA}/task1_axis_prediction.csv')
axes = t1['actual_axis'].values
loci = t1['locus'].values if 'locus' in t1.columns else np.random.choice(['T','P','C','D','L'], len(t1))

pos_pairs, neg_pairs = [], []
for i in range(len(t1)):
    for j in range(i+1, len(t1)):
        if axes[i] == axes[j]: pos_pairs.append((i, j))
        elif loci[i] == loci[j]: neg_pairs.append((i, j))

def evaluate_snn(test_model, k_eval=5):
    states = []
    for _, r in t1.iterrows():
        states.append(encode_root_snn(test_model, r['root']))
    states = np.array(states)
    
    # Contrastive Reward
    pos_dists = [np.linalg.norm(states[i] - states[j])**2 for (i,j) in pos_pairs[:500]]
    neg_dists = [np.linalg.norm(states[i] - states[j])**2 for (i,j) in neg_pairs[:500]]
    reward = np.mean(neg_dists) - (1.5 * np.mean(pos_dists)) if states.sum() > 0 else -100.0

    # ARI
    norm_states = StandardScaler().fit_transform(states) if states.sum() > 0 else states
    pred = KMeans(n_clusters=k_eval, random_state=42, n_init=1).fit_predict(norm_states)
    ari = adjusted_rand_score(LabelEncoder().fit_transform(axes), pred)
    return reward, ari, states.mean()

# ─────────────────────────────────────────────────────────────────
# 4. GRPO OPTIMIZATION 
# ─────────────────────────────────────────────────────────────────

torch.manual_seed(42)
np.random.seed(42)
model = TwoLayerSNN().to(device)

G = 8
ROUNDS = 20
SIGMA = 0.5
ETA = 0.1

print("Optimizing SNN Dynamics (VT, a, b, tau_w) via GRPO...", flush=True)

# Optimization targets
params = ['VT', 'a', 'b', 'tau_w']
base_vals = {p: getattr(model.L2, p).data.clone() for p in params}

for round_i in range(ROUNDS + 1):
    if round_i % 4 == 0:
        for p in params: getattr(model.L2, p).data = base_vals[p]
        reward, ari, mean_fire = evaluate_snn(model)
        print(f"Round {round_i:3d} | Reward = {reward:+.3f} | ARI = {ari:.4f} | Firing Rate = {mean_fire:.2f}", flush=True)

    if round_i == ROUNDS: break

    epsilons = {p: [torch.randn(1, 64).to(device) * SIGMA * base_vals[p].abs().mean() for _ in range(G)] for p in params}
    rewards = []
    for g_idx in range(G):
        for p in params:
            new_val = base_vals[p] + epsilons[p][g_idx]
            # Clamp to physical bounds
            if p == 'VT': new_val = torch.clamp(new_val, -65.0, -35.0)
            if p == 'tau_w': new_val = torch.clamp(new_val, 10.0, 500.0)
            if p in ['a', 'b']: new_val = torch.clamp(new_val, 0.0, 200.0)
            getattr(model.L2, p).data = new_val
        r, _, _ = evaluate_snn(model)
        rewards.append(r)
        
    mu_r, std_r = np.mean(rewards), np.std(rewards) + 1e-8
    adv = [(r - mu_r) / std_r for r in rewards]
    
    for p in params:
        delta = torch.sum(torch.stack([a * e for a, e in zip(adv, epsilons[p])]), dim=0)
        base_vals[p] = base_vals[p] + ETA * delta

print("\n=================================================================", flush=True)
print("PHASE 8 FINAL RESULTS", flush=True)
for p in params: getattr(model.L2, p).data = base_vals[p]
final_reward, final_ari, final_fire = evaluate_snn(model)
print(f"  Final ARI : {final_ari:.4f}", flush=True)
print(f"  Final Rate: {final_fire:.2f} spikes/root", flush=True)
print("=================================================================", flush=True)
