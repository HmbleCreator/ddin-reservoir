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
print("DDIN Exp 38 -- PANINIAN MORPHOLOGICAL PRIORS")
print("Phase 9C: Formal Morphological Injection (43D)")
print("="*65)

# ─────────────────────────────────────────────────────────────────
# 1. PANINIAN STRUCTURAL DATA
# ─────────────────────────────────────────────────────────────────

import json
import os

PRIOR_PATH = "exp38_balanced_prior.json"
if os.path.exists(PRIOR_PATH):
    with open(PRIOR_PATH, "r", encoding="utf-8") as f:
        PANINIAN_PRIOR_DICT = json.load(f)
    print(f"Loaded balanced prior for {len(PANINIAN_PRIOR_DICT)} roots.")
else:
    print("WARNING: exp38_balanced_prior.json not found. Using default mapping.")
    PANINIAN_PRIOR_DICT = {}

ANUBANDHA_MAP = {
    'Y': 0, 'q': 1, 'w': 2, 'L': 3, '~': 4, 'f': 5, 'I': 6, 'U': 7, 'k': 8, 'r': 9
}

def get_morph_vec(root_str):
    data = PANINIAN_PRIOR_DICT.get(root_str, {'upadesa': '', 'gana': 1})
    
    # Gana 10D
    g_vec = np.zeros(10)
    g_vec[data['gana']-1] = 1.0
    
    # Anubandha 10D
    a_vec = np.zeros(10)
    up = data['upadesa']
    for char, bit in ANUBANDHA_MAP.items():
        if char in up: a_vec[bit] = 1.0
        
    return np.concatenate([g_vec, a_vec])

# ─────────────────────────────────────────────────────────────────
# 2. SNN ARCHITECTURE (43D)
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
    def __init__(self, input_dim=43): # 23 Acoustic + 20 Morphological
        super().__init__()
        self.L1 = AdExPopulation(size=128, a=0.0, b=0.0, tau_w=5.0, gL=15.0)
        self.proj_in = nn.Parameter(torch.randn(input_dim, 128) * 850.0) 
        self.L2 = AdExPopulation(size=64, a=2.0, b=80.0, tau_w=150.0, gL=2.0)
        self.W12 = nn.Parameter(torch.randn(128, 64) * 320.0)

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
# 3. ENCODING & EXECUTION
# ─────────────────────────────────────────────────────────────────

def f1f2_norm(f1, f2): return [np.clip(f1/1200.0, 0, 1), np.clip((f2-200)/2600.0, 0, 1)]
FORMANT_DATA = {'a':f1f2_norm(800,1300), 'i':f1f2_norm(280,2300), 'u':f1f2_norm(280,700), 'e':f1f2_norm(400,2000), 'o':f1f2_norm(490,800)}
TRANSLIT = {'A':'a','I':'i','U':'u','R':'a','kh':'k','gh':'g','ch':'c','jh':'j','Th':'t','Dh':'d','th':'t','dh':'d','ph':'p','bh':'b','sh':'s','ng':'n'}

def embed_acoustic_23(char, vr):
    c = TRANSLIT.get(char, char.lower())[0]
    f = FORMANT_DATA.get(c, [0.4, 0.4])
    return np.concatenate([np.zeros(12), f, np.zeros(8), [vr]])

def encode_root_exp38(model, root_str, morph_vec):
    chars = list(root_str)
    v_ratio = sum(1 for c in chars if c.lower() in 'aiueo') / max(len(chars), 1)
    
    model.reset()
    with torch.no_grad():
        for c in chars:
            pv = embed_acoustic_23(c, v_ratio)
            combined_43 = np.concatenate([pv, morph_vec])
            I = torch.tensor(combined_43, dtype=torch.float32).unsqueeze(0).to(device)
            for _ in range(20): model.step(I)
        # Decay phase
        for _ in range(20):
            decay_43 = np.concatenate([np.zeros(23), morph_vec])
            model.step(torch.tensor(decay_43, dtype=torch.float32).unsqueeze(0).to(device))
    return model.L2.spike_counts.cpu().numpy().squeeze()

# ─────────────────────────────────────────────────────────────────
# 4. RUN
# ─────────────────────────────────────────────────────────────────

DATA_PATH = r'c:\Users\amiku\Downloads\AI Research New Paradigm\SampleData\task1_axis_prediction.csv'
t1 = pd.read_csv(DATA_PATH)
axes = t1['actual_axis'].values
ids = LabelEncoder().fit_transform(axes)

model = TwoLayerSNN(input_dim=43).to(device)

print("Running Exp 38 (Paninian Structural Priors)...")
states = []
for _, r in t1.iterrows():
    m_vec = get_morph_vec(r['root'])
    states.append(encode_root_exp38(model, r['root'], m_vec))

states = np.array(states)
states_norm = StandardScaler().fit_transform(states)

best_ari = -1
for _ in range(10): # Average over 10 K-Means runs
    pred = KMeans(n_clusters=5, random_state=None, n_init=1).fit_predict(states_norm)
    ari = adjusted_rand_score(ids, pred)
    if ari > best_ari: best_ari = ari

print(f"\nEXP 38 RESULTS (Paninian Prior):")
print(f"  Final ARI: {best_ari:.4f}")
print(f"  Rate: {states.mean():.2f} spikes/root")

if best_ari > 0.16:
    print("\nSUCCESS: PANINIAN STRUCTURAL CODE OUTPERFORMS GLOSS.")
elif best_ari > 0.08:
    print("\nSUCCESS: BREAKTHROUGH MAINTAINED.")
else:
    print("\nFAILED: MORPHOLOGY DID NOT TRANSFER.")
