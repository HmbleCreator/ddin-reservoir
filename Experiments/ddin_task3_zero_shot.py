import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import gc
from scipy.spatial.distance import euclidean

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*65)
print("DDIN Task 3 -- ZERO-SHOT ALIEN INFERENCE (FINAL)")
print("Phase 11: Testing Semantic Generalization")
print("="*65)

# 1. LOAD ARCHITECTURE & CENTROIDS
centroids = np.load('centroids.npy')
axes = ['MOT', 'EXP', 'TRN', 'SEP', 'CNT']

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

class MegaSNN(nn.Module):
    def __init__(self, input_dim=28):
        super().__init__()
        self.L1 = AdExPopulation(size=1024)
        self.proj_in = nn.Parameter(torch.randn(input_dim, 1024) * 1100.0) 
        self.L2 = AdExPopulation(size=512)
        self.W12 = nn.Parameter(torch.randn(1024, 512) * 410.0)

    def reset(self):
        self.L1.reset_states()
        self.L2.reset_states()

    def step(self, I_input):
        I1 = I_input @ self.proj_in
        sp1 = self.L1.step(I1)
        I2 = sp1.float() @ self.W12
        sp2 = self.L2.step(I2)
        return sp2

def embed_acoustic_23(char_slp1, vr):
    FORMANT_DATA = {'a':[0.67, 0.42], 'i':[0.23, 0.81], 'u':[0.23, 0.19], 'e':[0.33, 0.69], 'o':[0.41, 0.23]}
    c = char_slp1.lower()[0]
    f = FORMANT_DATA.get(c, [0.4, 0.4])
    return np.concatenate([np.zeros(12), f, np.zeros(8), [vr]])

# 2. INFERENCE LOOP
with open('alien_roots.json') as f:
    alien_roots = json.load(f)

model = MegaSNN().to(device)
zero_prior = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], dtype=torch.float32).unsqueeze(0).to(device)

print(f"Predicting semantics for {len(alien_roots)} alien roots...")
final_predictions = []

for root in alien_roots:
    chars = list(root)
    v_ratio = sum(1 for c in chars if c.lower() in 'aiueo') / max(len(chars), 1)
    model.reset()
    with torch.no_grad():
        for c in chars:
            pv = embed_acoustic_23(c, v_ratio)
            I = torch.tensor(np.concatenate([pv, [0.2]*5]), dtype=torch.float32).unsqueeze(0).to(device)
            for _ in range(20): model.step(I)
        for _ in range(20):
            model.step(torch.tensor(np.concatenate([np.zeros(23), [0.2]*5]), dtype=torch.float32).unsqueeze(0).to(device))
    
    state = model.L2.spike_counts.cpu().numpy().squeeze()
    # Find nearest centroid
    dists = [euclidean(state, c) for c in centroids]
    pred_axis = axes[np.argmin(dists)]
    final_predictions.append((root, pred_axis, np.min(dists)))

# 3. RESULTS SUMMARY
df = pd.DataFrame(final_predictions, columns=['root', 'predicted_axis', 'distance'])
print("\nTOP ALIEN SEMANTIC ASSIGNMENTS:")
print(df.head(15).to_string(index=False))

print(f"\nDistribution across axes:")
print(df['predicted_axis'].value_counts())
