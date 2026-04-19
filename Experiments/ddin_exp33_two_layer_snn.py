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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*65)
print("DDIN Exp 33 -- TWO-LAYER ADEX SNN (THE NEUROMORPHIC PORT)")
print("Phase 7B: Conquering the Seizure Boundary via Hierarchy")
print("="*65)

# ─────────────────────────────────────────────────────────────────
# 1. ADEX DIFFERENTIAL ENGINE (PyTorch Native)
# ─────────────────────────────────────────────────────────────────

class AdExPopulation(nn.Module):
    """
    Adaptive Exponential Integrate-and-Fire Population.
    Equations:
    C * dV/dt = -gL(V - EL) + gL*DeltaT * exp((V - VT)/DeltaT) - w + I
    tau_w * dw/dt = a(V - EL) - w
    """
    def __init__(self, size, dt=1.0, 
                 C=200.0, gL=10.0, EL=-70.0, VT=-50.0, DeltaT=2.0, 
                 Vpeak=0.0, Vreset=-58.0, 
                 a=2.0, b=80.0, tau_w=30.0):
        super().__init__()
        self.size = size
        self.dt = dt
        
        # Physical Parameters (per neuron)
        self.C = C
        self.gL = gL
        self.EL = EL
        self.VT = VT
        self.DeltaT = DeltaT
        self.Vpeak = Vpeak
        self.Vreset = Vreset
        
        # Adaptation Parameters
        self.a = nn.Parameter(torch.ones(size) * a)
        self.b = nn.Parameter(torch.ones(size) * b)
        self.tau_w = nn.Parameter(torch.ones(size) * tau_w)
        
        # State Variables
        self.V = torch.ones(1, size).to(device) * EL
        self.w = torch.zeros(1, size).to(device)
        self.spike_counts = torch.zeros(1, size).to(device)

    def reset_states(self):
        self.V = torch.ones(1, self.size).to(device) * self.EL
        self.w = torch.zeros(1, self.size).to(device)
        self.spike_counts = torch.zeros(1, self.size).to(device)

    def step(self, I_ext):
        # V-update
        # Explanatory term: gL * DeltaT * exp((V - VT)/DeltaT)
        exp_term = self.gL * self.DeltaT * torch.exp((self.V - self.VT) / self.DeltaT)
        dV = (-self.gL * (self.V - self.EL) + exp_term - self.w + I_ext) / self.C
        self.V += self.dt * dV
        
        # w-update
        dw = (self.a * (self.V - self.EL) - self.w) / self.tau_w
        self.w += self.dt * dw
        
        # Spike detection
        spikes = self.V >= self.Vpeak
        self.spike_counts += spikes.float()
        
        # Reset
        self.V = torch.where(spikes, torch.tensor(self.Vreset).to(device), self.V)
        self.w = torch.where(spikes, self.w + self.b, self.w)
        
        return spikes

# ─────────────────────────────────────────────────────────────────
# 2. PHONOLOGICAL ENCODING (Spike-Rate Input)
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
PRATYAHARA_SETS = [AC] # Simplified for this script

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

def embed_23(ph_char, vowel_ratio):
    ac16   = PHONEME_VECTORS_16.get(ph_char, [0.5]*16)
    manner = [ac16[d] for d in MANNER_DIMS]
    form   = [x * 4.0 for x in FORMANT_F1F2.get(ph_char, [0.4, 0.4])]
    prat   = [1.0 if ph_char in AC else 0.0 for _ in range(8)] # dummy prat for now
    return np.concatenate([manner, form, prat, [vowel_ratio]])

# ─────────────────────────────────────────────────────────────────
# 3. HIARARCHICAL SNN MODEL
# ─────────────────────────────────────────────────────────────────

class TwoLayerSNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: Fast Encoder
        # Low tau_w, Low b = Tonic spiking. 128 neurons.
        self.L1 = AdExPopulation(size=128, a=0.0, b=0.0, tau_w=5.0, gL=15.0)
        self.proj_in = nn.Parameter(torch.randn(23, 128) * 900.0) 
        
        # Layer 2: Slow Organizer
        # Reduced Leak (gL=2.0) for longer integration memory.
        self.L2 = AdExPopulation(size=64, a=2.0, b=80.0, tau_w=150.0, gL=2.0)
        self.W12 = nn.Parameter(torch.randn(128, 64) * 350.0) # Synaptic weights

    def reset(self):
        self.L1.reset_states()
        self.L2.reset_states()

    def step(self, I_input):
        # 1. Drive Layer 1
        I1 = I_input @ self.proj_in
        spikes1 = self.L1.step(I1)
        
        # 2. Drive Layer 2 (Shielded: only receives spikes from L1)
        # We simulate a simple current-based synapse for now
        I2 = spikes1.float() @ self.W12
        spikes2 = self.L2.step(I2)
        
        return spikes1, spikes2

# ─────────────────────────────────────────────────────────────────
# 4. INFERENCE LOOP
# ─────────────────────────────────────────────────────────────────

def encode_root_snn(model, root_str, steps_per_ph=20):
    chars = root_to_chars(root_str)
    if not chars: return np.zeros(64)
    
    n_vowels = sum(1 for c in chars if c in AC)
    v_ratio = n_vowels / max(len(chars), 1)
    
    model.reset()
    
    with torch.no_grad():
        for c in chars:
            pv = embed_23(c, v_ratio)
            I_in = torch.tensor(pv, dtype=torch.float32).unsqueeze(0).to(device)
            for _ in range(steps_per_ph):
                model.step(I_in)
                
        # Silence step (decay context)
        I_zero = torch.zeros(1, 23).to(device)
        for _ in range(20):
            model.step(I_zero)
            
    return model.L2.spike_counts.cpu().numpy()

# ─────────────────────────────────────────────────────────────────
# 5. EXECUTION & EVALUATION
# ─────────────────────────────────────────────────────────────────

DATA = r'c:\Users\amiku\Downloads\AI Research New Paradigm\SampleData'
t1 = pd.read_csv(f'{DATA}/task1_axis_prediction.csv')

torch.manual_seed(42)
np.random.seed(42)

model = TwoLayerSNN().to(device)
model.eval()

print(f"Processing {len(t1)} roots through SNN populations...")
start_time = time.time()

readout_states = []
for idx, r in t1.iterrows():
    if idx % 30 == 0: print(f"  Root {idx}/{len(t1)}...")
    counts = encode_root_snn(model, r['root'])
    readout_states.append(counts.squeeze())

readout_states = np.array(readout_states)
print(f"SNN Simulation Complete in {time.time() - start_time:.2f}s")

# Check for seizures
mean_firing = readout_states.mean()
print(f"Global Average Spikes per Root: {mean_firing:.2f}")
if mean_firing > 100:
    print(">>> WARNING: High firing rate detected. Potential Seizure Boundary.")
else:
    print(">>> SUCCESS: Firing rates are stable and asynchronous.")

# Evaluate ARI
scaler = StandardScaler()
norm_states = scaler.fit_transform(readout_states)

le = LabelEncoder()
axis_ids = le.fit_transform(t1['actual_axis'].values)

ari_scores = []
for i in range(10):
    # Standard 5-cluster evaluation
    pred = KMeans(n_clusters=5, random_state=i, n_init=5).fit_predict(norm_states)
    ari_scores.append(adjusted_rand_score(axis_ids, pred))

print("\n" + "="*65)
print(f"SNN Raw Capacity Result (Phase 7B Baseline)")
print(f"Mean ARI: {np.mean(ari_scores):.4f} (Max: {np.max(ari_scores):.4f})")
print(f"Previous ODE Baseline (Exp 29): 0.0492")
print("="*65)
