import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import gc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*65)
print("DDIN Exp 40b -- SCALING CURVE for rho_critical")
print("Mapping ARI vs N: 500, 1000, 1500 roots")
print("="*65)

def devanagari_to_slp1(text):
    vowels = {'अ':'a','आ':'A','इ':'i','ई':'I','उ':'u','ऊ':'U','ऋ':'f','ॠ':'F','ऌ':'x','ॡ':'X','ए':'e','ऐ':'E','ओ':'o','औ':'O'}
    vowel_signs = {'ा':'A','ि':'i','ी':'I','ु':'u','ू':'U','ृ':'f','ॄ':'F','ॢ':'x','ॣ':'X','े':'e','ै':'E','ो':'o','ौ':'O'}
    consonants = {'क':'k','ख':'K','ग':'g','घ':'G','ङ':'N','च':'c','छ':'C','ज':'j','झ':'J','ञ':'Y','ट':'w','ठ':'W','ड':'q','ढ':'Q','ण':'R','त':'t','थ':'T','द':'d','ध':'D','न':'n','प':'p','फ':'P','ब':'b','भ':'B','म':'m','य':'y','र':'r','ल':'l','व':'v','श':'S','ष':'z','स':'s','ह':'h'}
    misc = {'ं':'M', 'ः':'H', 'ँ':'~', '्':''}
    res = ""
    for i, char in enumerate(text):
        if char in vowels: res += vowels[char]
        elif char in consonants:
            res += consonants[char]
            if i + 1 < len(text) and (text[i+1] in vowel_signs or text[i+1] == '्'): pass
            else: res += 'a'
        elif char in vowel_signs: res += vowel_signs[char]
        elif char in misc: res += misc[char]
        else: res += char
    return res

def extract_artha_stem_tensor(artha_slp1):
    axis_stems = {
        0: ['gat', 'cal', 'gam', 'car', 'kram', 'vicar', 'sarp', 'pad'],
        1: ['satt', 'utpat', 'jan', 'Sabd', 'dIpt', 'prakAS', 'BAv', 'jIv'],
        2: ['pAk', 'vikAr', 'saMsk', 'kriy', 'nirmaA', 'kf'],
        3: ['hiMs', 'Bed', 'Cid', 'nAS', 'mAr', 'viDvaMs', 'tud'],
        4: ['DAr', 'pAl', 'saMvar', 'rakz', 'banD', 'sTA', 'ruD']
    }
    tensor = np.zeros(5)
    for i, stems in axis_stems.items():
        if any(s in artha_slp1 for s in stems): tensor[i] = 1.0
    s = np.sum(tensor)
    return tensor/s if s>0 else np.ones(5)*0.2

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
        self.theta = torch.ones(1, size).to(device) * 0.1
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
        self.theta += 0.005 * (spikes.float() - self.theta)
        return spikes

class MegaSNN(nn.Module):
    def __init__(self, input_dim=28):
        super().__init__()
        self.L1 = AdExPopulation(size=1024, a=0.0, b=0.0, tau_w=5.0, gL=15.0)
        mask1 = (torch.rand(input_dim, 1024) < 0.18).float()
        self.proj_in = nn.Parameter(torch.randn(input_dim, 1024) * 1100.0 * mask1)
        self.L2 = AdExPopulation(size=512, a=2.0, b=80.0, tau_w=150.0, gL=2.0)
        mask2 = (torch.rand(1024, 512) < 0.18).float()
        self.W12 = nn.Parameter(torch.randn(1024, 512) * 410.0 * mask2)

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

def process_corpus(model, dataset):
    all_states = []
    all_labels = []
    for i, item in enumerate(dataset):
        root_dv = item['dhatu']
        artha_dv = item['artha']
        root_slp1 = devanagari_to_slp1(root_dv)
        artha_slp1 = devanagari_to_slp1(artha_dv)
        a_tensor = extract_artha_stem_tensor(artha_slp1)
        label = np.argmax(a_tensor) if np.max(a_tensor) > 0.2 else -1
        chars = list(root_slp1)
        v_ratio = sum(1 for c in chars if c.lower() in 'aiueo') / max(len(chars), 1)
        model.reset()
        with torch.no_grad():
            for c in chars:
                pv = embed_acoustic_23(c, v_ratio)
                I = torch.tensor(np.concatenate([pv, a_tensor]), dtype=torch.float32).unsqueeze(0).to(device)
                for _ in range(20): model.step(I)
            for _ in range(20):
                model.step(torch.tensor(np.concatenate([np.zeros(23), a_tensor]), dtype=torch.float32).unsqueeze(0).to(device))
        all_states.append(model.L2.spike_counts.cpu().numpy().squeeze())
        all_labels.append(label)
    return np.array(all_states), np.array(all_labels)

with open(r'temp_ashtadhyayi_data/dhatu/data.txt', encoding='utf-8') as f:
    raw_data = json.load(f)
full_corpus = raw_data['data']

D_reservoir = 512
results = {}

for N in [500, 1000, 1500]:
    rho = N / D_reservoir
    print(f"\n{'='*60}")
    print(f"N = {N}  -->  rho = {rho:.4f}")
    print(f"{'='*60}")
    corpus_subset = full_corpus[:N]
    n_batches = max(1, N // 200)
    print(f"Processing {N} roots in {n_batches} mini-batches of ~200...")
    model = MegaSNN().to(device)
    states, labels = process_corpus(model, corpus_subset)
    mask = labels != -1
    valid_states = states[mask]
    valid_labels = labels[mask]
    states_norm = StandardScaler().fit_transform(valid_states)
    pred = KMeans(n_clusters=5, n_init=10).fit_predict(states_norm)
    ari = adjusted_rand_score(valid_labels, pred)
    mean_fr = states.mean()
    print(f"\n  --> ARI = {ari:.4f}  |  Mean FR = {mean_fr:.4f}  |  Valid roots: {len(valid_states)}/{N}")
    results[N] = {'ari': ari, 'rho': rho, 'mean_fr': mean_fr, 'n_valid': len(valid_states)}
    del model
    gc.collect()
    torch.cuda.empty_cache()

print("\n" + "="*65)
print("SCALING CURVE SUMMARY")
print("="*65)
print(f"{'N':>6}  {'rho':>8}  {'ARI':>8}")
print("-"*25)
for N in [500, 1000, 1500]:
    r = results[N]
    print(f"{N:>6}  {r['rho']:>8.4f}  {r['ari']:>8.4f}")
print("="*65)

print("\nPhase transition analysis:")
rho_critical = None
ari_values = [results[N]['ari'] for N in [500, 1000, 1500]]
if max(ari_values) - min(ari_values) > 0.5:
    print("  Large ARI range detected across scaling points.")
    for i, N in enumerate([500, 1000, 1500]):
        if results[N]['ari'] > 0.7:
            print(f"  --> rho_critical between {rho_critical:.4f} and {results[N]['rho']:.4f} (ARI jumped to {results[N]['ari']:.4f} at N={N})")
            break
        rho_critical = results[N]['rho']
else:
    print("  ARI range too small to identify sharp transition.")
    for N in [500, 1000, 1500]:
        print(f"  N={N}, rho={results[N]['rho']:.4f} --> ARI={results[N]['ari']:.4f}")
