import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import json
import gc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

SEEDS = [42, 43, 44, 45, 46]
D_RESERVOIR = 512

def devanagari_to_slp1(text):
    vowels  = {'अ':'a','आ':'A','इ':'i','ई':'I','उ':'u','ऊ':'U','ऋ':'f','ॠ':'F','ऌ':'x','ॡ':'X','ए':'e','ऐ':'E','ओ':'o','औ':'O'}
    vsigns  = {'ा':'A','ि':'i','ी':'I','ु':'u','ू':'U','ृ':'f','ॄ':'F','ॢ':'x','ॣ':'X','े':'e','ै':'E','ो':'o','ौ':'O'}
    cons    = {'क':'k','ख':'K','ग':'g','घ':'G','ङ':'N','च':'c','छ':'C','ज':'j','झ':'J','ञ':'Y',
               'ट':'w','ठ':'W','ड':'q','ढ':'Q','ण':'R','त':'t','थ':'T','द':'d','ध':'D','न':'n',
               'प':'p','फ':'P','ब':'b','भ':'B','म':'m','य':'y','र':'r','ल':'l','व':'v',
               'श':'S','ष':'z','स':'s','ह':'h'}
    misc    = {'ं':'M','ः':'H','ँ':'~','्':''}
    res = ""
    for i, char in enumerate(text):
        if char in vowels: res += vowels[char]
        elif char in cons:
            res += cons[char]
            if i+1 < len(text) and (text[i+1] in vsigns or text[i+1] == '्'): pass
            else: res += 'a'
        elif char in vsigns: res += vsigns[char]
        elif char in misc:   res += misc[char]
        else: res += char
    return res

def extract_artha_stem_tensor(artha_slp1):
    axis_stems = {
        0: ['gat','cal','gam','car','kram','vicar','sarp','pad'],
        1: ['satt','utpat','jan','Sabd','dIpt','prakAS','BAv','jIv'],
        2: ['pAk','vikAr','saMsk','kriy','nirmaA','kf'],
        3: ['hiMs','Bed','Cid','nAS','mAr','viDvaMs','tud'],
        4: ['DAr','pAl','saMvar','rakz','banD','sTA','ruD']
    }
    tensor = np.zeros(5)
    for i, stems in axis_stems.items():
        if any(s in artha_slp1 for s in stems): tensor[i] = 1.0
    s = np.sum(tensor)
    return tensor/s if s > 0 else np.ones(5)*0.2

class AdExPopulation(nn.Module):
    def __init__(self, size, dt=1.0, C=200.0, gL=10.0, EL=-70.0, VT=-50.0,
                 DeltaT=2.0, Vpeak=0.0, Vreset=-58.0, a=2.0, b=80.0, tau_w=30.0):
        super().__init__()
        self.size = size; self.dt = dt
        self.C = C; self.gL = gL; self.EL = EL
        self.DeltaT = DeltaT; self.Vpeak = Vpeak; self.Vreset = Vreset
        self.VT   = nn.Parameter(torch.ones(1, size) * VT)
        self.a    = nn.Parameter(torch.ones(1, size) * a)
        self.b    = nn.Parameter(torch.ones(1, size) * b)
        self.tau_w= nn.Parameter(torch.ones(1, size) * tau_w)
        self.V    = torch.ones(1, size) * EL
        self.w    = torch.zeros(1, size)
        self.theta= torch.ones(1, size) * 0.1
        self.spike_counts = torch.zeros(1, size)

    def reset_states(self):
        self.V = torch.ones(1, self.size) * self.EL
        self.w = torch.zeros(1, self.size)
        self.spike_counts = torch.zeros(1, self.size)

    def step(self, I_ext):
        exp_term = self.gL * self.DeltaT * torch.exp((self.V - self.VT) / self.DeltaT)
        dV = (-self.gL*(self.V-self.EL) + exp_term - self.w + I_ext) / self.C
        self.V += self.dt * dV
        dw = (self.a*(self.V-self.EL) - self.w) / self.tau_w
        self.w += self.dt * dw
        spikes = (self.V >= self.Vpeak)
        self.spike_counts += spikes.float()
        self.V = torch.where(spikes, torch.tensor(self.Vreset), self.V)
        self.w = torch.where(spikes, self.w + self.b, self.w)
        self.theta += 0.005 * (spikes.float() - self.theta)
        return spikes

class MegaSNN(nn.Module):
    def __init__(self, input_dim=28, seed=42):
        super().__init__()
        torch.manual_seed(seed)
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
        return self.L2.step(I2)

def embed_acoustic_23(char_slp1, vr):
    FORMANT_DATA = {'a':[0.67,0.42],'i':[0.23,0.81],'u':[0.23,0.19],
                    'e':[0.33,0.69],'o':[0.41,0.23]}
    c = char_slp1.lower()[0]
    f = FORMANT_DATA.get(c, [0.4, 0.4])
    return np.concatenate([np.zeros(12), f, np.zeros(8), [vr]])

def process_corpus(model, dataset, device):
    all_states, all_labels = [], []
    for item in dataset:
        root_slp1 = devanagari_to_slp1(item['dhatu'])
        artha_slp1= devanagari_to_slp1(item['artha'])
        a_tensor  = extract_artha_stem_tensor(artha_slp1)
        label     = np.argmax(a_tensor) if np.max(a_tensor) > 0.2 else -1
        chars     = list(root_slp1)
        v_ratio   = sum(1 for c in chars if c.lower() in 'aiueo') / max(len(chars), 1)
        model.reset()
        with torch.no_grad():
            for c in chars:
                pv = embed_acoustic_23(c, v_ratio)
                I  = torch.tensor(np.concatenate([pv, a_tensor]), dtype=torch.float32).unsqueeze(0).to(device)
                for _ in range(20): model.step(I)
            for _ in range(20):
                model.step(torch.tensor(np.concatenate([np.zeros(23), a_tensor]), dtype=torch.float32).unsqueeze(0).to(device))
        all_states.append(model.L2.spike_counts.cpu().numpy().squeeze())
        all_labels.append(label)
    return np.array(all_states), np.array(all_labels)

with open(r'temp_ashtadhyayi_data/dhatu/data.txt', encoding='utf-8') as f:
    full_corpus = json.load(f)['data']

print("="*65)
print("DDIN Exp 40c — 5-SEED SWEEP FOR BISTABILITY ANALYSIS")
print("="*65)

results = {}
for N in [500, 1000, 1500]:
    rho = N / D_RESERVOIR
    print(f"\n{'='*60}")
    print(f"N = {N}  |  rho = {rho:.4f}  |  5 seeds")
    print(f"{'='*60}")
    seed_aris = []
    corpus_subset = full_corpus[:N]
    for si, seed in enumerate(SEEDS):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = MegaSNN(input_dim=28, seed=seed).to(device)
        states, labels = process_corpus(model, corpus_subset, device)
        mask = labels != -1
        valid_states  = states[mask]
        valid_labels  = labels[mask]
        states_norm   = StandardScaler().fit_transform(valid_states)
        pred          = KMeans(n_clusters=5, n_init=10).fit_predict(states_norm)
        ari           = adjusted_rand_score(valid_labels, pred)
        seed_aris.append(ari)
        print(f"  seed={seed}: ARI={ari:.4f}  |  valid={len(valid_states)}/{N}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    mean_ari = np.mean(seed_aris)
    std_ari  = np.std(seed_aris)
    print(f"  --> mean={mean_ari:.4f}  std={std_ari:.4f}")
    results[N] = {'ari': seed_aris, 'mean': mean_ari, 'std': std_ari, 'rho': rho}

print("\n" + "="*65)
print("FINAL RESULTS TABLE")
print("="*65)
print(f"{'N':>6}  {'rho':>8}  {'mean':>8}  {'std':>8}  {'values'}")
print("-"*70)
for N in [500, 1000, 1500]:
    r = results[N]
    vals = '  '.join([f'{v:.4f}' for v in r['ari']])
    print(f"{N:>6}  {r['rho']:>8.4f}  {r['mean']:>8.4f}  {r['std']:>8.4f}")
    print(f"         {vals}")
print("="*65)

print("\nPhase transition analysis:")
for N in [500, 1000, 1500]:
    r = results[N]
    print(f"  N={N}, rho={r['rho']:.4f}: ARI={r['mean']:.4f} ± {r['std']:.4f}")
max_std_N = max(results.items(), key=lambda x: x[1]['std'])
print(f"\n  --> Maximum variance (bistability signature): N={max_std_N[0]}  std={max_std_N[1]['std']:.4f}")
