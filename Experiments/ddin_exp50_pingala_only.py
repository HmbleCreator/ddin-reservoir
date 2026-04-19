"""
Exp 50: Piᅁgala-Only Baseline
Tests whether pure phonological signal (acoustic + Piᅁgala, NO prior) achieves ARI > 0.15.
This is the critical test: can semantics emerge from physics + combinatorics alone?
GPU script - run on Kaggle/Colab.
"""
import torch
import torch.nn as nn
import numpy as np
import json
import gc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

LONG_VOWELS = set('AEIOUfF')
SHORT_VOWELS = set('aiux')
CONSONANTS = set('kKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzsSh')

def devanagari_to_slp1(text):
    vowels = {'अ':'a','आ':'A','इ':'i','ई':'I','उ':'u','ऊ':'U','ऋ':'f','ॠ':'F','ऌ':'x','ॡ':'X','ए':'e','ऐ':'E','ओ':'o','औ':'O'}
    vowel_signs = {'ा':'A','ि':'i','ी':'I','ु':'u','ू':'U','ृ':'f','ॄ':'F','ॢ':'x','ॣ':'X','े':'e','ै':'E','ो':'o','ौ':'O'}
    consonants = {'क':'k','ख':'K','ग':'g','घ':'G','ङ':'N','च':'c','छ':'C','ज':'j','झ':'J','ञ':'Y','ट':'w','ठ':'W','ड':'q','ढ':'Q','ण':'R','त':'t','थ':'T','द':'d','ध':'D','न':'n','प':'p','फ':'P','ब':'b','भ':'B','म':'m','य':'y','र':'r','ल':'l','व':'v','ल':'S','ष':'z','स':'s','ह':'h'}
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

def compute_pingala_address(root_slp1, max_syllables=4):
    chars = list(root_slp1)
    syllables = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c in LONG_VOWELS or c in SHORT_VOWELS:
            is_guru = c in LONG_VOWELS
            j = i + 1
            cluster = 0
            while j < len(chars) and chars[j] in CONSONANTS:
                cluster += 1
                j += 1
            if cluster > 1: is_guru = True
            syllables.append(1 if is_guru else 0)
            i = j
        else: i += 1
    addr = syllables[:max_syllables]
    while len(addr) < max_syllables: addr.append(0)
    return addr

def extract_artha_stem_tensor(artha_slp1):
    axis_stems = {0: ['gat','cal','gam','car','kram','vicar','sarp','pad'], 1: ['satt','utpat','jan','Sabd','dIpt','prakAS','BAv','jIv'], 2: ['pAk','vikAr','saMsk','kriy','nirmaA','kf'], 3: ['hiMs','Bed','Cid','nAS','mAr','viDvaMs','tud'], 4: ['DAr','pAl','saMvar','rakz','banD','sTA','ruD']}
    tensor = np.zeros(5)
    for i, stems in axis_stems.items():
        if any(s in artha_slp1 for s in stems): tensor[i] = 1.0
    s = np.sum(tensor)
    return tensor/s if s > 0 else np.ones(5)*0.2

def locus_features(root_slp1):
    c = root_slp1[0].lower() if root_slp1 else ''
    velar, palatal, retro, dental, labial = ['k','K','g','G','N'], ['c','C','j','J','Y'], ['w','W','q','Q','R'], ['t','T','d','D','n'], ['p','P','b','B','m']
    features = np.zeros(5)
    if c in velar: features[0] = 1.0
    elif c in palatal: features[1] = 1.0
    elif c in retro: features[2] = 1.0
    elif c in dental: features[3] = 1.0
    elif c in labial: features[4] = 1.0
    return features

def prosodic_features(root_slp1):
    chars = list(root_slp1)
    n_syllables = n_guru = n_laghu = 0
    i = 0
    while i < len(chars):
        c = chars[i]
        if c in LONG_VOWELS or c in SHORT_VOWELS:
            is_guru = c in LONG_VOWELS
            j, cluster = i + 1, 0
            while j < len(chars) and chars[j] in CONSONANTS: cluster += 1; j += 1
            if cluster > 1: is_guru = True
            n_syllables += 1
            if is_guru: n_guru += 1
            else: n_laghu += 1
            i = j
        else: i += 1
    total = max(n_guru + n_laghu, 1)
    return np.array([n_syllables, n_guru/total, n_laghu/total, float(n_guru-n_laghu)])

def acoustic_features(root_slp1):
    return np.concatenate([locus_features(root_slp1), prosodic_features(root_slp1)])

class AdExPopulation(nn.Module):
    def __init__(self, size, dt=1.0, C=200.0, gL=10.0, EL=-70.0, VT=-50.0, DeltaT=2.0, Vpeak=0.0, Vreset=-58.0, a=2.0, b=80.0, tau_w=30.0):
        super().__init__()
        self.size = size; self.dt = dt; self.C = C; self.gL = gL; self.EL = EL; self.DeltaT = DeltaT
        self.register_buffer('Vreset', torch.tensor([[Vreset]])); self.register_buffer('Vpeak', torch.tensor([[Vpeak]]))
        self.VT = nn.Parameter(torch.ones(1, size) * VT); self.a = nn.Parameter(torch.ones(1, size) * a); self.b = nn.Parameter(torch.ones(1, size) * b); self.tau_w = nn.Parameter(torch.ones(1, size) * tau_w)
        self.register_buffer('V', torch.ones(1, size) * EL); self.register_buffer('w', torch.zeros(1, size)); self.register_buffer('theta', torch.ones(1, size) * 0.1); self.register_buffer('spike_counts', torch.zeros(1, size))
    def reset_states(self): self.V.fill_(self.EL); self.w.zero_(); self.theta.fill_(0.1); self.spike_counts.zero_()
    def step(self, I_ext):
        exp_term = self.gL * self.DeltaT * torch.exp(torch.clamp((self.V - self.VT) / self.DeltaT, max=20.0))
        dV = (-self.gL*(self.V-self.EL) + exp_term - self.w + I_ext) / self.C
        self.V = self.V + self.dt * dV
        dw = (self.a*(self.V-self.EL) - self.w) / self.tau_w
        self.w = self.w + self.dt * dw
        spikes = (self.V >= self.Vpeak)
        self.spike_counts = self.spike_counts + spikes.float()
        self.V = torch.where(spikes, self.Vreset.expand_as(self.V), self.V); self.w = torch.where(spikes, self.w + self.b, self.w)
        self.theta = self.theta + 0.005 * (spikes.float() - self.theta)
        return spikes

class MegaSNN(nn.Module):
    def __init__(self, in_dim, n_neurons=512):
        super().__init__()
        self.L1 = AdExPopulation(size=n_neurons, a=0.0, b=0.0, tau_w=5.0, gL=15.0)
        mask1 = (torch.rand(in_dim, n_neurons) < 0.18).float()
        self.proj_in = nn.Parameter(torch.randn(in_dim, n_neurons) * 1100.0 * mask1)
        self.L2 = AdExPopulation(size=256, a=2.0, b=80.0, tau_w=150.0, gL=2.0)
        mask2 = (torch.rand(n_neurons, 256) < 0.18).float()
        self.W12 = nn.Parameter(torch.randn(n_neurons, 256) * 410.0 * mask2)
    def reset(self): self.L1.reset_states(); self.L2.reset_states()
    def step(self, I_in):
        I1 = I_in @ self.proj_in; sp1 = self.L1.step(I1); I2 = sp1.float() @ self.W12
        return self.L2.step(I2)

def run_experiment(seed, inputs, labels, device, n_neurons=512):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    model = MegaSNN(inputs.shape[1], n_neurons=n_neurons).to(device)
    all_states = []
    for row in inputs:
        model.reset()
        I = torch.tensor(row, dtype=torch.float32).unsqueeze(0).to(device)
        for _ in range(20): model.step(I)
        silence = torch.zeros(1, inputs.shape[1], device=device)
        for _ in range(20): model.step(silence)
        all_states.append(model.L2.spike_counts.cpu().numpy().squeeze())
    states_norm = StandardScaler().fit_transform(np.array(all_states))
    pred = KMeans(n_clusters=5, n_init=10, random_state=seed).fit_predict(states_norm)
    return adjusted_rand_score(labels, pred), np.mean(all_states)

print("="*70); print("Exp 50 -- Piᅁgala-Only Baseline (NO Prior)"); print("="*70)
with open('/content/data.txt', encoding='utf-8') as f: full_corpus = json.load(f)['data']
inputs, labels = [], []
for item in full_corpus[:2000]:
    root_slp1, artha_slp1 = devanagari_to_slp1(item['dhatu']), devanagari_to_slp1(item['artha'])
    a_tensor = extract_artha_stem_tensor(artha_slp1)
    label = int(np.argmax(a_tensor)) if np.max(a_tensor) > 0.2 else -1
    if label != -1: inputs.append(np.concatenate([acoustic_features(root_slp1), np.array(compute_pingala_address(root_slp1))])); labels.append(label)
inputs, labels = np.array(inputs), np.array(labels)
print(f"Loaded {len(labels)} roots, Dim: {inputs.shape[1]}")

results = []
for seed in [42, 43, 44, 45, 46]:
    ari, firing = run_experiment(seed, inputs, labels, device)
    results.append(ari); print(f"  seed={seed}: ARI={ari:.4f}  firing={firing:.3f}")

mean_ari = np.mean(results)
print("\n" + "="*70); print(f"Mean ARI: {mean_ari:.4f} ± {np.std(results):.4f}")
if mean_ari > 0.15: print("  *** PASS: Pure phonological signal achieves ARI > 0.15 ***")
else: print("  *** FAIL: ARI < 0.15 threshold ***")
print("="*70)