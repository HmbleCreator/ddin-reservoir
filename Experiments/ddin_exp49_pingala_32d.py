"""
Exp 49: Piᅁgala 32D Tensor Integration
Tests whether appending Piᅁgala prosodic address (4D) to acoustic+prior improves ARI.
GPU script - run on Kaggle/Colab.
"""
import torch
import torch.nn as nn
import numpy as np
import json
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
            if cluster > 1:
                is_guru = True
            syllables.append(1 if is_guru else 0)
            i = j
        else:
            i += 1
    addr = syllables[:max_syllables]
    while len(addr) < max_syllables:
        addr.append(0)
    return addr

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

def locus_features(root_slp1):
    c = root_slp1[0].lower() if root_slp1 else ''
    velar   = ['k','K','g','G','N']
    palatal = ['c','C','j','J','Y']
    retro   = ['w','W','q','Q','R']
    dental  = ['t','T','d','D','n']
    labial  = ['p','P','b','B','m']
    features = np.zeros(5)
    if c in velar:   features[0] = 1.0
    elif c in palatal: features[1] = 1.0
    elif c in retro:  features[2] = 1.0
    elif c in dental: features[3] = 1.0
    elif c in labial: features[4] = 1.0
    return features

def prosodic_features(root_slp1):
    chars = list(root_slp1)
    n_syllables = 0
    n_guru = 0
    n_laghu = 0
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
            if cluster > 1:
                is_guru = True
            n_syllables += 1
            if is_guru:
                n_guru += 1
            else:
                n_laghu += 1
            i = j
        else:
            i += 1
    total = max(n_guru + n_laghu, 1)
    return np.array([n_syllables, n_guru/total, n_laghu/total, n_guru-n_laghu])

def acoustic_features(root_slp1):
    locus_f = locus_features(root_slp1)
    pros_f = prosodic_features(root_slp1)
    return np.concatenate([locus_f, pros_f])

class AdExPopulation(nn.Module):
    def __init__(self, size, dt=1.0, C=200.0, gL=10.0, EL=-70.0, VT=-50.0,
                 DeltaT=2.0, Vpeak=0.0, Vreset=-58.0, a=2.0, b=80.0, tau_w=30.0):
        super().__init__()
        self.size = size; self.dt = dt
        self.C = C; self.gL = gL; self.EL = EL
        self.DeltaT = DeltaT
        self.register_buffer('Vreset', torch.tensor([[Vreset]]))
        self.register_buffer('Vpeak', torch.tensor([[Vpeak]]))
        self.VT   = nn.Parameter(torch.ones(1, size) * VT)
        self.a    = nn.Parameter(torch.ones(1, size) * a)
        self.b    = nn.Parameter(torch.ones(1, size) * b)
        self.tau_w= nn.Parameter(torch.ones(1, size) * tau_w)
        self.V = None
        self.w = None
        self.theta = None
        self.spike_counts = None

    def reset_states(self, batch_size):
        self.V = torch.ones(batch_size, self.size, device=self.VT.device) * self.EL
        self.w = torch.zeros(batch_size, self.size, device=self.VT.device)
        self.theta = torch.ones(batch_size, self.size, device=self.VT.device) * 0.1
        self.spike_counts = torch.zeros(batch_size, self.size, device=self.VT.device)

    def step(self, I_ext):
        exp_term = self.gL * self.DeltaT * torch.exp(torch.clamp((self.V - self.VT) / self.DeltaT, max=20.0))
        dV = (-self.gL*(self.V-self.EL) + exp_term - self.w + I_ext) / self.C
        self.V = self.V + self.dt * dV
        dw = (self.a*(self.V-self.EL) - self.w) / self.tau_w
        self.w = self.w + self.dt * dw
        spikes = (self.V >= self.Vpeak)
        self.spike_counts = self.spike_counts + spikes.float()
        self.V = torch.where(spikes, self.Vreset.expand_as(self.V), self.V)
        self.w = torch.where(spikes, self.w + self.b, self.w)
        self.theta = self.theta + 0.005 * (spikes.float() - self.theta)
        return spikes

class MegaSNN(nn.Module):
    def __init__(self, in_dim, n_neurons=512, dt=1.0):
        super().__init__()
        self.dt = dt
        self.reservoir = AdExPopulation(n_neurons, dt=dt)
        self.alpha = nn.Parameter(torch.ones(n_neurons) * 0.3)
        self.beta  = nn.Parameter(torch.ones(n_neurons) * 0.1)
        self.W_in  = nn.Parameter(torch.randn(in_dim, n_neurons) * 0.1)
        self.W     = nn.Parameter(torch.randn(n_neurons, n_neurons) * 0.06)
        self.n_neurons = n_neurons

    def forward(self, x, T=30):
        x = x.to(device)
        batch_size = x.shape[0]
        self.reservoir.reset_states(batch_size)
        for t in range(T):
            I = torch.matmul(x, self.W_in) + torch.matmul(self.reservoir.V, self.W)
            spikes = self.reservoir.step(I)
            self.reservoir.V = self.alpha * self.reservoir.V + spikes.float() * self.beta
        return self.reservoir.spike_counts

def load_corpus(n_roots=2000):
    with open(r'/content/data.txt', encoding='utf-8') as f:
        full_corpus = json.load(f)['data']
    corpus = full_corpus[:n_roots]
    inputs, labels = [], []
    for item in corpus:
        root_slp1  = devanagari_to_slp1(item['dhatu'])
        artha_slp1 = devanagari_to_slp1(item['artha'])
        a_tensor   = extract_artha_stem_tensor(artha_slp1)
        label      = int(np.argmax(a_tensor)) if np.max(a_tensor) > 0.2 else -1
        if label == -1: continue
        acoustic   = acoustic_features(root_slp1)
        pingala     = compute_pingala_address(root_slp1, max_syllables=4)
        full_input  = np.concatenate([acoustic, a_tensor, pingala])
        inputs.append(full_input)
        labels.append(label)
    return np.array(inputs), np.array(labels)

def run_experiment(seed, inputs, labels, n_neurons=512, T=30):
    torch.manual_seed(seed)
    np.random.seed(seed)
    in_dim = inputs.shape[1]
    model = MegaSNN(in_dim, n_neurons=n_neurons).to(device)
    X = torch.from_numpy(inputs).float()
    Y = torch.from_numpy(labels).long()
    with torch.no_grad():
        states = model(X, T=T)
    preds = states.argmax(dim=1).cpu().numpy()
    ari = adjusted_rand_score(Y.numpy(), preds)
    valid = np.sum(preds == Y.numpy())
    firing = states.mean().item()
    return ari, valid, len(Y), firing

print("="*70)
print("Exp 49 -- Piᅁgala 32D Tensor Integration")
print("="*70)
print("Input: 9D acoustic + 5D prior + 4D Piᅁgala = 18D")
print("Corpus: 2000 roots")
print()

inputs, labels = load_corpus(2000)
print(f"Loaded {len(labels)} labeled roots")
print(f"Input dim: {inputs.shape[1]}")

SEEDS = [42, 43, 44, 45, 46]
results = []

for seed in SEEDS:
    ari, valid, total, firing = run_experiment(seed, inputs, labels, n_neurons=512, T=30)
    results.append(ari)
    print(f"  seed={seed}: ARI={ari:.4f}  valid={valid}/{total}  firing={firing:.3f}")

mean_ari = np.mean(results)
std_ari = np.std(results)
print()
print("="*70)
print("RESULTS")
print("="*70)
print(f"  Mean ARI: {mean_ari:.4f} ± {std_ari:.4f}")
print(f"  Min: {np.min(results):.4f}  Max: {np.max(results):.4f}")
print()
print("Comparison (baseline from Phase 11A):")
print(f"  Baseline (no Piᅁgala): 0.9555 ± 0.0359")
print(f"  Piᅁgala 32D:          {mean_ari:.4f} ± {std_ari:.4f}")
delta = mean_ari - 0.9555
print(f"  Delta: {delta:+.4f}")
print("="*70)