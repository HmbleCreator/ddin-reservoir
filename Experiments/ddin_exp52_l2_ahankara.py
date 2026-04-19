"""
L2-A1: Ahankara Suspension - Zero-Prior Inference
The definitive test: does the network learn physics or memorize priors?
Train on 80%, test on 20% held-out roots with trained vs uniform theta.
GPU script - run on Kaggle/Colab.
"""
import warnings
warnings.filterwarnings('ignore')
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
    return tensor/s if s>0 else np.ones(5)*0.2

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

def run_experiment_trained(seed, inputs, labels, device, n_neurons=512):
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
    states = np.array(all_states)
    states_norm = StandardScaler().fit_transform(states)
    pred = KMeans(n_clusters=5, n_init=10, random_state=seed).fit_predict(states_norm)
    del model; gc.collect(); torch.cuda.empty_cache()
    return adjusted_rand_score(labels, pred)

def run_experiment_uniform(seed, inputs, labels, device, n_neurons=512):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    model = MegaSNN(inputs.shape[1], n_neurons=n_neurons).to(device)
    model.L2.a.data.fill_(0.0)
    model.L2.b.data.fill_(0.0)
    model.L2.tau_w.data.fill_(1.0)
    all_states = []
    for row in inputs:
        model.reset()
        I = torch.tensor(row, dtype=torch.float32).unsqueeze(0).to(device)
        for _ in range(20): model.step(I)
        silence = torch.zeros(1, inputs.shape[1], device=device)
        for _ in range(20): model.step(silence)
        all_states.append(model.L2.spike_counts.cpu().numpy().squeeze())
    states = np.array(all_states)
    states_norm = StandardScaler().fit_transform(states)
    pred = KMeans(n_clusters=5, n_init=10, random_state=seed).fit_predict(states_norm)
    del model; gc.collect(); torch.cuda.empty_cache()
    return adjusted_rand_score(labels, pred)

print("="*70)
print("L2-A1: Ahankara Suspension - Zero-Prior Inference")
print("="*70)
print("Train on 80%, test on 20% held-out roots")
print("Compare: trained theta vs uniform theta (ahankara suspended)")
print()

with open('/content/data.txt', encoding='utf-8') as f: full_corpus = json.load(f)['data']
inputs, labels = [], []
for item in full_corpus[:2000]:
    root_slp1, artha_slp1 = devanagari_to_slp1(item['dhatu']), devanagari_to_slp1(item['artha'])
    a_tensor = extract_artha_stem_tensor(artha_slp1)
    label = int(np.argmax(a_tensor)) if np.max(a_tensor) > 0.2 else -1
    if label != -1: inputs.append(np.concatenate([acoustic_features(root_slp1), a_tensor])); labels.append(label)
inputs, labels = np.array(inputs), np.array(labels)
print(f"Corpus: {len(labels)} roots, Input dim: {inputs.shape[1]}")

n_total = len(labels)
n_train = int(0.8 * n_total)
n_test = n_total - n_train

indices = np.random.permutation(n_total)
train_idx = indices[:n_train]
test_idx = indices[n_train:]

inputs_train, labels_train = inputs[train_idx], labels[train_idx]
inputs_test, labels_test = inputs[test_idx], labels[test_idx]

print(f"Train: {len(labels_train)} roots, Test: {len(labels_test)} roots")

SEEDS = [42, 43, 44]
results = []

for seed in SEEDS:
    print(f"\nSeed {seed}...")
    ari_train = run_experiment_trained(seed, inputs_train, labels_train, device)
    ari_test_trained = run_experiment_trained(seed, inputs_test, labels_test, device)
    ari_test_uniform = run_experiment_uniform(seed, inputs_test, labels_test, device)
    print(f"  Train ARI (trained):     {ari_train:.4f}")
    print(f"  Test ARI (trained):      {ari_test_trained:.4f}")
    print(f"  Test ARI (uniform):     {ari_test_uniform:.4f}")
    results.append({
        'seed': seed,
        'ari_train': ari_train,
        'ari_test_trained': ari_test_trained,
        'ari_test_uniform': ari_test_uniform
    })

print()
print("="*70)
print("SUMMARY")
print("="*70)
print(f"\n  {'Seed':<8} {'Train':<10} {'Test(trained)':<15} {'Test(uniform)':<15} {'Drop':<10}")
print("  " + "-"*58)
for r in results:
    drop = r['ari_test_trained'] - r['ari_test_uniform']
    print(f"  {r['seed']:<8} {r['ari_train']:<10.4f} {r['ari_test_trained']:<15.4f} {r['ari_test_uniform']:<15.4f} {drop:<10.4f}")

mean_train = np.mean([r['ari_train'] for r in results])
mean_trained = np.mean([r['ari_test_trained'] for r in results])
mean_uniform = np.mean([r['ari_test_uniform'] for r in results])
mean_drop = mean_trained - mean_uniform

print("  " + "-"*58)
print(f"  {'MEAN':<8} {mean_train:<10.4f} {mean_trained:<15.4f} {mean_uniform:<15.4f} {mean_drop:<10.4f}")

print()
print("="*70)
print("VERDICT")
print("="*70)
print(f"\n  Baseline (chance): 0.02")
print(f"  Acoustic only:      0.0858")
print(f"  Trained theta:      {mean_trained:.4f}")
print(f"  Uniform theta:      {mean_uniform:.4f}")
print()

if mean_uniform > 0.05:
    print("  *** PHYSICS LEARNED: Uniform theta achieves ARI >> chance ***")
    print("  --> Network learned the physics of meaning, not just priors")
elif mean_uniform > 0.02:
    print("  *** PARTIAL: Uniform theta above chance but reduced ***")
    print("  --> Network learned some physics, but priors contribute")
else:
    print("  *** PRIORS REQUIRED: Uniform theta at chance ***")
    print("  --> Network memorized priors, not physics")
print("="*70)