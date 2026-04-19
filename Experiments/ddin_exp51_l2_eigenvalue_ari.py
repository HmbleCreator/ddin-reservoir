"""
L2-C2: Eigenvalue-ARI Correlation
Tests whether eigenvalue entropy is causally predictive of ARI quality.
Uses existing 5-seed GPU sweep data. Run on GPU.
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
from scipy.stats import spearmanr, pearsonr

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

def compute_spectral_entropy(eigenvalues):
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) == 0: return 0.0
    eigenvalues = eigenvalues / eigenvalues.sum()
    return -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))

def compute_eigenvalue_metrics(states):
    cov = np.cov(states.T)
    eigenvals = np.linalg.eigvalsh(cov)
    eigenvals = np.abs(eigenvals)
    eigenvals = eigenvals[eigenvals > 1e-10]
    if len(eigenvals) == 0: return 0.0, 0.0, 0.0, 0.0
    eigenvals_sorted = np.sort(eigenvals)[::-1]
    total = eigenvals_sorted.sum()
    top10_mass = eigenvals_sorted[:10].sum() / total if total > 0 else 0.0
    top50_mass = eigenvals_sorted[:50].sum() / total if total > 0 else 0.0
    spectral_ent = compute_spectral_entropy(eigenvals_sorted / total)
    lambda_ratio = eigenvals_sorted[0] / eigenvals_sorted[49] if len(eigenvals_sorted) > 49 else float('inf')
    return spectral_ent, top10_mass, top50_mass, lambda_ratio

def run_experiment(seed, inputs, labels, device, n_neurons=512):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    model = MegaSNN(inputs.shape[1], n_neurons=n_neurons).to(device)
    all_states = []
    for i, row in enumerate(inputs):
        if i % 200 == 0: print(f"    processed {i}/{len(inputs)}...")
        model.reset()
        I = torch.tensor(row, dtype=torch.float32).unsqueeze(0).to(device)
        for _ in range(20): model.step(I)
        silence = torch.zeros(1, inputs.shape[1], device=device)
        for _ in range(20): model.step(silence)
        all_states.append(model.L2.spike_counts.cpu().numpy().squeeze())
    states = np.array(all_states)
    states_norm = StandardScaler().fit_transform(states)
    pred = KMeans(n_clusters=5, n_init=10, random_state=seed).fit_predict(states_norm)
    ari = adjusted_rand_score(labels, pred)
    spectral_ent, top10, top50, lambda_ratio = compute_eigenvalue_metrics(states)
    del model; gc.collect(); torch.cuda.empty_cache()
    return ari, spectral_ent, top10, top50, lambda_ratio

print("="*70)
print("L2-C2: Eigenvalue-ARI Correlation")
print("="*70)

with open('/content/data.txt', encoding='utf-8') as f: full_corpus = json.load(f)['data']
inputs, labels = [], []
for item in full_corpus[:2000]:
    root_slp1, artha_slp1 = devanagari_to_slp1(item['dhatu']), devanagari_to_slp1(item['artha'])
    a_tensor = extract_artha_stem_tensor(artha_slp1)
    label = int(np.argmax(a_tensor)) if np.max(a_tensor) > 0.2 else -1
    if label != -1: inputs.append(np.concatenate([acoustic_features(root_slp1), a_tensor])); labels.append(label)
inputs, labels = np.array(inputs), np.array(labels)
print(f"Corpus: {len(labels)} roots, Input dim: {inputs.shape[1]}")

SEEDS = [42, 43, 44, 45, 46]
results = []

for seed in SEEDS:
    print(f"\nSeed {SEEDS.index(seed)+1}/{len(SEEDS)} (seed={seed})...")
    ari, spectral_ent, top10, top50, lambda_ratio = run_experiment(seed, inputs, labels, device)
    results.append({'seed': seed, 'ari': ari, 'spectral_entropy': spectral_ent, 'top10_mass': top10, 'top50_mass': top50})
    print(f"  ARI={ari:.4f}  entropy={spectral_ent:.4f}  top10={top10:.4f}")

print()
print("="*70)
print("CORRELATION ANALYSIS")
print("="*70)

ari_vals = [r['ari'] for r in results]
ent_vals = [r['spectral_entropy'] for r in results]
top10_vals = [r['top10_mass'] for r in results]

pearson_r, pearson_p = pearsonr(ari_vals, ent_vals)
spearman_r, spearman_p = spearmanr(ari_vals, ent_vals)

print(f"\n  Spectral Entropy vs ARI:")
print(f"    Pearson r  = {pearson_r:.4f} (p={pearson_p:.4f})")
print(f"    Spearman r = {spearman_r:.4f} (p={spearman_p:.4f})")

pearson_r_top10, pearson_p_top10 = pearsonr(ari_vals, top10_vals)
print(f"\n  Top-10 Mass vs ARI:")
print(f"    Pearson r  = {pearson_r_top10:.4f} (p={pearson_p_top10:.4f})")

print()
print("="*70)
print("VERDICT")
print("="*70)
if abs(pearson_r) > 0.5:
    print(f"  CAUSAL: eigenvalue entropy is predictive of ARI (|r|={abs(pearson_r):.4f})")
    print("  --> Near-critical state is the mechanistic cause of attractor fusion")
elif abs(pearson_r) < 0.2:
    print(f"  EPIPHENOMENAL: eigenvalue entropy is NOT predictive (|r|={abs(pearson_r):.4f})")
    print("  --> Eigenvalue distribution is a consequence, not a cause")
else:
    print(f"  INCONCLUSIVE: moderate correlation (|r|={abs(pearson_r):.4f})")
    print("  --> More data needed")
print("="*70)