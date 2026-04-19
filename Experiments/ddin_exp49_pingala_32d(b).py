"""
Exp 49: Piṅgala 32D Tensor Integration
Tests whether appending Piṅgala prosodic address (4D) to acoustic+prior improves ARI.
GPU script - run on Colab (Runtime > Change runtime type > T4 GPU).
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

# ── Character sets ─────────────────────────────────────────────────────────────
LONG_VOWELS  = set('AEIOUfF')
SHORT_VOWELS = set('aiux')
CONSONANTS   = set('kKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzsh')

# ── Transliteration ────────────────────────────────────────────────────────────
def devanagari_to_slp1(text):
    vowels = {
        'अ':'a','आ':'A','इ':'i','ई':'I','उ':'u','ऊ':'U',
        'ऋ':'f','ॠ':'F','ऌ':'x','ॡ':'X',
        'ए':'e','ऐ':'E','ओ':'o','औ':'O'
    }
    vowel_signs = {
        'ा':'A','ि':'i','ी':'I','ु':'u','ू':'U','ृ':'f','ॄ':'F',
        'ॢ':'x','ॣ':'X','े':'e','ै':'E','ो':'o','ौ':'O'
    }
    consonants = {
        'क':'k','ख':'K','ग':'g','घ':'G','ङ':'N',
        'च':'c','छ':'C','ज':'j','झ':'J','ञ':'Y',
        'ट':'w','ठ':'W','ड':'q','ढ':'Q','ण':'R',
        'त':'t','थ':'T','द':'d','ध':'D','न':'n',
        'प':'p','फ':'P','ब':'b','भ':'B','म':'m',
        'य':'y','र':'r','ल':'l','व':'v',
        'श':'S','ष':'z','स':'s','ह':'h'
    }
    misc = {'ं':'M','ः':'H','ँ':'~','्':''}
    res = ""
    for i, char in enumerate(text):
        if char in vowels:
            res += vowels[char]
        elif char in consonants:
            res += consonants[char]
            if i + 1 < len(text) and (text[i+1] in vowel_signs or text[i+1] == '्'):
                pass
            else:
                res += 'a'
        elif char in vowel_signs:
            res += vowel_signs[char]
        elif char in misc:
            res += misc[char]
        else:
            res += char
    return res

# ── Piṅgala prosodic address ───────────────────────────────────────────────────
def compute_pingala_address(root_slp1, max_syllables=4):
    chars = list(root_slp1)
    syllables = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c in LONG_VOWELS or c in SHORT_VOWELS:
            is_guru = c in LONG_VOWELS
            j, cluster = i + 1, 0
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

# ── Semantic axis tensor ───────────────────────────────────────────────────────
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
        if any(s in artha_slp1 for s in stems):
            tensor[i] = 1.0
    s = np.sum(tensor)
    return tensor / s if s > 0 else np.ones(5) * 0.2

# ── Acoustic features ─────────────────────────────────────────────────────────
def locus_features(root_slp1):
    c = root_slp1[0].lower() if root_slp1 else ''
    velar   = ['k','K','g','G','N']
    palatal = ['c','C','j','J','Y']
    retro   = ['w','W','q','Q','R']
    dental  = ['t','T','d','D','n']
    labial  = ['p','P','b','B','m']
    features = np.zeros(5)
    if   c in velar:   features[0] = 1.0
    elif c in palatal: features[1] = 1.0
    elif c in retro:   features[2] = 1.0
    elif c in dental:  features[3] = 1.0
    elif c in labial:  features[4] = 1.0
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
            while j < len(chars) and chars[j] in CONSONANTS:
                cluster += 1
                j += 1
            if cluster > 1:
                is_guru = True
            n_syllables += 1
            if is_guru: n_guru += 1
            else:       n_laghu += 1
            i = j
        else:
            i += 1
    total = max(n_guru + n_laghu, 1)
    return np.array([n_syllables, n_guru/total, n_laghu/total, float(n_guru - n_laghu)])

def acoustic_features(root_slp1):
    return np.concatenate([locus_features(root_slp1), prosodic_features(root_slp1)])

# ── AdEx population ────────────────────────────────────────────────────────────
class AdExPopulation(nn.Module):
    def __init__(self, size, dt=1.0, C=200.0, gL=10.0, EL=-70.0, VT=-50.0,
                 DeltaT=2.0, Vpeak=0.0, Vreset=-58.0, a=2.0, b=80.0, tau_w=30.0):
        super().__init__()
        self.size   = size
        self.dt     = dt
        self.C      = C
        self.gL     = gL
        self.EL     = EL
        self.DeltaT = DeltaT
        self.register_buffer('Vpeak',  torch.tensor([[Vpeak]]))
        self.register_buffer('Vreset', torch.tensor([[Vreset]]))
        self.VT    = nn.Parameter(torch.ones(1, size) * VT)
        self.a     = nn.Parameter(torch.ones(1, size) * a)
        self.b     = nn.Parameter(torch.ones(1, size) * b)
        self.tau_w = nn.Parameter(torch.ones(1, size) * tau_w)
        self.register_buffer('V',            torch.ones(1, size) * EL)
        self.register_buffer('w',            torch.zeros(1, size))
        self.register_buffer('theta',        torch.ones(1, size) * 0.1)
        self.register_buffer('spike_counts', torch.zeros(1, size))

    def reset_states(self):
        self.V.fill_(self.EL)
        self.w.zero_()
        self.theta.fill_(0.1)
        self.spike_counts.zero_()

    def step(self, I_ext):
        exp_term = self.gL * self.DeltaT * torch.exp(
            torch.clamp((self.V - self.VT) / self.DeltaT, max=20.0)
        )
        dV = (-self.gL * (self.V - self.EL) + exp_term - self.w + I_ext) / self.C
        self.V = self.V + self.dt * dV
        dw = (self.a * (self.V - self.EL) - self.w) / self.tau_w
        self.w = self.w + self.dt * dw
        spikes = (self.V >= self.Vpeak)
        self.spike_counts = self.spike_counts + spikes.float()
        self.V     = torch.where(spikes, self.Vreset.expand_as(self.V), self.V)
        self.w     = torch.where(spikes, self.w + self.b, self.w)
        self.theta = self.theta + 0.005 * (spikes.float() - self.theta)
        return spikes

# ── Two-layer SNN (processes one root at a time, like Phase 11A) ──────────────
class MegaSNN(nn.Module):
    def __init__(self, in_dim, n_neurons=512):
        super().__init__()
        self.L1 = AdExPopulation(size=n_neurons, a=0.0, b=0.0,  tau_w=5.0,   gL=15.0)
        mask1   = (torch.rand(in_dim, n_neurons) < 0.18).float()
        self.proj_in = nn.Parameter(torch.randn(in_dim, n_neurons) * 1100.0 * mask1)

        self.L2 = AdExPopulation(size=256,      a=2.0, b=80.0, tau_w=150.0, gL=2.0)
        mask2   = (torch.rand(n_neurons, 256)   < 0.18).float()
        self.W12 = nn.Parameter(torch.randn(n_neurons, 256) * 410.0 * mask2)

    def reset(self):
        self.L1.reset_states()
        self.L2.reset_states()

    def step(self, I_input):
        I1  = I_input @ self.proj_in
        sp1 = self.L1.step(I1)
        I2  = sp1.float() @ self.W12
        return self.L2.step(I2)

# ── Corpus loader ─────────────────────────────────────────────────────────────
def load_corpus(path, n_roots=2000):
    with open(path, encoding='utf-8') as f:
        full_corpus = json.load(f)['data']
    corpus = full_corpus[:n_roots]
    rows, labels = [], []
    for item in corpus:
        root_slp1  = devanagari_to_slp1(item['dhatu'])
        artha_slp1 = devanagari_to_slp1(item['artha'])
        a_tensor   = extract_artha_stem_tensor(artha_slp1)
        label      = int(np.argmax(a_tensor)) if np.max(a_tensor) > 0.2 else -1
        if label == -1:
            continue
        acoustic = acoustic_features(root_slp1)            # 9D
        pingala  = compute_pingala_address(root_slp1, 4)   # 4D
        # 9D + 5D + 4D = 18D
        rows.append(np.concatenate([acoustic, a_tensor, pingala], dtype=np.float32))
        labels.append(label)
    return rows, np.array(labels)

# ── Per-root sequential processing (matches Phase 11A methodology) ────────────
def process_corpus(model, rows, labels, device):
    all_states = []
    for i, row in enumerate(rows):
        if i % 200 == 0:
            print(f"    processed {i}/{len(rows)}...")
        model.reset()
        with torch.no_grad():
            # Present the full 18D input vector for 20 steps
            I = torch.tensor(row, dtype=torch.float32).unsqueeze(0).to(device)
            for _ in range(20):
                model.step(I)
            # Silence / decay window
            silence = torch.zeros(1, len(row), dtype=torch.float32, device=device)
            for _ in range(20):
                model.step(silence)
        all_states.append(model.L2.spike_counts.cpu().numpy().squeeze())
    return np.array(all_states)

# ── Per-seed experiment ────────────────────────────────────────────────────────
def run_experiment(seed, rows, labels, device, n_neurons=512):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    in_dim = len(rows[0])
    model  = MegaSNN(in_dim, n_neurons=n_neurons).to(device)

    states      = process_corpus(model, rows, labels, device)
    states_norm = StandardScaler().fit_transform(states)
    pred        = KMeans(n_clusters=5, n_init=10, random_state=seed).fit_predict(states_norm)
    ari         = adjusted_rand_score(labels, pred)
    firing      = states.mean()

    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return ari, firing

# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Exp 49 -- Piṅgala 18D Tensor Integration")
print("=" * 70)
print("Input: 9D acoustic + 5D prior + 4D Piṅgala = 18D")
print("Corpus: 2000 roots | Processing: per-root sequential (matches Phase 11A)")
print()

# Auto-detect path for Colab vs Kaggle
import os
CORPUS_PATH = (
    '/content/data.txt'
    if os.path.exists('/content/data.txt')
    else '/kaggle/input/datasets/akkmit/dhatu-data/data.txt'
)
print(f"Corpus path: {CORPUS_PATH}")

rows, labels = load_corpus(CORPUS_PATH, n_roots=2000)
print(f"Loaded {len(labels)} labeled roots")
print(f"Input dim: {len(rows[0])}")
print()

SEEDS   = [42, 43, 44, 45, 46]
results = []

for seed in SEEDS:
    print(f"Seed {SEEDS.index(seed)+1}/{len(SEEDS)} (seed={seed})...")
    ari, firing = run_experiment(seed, rows, labels, device, n_neurons=512)
    results.append(ari)
    print(f"  ARI={ari:.4f}  firing={firing:.3f}")

mean_ari = np.mean(results)
std_ari  = np.std(results)

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"  Mean ARI : {mean_ari:.4f} ± {std_ari:.4f}")
print(f"  Min      : {np.min(results):.4f}  Max: {np.max(results):.4f}")
print()
print("Comparison (baseline from Phase 11A):")
print(f"  Baseline (no Piṅgala) : 0.9555 ± 0.0359")
print(f"  Piṅgala 18D           : {mean_ari:.4f} ± {std_ari:.4f}")
delta = mean_ari - 0.9555
print(f"  Delta                 : {delta:+.4f}")
if delta > 0.01:
    print("  --> Piṅgala prosodic address IMPROVES clustering")
elif delta < -0.01:
    print("  --> Piṅgala adds noise; consider weighting or ablating")
else:
    print("  --> Piṅgala effect neutral within noise floor")
print("=" * 70)