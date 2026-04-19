
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import numpy as np
import json
import gc
import os
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Pāṇinian Locus Taxonomy ───────────────────────────────────────────────────
# 5 classes based on initial consonant place of articulation
LOCUS_MAP = {
    'k': 0, 'K': 0, 'g': 0, 'G': 0, 'N': 0, # Kaṇṭhya (Velar)
    'c': 1, 'C': 1, 'j': 1, 'J': 1, 'Y': 1, 'y': 1, 'S': 1, # Tālavya (Palatal)
    'w': 2, 'W': 2, 'q': 2, 'Q': 2, 'R': 2, 'r': 2, 'z': 2, # Mūrdhanya (Retroflex)
    't': 3, 'T': 3, 'd': 3, 'D': 3, 'n': 3, 'l': 3, 's': 3, # Dantya (Dental)
    'p': 4, 'P': 4, 'b': 4, 'B': 4, 'm': 4, 'v': 4, # Oṣṭhya (Labial)
}

def get_locus_label(root_slp1):
    if not root_slp1: return -1
    char = root_slp1[0]
    return LOCUS_MAP.get(char, -1)

# ── Unit Tests for Locus Labeling ─────────────────────────────────────────────
def test_locus_labeling():
    assert get_locus_label('gam') == 0 # Velar
    assert get_locus_label('cal') == 1 # Palatal
    assert get_locus_label('qI') == 2  # Retroflex
    assert get_locus_label('tud') == 3 # Dental
    assert get_locus_label('pac') == 4 # Labial
    assert get_locus_label('aD') == -1 # Vowel start
    print("Unit tests passed: Locus Labeling is correct.")

# ── Acoustic Features ──────────────────────────────────────────────────────────
def formant_features(root_slp1):
    PHONEME_FORMANTS = {
        'a': (0.8, 0.3), 'A': (0.9, 0.2), 'i': (0.2, 0.8), 'I': (0.1, 0.9),
        'u': (0.3, 0.7), 'U': (0.2, 0.8), 'f': (0.4, 0.4), 'F': (0.3, 0.3),
        'k': (0.1, 0.1), 'K': (0.1, 0.2), 'g': (0.2, 0.1), 'G': (0.2, 0.2), 'N': (0.1, 0.3),
        'c': (0.3, 0.6), 'C': (0.3, 0.7), 'j': (0.4, 0.6), 'J': (0.4, 0.7), 'Y': (0.3, 0.5),
        'w': (0.4, 0.2), 'W': (0.4, 0.3), 'q': (0.5, 0.2), 'Q': (0.5, 0.3), 'R': (0.4, 0.4),
        't': (0.5, 0.5), 'T': (0.5, 0.6), 'd': (0.6, 0.5), 'D': (0.6, 0.6), 'n': (0.5, 0.4),
        'p': (0.3, 0.8), 'P': (0.3, 0.9), 'b': (0.4, 0.8), 'B': (0.4, 0.9), 'm': (0.4, 0.7),
        'y': (0.2, 0.6), 'r': (0.3, 0.5), 'l': (0.5, 0.5), 'v': (0.3, 0.7),
        'S': (0.4, 0.7), 'z': (0.5, 0.6), 's': (0.6, 0.6), 'h': (0.6, 0.4)
    }
    chars = list(root_slp1)
    formants = []
    for c in chars[:6]:
        f1, f2 = PHONEME_FORMANTS.get(c, (0.5, 0.5))
        formants.extend([f1, f2])
    while len(formants) < 12: formants.extend([0.0, 0.0])
    return np.array(formants[:12])

def locus_features(root_slp1):
    c = root_slp1[0] if root_slp1 else ''
    PHONEME_LOCUS = {
        'k': (0.1, 0.1), 'K': (0.1, 0.2), 'g': (0.2, 0.1), 'G': (0.2, 0.2), 'N': (0.1, 0.3),
        'c': (0.3, 0.6), 'C': (0.3, 0.7), 'j': (0.4, 0.6), 'J': (0.4, 0.7), 'Y': (0.3, 0.5),
        'w': (0.4, 0.2), 'W': (0.4, 0.3), 'q': (0.5, 0.2), 'Q': (0.5, 0.3), 'R': (0.4, 0.4),
        't': (0.5, 0.5), 'T': (0.5, 0.6), 'd': (0.6, 0.5), 'D': (0.6, 0.6), 'n': (0.5, 0.4),
        'p': (0.3, 0.8), 'P': (0.3, 0.9), 'b': (0.4, 0.8), 'B': (0.4, 0.9), 'm': (0.4, 0.7)
    }
    f1_t1, f1_t2 = PHONEME_LOCUS.get(c, (0.5, 0.5))
    return np.array([f1_t1, f1_t2, float(c in 'kKgGNcCjJWwTqQtTdDpPbBmyrlvSzsh'), 
                   float(c in 'kKgGNcCjJWwTqQtTdDpPbB'), float(c in 'KCTPh'), float(c in 'aeiouAEIOUfF')])

def phonation_features(root_slp1):
    c = root_slp1[0] if root_slp1 else ''
    return np.array([float(c in 'KgGjJqQdDbB'), float(c in 'KChWTP'), float(c in 'gGjJqQdDbBmnN')])

def acoustic_features(root_slp1):
    return np.concatenate([locus_features(root_slp1), phonation_features(root_slp1), formant_features(root_slp1)])

# ── AdEx population ────────────────────────────────────────────────────────────
class AdExPopulation(nn.Module):
    def __init__(self, size, dt=1.0, C=200.0, gL=1.5, EL=-70.0, VT=-50.0, DeltaT=2.0, Vpeak=0.0, Vreset=-58.0, a=2.0, b=10.0, tau_w=30.0):
        super().__init__()
        self.size = size; self.dt = dt; self.C = C; self.gL = gL; self.EL = EL; self.DeltaT = DeltaT
        self.register_buffer('Vreset', torch.tensor([[Vreset]])); self.register_buffer('Vpeak', torch.tensor([[Vpeak]]))
        self.v_thresh = nn.Parameter(torch.ones(1, size) * VT); self.v_reset = nn.Parameter(torch.ones(1, size) * Vreset)
        self.a = nn.Parameter(torch.ones(1, size) * a); self.b = nn.Parameter(torch.ones(1, size) * b); self.tau_w = nn.Parameter(torch.ones(1, size) * tau_w)
        self.register_buffer('V', torch.ones(1, size) * EL); self.register_buffer('w', torch.zeros(1, size)); self.register_buffer('spike_counts', torch.zeros(1, size))
    def reset_states(self): self.V.fill_(self.EL); self.w.zero_(); self.spike_counts.zero_()
    def step(self, I_ext):
        exp_term = self.gL * self.DeltaT * torch.exp(torch.clamp((self.V - self.v_thresh) / self.DeltaT, max=20.0))
        dV = (-self.gL*(self.V-self.EL) + exp_term - self.w + I_ext) / self.C
        self.V = self.V + self.dt * dV
        dw = (self.a*(self.V-self.EL) - self.w) / self.tau_w
        self.w = self.w + self.dt * dw
        spikes = (self.V >= self.Vpeak)
        self.spike_counts = self.spike_counts + spikes.float()
        self.V = torch.where(spikes, self.v_reset.expand_as(self.V), self.V); self.w = torch.where(spikes, self.w + self.b, self.w)
        return spikes

class MegaSNN(nn.Module):
    def __init__(self, n_neurons=512):
        super().__init__()
        self.L1 = AdExPopulation(size=n_neurons, a=0.0, b=0.0, tau_w=5.0, gL=1.5)
        self.proj_acoustic = nn.Parameter(torch.randn(21, n_neurons) * 12000.0 * (torch.rand(21, n_neurons) < 0.35).float())
        self.proj_prior = nn.Parameter(torch.randn(5, n_neurons) * 2000.0 * (torch.rand(5, n_neurons) < 0.35).float())
        self.L2 = AdExPopulation(size=256, a=2.0, b=10.0, tau_w=30.0, gL=1.0)
        self.W12 = nn.Parameter(torch.randn(n_neurons, 256) * 1800.0 * (torch.rand(n_neurons, 256) < 0.35).float())
    def reset(self): self.L1.reset_states(); self.L2.reset_states()
    def step(self, I_in):
        I_acoustic, I_prior = I_in[:, :21], I_in[:, 21:]
        I1 = (I_acoustic @ self.proj_acoustic) + (I_prior @ self.proj_prior)
        sp1 = self.L1.step(I1)
        return self.L2.step(sp1.float() @ self.W12)

# ── Permutation Test ──────────────────────────────────────────────────────────
def permutation_test(labels, pred, n_perms=1000):
    obs_ari = adjusted_rand_score(labels, pred)
    perm_aris = []
    labels_copy = labels.copy()
    for _ in range(n_perms):
        np.random.shuffle(labels_copy)
        perm_aris.append(adjusted_rand_score(labels_copy, pred))
    p_value = np.sum(np.array(perm_aris) >= obs_ari) / n_perms
    return obs_ari, p_value

# ── Experiment Core ───────────────────────────────────────────────────────────
def run_condition(inputs, labels, device, name, n_neurons=512):
    print(f"\nCondition: {name}")
    model = MegaSNN(n_neurons=n_neurons).to(device)
    all_states = []
    for row in inputs:
        model.reset()
        I = torch.tensor(row, dtype=torch.float32).unsqueeze(0).to(device)
        trajectory = []
        for _ in range(20):
            model.step(I)
            trajectory.append(model.L2.V.cpu().numpy().squeeze())
        all_states.append(trajectory[-1] + 70.0)
    
    states_norm = StandardScaler().fit_transform(np.array(all_states))
    pred = KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(states_norm)
    ari, p = permutation_test(labels, pred)
    print(f"  ARI = {ari:.4f} (p = {p:.4f})")
    return ari, p

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    test_locus_labeling()
    
    CORPUS_PATH = 'temp_ashtadhyayi_data/dhatu/data.txt'
    with open(CORPUS_PATH, encoding='utf-8') as f:
        full_corpus = json.load(f)['data']
    
    def devanagari_to_slp1_internal(text):
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

    processed = []
    for item in full_corpus:
        root = devanagari_to_slp1_internal(item['dhatu'])
        label = get_locus_label(root)
        if label != -1:
            processed.append((root, label))
    
    # Failure Mode 3 Guard: Report unique root count
    unique_roots = sorted(list(set([p[0] for p in processed])))
    print(f"Unique roots in corpus: {len(unique_roots)}")
    
    # For this experiment, we use unique roots to avoid duplication artifacts
    final_corpus = []
    seen = set()
    for root, label in processed:
        if root not in seen:
            final_corpus.append((root, label))
            seen.add(root)
            
    print(f"Total roots for experiment: {len(final_corpus)}")
    
    labels = np.array([p[1] for p in final_corpus])
    acoustic = np.array([acoustic_features(p[0]) for p in final_corpus])
    
    # ── Condition A: Acoustic-only ─────────────────────────────────────────────
    # Uniform 5D prior
    inputs_a = np.array([np.concatenate([ac, np.ones(5)*0.2]) for ac in acoustic])
    ari_a, p_a = run_condition(inputs_a, labels, device, "A: Acoustic-only (Uniform Prior)")
    
    # ── Condition B: Prior-only ───────────────────────────────────────────────
    # One-hot locus prior
    priors = []
    for lbl in labels:
        p = np.zeros(5)
        p[lbl] = 1.0
        priors.append(p)
    inputs_b = np.array([np.concatenate([np.zeros(21), p]) for p in priors])
    ari_b, p_b = run_condition(inputs_b, labels, device, "B: Prior-only (Locus Prior)")
    
    # ── Condition C: Full Fusion ──────────────────────────────────────────────
    inputs_c = np.array([np.concatenate([ac, p]) for ac, p in zip(acoustic, priors)])
    ari_c, p_c = run_condition(inputs_c, labels, device, "C: Full Fusion")
    
    # ── MLP Baseline ──────────────────────────────────────────────────────────
    print("\nMLP Baseline (Acoustic-only)")
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    scores = cross_val_score(clf, acoustic, labels, cv=5, scoring='adjusted_rand_score')
    print(f"  MLP ARI = {scores.mean():.4f} ± {scores.std():.4f}")

if __name__ == "__main__":
    main()
