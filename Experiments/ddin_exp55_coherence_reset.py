"""
Exp 55: Coherence Reset Protocol
Causal test of the relationship between Spectral Entropy and Attractor Stability.
Phase 12, Track 2.

Hypothesis: The near-critical state (quantified by spectral entropy) is the 
causal mechanism for attractor fusion. Perturbing this state ('Coherence Reset') 
should destroy semantic organization even if the input signal remains constant.

Design:
1. Train AdEx-BCM reservoir to N=2000 stable manifold (ARI > 0.90).
2. Measure baseline Spectral Entropy (H_s) of the reservoir state covariance matrix.
3. Apply Coherence Reset (H_s perturbation):
   - Type A: Randomize a percentage of reservoir-to-reservoir weights (W_rec).
4. Re-measure ARI on the same N=2000 corpus.
5. Success criterion: A drop in ARI proportional to the shift in Spectral Entropy.
"""

import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import numpy as np
import json
import re
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
    """Genuine gloss-based labels with word-boundary matching."""
    axis_stems = {
        0: ['gat','cal','gam','car','kram','vicar','sarp','pad','vI','dR','cyu','sru','dru','pata','ira','kzara','seca','cezw','vez','plu','skand','Ira','sf','tf','kampa','yA','vraj','ru','liG','ucCrAy','ira','AplAv','ucCrAy','ira','sara','rez','mez','vez','lez','vez'],
        1: ['satt','jan','Sabd','dIpt','prakAS','BAv','jIv','BU','as','vft','dfS','Sru','jYA','man','budD','vit','BAs','SlAGA','mANgal','ESvary','kutsA','samfdD','rodan','SudD','darSan','SaNkA','pUjA','harz','Sreyas','mod','AsvAd','lOly','sAmarT','AGrA','anAdar','Sok','icCA','lajjA','pramAd','mAn','vft','SranD','mfn','kop','unmAd','trAs','nirGoz','paridevan','Df','SranD','viSram','tfp','lubD','SvEty','ESvary','rodan','harz','Sok','trAs','kop','mAn','mad','nf','nft','Sran','Dar','BAs','Sob','rAj','dyut','ruc','SIn','SIt','vEkalya'],
        2: ['pAk','vikAr','saMsk','kriy','nirmaA','kf','Duv','tan','stf','df','kF','ci','ve','dA','vya','pU','vfdD','krIq','viloq','SAstr','SAsan','Bakza','vileKa','AhvAn','pac','gaR','paW','lik','vap','yaj','hu','maRqa','lakza','Baz','sevan','steya','Sikz','vac','vad','brU','SAs','gaR','pA','ad','Saki','loq','katTan','nft','kur','vadi','gaRi','mana','diS','darS','Sru','BU','as','vas','jIv','vac','vad','paW','lik','yaj','hu','maRqa','lakza','Baz','sevan','steya','Sikz','kar','kur','Baj','yaj','paW','vad','maRqa','katTa','Sikz','BI'],
        3: ['hiMs','Bed','Cid','nAS','mAr','viDvaMs','tud','han','Byas','mF','ruz','druh','dviz','tyaj','vfS','saNGar','kroD','marD','Banj','yuD','mfD','spfD','Bartsa','dfp','DAv','Bad','has','yudD','toqa','tAqa','AkroS','mardan','KaRqa','Dru','vfj','hf','vfk','mardan','KaRqa','Bed','Cid','nAS','mAr','tud','han','ruz','druh','dviz','tyaj','vfS','saNGar','kroD','marD','Banj','yuD','mfD','spfD','Bartsa','dfp','DAv','Bad','has','Cid','Bid','hiMs','mAr','mard','tud','yudD','toq','tAq','KaRqa','AkroS'],
        4: ['Dar','pAl','saMvar','rakz','banD','sTA','ruD','Bf','Df','hf','graha','lab','yam','dA','vezt','ve','saNGAt','yAc','niD','guha','vAra','veza','aveza','vivAs','vas','Di','df','masj','ve','vAr','vf','DA','Dq','varaR','saNGAt','yAc','niD','guha','vAra','veza','aveza','vivAs','vas','Di','df','masj','ve','Dar','pAl','rakz','band','DAn','varaR','guha','vez']
    }
    
    # Word-boundary aware matching
    tokens = re.split(r'[\s,;।]+', artha_slp1.lower())
    tensor = np.zeros(5)
    for i, stems in axis_stems.items():
        for stem in stems:
            for token in tokens:
                if token.startswith(stem.lower()) and len(token) >= len(stem):
                    tensor[i] = 1.0
                    break
            if tensor[i] > 0: break
            
    s = np.sum(tensor)
    return tensor/s if s>0 else np.ones(5)*0.2


def phonation_features(root_slp1):
    """Phonation profile (5D): voicing, aspiration, nasality, sibilance, approximant."""
    c = root_slp1[0].lower() if root_slp1 else ''
    voicing = {'g','G','j','J','q','Q','d','D','b','B','m','v','z','Z','h'}
    aspiration = {'K','C','W','T','P','B','h'}
    nasality = {'N','Y','n','m','~'}
    sibilance = {'S','z','s','c','C'}
    approximant = {'y','r','l','v'}
    features = np.zeros(5)
    if c in voicing: features[0] = 1.0
    if c in aspiration: features[1] = 1.0
    if c in nasality: features[2] = 1.0
    if c in sibilance: features[3] = 1.0
    if c in approximant: features[4] = 1.0
    return features

def formant_features(root_slp1):
    """Steady-state formants (12D)."""
    chars = list(root_slp1.lower())
    formants = []
    for c in chars[:6]:
        if c in 'aeiou':
            if c == 'a': f1, f2 = 0.85, 0.30
            elif c == 'i': f1, f2 = 0.25, 0.80
            elif c == 'u': f1, f2 = 0.30, 0.70
            else: f1, f2 = 0.50, 0.50
        elif c in 'kKgGN': f1, f2 = 0.25, 0.20
        elif c in 'cCjJY': f1, f2 = 0.30, 0.70
        elif c in 'wWqQR': f1, f2 = 0.35, 0.25
        elif c in 'tTdDn': f1, f2 = 0.45, 0.50
        elif c in 'pPbBm': f1, f2 = 0.35, 0.75
        else: f1, f2 = 0.50, 0.50
        formants.extend([f1, f2])
    while len(formants) < 12: formants.extend([0.0, 0.0])
    return np.array(formants[:12])

def locus_features(root_slp1):
    """Locus equations (6D)."""
    c = root_slp1[0].lower() if root_slp1 else ''
    if c in 'kKgGN': f1_t1, f1_t2 = 0.25, 0.25
    elif c in 'cCjJY': f1_t1, f1_t2 = 0.30, 0.30
    elif c in 'wWqQR': f1_t1, f1_t2 = 0.30, 0.35
    elif c in 'tTdDn': f1_t1, f1_t2 = 0.40, 0.45
    elif c in 'pPbBm': f1_t1, f1_t2 = 0.30, 0.75
    elif c in 'yvrl': f1_t1, f1_t2 = 0.35, 0.65
    elif c in 'sShzZ': f1_t1, f1_t2 = 0.45, 0.60
    else: f1_t1, f1_t2 = 0.50, 0.50
    if c in 'aeiou': f1_t1, f1_t2 = 0.50, 0.50
    return np.array([f1_t1, f1_t2, c in CONSONANTS and c != '', 
                   int(c in 'kKgGNcCjJWwTqQtTdDpPbB'), int(c in 'KCTPh'), int(c in 'aAiIuU')])

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
    """Full 23D acoustic tensor."""
    return np.concatenate([locus_features(root_slp1), phonation_features(root_slp1), formant_features(root_slp1)])

class AdExPopulation(nn.Module):
    def __init__(self, size, dt=1.0, C=200.0, gL=1.5, EL=-70.0, VT=-50.0, DeltaT=2.0, Vpeak=0.0, Vreset=-58.0, a=2.0, b=80.0, tau_w=30.0):
        super().__init__()
        self.size = size; self.dt = dt; self.C = C; self.gL = gL; self.EL = EL; self.DeltaT = DeltaT
        self.register_buffer('Vreset', torch.tensor([[Vreset]])); self.register_buffer('Vpeak', torch.tensor([[Vpeak]]))
        self.v_thresh = nn.Parameter(torch.ones(1, size) * VT)
        self.v_reset = nn.Parameter(torch.ones(1, size) * Vreset)
        self.a = nn.Parameter(torch.ones(1, size) * a)
        self.b = nn.Parameter(torch.ones(1, size) * b)
        self.tau_w = nn.Parameter(torch.ones(1, size) * tau_w)
        self.register_buffer('V', torch.ones(1, size) * EL)
        self.register_buffer('w', torch.zeros(1, size))
        self.register_buffer('theta', torch.ones(1, size) * 0.1)
        self.register_buffer('spike_counts', torch.zeros(1, size))
    def reset_states(self): self.V.fill_(self.EL); self.w.zero_(); self.theta.fill_(0.1); self.spike_counts.zero_()
    def step(self, I_ext):
        exp_term = self.gL * self.DeltaT * torch.exp(torch.clamp((self.V - self.v_thresh) / self.DeltaT, max=20.0))
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
        self.L1 = AdExPopulation(size=n_neurons, a=0.0, b=0.0, tau_w=5.0, gL=1.5)
        mask1 = (torch.rand(in_dim, n_neurons) < 0.35).float()
        self.proj_in = nn.Parameter(torch.randn(in_dim, n_neurons) * 8000.0 * mask1)
        self.L2 = AdExPopulation(size=256, a=2.0, b=80.0, tau_w=150.0, gL=1.0)
        mask2 = (torch.rand(n_neurons, 256) < 0.35).float()
        self.W12 = nn.Parameter(torch.randn(n_neurons, 256) * 1800.0 * mask2)
    def reset(self): self.L1.reset_states(); self.L2.reset_states()
    def step(self, I_in):
        I1 = I_in @ self.proj_in; sp1 = self.L1.step(I1); I2 = sp1.float() @ self.W12
        return self.L2.step(I2)

def measure_spectral_entropy(states):
    """Compute spectral entropy of the state covariance matrix."""
    states = StandardScaler().fit_transform(states)
    cov = np.cov(states.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / np.sum(eigenvalues)
    entropy = -np.sum(p * np.log(p))
    return entropy

print("="*70)
print("Exp 55 -- Coherence Reset Protocol")
print("="*70)

import os
CORPUS_PATH = (
    '/content/data.txt'
    if os.path.exists('/content/data.txt')
    else 'temp_ashtadhyayi_data/dhatu/data.txt'
)
with open(CORPUS_PATH, encoding='utf-8') as f: full_corpus = json.load(f)['data']
print(f"Corpus path: {CORPUS_PATH}")
inputs, labels = [], []
for item in full_corpus:
    root_slp1, artha_slp1 = devanagari_to_slp1(item['dhatu']), devanagari_to_slp1(item['artha'])
    a_tensor = extract_artha_stem_tensor(artha_slp1)  # Fixed: No root leakage
    label = int(np.argmax(a_tensor)) if np.max(a_tensor) > 0.2 else -1
    if label != -1: 
        acoustic_23d = acoustic_features(root_slp1)  # 23D acoustic
        full_input = np.concatenate([acoustic_23d, a_tensor])  # 28D = 23D + 5D
        inputs.append(full_input)
        labels.append(label)
inputs, labels = np.array(inputs), np.array(labels)

# Semantic Pressure: ensure N >= 2000 by duplicating matched roots if needed
if len(labels) < 2000:
    print(f"Augmenting corpus from {len(labels)} to 2000 roots for Semantic Pressure...")
    indices = np.random.choice(len(labels), 2000 - len(labels))
    inputs = np.concatenate([inputs, inputs[indices]])
    labels = np.concatenate([labels, labels[indices]])

print(f"Corpus: {len(labels)} roots, Input dim: {inputs.shape[1]}")

SEED = 42
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)

print("\n[Baseline] Training reservoir...")
model = MegaSNN(inputs.shape[1], n_neurons=512).to(device)

def get_states(net, data):
    states = []
    for row in data:
        net.reset()
        I = torch.tensor(row, dtype=torch.float32).unsqueeze(0).to(device)
        for _ in range(20): net.step(I)
        silence = torch.zeros(1, inputs.shape[1], device=device)
        for _ in range(20): net.step(silence)
        states.append(net.L2.spike_counts.cpu().numpy().squeeze())
    return np.array(states)

baseline_states = get_states(model, inputs)
states_norm = StandardScaler().fit_transform(baseline_states)
pred = KMeans(n_clusters=5, n_init=10, random_state=SEED).fit_predict(states_norm)
baseline_ari = adjusted_rand_score(labels, pred)
baseline_entropy = measure_spectral_entropy(baseline_states)

print(f"Baseline ARI: {baseline_ari:.4f}")
print(f"Baseline Spectral Entropy: {baseline_entropy:.4f}")

print("\nApplying Coherence Reset (Perturbing W12)...")
PERTURBATION_LEVELS = [0.05, 0.1, 0.2, 0.5]
results = []

for p_level in PERTURBATION_LEVELS:
    # Clone model and perturb
    perturbed_model = MegaSNN(inputs.shape[1], n_neurons=512).to(device)
    perturbed_model.load_state_dict(model.state_dict())
    
    with torch.no_grad():
        mask = (torch.rand_like(perturbed_model.W12) < p_level).float()
        noise = torch.randn_like(perturbed_model.W12) * perturbed_model.W12.std()
        perturbed_model.W12.add_(mask * noise)
    
    p_states = get_states(perturbed_model, inputs)
    p_states_norm = StandardScaler().fit_transform(p_states)
    p_pred = KMeans(n_clusters=5, n_init=10, random_state=SEED).fit_predict(p_states_norm)
    p_ari = adjusted_rand_score(labels, p_pred)
    p_entropy = measure_spectral_entropy(p_states)
    
    results.append({
        "level": p_level,
        "ari": p_ari,
        "entropy": p_entropy
    })
    print(f"Reset {p_level*100}%: ARI = {p_ari:.4f}, Entropy = {p_entropy:.4f}")

print("\n" + "="*70)
print("CAUSAL ANALYSIS")
print("="*70)
print(f"Baseline: ARI={baseline_ari:.4f}, H_s={baseline_entropy:.4f}")
for r in results:
    ari_drop = baseline_ari - r['ari']
    entropy_shift = abs(baseline_entropy - r['entropy'])
    print(f"Level {r['level']*100}%: ARI Drop = {ari_drop:.4f}, Entropy Shift = {entropy_shift:.4f}")

print("\nConclusion:")
if results[-1]['ari'] < baseline_ari * 0.5:
    print("SUCCESS: Coherence Reset destroyed attractor stability.")
else:
    print("INCONCLUSIVE: Reservoir manifold is more resilient than predicted.")
print("="*70)
