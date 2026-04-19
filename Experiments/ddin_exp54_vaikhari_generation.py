"""
Exp 54: Vaikharī Generation + Piṅgala Completion
Generates phonologically valid Sanskrit roots from semantic attractor states.
Uses trained decoder (Exp 47) + Piṅgala prosodic completion.
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

import re

def extract_artha_stem_tensor(artha_slp1):
    """Genuine gloss-based labels with word-boundary matching."""
    axis_stems = {
        0: ['gat','cal','gam','car','kram','vicar','sarp','pad','vI','dR','cyu','sru','dru','pata','ira','kzara','seca','cezw','vez','plu','skand','Ira','sf','tf','kampa','vraj','ru','liG','ucCrAy','ira','AplAv','ucCrAy','ira','sara','rez','mez','vez','lez','vez'],
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
    voicing = {'g','G','j','J','q','Q','d','D','b','B','m','v','z','Z','h'}  # voiced
    aspiration = {'K','C','W','T','P','B','h'}  # aspirated
    nasality = {'N','Y','n','m','~'}  # nasals
    sibilance = {'S','z','s','c','C'}  # sibilants
    approximant = {'y','r','l','v'}  # approximants
    features = np.zeros(5)
    if c in voicing: features[0] = 1.0
    if c in aspiration: features[1] = 1.0
    if c in nasality: features[2] = 1.0
    if c in sibilance: features[3] = 1.0
    if c in approximant: features[4] = 1.0
    return features

def formant_features(root_slp1):
    """Normalized high-fidelity formants (12D)."""
    chars = list(root_slp1.lower())
    formants = []
    # Differentiated phoneme-specific formants (Values in [0, 1])
    PHONEME_FORMANTS = {
        'a': (0.8, 0.3), 'A': (0.9, 0.2),
        'i': (0.2, 0.8), 'I': (0.1, 0.9),
        'u': (0.3, 0.7), 'U': (0.2, 0.8),
        'f': (0.4, 0.4), 'F': (0.3, 0.3),
        'k': (0.1, 0.1), 'K': (0.1, 0.2), 'g': (0.2, 0.1), 'G': (0.2, 0.2), 'N': (0.1, 0.3),
        'c': (0.3, 0.6), 'C': (0.3, 0.7), 'j': (0.4, 0.6), 'J': (0.4, 0.7), 'Y': (0.3, 0.5),
        'w': (0.4, 0.2), 'W': (0.4, 0.3), 'q': (0.5, 0.2), 'Q': (0.5, 0.3), 'R': (0.4, 0.4),
        't': (0.5, 0.5), 'T': (0.5, 0.6), 'd': (0.6, 0.5), 'D': (0.6, 0.6), 'n': (0.5, 0.4),
        'p': (0.3, 0.8), 'P': (0.3, 0.9), 'b': (0.4, 0.8), 'B': (0.4, 0.9), 'm': (0.4, 0.7),
        'y': (0.2, 0.6), 'r': (0.3, 0.5), 'l': (0.5, 0.5), 'v': (0.3, 0.7),
        'S': (0.4, 0.7), 'z': (0.5, 0.6), 's': (0.6, 0.6), 'h': (0.6, 0.4)
    }
    
    for c in chars[:6]:
        f1, f2 = PHONEME_FORMANTS.get(c, (0.5, 0.5))
        formants.extend([f1, f2])
    while len(formants) < 12: formants.extend([0.0, 0.0])
    return np.array(formants[:12])

def locus_features(root_slp1):
    """Normalized high-fidelity locus features (6D)."""
    c = root_slp1[0].lower() if root_slp1 else ''
    PHONEME_LOCUS = {
        'k': (0.1, 0.1), 'K': (0.1, 0.2), 'g': (0.2, 0.1), 'G': (0.2, 0.2), 'N': (0.1, 0.3),
        'c': (0.3, 0.6), 'C': (0.3, 0.7), 'j': (0.4, 0.6), 'J': (0.4, 0.7), 'Y': (0.3, 0.5),
        'w': (0.4, 0.2), 'W': (0.4, 0.3), 'q': (0.5, 0.2), 'Q': (0.5, 0.3), 'R': (0.4, 0.4),
        't': (0.5, 0.5), 'T': (0.5, 0.6), 'd': (0.6, 0.5), 'D': (0.6, 0.6), 'n': (0.5, 0.4),
        'p': (0.3, 0.8), 'P': (0.3, 0.9), 'b': (0.4, 0.8), 'B': (0.4, 0.9), 'm': (0.4, 0.7)
    }
    f1_t1, f1_t2 = PHONEME_LOCUS.get(c, (0.5, 0.5))
    if c in 'aeiouAEIOUfF': f1_t1, f1_t2 = 0.5, 0.5
    elif c in 'yrlv': f1_t1, f1_t2 = 0.3, 0.6
    elif c in 'Szsh': f1_t1, f1_t2 = 0.4, 0.6
    
    return np.array([f1_t1, f1_t2, float(c in CONSONANTS and c != ''), 
                   float(c in 'kKgGNcCjJWwTqQtTdDpPbB'), float(c in 'KCTPh'), float(c in 'aAiIuU')])
    
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
    """Full 23D acoustic tensor, normalized to [0, 1]."""
    features = np.concatenate([locus_features(root_slp1), phonation_features(root_slp1), formant_features(root_slp1)])
    # Ensure all features are in a consistent range
    return features

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
        self.V = torch.where(spikes, self.v_reset.expand_as(self.V), self.V)
        self.w = torch.where(spikes, self.w + self.b, self.w)
        self.theta = self.theta + 0.005 * (spikes.float() - self.theta)
        return spikes

class MegaSNN(nn.Module):
    def __init__(self, in_dim, n_neurons=512):
        super().__init__()
        self.L1 = AdExPopulation(size=n_neurons, a=0.0, b=0.0, tau_w=5.0, gL=1.5)
        self.mask_acoustic = (torch.rand(23, n_neurons) < 0.35).float()
        self.mask_prior = (torch.rand(5, n_neurons) < 0.35).float()
        # Scale input projections relative to input channel density
        self.proj_acoustic = nn.Parameter(torch.randn(23, n_neurons) * 12000.0 * self.mask_acoustic)
        # Prior is 5D but has 10x signal boost in Condition C, so scale proj_prior down to balance
        self.proj_prior = nn.Parameter(torch.randn(5, n_neurons) * 2000.0 * self.mask_prior)
        
        self.L2 = AdExPopulation(size=256, a=2.0, b=10.0, tau_w=30.0, gL=1.0)
        mask2 = (torch.rand(n_neurons, 256) < 0.35).float()
        self.W12 = nn.Parameter(torch.randn(n_neurons, 256) * 1800.0 * mask2)
    def reset(self): self.L1.reset_states(); self.L2.reset_states()
    def step(self, I_in):
        # I_in is [batch, 28] -> split into [batch, 23] and [batch, 5]
        I_acoustic = I_in[:, :23]
        I_prior = I_in[:, 23:]
        I1 = (I_acoustic @ self.proj_acoustic) + (I_prior @ self.proj_prior)
        sp1 = self.L1.step(I1)
        I2 = sp1.float() @ self.W12
        return self.L2.step(I2)

def get_states(net, data):
    states = []
    for row in data:
        net.reset()
        I = torch.tensor(row, dtype=torch.float32).unsqueeze(0).to(device)
        # Record membrane potential at every step
        trajectory = []
        for _ in range(20):
            net.step(I)
            trajectory.append(net.L2.V.cpu().numpy().squeeze())
        # Use only the LAST step for state representation to capture converged dynamics
        # Shift and scale to avoid near-zero values before averaging
        states.append(trajectory[-1] + 70.0)
    return np.array(states)

class VaikharīDecoder(nn.Module):
    def __init__(self, reservoir_dim=256, output_dim=23):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(reservoir_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.decoder(x)

def compute_pingala_address(root_slp1, max_syllables=4):
    chars = list(root_slp1)
    syllables = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c in LONG_VOWELS or c in SHORT_VOWELS:
            is_guru = c in LONG_VOWELS
            j, cluster = i + 1, 0
            while j < len(chars) and chars[j] in CONSONANTS: cluster += 1; j += 1
            if cluster > 1: is_guru = True
            syllables.append(1 if is_guru else 0)
            i = j
        else: i += 1
    addr = syllables[:max_syllables]
    while len(addr) < max_syllables: addr.append(0)
    return addr

def invert_acoustic_to_phoneme(acoustic_28d):
    """Invert full 28D acoustic features to phonological components."""
    # Extract locus from dimensions 0-5
    locus_vec = acoustic_28d[:6]
    # Extract phonation from dimensions 6-10
    phonation_vec = acoustic_28d[6:11]
    # Extract formants from dimensions 11-22
    formant_vec = acoustic_28d[11:23]
    
    # Determine initial consonant from locus (first 2 dims encode F1/F2 transition)
    f1_trans, f2_trans = locus_vec[0], locus_vec[1]
    
    # Classify locus region
    if f2_trans < 0.35:
        base_cons = 'k' if phonation_vec[0] < 0.5 else 'g'
    elif f2_trans < 0.55:
        base_cons = 'c' if phonation_vec[0] < 0.5 else 'j'
    elif f2_trans < 0.40:
        base_cons = 'w' if phonation_vec[0] < 0.5 else 'q'
    elif f2_trans < 0.65:
        base_cons = 't' if phonation_vec[0] < 0.5 else 'd'
    else:
        base_cons = 'p' if phonation_vec[0] < 0.5 else 'b'
    
    # Determine vowel quality from first formant pair
    f1, f2 = formant_vec[0], formant_vec[1]
    if f1 < 0.4 and f2 > 0.6:
        vowel = 'i'
    elif f1 < 0.4 and f2 < 0.4:
        vowel = 'u'
    elif f1 > 0.6:
        vowel = 'a'
    else:
        vowel = 'a'
    
    # Determine syllable count from prosodic portion
    n_syllables = int(np.clip(np.round(locus_vec[5] + 1), 1, 3))
    
    return base_cons, vowel, n_syllables

def pingala_completion(initial_consonant, vowel, n_syllables=2):
    """Generates a structured root using Piṅgala-like combinatorial rules."""
    # Common Sanskrit root patterns (CVC, CV, CVCC)
    patterns = {
        1: ["CV"],
        2: ["CVC", "CVV"],
        3: ["CVCC", "CVCV"]
    }
    
    # Select pattern based on n_syllables
    options = patterns.get(n_syllables, ["CVC"])
    pattern = options[0] # Default to first option for stability
    
    root = ""
    v_map = {'A': 'A', 'a': 'a'}
    
    if pattern == "CV":
        root = initial_consonant + vowel
    elif pattern == "CVC":
        # Use a terminal consonant based on the initial one (homorganic or common)
        term_map = {'k':'t', 'g':'m', 'c':'n', 'j':'l', 'w':'r', 'q':'d', 't':'s', 'd':'n', 'p':'l', 'b':'r'}
        term = term_map.get(initial_consonant.lower(), 'm')
        root = initial_consonant + vowel + term
    elif pattern == "CVCC":
        term_map = {'k':'kt', 'g':'mD', 'c':'nt', 'j':'ly', 'w':'rt', 'q':'nd', 't':'st', 'd':'nd', 'p':'lp', 'b':'rb'}
        term = term_map.get(initial_consonant.lower(), 'nt')
        root = initial_consonant + vowel + term
    elif pattern == "CVCV":
        root = initial_consonant + vowel + initial_consonant.lower() + 'a'
        
    return root

print("="*70)
print("Exp 54 -- Vaikhari Generation + Pingala Completion")
print("="*70)

import os
CORPUS_PATH = (
    '/content/data.txt'
    if os.path.exists('/content/data.txt')
    else 'temp_ashtadhyayi_data/dhatu/data.txt'
)
with open(CORPUS_PATH, encoding='utf-8') as f: full_corpus = json.load(f)['data']
print(f"Corpus path: {CORPUS_PATH}")
# Build input tensor: 23D acoustic + 5D prior = 28D total
inputs, labels = [], []
for item in full_corpus:
    root_slp1, artha_slp1 = devanagari_to_slp1(item['dhatu']), devanagari_to_slp1(item['artha'])
    a_tensor = extract_artha_stem_tensor(artha_slp1)  # Fixed: No root leakage
    label = int(np.argmax(a_tensor)) if np.max(a_tensor) > 0.2 else -1
    if label != -1: 
        acoustic_23d = acoustic_features(root_slp1)  # 23D acoustic
        # Boost the semantic prior signal by 10x to ensure it drives the reservoir
        full_input = np.concatenate([acoustic_23d, a_tensor * 10.0])  # 28D = 23D + 5D
        inputs.append(full_input)
        labels.append(label)
inputs, labels = np.array(inputs), np.array(labels)

# Semantic Pressure: ensure N >= 2000 by duplicating matched roots if needed
if len(labels) < 2000:
    print(f"Augmenting corpus from {len(labels)} to 2000 roots for Semantic Pressure...")
    indices = np.random.choice(len(labels), 2000 - len(labels))
    inputs = np.concatenate([inputs, inputs[indices]])
    labels = np.concatenate([labels, labels[indices]])

print(f"Corpus: {len(labels)} roots, Input dim: {inputs.shape[1]} (28D = 23D acoustic + 5D prior)")
print(f"Inputs Mean: {inputs.mean():.4f}, Std: {inputs.std():.4f}")

AXIS_NAMES = ['Motion (MOT)', 'Experiential (EXP)', 'Transformation (TRN)', 'Separation (SEP)', 'Containment (CNT)']

SEED = 42
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)

print("\nTraining reservoir...")
model = MegaSNN(inputs.shape[1], n_neurons=512).to(device)

# Spike rate diagnostic
spike_rates = []
for row in inputs[:100]:
    model.reset()
    I = torch.tensor(row, dtype=torch.float32).unsqueeze(0).to(device)
    for _ in range(20): model.step(I)
    rate = model.L2.spike_counts.mean().item()
    spike_rates.append(rate)
print(f"Mean L2 spikes/root: {np.mean(spike_rates):.4f}")
print(f"Max L2 spikes/root: {np.max(spike_rates):.4f}")

states = get_states(model, inputs)
print(f"States shape: {states.shape}, Mean: {states.mean():.4f}, Std: {states.std():.4f}")
if states.std() < 1e-6:
    print("WARNING: Reservoir states have zero variance! (Cluster collapse inevitable)")
states_norm = StandardScaler().fit_transform(states)
pred = KMeans(n_clusters=5, n_init=10, random_state=SEED).fit_predict(states_norm)
ari = adjusted_rand_score(labels, pred)
print(f"Reservoir ARI: {ari:.4f}")

print("\nTraining Vaikhari decoder...")
torch.set_grad_enabled(True)
decoder = VaikharīDecoder(reservoir_dim=256, output_dim=23).to(device)
optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
criterion = nn.MSELoss()
X_train = torch.from_numpy(states).float().to(device)
Y_train = torch.from_numpy(inputs[:, :23]).float().to(device)  # 23D acoustic features
for epoch in range(200):
    optimizer.zero_grad()
    recon = decoder(X_train)
    loss = criterion(recon, Y_train)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0: print(f"  Epoch {epoch}: loss = {loss.item():.6f}")
print(f"  Final loss: {loss.item():.6f}")
torch.set_grad_enabled(False)

# --- Ablation Study ---
print("\n" + "="*70)
print("ABLATION STUDY: ACOUSTIC VS PRIOR VS FUSION")
print("="*70)

# Condition A: Acoustic-only (uniform prior)
print("\nCondition A: Acoustic-only (uniform prior)")
inputs_acoustic = inputs.copy()
inputs_acoustic[:, -5:] = 0.2 # Uniform 5D prior
model_a = MegaSNN(inputs.shape[1], n_neurons=512).to(device)
states_a = get_states(model_a, inputs_acoustic)
states_a_norm = StandardScaler().fit_transform(states_a)
ari_a = adjusted_rand_score(labels, KMeans(n_clusters=5, n_init=10, random_state=SEED).fit_predict(states_a_norm))
print(f"Acoustic-only Reservoir ARI: {ari_a:.4f}")

# Condition B: Prior-only (zero acoustic)
print("\nCondition B: Prior-only (zero acoustic)")
inputs_prior = inputs.copy()
inputs_prior[:, :23] = 0.0 # Zero 23D acoustic
model_b = MegaSNN(inputs.shape[1], n_neurons=512).to(device)
states_b = get_states(model_b, inputs_prior)
states_b_norm = StandardScaler().fit_transform(states_b)
ari_b = adjusted_rand_score(labels, KMeans(n_clusters=5, n_init=10, random_state=SEED).fit_predict(states_b_norm))
print(f"Prior-only Reservoir ARI: {ari_b:.4f}")

# Condition C: Full Fusion
print("\nCondition C: Full Fusion")
ari_fusion = adjusted_rand_score(labels, KMeans(n_clusters=5, n_init=10, random_state=SEED).fit_predict(StandardScaler().fit_transform(states)))
print(f"Full Fusion Reservoir ARI: {ari_fusion:.4f}")

print(f"\nDelta ARI (Fusion - Prior): {ari_fusion - ari_b:.4f}")

print("\n" + "="*70)
print("GENERATING NOVEL ROOTS FROM ATTRACTOR INTERIORS")
print("="*70)

print("\nSampling 10 roots per semantic axis...")
generated_roots = {i: [] for i in range(5)}

for axis in range(5):
    axis_indices = np.where(labels == axis)[0]
    if len(axis_indices) > 0:
        # Use states directly for sampling to avoid normalization artifacts
        axis_states = states[axis_indices]
        axis_centroid = axis_states.mean(axis=0)
        axis_std = np.std(axis_states, axis=0)
        
        for _ in range(10):
            # Sample with small noise scaled by state std
            noise = np.random.randn(256) * 0.05 * axis_std
            sample_tensor = torch.tensor(axis_centroid + noise, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # decoder output is 23D acoustic features
                recon_acoustic = decoder(sample_tensor).cpu().numpy().squeeze()
            initial_cons, vowel_quality, n_syllables = invert_acoustic_to_phoneme(recon_acoustic)
            generated_root = pingala_completion(initial_cons, vowel_quality, n_syllables)
            generated_roots[axis].append(generated_root)
    else:
        print(f"  Axis {axis}: no samples found in corpus")

print("\n" + "-"*70)
print("GENERATED ROOTS (for human evaluation)")
print("-"*70)
for axis in range(5):
    print(f"\n{AXIS_NAMES[axis]}:")
    for root in generated_roots[axis]:
        print(f"  {root}")

# Save results for Human Evaluation (only axes with generated roots)
output_data = []
for axis, roots in generated_roots.items():
    for root in roots:
        output_data.append({
            "basin": AXIS_NAMES[axis],
            "generated_root": root
        })

total_saved = len(output_data)
if total_saved > 0:
    with open('vaikhari_generated.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved {total_saved} generated roots to vaikhari_generated.json")
else:
    print("\nWARNING: No roots generated - check corpus matching")

print("\n" + "="*70)
print("HUMAN EVALUATION PROTOCOL")
print("="*70)
print("""
For each generated root, ask 3 evaluators (Sanskrit phonetics background):
  1. "Is this a phonologically valid Sanskrit-like sequence?" (1-5 scale)
  2. "Which semantic category does this feel most consistent with?" (5-choice)

Success criterion: > 20% agreement between evaluator assignment and source basin.
If evaluators assign Motion roots to Motion basin at > 30%, the generative claim holds.
""")
print("="*70)