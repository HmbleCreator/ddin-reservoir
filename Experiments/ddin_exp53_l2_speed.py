
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

torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Character sets ─────────────────────────────────────────────────────────────
LONG_VOWELS  = set('AEIOUfF')
SHORT_VOWELS = set('aiux')
CONSONANTS   = set('kKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzsh')

# ── Transliteration ────────────────────────────────────────────────────────────
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

# ── Corrected Label Logic ──────────────────────────────────────────────────────
def extract_artha_stem_tensor(artha_slp1):
    """Genuine gloss-based labels with word-boundary matching."""
    axis_stems = {
        0: ['gat','cal','gam','car','kram','vicar','sarp','pad','vI','dR','cyu','sru','dru','pata','ira','kzara','seca','cezw','vez','plu','skand','Ira','sf','tf','kampa','vraj','ru','liG','ucCrAy','ira','AplAv','ucCrAy','ira','sara','rez','mez','vez','lez','vez'],
        1: ['satt','jan','Sabd','dIpt','prakAS','BAv','jIv','BU','as','vft','dfS','Sru','jYA','man','budD','vit','BAs','SlAGA','mANgal','ESvary','kutsA','samfdD','rodan','SudD','darSan','SaNkA','pUjA','harz','Sreyas','mod','AsvAd','lOly','sAmarT','AGrA','anAdar','Sok','icCA','lajjA','pramAd','mAn','vft','SranD','mfn','kop','unmAd','trAs','nirGoz','paridevan','Df','SranD','viSram','tfp','lubD','SvEty','ESvary','rodan','harz','Sok','trAs','kop','mAn','mad','nf','nft','Sran','Dar','BAs','Sob','rAj','dyut','ruc','SIn','SIt','vEkalya'],
        2: ['pAk','vikAr','saMsk','kriy','nirmaA','kf','Duv','tan','stf','df','kF','ci','ve','dA','vya','pU','vfdD','krIq','viloq','SAstr','SAsan','Bakza','vileKa','AhvAn','pac','gaR','paW','lik','vap','yaj','hu','maRqa','lakza','Baz','sevan','steya','Sikz','vac','vad','brU','SAs','gaR','pA','ad','Saki','loq','katTan','nft','kur','vadi','gaRi','mana','diS','darS','Sru','BU','as','vas','jIv','vac','vad','paW','lik','yaj','hu','maRqa','lakza','Baz','sevan','steya','Sikz','kar','kur','Baj','yaj','paW','vad','maRqa','katTa','Sikz','BI'],
        3: ['hiMs','Bed','Cid','nAS','mAr','viDvaMs','tud','han','Byas','mF','ruz','druh','dviz','tyaj','vfS','saNGar','kroD','marD','Banj','yuD','mfD','spfD','Bartsa','dfp','DAv','Bad','has','yudD','toqa','tAqa','AkroS','mardan','KaRqa','Dru','vfj','hf','vfk','mardan','KaRqa','Bed','Cid','nAS','mAr','tud','han','ruz','druh','dviz','tyaj','vfS','saNGar','kroD','marD','Banj','yuD','mfD','spfD','Bartsa','dfp','DAv','Bad','has','Cid','Bid','hiMs','mAr','mard','tud','yudD','toq','tAq','KaRqa','AkroS'],
        4: ['Dar','pAl','saMvar','rakz','banD','sTA','ruD','Bf','Df','hf','graha','lab','yam','dA','vezt','ve','saNGAt','yAc','niD','guha','vAra','veza','aveza','vivAs','vas','Di','df','masj','ve','vAr','vf','DA','Dq','varaR','saNGAt','yAc','niD','guha','vAra','veza','aveza','vivAs','vas','Di','df','masj','ve','Dar','pAl','rakz','band','DAn','varaR','guha','vez']
    }
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

# ── High-Fidelity Acoustic Features ───────────────────────────────────────────
def formant_features_partial(root_slp1, max_phonemes=1):
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
    chars = list(root_slp1.lower())
    formants = []
    for c in chars[:max_phonemes]:
        f1, f2 = PHONEME_FORMANTS.get(c, (0.5, 0.5))
        formants.extend([f1, f2])
    while len(formants) < 12: formants.extend([0.0, 0.0])
    return np.array(formants[:12])

def locus_features(root_slp1):
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

def phonation_features(root_slp1):
    c = root_slp1[0].lower() if root_slp1 else ''
    return np.array([float(c in 'KgGjJqQdDbB'), float(c in 'KChWTP'), float(c in 'gGjJqQdDbBmnN')])

def acoustic_features_partial(root_slp1, max_phonemes=1):
    return np.concatenate([locus_features(root_slp1), phonation_features(root_slp1), formant_features_partial(root_slp1, max_phonemes)])

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
    def __init__(self, in_dim, n_neurons=512):
        super().__init__()
        self.L1 = AdExPopulation(size=n_neurons, a=0.0, b=0.0, tau_w=5.0, gL=1.5)
        # Acoustic features are 21D: locus(6) + phonation(3) + formant(12)
        self.proj_acoustic = nn.Parameter(torch.randn(21, n_neurons) * 12000.0 * (torch.rand(21, n_neurons) < 0.35).float())
        self.proj_prior = nn.Parameter(torch.randn(5, n_neurons) * 2000.0 * (torch.rand(5, n_neurons) < 0.35).float())
        self.L2 = AdExPopulation(size=256, a=2.0, b=10.0, tau_w=30.0, gL=1.0)
        self.W12 = nn.Parameter(torch.randn(n_neurons, 256) * 1800.0 * (torch.rand(n_neurons, 256) < 0.35).float())
    def reset(self): self.L1.reset_states(); self.L2.reset_states()
    def step(self, I_in):
        # I_in is 26D: 21D acoustic + 5D prior
        I_acoustic, I_prior = I_in[:, :21], I_in[:, 21:]
        I1 = (I_acoustic @ self.proj_acoustic) + (I_prior @ self.proj_prior)
        sp1 = self.L1.step(I1)
        return self.L2.step(sp1.float() @ self.W12)

# ── Per-seed experiment ────────────────────────────────────────────────────────
def run_experiment(seed, inputs, labels, device, n_neurons=512):
    torch.manual_seed(seed); np.random.seed(seed)
    in_dim = inputs.shape[1]
    model  = MegaSNN(in_dim, n_neurons=n_neurons).to(device)
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
    pred = KMeans(n_clusters=5, n_init=10, random_state=seed).fit_predict(states_norm)
    return adjusted_rand_score(labels, pred)

# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("EXP 53 REDUX: Speed-1 Finding with Honest Labels (Acoustic-only)")
print("=" * 70)

CORPUS_PATH = 'temp_ashtadhyayi_data/dhatu/data.txt'
with open(CORPUS_PATH, encoding='utf-8') as f:
    full_corpus = json.load(f)['data']

base = []
for item in full_corpus:
    root_slp1 = devanagari_to_slp1(item['dhatu'])
    artha_slp1 = devanagari_to_slp1(item['artha'])
    a_tensor = extract_artha_stem_tensor(artha_slp1)
    label = int(np.argmax(a_tensor)) if np.max(a_tensor) > 0.2 else -1
    if label != -1: base.append((root_slp1, a_tensor, label))

print(f"Corpus: {len(base)} labeled roots")

SEEDS = [42, 43, 44]
PHONEME_STEPS = [1, 2, 3, 4, 5, 10]

results = {s: [] for s in SEEDS}
for n_phonemes in PHONEME_STEPS:
    print(f"\nTesting {n_phonemes} phonemes...")
    # Acoustic-only (Uniform 5D prior)
    inputs = np.array([np.concatenate([acoustic_features_partial(r, n_phonemes), np.ones(5)*0.2]) for r, a, _ in base])
    labels = np.array([lbl for _, _, lbl in base])
    
    for seed in SEEDS:
        ari = run_experiment(seed, inputs, labels, device)
        results[seed].append(ari)
        print(f"  seed={seed}: ARI = {ari:.4f}")

print("\n" + "=" * 70)
print("FINAL RESULTS: Speed-1 Integration Curve (Acoustic-only)")
print("=" * 70)
print(f"{'Phonemes':<10} | {'Mean ARI':<10} | {'Std Dev'}")
print("-" * 35)
for i, n in enumerate(PHONEME_STEPS):
    aris = [results[seed][i] for seed in SEEDS]
    print(f"{n:<10} | {np.mean(aris):.4f} | {np.std(aris):.4f}")
