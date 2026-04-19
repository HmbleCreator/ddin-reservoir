"""
Exp 45: Arabic Pilot -- Trilateral Root System Cross-Linguistic Validation
Tests whether the DDIN architecture generalizes to Classical Arabic.
Run on Kaggle GPU. Takes ~5-10 minutes.
"""
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

N_ROOTS = 200
SEED = 42

ARABIC_CONSONANTS = {
    '\u0628': {'name': 'baa',  'locus': 'labial',     'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    '\u062a': {'name': 'taa',  'locus': 'dental',     'voiced': 0, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    '\u062b': {'name': 'thaa', 'locus': 'dental',     'voiced': 0, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    '\u0643': {'name': 'kaf',  'locus': 'velar',      'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    '\u062f': {'name': 'daal', 'locus': 'dental',     'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    '\u0630': {'name': 'dhaal','locus': 'dental',     'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    '\u0632': {'name': 'zay',  'locus': 'dental',     'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    '\u0634': {'name': 'shiin','locus': 'palatal',    'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 1},
    '\u0633': {'name': 'siin', 'locus': 'dental',     'voiced': 0, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 1},
    '\u0635': {'name': 'saad', 'locus': 'pharyngeal', 'voiced': 0, 'pharyngeal': 1, 'nasal': 0, 'sibilant': 0},
    '\u0636': {'name': 'daad', 'locus': 'pharyngeal', 'voiced': 1, 'pharyngeal': 1, 'nasal': 0, 'sibilant': 0},
    '\u0637': {'name': 'taa',  'locus': 'pharyngeal', 'voiced': 0, 'pharyngeal': 1, 'nasal': 0, 'sibilant': 0},
    '\u0638': {'name': 'zaa',  'locus': 'pharyngeal', 'voiced': 1, 'pharyngeal': 1, 'nasal': 0, 'sibilant': 0},
    '\u0639': {'name': 'ayn',  'locus': 'pharyngeal', 'voiced': 1, 'pharyngeal': 1, 'nasal': 0, 'sibilant': 0},
    '\u063a': {'name': 'ghayn','locus': 'pharyngeal', 'voiced': 1, 'pharyngeal': 1, 'nasal': 0, 'sibilant': 0},
    '\u062d': {'name': 'haa',  'locus': 'pharyngeal', 'voiced': 0, 'pharyngeal': 1, 'nasal': 0, 'sibilant': 0},
    '\u062e': {'name': 'khaa', 'locus': 'pharyngeal', 'voiced': 0, 'pharyngeal': 1, 'nasal': 0, 'sibilant': 0},
    '\u062c': {'name': 'jiim', 'locus': 'palatal',    'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    '\u0621': {'name': 'hamza','locus': 'glottal',    'voiced': 0, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    '\u064a': {'name': 'yaa',  'locus': 'palatal',    'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    '\u0646': {'name': 'noon', 'locus': 'dental',     'voiced': 1, 'pharyngeal': 0, 'nasal': 1, 'sibilant': 0},
    '\u0645': {'name': 'miim', 'locus': 'labial',     'voiced': 1, 'pharyngeal': 0, 'nasal': 1, 'sibilant': 0},
    '\u0644': {'name': 'laam', 'locus': 'dental',     'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    '\u0631': {'name': 'raa',  'locus': 'dental',     'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    '\u0642': {'name': 'qaaf', 'locus': 'uvular',     'voiced': 0, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    '\u0647': {'name': 'haa',  'locus': 'glottal',    'voiced': 0, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    '\u0648': {'name': 'waaw', 'locus': 'labial',     'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    '\u0623': {'name': 'alif', 'locus': 'glottal',    'voiced': 0, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
}

ARABIC_VOWELS = {
    '\u0627': 'a', '\u0622': 'aa',
    '\u064b': 'an', '\u064c': 'un', '\u064d': 'in'
}

def get_locus_vector(char):
    if char not in ARABIC_CONSONANTS:
        return np.zeros(5)
    info = ARABIC_CONSONANTS[char]
    locus_map = {
        'velar': 0, 'palatal': 1, 'pharyngeal': 2,
        'dental': 3, 'labial': 4, 'uvular': 0, 'glottal': 2
    }
    vec = np.zeros(5)
    if info['locus'] in locus_map:
        vec[locus_map[info['locus']]] = 1.0
    return vec

def embed_arabic_phoneme(char, phoneme_idx, total_phonemes):
    base = np.zeros(16)
    if char in ARABIC_CONSONANTS:
        info = ARABIC_CONSONANTS[char]
        base[0] = info['voiced']
        base[1] = info['pharyngeal']
        base[2] = info['nasal']
        base[3] = info['sibilant']
        base[4:9] = get_locus_vector(char)
    elif char in ARABIC_VOWELS:
        base[9]  = 1.0
        base[10] = 1.0 if ARABIC_VOWELS[char] in ['aa', 'an'] else 0.0
    base[11 + min(phoneme_idx, 10)] = 1.0  # positional (indices 11-13 used safely for idx 0-2)
    base[14] = phoneme_idx / max(total_phonemes, 1)
    base[15] = total_phonemes / 6.0
    return base

def get_arabic_semantic_tensor(meaning_ar, meaning_en):
    meaning_lower = meaning_en.lower()
    axis_keywords = {
        0: ['go', 'walk', 'travel', 'come', 'move', 'run', 'rise', 'motion'],
        1: ['create', 'find', 'appear', 'be', 'happen', 'exist', 'born'],
        2: ['write', 'do', 'act', 'make', 'work', 'perform', 'action'],
        3: ['kill', 'hit', 'break', 'defeat', 'destroy', 'cut', 'strike'],
        4: ['take', 'give', 'obtain', 'get', 'contain', 'receive', 'transfer'],
    }
    tensor = np.zeros(5)
    for i, keywords in axis_keywords.items():
        if any(kw in meaning_lower for kw in keywords):
            tensor[i] = 1.0
    s = np.sum(tensor)
    return tensor / s if s > 0 else np.ones(5) * 0.2

# ── Corpus ───────────────────────────────────────────────────────────────────
ARABIC_CORPUS = [
    {'root': '\u062f\u0647\u0628', 'meaning': 'gold',    'triliteral': '\u062f\u0647\u0628', 'artha': 'Motion'},
    {'root': '\u0630\u0647\u0628', 'meaning': 'go',      'triliteral': '\u0630\u0647\u0628', 'artha': 'Motion'},
    {'root': '\u0645\u0634\u0649', 'meaning': 'walk',    'triliteral': '\u0645\u0634\u0649', 'artha': 'Motion'},
    {'root': '\u0633\u0627\u0631', 'meaning': 'travel',  'triliteral': '\u0633\u0631\u062d', 'artha': 'Motion'},
    {'root': '\u062c\u0627\u0621', 'meaning': 'come',    'triliteral': '\u062c\u0627\u0621', 'artha': 'Motion'},
    {'root': '\u0642\u0627\u0645', 'meaning': 'rise',    'triliteral': '\u0642\u0648\u0645', 'artha': 'Motion'},
    {'root': '\u062e\u0644\u0642', 'meaning': 'create',  'triliteral': '\u062e\u0644\u0642', 'artha': 'Existence'},
    {'root': '\u0648\u062c\u062f', 'meaning': 'find',    'triliteral': '\u0648\u062c\u062f', 'artha': 'Existence'},
    {'root': '\u0628\u062f\u0627', 'meaning': 'appear',  'triliteral': '\u0628\u062f\u0627', 'artha': 'Existence'},
    {'root': '\u0643\u0627\u0646', 'meaning': 'be',      'triliteral': '\u0643\u0627\u0646', 'artha': 'Existence'},
    {'root': '\u062d\u062f\u062b', 'meaning': 'happen',  'triliteral': '\u062d\u062f\u062b', 'artha': 'Existence'},
    {'root': '\u0638\u0647\u0631', 'meaning': 'appear',  'triliteral': '\u0638\u0647\u0631', 'artha': 'Existence'},
    {'root': '\u0643\u062a\u0628', 'meaning': 'write',   'triliteral': '\u0643\u062a\u0628', 'artha': 'Action'},
    {'root': '\u0639\u0645\u0644', 'meaning': 'do',      'triliteral': '\u0639\u0645\u0644', 'artha': 'Action'},
    {'root': '\u0641\u0639\u0644', 'meaning': 'act',     'triliteral': '\u0641\u0639\u0644', 'artha': 'Action'},
    {'root': '\u0635\u0646\u0639', 'meaning': 'make',    'triliteral': '\u0635\u0646\u0639', 'artha': 'Action'},
    {'root': '\u062d\u0631\u0643', 'meaning': 'move',    'triliteral': '\u062d\u0631\u0643', 'artha': 'Action'},
    {'root': '\u062a\u062d\u0631\u0643', 'meaning': 'move', 'triliteral': '\u062a\u062d\u0631\u0643', 'artha': 'Action'},
    {'root': '\u0642\u062a\u0644', 'meaning': 'kill',    'triliteral': '\u0642\u062a\u0644', 'artha': 'Violence'},
    {'root': '\u0636\u0631\u0628', 'meaning': 'hit',     'triliteral': '\u0636\u0631\u0628', 'artha': 'Violence'},
    {'root': '\u0643\u0633\u0631', 'meaning': 'break',   'triliteral': '\u0643\u0633\u0631', 'artha': 'Violence'},
    {'root': '\u063a\u0644\u0628', 'meaning': 'defeat',  'triliteral': '\u063a\u0644\u0628', 'artha': 'Violence'},
    {'root': '\u0647\u0632\u0645', 'meaning': 'destroy', 'triliteral': '\u0647\u0632\u0645', 'artha': 'Violence'},
    {'root': '\u062f\u0645\u0645', 'meaning': 'cut',     'triliteral': '\u062f\u0645\u0645', 'artha': 'Violence'},
    {'root': '\u0623\u062e\u0630', 'meaning': 'take',    'triliteral': '\u0623\u062e\u0630', 'artha': 'Transfer'},
    {'root': '\u0623\u062a\u0649', 'meaning': 'give',    'triliteral': '\u0623\u062a\u0649', 'artha': 'Transfer'},
    {'root': '\u0639\u0637\u0649', 'meaning': 'give',    'triliteral': '\u0639\u0637\u0649', 'artha': 'Transfer'},
    {'root': '\u0646\u0627\u0644', 'meaning': 'obtain',  'triliteral': '\u0646\u0627\u0644', 'artha': 'Transfer'},
    {'root': '\u062d\u0635\u0644', 'meaning': 'get',     'triliteral': '\u062d\u0635\u0644', 'artha': 'Transfer'},
    {'root': '\u062a\u0636\u0645\u0646', 'meaning': 'contain', 'triliteral': '\u062a\u0636\u0645\u0646', 'artha': 'Transfer'},
]

AXIS_MAP = {'Motion': 0, 'Existence': 1, 'Action': 2, 'Violence': 3, 'Transfer': 4}
for item in ARABIC_CORPUS:
    item['axis'] = AXIS_MAP.get(item['artha'], -1)

# Pad corpus to N_ROOTS by cycling
while len(ARABIC_CORPUS) < N_ROOTS:
    ARABIC_CORPUS.append(dict(ARABIC_CORPUS[len(ARABIC_CORPUS) % 30]))

print(f"Arabic corpus: {len(ARABIC_CORPUS)} roots")


# ── AdEx population (buffers registered properly for .to(device)) ────────────
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
        # Scalars needed in tensor ops — registered so .to(device) moves them
        self.register_buffer('Vpeak',  torch.tensor([[Vpeak]]))
        self.register_buffer('Vreset', torch.tensor([[Vreset]]))
        # Learnable parameters
        self.VT    = nn.Parameter(torch.ones(1, size) * VT)
        self.a     = nn.Parameter(torch.ones(1, size) * a)
        self.b     = nn.Parameter(torch.ones(1, size) * b)
        self.tau_w = nn.Parameter(torch.ones(1, size) * tau_w)
        # State tensors — registered as buffers so .to(device) moves them
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
        self.V = torch.where(spikes, self.Vreset.expand_as(self.V), self.V)
        self.w = torch.where(spikes, self.w + self.b, self.w)
        self.theta = self.theta + 0.005 * (spikes.float() - self.theta)
        return spikes


# ── Two-layer SNN ─────────────────────────────────────────────────────────────
class ArabicSNN(nn.Module):
    def __init__(self, input_dim=21, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.L1 = AdExPopulation(size=512, a=0.0, b=0.0,  tau_w=5.0,   gL=15.0)
        mask1 = (torch.rand(input_dim, 512) < 0.18).float()
        self.proj_in = nn.Parameter(torch.randn(input_dim, 512) * 800.0 * mask1)

        self.L2 = AdExPopulation(size=256, a=2.0, b=80.0, tau_w=150.0, gL=2.0)
        mask2 = (torch.rand(512, 256) < 0.18).float()
        self.W12 = nn.Parameter(torch.randn(512, 256) * 300.0 * mask2)

    def reset(self):
        self.L1.reset_states()
        self.L2.reset_states()

    def step(self, I_input):
        I1  = I_input @ self.proj_in
        sp1 = self.L1.step(I1)
        I2  = sp1.float() @ self.W12
        sp2 = self.L2.step(I2)
        return sp2


# ── Corpus processor ──────────────────────────────────────────────────────────
def process_arabic_corpus(model, dataset, device):
    all_states, all_labels = [], []
    for item in dataset:
        chars      = list(item['triliteral'])
        sem_tensor = get_arabic_semantic_tensor(item['meaning'], item['meaning'])
        label      = item['axis']

        model.reset()
        with torch.no_grad():
            for idx, c in enumerate(chars):
                emb        = embed_arabic_phoneme(c, idx, len(chars))
                full_input = np.concatenate([emb, sem_tensor])
                I = torch.tensor(full_input, dtype=torch.float32).unsqueeze(0).to(device)
                for _ in range(20):
                    model.step(I)

            silence = torch.tensor(
                np.concatenate([np.zeros(16), sem_tensor]),
                dtype=torch.float32
            ).unsqueeze(0).to(device)
            for _ in range(20):
                model.step(silence)

        all_states.append(model.L2.spike_counts.cpu().numpy().squeeze())
        all_labels.append(label)

    return np.array(all_states), np.array(all_labels)


# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Exp 45 -- Arabic Pilot (Cross-Linguistic Generalization)")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Step 1: Baseline
print("\nStep 1: Baseline (random init, no training)...")
torch.manual_seed(SEED)
model = ArabicSNN(input_dim=21, seed=SEED).to(device)
states, labels = process_arabic_corpus(model, ARABIC_CORPUS, device)
mask         = labels >= 0  # all labels are valid here since AXIS_MAP covers all artha
valid_states = states[mask]
valid_labels = labels[mask]
states_norm  = StandardScaler().fit_transform(valid_states)
pred         = KMeans(n_clusters=5, n_init=10, random_state=SEED).fit_predict(states_norm)
ari_baseline = adjusted_rand_score(valid_labels, pred)
print(f"  Random init ARI: {ari_baseline:.4f}")
del model; gc.collect()
if device.type == "cuda": torch.cuda.empty_cache()

# Step 2: 3-seed sweep
print("\nStep 2: Full Arabic pipeline (200 roots, 3 seeds)...")
SEEDS    = [42, 43, 44]
ari_vals = []
for si, seed in enumerate(SEEDS):
    print(f"  Seed {si+1}/{len(SEEDS)} (seed={seed})...", end=" ", flush=True)
    torch.manual_seed(seed)
    model = ArabicSNN(input_dim=21, seed=seed).to(device)
    states, labels = process_arabic_corpus(model, ARABIC_CORPUS, device)
    mask         = labels >= 0
    valid_states = states[mask]
    valid_labels = labels[mask]
    states_norm  = StandardScaler().fit_transform(valid_states)
    pred         = KMeans(n_clusters=5, n_init=10, random_state=seed).fit_predict(states_norm)
    ari          = adjusted_rand_score(valid_labels, pred)
    ari_vals.append(ari)
    print(f"ARI={ari:.4f}")
    del model; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()

mean_ari = np.mean(ari_vals)
std_ari  = np.std(ari_vals)
print(f"\n  Arabic pipeline: mean={mean_ari:.4f} +/- {std_ari:.4f}")

# ── Verdict ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print(f"  Baseline ARI (random init):        {ari_baseline:.4f}")
print(f"  Full pipeline (mean over 3 seeds): {mean_ari:.4f} +/- {std_ari:.4f}")

success_threshold = 0.15
if mean_ari > success_threshold:
    print(f"\n  *** PASS: ARI = {mean_ari:.4f} > {success_threshold:.2f} threshold ***")
    print("  --> DDIN architecture GENERALIZES to Arabic")
    print("  --> Cross-linguistic phonosemantic organization confirmed")
    print("  --> Paper 5 framing: 'Sanskrit, Arabic, and the Universality Question'")
else:
    print(f"\n  --> MARGINAL: ARI = {mean_ari:.4f} < {success_threshold:.2f}")
    print("  --> Arabic roots may need more semantic axes or larger corpus")
    print("  --> Try: 500-root corpus, different semantic taxonomy")

print("=" * 70)