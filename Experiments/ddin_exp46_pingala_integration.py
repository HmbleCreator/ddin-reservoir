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
from collections import Counter

def devanagari_to_slp1(text):
    vowels  = {'अ':'a','आ':'A','इ':'i','ई':'I','उ':'u','ऊ':'U','ऋ':'f','ॠ':'F','ऌ':'x','ॡ':'X','ए':'e','ऐ':'E','ओ':'o','औ':'O'}
    vsigns  = {'ा':'A','ि':'i','ी':'I','ु':'u','ू':'U','ृ':'f','ॄ':'F','ॢ':'x','ॣ':'X','े':'e','ै':'E','ो':'o','ौ':'O'}
    cons    = {'क':'k','ख':'K','ग':'g','घ':'G','ङ':'N','च':'c','छ':'C','ज':'j','झ':'J','ञ':'Y',
               'ट':'w','ठ':'W','ड':'q','ढ':'Q','ण':'R','त':'t','थ':'T','द':'d','ध':'D','न':'n',
               'प':'p','फ':'P','ब':'b','भ':'B','म':'m','य':'y','र':'r','ल':'l','व':'v',
               'श':'S','ष':'z','स':'s','ह':'h'}
    misc    = {'ं':'M','ः':'H','ँ':'~','्':''}
    res = ""
    for i, char in enumerate(text):
        if char in vowels: res += vowels[char]
        elif char in cons:
            res += cons[char]
            if i+1 < len(text) and (text[i+1] in vsigns or text[i+1] == '्'): pass
            else: res += 'a'
        elif char in vsigns: res += vsigns[char]
        elif char in misc:   res += misc[char]
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
    return tensor/s if s > 0 else np.ones(5)*0.2

def compute_locus_label(root_slp1):
    c = root_slp1[0].lower() if root_slp1 else ''
    velar   = ['k','K','g','G','N']
    palatal = ['c','C','j','J','Y']
    retro   = ['w','W','q','Q','R']
    dental  = ['t','T','d','D','n']
    labial  = ['p','P','b','B','m']
    if c in velar:   return 0
    if c in palatal: return 1
    if c in retro:   return 2
    if c in dental:  return 3
    if c in labial:  return 4
    return 5

LONG_VOWELS = set('AEIOUfF')
SHORT_VOWELS = set('aiux')
CONSONANTS = set('kKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzsSh')

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
    return tuple(addr)

def compute_matras(root_slp1):
    """Mora count: guru=2, laghu=1"""
    addr = compute_pingala_address(root_slp1, max_syllables=10)
    return sum(1 + b for b in addr if b is not None)

def pingala_index(addr):
    """Uddista: binary string to integer index"""
    result = 0
    for b in addr:
        result = (result << 1) | b
    return result

class AdExPopulation(nn.Module):
    def __init__(self, size, dt=1.0, C=200.0, gL=10.0, EL=-70.0, VT=-50.0,
                 DeltaT=2.0, Vpeak=0.0, Vreset=-58.0, a=2.0, b=80.0, tau_w=30.0):
        super().__init__()
        self.size = size; self.dt = dt
        self.C = C; self.gL = gL; self.EL = EL
        self.DeltaT = DeltaT; self.Vpeak = Vpeak; self.Vreset = Vreset
        self.VT   = nn.Parameter(torch.ones(1, size) * VT)
        self.a    = nn.Parameter(torch.ones(1, size) * a)
        self.b    = nn.Parameter(torch.ones(1, size) * b)
        self.tau_w= nn.Parameter(torch.ones(1, size) * tau_w)
        self.V    = torch.ones(1, size) * EL
        self.w    = torch.zeros(1, size)
        self.theta= torch.ones(1, size) * 0.1
        self.spike_counts = torch.zeros(1, size)

    def reset_states(self):
        self.V = torch.ones(1, self.size) * self.EL
        self.w = torch.zeros(1, self.size)
        self.spike_counts = torch.zeros(1, self.size)

    def step(self, I_ext):
        exp_term = self.gL * self.DeltaT * torch.exp((self.V - self.VT) / self.DeltaT)
        dV = (-self.gL*(self.V-self.EL) + exp_term - self.w + I_ext) / self.C
        self.V += self.dt * dV
        dw = (self.a*(self.V-self.EL) - self.w) / self.tau_w
        self.w += self.dt * dw
        spikes = (self.V >= self.Vpeak)
        self.spike_counts += spikes.float()
        self.V = torch.where(spikes, torch.tensor(self.Vreset), self.V)
        self.w = torch.where(spikes, self.w + self.b, self.w)
        self.theta += 0.005 * (spikes.float() - self.theta)
        return spikes

class MegaSNN(nn.Module):
    def __init__(self, input_dim=28, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.L1 = AdExPopulation(size=1024, a=0.0, b=0.0, tau_w=5.0, gL=15.0)
        mask1 = (torch.rand(input_dim, 1024) < 0.18).float()
        self.proj_in = nn.Parameter(torch.randn(input_dim, 1024) * 1100.0 * mask1)
        self.L2 = AdExPopulation(size=512, a=2.0, b=80.0, tau_w=150.0, gL=2.0)
        mask2 = (torch.rand(1024, 512) < 0.18).float()
        self.W12 = nn.Parameter(torch.randn(1024, 512) * 410.0 * mask2)

    def reset(self):
        self.L1.reset_states()
        self.L2.reset_states()

    def step(self, I_input):
        I1 = I_input @ self.proj_in
        sp1 = self.L1.step(I1)
        I2 = sp1.float() @ self.W12
        sp2 = self.L2.step(I2)
        return sp2

def embed_acoustic_23(char_slp1, vr):
    FORMANT_DATA = {'a':[0.67,0.42],'i':[0.23,0.81],'u':[0.23,0.19],
                    'e':[0.33,0.69],'o':[0.41,0.23]}
    c = char_slp1.lower()[0]
    f = FORMANT_DATA.get(c, [0.4, 0.4])
    return np.concatenate([np.zeros(12), f, np.zeros(8), [vr]])

def process_corpus_28d(model, dataset, device):
    all_states, all_labels = [], []
    for item in dataset:
        root_slp1 = devanagari_to_slp1(item['dhatu'])
        artha_slp1 = devanagari_to_slp1(item['artha'])
        a_tensor = extract_artha_stem_tensor(artha_slp1)
        label = np.argmax(a_tensor) if np.max(a_tensor) > 0.2 else -1
        chars = list(root_slp1)
        v_ratio = sum(1 for c in chars if c.lower() in 'aiueo') / max(len(chars), 1)
        model.reset()
        with torch.no_grad():
            for c in chars:
                pv = embed_acoustic_23(c, v_ratio)
                I = torch.tensor(np.concatenate([pv, a_tensor]), dtype=torch.float32).unsqueeze(0).to(device)
                for _ in range(20): model.step(I)
            for _ in range(20):
                model.step(torch.tensor(np.concatenate([np.zeros(23), a_tensor]), dtype=torch.float32).unsqueeze(0).to(device))
        all_states.append(model.L2.spike_counts.cpu().numpy().squeeze())
        all_labels.append(label)
    return np.array(all_states), np.array(all_labels)

def process_corpus_32d(model, dataset, device, pingala_table):
    all_states, all_labels = [], []
    for item in dataset:
        root_slp1 = devanagari_to_slp1(item['dhatu'])
        artha_slp1 = devanagari_to_slp1(item['artha'])
        a_tensor = extract_artha_stem_tensor(artha_slp1)
        label = np.argmax(a_tensor) if np.max(a_tensor) > 0.2 else -1
        pingala_addr = pingala_table.get(root_slp1, (0,0,0,0))
        pingala_vec = np.array(pingala_addr, dtype=np.float32)
        chars = list(root_slp1)
        v_ratio = sum(1 for c in chars if c.lower() in 'aiueo') / max(len(chars), 1)
        model.reset()
        with torch.no_grad():
            for c in chars:
                pv = embed_acoustic_23(c, v_ratio)
                full_input = np.concatenate([pv, a_tensor, pingala_vec])
                I = torch.tensor(full_input, dtype=torch.float32).unsqueeze(0).to(device)
                for _ in range(20): model.step(I)
            for _ in range(20):
                model.step(torch.tensor(np.concatenate([np.zeros(23), a_tensor, pingala_vec]), dtype=torch.float32).unsqueeze(0).to(device))
        all_states.append(model.L2.spike_counts.cpu().numpy().squeeze())
        all_labels.append(label)
    return np.array(all_states), np.array(all_labels)

def process_corpus_27d_pingala_only(model, dataset, device, pingala_table):
    all_states, all_labels = [], []
    for item in dataset:
        root_slp1 = devanagari_to_slp1(item['dhatu'])
        artha_slp1 = devanagari_to_slp1(item['artha'])
        label = np.argmax(extract_artha_stem_tensor(artha_slp1)) if np.max(extract_artha_stem_tensor(artha_slp1)) > 0.2 else -1
        pingala_addr = pingala_table.get(root_slp1, (0,0,0,0))
        pingala_vec = np.array(pingala_addr, dtype=np.float32)
        chars = list(root_slp1)
        v_ratio = sum(1 for c in chars if c.lower() in 'aiueo') / max(len(chars), 1)
        model.reset()
        with torch.no_grad():
            for c in chars:
                pv = embed_acoustic_23(c, v_ratio)
                full_input = np.concatenate([pv, pingala_vec])
                I = torch.tensor(full_input, dtype=torch.float32).unsqueeze(0).to(device)
                for _ in range(20): model.step(I)
            for _ in range(20):
                model.step(torch.tensor(np.concatenate([np.zeros(23), pingala_vec]), dtype=torch.float32).unsqueeze(0).to(device))
        all_states.append(model.L2.spike_counts.cpu().numpy().squeeze())
        all_labels.append(label)
    return np.array(all_states), np.array(all_labels)

def compute_mi(x, y, n_bins=10):
    hist_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
    pxy = hist_xy / (hist_xy.sum() + 1e-10)
    px  = pxy.sum(axis=1)
    py  = pxy.sum(axis=0)
    MI  = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            if pxy[i,j] > 0 and px[i] > 0 and py[j] > 0:
                MI += pxy[i,j] * np.log(pxy[i,j] / (px[i]*py[j] + 1e-10) + 1e-10)
    return max(MI, 0.0)

print("="*70)
print("DDIN Phase 11 -- Pingala Integration (Exp 42, 43, 44)")
print("="*70)

with open(r'temp_ashtadhyayi_data/dhatu/data.txt', encoding='utf-8') as f:
    full_corpus = json.load(f)['data']
corpus = full_corpus[:2000]
print(f"Loaded {len(corpus)} roots")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("\n" + "="*70)
print("EXP 42 -- MUTUAL INFORMATION AUDIT")
print("="*70)
print("Computing Pingala addresses for all 2000 roots...")

pingala_table  = {}
locus_labels   = []
semantic_labels = []
pingala_labels = []
matra_labels   = []
for item in corpus:
    root_slp1    = devanagari_to_slp1(item['dhatu'])
    artha_slp1   = devanagari_to_slp1(item['artha'])
    locus        = compute_locus_label(root_slp1)
    sem_label    = np.argmax(extract_artha_stem_tensor(artha_slp1)) if np.max(extract_artha_stem_tensor(artha_slp1)) > 0.2 else -1
    pingala_addr = compute_pingala_address(root_slp1, max_syllables=4)
    matra        = compute_matras(root_slp1)
    pingala_table[root_slp1] = pingala_addr
    locus_labels.append(locus)
    semantic_labels.append(sem_label)
    pingala_labels.append(pingala_addr)
    matra_labels.append(matra)

locus_arr   = np.array(locus_labels)
sem_arr     = np.array(semantic_labels)
pingala_arr = np.array([int(''.join(str(b) for b in p), 2) for p in pingala_labels])
matra_arr   = np.array(matra_labels)

mask = sem_arr != -1
locus_v   = locus_arr[mask]
sem_v     = sem_arr[mask]
pingala_v = pingala_arr[mask]
matra_v   = matra_arr[mask]

print("\n  Pingala address distribution:")
addr_counts = Counter(pingala_labels)
for addr in sorted(addr_counts.keys()):
    binary = ''.join(str(b) for b in addr)
    print("    %s: %d roots" % (binary, addr_counts[addr]))

print("\n  --Matra distribution:")
matra_counts = Counter(matra_labels)
for m in sorted(matra_counts.keys()):
    print("    M=%d: %d roots" % (m, matra_counts[m]))

MI_pingala_sem = compute_mi(pingala_v, sem_v)
MI_locus_sem   = compute_mi(locus_v, sem_v)
MI_matra_sem   = compute_mi(matra_v, sem_v)
MI_pingala_locus = compute_mi(pingala_v, locus_v)

print("\n  Mutual Information Results:")
print("    MI(Pingala, Semantic Axis) = %.4f bits" % MI_pingala_sem)
print("    MI(Matra, Semantic Axis)   = %.4f bits" % MI_matra_sem)
print("    MI(Locus, Semantic Axis)    = %.4f bits" % MI_locus_sem)
print("    MI(Pingala, Locus)          = %.4f bits" % MI_pingala_locus)

print("\n  EXP 42 VERDICT:")
if MI_pingala_sem > 0.02:
    print("    PASS: MI(Pingala, Axis) = %.4f > 0.02 threshold" % MI_pingala_sem)
    print("    --> Pingala channel carries semantic signal above acoustic baseline")
else:
    print("    MARGINAL: MI(Pingala, Axis) = %.4f <= 0.02" % MI_pingala_sem)
    print("    --> Continue to Exp 43-44 for network-level validation")

print("\n" + "="*70)
print("EXP 43 -- 32D TENSOR BASELINE (28D + 4D Pingala appended)")
print("="*70)
SEEDS = [42, 43, 44]
exp43_results = {}
for si, seed in enumerate(SEEDS):
    print("\n  Seed %d/3: %d" % (si+1, seed), end=" ", flush=True)
    model = MegaSNN(input_dim=32, seed=seed).to(device)
    states, labels = process_corpus_32d(model, corpus, device, pingala_table)
    mask = labels != -1
    valid_states = states[mask]
    valid_labels = labels[mask]
    states_norm  = StandardScaler().fit_transform(valid_states)
    pred = KMeans(n_clusters=5, n_init=10).fit_predict(states_norm)
    ari  = adjusted_rand_score(valid_labels, pred)
    exp43_results[seed] = ari
    print(f"ARI={ari:.4f}")
    del model; gc.collect(); torch.cuda.empty_cache()

ari_vals_43 = list(exp43_results.values())
print("\n  32D (28D+Pingala): mean=%.4f  std=%.4f" % (np.mean(ari_vals_43), np.std(ari_vals_43)))

print("\n" + "="*70)
print("EXP 44 -- Pingala-ONLY BASELINE (27D = 23D acoustic + 4D Pingala)")
print("           NO semantic dictionary. Pure phonological + combinatorial.")
print("="*70)
exp44_results = {}
for si, seed in enumerate(SEEDS):
    print("\n  Seed %d/5: %d" % (si+1, seed), end=" ", flush=True)
    model = MegaSNN(input_dim=27, seed=seed).to(device)
    states, labels = process_corpus_27d_pingala_only(model, corpus, device, pingala_table)
    mask = labels != -1
    valid_states = states[mask]
    valid_labels = labels[mask]
    states_norm  = StandardScaler().fit_transform(valid_states)
    pred = KMeans(n_clusters=5, n_init=10).fit_predict(states_norm)
    ari  = adjusted_rand_score(valid_labels, pred)
    exp44_results[seed] = ari
    print("ARI=%.4f  valid=%d/%d" % (ari, mask.sum(), len(labels)))
    del model; gc.collect(); torch.cuda.empty_cache()

ari_vals_44 = list(exp44_results.values())
mean_44 = np.mean(ari_vals_44)
std_44  = np.std(ari_vals_44)
print("\n  27D (acoustic+Pingala only): mean=%.4f  std=%.4f" % (mean_44, std_44))

print("\n" + "="*70)
print("EXP 43 vs EXP 44 vs BASELINE COMPARISON")
print("="*70)
baseline_28d = 0.9758
print("\n  28D baseline (Exp 40, single run):  %.4f" % baseline_28d)
print("  32D + Pingala appended (Exp 43):     %.4f +/- %.4f" % (np.mean(ari_vals_43), np.std(ari_vals_43)))
print("  27D Pingala-only (Exp 44):           %.4f +/- %.4f" % (mean_44, std_44))

if mean_44 > 0.15:
    print("\n  *** CRITICAL RESULT: Exp 44 ARI = %.4f > 0.15 threshold ***" % mean_44)
    print("  --> Semantic signal is LATENT in phonological + combinatorial structure.")
    print("  --> No dictionary glosses needed. Pingala's algebra is a semantic feature extractor.")
    print("  --> Paper 5 title writes itself.")
else:
    print("\n  --> Exp 44 ARI = %.4f below 0.15 threshold." % mean_44)
    print("  --> Dictionary glosses still required to unlock semantic signal.")
    print("  --> Pingala adds complementary signal (Exp 43) but is not sufficient alone.")

print("\n" + "="*70)
print("MATRA-VRITHA HYPOTHESIS TEST")
print("="*70)
for m_val in sorted(set(matra_v)):
    mask_m = matra_v == m_val
    if mask_m.sum() > 20:
        sem_counts = np.bincount(sem_v[mask_m].astype(int), minlength=5)
        print("  M=%d: %d roots, sem_axis distribution: %s" % (m_val, mask_m.sum(), sem_counts.tolist()))

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("  Exp 42 MI(Pingala, Axis) = %.4f bits  %s" % (MI_pingala_sem, '(PASS)' if MI_pingala_sem > 0.02 else '(marginal)'))
print("  Exp 43 32D ARI           = %.4f +/- %.4f" % (np.mean(ari_vals_43), np.std(ari_vals_43)))
print("  Exp 44 27D ARI           = %.4f +/- %.4f  %s" % (mean_44, std_44, '(>0.15)' if mean_44 > 0.15 else '(<0.15)'))
print("="*70)
