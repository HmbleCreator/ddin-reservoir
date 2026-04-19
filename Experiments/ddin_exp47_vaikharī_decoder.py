"""
Exp 47: VaikharI Decoder -- Generative Decoding from Attractor States
Trains a decoder MLP: reservoir_state (512D) -> acoustic_features (23D)
Then samples from each semantic attractor basin and decodes to phoneme sequences.
Human evaluation: 3 linguists judge whether decoded sequences feel semantically appropriate.
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

N_ROOTS = 2000
D_RESERVOIR = 512
LATENT_DIM = 64
SEED = 42

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
    FORMANT_DATA = {'a':[0.67, 0.42], 'i':[0.23, 0.81], 'u':[0.23, 0.19], 'e':[0.33, 0.69], 'o':[0.41, 0.23]}
    c = char_slp1.lower()[0]
    f = FORMANT_DATA.get(c, [0.4, 0.4])
    return np.concatenate([np.zeros(12), f, np.zeros(8), [vr]])

def process_corpus(model, dataset, device):
    all_states, all_labels = [], []
    for item in dataset:
        root_slp1    = devanagari_to_slp1(item['dhatu'])
        artha_slp1   = devanagari_to_slp1(item['artha'])
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

class DecoderMLP(nn.Module):
    def __init__(self, reservoir_dim=512, latent_dim=64, output_dim=23):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(reservoir_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)

def acoustic_to_phoneme(vec, centroid_map):
    cos_sim = np.dot(centroid_map, vec) / (np.linalg.norm(centroid_map, axis=1) * np.linalg.norm(vec) + 1e-10)
    best = np.argmax(cos_sim)
    return best, cos_sim[best]

print("="*70)
print("Exp 47 -- VaikharI Decoder (Generative from Attractor States)")
print("="*70)

with open(r'temp_ashtadhyayi_data/dhatu/data.txt', encoding='utf-8') as f:
    raw_data = json.load(f)
corpus = raw_data['data'][:N_ROOTS]
device = torch.device("cpu")
print("Device: %s" % device)
print("Corpus: %d roots" % len(corpus))

print("\nStep 1: Collect reservoir states from trained network...")
torch.manual_seed(SEED)
model = MegaSNN(input_dim=28, seed=SEED).to(device)
states, labels = process_corpus(model, corpus, device)
mask = labels != -1
valid_states = states[mask]
valid_labels = labels[mask]
states_norm = StandardScaler().fit_transform(valid_states)
pred = KMeans(n_clusters=5, n_init=10).fit_predict(states_norm)
ari = adjusted_rand_score(valid_labels, pred)
print("  Reservoir ARI: %.4f" % ari)
print("  Valid roots: %d" % mask.sum())

print("\nStep 2: Train decoder MLP (512D -> 23D acoustic reconstruction)...")
decoder = DecoderMLP(reservoir_dim=512, latent_dim=LATENT_DIM, output_dim=23).to(device)
optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
criterion = nn.MSELoss()

X_train = torch.tensor(states_norm, dtype=torch.float32)
y_train = torch.zeros(len(corpus), 23)
idx = 0
for i, item in enumerate(corpus):
    if labels[i] == -1:
        continue
    root_slp1  = devanagari_to_slp1(item['dhatu'])
    chars = list(root_slp1)
    v_ratio = sum(1 for c in chars if c.lower() in 'aiueo') / max(len(chars), 1)
    pv_full = np.zeros(23)
    for c in chars:
        pv_full += embed_acoustic_23(c, v_ratio) / max(len(chars), 1)
    y_train[idx] = torch.tensor(pv_full, dtype=torch.float32)
    idx += 1

valid_mask = labels != -1
X_train = torch.tensor(states_norm, dtype=torch.float32)
y_train_t = y_train[:len(X_train)]

print("  Training decoder for 200 epochs...")
for epoch in range(200):
    decoder.train()
    pred_out = decoder(X_train)
    loss = criterion(pred_out, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print("  Epoch %d: loss=%.6f" % (epoch+1, loss.item()))

print("\nStep 3: Compute attractor basin centroids...")
kmeans = KMeans(n_clusters=5, n_init=10).fit(states_norm)
centroids = kmeans.cluster_centers_

print("\nStep 4: Generate 10 decoded sequences per semantic axis...")
print("  (Sampling from each attractor basin and decoding to phonemes)")
print()

AXIS_NAMES = ['MOT', 'SEP', 'CNV', 'EXP', 'TRN']
AXIS_EXAMPLES = {
    0: ['gam (go)', 'cal (move)', 'pad (fall)', 'sarp (crawl)'],
    1: ['jan (be born)', 'utpat (sprout)', 'dIpt (shine)', 'jIv (live)'],
    2: ['pAk (cook)', 'kf (do)', 'kriy (act)', 'saMsk (adorn)'],
    3: ['hiMs (injure)', 'Cit (notice)', 'nAS (destroy)', 'mAr (strike)'],
    4: ['DAr (hold)', 'pAl (protect)', 'rakz (protect)', 'banD (bind)']
}

generated_roots = {}
for axis in range(5):
    print("  Axis %d (%s):" % (axis, AXIS_NAMES[axis]))
    print("    Natural examples: %s" % ', '.join(AXIS_EXAMPLES[axis][:3]))

    mask_axis = (pred == axis)
    if mask_axis.sum() < 2:
        print("    [Too few points in basin -- skipping]")
        continue

    axis_states = states_norm[mask_axis]
    n_samples = min(10, len(axis_states))
    sampled_idx = np.random.choice(len(axis_states), n_samples, replace=False)
    sampled_states = axis_states[sampled_idx]

    decoder.eval()
    decoded_vectors = decoder(torch.tensor(sampled_states, dtype=torch.float32).to(device)).detach().cpu().numpy()

    print("    Generated sequences (acoustic reconstruction -> phoneme):")
    for j, dec_vec in enumerate(decoded_vectors):
        vr_guess = dec_vec[22]
        f1_guess = dec_vec[12]
        f2_guess = dec_vec[13]
        best_vowel = min(range(5), key=lambda k: abs([0.67, 0.23, 0.23, 0.33, 0.41][k] - f1_guess) + abs([0.42, 0.81, 0.19, 0.69, 0.23][k] - f2_guess))
        vowel_chars = ['a', 'i', 'u', 'e', 'o']
        vr_char = 'a' if vr_guess < 0.5 else 'A'
        print("      [%d] F1=%.2f F2=%.2f VR=%.2f -> v=%s r=%s" % (
            j+1, f1_guess, f2_guess, vr_guess, vowel_chars[best_vowel], vr_char))

print("\n" + "="*70)
print("HUMAN EVALUATION PROTOCOL")
print("="*70)
print("""
  Blind evaluation: Show 3 linguists the 50 generated sequences (10 per axis)
  without axis labels. Ask: "Does this phonological sequence feel consistent
  with the semantic class [MOTION/SEPARATION/etc.]?"

  Scoring:
    - If agreement > 60%% above chance: decoder is generating semantically
      appropriate phonological forms
    - If agreement < 60%%: decoder is not producing meaningful output

  Success criterion: mean agreement > 60%% across axes
""")
print("="*70)
print("\nDecoder architecture:")
print(decoder)
print("\nTotal parameters: %d" % sum(p.numel() for p in decoder.parameters()))
print("="*70)
