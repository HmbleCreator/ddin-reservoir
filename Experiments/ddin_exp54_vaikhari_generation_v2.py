"""
DDIN Exp 54 v2 -- Vaikharī Generation & Piṅgala Completion
Phase 12: Transform DDIN from classifier to generator
Key insight: Use Piṅgala as generative constraint, not input feature
Self-contained: generates attractor states from scratch

Pipeline:
  Attractor state → MLP decoder → 23D acoustic → Piṅgala completion → Full root

Run on Colab. Takes ~15-20 minutes.
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

N_BASINS = 5
N_SAMPLES_PER_BASIN = 20
TOTAL_GENERATED = 100

SEEDS = [42, 43, 44]

N_ROOTS = 2000

DHATU_CORPUS = [
    {'root': 'gam', 'slp1': 'gam', 'artha': 'Motion'},
    {'root': 'gam', 'slp1': 'gam', 'artha': 'Motion'},
    {'root': 'gachCh', 'slp1': 'gam', 'artha': 'Motion'},
    {'root': 'cal', 'slp1': 'cal', 'artha': 'Motion'},
    {'root': 'pad', 'slp1': 'pad', 'artha': 'Motion'},
    {'root': 'kram', 'slp1': 'kram', 'artha': 'Motion'},
    {'root': 'car', 'slp1': 'car', 'artha': 'Motion'},
    {'root': 'sarp', 'slp1': 'sarp', 'artha': 'Motion'},
    {'root': 'i', 'slp1': 'i', 'artha': 'Motion'},
    {'root': 'e', 'slp1': 'e', 'artha': 'Motion'},
    {'root': 'vI', 'slp1': 'vI', 'artha': 'Motion'},
    {'root': 'vR', 'slp1': 'vR', 'artha': 'Motion'},
    {'root': 'jan', 'slp1': 'jan', 'artha': 'Existence'},
    {'root': 'jI', 'slp1': 'jI', 'artha': 'Existence'},
    {'root': 'sthA', 'slp1': 'sTA', 'artha': 'Existence'},
    {'root': 'As', 'slp1': 'As', 'artha': 'Existence'},
    {'root': 'BU', 'slp1': 'BU', 'artha': 'Existence'},
    {'root': 'vid', 'slp1': 'vid', 'artha': 'Existence'},
    {'root': 'rud', 'slp1': 'rud', 'artha': 'Existence'},
    {'root': 'mad', 'slp1': 'mad', 'artha': 'Existence'},
    {'root': 'kR', 'slp1': 'kR', 'artha': 'Action'},
    {'root': 'kAr', 'slp1': 'kAr', 'artha': 'Action'},
    {'root': 'kF', 'slp1': 'kF', 'artha': 'Action'},
    {'root': 'viDA', 'slp1': 'viBA', 'artha': 'Action'},
    {'root': 'sAm', 'slp1': 'sAm', 'artha': 'Action'},
    {'root': 'ci', 'slp1': 'ci', 'artha': 'Action'},
    {'root': 'li', 'slp1': 'li', 'artha': 'Action'},
    {'root': 'ri', 'slp1': 'ri', 'artha': 'Action'},
    {'root': 'tan', 'slp1': 'tan', 'artha': 'Action'},
    {'root': 'tan', 'slp1': 'tan', 'artha': 'Action'},
    {'root': 'han', 'slp1': 'han', 'artha': 'Violence'},
    {'root': 'hiMs', 'slp1': 'hiMs', 'artha': 'Violence'},
    {'root': 'Cid', 'slp1': 'Cid', 'artha': 'Violence'},
    {'root': 'nAS', 'slp1': 'nAS', 'artha': 'Violence'},
    {'root': 'mAr', 'slp1': 'mAr', 'artha': 'Violence'},
    {'root': 'tuj', 'slp1': 'tuj', 'artha': 'Violence'},
    {'root': 'druh', 'slp1': 'druh', 'artha': 'Violence'},
    {'root': 'daMS', 'slp1': 'daMS', 'artha': 'Violence'},
    {'root': 'pU', 'slp1': 'pU', 'artha': 'Transfer'},
    {'root': 'pAl', 'slp1': 'pAl', 'artha': 'Transfer'},
    {'root': 'rA', 'slp1': 'rA', 'artha': 'Transfer'},
    {'root': 'grah', 'slp1': 'grah', 'artha': 'Transfer'},
    {'root': 'lab', 'slp1': 'lab', 'artha': 'Transfer'},
    {'root': 'rad', 'slp1': 'rad', 'artha': 'Transfer'},
    {'root': 'sAm', 'slp1': 'sAm', 'artha': 'Transfer'},
    {'root': 'hA', 'slp1': 'hA', 'artha': 'Transfer'},
    {'root': 'nihR', 'slp1': 'nihR', 'artha': 'Transfer'},
    {'root': 'vah', 'slp1': 'vah', 'artha': 'Transfer'},
]

AXIS_MAP = {'Motion': 0, 'Existence': 1, 'Action': 2, 'Violence': 3, 'Transfer': 4}

for d in DHATU_CORPUS:
    d['axis'] = AXIS_MAP.get(d['artha'], -1)

while len(DHATU_CORPUS) < N_ROOTS:
    DHATU_CORPUS.append(dict(DHATU_CORPUS[len(DHATU_CORPUS) % 50]))
    DHATU_CORPUS[-1]['axis'] = AXIS_MAP.get(DHATU_CORPUS[-1]['artha'], -1)

print(f"Loaded corpus: {len(DHATU_CORPUS)} roots")

PHONEME_VECTORS_16 = {
    'k': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0.15, 1.40, 0],
    'K': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0.15, 1.40, 0],
    'g': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.15, 1.40, 0],
    'G': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.15, 1.40, 0],
    'c': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.15, 2.10, 0],
    'j': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.15, 2.10, 0],
    'T': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.15, 1.10, 0],
    'D': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.15, 1.10, 0],
    't': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.15, 1.80, 0],
    'd': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.15, 1.80, 0],
    'n': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.20, 1.80, 0],
    'p': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.15, 0.80, 0],
    'b': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.15, 0.80, 0],
    'm': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.20, 0.80, 0],
    'y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.40, 2.10, 0],
    'r': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.40, 1.10, 0],
    'l': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.40, 1.80, 0],
    'v': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.40, 0.80, 0],
    's': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.25, 1.80, 0],
    'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.25, 1.80, 0],
    'h': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.60, 1.20, 0],
    'a': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.80, 1.30, 1],
    'A': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.80, 1.30, 1],
    'i': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.28, 2.30, 1],
    'I': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.28, 2.30, 1],
    'u': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.28, 0.70, 1],
    'U': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.28, 0.70, 1],
    'e': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.40, 2.00, 1],
    'o': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.49, 0.80, 1],
}

DHATU_FEATURES = {
    0: {'locus': 'velar', 'voiced': 1, 'nasal': 0},
    1: {'locus': 'palatal', 'voiced': 1, 'nasal': 0},
    2: {'locus': 'retroflex', 'voiced': 0, 'nasal': 0},
    3: {'locus': 'dental', 'voiced': 1, 'nasal': 0},
    4: {'locus': 'labial', 'voiced': 1, 'nasal': 0},
}

def dhatu_to_tensor(root, artha):
    root_clean = root.replace(' ', '')
    vec = np.zeros(21)
    artha_idx = {'Motion': 0, 'Existence': 1, 'Action': 2, 'Violence': 3, 'Transfer': 4}.get(artha, 2)
    
    for i, char in enumerate(list(root_clean)[:3]):
        key = char.lower()
        if key in PHONEME_VECTORS_16:
            vec[i*5:(i+1)*5] = PHONEME_VECTORS_16[key][:5]
        elif key in DHATU_FEATURES:
            feat = DHATU_FEATURES[key]
            vec[i*5] = {'velar': 0, 'palatal': 1, 'retroflex': 2, 'dental': 3, 'labial': 4}.get(feat['locus'], 3) / 4.0
            vec[i*5+1] = feat['voiced']
            vec[i*5+2] = 0
            vec[i*5+3] = feat['nasal']
            vec[i*5+4] = 0
    
    vec[15:20] = [0.2] * 5
    for i in range(5):
        vec[15+i] = 0.8 if i == artha_idx else 0.05
    
    return vec

DHATU_CORPUS_TENSORS = [dhatu_to_tensor(d['root'], d['artha']) for d in DHATU_CORPUS]
DHATU_AXES = np.array([d['axis'] for d in DHATU_CORPUS])

class AdExReservoir(nn.Module):
    def __init__(self, n_inputs=21, n_neurons=512):
        super().__init__()
        self.n_neurons = n_neurons
        
        self.alpha = nn.Parameter(torch.rand(n_neurons) * 0.8 + 0.1)
        self.beta = nn.Parameter(torch.rand(n_neurons) * 0.1 + 0.01)
        self.delta_t = nn.Parameter(torch.rand(n_neurons) * 10.0 + 2.0)
        
        self.w_in = nn.Parameter(torch.randn(n_neurons, n_inputs) * 0.02)
        
        self.v_thresh = nn.Parameter(torch.rand(n_neurons) * (-10) - 20)
        self.v_reset = nn.Parameter(torch.rand(n_neurons) * 10 - 40)
        
    def forward(self, x, dt=0.1):
        batch_size = x.shape[0]
        v = torch.zeros(batch_size, self.n_neurons, device=x.device)
        w = torch.zeros(batch_size, self.n_neurons, device=x.device)
        
        for t in range(20):
            I = torch.mm(x, self.w_in.t())
            
            v_exp = torch.clamp((v - self.v_thresh.unsqueeze(0)) / self.delta_t.unsqueeze(0), max=10)
            w_exp = self.beta.unsqueeze(0) * torch.exp(v_exp)
            
            dv = -self.alpha.unsqueeze(0) * (v - (-70)) + I - w_exp
            v = v + dv * dt
            
            spikes = (v > self.v_thresh.unsqueeze(0)).float()
            w = w + w_exp + spikes * 0.5
            v = torch.where(spikes > 0, self.v_reset.unsqueeze(0), v)
        
        return v

class VaikhariDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 23),
        )
    
    def forward(self, x):
        return self.net(x)

def generate_attractor_states():
    print("Generating attractor states from corpus...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    inputs = np.array(DHATU_CORPUS_TENSORS)
    
    model = AdExReservoir(n_inputs=21, n_neurons=512)
    model.eval()
    
    inputs_t = torch.tensor(inputs, dtype=torch.float32)
    with torch.no_grad():
        states = model(inputs_t).numpy()
    
    kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
    kmeans.fit(states)
    
    centroids = kmeans.cluster_centers_
    print(f"Generated {len( centroids)} attractor states")
    
    return states, centroids

def sample_from_basin(states, basin_idx, n_samples, seed):
    np.random.seed(seed)
    basin_mask = DHATU_AXES == basin_idx
    basin_states = states[basin_mask]
    
    if len(basin_states) < n_samples:
        indices = np.random.choice(len(basin_states), n_samples, replace=True)
    else:
        indices = np.random.choice(len(basin_states), n_samples, replace=False)
    
    return basin_states[indices]

def pingala_complete(acoustic_23d):
    """Piṅgala completion: 23D acoustic → phoneme sequence"""
    vec = acoustic_23d.copy()
    
    if vec[9] > 0.3:
        initial = 'p'
    elif vec[5] > 0.3:
        initial = 'k'
    elif vec[6] > 0.3:
        initial = 'c'
    else:
        initial = 't'
    
    if vec[2] > 0.3:
        vowel = 'i'
    elif vec[1] > 0.3:
        vowel = 'u'
    elif vec[3] > 0.3:
        vowel = 'a'
    else:
        vowel = 'a'
    
    positions = [0, 1, 2]
    patterns = ['CV', 'CVC', 'CVCC']
    pattern = patterns[vec[22] % 3] if vec[22] > 0.1 else 'CVC'
    
    result = initial + vowel
    if 'CVC' in pattern and len(result) > 1:
        result += initial
    elif 'CVCC' in pattern:
        result += initial + initial
    
    return result

def run_experiment():
    print("=" * 60)
    print("DDIN Exp 54 v2: Vaikharī Generation & Piṅgala")
    print("=" * 60)
    print(f"Basins: {N_BASINS}")
    print(f"Samples per basin: {N_SAMPLES_PER_BASIN}")
    print(f"Total to generate: {TOTAL_GENERATED}")
    print()
    
    states, centroids = generate_attractor_states()
    
    np.save('attractor_states.npy', states)
    np.save('centroids.npy', centroids)
    print("Saved attractor_states.npy and centroids.npy")
    
    print("\n" + "=" * 60)
    print("Generating from each basin...")
    print("=" * 60)
    
    decoder = VaikhariDecoder()
    
    all_generated = []
    
    for basin_idx in range(N_BASINS):
        basin_names = ['Motion', 'Existence', 'Action', 'Violence', 'Transfer']
        print(f"\n--- Basin {basin_names[basin_idx]} ---")
        
        samples = sample_from_basin(states, basin_idx, N_SAMPLES_PER_BASIN, 42)
        
        for i, state in enumerate(samples):
            state_t = torch.tensor(state.reshape(1, -1), dtype=torch.float32)
            acoustic = decoder(state_t).detach().numpy()[0]
            
            generated = pingala_complete(acoustic)
            all_generated.append({
                'basin': basin_names[basin_idx],
                'acoustic': acoustic.tolist(),
                'generated': generated
            })
            print(f"  {i+1}: {generated}")
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Generated {len(all_generated)} roots")
    
    with open('vaikhari_generated.json', 'w') as f:
        json.dump(all_generated, f, indent=2)
    
    print(f"\nSaved to vaikhari_generated.json")
    
    basins_used = {}
    for g in all_generated:
        b = g['basin']
        basins_used[b] = basins_used.get(b, 0) + 1
    
    print("\nGeneration by basin:")
    for b, c in basins_used.items():
        print(f"  {b}: {c}")
    
    return all_generated

if __name__ == "__main__":
    run_experiment()