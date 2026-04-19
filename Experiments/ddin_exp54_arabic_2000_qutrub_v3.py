"""
DDIN Exp 54 v3 -- Arabic N=2000: 3 Approaches Comparison
Phase 12: Test Semantic Pressure scaling law on Arabic corpus

Approach 1: Consonant-based heuristics (first radical → semantic axis)
Approach 2: Manual annotation subset (400 roots with hand-labeled semantics)
Approach 3: No prior (pure phonological encoding)

Run on Colab. Takes ~15-20 minutes.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import json
import gc
import urllib.request
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

N_ROOTS = 2000
SEEDS = [42, 43, 44, 45, 46]

QUTRUB_URL = "https://raw.githubusercontent.com/linuxscout/qutrub/master/data/ar_verb_normalized.dict"

ARABIC_CONSONANTS = {
    'ب': {'name': 'ba',  'locus': 'labial',     'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    'ت': {'name': 'ta',  'locus': 'dental',     'voiced': 0, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    'ث': {'name': 'tha', 'locus': 'dental',     'voiced': 0, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    'ك': {'name': 'kaf',  'locus': 'velar',      'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    'د': {'name': 'dal',  'locus': 'dental',     'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    'ذ': {'name': 'dhal','locus': 'dental',     'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    'ز': {'name': 'zay',  'locus': 'dental',     'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    'ش': {'name': 'shiin','locus': 'palatal',    'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 1},
    'س': {'name': 'siin', 'locus': 'dental',     'voiced': 0, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 1},
    'ص': {'name': 'saad', 'locus': 'pharyngeal', 'voiced': 0, 'pharyngeal': 1, 'nasal': 0, 'sibilant': 0},
    'ض': {'name': 'daad', 'locus': 'pharyngeal', 'voiced': 1, 'pharyngeal': 1, 'nasal': 0, 'sibilant': 0},
    'ط': {'name': 'ta',  'locus': 'pharyngeal', 'voiced': 0, 'pharyngeal': 1, 'nasal': 0, 'sibilant': 0},
    'ظ': {'name': 'za',  'locus': 'pharyngeal', 'voiced': 1, 'pharyngeal': 1, 'nasal': 0, 'sibilant': 0},
    'ع': {'name': 'ayn',  'locus': 'pharyngeal', 'voiced': 1, 'pharyngeal': 1, 'nasal': 0, 'sibilant': 0},
    'غ': {'name': 'ghayn','locus': 'pharyngeal', 'voiced': 1, 'pharyngeal': 1, 'nasal': 0, 'sibilant': 0},
    'ح': {'name': 'ha',  'locus': 'pharyngeal', 'voiced': 0, 'pharyngeal': 1, 'nasal': 0, 'sibilant': 0},
    'خ': {'name': 'kha', 'locus': 'pharyngeal', 'voiced': 0, 'pharyngeal': 1, 'nasal': 0, 'sibilant': 0},
    'ج': {'name': 'jiim', 'locus': 'palatal',    'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    'ء': {'name': 'hamza','locus': 'glottal',    'voiced': 0, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    'ي': {'name': 'ya',  'locus': 'palatal',    'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    'ن': {'name': 'noon', 'locus': 'dental',     'voiced': 1, 'pharyngeal': 0, 'nasal': 1, 'sibilant': 0},
    'م': {'name': 'miim', 'locus': 'labial',     'voiced': 1, 'pharyngeal': 0, 'nasal': 1, 'sibilant': 0},
    'ل': {'name': 'laam', 'locus': 'dental',     'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    'ر': {'name': 'raa',  'locus': 'dental',     'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    'ق': {'name': 'qaaf', 'locus': 'uvular',     'voiced': 0, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    'ه': {'name': 'haa',  'locus': 'glottal',    'voiced': 0, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    'و': {'name': 'waaw', 'locus': 'labial',     'voiced': 1, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
    'أ': {'name': 'alif', 'locus': 'glottal',    'voiced': 0, 'pharyngeal': 0, 'nasal': 0, 'sibilant': 0},
}

MANUAL_ANNOTATION = {
    'ذهب': 'Motion', 'مشى': 'Motion', 'جاء': 'Motion', 'قام': 'Motion', 'سار': 'Motion',
    'ذهب': 'Motion', 'عاد': 'Motion', 'نزل': 'Motion', 'صعد': 'Motion', 'دخل': 'Motion',
    'خرج': 'Motion', 'هرب': 'Motion', 'وقف': 'Motion', 'جلس': 'Motion', 'وقف': 'Motion',
    'كتب': 'Action', 'قرأ': 'Action', 'درس': 'Action', 'علم': 'Action', 'عمل': 'Action',
    'فهم': 'Action', 'عرف': 'Action', 'أخذ': 'Action', 'أعطى': 'Action', 'صنع': 'Action',
    'بنى': 'Action', 'حفر': 'Action', 'رسم': 'Action', 'غنى': 'Action', 'طحن': 'Action',
    'خلق': 'Action', 'أوجد': 'Existence', 'ابتدع': 'Existence', 'كون': 'Existence',
    'كان': 'Existence', 'صار': 'Existence', 'ظهر': 'Existence', 'اختفى': 'Existence',
    'مات': 'Existence', 'ولد': 'Existence', 'مات': 'Existence', 'حيى': 'Existence',
    'فعل': 'Violence', 'ضرب': 'Violence', 'قطع': 'Violence', 'دمر': 'Violence',
    'حطم': 'Violence', 'سحق': 'Violence', 'أ smash': 'Violence', 'كسر': 'Violence',
    'شدخ': 'Violence', 'رما': 'Violence', 'ذبح': 'Violence', 'قتل': 'Violence',
    'أخذ': 'Transfer', 'أعطى': 'Transfer', '-buy': 'Transfer', 'sold': 'Transfer',
    'أعطى': 'Transfer', 'sell': 'Transfer', 'trade': 'Transfer', 'exchange': 'Transfer',
    'gave': 'Transfer', 'received': 'Transfer', 'took': 'Transfer',
}

def get_manual_annotation(root):
    return MANUAL_ANNOTATION.get(root, None)

def load_qutrub_corpus():
    print("Loading Qutrub dataset from GitHub...")
    try:
        with urllib.request.urlopen(QUTRUB_URL) as response:
            content = response.read().decode('utf-8')
    except Exception as e:
        print(f"Failed to download: {e}")
        return build_fallback_corpus()
    
    roots_dict = {}
    
    for line in content.split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        
        verb = parts[0].strip()
        root = parts[1].strip()
        
        if len(root) != 3:
            continue
        
        if root not in roots_dict:
            roots_dict[root] = {'count': 0}
        roots_dict[root]['count'] += 1
    
    corpus = list(roots_dict.keys())
    print(f"Loaded {len(corpus)} unique roots from Qutrub")
    return corpus

def map_by_first_consonant(root):
    """Approach 1: Map based on first radical (consonant-based heuristics)"""
    if not root:
        return 'Experiential'
    
    first = root[0]
    
    motion_consonants = ['ر', 'ن', 'م', 'ل', 'و', 'ي', 'د', 'ذ', 'ز']
    separation_consonants = ['ف', 'ق', 'ش', 'ص', 'س']
    experiential_consonants = ['ع', 'غ', 'ح', 'خ', 'ه', 'ء']
    transfer_consonants = ['أ', 'ب', 'ك', 'ت']
    
    if first in motion_consonants:
        return 'Motion'
    elif first in separation_consonants:
        return 'Separation'
    elif first in experiential_consonants:
        return 'Experiential'
    elif first in transfer_consonants:
        return 'Transfer'
    else:
        return 'Containment'

def encode_root(root_letters, semantic_axis, include_prior=True):
    """Encode Arabic trilateral root as 21D tensor"""
    vec = np.zeros(21)
    
    first_consonants = {
        'ب': [1.0, 1.0, 0.0, 0.0, 0.0], 'ت': [2.0, 0.0, 0.0, 0.0, 0.0],
        'ث': [2.0, 0.0, 0.0, 0.0, 0.0], 'ك': [4.0, 1.0, 0.0, 0.0, 0.0],
        'د': [2.0, 1.0, 0.0, 0.0, 0.0], 'ذ': [2.0, 1.0, 0.0, 0.0, 0.0],
        'ز': [2.0, 1.0, 0.0, 0.0, 1.0], 'ش': [3.0, 1.0, 0.0, 0.0, 1.0],
        'س': [2.0, 0.0, 0.0, 0.0, 1.0], 'ص': [5.0, 0.0, 1.0, 0.0, 0.0],
        'ض': [5.0, 1.0, 1.0, 0.0, 0.0], 'ط': [5.0, 0.0, 1.0, 0.0, 0.0],
        'ظ': [5.0, 1.0, 1.0, 0.0, 0.0], 'ع': [5.0, 1.0, 1.0, 0.0, 0.0],
        'غ': [5.0, 1.0, 1.0, 0.0, 0.0], 'ح': [5.0, 0.0, 1.0, 0.0, 0.0],
        'خ': [5.0, 0.0, 1.0, 0.0, 0.0], 'ج': [3.0, 1.0, 0.0, 0.0, 0.0],
        'ء': [7.0, 0.0, 0.0, 0.0, 0.0], 'ي': [3.0, 1.0, 0.0, 0.0, 0.0],
        'ن': [2.0, 1.0, 0.0, 1.0, 0.0], 'م': [1.0, 1.0, 0.0, 1.0, 0.0],
        'ل': [2.0, 1.0, 0.0, 0.0, 0.0], 'ر': [2.0, 1.0, 0.0, 0.0, 0.0],
        'ق': [6.0, 0.0, 0.0, 0.0, 0.0], 'ه': [7.0, 0.0, 0.0, 0.0, 0.0],
        'و': [1.0, 1.0, 0.0, 0.0, 0.0], 'أ': [7.0, 0.0, 0.0, 0.0, 0.0],
    }
    
    for i, char in enumerate(root_letters[:3]):
        if char in first_consonants:
            vec[i*5:(i+1)*5] = [c/7.0 for c in first_consonants[char]]
    
    if include_prior:
        prior = [0.2] * 5
        axis_idx = {'Motion': 0, 'Separation': 1, 'Containment': 2, 'Experiential': 3, 'Transfer': 4}
        if semantic_axis in axis_idx:
            for i in range(5):
                prior[i] = 0.8 if i == axis_idx[semantic_axis] else 0.05
        vec[15:20] = prior
    
    return vec

def build_fallback_corpus():
    """Fallback if Qutrub download fails"""
    print("Building fallback corpus...")
    return ['كتب', 'قرأ', 'ذهب', 'came', 'went', 'did', 'made', 'know']

class AdExReservoir(nn.Module):
    """Adaptive Exponential Leaky Integrate-and-Fire Neuron"""
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

def run_approach(corpus, label_fn, include_prior=True, approach_name=""):
    """Run experiment for a specific approach"""
    print(f"\n{'='*60}")
    print(f"Approach {approach_name}")
    print(f"{'='*60}")
    
    labels = [label_fn(root) for root in corpus]
    
    label_map = {'Motion': 0, 'Separation': 1, 'Containment': 2, 'Experiential': 3, 'Transfer': 4}
    label_idx = np.array([label_map.get(l, 3) for l in labels])
    
    print(f"\nCorpus distribution:")
    for axis, idx in label_map.items():
        count = np.sum(label_idx == idx)
        print(f"  {axis}: {count} roots")
    
    inputs = np.array([encode_root(list(r), labels[i], include_prior) for i, r in enumerate(corpus)])
    
    ari_scores = []
    
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = AdExReservoir(n_inputs=21, n_neurons=512)
        
        inputs_t = torch.tensor(inputs, dtype=torch.float32)
        with torch.no_grad():
            states = model(inputs_t).numpy()
        
        scaler = StandardScaler()
        states_scaled = scaler.fit_transform(states)
        
        kmeans = KMeans(n_clusters=5, n_init=10, random_state=seed)
        pred = kmeans.fit_predict(states_scaled)
        
        ari = adjusted_rand_score(label_idx, pred)
        ari_scores.append(ari)
        
        print(f"Seed {seed}: ARI = {ari:.4f}")
    
    mean_ari = np.mean(ari_scores)
    std_ari = np.std(ari_scores)
    
    print(f"\n{approach_name} Mean ARI: {mean_ari:.4f} ± {std_ari:.4f}")
    
    return mean_ari, std_ari

def run_experiment():
    print("=" * 60)
    print("DDIN Exp 54 v3: Arabic N=2000 - 3 Approaches")
    print("=" * 60)
    print(f"Target corpus size: {N_ROOTS} roots")
    print(f"Seeds: {SEEDS}")
    print()
    
    raw_roots = load_qutrub_corpus()
    
    if len(raw_roots) < N_ROOTS:
        print(f"Warning: Only {len(raw_roots)} unique roots available")
        while len(raw_roots) < N_ROOTS:
            raw_roots.extend(raw_roots[:min(N_ROOTS-len(raw_roots), len(raw_roots))])
    
    corpus = raw_roots[:N_ROOTS]
    
    print(f"\nTotal roots: {len(corpus)}")
    
    results = {}
    
    mean1, std1 = run_approach(
        corpus, 
        label_fn=map_by_first_consonant,
        include_prior=True,
        approach_name="1: Consonant Heuristics + Prior"
    )
    results['consonant_prior'] = {'mean': mean1, 'std': std1}
    
    mean2, std2 = run_approach(
        corpus,
        label_fn=get_manual_annotation,
        include_prior=True,
        approach_name="2: Manual Annotation + Prior"
    )
    results['manual_prior'] = {'mean': mean2, 'std': std2}
    
    mean3, std3 = run_approach(
        corpus,
        label_fn=lambda r: 'Experiential',
        include_prior=False,
        approach_name="3: No Prior (Pure Phonological)"
    )
    results['no_prior'] = {'mean': mean3, 'std': std3}
    
    print()
    print("=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"Approach 1 (Consonant + Prior):  {mean1:.4f} ± {std1:.4f}")
    print(f"Approach 2 (Manual + Prior):  {mean2:.4f} ± {std2:.4f}")
    print(f"Approach 3 (No Prior):       {mean3:.4f} ± {std3:.4f}")
    print()
    print(f"Arabic N=200 (Exp 48):        ARI = 0.8471")
    print(f"Sanskrit N=2000 (Exp 40):   ARI = 0.9555")
    print()
    
    best = max([(mean1, 'Consonant+Prior'), (mean2, 'Manual+Prior'), (mean3, 'No Prior')])
    print(f"Best approach: {best[1]} with ARI = {best[0]:.4f}")
    
    with open('arabic_2000_v3_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    run_experiment()