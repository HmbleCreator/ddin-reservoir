import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score, silhouette_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*70)
print("DDIN v29 — TWO-LAYER BASELINE (Phase 6: Hierarchical Architecture)")
print("="*70)

# ─────────────────────────────────────────────────────────────────
# 1.  DATA
# ─────────────────────────────────────────────────────────────────

DATA = r'c:\Users\amiku\Downloads\AI Research New Paradigm\SampleData'
df = pd.read_csv(f'{DATA}/task1_axis_prediction.csv')

print(f"\nDataset: {len(df)} roots, {df.actual_axis.nunique()} axes")

# ─────────────────────────────────────────────────────────────────
# 2.  TWO-LAYER MODEL
# ─────────────────────────────────────────────────────────────────

class TwoLayerDDIN(nn.Module):
    """
    Two-layer DDIN:
    - Layer 1 (Parā): fast phoneme encoder (W=0)
    - Layer 2 (Paśyantī): slow semantic organizer (W can be non-zero)
    """
    def __init__(self, in_dim=23, dim1=128, dim2=64):
        super().__init__()
        
        # Layer 1: phoneme encoder (fast dynamics, W=0)
        self.layer1_dim = dim1
        self.layer1_alpha = nn.Parameter(torch.rand(dim1) * 0.8 + 0.1)  # fast decay
        self.layer1_beta = nn.Parameter(torch.rand(dim1) * 0.8 + 0.1)
        self.layer1_proj = nn.Parameter(torch.randn(in_dim, dim1) * 0.1)
        self.layer1_W = nn.Parameter(torch.zeros(dim1, dim1))  # W=0
        
        # Layer 2: semantic organizer (slower dynamics, can have structure)
        self.layer2_dim = dim2
        self.layer2_alpha = nn.Parameter(torch.rand(dim2) * 0.6 + 0.3)  # slower
        self.layer2_beta = nn.Parameter(torch.rand(dim2) * 0.8 + 0.1)
        self.layer2_proj = nn.Parameter(torch.randn(dim1, dim2) * 0.1)
        # Layer 2 W: sparse structured ( Dhātu-like)
        self.layer2_W = nn.Parameter(torch.randn(dim2, dim2) * 0.02)

    def forward_layer1(self, x, u, dt=0.020):
        drive = u @ self.layer1_proj
        dx = -self.layer1_alpha * x + torch.tanh(x @ self.layer1_W) + self.layer1_beta * drive
        return x + dt * dx

    def forward_layer2(self, h1, dt=0.050):
        """Layer 2 receives Layer 1 final state as input"""
        drive = h1 @ self.layer2_proj
        dx = -self.layer2_alpha * h1[:, :self.layer2_dim] + torch.tanh(h1[:, :self.layer2_dim] @ self.layer2_W) + self.layer2_beta * drive
        return h1[:, :self.layer2_dim] + dt * dx

    def forward(self, u_seq, return_layer1=False):
        """Forward pass through both layers"""
        # Layer 1: process phoneme sequence
        x = torch.zeros(1, self.layer1_dim).to(device)
        for u in u_seq:
            x = self.forward_layer1(x, u.unsqueeze(0), dt=0.020)
        
        # Silence pass for Layer 1
        u_zero = torch.zeros(1, 23).to(device)
        for _ in range(2):
            x = self.forward_layer1(x, u_zero, dt=0.050)
        
        layer1_output = x
        
        if return_layer1:
            return layer1_output
        
        # Layer 2: semantic integration
        h1 = layer1_output.unsqueeze(0).repeat(self.layer2_dim, 1).T if layer1_output.dim() == 1 else layer1_output.unsqueeze(0)
        # Actually: create a state for Layer 2
        h2 = torch.zeros(1, self.layer2_dim).to(device)
        for _ in range(3):  # integrate a few steps
            h2 = self.forward_layer2(h1, dt=0.050)
        
        return layer1_output, h2

# ─────────────────────────────────────────────────────────────────
# 3.  PHONOLOGICAL MAPPINGS (same as before)
# ─────────────────────────────────────────────────────────────────

def f1f2_norm(f1_hz, f2_hz):
    return [round(np.clip(f1_hz/1200.0, 0, 1), 4),
            round(np.clip((f2_hz-200)/2600.0, 0, 1), 4)]

FORMANT_F1F2 = {
    'a':f1f2_norm(800,1300), 'A':f1f2_norm(800,1300),
    'i':f1f2_norm(280,2300), 'I':f1f2_norm(280,2300),
    'u':f1f2_norm(280,700), 'U':f1f2_norm(280,700),
    'R':f1f2_norm(490,1380), 'L':f1f2_norm(490,1800), 'e':f1f2_norm(400,2000), 'o':f1f2_norm(490,800),
    'k':f1f2_norm(150,1400),'K':f1f2_norm(150,1400),'g':f1f2_norm(150,1400),
    'G':f1f2_norm(150,1400),'N':f1f2_norm(200,1400),'h':f1f2_norm(600,1200),
    'c':f1f2_norm(150,2100),'C':f1f2_norm(150,2100),'j':f1f2_norm(150,2100),
    'J':f1f2_norm(150,2100),'y':f1f2_norm(400,2100),'z':f1f2_norm(250,2100),
    'T':f1f2_norm(150,1100),'Q':f1f2_norm(150,1100),'D':f1f2_norm(150,1100),
    'X':f1f2_norm(150,1100),'r':f1f2_norm(400,1100),'x':f1f2_norm(250,1100),
    't':f1f2_norm(150,1800),'H':f1f2_norm(150,1800),'d':f1f2_norm(150,1800),
    'W':f1f2_norm(150,1800),'n':f1f2_norm(200,1800),'l':f1f2_norm(400,1800),
    's':f1f2_norm(250,1800),
    'p':f1f2_norm(150,800),'P':f1f2_norm(150,800),'b':f1f2_norm(150,800),
    'B':f1f2_norm(150,800),'m':f1f2_norm(200,800),'v':f1f2_norm(400,800),
    'S':f1f2_norm(250,2100), 'M':f1f2_norm(200,800), 'L':f1f2_norm(490,1800),
}

PHONEME_VECTORS_16 = {
    'a':[0.50,0.50,0.0,1.0,0.0,0.0,0.50,0.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,0.7],
    'A':[0.50,0.50,0.0,1.0,0.0,0.0,0.50,1.0,1.0,1.0,1.0,1.0,0.0,0.0,1.0,0.7],
    'i':[0.25,0.80,0.0,1.0,0.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.3,0.7],
    'I':[0.25,0.80,0.0,1.0,0.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.3,0.7],
    'u':[0.75,0.80,0.0,1.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.2,0.7],
    'U':[0.75,0.80,0.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.2,0.7],
    'R':[0.50,0.60,0.0,1.0,0.0,0.5,0.0,0.0,1.0,0.9,1.0,1.0,0.0,0.0,0.5,0.7],
    'L':[0.75,0.70,0.0,1.0,0.0,0.5,0.0,0.0,1.0,0.9,1.0,1.0,0.0,0.0,0.5,0.7],
    'e':[0.25,0.70,0.0,1.0,0.0,0.7,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.7,0.7],
    'o':[0.75,0.70,0.0,1.0,0.0,0.7,1.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.7,0.7],
    'k':[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    'K':[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.35,0.0,0.0,0.0,0.0],
    'g':[0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.2,0.0,0.0,0.0,0.0],
    'G':[0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.35,0.0,0.0,0.0,0.0],
    'N':[0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.5,0.5,0.5,0.0,0.0,0.0,1.0],
    'h':[0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.3,1.0,0.65,0.0,0.0,0.0,0.0],
    'c':[0.25,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],
    'C':[0.25,0.25,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.35,0.0,1.0,0.0,0.0],
    'j':[0.25,0.25,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.2,0.0,1.0,0.0,0.0],
    'J':[0.25,0.25,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.35,0.0,1.0,0.0,0.0],
    'y':[0.25,0.25,0.0,1.0,0.0,0.8,0.0,0.0,1.0,0.8,1.0,0.85,0.0,1.0,0.0,0.4],
    'z':[0.25,0.25,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.2,1.0,0.55,1.0,1.0,0.0,0.0],
    'T':[0.5,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],
    'Q':[0.5,0.5,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.35,0.0,1.0,0.0,0.0],
    'D':[0.5,0.5,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.2,0.0,1.0,0.0,0.0],
    'X':[0.5,0.5,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.35,0.0,1.0,0.0,0.0],
    'r':[0.5,0.5,0.0,1.0,0.0,0.3,0.0,0.0,1.0,0.8,1.0,0.75,0.0,1.0,0.0,0.4],
    'x':[0.5,0.5,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.2,1.0,0.55,1.0,1.0,0.0,0.0],
    't':[0.75,0.75,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],
    'H':[0.75,0.75,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.35,0.0,1.0,0.0,0.0],
    'd':[0.75,0.75,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.2,0.0,1.0,0.0,0.0],
    'W':[0.75,0.75,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.35,0.0,1.0,0.0,0.0],
    'n':[0.75,0.75,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.5,0.5,0.5,0.0,1.0,0.0,1.0],
    'l':[0.75,0.7,0.0,1.0,0.0,0.3,0.0,0.0,1.0,0.8,1.0,0.75,0.0,1.0,0.0,0.4],
    's':[0.75,0.75,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.2,1.0,0.55,1.0,1.0,0.0,0.0],
    'p':[1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    'P':[1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.35,0.0,0.0,0.0,0.0],
    'b':[1.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.2,0.0,0.0,0.0,0.0],
    'B':[1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.35,0.0,0.0,0.0,0.0],
    'm':[1.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.5,0.5,0.5,0.0,0.0,0.0,1.0],
    'v':[1.0,1.0,0.0,1.0,0.0,0.2,1.0,0.0,1.0,0.7,1.0,0.85,0.0,0.0,0.0,0.4],
    'S':[0.25,0.25,1.0,1.0,0.0,0.0,0.0,0.0,1.0,0.1,1.0,0.35,1.0,1.0,0.0,0.0],
    'M':[1.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.5,0.5,0.5,0.0,0.0,0.0,1.0],
    'L':[0.75,0.70,0.0,1.0,0.0,0.3,0.0,0.0,1.0,0.8,1.0,0.75,0.0,1.0,0.0,0.4],
}

MANNER_DIMS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15]
AC = set(['a','A','i','I','u','U','R','L','e','o'])
PRATYAHARA_SETS = [AC, set(PHONEME_VECTORS_16.keys()) - AC, set(['y','r','l','v']),
                  set(['K','G','C','J','Q','X','H','W','P','B','z','x','s','h']),
                  set(['z','x','s','h']), set(['g','j','D','d','b']), set(['N','n','m']), set(['a','i','u','R','L'])]

TRANSLIT = {'A':'A','I':'I','U':'U','R':'R','kh':'K','gh':'G','ch':'C','jh':'J','Th':'Q','Dh':'X',
           'th':'H','dh':'W','ph':'P','bh':'B','sh':'z','sh2':'x','ng':'N','S':'x', 'M':'m'}

def root_to_chars(s):
    chars = []
    i = 0
    while i < len(s):
        if i+1 < len(s) and s[i:i+2] in TRANSLIT:
            chars.append(TRANSLIT[s[i:i+2]])
            i += 2
        else:
            chars.append(TRANSLIT.get(s[i], s[i]))
            i += 1
    return chars

FORMANT_WEIGHT = 4.0

def embed_22(ph_char):
    ac16 = PHONEME_VECTORS_16.get(ph_char, [0.5]*16)
    manner = [ac16[d] for d in MANNER_DIMS]
    form = [x * FORMANT_WEIGHT for x in FORMANT_F1F2.get(ph_char, [0.4, 0.4])]
    prat = [1.0 if ph_char in S else 0.0 for S in PRATYAHARA_SETS]
    return np.array(manner + form + prat, dtype=np.float32)

def encode_two_layer(model, root_str):
    """Encode a root through both layers"""
    chars = root_to_chars(root_str)
    if not chars:
        return np.zeros(64, dtype=np.float32)  # Layer 2 output dim
    
    n_vowels = sum(1 for c in chars if c in AC)
    vowel_ratio = n_vowels / max(len(chars), 1)
    
    # Build sequence of inputs
    u_seq = []
    for c in chars:
        ph_vec = embed_22(c)
        ph_vec_23 = np.concatenate([ph_vec, [vowel_ratio]])
        u_seq.append(torch.tensor(ph_vec_23, dtype=torch.float32).to(device))
    
    # Layer 1 encoding
    x = torch.zeros(1, model.layer1_dim).to(device)
    with torch.no_grad():
        for u in u_seq:
            x = model.forward_layer1(x, u.unsqueeze(0), dt=0.020)
        
        # Silence
        u_zero = torch.zeros(1, 23).to(device)
        for _ in range(2):
            x = model.forward_layer1(x, u_zero, dt=0.050)
        
        layer1_out = x.squeeze()
        
        # Layer 2 encoding
        h2 = torch.zeros(1, model.layer2_dim).to(device)
        # Layer 2 receives Layer 1 output projected through layer2_proj
        for _ in range(3):
            h2 = model.forward_layer2(layer1_out.unsqueeze(0), dt=0.050)
    
    return h2.squeeze().cpu().numpy()

# ─────────────────────────────────────────────────────────────────
# 4.  EVALUATION
# ─────────────────────────────────────────────────────────────────

def evaluate_two_layer(model, test_df, k_eval=5, use_layer2=True):
    """Evaluate the two-layer system"""
    states = []
    for _, r in test_df.iterrows():
        rep = encode_two_layer(model, r['root'])
        states.append(rep)
    
    states = np.array(states)
    scaler = StandardScaler()
    states = scaler.fit_transform(states)
    
    kmeans = KMeans(n_clusters=k_eval, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(states)
    
    le = LabelEncoder()
    true_labels = le.fit_transform(test_df['actual_axis'].values)
    
    ari = adjusted_rand_score(true_labels, pred_labels)
    sil = silhouette_score(states, pred_labels)
    return ari, sil

# ─────────────────────────────────────────────────────────────────
# 5.  EXPERIMENT 29: TWO-LAYER BASELINE
# ─────────────────────────────────────────────────────────────────

np.random.seed(42)
torch.manual_seed(42)

model = TwoLayerDDIN(in_dim=23, dim1=128, dim2=64).to(device)
model.eval()

k_clusters = df['actual_axis'].nunique()

print(f"\nRunning two-layer baseline (Layer 1: 128 neurons, Layer 2: 64 neurons)...")

ari, sil = evaluate_two_layer(model, df, k_eval=k_clusters)

print(f"\n" + "="*70)
print("EXP 29 RESULT: TWO-LAYER BASELINE")
print("="*70)
print(f"Layer 1 output dimension: 128")
print(f"Layer 2 output dimension: 64")
print(f"Layer 2 W: sparse structured (Dhatu-like)")
print(f"")
print(f"ARI: {ari:.4f}")
print(f"Silhouette: {sil:+.4f}")
print(f"")
print(f"Target: ARI > 0.08 (Layer 2 adds signal over single-layer)")

if ari > 0.08:
    print(f"\n>>> SUCCESS! Layer 2 adds signal - two-layer hierarchy works!")
elif ari > 0.06:
    print(f"\n>>> PARTIAL: Slight improvement over single-layer ceiling")
else:
    print(f"\n>>> BELOW TARGET: Ceiling persists even with two layers")

# Also test Layer 1 alone for comparison
# Just use the one-layer baseline
class OneLayerBaseline(nn.Module):
    def __init__(self, in_dim=23, dim=128):
        super().__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.rand(dim) * 0.8 + 0.1)
        self.beta = nn.Parameter(torch.rand(dim) * 0.8 + 0.1)
        self.proj = nn.Parameter(torch.randn(in_dim, dim) * 0.1)
        self.W = nn.Parameter(torch.zeros(dim, dim))

    def forward(self, x, u, dt=0.020):
        drive = u @ self.proj
        dx = -self.alpha * x + torch.tanh(x @ self.W) + self.beta * drive
        return x + dt * dx

def encode_one_layer(model, root_str):
    chars = root_to_chars(root_str)
    if not chars:
        return np.zeros(model.dim, dtype=np.float32)
    
    n_vowels = sum(1 for c in chars if c in AC)
    vowel_ratio = n_vowels / max(len(chars), 1)
    
    x = torch.zeros(1, model.dim).to(device)
    with torch.no_grad():
        for c in chars:
            ph_vec = embed_22(c)
            ph_vec_23 = np.concatenate([ph_vec, [vowel_ratio]])
            u = torch.tensor(ph_vec_23, dtype=torch.float32).unsqueeze(0).to(device)
            x = model(x, u, dt=0.020)
        u_zero = torch.zeros(1, 23).to(device)
        for _ in range(2):
            x = model(x, u_zero, dt=0.050)
    return x.cpu().squeeze().numpy()

def evaluate_one_layer(model, test_df, k_eval=5):
    states = np.array([encode_one_layer(model, r['root']) for _, r in test_df.iterrows()])
    scaler = StandardScaler()
    states = scaler.fit_transform(states)
    kmeans = KMeans(n_clusters=k_eval, random_state=42, n_init=10)
    pred = kmeans.fit_predict(states)
    le = LabelEncoder()
    true = le.fit_transform(test_df['actual_axis'].values)
    return adjusted_rand_score(true, pred)

model1 = OneLayerBaseline(in_dim=23, dim=128).to(device)
model1.eval()
ari_1 = evaluate_one_layer(model1, df, k_eval=k_clusters)

print(f"\nComparison:")
print(f"  Single-layer (baseline): ARI = {ari_1:.4f}")
print(f"  Two-layer (Exp 29):   ARI = {ari:.4f}")
print(f"  Delta:              {ari - ari_1:+.4f}")

print("="*70)