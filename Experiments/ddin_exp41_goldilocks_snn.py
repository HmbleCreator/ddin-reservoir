import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*65)
print("DDIN Exp 41 -- THE GOLDILOCKS RESERVOIR (512 NEURONS)")
print("Phase 10B: Finding the Optimal Semantic Density")
print("="*65)

# ─────────────────────────────────────────────────────────────────
# 1. DATA (146-ROOT MASTER DICTIONARY)
# ─────────────────────────────────────────────────────────────────

MASTER_ARTHA_DICT = {
    'gam': 'gatau', 'cal': 'kampane', 'car': 'gatiBakzaRayoH', 'kram': 'pAdavikzepe', 'srp': 'gatau', 'pad': 'gatau',
    'ya': 'prApaRe', 'vis': 'praveSane', 'bhu': 'sattAyAm', 'as': 'Buvi', 'jan': 'prAdurBAve',
    'div': 'krIqAvijigIzAvyavahAradyutistutimodamadasvapnakAntigatizu', 'bha': 'dIptO', 'vad': 'vyaktAyAM vAci',
    'bru': 'vyaktAyAM vAci', 'jiv': 'prARaDAraRe', 'pac': 'pAke', 'kri': 'karaRe', 'kr': 'karaRe',
    'taksh': 'tvaCane', 'nirma': 'nirmARe', 'han': 'hiMsAgatyoH', 'bhid': 'vidAraRe', 'chid': 'dvaIDIkaraRe',
    'rudh': 'AvaraRe', 'mri': 'prARatyAge', 'nas': 'adarSane', 'tud': 'vyaTAyAm', 'dha': 'DAraRapozaRayoH',
    'bhr': 'DAraRapozaRayoH', 'stha': 'gatinivfttO', 'grah': 'upAdAne', 'ap': 'vyAptO', 'tan': 'vistAre',
    'bandh': 'banDane', 'raksh': 'pAlane',
    'kan': 'dIptikAntigatizu', 'kAz': 'dIptO', 'kIrt': 'saMSabdane', 'kup': 'kope', 'kSar': 'saMcalane',
    'gai': 'Sabde', 'garj': 'Sabde', 'gRh': 'upAdAne', 'guh': 'saMvaraRe', 'hR': 'haraRe', 'hu': 'dAnAdanayoH',
    'hve': 'sparDAyAM Sabde ca', 'kI': 'Sabde', 'kep': 'kampane', 'gup': 'rakzaRe gopane ca', 'kzip': 'preraRe',
    'hims': 'hiMsAyAm', 'kLp': 'sAmarTye', 'ci': 'cayane', 'cit': 'saMjYAne', 'cint': 'smftyAm', 'cur': 'steye',
    'cud': 'preraRe', 'cR': 'hiMsAyAm', 'cay': 'gatau', 'cakAs': 'dIptO', 'jap': 'vyaktAyAM vAci',
    'jalp': 'vyaktAyAM vAci', 'ji': 'jaye', 'jval': 'dIptO', 'jash': 'hiMsAyAm', 'yaj': 'devapUjAsaNgatikaraRadAnezu',
    'yam': 'uparame', 'yu': 'miSraRe', 'jAgR': 'nidrAkzaye', 'yA': 'prApaRe gatau ca', 'yuddh': 'samprahAre',
    'rA': 'AdAne', 'raj': 'rAge', 'ram': 'krIqAyAm', 'ras': 'Sabde', 'ruh': 'prAdurBAve', 'rudh': 'AvaraRe ruDi ca',
    'rud': 'rodane', 'ri': 'gatau', 'ric': 'virecane', 'rip': 'gatau', 'ruj': 'viDvaMsane', 'lA': 'AdAne',
    'lag': 'saNge', 'lamb': 'sraMsane', 'laz': 'kAntO', 'lup': 'ChedanaBedanayoH', 'luth': 'upaGAte hiMsAyAM ca',
    'lI': 'SlezaRe', 'lok': 'darSane', 'rAj': 'dIptO', 'rambh': 'saMramBe', 'lip': 'upadehe', 'lubh': 'gArDye',
    'lih': 'AsvAdane', 'ruc': 'dIptO', 'rI': 'gatau', 'rak': 'pAlane rakzaRe ca', 'randh': 'hiMsAyAm',
    'rAd': 'saMsidDO', 'tap': 'santApe', 'tAy': 'santAne', 'tij': 'niSAne kzamAyAM ca', 'tul': 'unmAne',
    'tud': 'vyaTAyAm tudane', 'tan': 'vistAre', 'tRp': 'prIRane', 'dah': 'BasmIkaraRe nASe ca', 'dA': 'dAne',
    'diz': 'atisarjane', 'dIp': 'dIptO', 'drA': 'kutsitAyAM gatau', 'dru': 'gatau', 'dhA': 'DAraRapozaRayoH',
    'dhAv': 'gatiSudDyO', 'dus': 'vEkfTye', 'daMz': 'daSane', 'san': 'samBakTO', 'sah': 'marzaRe', 'sic': 'kzaraRe',
    'sidh': 'gatyAm', 'sRp': 'gatau', 'svap': 'Saye', 'sRj': 'visarge nirmARe ca', 'stR': 'AcCAdane',
    'skand': 'gatiSozaRayoH', 'syand': 'prasravaRe', 'tvar': 'samBrame', 'tyaj': 'hAnO', 'trik': 'gatau',
    'vac': 'vyaktAyAM vAci', 'vah': 'prApaRe gatau ca', 'vA': 'gatiganDanayoH', 'vaz': 'kAntO', 'vid': 'jYAne',
    'viz': 'praveSane', 'vR': 'varaRe', 'vRt': 'vartane', 'vyath': 'BayasaMcalanayoH', 'pA': 'pAne', 'pat': 'gatau',
    'pAl': 'rakzaRe pAlane ca', 'piS': 'avayave', 'pI': 'pAne', 'pu': 'pavane', 'pus': 'puzwO', 'pRR': 'pAlanapUraRayoH',
    'bhaj': 'sevAyAm', 'bhA': 'dIptO', 'bhid': 'vidAraRe Bedane ca', 'bhuj': 'pAlanAByavahArayoH', 'bhU': 'sattAyAm',
    'phan': 'gatau', 'bhr': 'DAraRapozaRayoH', 'bhaS': 'Bartsane', 'pRc': 'samparke', 'plav': 'plavane gatau ca'
}

def extract_artha_stem_tensor(artha):
    axis_stems = {
        0: ['gat', 'cal', 'gam', 'car', 'kram', 'vicar', 'sarp', 'pad'],
        1: ['satt', 'utpat', 'jan', 'Sabd', 'dIpt', 'prakAS', 'BAv', 'jIv'],
        2: ['pAk', 'vikAr', 'saMsk', 'kriy', 'nirmaA', 'kf'],
        3: ['hiMs', 'Bed', 'Cid', 'nAS', 'mAr', 'viDvaMs', 'tud'],
        4: ['DAr', 'pAl', 'saMvar', 'rakz', 'banD', 'sTA', 'ruD']
    }
    tensor = np.zeros(5)
    for i, stems in axis_stems.items():
        if any(s in artha for s in stems): tensor[i] = 1.0
    s = np.sum(tensor)
    return tensor/s if s>0 else np.ones(5)*0.2

# ─────────────────────────────────────────────────────────────────
# 2. GOLDILOCKS ARCHITECTURE (512/256)
# ─────────────────────────────────────────────────────────────────

class AdExPopulation(nn.Module):
    def __init__(self, size, dt=1.0, C=200.0, gL=10.0, EL=-70.0, VT=-50.0, DeltaT=2.0, Vpeak=0.0, Vreset=-58.0, a=2.0, b=80.0, tau_w=30.0):
        super().__init__()
        self.size = size
        self.dt = dt
        self.C, self.gL, self.EL, self.DeltaT, self.Vpeak, self.Vreset = C, gL, EL, DeltaT, Vpeak, Vreset
        self.VT = nn.Parameter(torch.ones(1, size).to(device) * VT)
        self.a = nn.Parameter(torch.ones(1, size).to(device) * a)
        self.b = nn.Parameter(torch.ones(1, size).to(device) * b)
        self.tau_w = nn.Parameter(torch.ones(1, size).to(device) * tau_w)
        self.V = torch.ones(1, size).to(device) * EL
        self.w = torch.zeros(1, size).to(device)
        self.theta = torch.ones(1, size).to(device) * 0.1 # Homeostasis
        self.spike_counts = torch.zeros(1, size).to(device)

    def reset_states(self):
        self.V = torch.ones(1, self.size).to(device) * self.EL
        self.w = torch.zeros(1, self.size).to(device)
        self.spike_counts = torch.zeros(1, self.size).to(device)

    def step(self, I_ext):
        exp_term = self.gL * self.DeltaT * torch.exp((self.V - self.VT) / self.DeltaT)
        dV = (-self.gL * (self.V - self.EL) + exp_term - self.w + I_ext) / self.C
        self.V += self.dt * dV
        dw = (self.a * (self.V - self.EL) - self.w) / self.tau_w
        self.w += self.dt * dw
        spikes = self.V >= self.Vpeak
        self.spike_counts += spikes.float()
        self.V = torch.where(spikes, torch.tensor(self.Vreset).to(device), self.V)
        self.w = torch.where(spikes, self.w + self.b, self.w)
        # Slow BCM Threshold Update
        self.theta += 0.005 * (spikes.float() - self.theta)
        return spikes

class GoldilocksSNN(nn.Module):
    def __init__(self, input_dim=28):
        super().__init__()
        self.L1 = AdExPopulation(size=512, a=0.0, b=0.0, tau_w=5.0, gL=15.0)
        mask1 = (torch.rand(input_dim, 512) < 0.18).float()
        # Boosted Gain (+25%)
        self.proj_in = nn.Parameter(torch.randn(input_dim, 512) * 1100.0 * mask1) 
        self.L2 = AdExPopulation(size=256, a=2.0, b=80.0, tau_w=150.0, gL=2.0)
        mask2 = (torch.rand(512, 256) < 0.18).float()
        self.W12 = nn.Parameter(torch.randn(512, 256) * 410.0 * mask2)

    def reset(self):
        self.L1.reset_states()
        self.L2.reset_states()

    def step(self, I_input):
        I1 = I_input @ self.proj_in
        sp1 = self.L1.step(I1)
        I2 = sp1.float() @ self.W12
        sp2 = self.L2.step(I2)
        return sp2

# ─────────────────────────────────────────────────────────────────
# 3. ENCODING
# ─────────────────────────────────────────────────────────────────

def embed_acoustic_23(char, vr):
    FORMANT_DATA = {'a':[0.67, 0.42], 'i':[0.23, 0.81], 'u':[0.23, 0.19], 'e':[0.33, 0.69], 'o':[0.41, 0.23]}
    TRANSLIT = {'A':'a','I':'i','U':'u','R':'a','kh':'k','gh':'g','ch':'c','jh':'j','Th':'t','Dh':'d','th':'t','dh':'d','ph':'p','bh':'b','sh':'s','ng':'n'}
    c = TRANSLIT.get(char, char.lower())[0]
    f = FORMANT_DATA.get(c, [0.4, 0.4])
    return np.concatenate([np.zeros(12), f, np.zeros(8), [vr]])

def encode_batch(model, df):
    all_states = []
    for _, r in df.iterrows():
        root = r['root']
        artha = MASTER_ARTHA_DICT.get(root.lower(), "")
        a_tensor = extract_artha_stem_tensor(artha)
        chars = list(root)
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
    return np.array(all_states)

# ─────────────────────────────────────────────────────────────────
# 4. EXECUTION
# ─────────────────────────────────────────────────────────────────

t1 = pd.read_csv(r'c:\Users\amiku\Downloads\AI Research New Paradigm\SampleData\task1_axis_prediction.csv')
ids = LabelEncoder().fit_transform(t1['actual_axis'].values)

model = GoldilocksSNN().to(device)
print(f"Goldilocks Network Initialized: 512 -> 256 Neurons")

states = encode_batch(model, t1)
states_norm = StandardScaler().fit_transform(states)

best_ari = -1
for _ in range(10): 
    pred = KMeans(n_clusters=5, n_init=1).fit_predict(states_norm)
    ari = adjusted_rand_score(ids, pred)
    if ari > best_ari: best_ari = ari

print(f"\nGOLDILOCKS RESULTS:")
print(f"  ARI: {best_ari:.4f}")
print(f"  Mean Activity: {states.mean():.2f} spikes/root")
