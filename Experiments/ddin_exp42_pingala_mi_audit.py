"""
Exp 42: Pingala Mutual Information Audit
No network required. Computes MI between Pingala address, Locus, and Semantic Axis.
Runs in seconds on CPU.
"""
import numpy as np
import json
from collections import Counter
from scipy.stats import entropy

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
    addr = compute_pingala_address(root_slp1, max_syllables=10)
    return sum(1 + b for b in addr)

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

def compute_mi_discrete(x, y, n_bins=10):
    min_bins = min(n_bins, max(int(np.sqrt(len(x) / 10)), 5))
    hist_xy, _, _ = np.histogram2d(x, y, bins=min_bins)
    pxy = hist_xy / (hist_xy.sum() + 1e-10)
    px  = pxy.sum(axis=1, keepdims=True)
    py  = pxy.sum(axis=0, keepdims=True)
    MI  = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            if pxy[i,j] > 1e-10 and px[i,0] > 1e-10 and py[0,j] > 1e-10:
                MI += pxy[i,j] * np.log(pxy[i,j] / (px[i,0] * py[0,j] + 1e-10) + 1e-10)
    return max(MI, 0.0)

print("="*70)
print("Exp 42 -- Pingala Mutual Information Audit")
print("="*70)

with open(r'temp_ashtadhyayi_data/dhatu/data.txt', encoding='utf-8') as f:
    full_corpus = json.load(f)['data']
corpus = full_corpus[:2000]
print("Corpus: %d roots" % len(corpus))

pingala_table = {}
locus_list, sem_list, pingala_list, matra_list = [], [], [], []

for item in corpus:
    root_slp1  = devanagari_to_slp1(item['dhatu'])
    artha_slp1 = devanagari_to_slp1(item['artha'])
    a_tensor   = extract_artha_stem_tensor(artha_slp1)
    label      = int(np.argmax(a_tensor)) if np.max(a_tensor) > 0.2 else -1
    locus      = compute_locus_label(root_slp1)
    pingala    = compute_pingala_address(root_slp1, max_syllables=4)
    matra      = compute_matras(root_slp1)

    pingala_table[root_slp1] = pingala
    locus_list.append(locus)
    sem_list.append(label)
    pingala_list.append(pingala)
    matra_list.append(matra)

locus_arr   = np.array(locus_list)
sem_arr     = np.array(sem_list)
matra_arr   = np.array(matra_list)
pingala_idx = np.array([int(''.join(str(b) for b in p), 2) for p in pingala_list])

mask = sem_arr != -1
locus_v   = locus_arr[mask]
sem_v     = sem_arr[mask]
matra_v   = matra_arr[mask]
pingala_v = pingala_idx[mask]

print("\n" + "="*70)
print("PiMGALA ADDRESS DISTRIBUTION")
print("="*70)
addr_counts = Counter(pingala_list)
for addr in sorted(addr_counts.keys()):
    binary = ''.join(str(b) for b in addr)
    print("  %s: %3d roots (%5.1f%%)" % (binary, addr_counts[addr], 100*addr_counts[addr]/len(corpus)))

print("\n" + "="*70)
print("MATRA DISTRIBUTION (M = total morae)")
print("="*70)
matra_counts = Counter(matra_list)
for m in sorted(matra_counts.keys()):
    pct = 100 * matra_counts[m] / len(corpus)
    print("  M=%d: %3d roots (%5.1f%%)" % (m, matra_counts[m], pct))

print("\n" + "="*70)
print("MUTUAL INFORMATION ANALYSIS")
print("="*70)
MI_pingala_sem   = compute_mi_discrete(pingala_v.astype(float), sem_v.astype(float))
MI_locus_sem     = compute_mi_discrete(locus_v.astype(float), sem_v.astype(float))
MI_matra_sem     = compute_mi_discrete(matra_v.astype(float), sem_v.astype(float))
MI_pingala_locus = compute_mi_discrete(pingala_v.astype(float), locus_v.astype(float))

print("\n  Channel               MI(bits)   Interpretation")
print("  " + "-"*60)
print("  PiMgala -> Axis:     %6.4f     (binary prosodic address)" % MI_pingala_sem)
print("  Matra   -> Axis:     %6.4f     (total mora count)" % MI_matra_sem)
print("  Locus   -> Axis:     %6.4f     (place of articulation)" % MI_locus_sem)
print("  PiMgala -> Locus:    %6.4f     (should be ~0 for orthogonality)" % MI_pingala_locus)

H_sem = entropy(np.bincount(sem_v, minlength=5))
H_locus = entropy(np.bincount(locus_v, minlength=6)[:5])
NMI_pingala = MI_pingala_sem / (H_sem + 1e-10)
NMI_locus   = MI_locus_sem   / (H_locus + 1e-10)

print("\n  Normalized MI:")
print("    NMI(PiMgala, Axis) = %.4f" % NMI_pingala)
print("    NMI(Locus, Axis)    = %.4f" % NMI_locus)

print("\n" + "="*70)
print("VERDICT")
print("="*70)
if MI_pingala_sem > MI_locus_sem:
    ratio = MI_pingala_sem / (MI_locus_sem + 1e-10)
    print("  PiMgala channel carries MORE semantic signal than Locus (%.2fx)" % ratio)
    print("  --> PiMgala is a better predictor of semantic axis than Locus")
    print("  --> Adding PiMgala to input tensor should improve ARI")
else:
    ratio = MI_locus_sem / (MI_pingala_sem + 1e-10)
    print("  Locus channel carries more signal (%.2fx PiMgala)" % ratio)

if MI_pingala_locus < 0.05:
    print("  PiMgala and Locus are NEARLY ORTHOGONAL (MI=%.4f)" % MI_pingala_locus)
    print("  --> Adding PiMgala introduces NEW information, not redundancy")

print("\n  Success criterion (MI > 0.02 bits): %s" % ("PASS" if MI_pingala_sem > 0.02 else "FAIL"))
print("="*70)
