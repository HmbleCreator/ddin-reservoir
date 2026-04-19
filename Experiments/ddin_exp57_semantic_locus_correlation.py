
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import json
import re
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score

# ── Pāṇinian Locus Taxonomy ───────────────────────────────────────────────────
LOCUS_MAP = {
    'k': 0, 'K': 0, 'g': 0, 'G': 0, 'N': 0, # Kaṇṭhya (Velar)
    'c': 1, 'C': 1, 'j': 1, 'J': 1, 'Y': 1, 'y': 1, 'S': 1, # Tālavya (Palatal)
    'w': 2, 'W': 2, 'q': 2, 'Q': 2, 'R': 2, 'r': 2, 'z': 2, # Mūrdhanya (Retroflex)
    't': 3, 'T': 3, 'd': 3, 'D': 3, 'n': 3, 'l': 3, 's': 3, # Dantya (Dental)
    'p': 4, 'P': 4, 'b': 4, 'B': 4, 'm': 4, 'v': 4, # Oṣṭhya (Labial)
}

def get_locus_label(root_slp1):
    if not root_slp1: return -1
    char = root_slp1[0]
    return LOCUS_MAP.get(char, -1)

# ── Semantic Axis Stems (Corrected Phase 12 logic) ───────────────────────────
def extract_artha_stem_tensor(artha_slp1):
    axis_stems = {
        0: ['gat','cal','gam','car','kram','vicar','sarp','pad','vI','dR','cyu','sru','dru','pata','ira','kzara','seca','cezw','vez','plu','skand','Ira','sf','tf','kampa','yA','vraj','ru','liG','ucCrAy','ira','AplAv','ucCrAy','ira','sara','rez','mez','vez','lez','vez'],
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

# ── Transliteration ──────────────────────────────────────────────────────────
def devanagari_to_slp1(text):
    vowels = {'अ':'a','आ':'A','इ':'i','ई':'I','उ':'u','ऊ':'U','ऋ':'f','ॠ':'F','ऌ':'x','ॡ':'X','ए':'e','ऐ':'E','ओ':'o','औ':'O'}
    vowel_signs = {'ा':'A','ि':'i','ी':'I','ु':'u','ू':'U','ृ':'f','ॄ':'F','ॢ':'x','ॣ':'X','े':'e','ै':'E','ो':'o','ौ':'O'}
    consonants = {'क':'k','ख':'K','ग':'g','घ':'G','ङ':'N','च':'c','छ':'C','ज':'j','झ':'J','ञ':'Y','ट':'w','ठ':'W','ड':'q','ढ':'Q','ण':'R','त':'t','थ':'T','द':'d','ध':'D','न':'n','प':'p','फ':'P','ब':'b','भ':'B','म':'m','य':'y','r':'r','ल':'l','व':'v','श':'S','ष':'z','स':'s','ह':'h'}
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

def main():
    CORPUS_PATH = 'temp_ashtadhyayi_data/dhatu/data.txt'
    with open(CORPUS_PATH, encoding='utf-8') as f:
        full_corpus = json.load(f)['data']
    
    # Extract unique roots with both labels
    data = []
    seen = set()
    for item in full_corpus:
        root = devanagari_to_slp1(item['dhatu'])
        artha = devanagari_to_slp1(item['artha'])
        
        locus_label = get_locus_label(root)
        semantic_tensor = extract_artha_stem_tensor(artha)
        semantic_label = int(np.argmax(semantic_tensor)) if np.max(semantic_tensor) > 0.2 else -1
        
        if locus_label != -1 and semantic_label != -1 and root not in seen:
            data.append((root, semantic_label, locus_label))
            seen.add(root)
            
    print(f"Total roots for correlation analysis: {len(data)}")
    
    semantic_labels = np.array([d[1] for d in data])
    locus_labels = np.array([d[2] for d in data])
    
    # ── Correlation Metrics ───────────────────────────────────────────────────
    ari = adjusted_rand_score(semantic_labels, locus_labels)
    mi = mutual_info_score(semantic_labels, locus_labels)
    nmi = normalized_mutual_info_score(semantic_labels, locus_labels)
    
    print("\n--- Correlation Metrics ---")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Mutual Information (MI): {mi:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    
    # ── Contingency Table ─────────────────────────────────────────────────────
    # Rows: Semantic Axis, Columns: Locus Axis
    # Semantic: 0:MOT, 1:EXP, 2:TRN, 3:SEP, 4:CNT
    # Locus: 0:Vel, 1:Pal, 2:Ret, 3:Den, 4:Lab
    semantic_names = ['MOT', 'EXP', 'TRN', 'SEP', 'CNT']
    locus_names = ['Vel', 'Pal', 'Ret', 'Den', 'Lab']
    
    table = np.zeros((5, 5))
    for s, l in zip(semantic_labels, locus_labels):
        table[s, l] += 1
        
    print("\n--- Contingency Table (Rows=Semantic, Cols=Locus) ---")
    header = "      " + "".join([f"{n:<8}" for n in locus_names])
    print(header)
    for i, s_name in enumerate(semantic_names):
        row = f"{s_name:<6}" + "".join([f"{int(v):<8}" for v in table[i, :]])
        print(row)
        
    # ── Permutation Test for ARI Correlation ─────────────────────────────────
    obs_ari = ari
    n_perms = 1000
    perm_aris = []
    semantic_copy = semantic_labels.copy()
    for _ in range(n_perms):
        np.random.shuffle(semantic_copy)
        perm_aris.append(adjusted_rand_score(semantic_copy, locus_labels))
    p_value = np.sum(np.array(perm_aris) >= obs_ari) / n_perms
    print(f"\nPermutation Test ARI (n=1000): p = {p_value:.4f}")

if __name__ == "__main__":
    main()
