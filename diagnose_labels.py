
import json
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

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

CORPUS_PATH = 'temp_ashtadhyayi_data/dhatu/data.txt'
with open(CORPUS_PATH, encoding='utf-8') as f:
    full_corpus = json.load(f)['data']

priors = []
labels = []
for item in full_corpus:
    artha = devanagari_to_slp1(item['artha'])
    tensor = extract_artha_stem_tensor(artha)
    if np.max(tensor) > 0.2:
        priors.append(tensor)
        labels.append(np.argmax(tensor))

priors = np.array(priors)
labels = np.array(labels)
scaler = StandardScaler()
priors_scaled = scaler.fit_transform(priors)

print("\n--- Baseline ARI (K-Means on 5D Prior Vectors) ---")
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
pred = kmeans.fit_predict(priors_scaled)
print(f"K-Means (Prior only) ARI: {adjusted_rand_score(labels, pred):.4f}")

print("\n--- MLP Baseline ARI (Feedforward limit) ---")
clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
ari_scorer = make_scorer(adjusted_rand_score)
scores = cross_val_score(clf, priors_scaled, labels, cv=5, scoring=ari_scorer)
print(f"MLP (Prior-only) cross-val ARI: {scores.mean():.4f} ± {scores.std():.4f}")
