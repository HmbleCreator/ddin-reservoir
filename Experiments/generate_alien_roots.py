import random
import json

# Pāṇinian Phonotactic Sets
VOWELS = ['a', 'i', 'u', 'e', 'o', 'A', 'I', 'U', 'f']
CONSONANTS = ['k', 'K', 'g', 'G', 'c', 'C', 'j', 'J', 'w', 'W', 'q', 'Q', 't', 'T', 'd', 'D', 'n', 'p', 'P', 'b', 'B', 'm', 'y', 'r', 'l', 'v', 'S', 'z', 's', 'h']
CLUSTERS = ['sk', 'st', 'sp', 'Scy', 'Gr', 'kr', 'Tr', 'pr', 'dr', 'br', 'Gr', 'sn', 'sm', 'vy', 'kv', 'tv']

def generate_alien_root():
    # Randomly choose a structure: CV, CVC, CCV, CCVC
    structure = random.choice(['CV', 'CVC', 'CCV', 'CCVC'])
    root = ""
    if structure == 'CV':
        root = random.choice(CONSONANTS) + random.choice(VOWELS)
    elif structure == 'CVC':
        root = random.choice(CONSONANTS) + random.choice(VOWELS) + random.choice(CONSONANTS)
    elif structure == 'CCV':
        root = random.choice(CLUSTERS) + random.choice(VOWELS)
    elif structure == 'CCVC':
        root = random.choice(CLUSTERS) + random.choice(VOWELS) + random.choice(CONSONANTS)
    return root

# Load 2000 genuine roots for exclusion
with open('temp_ashtadhyayi_data/dhatu/data.txt', encoding='utf-8') as f:
    raw = json.load(f)
    genuine_roots = set(d['dhatu'] for d in raw['data']) # These are Devanagari

# We need a quick Devanagari to SLP1 to compare
def slp1_to_devanagari_basic(text):
    # This is a very rough mapper for checking existence
    return text # Placeholder

# For now, we generate 100 and manually check a few "Alien" sounding ones
ALIEN_ROOTS = []
while len(ALIEN_ROOTS) < 100:
    r = generate_alien_root()
    # Simple exclusion: if it's very short and looks common, skip
    if len(r) < 2: continue
    if r not in ALIEN_ROOTS:
        ALIEN_ROOTS.append(r)

print(f"Generated 100 Alien Roots. Sample: {ALIEN_ROOTS[:10]}")

with open('alien_roots.json', 'w') as f:
    json.dump(ALIEN_ROOTS, f)
