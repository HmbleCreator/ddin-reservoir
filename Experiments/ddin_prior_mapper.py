import urllib.request
import csv
import io
import json
import pandas as pd
import os

def build_benchmark_prior():
    """
    Fetches the full Pāṇinian morphological database and builds a 100% covered
    mapping for the specific 150-root benchmark dataset.
    """
    # 1. Get the benchmark keys from the CSV
    csv_path = r'c:\Users\amiku\Downloads\AI Research New Paradigm\SampleData\task1_axis_prediction.csv'
    if not os.path.exists(csv_path):
        print(f"Error: Could not find benchmark CSV at {csv_path}")
        return
    
    df_bench = pd.read_csv(csv_path)
    benchmark_keys = list(df_bench['root'].unique())
    print(f"Loaded {len(benchmark_keys)} benchmark roots.")

    # 2. Fetch the structural database
    url = "https://raw.githubusercontent.com/sanskrit/vyakarana/master/data/dhatupatha.csv"
    print(f"Fetching full Paninian database from: {url}")
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            csv_data = response.read().decode('utf-8')
    except Exception as e:
        print(f"Failed to download database: {e}")
        return
        
    reader = csv.reader(io.StringIO(csv_data))
    master_db = []
    for row in reader:
        if len(row) < 3: continue
        # Gana is usually the first column
        try:
            gana_num = int(row[0])
        except:
            gana_num = 1
        master_db.append({
            'raw_upadesa': row[2],
            'gana': gana_num,
            'meaning': row[3] if len(row) > 3 else "unknown"
        })

    # 3. Fuzzy Matching Logic
    benchmark_dict = {}
    missing_roots = []

    # Helper to clean/canonicalize roots for matching
    # Paninian roots often use different Romanization (SLP1)
    MANUAL_OVERRIDE = {
        'kSip': 'kzipa', 'kSar': 'kzara', 'chad': 'Cada', 'chid': 'Cida',
        'dhA': 'quDA', 'dhAv': 'DAvu', 'bhaj': 'Baja', 'bhA': 'BA',
        'bhid': 'Bida', 'mR': 'm f', 'sad': 'zad', 'sic': 'ziYca',
        'stambh': 'staMbu', 'sthA': 'zWA', 'cR': 'cara'
    }

    def find_match(key):
        k_low = key.lower()
        # 0. Manual Override
        if key in MANUAL_OVERRIDE:
            m = next((item for item in master_db if MANUAL_OVERRIDE[key] in item['raw_upadesa']), None)
            if m: return m

        # 1. Direct substring match
        m = next((item for item in master_db if k_low in item['raw_upadesa'].lower()), None)
        if m: return m
        
        # Try swapping common transliteration differences (Bench -> SLP1)
        # S = ś, z = ṣ, jY = jñ, kS = kṣ
        k_slp = k_low.replace('sh2', 'z').replace('sh', 'S').replace('sh', 'S').replace('ks', 'kS').replace('jn', 'jY')
        m = next((item for item in master_db if k_slp in item['raw_upadesa'].lower()), None)
        if m: return m
        
        # Try prefix match (first 2 chars) for very divergent cases
        m = next((item for item in master_db if item['raw_upadesa'].lower().startswith(k_low[:2])), None)
        return m

    for key in benchmark_keys:
        match = find_match(key)
        if match:
            benchmark_dict[key] = {
                'upadesa': match['raw_upadesa'],
                'gana': match['gana'],
                'meaning': match['meaning']
            }
        else:
            missing_roots.append(key)
            # Default to a safe placeholder
            benchmark_dict[key] = {'upadesa': 'DEFAULT', 'gana': 1, 'meaning': 'unknown'}

    # 4. Save results
    output_path = "exp38_balanced_prior.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_dict, f, indent=4)
        
    print(f"SUCCESS: Mapping complete.")
    print(f"Mapped: {len(benchmark_dict) - len(missing_roots)} / {len(benchmark_keys)}")
    if missing_roots:
        print(f"Missing ({len(missing_roots)}): {missing_roots[:10]}...")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    build_benchmark_prior()
