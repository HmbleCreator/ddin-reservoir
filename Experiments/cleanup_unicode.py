# -*- coding: utf-8 -*-
import sys, os

files = [
    'ddin_exp42_pingala_mi_audit.py',
    'ddin_exp45_replication_ablation_criticality.py',
    'ddin_exp46_pingala_integration.py',
    'ddin_exp45b_5seed_gpu_sweep.py',
]

for fname in files:
    path = os.path.join(os.path.dirname(__file__), fname)
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        text = raw.decode('utf-8')
    except Exception as e:
        print('Cannot read %s: %s' % (fname, e))
        continue

    for old, new in [
        ('\u1e45', 'n'), ('\u1e46', 'N'), ('\u1e47', 'n'), ('\u1e48', 'N'),
        ('\u1e49', 'N'), ('\u1e6d', 't'), ('\u1e6f', 'f'), ('\u0101', 'a'),
        ('\u012b', 'i'), ('\u016b', 'u'), ('\u0113', 'e'), ('\u014d', 'o'),
        ('\u1e41', 'm'), ('\u1e44', 'M'), ('\u1e0d', 'd'), ('\u1e0f', 'd'),
        ('\u1e1d', 'e'), ('\u1e25', 'h'), ('\u1e2b', 'h'), ('\u1e35', 'k'),
        ('\u1e3f', 'm'), ('\u1e43', 'm'), ('\u1e55', 'p'), ('\u1e57', 'p'),
        ('\u1e63', 's'), ('\u1e6b', 't'), ('\u1e7b', 'u'), ('\u1e81', 'w'),
        ('\u1e83', 'w'), ('\u1e85', 'w'), ('\u1e9b', 's'), ('\u1ea1', 'a'),
        ('\u1ea3', 'a'), ('\u1eb7', 'u'), ('\u2014', '-'), ('\u2013', '-'),
        ('\u2018', "'"), ('\u2019', "'"), ('\u02bb', "'"), ('\u02bc', "'"),
    ]:
        text = text.replace(old, new)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print('Cleaned: %s' % fname)
