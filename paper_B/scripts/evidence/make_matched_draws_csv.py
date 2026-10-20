"""Matched training draws of Table S5 (E2-8 of ReliSense_PaperB_evidence_v2.py), regenerated from its random seed.
The loop below repeats the draw order of E2-8 exactly (seed 0; for each real-damage L8 test bearing, 30 draws of
K002 + three real outer-race + three real inner-race bearings other than the test bearing).
    python make_matched_draws_csv.py ../../results/ReliSense_PaperB_matched_draws_v1.csv
"""
import csv, sys
import numpy as np

SEED = 0
L8_TEST = ['K001', 'KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']
REAL_OR = ['KA04', 'KA15', 'KA16', 'KA22', 'KA30']
REAL_IR = ['KI04', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']

rngm = np.random.default_rng(SEED)
real_test = [b for b in L8_TEST if b != 'K001']
rows = []
for b in real_test:
    for k in range(30):
        ors = [x for x in REAL_OR if x != b]; irs = [x for x in REAL_IR if x != b]
        tb = ['K002'] + list(rngm.choice(ors, 3, replace=False)) + list(rngm.choice(irs, 3, replace=False))
        rows.append((b, k + 1, ' '.join(str(x) for x in tb)))
with open(sys.argv[1], 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['test_bearing', 'draw', 'training_bearings']); w.writerows(rows)
print(f'{len(rows)} draws written to {sys.argv[1]}')
