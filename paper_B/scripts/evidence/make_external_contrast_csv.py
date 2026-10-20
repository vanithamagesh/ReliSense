"""Contrast and accuracy of the 16 damaged external bearings (Fig. 11b, Section IV-B; Table S10 and S11).
Gamma: natural-log amplitude contrast of (24) from phase12_numbers.json; accuracy: probe (P + LR), HUST LOTO and
CWRU LOSO, 3-class, per unit from phase10_results.json.
    python make_external_contrast_csv.py ../../results/phase12_numbers.json ../../results/phase10_results.json ../../results/external_contrast_accuracy.csv
"""
import csv, json, sys
from scipy.stats import spearmanr

N12, P10, OUT = sys.argv[1:4]
ext = json.load(open(N12))['external']; j = json.load(open(P10))
acc = {'HUST': j['hust']['3-class']['LOTO']['ReliSense (P + LR)']['per_unit'],
       'CWRU': j['cwru']['3-class']['LOSO']['ReliSense (P + LR)']['per_unit']}
gam = {'HUST': ext['hust_contrast'], 'CWRU': ext['cwru_contrast']}
race = lambda u: 'OR' if u.startswith('O') else 'IR' if u.startswith('I') else None
rows = [(ds, u, race(u), gam[ds][u], round(100 * acc[ds][u], 1)) for ds in ('HUST', 'CWRU') for u in sorted(gam[ds]) if race(u)]
with open(OUT, 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['dataset', 'bearing', 'race', 'gamma', 'accuracy']); w.writerows(rows)
r = spearmanr([x[3] for x in rows], [x[4] for x in rows])
print(f'{len(rows)} bearings; Spearman rho {r.correlation:.3f}, p {r.pvalue:.4f}; '
      f'Gamma > 1.1: {sum(x[3] > 1.1 for x in rows)} bearings, lowest accuracy among them {min(x[4] for x in rows if x[3] > 1.1):.0f}%')
