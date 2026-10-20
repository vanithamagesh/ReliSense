"""Fig. 13 of Paper B v11.70: per-bearing LOBO accuracy of the probe, its kurtogram and pre-whitened variants, the
threshold rule and WDCNN on the envelope, redrawn with 8-pt labels at the full text width (7.0 in), 600 dpi.
Same data and layout as fig_matrix_B in fig_B_core.py (phase 2 per-bearing results, phase 19 LR = phase 2 lr_fixed,
WDCNN on the envelope, labels_32.csv).
    python -E -P make_fig13_v11_70.py <phase2_per_bearing.json> <phase19_results.json> <wdcnn_env_lobo.json> <labels_32.csv> <out.png>
"""
import csv, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P2F, P19F, WDF, LABF, OUT = sys.argv[1:6]
FS = 8
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
                     'font.size': FS, 'axes.linewidth': 0.5, 'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
INK, LIGHT = '#1a1a1a', '#bdbdbd'
LAB = {r['bearing']: r for r in csv.DictReader(open(LABF))}
P2 = json.load(open(P2F)); P19 = json.load(open(P19F)); WD = json.load(open(WDF))
MOD = [('LR', 'Kin.\ncore'), ('lr_sk', 'Core,\nkurtogr.'), ('lr_cpw', 'Core,\npre-wh.'), ('rule', 'Thresh.\nrule'), ('wdcnn', 'WDCNN\n(env.)')]
acc = {m: {} for m, _ in MOD}
for r in P2:
    if r['protocol'] == 'LOBO' and r['model'] in acc: acc[r['model']][r['bearing']] = 100 * r['acc_mean']
acc['wdcnn'] = {b: 100 * v for b, v in WD.items()}
acc['LR'] = {b: 100 * v for b, v in P19['lobo_per_bearing']['LR'].items()}
BEAR = sorted(acc['LR'])
SIZE = {'1': '≤ 2 mm', '2': '2–4.5 mm', '3': '4.5–13.5 mm'}
SH = {'EDM': 'EDM', 'drilling': 'drilling', 'electric engraver': 'engraver', 'fatigue: pitting': 'pitting',
      'plastic deformation: indentations': 'indentation'}
hb = sorted([b for b in BEAR if LAB[b]['origin'] == 'healthy'], key=lambda b: float(LAB[b]['mode'].split()[1].lstrip('>')))
key = lambda b: (LAB[b]['location'] != 'OR', LAB[b]['extent'], LAB[b]['mode'], b)
art = sorted([b for b in BEAR if LAB[b]['origin'] == 'artificial'], key=key)
real = sorted([b for b in BEAR if LAB[b]['origin'] == 'real'], key=key)
order = hb + art + real
A = np.array([[acc[m][b] for m, _ in MOD] for b in order])
assert A.shape == (29, 5)

fig = plt.figure(figsize=(7.4, 6.3)); gs = fig.add_gridspec(1, 3, width_ratios=[2.75, 2.35, 0.07], wspace=0.04)
at, ax, cax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
im = ax.imshow(A, cmap='Blues', vmin=0, vmax=100, aspect='auto')
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        ax.text(j, i, f'{A[i, j]:.0f}', ha='center', va='center', fontsize=FS, color='white' if A[i, j] > 62 else INK)
ax.set_xticks(range(len(MOD))); ax.set_xticklabels([m[1] for m in MOD], fontsize=FS); ax.xaxis.tick_top()
ax.set_yticks([]); ax.tick_params(length=0)
for yy in (len(hb) - 0.5, len(hb) + len(art) - 0.5): ax.axhline(yy, color='white', lw=1.6)
cb = fig.colorbar(im, cax=cax); cb.set_label('LOBO accuracy (%)', fontsize=FS); cb.ax.tick_params(labelsize=FS, width=0.5)
cb.outline.set_linewidth(0.4)
at.set_xlim(0, 1); at.set_ylim(len(order) - 0.5, -0.5); at.axis('off')
xs = [0.0, 0.15, 0.36, 0.47, 0.69, 0.92]
for x, h in zip(xs, ['Bearing', 'Origin', 'Race', 'Size', 'Method', 'Arr.']):
    at.text(x, -1.3, h, fontsize=FS, va='center', style='italic')
for i, b in enumerate(order):
    l = LAB[b]
    v = [b, 'healthy', '–', '–', 'run-in ' + l['mode'].split()[1] + ' h', '–'] if l['origin'] == 'healthy' else \
        [b, l['origin'], l['location'], SIZE[l['extent']], SH[l['mode']], l['arrangement'] + ('/d' if l['characteristic'] == 'distributed' else '')]
    for x, t in zip(xs, v): at.text(x, i, t, fontsize=FS, va='center')
for yy in (len(hb) - 0.5, len(hb) + len(art) - 0.5): at.axhline(yy, color=LIGHT, lw=0.5)
fig.savefig(OUT); plt.close(fig)
from PIL import Image
w, h = Image.open(OUT).size
print(f'written {OUT}: {w / 600:.2f} x {h / 600:.2f} in; all labels {FS} pt')
