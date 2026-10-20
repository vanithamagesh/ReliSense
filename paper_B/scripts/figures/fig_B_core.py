"""Paper B figures (v1.0): the kinematic core in place of ReliSense, from stored results only.
fig_conditions_B  (a), (c) kinematic core by operating condition (phase 5); (b) speed transfer (phase 6, E2), as fig_conditions
fig_matrix_B      per-bearing LOBO accuracy of the core, its kurtogram and pre-whitened variants, the threshold rule and WDCNN (phases 2, 4, 19)
fig_line_B        HUST pitch, defect-line contrast and CWRU size with the core accuracy (phases 12, 20, 21)
fig_disc_size_B   extent level and damage mode against core LOBO accuracy (phase 22, labels_32.csv)
fig_disc_a2r_B    L8 (artificial only) against LOBO (with real damage) for the core (phases 5, 22)"""
from v326 import P19, P22, ext
from scipy.stats import spearmanr
import json, csv
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from ieee_style import plt, C, COL1, COL2, panel, save

P5 = json.load(open('phase5_numbers.json')); P6 = json.load(open('phase6_numbers.json'))
J = json.load(open('phase10_results.json')); N12 = json.load(open('phase12_numbers.json'))
LAB = {r['bearing']: r for r in csv.DictReader(open('labels_32.csv'))}
CL = ['H', 'OR', 'IR']


import json
import numpy as np
from matplotlib.lines import Line2D
from ieee_style import plt, C, COL1, COL2, save
from v326 import P22, VW

P6 = json.load(open('phase6_numbers.json'))
FAM = ['BPFO', 'BPFI', 'BSF', 'FTF', 'shaft']; KEYS = [f'{k}_h{h}' for k in FAM for h in (1, 2, 3)]
n_, d_, D_ = 8, 6.75, 28.55; q = d_ / D_; BPFO, BPFI = n_ / 2 * (1 - q), n_ / 2 * (1 + q)
CLSC = {0: C['H'], 1: C['OR'], 2: C['IR'], 3: C['B']}


# ------------------------------------------------------------------ operating conditions
CN = ['N15_M07_F10', 'N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04']; CL = ['1500 rpm', '900 rpm', '0.1 Nm', '400 N']
fig = plt.figure(figsize=(COL2, 4.3)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.75, wspace=0.3)
ax = fig.add_subplot(gs[0, 0]); xs = np.arange(4); w = 0.2; P5 = json.load(open('phase5_numbers.json')); BC = {c: dict(acc=P5['lobo_accuracy_per_condition'][c], recall_H=P5['lobo_recall_per_condition_H_OR_IR'][c][0], recall_OR=P5['lobo_recall_per_condition_H_OR_IR'][c][1], recall_IR=P5['lobo_recall_per_condition_H_OR_IR'][c][2]) for c in CN}
series = [('all classes', [100 * BC[c]['acc'] for c in CN], C['rs']), ('recall H', [100 * BC[c]['recall_H'] for c in CN], C['H']),
          ('recall OR', [100 * BC[c]['recall_OR'] for c in CN], C['OR']), ('recall IR', [100 * BC[c]['recall_IR'] for c in CN], C['IR'])]
for k, (lab, vals, col) in enumerate(series): ax.bar(xs + (k - 1.5) * w, vals, w * 0.9, color=col, label=lab)
for x, a in zip(xs, series[0][1]): ax.text(x - 1.5 * w, a + 2, f'{a:.1f}', ha='center', fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels(CL); ax.set_ylim(0, 132); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel('LOBO accuracy, recall (%)'); ax.tick_params(axis='x', length=0)
ax.legend(ncol=4, loc='upper center', handlelength=0.9, columnspacing=0.6, handletextpad=0.3, bbox_to_anchor=(0.5, 1.0), fontsize=7)
ax.set_title('(a) Kinematic core by operating condition', loc='left')
ax = fig.add_subplot(gs[0, 1]); E2 = P6['E2_speed_transfer']; names = list(E2); lab2 = ['kinematic\n(orders)', 'band energies\n(Hz)', 'time\nstatistics', 'time + band\n(RF)']
for k, (key, col, hatch, lab) in enumerate((('train1500_test1500', C['rs'], None, 'tested at 1500 rpm'), ('train1500_test900', 'white', '////', 'tested at 900 rpm'))):
    vals = [100 * E2[n][key] for n in names]
    ax.bar(np.arange(4) + (k - 0.5) * 0.38, vals, 0.35, color=col, edgecolor=C['rs'], hatch=hatch, lw=0.6, label=lab)
    for x, y_ in zip(np.arange(4) + (k - 0.5) * 0.38, vals): ax.text(x, y_ + 2, f'{y_:.0f}', ha='center', fontsize=7)
ax.axhline(100 / 3, color='#999999', lw=0.5, ls=(0, (2, 1.5))); ax.set_xticks(range(4)); ax.set_xticklabels(lab2); ax.tick_params(axis='x', length=0)
ax.set_ylim(0, 132); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel('LOBO accuracy (%)'); ax.legend(loc='upper right', ncol=2, columnspacing=0.8, handlelength=1.2)
ax.set_title('(b) Kinematic core and generic inputs, trained at 1500 rpm', loc='left')
ax = fig.add_subplot(gs[1, :]); BB = P5['lobo_accuracy_bearing_by_condition']; bs = list(P22['lobo_bearing_by_condition'])
M = np.array([BB[b] for b in bs]).T * 100
im = ax.imshow(M, cmap='Blues', vmin=0, vmax=100, aspect='auto')
ax.set_yticks(range(4)); ax.set_yticklabels(CL); ax.set_xticks(range(len(bs))); ax.set_xticklabels(bs, rotation=90); ax.tick_params(length=0)
for x in (5.5, 17.5): ax.axvline(x, color='white', lw=2)
for x, t in ((2.5, 'healthy'), (11.5, 'artificial damage'), (23.5, 'real damage')): ax.text(x, -0.62, t, ha='center', va='bottom')
cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label('Accuracy (%)'); cb.outline.set_linewidth(0.4)
ax.set_title('(c) Kinematic core, LOBO accuracy of every bearing by condition', loc='left', pad=17)
save(fig, 'fig_conditions_B.png')

# ------------------------------------------------------------------ damage matrix (per-bearing LOBO accuracy of six models)
P2 = json.load(open('phase2_per_bearing.json')); WD = json.load(open('wdcnn_env_lobo.json'))
MOD = [('LR', 'Kin.\ncore'), ('lr_sk', 'Core,\nkurtogr.'), ('lr_cpw', 'Core,\npre-wh.'), ('rule', 'Thresh.\nrule'), ('wdcnn', 'WDCNN\n(env.)')]
acc = {m: {} for m, _ in MOD}
for r in P2:
    if r['protocol'] == 'LOBO' and r['model'] in acc: acc[r['model']][r['bearing']] = 100 * r['acc_mean']
acc['wdcnn'] = {b: 100 * v for b, v in WD.items()}
acc['LR'] = {b: 100 * v for b, v in P19['lobo_per_bearing']['LR'].items()}
BEAR = sorted(acc['LR'])
SIZE = {'1': '≤ 2 mm', '2': '2–4.5 mm', '3': '4.5–13.5 mm'}
SH = {'EDM': 'EDM', 'drilling': 'drilling', 'electric engraver': 'engraver', 'fatigue: pitting': 'pitting', 'plastic deformation: indentations': 'indentation'}
hb = sorted([b for b in BEAR if LAB[b]['origin'] == 'healthy'], key=lambda b: float(LAB[b]['mode'].split()[1].lstrip('>')))
key = lambda b: (LAB[b]['location'] != 'OR', LAB[b]['extent'], LAB[b]['mode'], b)
art = sorted([b for b in BEAR if LAB[b]['origin'] == 'artificial'], key=key); real = sorted([b for b in BEAR if LAB[b]['origin'] == 'real'], key=key)
order = hb + art + real; A = np.array([[acc[m][b] for m, _ in MOD] for b in order])
fig = plt.figure(figsize=(COL2, 5.6)); gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 2.0, 0.08], wspace=0.04)
at, ax, cax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
im = ax.imshow(A, cmap='Blues', vmin=0, vmax=100, aspect='auto')
for i in range(A.shape[0]):
    for j in range(A.shape[1]): ax.text(j, i, f'{A[i, j]:.0f}', ha='center', va='center', fontsize=7, color='white' if A[i, j] > 62 else C['ink'])
ax.set_xticks(range(len(MOD))); ax.set_xticklabels([m[1] for m in MOD], fontsize=6.5); ax.xaxis.tick_top(); ax.set_yticks([]); ax.tick_params(length=0)
for yy in (len(hb) - 0.5, len(hb) + len(art) - 0.5): ax.axhline(yy, color='white', lw=1.6)
cb = fig.colorbar(im, cax=cax); cb.set_label('LOBO accuracy (%)'); cb.outline.set_linewidth(0.4)
at.set_xlim(0, 1); at.set_ylim(len(order) - 0.5, -0.5); at.axis('off')
xs = [0.0, 0.16, 0.37, 0.48, 0.70, 0.93]
for x, h in zip(xs, ['Bearing', 'Origin', 'Race', 'Size', 'Method', 'Arr.']): at.text(x, -1.25, h, fontsize=7, va='center', style='italic')
for i, b in enumerate(order):
    l = LAB[b]
    v = [b, 'healthy', '–', '–', 'run-in ' + l['mode'].split()[1] + ' h', '–'] if l['origin'] == 'healthy' else \
        [b, l['origin'], l['location'], SIZE[l['extent']], SH[l['mode']], l['arrangement'] + ('/d' if l['characteristic'] == 'distributed' else '')]
    for x, t in zip(xs, v): at.text(x, i, t, fontsize=6.5, va='center')
for yy in (len(hb) - 0.5, len(hb) + len(art) - 0.5): at.axhline(yy, color=C['light'], lw=0.5)
save(fig, 'fig_matrix_B.png')

M3 = ['LR', 'time + band + RF', 'band energies + LR']
ML = {'LR': 'kinematic core', 'time + band + RF': 'time + band, RF', 'band energies + LR': 'band energies, LR'}
MC = {'LR': C['rs'], 'time + band + RF': C['rf'], 'band energies + LR': C['be']}; MK = dict(zip(M3, 'osD'))
# ------------------------------------------------------------------ line strength across bearing designs (final ReliSense)
E12 = N12['external']; RSM = 'LR'; OTH = ['time + band + RF', 'band energies + LR']
hu = ext('hust', 'LOTO', RSM)['per_unit']; cw = ext('cwru', 'LOSO', RSM)['per_unit']
fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.6), gridspec_kw={'wspace': 0.42, 'width_ratios': [1, 1.2, 1]})
GEO = E12['hust_by_type']; types = list(GEO)
for m in [RSM] + OTH:
    u = ext('hust', 'LOTO', m)['per_unit']; acc_t = [100 * np.mean([u[c + t[-1]] for c in 'NOI']) for t in types]
    axs[0].plot([GEO[t]['pitch_mm'] for t in types], acc_t, marker=MK[m], ms=3.4, lw=0.8, color=MC[m], mfc=MC[m] if m == RSM else 'white', mew=0.7, label=ML[m])
axs[0].set_xticks([GEO[t]['pitch_mm'] for t in types]); axs[0].set_xticklabels([t[-2:] for t in types]); axs[0].set_xlabel('Unseen type 62xx (pitch 33.5–60 mm)'); axs[0].set_ylabel('Accuracy (%)'); axs[0].set_ylim(0, 105); axs[0].set_title('(a) HUST, LOTO', loc='left')
hc, cc = E12['hust_contrast'], E12['cwru_contrast']; xs_, ys_ = [], []
for dct, src, filled, nm in ((hc, hu, True, 'HUST'), (cc, cw, False, 'CWRU')):
    for u, v in dct.items():
        if u[0] not in 'OI': continue
        y = 100 * src[u]; col = C['OR'] if u[0] == 'O' else C['IR']; xs_.append(v); ys_.append(y)
        axs[1].plot(v, y, 'o' if u[0] == 'O' else '^', ms=3.8, color=col, mfc=col if filled else 'white', mew=0.7)
        if y < 99:
            lab = f'{nm} {"OR" if u[0] == "O" else "IR"} {("620" + u[1]) if nm == "HUST" else u[2:]}'
            off = {'HUST IR 6204': (5, 7), 'HUST OR 6208': (5, 9), 'CWRU OR 014': (5, -8), 'HUST IR 6207': (5, -6), 'HUST OR 6206': (5, 2)}.get(lab, (5, 0))
            axs[1].annotate(lab, (v, y), xytext=off, textcoords='offset points', fontsize=7, va='center')
rho, pv = spearmanr(xs_, ys_)
axs[1].text(0.97, 0.3, f'Spearman ρ = {rho:.2f}\n' + (f'p = {pv:.3f}' if pv >= 0.001 else 'p < 0.001') + f', n = {len(xs_)}', transform=axs[1].transAxes, ha='right', va='bottom')
axs[1].set_xlabel('Signature contrast (log ratio)'); axs[1].set_ylabel('Kinematic-core accuracy (%)'); axs[1].set_ylim(-16, 108); axs[1].set_xlim(-0.1, 4.1); axs[1].set_title('(b) Defect line and accuracy', loc='left')
hl = [Line2D([], [], ls='', marker='o', color=C['OR'], ms=3.8, label='outer race'), Line2D([], [], ls='', marker='^', color=C['IR'], ms=3.8, label='inner race'),
      Line2D([], [], ls='', marker='o', color='#777777', ms=3.8, label='HUST'), Line2D([], [], ls='', marker='o', mfc='white', mec='#777777', ms=3.8, label='CWRU')]
pass  # markers explained in the caption
x3 = np.arange(3); sz = ['007', '014', '021']
for k, m in enumerate([RSM] + OTH):
    u = ext('cwru', 'LOSO', m)['per_unit']
    axs[2].bar(x3 + (k - 1) * 0.26, [100 * np.mean([u['IR' + s_], u['OR' + s_]]) for s_ in sz], 0.24, color=MC[m] if k == 0 else 'white', edgecolor=MC[m], lw=0.7, hatch=None if k == 0 else ('////' if k == 1 else '....'))
axs[2].set_xticks(x3); axs[2].set_xticklabels(['0.18', '0.36', '0.53']); axs[2].set_ylim(0, 105)
axs[2].set_xlabel('Unseen fault diameter (mm)'); axs[2].set_ylabel('Accuracy, IR and OR (%)'); axs[2].set_title('(c) CWRU, LOSO', loc='left')
h_ = [Line2D([], [], color=MC[m], marker=MK[m], ms=3.4, mfc=MC[m] if m == RSM else 'white', lw=0.8, label=ML[m]) for m in [RSM] + OTH]
fig.legend(handles=h_, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))
print('contrast vs ReliSense accuracy: rho', round(rho, 3), 'p', round(pv, 4), 'n', len(xs_))
save(fig, 'fig_line_B.png')
