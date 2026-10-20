"""Paper B variants (v1.0) with the kinematic core in place of ReliSense; from fig_discussion2.py v1.0.
Discussion figures, part 2 (v1.0, manuscript v3.32), from stored results only.
fig_disc_inputs: accuracy on unseen bearings by input (generic or kinematic) and classifier (LR or RF): Paderborn LOBO (phases 19/22,
                 band energies and time + band statistics with LR from Table repr), HUST and CWRU (phases 20/21).
fig_disc_size  : Paderborn LOBO accuracy of ReliSense against extent level, and real damage by damage mode (phase 22, labels_32.csv).
fig_disc_a2r   : real-damage test bearings of L8, trained on artificial damage only (L8) and with real damage of other bearings (LOBO)."""
import csv, json
import numpy as np
from scipy.stats import spearmanr
from matplotlib.lines import Line2D
from ieee_style import plt, C, COL2, save, panel
from v326 import P22, FD, ext

E = P22['protocols']['LOBO']['LR']['per_bearing']   # kinematic core

# ------------------------------------------------------------------ part 4: size does not predict
L = {r['bearing']: r for r in csv.DictReader(open('labels_32.csv')) if r['bearing'] in E}
dam = [b for b in L if L[b]['origin'] != 'healthy']
fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.25), gridspec_kw={'wspace': 0.3, 'width_ratios': [1, 1.15]})
ax = axs[0]; rng = np.random.default_rng(0)
for b in sorted(dam):
    lv = int(L[b]['extent']); real = L[b]['origin'] == 'real'; col = C['OR'] if b.startswith('KA') else C['IR']
    x = lv + (0.12 if real else -0.12) + rng.uniform(-0.06, 0.06)
    ax.plot(x, 100 * E[b], 'o' if b.startswith('KA') else '^', color=col, mfc='white' if real else col, mew=0.8, ms=3.8)
rho, p = spearmanr([int(L[b]['extent']) for b in dam], [E[b] for b in dam])
ax.set_title(f'Spearman ρ = {rho:.2f}, p = {p:.2f}, n = {len(dam)}', loc='right')
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['1\n(≤ 2 mm)', '2\n(2–4.5 mm)', '3\n(4.5–13.5 mm)']); ax.set_xlim(0.5, 3.5); ax.set_ylim(35, 104)
ax.set_xlabel('Extent level'); ax.set_ylabel('LOBO accuracy (%)'); ax.grid(axis='y', color=C['grid'], lw=0.5); ax.set_axisbelow(True)
ax.legend(handles=[Line2D([], [], ls='', marker='o', color=C['OR'], ms=3.8, label='outer, artificial'), Line2D([], [], ls='', marker='o', color=C['OR'], mfc='white', mew=0.8, ms=3.8, label='outer, real'),
                   Line2D([], [], ls='', marker='^', color=C['IR'], ms=3.8, label='inner, artificial'), Line2D([], [], ls='', marker='^', color=C['IR'], mfc='white', mew=0.8, ms=3.8, label='inner, real')],
          loc='upper center', bbox_to_anchor=(0.5, -0.36), ncol=2, handletextpad=0.1, columnspacing=0.8)
panel(ax, '(a)')
ax = axs[1]; real = sorted([b for b in dam if L[b]['origin'] == 'real'], key=lambda b: E[b])
for i, b in enumerate(real):
    ind = 'indent' in L[b]['mode']; col = C['OR'] if b.startswith('KA') else C['IR']
    ax.barh(i, 100 * E[b], height=0.62, color=col if not ind else 'white', edgecolor=col, lw=0.8, hatch='////' if ind else None)
    arr = {'S': 'single', 'R': 'repeated', 'M': 'multiple'}[L[b]['arrangement']] + (', distributed' if L[b]['characteristic'] == 'distributed' else '')
    ax.text(101, i, f"level {L[b]['extent']}, {arr}", va='center', ha='left', fontsize=7)
ax.set_yticks(range(len(real))); ax.set_yticklabels(real); ax.set_xlim(0, 100); ax.set_xlabel('LOBO accuracy (%)')
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc='#8c8c8c', ec='#8c8c8c', label='fatigue pitting'), plt.Rectangle((0, 0), 1, 1, fc='white', ec='#555555', hatch='////', lw=0.8, label='plastic indentation')],
          loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)
panel(ax, '(b)', x=-0.13)
save(fig, 'fig_disc_size_B.png')

# ------------------------------------------------------------------ part 5: artificial-only training vs training with real damage
P = FD['prob'].astype(float); y = FD['label'].astype(int); B = FD['bearing'].astype(str); cnd = FD['condition'].astype(str)
L8a = P22['protocols']['L8_all']['LR']['per_bearing']; L80 = P22['protocols']['L8_c0']['LR']['per_bearing']
TB = [b for b in L8a if b != 'K001']; TB = sorted(TB, key=lambda b: E[b])
P5 = json.load(open('phase5_numbers.json')); lobo0 = {b: P5['lobo_accuracy_bearing_by_condition'][b][0] for b in TB}   # condition 0 = N15_M07_F10
fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.3), sharey=True, gridspec_kw={'wspace': 0.08})
for ax, A, Lb, t in ((axs[0], L80, lobo0, 'condition 0'), (axs[1], L8a, E, 'all four conditions')):
    for i, b in enumerate(TB):
        a, l = 100 * A[b], 100 * Lb[b]; col = C['OR'] if b.startswith('KA') else C['IR']
        ax.plot([l, a], [i, i], color=C['light'], lw=1.2, zorder=1)
        ax.plot(a, i, 'o', color=col, ms=4.2, zorder=2); ax.plot(l, i, 'o', color=col, mfc='white', mew=0.9, ms=4.2, zorder=3)
    ax.set_xlim(25, 104); ax.set_xlabel('Accuracy (%)'); ax.set_title(t, loc='left'); ax.grid(axis='x', color=C['grid'], lw=0.5); ax.set_axisbelow(True)
    print(t, 'L8 mean', round(100 * np.mean([A[b] for b in TB]), 1), 'LOBO mean', round(100 * np.mean([Lb[b] for b in TB]), 1), 'L8 higher on', sum(A[b] > Lb[b] + 1e-9 for b in TB), 'lower on', sum(A[b] < Lb[b] - 1e-9 for b in TB))
axs[0].set_yticks(range(len(TB))); axs[0].set_yticklabels(TB)
panel(axs[0], '(a)', x=-0.1); panel(axs[1], '(b)')
fig.legend(handles=[Line2D([], [], ls='', marker='o', color='#555555', ms=4.2, label='trained on artificial damage only (L8)'),
                    Line2D([], [], ls='', marker='o', color='#555555', mfc='white', mew=0.9, ms=4.2, label='trained with real damage of other bearings (LOBO)')],
           loc='lower center', bbox_to_anchor=(0.5, -0.17), ncol=2)
save(fig, 'fig_disc_a2r_B.png')
