"""Paper B variant of fig_disc_inputs (v1.0): generic inputs, kinematic core and envelope features only."""
"""Discussion figures, part 2 (v1.0, manuscript v3.32), from stored results only.
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

E = P22['protocols']['LOBO']['ENSX']['per_bearing']

# ------------------------------------------------------------------ part 1: input generality vs classifier flexibility
SET = [('Paderborn\nLOBO', None, None), ('HUST\nLOTO', 'hust', 'LOTO'), ('HUST\n1TYPE', 'hust', '1TYPE'), ('CWRU\nLOSO', 'cwru', 'LOSO'), ('CWRU\n1SIZE', 'cwru', '1SIZE')]
PAD = {'band energies + LR': 50.1, 'time + band + LR': 55.2, 'LR': 100 * P22['protocols']['LOBO']['LR']['acc'], 'LRX': 100 * P22['protocols']['LOBO']['LRX']['acc'],
       'RFenv': 100 * P22['protocols']['LOBO']['RF']['acc'], 'ENSX': 100 * P22['protocols']['LOBO']['ENSX']['acc']}
M = [('band energies + LR', 'Band energies, LR', '#9a9a9a', 'o', True), ('time + band + RF', 'Time + band statistics, RF', '#9a9a9a', 's', True),
     ('LR', 'Kinematic core, LR', C['rs'], 'o', False), ('RFenv', 'Envelope features, RF', C['rs'], 's', False)]
fig, ax = plt.subplots(figsize=(COL2, 2.35)); dx = np.linspace(-0.3, 0.3, len(M))
for j, (k, lab, col, mk, gen) in enumerate(M):
    v = []
    for i, (_, ds, pr) in enumerate(SET):
        if ds is None: val = PAD['time + band + LR'] if k == 'time + band + RF' else PAD[k]
        else: val = 100 * ext(ds, pr, k)['acc']
        v.append(val)
        mk_ = 'o' if (ds is None and k == 'time + band + RF') else mk
        ax.plot(i + dx[j], val, mk_, color=col, mfc='white' if gen else col, mew=0.9, ms=6 if mk == '*' else 4)
    print(lab, [round(x, 1) for x in v])
for i in range(len(SET) - 1): ax.axvline(i + 0.5, color=C['grid'], lw=0.6)
ax.set_xticks(range(len(SET))); ax.set_xticklabels([s[0] for s in SET]); ax.set_xlim(-0.5, len(SET) - 0.5); ax.set_ylim(35, 92)
ax.set_ylabel('Accuracy on unseen\nbearings (%)'); ax.grid(axis='y', color=C['grid'], lw=0.5); ax.set_axisbelow(True)
h = [Line2D([], [], ls='', marker=mk, color=col, mfc='white' if gen else col, mew=0.9, ms=6 if mk == '*' else 4, label=lab) for _, lab, col, mk, gen in M]
ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=2, handletextpad=0.2, columnspacing=1.0)
save(fig, 'fig_disc_inputs_B.png')

