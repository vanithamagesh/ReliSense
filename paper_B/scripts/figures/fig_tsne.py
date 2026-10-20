"""Fig. t-SNE (v1.0) from phase16_tsne.npz: four representations of all 2319 recordings of the 29 single-damage bearings.
Top: coloured by damage class (filled: artificial damage or healthy, open: real damage). Bottom: coloured by bearing.
Panel titles give the silhouette scores computed in the standardized feature space (not in the embedding)."""
import numpy as np
import matplotlib
from ieee_style import plt, C, COL2, save

Z = dict(np.load('phase16_tsne.npz', allow_pickle=True)); y = Z['label']; B = Z['bearing']; org = Z['meta_origin']
TITLE = ['15 kinematic features', 'Order spectrum', '32 band energies', 'Time + band statistics']
CC = {0: C['H'], 1: C['OR'], 2: C['IR']}
ub = sorted(set(B)); cmap = [matplotlib.colormaps['tab20'](i % 20) if i < 20 else matplotlib.colormaps['tab20b'](2 * (i - 20) + 1) for i in range(len(ub))]
rng = np.random.default_rng(0); order = rng.permutation(len(y))
fig, axs = plt.subplots(2, 4, figsize=(COL2, 3.75), gridspec_kw={'hspace': 0.28, 'wspace': 0.06})
for j in range(4):
    E = Z[f'E{j}'].astype(float); sc, sb = float(Z[f'sil_class{j}']), float(Z[f'sil_bearing{j}'])
    ax = axs[0, j]
    for k in (0, 1, 2):
        for real in (False, True):
            m = (y == k) & ((org == 'real') == real)
            if m.any():
                ax.scatter(E[m, 0], E[m, 1], s=1.4 if not real else 2.2, lw=0.3 if real else 0, facecolor='none' if real else CC[k],
                           edgecolor=CC[k] if real else 'none', alpha=0.75, rasterized=True)
    ax.set_title(f'({chr(97 + j)}) {TITLE[j]}\nsilhouette, class: {sc:.3f}', loc='left', fontsize=6.6)
    ax = axs[1, j]; bi = np.array([ub.index(b) for b in B])
    ax.scatter(E[order, 0], E[order, 1], s=1.2, lw=0, c=[cmap[i] for i in bi[order]], alpha=0.85, rasterized=True)
    ax.set_title(f'({chr(101 + j)}) silhouette, bearing: {sb:.3f}', loc='left', fontsize=6.6)
for ax in axs.ravel(): ax.set_xticks([]); ax.set_yticks([])
from matplotlib.lines import Line2D
h = [Line2D([], [], ls='', marker='o', ms=3, color=CC[0], label='healthy'), Line2D([], [], ls='', marker='o', ms=3, color=CC[1], label='OR, artificial'),
     Line2D([], [], ls='', marker='o', ms=3, mfc='none', mec=CC[1], label='OR, real'), Line2D([], [], ls='', marker='o', ms=3, color=CC[2], label='IR, artificial'),
     Line2D([], [], ls='', marker='o', ms=3, mfc='none', mec=CC[2], label='IR, real')]
fig.legend(handles=h, loc='upper center', ncol=5, fontsize=6.3, bbox_to_anchor=(0.5, 1.035), handletextpad=0.2, columnspacing=1.2)
fig.text(0.5, 0.075, 'Bottom row: one colour per bearing (29 bearings)', ha='center', fontsize=6.3, color='#444444')
save(fig, 'fig_tsne.png')
