"""Fig. 12 of Paper B v11.64 without the double-damage bearings KB23, KB24 and KB27, at 600 dpi.
(a) first-harmonic BPFO and BPFI features of the 29 bearings: box statistics of phase 22 (original implementation),
    the KB rows dropped; (b) t-SNE of the 15 standardized features of the 2319 recordings of the 29 bearings (phase 16,
    perplexity 30, PCA initialisation, random_state 0), replacing the PCA that had been fitted with the KB bearings.
    python -I make_fig12_v11_64.py <phase22_views.npz> <phase16_tsne.npz> <out.png>
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

VIEWS, TSNE, OUT = sys.argv[1:4]
C = dict(H='#5b5b5b', OR='#0072b2', IR='#d55e00', grid='#e4e4e4')
CLSC = {0: C['H'], 1: C['OR'], 2: C['IR']}
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.edgecolor': '#1a1a1a', 'legend.frameon': False,
    'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})

V = np.load(VIEWS, allow_pickle=False); T = np.load(TSNE, allow_pickle=False)
keep = np.array([int(l) < 3 for l in V['box_label']])                     # drops KB23, KB24, KB27 (label 3)
bb = [str(b) for b in V['box_bearings'][keep]]; lab = V['box_label'][keep].astype(int)
assert len(bb) == 29 and not any(b.startswith('KB') for b in bb)
assert str(T['names'][0]) == '15 kinematic features'
E = T['E0'].astype(float); yl = T['label'].astype(int); real = T['meta_origin'] == 'real'
assert len(E) == 2319 and len(set(T['bearing'])) == 29 and yl.max() == 2

fig = plt.figure(figsize=(7.0, 3.3)); gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1.0], wspace=0.22, hspace=0.12)
for i, (key, nm) in enumerate((('box_bpfo1', r'$\varphi_{\mathrm{BPFO},1}$'), ('box_bpfi1', r'$\varphi_{\mathrm{BPFI},1}$'))):
    ax = fig.add_subplot(gs[i, 0]); st = V[key][keep].astype(float)
    stats = [dict(whislo=s[0], q1=s[1], med=s[2], q3=s[3], whishi=s[4], fliers=[]) for s in st]
    bp = ax.bxp(stats, positions=range(len(bb)), widths=0.62, patch_artist=True, showfliers=False,
                medianprops=dict(color='white', lw=0.9), whiskerprops=dict(lw=0.5, color='#555555'),
                capprops=dict(lw=0.5, color='#555555'), boxprops=dict(lw=0))
    for p, l in zip(bp['boxes'], lab): p.set_facecolor(CLSC[l])
    ax.set_ylabel(nm); ax.set_xlim(-0.8, len(bb) - 0.2); ax.grid(axis='y', color=C['grid'], lw=0.4); ax.set_axisbelow(True)
    for x in (5.5, 17.5): ax.axvline(x, color='#bbbbbb', lw=0.5)
    ax.set_xticks(range(len(bb)))
    if i == 0:
        ax.set_xticklabels([]); ax.text(-0.11, 1.04, '(a)', transform=ax.transAxes, fontsize=8)
        for x, t in ((2.5, 'healthy'), (11.5, 'artificial'), (23.0, 'real')):
            ax.text(x, 1.02, t, transform=ax.get_xaxis_transform(), ha='center', va='bottom')
    else:
        ax.set_xticklabels(bb, rotation=90, fontsize=7)
ax = fig.add_subplot(gs[:, 1]); idx = np.random.default_rng(0).permutation(len(yl))
for k in (0, 1, 2):
    for r in (False, True):
        m = (yl[idx] == k) & (real[idx] == r)
        if m.any():
            ax.scatter(E[idx][m, 0], E[idx][m, 1], s=3 if r else 2.2, marker='s' if r else 'o', lw=0, alpha=0.55,
                       color=CLSC[k], rasterized=True)
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2', labelpad=2)
ax.text(-0.08, 1.02, '(b)', transform=ax.transAxes, fontsize=8)
h = [Line2D([], [], ls='', marker='o', ms=4, color=CLSC[k], label=t) for k, t in ((0, 'healthy'), (1, 'outer ring'), (2, 'inner ring'))]
h += [Line2D([], [], ls='', marker='s', ms=4, color='#777777', label='real damage')]
ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, handletextpad=0.1, columnspacing=0.9, labelspacing=0.3)
fig.savefig(OUT); plt.close(fig)
from PIL import Image
im = Image.open(OUT); print(f'written {OUT}: {im.width / 600:.2f} x {im.height / 600:.2f} in, {len(bb)} bearings, {len(E)} recordings')
