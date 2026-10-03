"""Redraw of the Phase 6 figures for manuscript v2.3. Every value is read from p6figs/phase6_numbers.json
(the real Step 12 output); only the layout changes (no overlapping labels, random-order range shown in E1)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'rs_figs'); os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(HERE, 'p6figs', 'phase6_numbers.json')))
BLUE, ORANGE, AQUA, GREY, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#9a9a93', '#222222', '#5f5e58'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': '#b5b4ad', 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5df', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'savefig.dpi': 400, 'savefig.bbox': 'tight'})
save = lambda fig, n: (fig.savefig(os.path.join(OUT, n + '.png')), plt.close(fig))

# ---------------------------------------------------------------- E1 wrong-physics control
sc = sorted(R['E1_scaled'], key=float); s = [float(k) for k in sc]; dr = R['E1_random']['draws']
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, key, c, title in ((axs[0], 'L8_cond0', ORANGE, '(a) L8, condition 0'), (axs[1], 'LOBO', BLUE, '(b) leave one bearing out')):
    v = np.array([R['E1_scaled'][k][key] for k in sc]) * 100; rv = np.array([d[key] for d in dr]) * 100
    ax.axhspan(rv.min(), rv.max(), color=GREY, alpha=0.18, lw=0)
    ax.axhline(rv.mean(), color=MUTED, ls=':', lw=1)
    ax.plot(s, v, marker='o', color=c, lw=1.8, ms=5)
    ax.plot([1.0], [v[s.index(1.0)]], marker='o', ms=9, mfc='none', mec=INK, mew=1.2)
    ax.axvline(1.0, color=INK, lw=0.7, ls='--'); ax.axhline(100 / 3, color=MUTED, lw=0.6)
    ax.set_title(title, fontsize=8.5, loc='left'); ax.set_xlabel('scale applied to all fault orders')
    ax.text(0.80, rv.max() + 1.2, 'range of 20 random order sets', fontsize=6.5, color=MUTED)
    ax.text(0.80, rv.mean() - 4.0, 'random mean', fontsize=6.5, color=MUTED, ha='left')
    ax.text(1.005, 36, 'true 6203\nkinematics', fontsize=6.5, color=INK)
axs[0].set_ylabel('recording accuracy (%)'); axs[0].set_ylim(30, 100)
fig.tight_layout(); save(fig, 'figE1_wrong_physics')

# ---------------------------------------------------------------- E2 speed transfer
e2 = R['E2_speed_transfer']; names = ['kinematic peaks\n(orders) + LR', 'band energies\n(Hz) + LR', 'time statistics\n+ LR', 'time + band\nenergies + RF']
fig, ax = plt.subplots(figsize=(6.0, 3.0)); x = np.arange(4); w = 0.36
for j, (key, c, lab) in enumerate((('train1500_test1500', BLUE, 'test at 1500 rpm (speed seen in training)'), ('train1500_test900', ORANGE, 'test at 900 rpm (unseen speed)'))):
    vals = [v[key] * 100 for v in e2.values()]
    ax.bar(x + (j - 0.5) * w, vals, w * 0.92, color=c, label=lab)
    for i, val in enumerate(vals): ax.text(i + (j - 0.5) * w, val + 1.5, f'{val:.1f}', ha='center', fontsize=6.5, color=INK)
ax.axhline(100 / 3, color=MUTED, lw=0.6, ls=':'); ax.text(-0.68, 34.8, 'chance', fontsize=6.5, color=MUTED, ha='left'); ax.set_xlim(-0.72, 3.5)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.2); ax.set_ylim(0, 112); ax.set_ylabel('held-out-bearing accuracy (%)')
ax.legend(fontsize=7, loc='upper right', ncol=1); save(fig, 'figE2_speed_transfer')

# ---------------------------------------------------------------- E3 representation ladder
q, e3 = R['E3_quoted_phase4'], R['E3_ladder']
rows = [('WDCNN,\nraw vibration', q['WDCNN raw']), ('WDCNN,\nenvelope', q['WDCNN envelope']), ('order spectrum\n+ LR', e3['order spectrum + LR']),
        ('order spectrum\n+ 1-D CNN', e3['order spectrum + 1-D CNN']), ('15 kinematic\npeaks + LR', e3['kinematic peaks + LR'])]
fig, ax = plt.subplots(figsize=(6.6, 3.4)); x = np.arange(len(rows)); w = 0.26
for j, (key, c, lab) in enumerate((('L8_cond0', BLUE, 'L8, condition 0'), ('L8_all', AQUA, 'L8, all conditions'), ('LOBO', ORANGE, 'LOBO'))):
    ax.bar(x + (j - 1) * w, [r[1][key] * 100 for r in rows], w * 0.92, color=c, label=lab)
ax.axhline(100 / 3, color=MUTED, lw=0.6, ls=':')
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=7.2); ax.set_ylim(0, 122); ax.set_yticks(range(0, 101, 20))
ax.set_ylabel('recording accuracy (%)'); ax.legend(fontsize=7, ncol=3, loc='upper left', bbox_to_anchor=(0, 1.0))
ax.annotate('', xy=(4.45, 106), xytext=(-0.45, 106), arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.8))
ax.text(2, 108, 'raw waveform  →  envelope  →  order axis  →  kinematic orders only', ha='center', fontsize=7, color=MUTED)
ax.grid(axis='x', visible=False); save(fig, 'figE3_ladder')

# ---------------------------------------------------------------- E5 data efficiency
e5 = R['E5_data_efficiency']
fig, ax = plt.subplots(figsize=(5.4, 3.2))
for (k, v), c, off, mk in zip(e5.items(), (BLUE, ORANGE, AQUA), (-0.12, 0.0, 0.12), ('o', 's', 'D')):
    ks = sorted(v, key=int); mu = np.array([v[t][0] for t in ks]) * 100; sd = np.array([v[t][1] for t in ks]) * 100
    xs = np.array([int(t) for t in ks]) + off
    ax.errorbar(xs, mu, yerr=sd, color=c, marker=mk, ms=5, lw=1.6, capsize=2.5, elinewidth=0.8, label=k)
ax.axhline(100 / 3, color=MUTED, lw=0.6, ls=':'); ax.text(5.35, 34.5, 'chance', fontsize=6.5, color=MUTED, ha='right')
ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xlabel('training bearings per class'); ax.set_ylabel('accuracy on new bearings (%)')
ax.set_ylim(15, 105); ax.legend(fontsize=7, loc='upper left', ncol=1); save(fig, 'figE5_data_efficiency')
print('ok')
