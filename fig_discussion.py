"""Discussion figures (v1.0, manuscript v3.31), all from stored results, no new computation of models.
fig_disc_branches: per-bearing LOBO accuracy of the kinematic and envelope branches and gain of ReliSense over the core (phase 22).
fig_disc_healthy : the six healthy bearings under LOBO, diagnosis before abstention and at kappa = 0.7 (phase 22 posteriors, tau).
fig_disc_networks: ReliSense against WDCNN and the phase 23 networks under five protocols, and per bearing under LOBO."""
import json
import numpy as np
from matplotlib.lines import Line2D
from ieee_style import plt, C, COL2, save, panel
from v326 import P22, FD

PB = {m: P22['protocols']['LOBO'][m]['per_bearing'] for m in ('LR', 'LRX', 'RF', 'ENSX')}
BS = sorted(PB['ENSX']); cls = lambda b: 0 if b.startswith('K0') else (1 if b.startswith('KA') else 2)
CC = {0: ('healthy', C['H']), 1: ('outer race', C['OR']), 2: ('inner race', C['IR'])}
MK = {0: 's', 1: 'o', 2: '^'}
leg = [Line2D([], [], ls='', marker=MK[k], color=CC[k][1], mfc='none' if k == 0 else CC[k][1], mew=0.8, ms=3.6, label=CC[k][0]) for k in CC]
OFF1 = {'K004': (0, 6, 'center'), 'K005': (2, -9, 'left'), 'KI03': (-5, 0, 'right'), 'KI17': (5, 0, 'left'), 'KI05': (5, 0, 'left'), 'K002': (5, 0, 'left')}

# ------------------------------------------------------------------ two branches
fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.5), gridspec_kw={'wspace': 0.28, 'width_ratios': [1, 1.7]})
ax = axs[0]
for b in BS:
    x, y = 100 * PB['LRX'][b], 100 * PB['RF'][b]; ax.plot(x, y, MK[cls(b)], color=CC[cls(b)][1], ms=3.6, mfc='none' if cls(b) == 0 else CC[cls(b)][1], mew=0.8)
    if b in OFF1: ax.annotate(b, (x, y), xytext=OFF1[b][:2], textcoords='offset points', ha=OFF1[b][2], va='center', fontsize=7)
ax.plot([0, 100], [0, 100], color=C['light'], lw=0.6, zorder=0); ax.set_xlim(-4, 104); ax.set_ylim(-4, 104); ax.set_aspect('equal')
ax.set_xlabel('Kinematic branch (%)'); ax.set_ylabel('Envelope branch (%)'); ax.legend(handles=leg, loc='upper left', handletextpad=0.1, borderaxespad=0.2, labelspacing=0.25)
panel(ax, '(a)')
ax = axs[1]; g = np.array([100 * (PB['ENSX'][b] - PB['LR'][b]) for b in BS]); o = np.argsort(-g)
ax.bar(np.arange(len(BS)), g[o], width=0.72, color=[CC[cls(BS[i])][1] for i in o], lw=0)
ax.axhline(0, color=C['ink'], lw=0.5); ax.set_xticks(np.arange(len(BS))); ax.set_xticklabels([BS[i] for i in o], rotation=90, fontsize=7)
ax.set_xlim(-0.7, len(BS) - 0.3); ax.set_ylabel('ReliSense − kinematic core\n(points)'); ax.grid(axis='y', color=C['grid'], lw=0.5); ax.set_axisbelow(True)
panel(ax, '(b)')
save(fig, 'fig_disc_branches.png')
print('better/worse vs core', (g > 1e-9).sum(), (g < -1e-9).sum())

# ------------------------------------------------------------------ healthy bearings
P = FD['prob'].astype(float); y = FD['label'].astype(int); B = np.array([str(b) for b in FD['bearing']]); tb = [str(b) for b in FD['tau_bearings']]
pred = P.argmax(1); conf = P.max(1); tau = np.array([FD['tau'][tb.index(b), 2] for b in B]); acc = conf >= tau
HB = sorted(set(B[y == 0]))
fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.05), sharey=True, gridspec_kw={'wspace': 0.3})
seg0 = [('diagnosed healthy', lambda m: pred[m] == 0, '#bdbdbd'), ('diagnosed outer race', lambda m: pred[m] == 1, C['OR']), ('diagnosed inner race', lambda m: pred[m] == 2, C['IR'])]
seg1 = [('accepted, healthy', lambda m: acc[m] & (pred[m] == 0), '#bdbdbd'), ('accepted, damaged', lambda m: acc[m] & (pred[m] != 0), C['IR']), ('referred', lambda m: ~acc[m], '#e9dcc4')]
for ax, seg, t in ((axs[0], seg0, 'before abstention'), (axs[1], seg1, r'with abstention, $\kappa$ = 0.7')):
    for i, b in enumerate(HB):
        m = B == b; left = 0
        for lab, f, col in seg:
            v = 100 * f(m).mean(); ax.barh(i, v, left=left, height=0.62, color=col, lw=0, label=lab if i == 0 else None)
            if v >= 12 and col != '#bdbdbd' or (v >= 12 and lab.startswith('accepted, h')): pass
            left += v
        if seg is seg1:
            d = 100 * (acc[m] & (pred[m] != 0)).mean()
            ax.text(102, i, f'{d:.1f}', va='center', ha='left', fontsize=7, color=C['ink'])
        else:
            fa = 100 * (pred[m] != 0).mean()
            ax.text(102, i, f'{fa:.1f}', va='center', ha='left', fontsize=7, color=C['ink'])
    ax.set_xlim(0, 100); ax.set_xlabel('Recordings of the bearing (%)'); ax.set_title(t, loc='left')
    ax.text(102, -0.75, 'FA (%)', ha='left', va='bottom', fontsize=7)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=2, handlelength=1.0, columnspacing=0.9, handletextpad=0.4)
axs[0].set_yticks(range(len(HB))); axs[0].set_yticklabels(HB); axs[0].invert_yaxis()
panel(axs[0], '(a)', x=-0.1); panel(axs[1], '(b)', x=-0.02)
save(fig, 'fig_disc_healthy.png')
print('FA before', 100 * (pred[y == 0] != 0).mean(), 'after', 100 * np.mean([(acc[B == b] & (pred[B == b] != 0)).mean() for b in HB]))

# ------------------------------------------------------------------ networks
J = json.load(open('phase23_results.json'))
PR = ['L8_c0', 'L8_all', 'L10', 'A2R', 'LOBO']; PRL = ['L8 cond. 0', 'L8 all', 'L10', 'A2R', 'LOBO']
acc_ = lambda m, p: 100 * np.mean([r['acc'] for r in J[m][p]])
ROWS = [('ReliSense', [95.9, 87.0, 78.7, 77.7, 83.7], C['rs'], 'o', 4.2),
        ('Kinematic core', [93.6, 82.8, 74.6, 73.0, 80.1], C['rs'], 'o', 3.2),
        ('Fault-order-guided ResNet', [acc_('fo_resnet', p) for p in PR], C['be'], 'D', 3.4),
        ('WDCNN, envelope', [70.8, 59.9, 59.6, 69.0, 68.1], C['wd'], 's', 3.2),
        ('ResNet-1D, envelope', [acc_('resnet_env', p) for p in PR], C['rf'], 's', 3.2),
        ('Impulse-response kernel net', [acc_('irk_raw', p) for p in PR], C['be'], 'v', 3.4),
        ('ResNet-1D, raw', [acc_('resnet_raw', p) for p in PR], C['rf'], 'x', 3.4)]
fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.65), gridspec_kw={'wspace': 0.3, 'width_ratios': [1.55, 1]})
ax = axs[0]; dx = np.linspace(-0.27, 0.27, len(ROWS))
for j, (lab, v, col, mk, ms) in enumerate(ROWS):
    mfc = 'none' if lab == 'Kinematic core' else col
    ax.plot(np.arange(5) + dx[j], v, mk, color=col, mfc=mfc, ms=ms, mew=0.8, label=lab, ls='')
for i in range(4): ax.axvline(i + 0.5, color=C['grid'], lw=0.6)
ax.set_xticks(range(5)); ax.set_xticklabels(PRL); ax.set_xlim(-0.5, 4.5); ax.set_ylim(35, 100); ax.set_ylabel('Recording accuracy (%)')
ax.grid(axis='y', color=C['grid'], lw=0.5); ax.set_axisbelow(True)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, handletextpad=0.2, columnspacing=0.8)
panel(ax, '(a)')
ax = axs[1]; fo = {}
for r in J['fo_resnet']['LOBO']: fo.update(r['per_bearing'])
for b in BS:
    x, yv = 100 * PB['ENSX'][b], 100 * fo[b]; ax.plot(x, yv, MK[cls(b)], color=CC[cls(b)][1], ms=3.6, mfc='none' if cls(b) == 0 else CC[cls(b)][1], mew=0.8)
    if abs(x - yv) >= 30: ax.annotate(b, (x, yv), xytext={'K002': (-5, 0), 'KA15': (-5, 0)}.get(b, (5, 0)), textcoords='offset points', ha={'K002': 'right', 'KA15': 'right'}.get(b, 'left'), va='center', fontsize=7)
ax.plot([0, 100], [0, 100], color=C['light'], lw=0.6, zorder=0); ax.set_xlim(-4, 104); ax.set_ylim(-4, 104); ax.set_aspect('equal')
ax.set_xlabel('ReliSense, LOBO (%)'); ax.set_ylabel('Fault-order-guided ResNet (%)'); ax.legend(handles=leg, loc='upper left', handletextpad=0.1, borderaxespad=0.2, labelspacing=0.25)
panel(ax, '(b)')
save(fig, 'fig_disc_networks.png')
d = np.array([100 * (PB['ENSX'][b] - fo[b]) for b in BS]); print('ReliSense vs fo per bearing', (d > 1e-9).sum(), (d < -1e-9).sum(), round(d.mean(), 1))
