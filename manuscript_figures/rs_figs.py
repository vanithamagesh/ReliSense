"""Figures for ReliSense manuscript v2.1. All values are copied from the real study outputs
(Steps 5, 6, 7, 8, 9, 10 and the v1.4 15-bearing tables); nothing is simulated."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
import numpy as np, os

OUT = os.path.join(os.path.dirname(__file__), 'rs_figs'); os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, GREY, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#9a9a93', '#222222', '#5f5e58'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': '#b5b4ad', 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5df', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'savefig.dpi': 600, 'savefig.bbox': 'tight'})

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.png')); plt.close(fig)

# ---------------------------------------------------------------- Fig 1: framework
fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 42)
def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec='#8c8b84', lw=0.8))
    ax.text(x + w / 2, y + h - 2.4, title, ha='center', va='top', fontsize=8.5, weight='bold', color=INK)
    ax.text(x + w / 2, y + h - 6.6, body, ha='center', va='top', fontsize=7, color=MUTED, linespacing=1.35)
def arrow(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color='#6d6c66', lw=0.9))
box(1, 24, 17, 16, 'Recording', '4 s, 64 kHz\nvibration +\n2 phase currents', '#f3f2ec')
box(23, 24, 20, 16, 'Envelope analysis', 'band-pass 2–12 kHz\nsquared envelope\nspectrum (Hann)', '#dbe8f8')
box(48, 24, 20, 16, 'Kinematic features', 'peak/background at\n1–3 × BPFO, BPFI,\nBSF, FTF, shaft (15)', '#dbe8f8')
box(73, 24, 26, 16, 'Diagnosis', 'standardise + logistic\nregression (C = 0.3)\nfixed before testing', '#dbe8f8')
box(73, 3, 26, 16, 'Selective decision', 'abstain if confidence <\nthreshold set on training\nbearings only', '#fde3d6')
box(23, 3, 45, 16, 'Sensor-health monitor (learned)', 'quality output per channel, trained with injected\nnoise, bias, gain, drift and clipping;\nreports whether the measurement can be trusted', '#d6f1e7')
arrow(18.5, 32, 22.5, 32); arrow(43.5, 32, 47.5, 32); arrow(68.5, 32, 72.5, 32); arrow(86, 23.5, 86, 19.5)
arrow(9.5, 23.5, 22.5, 12)
ax.text(86, 0.2, 'output: H / OR / IR, or "not sure"', ha='center', va='bottom', fontsize=7, color=INK, style='italic')
save(fig, 'fig1_framework')

# ---------------------------------------------------------------- Fig 2: protocols
bear = ['K001', 'K002', 'K003', 'K004', 'K005', 'K006', 'KA01', 'KA03', 'KA05', 'KA06', 'KA07', 'KA08', 'KA09', 'KI01', 'KI03',
        'KI05', 'KI07', 'KI08', 'KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KI04', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']
L8tr = ['K002', 'KA01', 'KA05', 'KA07', 'KI01', 'KI05', 'KI07']
L8te = ['K001', 'KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']
L10p = ['K001', 'K002', 'K003', 'K004', 'K005', 'KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KI04', 'KI14', 'KI16', 'KI18', 'KI21']
art = bear[6:18]; real = bear[18:]
A2Rtr = ['K001', 'K002', 'K003'] + art; A2Rte = ['K004', 'K005', 'K006'] + real
rows = {'L8': [2 if b in L8te else 1 if b in L8tr else 0 for b in bear],
        'L10': [3 if b in L10p else 0 for b in bear],
        'A2R': [2 if b in A2Rte else 1 for b in bear],
        'LOBO': [3 for b in bear]}
cmap = {0: '#efeee9', 1: BLUE, 2: ORANGE, 3: AQUA}
fig, ax = plt.subplots(figsize=(7.2, 1.9)); ax.grid(False)
for i, (k, v) in enumerate(rows.items()):
    for j, c in enumerate(v):
        ax.add_patch(plt.Rectangle((j + 0.06, 3 - i + 0.08), 0.88, 0.84, color=cmap[c], lw=0))
ax.set_xlim(0, len(bear)); ax.set_ylim(0, 4); ax.set_yticks([3.5, 2.5, 1.5, 0.5]); ax.set_yticklabels(list(rows))
ax.set_xticks(np.arange(len(bear)) + 0.5); ax.set_xticklabels(bear, rotation=90, fontsize=6.5)
for s in ax.spines.values(): s.set_visible(False)
for x, lab in ((3, 'healthy'), (12, 'artificial damage'), (23.5, 'real damage')):
    ax.text(x, 4.15, lab, ha='center', fontsize=7.5, color=MUTED)
for x in (6, 18): ax.axvline(x, color='#8c8b84', lw=0.8)
ax.legend(handles=[Patch(color=BLUE, label='training'), Patch(color=ORANGE, label='test'),
                   Patch(color=AQUA, label='rotating train/test pool'), Patch(color='#efeee9', label='not used')],
          ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.42), fontsize=7.5)
save(fig, 'fig2_protocols')

# ---------------------------------------------------------------- Fig 3: label check scatter (Step 5, fixed band)
pts = [('K001', 0, .56, .75), ('K002', 0, .62, .77), ('K003', 0, .69, .74), ('K004', 0, .83, .76), ('K005', 0, .84, .78), ('K006', 0, 1.13, .69),
       ('KA01', 1, 5.03, .78), ('KA03', 1, 2.22, .74), ('KA05', 1, 1.83, .73), ('KA06', 1, 2.76, .72), ('KA07', 1, 2.63, .70), ('KA08', 1, 1.72, .68), ('KA09', 1, 2.68, .75),
       ('KI01', 2, 2.65, 3.21), ('KI03', 2, 1.01, 1.71), ('KI05', 2, .78, 1.73), ('KI07', 2, .91, 1.55), ('KI08', 2, 1.23, 2.10),
       ('KA04', 1, 3.44, .85), ('KA15', 1, 1.05, .85), ('KA16', 1, 3.39, .77), ('KA22', 1, .93, .65), ('KA30', 1, 1.42, 1.04),
       ('KI04', 2, 1.65, 2.63), ('KI14', 2, 1.05, 1.81), ('KI16', 2, 1.72, 1.54), ('KI17', 2, .87, .93), ('KI18', 2, 2.65, 3.40), ('KI21', 2, 1.50, 1.72),
       ('KB23', 3, 2.27, 1.58), ('KB24', 3, 2.00, 2.32), ('KB27', 3, 1.38, .99)]
col = {0: GREY, 1: BLUE, 2: ORANGE, 3: AQUA}; name = {0: 'healthy', 1: 'outer ring', 2: 'inner ring', 3: 'both rings'}
fig, ax = plt.subplots(figsize=(4.6, 3.8))
for b, c, o, i in pts:
    realb = b[:2] in ('KB',) or b in [x for x in real]
    ax.scatter(o, i, s=34, color=col[c], marker='o' if not realb else 's', edgecolor='white', linewidth=1.2, zorder=3)
offs = {'KA15': (8, -3), 'KA22': (16, -3), 'KA30': (6, 6), 'KI17': (-22, 7), 'K004': (-30, -12), 'K005': (-8, -19),
        'KB27': (8, -12), 'KI16': (5, 4), 'KA01': (-10, 7), 'KI18': (5, 3)}
for b, off in offs.items():
    o, i = [(p[2], p[3]) for p in pts if p[0] == b][0]
    ax.annotate(b, (o, i), xytext=off, textcoords='offset points', fontsize=6.5, color=INK,
                arrowprops=dict(arrowstyle='-', color='#b5b4ad', lw=0.5) if b in ('K004', 'K005', 'KB27', 'KA22') else None)
ax.set_xlabel('median BPFO feature (log peak/background)'); ax.set_ylabel('median BPFI feature (log peak/background)')
h = [Line2D([], [], marker='o', ls='', color=col[k], label=name[k]) for k in range(4)]
h += [Line2D([], [], marker='o', ls='', color=MUTED, label='healthy / artificial'), Line2D([], [], marker='s', ls='', color=MUTED, label='real damage')]
ax.legend(handles=h, fontsize=7, loc='upper right')
save(fig, 'fig3_label_check')

# ---------------------------------------------------------------- Fig 4: main comparison
prot = ['L8\ncond. 0', 'L8\nall cond.', 'L10', 'A2R', 'LOBO']
meth = [('Envelope features + LR (primary)', [93.6, 82.8, 74.6, 73.0, 80.1], [0, 0, 11.8, 0, 0], BLUE),
        ('All envelope features + RF', [84.1, 81.7, 72.6, 76.3, 82.5], [0, 0, 10.9, 0, 0], AQUA),
        ('WDCNN, envelope', [70.8, 59.9, 59.6, 69.0, 68.1], [11.2, 2.1, 15.6, 1.2, 0], ORANGE),
        ('WDCNN, raw vibration', [46.4, 47.5, 58.1, 66.0, 61.9], [6.2, 2.2, 13.6, 1.5, 0], YELLOW)]
fig, ax = plt.subplots(figsize=(7.2, 3.2)); x = np.arange(len(prot)); w = 0.2
for k, (lab, v, e, c) in enumerate(meth):
    ax.bar(x + (k - 1.5) * w, v, w * 0.92, color=c, label=lab, yerr=[ee if ee else np.nan for ee in e],
           error_kw=dict(ecolor='#555', lw=0.7, capsize=1.8))
for k, val in enumerate(meth[0][1]):
    ax.text(x[k] - 1.5 * w, val + 1.2, f'{val:.1f}', ha='center', fontsize=6.5, color=INK)
ax.plot([x[0] - 0.42, x[0] + 0.42], [65.9, 65.9], color=INK, lw=1.1, ls='--', label='published [2], vibration (best)')
ax.plot([x[2] - 0.42, x[2] + 0.42], [98.5, 98.5], color=INK, lw=1.1, ls='--')
ax.axhline(100 / 3, color=MUTED, lw=0.8, ls=':'); ax.text(4.55, 34.5, 'chance', fontsize=6.5, color=MUTED, ha='right')
ax.set_xticks(x); ax.set_xticklabels(prot); ax.set_ylabel('recording accuracy (%)'); ax.set_ylim(0, 105)
ax.legend(fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.17))
save(fig, 'fig4_main_comparison')

# ---------------------------------------------------------------- Fig 5: LOBO per bearing, physics vs CNN
phys = dict(K001=.80, K002=.85, K003=.94, K004=.15, K005=.07, K006=.72, KA01=1.00, KA03=1.00, KA05=1.00, KA06=1.00, KA07=1.00, KA08=.87, KA09=.75,
            KI01=1.00, KI03=.94, KI05=.75, KI07=.70, KI08=.97, KA04=.99, KA15=.61, KA16=1.00, KA22=.62, KA30=.46, KI04=1.00, KI14=.80, KI16=.97,
            KI17=.39, KI18=1.00, KI21=.86)
cnn = dict(K001=1.00, K002=.00, K003=1.00, K004=.40, K005=.04, K006=.99, KA01=1.00, KA03=.74, KA04=.97, KA05=.51, KA06=1.00, KA07=.61, KA08=.99,
           KA09=.05, KA15=.93, KA16=1.00, KA22=.71, KA30=.74, KI01=1.00, KI03=.86, KI04=.72, KI05=.19, KI07=.10, KI08=.47, KI14=.75, KI16=.75,
           KI17=.45, KI18=1.00, KI21=.79)
fig, ax = plt.subplots(figsize=(7.2, 2.9)); xs = np.arange(len(bear))
for j, b in enumerate(bear):
    ax.plot([j, j], [phys[b] * 100, cnn[b] * 100], color='#d6d5cf', lw=1, zorder=1)
ax.scatter(xs, [phys[b] * 100 for b in bear], s=26, color=BLUE, edgecolor='white', lw=1, zorder=3, label='envelope features + LR (primary)')
ax.scatter(xs, [cnn[b] * 100 for b in bear], s=26, color=ORANGE, marker='D', edgecolor='white', lw=1, zorder=3, label='WDCNN, envelope')
for xv in (5.5, 17.5): ax.axvline(xv, color='#8c8b84', lw=0.8)
for xv, lab in ((2.5, 'healthy'), (11.5, 'artificial damage'), (23, 'real damage')):
    ax.text(xv, 108, lab, ha='center', fontsize=7.5, color=MUTED)
ax.set_xticks(xs); ax.set_xticklabels(bear, rotation=90, fontsize=6.5); ax.set_ylim(-5, 114); ax.set_ylabel('accuracy on held-out bearing (%)')
ax.legend(fontsize=7, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.3))
save(fig, 'fig5_lobo_per_bearing')

# ---------------------------------------------------------------- Fig 6: by damage level
grp = [('Artificial,\nlevel 1', 94.8), ('Artificial,\nlevel 2', 88.3), ('Real,\nlevel 1', 71.7), ('Real,\nlevel 2', 100.0), ('Real,\nlevel 3', 97.5), ('Healthy', 59.0)]
fig, ax = plt.subplots(figsize=(4.8, 2.8))
ax.bar(range(len(grp)), [g[1] for g in grp], 0.62, color=BLUE)
for k, g in enumerate(grp): ax.text(k, g[1] + 1.5, f'{g[1]:.1f}', ha='center', fontsize=7, color=INK)
ax.set_xticks(range(len(grp))); ax.set_xticklabels([g[0] for g in grp], fontsize=7.5); ax.set_ylim(0, 108); ax.set_ylabel('LOBO accuracy (%)')
save(fig, 'fig6_damage_level')

# ---------------------------------------------------------------- Fig 7: risk-coverage + training-only thresholds
cov = [100, 90, 80, 70, 60, 50]; acc = [80.1, 84.3, 88.7, 93.7, 97.4, 99.4]
thr_cov = [100.0, 89.9, 78.7, 68.8]; thr_acc = [80.1, 84.5, 89.5, 94.8]
fig, ax = plt.subplots(figsize=(4.6, 3.1))
ax.plot(cov, acc, color=BLUE, lw=2, marker='o', ms=4, label='ranked by confidence (descriptive)')
ax.scatter(thr_cov, thr_acc, s=46, color=ORANGE, marker='D', edgecolor='white', lw=1, zorder=4, label='threshold from training bearings only')
for c_, a_ in zip(thr_cov[1:], thr_acc[1:]): ax.annotate(f'{a_:.1f}', (c_, a_), xytext=(5, -10), textcoords='offset points', fontsize=6.5, color=INK)
ax.invert_xaxis(); ax.set_xlabel('coverage: share of recordings accepted (%)'); ax.set_ylabel('accuracy on accepted (%)'); ax.set_ylim(75, 101)
ax.legend(fontsize=7, loc='lower right')
save(fig, 'fig7_risk_coverage')

# ---------------------------------------------------------------- Fig 8: per-bearing acceptance vs accuracy (Step 9)
st9 = [('K001', .65, .80), ('K002', .39, .85), ('K003', .74, .94), ('K004', .07, .15), ('K005', .36, .07), ('K006', .33, .72), ('KA01', 1.0, 1.0), ('KA03', .99, 1.0),
       ('KA05', .89, 1.0), ('KA06', .99, 1.0), ('KA07', .95, 1.0), ('KA08', .65, .87), ('KA09', .81, .75), ('KI01', 1.0, 1.0), ('KI03', .69, .94), ('KI05', .64, .75),
       ('KI07', .65, .70), ('KI08', .90, .97), ('KA04', 1.0, .99), ('KA15', .20, .61), ('KA16', 1.0, 1.0), ('KA22', .20, .62), ('KA30', .19, .46), ('KI04', 1.0, 1.0),
       ('KI14', .80, .80), ('KI16', .90, .97), ('KI17', .33, .39), ('KI18', 1.0, 1.0), ('KI21', .66, .86)]
fig, ax = plt.subplots(figsize=(4.6, 3.4))
gc = lambda b: GREY if b.startswith('K0') else (BLUE if b.startswith('KA') else ORANGE)
for b, a, c in st9:
    ax.scatter(c * 100, a * 100, s=30, color=gc(b), edgecolor='white', lw=1, zorder=3)
for b, lab in (('K004', 'K004'), ('K005', 'K005'), ('KA22', 'KA15, KA22'), ('KA30', 'KA30'), ('KI17', 'KI17')):
    a, c = [(p[1], p[2]) for p in st9 if p[0] == b][0]; ax.annotate(lab, (c * 100, a * 100), xytext=(5, 3), textcoords='offset points', fontsize=6.5)
ax.set_xlabel('accuracy on all recordings of the bearing (%)'); ax.set_ylabel('recordings accepted (%)')
ax.legend(handles=[Line2D([], [], marker='o', ls='', color=GREY, label='healthy'), Line2D([], [], marker='o', ls='', color=BLUE, label='outer ring'),
                   Line2D([], [], marker='o', ls='', color=ORANGE, label='inner ring')], fontsize=7, loc='upper left')
save(fig, 'fig8_acceptance')

# ---------------------------------------------------------------- Fig 9: ladder on L10
rows9 = [('Primary model\n(fixed in advance)', 73.5, 74.6), ('Nested selection\namong 72 (honest)', 55.3, 69.3), ('Best of 72 after\ntesting (not valid)', 74.8, 76.2)]
fig, ax = plt.subplots(figsize=(4.8, 2.9)); x = np.arange(3); w = 0.34
ax.bar(x - w / 2, [r[1] for r in rows9], w * 0.92, color=BLUE, label='condition 0')
ax.bar(x + w / 2, [r[2] for r in rows9], w * 0.92, color=AQUA, label='all conditions')
for k, r in enumerate(rows9):
    ax.text(k - w / 2, r[1] + 1.5, f'{r[1]:.1f}', ha='center', fontsize=6.5); ax.text(k + w / 2, r[2] + 1.5, f'{r[2]:.1f}', ha='center', fontsize=6.5)
ax.axhline(98.5, color=INK, ls='--', lw=1); ax.text(2.45, 101.0, 'published [2]: 98.5', ha='right', fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows9], fontsize=7); ax.set_ylim(0, 108); ax.set_ylabel('L10 recording accuracy (%)')
ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(0, 0.93))
save(fig, 'fig9_ladder')

# ---------------------------------------------------------------- Fig 10: sensor degradation (two panels, one axis each)
fam = ['clean', 'noise', 'bias/gain', 'drift', 'clipping', 'spikes,\nquantis.\n(test only)']
f1 = [85.0, 83.3, 85.0, 84.6, 84.7, 84.7]; auc = [np.nan, 96.0, 96.5, 96.2, 97.1, 67.3]; auc_sd = [0, 2.9, 2.4, 5.9, 1.6, 3.7]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
a1.bar(range(6), f1, 0.6, color=BLUE)
for k, v in enumerate(f1): a1.text(k, v + 1.5, f'{v:.1f}', ha='center', fontsize=6.5)
a1.set_xticks(range(6)); a1.set_xticklabels(fam, fontsize=6.5); a1.set_ylim(0, 100); a1.set_ylabel('diagnosis macro F1 (%)')
a1.set_title('(a) physics diagnosis under sensor errors', fontsize=8, loc='left', color=INK)
a2.bar(range(1, 6), auc[1:], 0.6, color=AQUA, yerr=auc_sd[1:], error_kw=dict(ecolor='#555', lw=0.7, capsize=2))
for k in range(1, 6): a2.text(k, auc[k] + auc_sd[k] + 1.5, f'{auc[k]:.1f}', ha='center', fontsize=6.5)
a2.axhline(50, color=MUTED, ls=':', lw=0.8); a2.text(0.45, 51.5, 'chance', fontsize=6.5, color=MUTED, ha='left')
a2.set_xticks(range(1, 6)); a2.set_xticklabels(fam[1:], fontsize=6.5); a2.set_xlim(0.4, 5.6); a2.set_ylim(0, 108); a2.set_ylabel('monitor AUROC (%)')
a2.set_title('(b) sensor-health monitor', fontsize=8, loc='left', color=INK)
fig.tight_layout(); save(fig, 'fig10_sensor_health')
print('figures in', OUT, sorted(os.listdir(OUT)))
