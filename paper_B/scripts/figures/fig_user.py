"""Figures from the phase 5-7 and phase 22 results (v2.0, manuscript v3.26). All panels are drawn from data in one style.
fig_weights     weights of the kinematic core (phase 6, E4)
fig_conditions  (a), (c) final ReliSense by operating condition (phase 22); (b) speed transfer of four inputs (phase 6, E2)
fig_efficiency  data efficiency of three inputs (phase 6, E5), mean +- SD as error bars
fig_tf          (a) spectrograms of 0.5 s of the 8-kHz vibration, (b) envelope order spectra of the 20 condition-0 recordings
                of four bearings as heat maps (phase 22 views; replaces the delivered phase 5 plots)
fig_featdist    (a) first-harmonic BPFO and BPFI features of every bearing, (b) PCA of the 15 features (phase 22 views)"""
import json
import numpy as np
from matplotlib.lines import Line2D
from ieee_style import plt, C, COL1, COL2, save
from v326 import P22, VW

P6 = json.load(open('phase6_numbers.json'))
FAM = ['BPFO', 'BPFI', 'BSF', 'FTF', 'shaft']; KEYS = [f'{k}_h{h}' for k in FAM for h in (1, 2, 3)]
n_, d_, D_ = 8, 6.75, 28.55; q = d_ / D_; BPFO, BPFI = n_ / 2 * (1 - q), n_ / 2 * (1 + q)
CLSC = {0: C['H'], 1: C['OR'], 2: C['IR'], 3: C['B']}

# ------------------------------------------------------------------ learned weights of the kinematic core (E4)
W = np.array([[P6['E4_weights'][c][k] for k in KEYS] for c in ('H', 'OR', 'IR')])
fig, ax = plt.subplots(figsize=(COL2, 1.75)); v = np.abs(W).max()
im = ax.imshow(W, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
for (r, c), x in np.ndenumerate(W): ax.text(c, r, f'{x:.2f}', ha='center', va='center', fontsize=7, color='white' if abs(x) > 0.6 * v else C['ink'])
ax.set_yticks(range(3)); ax.set_yticklabels(['H', 'OR', 'IR']); ax.set_xticks(range(15)); ax.set_xticklabels([f'{k} {h}' for k in FAM for h in (1, 2, 3)], rotation=45, ha='right')
for x in (2.5, 5.5, 8.5, 11.5): ax.axvline(x, color='white', lw=1.6)
ax.tick_params(length=0); cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label('Weight'); cb.outline.set_linewidth(0.4)
save(fig, 'fig_weights.png')

# ------------------------------------------------------------------ operating conditions
CN = ['N15_M07_F10', 'N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04']; CL = ['1500 rpm', '900 rpm', '0.1 Nm', '400 N']
fig = plt.figure(figsize=(COL2, 4.3)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.75, wspace=0.3)
ax = fig.add_subplot(gs[0, 0]); xs = np.arange(4); w = 0.2; BC = P22['lobo_by_condition']
series = [('all classes', [100 * BC[c]['acc'] for c in CN], C['rs']), ('recall H', [100 * BC[c]['recall_H'] for c in CN], C['H']),
          ('recall OR', [100 * BC[c]['recall_OR'] for c in CN], C['OR']), ('recall IR', [100 * BC[c]['recall_IR'] for c in CN], C['IR'])]
for k, (lab, vals, col) in enumerate(series): ax.bar(xs + (k - 1.5) * w, vals, w * 0.9, color=col, label=lab)
for x, a in zip(xs, series[0][1]): ax.text(x - 1.5 * w, a + 2, f'{a:.1f}', ha='center', fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels(CL); ax.set_ylim(0, 132); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel('LOBO accuracy, recall (%)'); ax.tick_params(axis='x', length=0)
ax.legend(ncol=4, loc='upper center', handlelength=0.9, columnspacing=0.6, handletextpad=0.3, bbox_to_anchor=(0.5, 1.0), fontsize=7)
ax.set_title('(a) ReliSense by operating condition', loc='left')
ax = fig.add_subplot(gs[0, 1]); E2 = P6['E2_speed_transfer']; names = list(E2); lab2 = ['kinematic\n(orders)', 'band energies\n(Hz)', 'time\nstatistics', 'time + band\n(RF)']
for k, (key, col, hatch, lab) in enumerate((('train1500_test1500', C['rs'], None, 'tested at 1500 rpm'), ('train1500_test900', 'white', '////', 'tested at 900 rpm'))):
    vals = [100 * E2[n][key] for n in names]
    ax.bar(np.arange(4) + (k - 0.5) * 0.38, vals, 0.35, color=col, edgecolor=C['rs'], hatch=hatch, lw=0.6, label=lab)
    for x, y_ in zip(np.arange(4) + (k - 0.5) * 0.38, vals): ax.text(x, y_ + 2, f'{y_:.0f}', ha='center', fontsize=7)
ax.axhline(100 / 3, color='#999999', lw=0.5, ls=(0, (2, 1.5))); ax.set_xticks(range(4)); ax.set_xticklabels(lab2); ax.tick_params(axis='x', length=0)
ax.set_ylim(0, 132); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel('LOBO accuracy (%)'); ax.legend(loc='upper right', ncol=2, columnspacing=0.8, handlelength=1.2)
ax.set_title('(b) Kinematic core and generic inputs, trained at 1500 rpm', loc='left')
ax = fig.add_subplot(gs[1, :]); BB = P22['lobo_bearing_by_condition']; bs = list(BB)
M = np.array([BB[b] for b in bs]).T * 100
im = ax.imshow(M, cmap='Blues', vmin=0, vmax=100, aspect='auto')
ax.set_yticks(range(4)); ax.set_yticklabels(CL); ax.set_xticks(range(len(bs))); ax.set_xticklabels(bs, rotation=90); ax.tick_params(length=0)
for x in (5.5, 17.5): ax.axvline(x, color='white', lw=2)
for x, t in ((2.5, 'healthy'), (11.5, 'artificial damage'), (23.5, 'real damage')): ax.text(x, -0.62, t, ha='center', va='bottom')
cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label('Accuracy (%)'); cb.outline.set_linewidth(0.4)
ax.set_title('(c) ReliSense, LOBO accuracy of every bearing by condition', loc='left', pad=17)
save(fig, 'fig_conditions.png')

# ------------------------------------------------------------------ data efficiency (E5): error bars instead of overlapping bands
E5 = P6['E5_data_efficiency']; ks = np.array([1, 2, 3, 4, 5])
fig, ax = plt.subplots(figsize=(COL1, 2.45))
for j, (name, lab, col, mk) in enumerate((('kinematic peaks + LR', 'kinematic core + LR', C['rs'], 'o'), ('band energies + LR', 'band energies + LR', C['rf'], 's'),
                                          ('time + band + RF', 'time + band + RF', C['be'], 'D'))):
    mu = np.array([E5[name][str(k)][0] for k in ks]) * 100; sd = np.array([E5[name][str(k)][1] for k in ks]) * 100
    ax.errorbar(ks + (j - 1) * 0.09, mu, yerr=sd, fmt='-' + mk, color=col, ms=3.6, lw=0.9, elinewidth=0.6, capsize=1.8, capthick=0.6, mec='white', mew=0.4, label=lab)
ax.axhline(100 / 3, color='#999999', lw=0.5, ls=(0, (2, 1.5))); ax.text(5.35, 35, 'chance', fontsize=7, color='#666666', ha='right', va='bottom')
ax.set_xticks(ks); ax.set_xlim(0.7, 5.4); ax.set_xlabel('Training bearings per class'); ax.set_ylabel('Accuracy on unseen bearings (%)'); ax.set_ylim(10, 100)
ax.grid(axis='y', color=C['grid'], lw=0.4); ax.set_axisbelow(True)
ax.legend(loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=7, handlelength=1.6, columnspacing=0.8)
save(fig, 'fig_efficiency.png')

# ------------------------------------------------------------------ raw views: spectrograms and envelope order spectra (phase 22)
def unq(key):
    lo, hi = VW[key + '_lohi'].astype(float); return lo + VW[key].astype(float) / 255 * (hi - lo)
f_, t_ = VW['sg_f'].astype(float), VW['sg_t'].astype(float)
fig = plt.figure(figsize=(COL2, 4.25)); gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=0.62)
g1 = gs[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.12); g2 = gs[1].subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.12)
S = {b: unq('sg_' + b) for b in ('K001', 'KA16', 'KI18')}; vmax = max(np.percentile(s, 99.5) for s in S.values()); vmin = vmax - 60
for j, (b, t) in enumerate((('K001', 'Healthy K001'), ('KA16', 'Outer race KA16'), ('KI18', 'Inner race KI18'))):
    ax = fig.add_subplot(g1[j]); im = ax.pcolormesh(1000 * t_, f_ / 1000, S[b], cmap='magma', vmin=vmin, vmax=vmax, shading='auto', rasterized=True)
    ax.set_title(t, loc='left'); ax.set_xlabel('Time (ms)')
    if j == 0: ax.set_ylabel('Frequency (kHz)')
    else: ax.set_yticklabels([])
    if j == 0: ax.text(-0.22, 1.13, '(a)', transform=ax.transAxes, fontsize=8)
cb = fig.colorbar(im, cax=fig.add_subplot(g1[3])); cb.set_label('Power (dB)'); cb.outline.set_linewidth(0.4)
o = VW['wf_orders'].astype(float); WF = {b: unq('wf_' + b) for b in ('K001', 'KA16', 'KA30', 'KI18')}; wmax = max(np.percentile(x, 99.5) for x in WF.values())
for j, (b, t) in enumerate((('K001', 'K001, healthy'), ('KA16', 'KA16, OR, real, level 2'), ('KA30', 'KA30, OR, real, level 1'), ('KI18', 'KI18, IR, real, level 2'))):
    ax = fig.add_subplot(g2[j]); A = WF[b]
    im = ax.imshow(np.clip(A, 0, None), aspect='auto', cmap='Blues', vmin=0, vmax=2.6, extent=[o[0], o[-1], A.shape[0] + 0.5, 0.5], interpolation='nearest')
    for h in (1, 2, 3):
        ax.plot([h * BPFO] * 2, [-0.1, -1.6], color=C['OR'], lw=1.1, clip_on=False)
        if h * BPFI <= o[-1]: ax.plot([h * BPFI] * 2, [-0.1, -1.6], color=C['IR'], lw=1.1, clip_on=False)
    ax.set_ylim(A.shape[0] + 0.5, 0.5); ax.set_title(t, loc='left', pad=11); ax.set_xlabel(r'Order $r$'); ax.set_xticks([0, 3, 6, 9])
    if j == 0: ax.set_ylabel('Recording'); ax.text(-0.3, 1.2, '(b)', transform=ax.transAxes, fontsize=8)
    else: ax.set_yticklabels([])
cb = fig.colorbar(im, cax=fig.add_subplot(g2[4]), extend='max'); cb.set_label('log SES / median'); cb.outline.set_linewidth(0.4)
fig.legend(handles=[Line2D([], [], color=C['OR'], lw=1.2, label='BPFO harmonics'), Line2D([], [], color=C['IR'], lw=1.2, label='BPFI harmonics')],
           loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.04), frameon=False)
save(fig, 'fig_tf.png')
print('spectrograms from', [str(VW['sg_' + b + '_rec']) for b in ('K001', 'KA16', 'KI18')])

# ------------------------------------------------------------------ feature distributions (phase 22)
bb = [str(b) for b in VW['box_bearings']]; lab = VW['box_label'].astype(int)
fig = plt.figure(figsize=(COL2, 3.3)); gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1.0], wspace=0.22, hspace=0.12)
for i, (key, nm) in enumerate((('box_bpfo1', r'$\varphi_{\mathrm{BPFO},1}$'), ('box_bpfi1', r'$\varphi_{\mathrm{BPFI},1}$'))):
    ax = fig.add_subplot(gs[i, 0]); st = VW[key].astype(float)
    stats = [dict(whislo=s[0], q1=s[1], med=s[2], q3=s[3], whishi=s[4], fliers=[]) for s in st]
    bp = ax.bxp(stats, positions=range(len(bb)), widths=0.62, patch_artist=True, showfliers=False,
                medianprops=dict(color='white', lw=0.9), whiskerprops=dict(lw=0.5, color='#555555'), capprops=dict(lw=0.5, color='#555555'), boxprops=dict(lw=0))
    for p, l in zip(bp['boxes'], lab): p.set_facecolor(CLSC[l])
    ax.set_ylabel(nm); ax.set_xlim(-0.8, len(bb) - 0.2); ax.grid(axis='y', color=C['grid'], lw=0.4); ax.set_axisbelow(True)
    for x in (5.5, 17.5, 28.5): ax.axvline(x, color='#bbbbbb', lw=0.5)
    ax.set_xticks(range(len(bb)))
    if i == 0:
        ax.set_xticklabels([]); ax.text(-0.11, 1.04, '(a)', transform=ax.transAxes, fontsize=8)
        for x, t in ((2.5, 'healthy'), (11.5, 'artificial'), (23, 'real'), (30, 'both')): ax.text(x, 1.02, t, transform=ax.get_xaxis_transform(), ha='center', va='bottom')
    else: ax.set_xticklabels(bb, rotation=90, fontsize=7)
ax = fig.add_subplot(gs[:, 1]); P = VW['pca'].astype(float); yl = VW['pca_label'].astype(int); org = VW['pca_origin']; evr = VW['pca_evr']
rng = np.random.default_rng(0); idx = rng.permutation(len(yl))
for k in (0, 1, 2, 3):
    for real in (False, True):
        m = (yl[idx] == k) & ((org[idx] == 'real') == real)
        if m.any(): ax.scatter(P[idx][m, 0], P[idx][m, 1], s=3 if real else 2.2, marker='s' if real else 'o', lw=0, alpha=0.55, color=CLSC[k], rasterized=True)
ax.set_xlabel(f'PC 1 ({100 * evr[0]:.1f}%)'); ax.set_ylabel(f'PC 2 ({100 * evr[1]:.1f}%)', labelpad=1); ax.text(-0.17, 1.02, '(b)', transform=ax.transAxes, fontsize=8)
h = [Line2D([], [], ls='', marker='o', ms=4, color=CLSC[k], label=t) for k, t in ((0, 'healthy'), (1, 'outer ring'), (2, 'inner ring'), (3, 'both rings'))]
h += [Line2D([], [], ls='', marker='s', ms=4, color='#777777', label='real damage')]
ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.45, -0.16), ncol=3, handletextpad=0.1, columnspacing=0.7, labelspacing=0.3)
save(fig, 'fig_featdist.png')
print('PCA explained', [round(100 * float(x), 1) for x in evr])
