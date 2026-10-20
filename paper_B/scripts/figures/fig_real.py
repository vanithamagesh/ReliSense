"""Figures from phase13_figdata.npz (real Paderborn data, phase 13) and phase6_numbers.json (v2.0, manuscript v3.26).
fig_calib and fig_abstain use the final ReliSense (two branches; LOBO posteriors and training-only thresholds of phase 22)."""
import json
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from ieee_style import plt, C, COL1, COL2, panel, save

Z = dict(np.load('phase13_figdata.npz', allow_pickle=True)); P6 = json.load(open('phase6_numbers.json'))
grid = Z['grid'].astype(float); FN = list(Z['feature_names'])
n, d, D = 8, 6.75, 28.55; q = d / D
FO = {'BPFO': n / 2 * (1 - q), 'BPFI': n / 2 * (1 + q), 'BSF': D / (2 * d) * (1 - q ** 2), 'FTF': 0.5 * (1 - q), 'shaft': 1.0}
prob = Z['lobo_prob']; Bb = Z['lobo_bearing']; y = Z['lobo_label']; keep = ~np.isnan(prob).any(1)
conf = np.where(keep, np.nan_to_num(prob).max(1), np.nan); pred = np.nan_to_num(prob).argmax(1); ok = (pred == y) & keep
tau = {b: t for b, t in zip(Z['tau_bearings'], Z['tau'])}; KAP = list(Z['tau_kappa'])
CLS = {0: ('H', C['H']), 1: ('OR', C['OR']), 2: ('IR', C['IR'])}

# ------------------------------------------------------------------ framework: Modules I-III with real data
fig = plt.figure(figsize=(COL2, 2.55))
gs = fig.add_gridspec(2, 4, width_ratios=[1.35, 0.75, 1.0, 1.0], height_ratios=[1, 1], hspace=0.55, wspace=0.42)
ax1 = fig.add_subplot(gs[0, 0]); ax1b = fig.add_subplot(gs[1, 0])
for ax, b, col in ((ax1, 'KA16', C['OR']), (ax1b, 'K001', C['H'])):
    ax.plot(grid, Z[f'order_{b}'], lw=0.45, color=col)
    for h in (1, 2, 3): ax.axvline(h * FO['BPFO'], color=C['ink'], lw=0.4, ls=(0, (3, 2)))
    ax.set_xlim(0.2, 10.5); ax.set_ylim(-1.5, 6.5); ax.set_yticks([0, 3, 6]); ax.set_ylabel('log SES', labelpad=1)
    ax.text(0.98, 0.9, f'{b} ({"outer race" if b == "KA16" else "healthy"})', transform=ax.transAxes, ha='right', va='top', fontsize=6.5)
ax1.set_xticklabels([]); ax1b.set_xlabel(r'Order $r = f/f_r$'); ax1.set_title('(a) Envelope order spectrum', loc='left')
axf = fig.add_subplot(gs[:, 1])
M = np.stack([Z['feat_KA16'], Z['feat_KI18'], Z['feat_K001']]).reshape(3, 5, 3)
show = np.concatenate([M[i] for i in range(3)], 1)   # 5 orders x 9 (3 bearings x 3 harmonics)
im = axf.imshow(show, cmap='Blues', vmin=0, vmax=max(4, float(show.max())), aspect='auto')
axf.set_yticks(range(5)); axf.set_yticklabels(['BPFO', 'BPFI', 'BSF', 'FTF', 'shaft'], fontsize=6.5)
axf.set_xticks([1, 4, 7]); axf.set_xticklabels(['KA16', 'KI18', 'K001'], fontsize=6.5); axf.tick_params(length=0)
for xv in (2.5, 5.5): axf.axvline(xv, color='white', lw=1.2)
axf.set_title(r'(b) Features $\varphi_{kh}$', loc='left')
ax3 = fig.add_subplot(gs[:, 2]); ex = {}
for b in ('KA16', 'KI18', 'K001'):
    m = (Bb == b) & keep; ex[b] = prob[m].mean(0)
xs = np.arange(3); w = 0.26
for j, (k, (nm, col)) in enumerate(CLS.items()):
    ax3.bar(xs + (j - 1) * w, [ex[b][k] for b in ('KA16', 'KI18', 'K001')], w * 0.92, color=col, label=nm)
ax3.axhline(np.mean(Z['tau'][:, 2]), color=C['ink'], lw=0.6, ls=(0, (4, 2)), label=r'mean $\tau_b$, $\kappa$ = 0.7')
ax3.set_xticks(xs); ax3.set_xticklabels(['KA16', 'KI18', 'K001'], fontsize=6.5); ax3.set_ylim(0, 1.3); ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax3.set_ylabel(r'Mean $p(k\,|\,\mathbf{z})$, LOBO')
ax3.legend(fontsize=6, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.01), handlelength=0.8, columnspacing=0.6); ax3.set_title('(c) Diagnosis, unseen bearing', loc='left')
ax4 = fig.add_subplot(gs[:, 3]); sc = P6['E1_scaled']; ss = sorted(float(s) for s in sc)
ax4.plot(ss, [100 * sc[f'{s:g}' if f'{s:g}' in sc else str(s)]['L8_cond0'] for s in ss], '-o', color=C['light'], mfc='white', mec=C['ink'], ms=2.6, lw=0.7, mew=0.5)
ax4.plot([1.0], [100 * sc['1.0']['L8_cond0']], 'o', ms=4.5, color=C['rs'])
ax4.set_xlabel('Scale of fault orders'); ax4.set_ylabel('L8 accuracy (%)'); ax4.set_title('(d) Verification', loc='left')
fig.canvas.draw(); bb = {k: a.get_position() for k, a in (('a', ax1), ('b', axf), ('c', ax3), ('d', ax4))}
top = bb['a'].y1 + 0.13
for (l, r), t in ((('a', 'b'), 'Module I: kinematic signature'), (('c', 'c'), 'Module II: diagnosis and abstention'), (('d', 'd'), 'Module III: verification')):
    x0, x1 = bb[l].x0, bb[r].x1
    fig.add_artist(Line2D([x0, x1], [top, top], transform=fig.transFigure, color=C['ink'], lw=0.5))
    fig.text((x0 + x1) / 2, top + 0.015, t, ha='center', va='bottom', fontsize=7.5, style='italic')
save(fig, 'fig_framework.png')

# ------------------------------------------------------------------ calibration of tau_b: nested split (real bearings) + pooled inner confidences
from v326 import FD
single = [str(b) for b in FD['tau_bearings']]; b0 = 'KA22'; ib = single.index(b0); pool = FD['inner_conf_KA22'].astype(float); TAU22 = FD['tau'].astype(float)
fig = plt.figure(figsize=(COL2, 2.6)); gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.28)
ax = fig.add_subplot(gs[0]); rows = ['outer split'] + [f'inner {i}' for i in (1, 2, 3)] + ['...'] + ['inner 28']
inner_v = [x for x in single if x != b0]; vs = [inner_v[0], inner_v[1], inner_v[2], None, inner_v[-1]]
for i, r in enumerate(rows):
    for j, b in enumerate(single):
        if r == '...': continue
        if b == b0: fc = C['IR'] if i == 0 else 'white'
        elif i == 0: fc = C['rs']
        else: fc = '#e8a33d' if b == vs[i - 1] else '#9db9d6'
        ax.add_patch(Rectangle((j + 0.07, i + 0.1), 0.86, 0.8, fc=fc, ec='#9a9a9a' if fc == 'white' else 'none', lw=0.3))
    ax.text(-0.4, i + 0.5, r, ha='right', va='center', fontsize=6.5)
for j, b in enumerate(single): ax.text(j + 0.5, 6.15, b, rotation=90, ha='center', va='top', fontsize=7)
ax.set_xlim(-5.2, len(single) + 0.2); ax.set_ylim(8.3, -0.4); ax.axis('off'); ax.text(-4.5, -0.75, '(a)', fontsize=8)
hh = [Rectangle((0, 0), 1, 1, fc=c) for c in (C['IR'], C['rs'], '#e8a33d', '#9db9d6')]
ax.legend(hh, [f'test bearing b ({b0})', r'training bearings $\mathcal{T}_b$', r'inner held-out bearing $v$', r'inner training $\mathcal{T}_b \setminus \{v\}$'], loc='lower center', bbox_to_anchor=(0.45, -0.34), ncol=2, fontsize=7)
ax2 = fig.add_subplot(gs[1]); ax2.hist(pool, bins=np.linspace(1 / 3, 1, 34), color='#9db9d6', edgecolor='white', lw=0.3)
for k, t, ls in zip(KAP, TAU22[ib], ('-', (0, (4, 2)), (0, (1, 1.5)))):
    ax2.axvline(t, color=C['ink'], lw=0.7, ls=ls, label=rf'$\kappa$ = {k:.1f}: $\tau_b$ = {t:.2f}')
ax2.set_xlabel(rf'Confidence of inner models on $v$ ($b$ = {b0})'); ax2.set_ylabel('Recordings'); ax2.set_ylim(0, 1.3 * np.histogram(pool, bins=np.linspace(1 / 3, 1, 34))[0].max()); ax2.legend(fontsize=6.3, loc='upper left', bbox_to_anchor=(0.0, 1.0), facecolor='white', framealpha=1, frameon=True, edgecolor='none').set_zorder(5); panel(ax2, '(b)', -0.02, 1.0)
save(fig, 'fig_calib.png')

# ------------------------------------------------------------------ signature map
Mb = Z['sig_map'].astype(float); names = list(Z['sig_bearings']); lab = list(Z['sig_labels'])
fig, ax = plt.subplots(figsize=(COL2, 3.9))
im = ax.imshow(np.clip(Mb, -0.5, 3.0), aspect='auto', cmap='magma', extent=[grid[0], grid[-1], len(names) - 0.5, -0.5], interpolation='nearest')
for k, ls, col in (('BPFO', (0, (4, 2)), '#9fd3ff'), ('BPFI', (0, (1, 1.5)), '#ffffff')):
    for h in (1, 2, 3):
        if h * FO[k] <= grid[-1]: ax.axvline(h * FO[k], color=col, lw=0.55, ls=ls)
TAG = {0: 'H', 1: 'OR', 2: 'IR', 3: 'OR+IR'}
ax.set_yticks(range(len(names))); ax.set_yticklabels([f'{b} {TAG[l]}' for b, l in zip(names, lab)], fontsize=5.6)
bounds = [i - 0.5 for i in range(1, len(names)) if (names[i][:2] != names[i - 1][:2]) or (lab[i] != lab[i - 1] and names[i][:2] == 'KB')]
for yy in (5.5, 17.5, 28.5): ax.axhline(yy, color='white', lw=0.8)
ax.set_xlabel(r'Order $r = f/f_r$'); cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.035); cb.set_label('log SES / median'); cb.outline.set_linewidth(0.4)
for yy, t in ((2.5, 'healthy'), (11.5, 'artificial'), (23, 'real'), (30, 'both rings')): ax.text(12.12, yy, t, rotation=90, va='center', ha='left', fontsize=6.3, clip_on=False)
ax.legend(handles=[Line2D([], [], color='#4a8fd0', ls=(0, (4, 2)), lw=0.8, label='BPFO harmonics'), Line2D([], [], color='#555555', ls=(0, (1, 1.5)), lw=0.8, label='BPFI harmonics')],
          loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=7)
save(fig, 'fig_sigmap.png')

# ------------------------------------------------------------------ folding fields
fr = Z['fold_rates']; cases = [('K001', 'BPFO', 'Healthy K001, BPFO', 0), ('KA16', 'BPFO', 'Outer race KA16, BPFO', 1), ('KI18', 'BPFI', 'Inner race KI18, BPFI', 2)]
fig, axs = plt.subplots(2, 4, figsize=(COL2, 3.5), gridspec_kw={'hspace': 0.28, 'wspace': 0.14, 'width_ratios': [1, 1, 1, 1.15]})
for j, (b, k, t, r) in enumerate(cases):
    A = Z[f'fold_{b}'].astype(float); A10 = Z[f'fold10_{b}'].astype(float); vmax = np.percentile(np.concatenate([A.ravel(), A10.ravel()]), 99)
    for i, (F_, lab_) in enumerate(((A, rf'measured period, $f^*$ = {fr[r, 1]:.2f} Hz'), (A10, 'period offset by 10%'))):
        ax = axs[i, j]; ax.imshow(np.clip(F_, 0, vmax), aspect='auto', cmap='Blues', extent=[0, 1, F_.shape[0], 0], interpolation='nearest')
        ax.set_title((t + '\n' if i == 0 else '') + lab_, fontsize=6.8, loc='left')
        if j == 0: ax.set_ylabel(r'Fault period $i$')
        else: ax.set_yticklabels([])
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(['0', '0.5', '1'] if i == 1 else [])
        if i == 1: ax.set_xlabel(r'Phase $\theta$ within one period')
        else: ax.set_xticklabels([])
for i, (b_, t_) in enumerate((('KI18', 'Inner race KI18'), ('K001', 'Healthy K001'))):
    ax = axs[i, 3]; Fs = Z[f'shaft_{b_}'].astype(float); ax.imshow(np.clip(Fs, 0, np.percentile(Fs, 99)), aspect='auto', cmap='Blues', extent=[0, 360, Fs.shape[0], 0], interpolation='nearest')
    ax.set_title(t_ + ('\nfolded at shaft period' if i == 0 else ''), fontsize=6.8, loc='left'); ax.set_ylabel('Revolution', labelpad=1); ax.yaxis.set_label_position('right'); ax.yaxis.tick_right()
    ax.set_xticks([0, 180, 360])
    if i == 1: ax.set_xlabel('Shaft angle (deg)')
    else: ax.set_xticklabels([])
save(fig, 'fig_folding.png')

# ------------------------------------------------------------------ selective classification (final ReliSense, phase 22)
prob = FD['prob'].astype(float); Bb = np.array([str(b) for b in FD['bearing']]); y = FD['label'].astype(int); keep = np.ones(len(y), bool)
conf = prob.max(1); pred = prob.argmax(1); ok = pred == y; tau = {b: t for b, t in zip(single, TAU22)}
fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.45), gridspec_kw={'wspace': 0.42})
bins = np.linspace(1 / 3, 1, 27)
axs[0].hist(conf[ok], bins=bins, color='#8fb3d9', label=f'correct (n = {ok.sum()})')
axs[0].hist(conf[keep & ~ok], bins=bins, histtype='step', color=C['IR'], lw=1.0, label=f'incorrect (n = {(keep & ~ok).sum()})')
axs[0].set_yscale('log'); axs[0].set_xlabel(r'Confidence $c$'); axs[0].set_ylabel('Recordings'); axs[0].legend(fontsize=6.2, loc='upper left'); axs[0].set_title('(a)', loc='left')
o = np.argsort(-conf[keep]); okk = ok[keep][o]; cov = np.arange(1, len(o) + 1) / len(o); accc = np.cumsum(okk) / np.arange(1, len(o) + 1)
axs[1].plot(100 * cov, 100 * accc, color=C['light'], lw=0.9, label='ranked by confidence')
pts = []
for jk, k in enumerate(KAP):
    acc_m = np.array([keep[i] and conf[i] >= tau[Bb[i]][jk] for i in range(len(y))]); pts.append((100 * acc_m[keep].mean(), 100 * ok[acc_m].mean()))
pts = [(100.0, 100 * ok[keep].mean())] + pts
axs[1].plot(*zip(*pts), 'o', ms=3.4, color=C['rs'], label=r'training-only $\tau_b$')
for (cx, cy), lab_ in zip(pts, ['all', 'κ = 0.9', 'κ = 0.8', 'κ = 0.7']): axs[1].annotate(f'{cy:.1f}', (cx, cy), xytext=(3, -9), textcoords='offset points', fontsize=7)
axs[1].set_xlim(102, 45); axs[1].set_ylim(82, 101); axs[1].set_xlabel('Coverage (%)'); axs[1].set_ylabel('Selective accuracy (%)'); axs[1].legend(fontsize=6.2, loc='lower right'); axs[1].set_title('(b)', loc='left')
acc70 = np.array([keep[i] and conf[i] >= tau[Bb[i]][2] for i in range(len(y))])
for b in single:
    m = Bb == b; cl = int(y[m][0]); xa, ya = 100 * ok[m].mean(), 100 * acc70[m].mean()
    axs[2].plot(xa, ya, 'o', ms=3.2, color=CLS[cl][1])
    OFF = {'K004': (-4, -2, 'right'), 'K005': (5, -4, 'left'), 'KA30': (-4, 3, 'right'), 'K002': (5, -2, 'left'), 'KI05': (5, -3, 'left')}
    if b in OFF: axs[2].annotate(b, (xa, ya), xytext=OFF[b][:2], textcoords='offset points', fontsize=7, ha=OFF[b][2], va='center')
axs[2].set_xlabel('Accuracy of the bearing (%)'); axs[2].set_ylabel(r'Accepted at $\kappa$ = 0.7 (%)'); axs[2].set_xlim(-3, 105); axs[2].set_ylim(-3, 105); axs[2].set_title('(c)', loc='left')
axs[2].legend(handles=[Line2D([], [], ls='', marker='o', color=CLS[k][1], ms=3.2, label=CLS[k][0]) for k in CLS], fontsize=6.2, loc='upper left', handletextpad=0.1)
save(fig, 'fig_abstain.png')
print('check', pts)
