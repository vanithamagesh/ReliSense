"""Phase 5 (v1.1): data figures and supporting numbers for the manuscript, from the real 32-bearing files.

Writes PNG figures (300 dpi) and phase5_numbers.json to --out, then a zip of everything.
  D1 signals          raw vibration and envelope windows of a healthy, an outer-race and an inner-race bearing
  D2 order_mean       mean envelope order spectrum per group (healthy / artificial OR, IR / real OR, IR)
  D3 order_examples   order spectra of clear and weak bearings (KA16 vs KA30, KI18 vs KI17, K001 vs K005)
  D4 feature_boxes    BPFO and BPFI features of every bearing (box plots over its recordings)
  D5 feature_pca      2-D principal components of the 15 envelope features (descriptive)
  D6 conditions       LOBO accuracy of the primary model per operating condition
  D7 confusions       confusion matrices of the primary model (L8 condition 0, A2R, LOBO)
  D8 confidence       LOBO confidence of correct and wrong recordings
  D9 kurtogram        bands selected by the kurtogram, per bearing group
  D10 speed           measured shaft speed per operating condition
  D11 distribution    histogram with fitted normal law and Q-Q plot (healthy BPFO feature; measured speed)
  D12 waterfall       3-D surfaces of the envelope order spectrum over all recordings of four bearings
  D13 spectrogram     time-frequency images of raw vibration (healthy vs outer- and inner-race damage)
  D14 heatmap         LOBO accuracy of every bearing at every operating condition
The primary model (15 fixed-band features, standardisation + logistic regression C = 0.3) is the one fixed
in advance; nothing here is tuned on test bearings.
"""
from __future__ import annotations
import argparse, csv, json, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import physics64 as P
from phase2_physics import L8_TRAIN, L8_TEST

BLUE, ORANGE, AQUA, YELLOW, GREY, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#9a9a93', '#222222', '#5f5e58'
VIOLET = '#4a3aa7'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': '#b5b4ad', 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5df', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
COND0 = 'N15_M07_F10'
CONDS = ['N15_M07_F10', 'N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04']
C15 = 'N15_M07_F10'
CLS = ['H', 'OR', 'IR']


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--physics', default='pb32_physics.npz')
    ap.add_argument('--windows', default='pb32_4096.npz')
    ap.add_argument('--envelope', default='pb32_4096_envelope.npz')
    ap.add_argument('--labels', default='labels_32.csv')
    ap.add_argument('--out', default='phase5_figs')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); nums = {}
    d = dict(np.load(a.physics)); B, y, cond, org = d['bearing'], d['label'], d['condition'], d['origin']
    meta = {r['bearing']: r for r in csv.DictReader(open(a.labels))}
    grid = d['order_grid']; O = d['order_fixed'].astype(float); F = d['feat_fixed']; names = list(d['feature_names'])
    fo = P.fault_orders()
    save = lambda fig, n: (fig.savefig(out / f'{n}.png'), plt.close(fig))

    # ---------------- D1 signals (first window of the first recording at condition 0)
    w = np.load(a.windows); e = np.load(a.envelope)
    WB, WR, WC = w['bearing'], w['recording'], w['condition']
    pick = [('K001', 'healthy (K001)'), ('KA16', 'real outer-race damage (KA16)'), ('KI18', 'real inner-race damage (KI18)')]
    fig, axs = plt.subplots(3, 2, figsize=(7.2, 5.0), sharex='col')
    t = np.arange(4096) / 8000.0
    for r, (b, lab) in enumerate(pick):
        idx = np.flatnonzero((WB == b) & (WC == COND0))
        if len(idx) == 0: continue
        i0 = idx[0]
        axs[r, 0].plot(t, w['X'][i0, 0], color=BLUE, lw=0.4); axs[r, 1].plot(t, e['X'][i0, 0], color=ORANGE, lw=0.4)
        axs[r, 0].set_ylabel(lab, fontsize=7.5)
    axs[0, 0].set_title('(a) vibration, 8 kHz', fontsize=8, loc='left'); axs[0, 1].set_title('(b) 2–12 kHz envelope', fontsize=8, loc='left')
    axs[2, 0].set_xlabel('time (s)'); axs[2, 1].set_xlabel('time (s)')
    fig.tight_layout(); save(fig, 'D1_signals')
    del w, e

    # ---------------- D2 mean order spectra per group
    groups = [('healthy', (y == 0)), ('artificial OR', (y == 1) & (org == 'artificial')), ('real OR', (y == 1) & (org == 'real')),
              ('artificial IR', (y == 2) & (org == 'artificial')), ('real IR', (y == 2) & (org == 'real'))]
    colg = [GREY, BLUE, VIOLET, ORANGE, YELLOW]
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    for ax, keep in zip(axs, (('healthy', 'artificial OR', 'real OR'), ('healthy', 'artificial IR', 'real IR'))):
        for (g, m), c in zip(groups, colg):
            if g in keep: ax.plot(grid, O[m].mean(0), color=c, lw=1.2, label=f'{g} (n = {int(m.sum())})')
        for k, ls in (('BPFO', '--'), ('BPFI', ':')):
            for h in (1, 2):
                if h * fo[k] <= grid[-1]: ax.axvline(h * fo[k], color=INK, lw=0.7, ls=ls)
        ax.set_ylabel('log SES / median'); ax.legend(fontsize=7, loc='upper right')
    axs[0].set_title('(a) outer-race groups (dashed: BPFO harmonics, dotted: BPFI harmonics)', fontsize=8, loc='left')
    axs[1].set_title('(b) inner-race groups', fontsize=8, loc='left'); axs[1].set_xlabel('order (multiples of shaft frequency)')
    fig.tight_layout(); save(fig, 'D2_order_mean')

    # ---------------- D3 clear vs weak examples (median spectrum over the bearing's condition-0 recordings)
    pairs = [('KA16', 'KA30', 'BPFO'), ('KI18', 'KI17', 'BPFI'), ('K001', 'K005', 'BPFO')]
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True)
    for ax, (b1, b2, k) in zip(axs, pairs):
        for b, c in ((b1, BLUE), (b2, ORANGE)):
            m = (B == b) & (cond == COND0)
            if m.any(): ax.plot(grid, np.median(O[m], 0), color=c, lw=1.0, label=f'{b} ({meta[b]["origin"]}, level {meta[b]["extent"]})')
        for h in (1, 2, 3):
            if h * fo[k] <= grid[-1]: ax.axvline(h * fo[k], color=INK, lw=0.7, ls='--')
        ax.set_ylabel('log SES / median'); ax.legend(fontsize=7, loc='upper right'); ax.set_title(f'dashed: {k} harmonics', fontsize=7.5, loc='left')
    axs[2].set_xlabel('order (multiples of shaft frequency)'); fig.tight_layout(); save(fig, 'D3_order_examples')

    # ---------------- D4 feature boxes per bearing
    order_b = [b for b in meta if b in set(B)]
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
    for ax, k, c in ((axs[0], 'BPFO_h1', BLUE), (axs[1], 'BPFI_h1', ORANGE)):
        data = [F[B == b, names.index(k)] for b in order_b]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
        for patch, b in zip(bp['boxes'], order_b):
            lab = int(meta[b]['label']); patch.set_facecolor({0: GREY, 1: BLUE, 2: ORANGE, 3: AQUA}[lab]); patch.set_alpha(0.85); patch.set_linewidth(0.4)
        for med in bp['medians']: med.set_color(INK)
        ax.set_ylabel(k.replace('_h1', ' (1st harmonic)'))
        for xv in (6.5, 18.5, 29.5): ax.axvline(xv, color='#8c8b84', lw=0.7)
    axs[1].set_xticks(range(1, len(order_b) + 1)); axs[1].set_xticklabels(order_b, rotation=90, fontsize=6.5)
    axs[0].legend(handles=[Line2D([], [], marker='s', ls='', color=c, label=l) for c, l in ((GREY, 'healthy'), (BLUE, 'outer ring'), (ORANGE, 'inner ring'), (AQUA, 'both rings'))],
                  ncol=4, fontsize=7, loc='upper right')
    fig.tight_layout(); save(fig, 'D4_feature_boxes')

    # ---------------- D5 PCA of the 15 features (descriptive, all recordings)
    Z = StandardScaler().fit_transform(F); pca = PCA(2).fit(Z); C2 = pca.transform(Z)
    nums['pca_explained_variance'] = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for lab, c, nm in ((0, GREY, 'healthy'), (1, BLUE, 'outer ring'), (2, ORANGE, 'inner ring'), (3, AQUA, 'both rings')):
        for o, mk in (('healthy', 'o'), ('artificial', 'o'), ('real', 's')):
            m = (y == lab) & (org == o)
            if m.any(): ax.scatter(C2[m, 0], C2[m, 1], s=5, color=c, marker=mk, alpha=0.55, lw=0)
    ax.set_xlabel(f'PC 1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)'); ax.set_ylabel(f'PC 2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)')
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color=c, label=l) for c, l in ((GREY, 'healthy'), (BLUE, 'outer ring'), (ORANGE, 'inner ring'), (AQUA, 'both rings'))]
              + [Line2D([], [], marker='s', ls='', color=MUTED, label='real damage (squares)')], fontsize=7, loc='best')
    save(fig, 'D5_feature_pca')

    # ---------------- LOBO predictions of the primary model (used by D6-D8)
    single = [b for b in order_b if int(meta[b]['label']) < 3]
    prob = np.full((len(y), 3), np.nan)
    for b in single:
        tr, te = np.isin(B, [x for x in single if x != b]), B == b
        prob[te] = lr().fit(F[tr], y[tr]).predict_proba(F[te])
    keep = np.isin(B, single); pred = np.nan_to_num(prob).argmax(1); ok = (pred == y) & keep

    # ---------------- D6 per-condition accuracy
    cond_acc = {c: float(ok[keep & (cond == c)].sum() / (keep & (cond == c)).sum()) for c in CONDS}
    nums['lobo_accuracy_per_condition'] = {c: round(v, 4) for c, v in cond_acc.items()}
    per_cls = {}
    for c in CONDS:
        per_cls[c] = [float(ok[keep & (cond == c) & (y == k)].sum() / max((keep & (cond == c) & (y == k)).sum(), 1)) for k in range(3)]
    nums['lobo_recall_per_condition_H_OR_IR'] = {c: [round(v, 4) for v in per_cls[c]] for c in CONDS}
    fig, ax = plt.subplots(figsize=(5.6, 3.0)); xx = np.arange(4); wd = 0.2
    ax.bar(xx - 1.5 * wd, [cond_acc[c] * 100 for c in CONDS], wd * 0.92, color=INK, label='all classes')
    for k, col in zip(range(3), (GREY, BLUE, ORANGE)):
        ax.bar(xx + (k - 0.5) * wd, [per_cls[c][k] * 100 for c in CONDS], wd * 0.92, color=col, label=f'recall {CLS[k]}')
    for j, c in enumerate(CONDS): ax.text(j - 1.5 * wd, cond_acc[c] * 100 + 1.5, f'{cond_acc[c] * 100:.1f}', ha='center', fontsize=6.5)
    ax.set_xticks(xx); ax.set_xticklabels(CONDS, fontsize=7.5); ax.set_ylim(0, 108); ax.set_ylabel('LOBO accuracy / recall (%)')
    ax.legend(fontsize=7, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.15)); save(fig, 'D6_conditions')

    # ---------------- D7 confusion matrices
    def cm_of(trm, tem):
        p = lr().fit(F[trm], y[trm]).predict(F[tem]); M = np.zeros((3, 3), int)
        for t_, q in zip(y[tem], p): M[t_, q] += 1
        return M
    healthy = [b for b in single if meta[b]['origin'] == 'healthy']
    a2r_tr = healthy[:3] + [b for b in single if meta[b]['origin'] == 'artificial']; a2r_te = healthy[3:] + [b for b in single if meta[b]['origin'] == 'real']
    c0 = cond == COND0
    mats = {'L8, condition 0': cm_of(np.isin(B, L8_TRAIN) & c0, np.isin(B, L8_TEST) & c0),
            'A2R': cm_of(np.isin(B, a2r_tr), np.isin(B, a2r_te))}
    M = np.zeros((3, 3), int)
    for t_, q in zip(y[keep], pred[keep]): M[t_, q] += 1
    mats['LOBO'] = M
    nums['confusion_matrices'] = {k: v.tolist() for k, v in mats.items()}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (k, Mx) in zip(axs, mats.items()):
        R = Mx / Mx.sum(1, keepdims=True); ax.imshow(R, cmap='Blues', vmin=0, vmax=1); ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{Mx[i, j]}\n{100 * R[i, j]:.0f}%', ha='center', va='center', fontsize=6.5, color='white' if R[i, j] > 0.6 else INK)
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(CLS); ax.set_yticklabels(CLS)
        ax.set_title(k, fontsize=8); ax.set_xlabel('predicted')
    axs[0].set_ylabel('true'); fig.tight_layout(); save(fig, 'D7_confusions')

    # ---------------- D8 confidence of correct and wrong recordings (LOBO)
    conf = np.nan_to_num(prob).max(1)
    nums['lobo_confidence_median_correct_wrong'] = [round(float(np.median(conf[ok])), 4), round(float(np.median(conf[keep & ~ok])), 4)]
    fig, ax = plt.subplots(figsize=(5.0, 3.0)); bins = np.linspace(1 / 3, 1, 26)
    ax.hist(conf[ok], bins=bins, color=BLUE, alpha=0.85, label=f'correct (n = {int(ok.sum())})')
    ax.hist(conf[keep & ~ok], bins=bins, color=ORANGE, alpha=0.85, label=f'wrong (n = {int((keep & ~ok).sum())})')
    ax.set_xlabel('confidence (largest class probability)'); ax.set_ylabel('recordings'); ax.legend(fontsize=7)
    save(fig, 'D8_confidence')

    # ---------------- D9 kurtogram bands
    band = d['band_sk']; centre = band.mean(1) / 1000.0; bw = (band[:, 1] - band[:, 0]) / 1000.0
    nums['kurtogram_band_centre_khz_median_by_group'] = {g: round(float(np.median(centre[m])), 3) for g, m in groups}
    nums['kurtogram_band_width_khz_median'] = round(float(np.median(bw)), 3)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.boxplot([centre[m] for _, m in groups], widths=0.55, showfliers=False)
    ax.axhspan(2, 12, color=BLUE, alpha=0.08); ax.text(5.45, 11.5, 'fixed band 2–12 kHz', ha='right', fontsize=6.5, color=BLUE)
    ax.set_xticks(range(1, 6)); ax.set_xticklabels([g for g, _ in groups], fontsize=7.5); ax.set_ylabel('kurtogram band centre (kHz)')
    save(fig, 'D9_kurtogram')

    # ---------------- D10 measured speed
    rpm, src = d['rpm'], d['rpm_source']
    nums['measured_speed_share'] = round(float(np.mean(src == 'measured')), 4)
    nums['measured_speed_rpm_median_by_condition'] = {c: round(float(np.median(rpm[(cond == c) & (src == 'measured')])), 1) for c in CONDS if ((cond == c) & (src == 'measured')).any()}
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    ax.boxplot([rpm[cond == c] for c in CONDS], widths=0.55, showfliers=True, flierprops=dict(markersize=2))
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(CONDS, fontsize=7.5); ax.set_ylabel('shaft speed used (rpm)')
    save(fig, 'D10_speed')


    # ---------------- D11 distribution checks: histogram + fitted normal + Q-Q
    from scipy import stats as st
    hb = F[(y == 0), names.index('BPFO_h1')]
    sp = rpm[(cond == C15) & (src == 'measured')] if ((cond == C15) & (src == 'measured')).any() else rpm[cond == C15]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for col, (vals, lab) in enumerate(((hb, 'BPFO feature, healthy bearings'), (sp, 'measured shaft speed, condition 0 (rpm)'))):
        mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
        ax = axs[0, col]; ax.hist(vals, bins=30, density=True, color='#9ec5f4', edgecolor='white', label='observed')
        xs = np.linspace(vals.min(), vals.max(), 300); ax.plot(xs, st.norm.pdf(xs, mu, sd), color='#e34948', lw=1.6, label=f'normal fit (\u03bc={mu:.2f}, \u03c3={sd:.2f})')
        ax.set_xlabel(lab, fontsize=7.5); ax.set_ylabel('probability density'); ax.legend(fontsize=6.5)
        (osm, osr), (slope, icpt, r) = st.probplot(vals, dist='norm')
        ax = axs[1, col]; ax.scatter(osm, osr, s=4, color='#4a3aa7', alpha=0.7); ax.plot(osm, slope * osm + icpt, color='#e34948', ls='--', lw=1)
        ax.set_xlabel('theoretical quantiles (normal)'); ax.set_ylabel('observed quantiles'); ax.set_title(f'Q\u2013Q plot, r = {r:.3f}', fontsize=8)
        nums[f'distribution_{col}'] = {'what': lab, 'n': int(len(vals)), 'mean': round(mu, 4), 'sd': round(sd, 4), 'qq_r': round(float(r), 4),
                                       'shapiro_p_on_500': round(float(st.shapiro(np.random.default_rng(0).choice(vals, min(500, len(vals)), replace=False))[1]), 4)}
    fig.tight_layout(); save(fig, 'D11_distribution')

    # ---------------- D12 3-D waterfalls of the order spectrum over all recordings of a bearing
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7.2, 6.2)); sel = (grid >= 0.5) & (grid <= 11.0)
    for k, b in enumerate(('K001', 'KA16', 'KA30', 'KI18')):
        ax = fig.add_subplot(2, 2, k + 1, projection='3d'); m = np.flatnonzero(B == b)
        m = m[np.argsort([CONDS.index(cc) for cc in cond[m]], kind='stable')]
        Zs = np.clip(O[m][:, sel], 0, None); Xg, Yg = np.meshgrid(grid[sel], np.arange(len(m)))
        ax.plot_surface(Xg, Yg, Zs, cmap='viridis', linewidth=0, antialiased=False, rstride=1, cstride=2)
        for kk, ls_ in (('BPFO', '--'), ('BPFI', ':')):
            for h in (1, 2):
                xv = h * fo[kk]
                if xv <= 11: ax.plot([xv, xv], [0, len(m) - 1], [0, 0], color='#e34948' if kk == 'BPFO' else '#eda100', lw=0.8, ls=ls_)
        ax.set_xlabel('order', fontsize=7, labelpad=-2); ax.set_ylabel('recording', fontsize=7, labelpad=-2); ax.set_zlabel('log SES', fontsize=7, labelpad=-4)
        ax.tick_params(labelsize=5.5, pad=-1); ax.set_title(f'{b} ({meta[b]["origin"]}' + (f', level {meta[b]["extent"]})' if meta[b]['origin'] != 'healthy' else ')'), fontsize=8)
        ax.view_init(elev=32, azim=-62)
    fig.tight_layout(); save(fig, 'D12_waterfall')

    # ---------------- D13 spectrograms of raw vibration
    from scipy.signal import spectrogram
    w = np.load(a.windows); WB, WC = w['bearing'], w['condition']
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, (b, lab) in zip(axs, (('K001', 'healthy K001'), ('KA16', 'outer race KA16'), ('KI18', 'inner race KI18'))):
        idx = np.flatnonzero((WB == b) & (WC == COND0))
        if len(idx) == 0: continue
        f_, t_, Sxx = spectrogram(w['X'][idx[0], 0].astype(float), fs=8000, nperseg=128, noverlap=96)
        im = ax.pcolormesh(t_ * 1000, f_ / 1000, 10 * np.log10(Sxx + 1e-12), shading='auto', cmap='viridis')
        ax.set_title(lab, fontsize=8); ax.set_xlabel('time (ms)'); ax.grid(False)
    axs[0].set_ylabel('frequency (kHz)'); fig.colorbar(im, ax=axs, shrink=0.85, label='power (dB)')
    save(fig, 'D13_spectrogram'); del w

    # ---------------- D14 heatmap of LOBO accuracy per bearing and condition
    H = np.array([[ok[(B == b) & (cond == cc)].mean() if ((B == b) & (cond == cc)).any() else np.nan for cc in CONDS] for b in single])
    nums['lobo_accuracy_bearing_by_condition'] = {b: [round(float(v), 3) for v in row] for b, row in zip(single, H)}
    fig, ax = plt.subplots(figsize=(7.2, 2.6)); ax.grid(False)
    im = ax.imshow(H.T * 100, aspect='auto', cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(len(single))); ax.set_xticklabels(single, rotation=90, fontsize=6.5); ax.set_yticks(range(4)); ax.set_yticklabels(CONDS, fontsize=7)
    fig.colorbar(im, ax=ax, label='accuracy (%)', shrink=0.9); save(fig, 'D14_heatmap')

    nums['n_recordings'] = int(len(y)); nums['lobo_overall_accuracy'] = round(float(ok[keep].mean()), 4)
    (out / 'phase5_numbers.json').write_text(json.dumps(nums, indent=1))
    shutil.make_archive(str(out), 'zip', out)
    print(json.dumps(nums, indent=1)); print('figures and zip written to', out, 'and', str(out) + '.zip')


if __name__ == '__main__':
    main()
