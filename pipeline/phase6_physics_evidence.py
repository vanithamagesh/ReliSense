"""Phase 6 (v1.0): does the physics carry the diagnosis? Five pre-declared experiments (CPU).

E1 wrong-physics control   peak features computed from the order spectrum at the true fault orders and at
                           wrong ones (all fault orders scaled by 0.80-1.20; 20 sets of random orders).
E2 speed transfer          train on the 1500-rpm conditions, test on the 900-rpm condition of the held-out
                           bearing (LOBO). Kinematic order features vs. spectral band energies in Hz vs. time statistics.
E3 representation ladder   logistic regression on the 590-bin order spectrum, a small 1-D CNN on the order spectrum
                           (OrderCNN), and the 15 peak features; LOBO and L8. (Raw and envelope WDCNN results come
                           from Phase 4 and are only quoted.)
E4 interpretability        standardised logistic-regression weights of the 15 peak features (fit on all 29 bearings).
E5 data efficiency         accuracy on held-out bearings when only k = 1..5 training bearings per class are used
                           (30 random draws per k): peak features + LR, band energies + LR, time+band statistics + RF.
Every split is by bearing; scaling uses training recordings only; all settings are fixed in this file.
"""
from __future__ import annotations
import argparse, csv, json, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import physics64 as P
from phase2_physics import L8_TRAIN, L8_TEST
from phase3b_ladder import build_features

BLUE, ORANGE, AQUA, YELLOW, GREY, INK, MUTED, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#9a9a93', '#222222', '#5f5e58', '#4a3aa7'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': '#b5b4ad', 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5df', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
COND0 = 'N15_M07_F10'; SLOW = 'N09_M07_F10'


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))


def order_peaks(O, grid, orders, harmonics=(1, 2)):
    """Peak-to-background from the log order spectrum: max within +-2% (>= 0.04 order) minus median within +-0.4 order."""
    cols = []
    for o in orders:
        for h in harmonics:
            c = h * o; w = max(0.02 * c, 0.04)
            band = np.abs(grid - c) <= w; bg = (np.abs(grid - c) <= 0.4) & ~band
            if not band.any() or not bg.any(): cols.append(np.zeros(len(O))); continue
            cols.append(O[:, band].max(1) - np.median(O[:, bg], 1))
    return np.stack(cols, 1)


def lobo_acc(X, y, B, single, mask=None, test_mask=None, model=lr):
    ok = n = 0; per = {}
    for b in single:
        tr = np.isin(B, [x for x in single if x != b]); te = B == b
        if mask is not None: tr &= mask
        if test_mask is not None: te &= test_mask
        if not te.any() or not tr.any(): continue
        p = model().fit(X[tr], y[tr]).predict(X[te]); ok += (p == y[te]).sum(); n += te.sum(); per[b] = float((p == y[te]).mean())
    return ok / max(n, 1), per


def l8_acc(X, y, B, cond, c0=True, model=lr):
    tr, te = np.isin(B, L8_TRAIN), np.isin(B, L8_TEST)
    if c0: tr &= cond == COND0; te &= cond == COND0
    return float((model().fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean())


def order_cnn_fit_predict(Xtr, ytr, Xte, seed=0, epochs=60):
    import torch
    from torch import nn
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = Xtr.mean(), Xtr.std() + 1e-8
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32).unsqueeze(1); yt = torch.tensor(ytr, dtype=torch.long)
    net = nn.Sequential(nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
                        nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                        nn.Conv1d(32, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.AdaptiveMaxPool1d(8),
                        nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 3))
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-3)
    wts = torch.tensor(len(ytr) / (3 * np.maximum(np.bincount(ytr, minlength=3), 1)), dtype=torch.float32)
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(yt)).split(64):
            sh = int(np.random.randint(-2, 3))           # small order shift: speed and slip augmentation
            loss = nn.functional.cross_entropy(net(torch.roll(xt[b], sh, 2)), yt[b], weight=wts)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor((Xte - mu) / sd, dtype=torch.float32).unsqueeze(1)).argmax(1).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--physics', default='pb32_physics.npz')
    ap.add_argument('--windows', default='pb32_4096.npz')
    ap.add_argument('--envelope', default='pb32_4096_envelope.npz')
    ap.add_argument('--labels', default='labels_32.csv')
    ap.add_argument('--out', default='phase6_out')
    ap.add_argument('--draws', type=int, default=30)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); R = {}
    d = dict(np.load(a.physics)); meta = {r['bearing']: r for r in csv.DictReader(open(a.labels))}
    keep = d['label'] < 3
    d = {k: (v[keep] if hasattr(v, 'shape') and v.shape[:1] == keep.shape else v) for k, v in d.items()}
    B, y, cond = d['bearing'], d['label'], d['condition']; grid = d['order_grid']; O = d['order_fixed'].astype(float)
    single = [b for b in meta if b in set(B)]
    fo = P.fault_orders(); faults = [fo['BPFO'], fo['BPFI'], fo['BSF'], fo['FTF']]
    save = lambda fig, n: (fig.savefig(out / f'{n}.png'), plt.close(fig))

    # ---------------- E1 wrong-physics control
    print('E1 ...', flush=True)
    scales = [0.80, 0.85, 0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.15, 1.20]; e1 = {}
    for s in scales:
        X = np.concatenate([order_peaks(O, grid, [f * s for f in faults]), order_peaks(O, grid, [1.0], (1, 2, 3))], 1)
        e1[s] = {'LOBO': lobo_acc(X, y, B, single)[0], 'L8_cond0': l8_acc(X, y, B, cond)}
    rng = np.random.default_rng(0); rnd = []
    for _ in range(20):
        ro = list(rng.uniform(0.3, 5.9, 4))
        X = np.concatenate([order_peaks(O, grid, ro), order_peaks(O, grid, [1.0], (1, 2, 3))], 1)
        rnd.append({'orders': [round(v, 3) for v in ro], 'LOBO': lobo_acc(X, y, B, single)[0], 'L8_cond0': l8_acc(X, y, B, cond)})
    R['E1_scaled'] = {str(k): {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in e1.items()}
    R['E1_random'] = {'LOBO_mean': round(float(np.mean([r['LOBO'] for r in rnd])), 4), 'LOBO_max': round(float(np.max([r['LOBO'] for r in rnd])), 4),
                      'L8_mean': round(float(np.mean([r['L8_cond0'] for r in rnd])), 4), 'L8_max': round(float(np.max([r['L8_cond0'] for r in rnd])), 4), 'draws': rnd}
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for key, c, lab in (('LOBO', BLUE, 'LOBO'), ('L8_cond0', ORANGE, 'L8, condition 0')):
        ax.plot(scales, [e1[s][key] * 100 for s in scales], marker='o', color=c, lw=1.8, label=lab)
        m = np.mean([r[key] for r in rnd]) * 100; ax.axhline(m, color=c, ls=':', lw=1)
    ax.axvline(1.0, color=INK, lw=0.8, ls='--'); ax.text(1.003, 34, 'true 6203\nkinematics', fontsize=6.5)
    ax.axhline(100 / 3, color=MUTED, lw=0.6); ax.set_xlabel('scale applied to all fault orders'); ax.set_ylabel('recording accuracy (%)')
    ax.set_ylim(25, 100); ax.legend(fontsize=7, title='dotted: random orders (mean of 20)', title_fontsize=6.5); save(fig, 'E1_wrong_physics')

    # ---------------- feature sets needing windows (E2, E5)
    print('features from windows ...', flush=True)
    phys_full = dict(np.load(a.physics)); win = np.load(a.windows); env = np.load(a.envelope)
    feats, has = build_features(phys_full, {'X': win['X'], 'recording': win['recording']}, {'X': env['X']})
    del win, env
    Pf = d['feat_fixed']; S = feats['S'][keep]; T = feats['T'][keep]; TS = np.concatenate([T, S], 1)

    # ---------------- E2 speed transfer: train 1500 rpm, test 900 rpm (held-out bearing)
    print('E2 ...', flush=True)
    fast = cond != SLOW; slow = cond == SLOW; e2 = {}
    for name, X in (('kinematic peaks (orders)', Pf), ('band energies (Hz)', S), ('time statistics', T), ('time + band (RF)', TS)):
        mdl = (lambda: RandomForestClassifier(300, random_state=0, n_jobs=-1)) if 'RF' in name else lr
        cross = lobo_acc(X, y, B, single, mask=fast, test_mask=slow, model=mdl)[0]
        same = lobo_acc(X, y, B, single, mask=fast, test_mask=fast, model=mdl)[0]
        e2[name] = {'train1500_test1500': round(float(same), 4), 'train1500_test900': round(float(cross), 4)}
    R['E2_speed_transfer'] = e2
    fig, ax = plt.subplots(figsize=(5.6, 3.0)); x = np.arange(len(e2)); w = 0.36
    ax.bar(x - w / 2, [v['train1500_test1500'] * 100 for v in e2.values()], w * 0.92, color=BLUE, label='test at 1500 rpm')
    ax.bar(x + w / 2, [v['train1500_test900'] * 100 for v in e2.values()], w * 0.92, color=ORANGE, label='test at 900 rpm (unseen speed)')
    for i, v in enumerate(e2.values()):
        ax.text(i - w / 2, v['train1500_test1500'] * 100 + 1.5, f"{v['train1500_test1500'] * 100:.1f}", ha='center', fontsize=6.5)
        ax.text(i + w / 2, v['train1500_test900'] * 100 + 1.5, f"{v['train1500_test900'] * 100:.1f}", ha='center', fontsize=6.5)
    ax.axhline(100 / 3, color=MUTED, lw=0.6, ls=':'); ax.set_xticks(x); ax.set_xticklabels(list(e2), fontsize=7.5); ax.set_ylim(0, 108)
    ax.set_ylabel('LOBO accuracy (%)'); ax.legend(fontsize=7, loc='upper right'); save(fig, 'E2_speed_transfer')

    # ---------------- E3 representation ladder
    print('E3 ...', flush=True)
    sel = grid <= 11.8; Osel = O[:, sel]
    def lobo_cnn(mask_tr=None, mask_te=None):
        ok = n = 0
        for b in single:
            tr = np.isin(B, [x for x in single if x != b]); te = B == b
            p = order_cnn_fit_predict(Osel[tr], y[tr], Osel[te]); ok += (p == y[te]).sum(); n += te.sum()
        return ok / n
    def l8_cnn(c0):
        tr, te = np.isin(B, L8_TRAIN), np.isin(B, L8_TEST)
        if c0: tr &= cond == COND0; te &= cond == COND0
        return float((order_cnn_fit_predict(Osel[tr], y[tr], Osel[te]) == y[te]).mean())
    e3 = {'order spectrum + LR': {'LOBO': lobo_acc(Osel, y, B, single)[0], 'L8_cond0': l8_acc(Osel, y, B, cond), 'L8_all': l8_acc(Osel, y, B, cond, c0=False)},
          'order spectrum + 1-D CNN': {'LOBO': lobo_cnn(), 'L8_cond0': l8_cnn(True), 'L8_all': l8_cnn(False)},
          'kinematic peaks + LR': {'LOBO': lobo_acc(Pf, y, B, single)[0], 'L8_cond0': l8_acc(Pf, y, B, cond), 'L8_all': l8_acc(Pf, y, B, cond, c0=False)}}
    R['E3_ladder'] = {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in e3.items()}
    R['E3_quoted_phase4'] = {'WDCNN raw': {'LOBO': 0.619, 'L8_cond0': 0.464, 'L8_all': 0.475}, 'WDCNN envelope': {'LOBO': 0.681, 'L8_cond0': 0.708, 'L8_all': 0.599}}
    rows = [('WDCNN, raw\nvibration', R['E3_quoted_phase4']['WDCNN raw']), ('WDCNN,\nenvelope', R['E3_quoted_phase4']['WDCNN envelope']),
            ('order spectrum\n+ LR', e3['order spectrum + LR']), ('order spectrum\n+ 1-D CNN', e3['order spectrum + 1-D CNN']), ('kinematic\npeaks + LR', e3['kinematic peaks + LR'])]
    fig, ax = plt.subplots(figsize=(6.4, 3.2)); x = np.arange(len(rows)); w = 0.26
    for j, (key, c, lab) in enumerate((('L8_cond0', BLUE, 'L8, condition 0'), ('L8_all', AQUA, 'L8, all conditions'), ('LOBO', ORANGE, 'LOBO'))):
        ax.bar(x + (j - 1) * w, [r[1][key] * 100 for r in rows], w * 0.92, color=c, label=lab)
    ax.axhline(100 / 3, color=MUTED, lw=0.6, ls=':'); ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=7)
    ax.set_ylim(0, 105); ax.set_ylabel('recording accuracy (%)'); ax.legend(fontsize=7, ncol=3, loc='upper left')
    ax.annotate('', xy=(4.4, 101), xytext=(-0.4, 101), arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.8))
    ax.text(2, 102.5, 'more physics in the representation', ha='center', fontsize=7, color=MUTED); save(fig, 'E3_ladder')

    # ---------------- E4 interpretability: standardised LR weights
    m = lr().fit(Pf, y); W = m[-1].coef_; names = list(d['feature_names'])
    R['E4_weights'] = {cls: {n: round(float(v), 3) for n, v in zip(names, W[k])} for k, cls in enumerate(('H', 'OR', 'IR'))}
    fig, ax = plt.subplots(figsize=(7.0, 2.2)); ax.grid(False); lim = np.abs(W).max()
    im = ax.imshow(W, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
    ax.set_yticks(range(3)); ax.set_yticklabels(['healthy', 'outer ring', 'inner ring']); ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('_h', ' h') for n in names], rotation=60, fontsize=6.5); fig.colorbar(im, ax=ax, label='weight', shrink=0.9)
    save(fig, 'E4_weights')

    # ---------------- E5 data efficiency
    print('E5 ...', flush=True)
    rng = np.random.default_rng(1); cls_b = {k: [b for b in single if int(meta[b]['label']) == k] for k in range(3)}
    sets = {'kinematic peaks + LR': (Pf, lr), 'band energies + LR': (S, lr), 'time + band + RF': (TS, lambda: RandomForestClassifier(300, random_state=0, n_jobs=-1))}
    e5 = {k: {} for k in sets}
    for kk in (1, 2, 3, 4, 5):
        acc = {k: [] for k in sets}
        for _ in range(a.draws):
            te_b = [rng.choice(cls_b[c]) for c in range(3)]
            tr_b = sum([list(rng.choice([b for b in cls_b[c] if b not in te_b], kk, replace=False)) for c in range(3)], [])
            tr, te = np.isin(B, tr_b), np.isin(B, te_b)
            for k, (X, mdl) in sets.items(): acc[k].append(float((mdl().fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean()))
        for k in sets: e5[k][kk] = [round(float(np.mean(acc[k])), 4), round(float(np.std(acc[k])), 4)]
    R['E5_data_efficiency'] = e5
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    for (k, v), c in zip(e5.items(), (BLUE, ORANGE, AQUA)):
        ks = sorted(v); mu = np.array([v[q][0] for q in ks]) * 100; sd = np.array([v[q][1] for q in ks]) * 100
        ax.plot(ks, mu, marker='o', color=c, lw=1.8, label=k); ax.fill_between(ks, mu - sd, mu + sd, color=c, alpha=0.12, lw=0)
    ax.axhline(100 / 3, color=MUTED, lw=0.6, ls=':'); ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xlabel('training bearings per class')
    ax.set_ylabel('accuracy on new bearings (%)'); ax.set_ylim(20, 100); ax.legend(fontsize=7, loc='lower right'); save(fig, 'E5_data_efficiency')

    (out / 'phase6_numbers.json').write_text(json.dumps(R, indent=1))
    shutil.make_archive(str(out), 'zip', out)
    short = {k: v for k, v in R.items() if k not in ('E1_random', 'E4_weights')}
    short['E1_random_summary'] = {k: v for k, v in R['E1_random'].items() if k != 'draws'}
    print(json.dumps(short, indent=1)); print('written', out, str(out) + '.zip')


if __name__ == '__main__':
    main()
