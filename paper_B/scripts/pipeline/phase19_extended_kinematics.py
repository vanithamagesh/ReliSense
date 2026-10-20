"""Phase 19 (v1.0): extended kinematic branch on the Paderborn physics file (CPU, about 5-10 minutes; checkpoints).

Pre-specified before any result is seen; every model below is reported, none is dropped afterwards.
The 16 new features are read from the saved order spectrum order_fixed of pb32_physics.npz (log squared-envelope
spectrum on the fixed 2-12 kHz band, orders 0.2-12.0, step 0.02, divided by its median), so no raw signal is needed.
Each new feature = max of O(r) within +-max(2 %, 0.03) of the centre order minus the median of O(r) within +-0.4 orders
(window excluded); this is the peak-to-background ratio of phase 2 written on the order axis.
  sidebands of the inner race   h*BPFI +- 1 (shaft),  h = 1, 2                 4 features
  sidebands of the outer race   h*BPFO +- FTF (cage),  h = 1, 2                4 features
  sidebands of the ball         h*BSF  +- FTF (cage),  h = 1, 2                4 features
  higher harmonics              4*BSF, 5*BSF, 4*FTF, 5*FTF                     4 features
(4 and 5 times BPFO and BPFI lie above order 12 and are not available.)
Models (settings of phases 2 and 13, nothing tuned):
  LR      ReliSense, 15 features, LR C = 0.3                      (from phase 17 checkpoints)
  RF      envelope branch, 62 features, 500 trees, seed 0         (from phase 17 checkpoints)
  ENS     mean of LR and RF                                       (phase 17)
  LRX     extended kinematic branch, 15 + 16 = 31 features, LR C = 0.3
  ENSX    mean of LRX and RF
Protocols as phase 17: L8 (cond. 0 and all), L10 (10 splits), A2R, LOBO; LOBO abstention with inner LOBO (pooled
threshold, kappa 0.9 / 0.8 / 0.7). If a phase 17 checkpoint is missing, LR and RF are refitted here.
Output: <root>/phase19_results.json and <root>/checkpoints/phase19/
Colab:  !python /content/drive/MyDrive/ReliSense_study/phase19_extended_kinematics.py --root /content/drive/MyDrive/ReliSense_study
"""
import argparse, itertools, json, os, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser(); ap.add_argument('--root', default='/content/drive/MyDrive/ReliSense_study'); ap.add_argument('--fresh', action='store_true')
ap.add_argument('--physics', default=None); ap.add_argument('--trees', type=int, default=500)
a = ap.parse_args(); ROOT = a.root; CK = ROOT + '/checkpoints/phase19'; CK17 = ROOT + '/checkpoints/phase17'; os.makedirs(CK, exist_ok=True); t0 = time.time()
d = dict(np.load(a.physics or ROOT + '/pb32/pb32_physics.npz', allow_pickle=True))
B, y, cond = d['bearing'], d['label'].astype(int), d['condition']
grid = d['order_grid'].astype(float); O = np.nan_to_num(d['order_fixed'].astype(float))

# ------------------------------------------------------------------ fault orders of the 6203 (n = 8, d = 6.75 mm, D = 28.55 mm, contact angle 0)
n_, d_, D_ = 8, 6.75, 28.55; q = d_ / D_
BPFO, BPFI, BSF, FTF = n_ / 2 * (1 - q), n_ / 2 * (1 + q), D_ / (2 * d_) * (1 - q ** 2), (1 - q) / 2
CENTRES = ([(f'BPFI{h}{s}1', h * BPFI + sg) for h in (1, 2) for s, sg in (('-', -1), ('+', 1))] +
           [(f'BPFO{h}{s}FTF', h * BPFO + sg * FTF) for h in (1, 2) for s, sg in (('-', -1), ('+', 1))] +
           [(f'BSF{h}{s}FTF', h * BSF + sg * FTF) for h in (1, 2) for s, sg in (('-', -1), ('+', 1))] +
           [('BSF4', 4 * BSF), ('BSF5', 5 * BSF), ('FTF4', 4 * FTF), ('FTF5', 5 * FTF)])
assert all(grid[0] + 0.4 <= r <= grid[-1] - 0.4 for _, r in CENTRES), 'centre order outside the saved grid'


def ext_features(O):
    out = []
    for _, r0 in CENTRES:
        w = max(0.02 * r0, 0.03); win = np.abs(grid - r0) <= w; bg = (np.abs(grid - r0) <= 0.4) & ~win
        out.append(O[:, win].max(1) - np.median(O[:, bg], 1))
    return np.stack(out, 1)


X_lr = np.nan_to_num(d['feat_fixed'].astype(float)); X_x = np.c_[X_lr, ext_features(O)]
X_rf = np.nan_to_num(np.concatenate([d[f'feat_{v}'] for v in ('fixed', 'sk', 'cpw', 'cpw_sk')] +
                                    [np.log(d['stat_kurtosis'])[:, None], np.log(d['stat_crest'])[:, None]], 1).astype(float))
print(f'features: ReliSense {X_lr.shape[1]}, extended {X_x.shape[1]}, envelope branch {X_rf.shape[1]}', flush=True)
COND0 = 'N15_M07_F10'; KAP = (0.9, 0.8, 0.7); M5 = ('LR', 'RF', 'ENS', 'LRX', 'ENSX')
L8_TRAIN = ['K002', 'KA01', 'KA05', 'KA07', 'KI01', 'KI05', 'KI07']
L8_TEST = ['K001', 'KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']
L10 = [['K001', 'K002', 'K003', 'K004', 'K005'], ['KA04', 'KA15', 'KA16', 'KA22', 'KA30'], ['KI04', 'KI14', 'KI16', 'KI18', 'KI21']]
bearings = list(dict.fromkeys(B)); lab = {b: int(y[B == b][0]) for b in bearings}; org = {b: str(d['origin'][B == b][0]) for b in bearings}
single = [b for b in bearings if lab[b] < 3]


def proba(m, X):
    P = np.zeros((len(X), 3)); P[:, m.classes_] = m.predict_proba(X); return P


def lr_fit(X, tr): return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)).fit(X[tr], y[tr])


def base_probs(tr, te):
    """LR and RF of phase 17 (refitted only when no phase 17 checkpoint exists)."""
    rf = RandomForestClassifier(a.trees, random_state=0, n_jobs=-1).fit(X_rf[tr], y[tr])
    return proba(lr_fit(X_lr, tr), X_lr[te]), proba(rf, X_rf[te])


def five(pl, pr, tr, te):
    px = proba(lr_fit(X_x, tr), X_x[te]); return {'LR': pl, 'RF': pr, 'ENS': (pl + pr) / 2, 'LRX': px, 'ENSX': (px + pr) / 2}


def ck(name, fn):
    p = f'{CK}/{name}.npz'
    if os.path.exists(p) and not a.fresh: return dict(np.load(p))
    r = fn(); np.savez_compressed(p, **r); return r


def ck17(name):
    p = f'{CK17}/{name}.npz'; return dict(np.load(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ all protocols
def protocol_splits():
    yield 'L8_c0', 0, L8_TRAIN, L8_TEST, True
    yield 'L8_all', 0, L8_TRAIN, L8_TEST, False
    for k, tr in enumerate(itertools.combinations(range(5), 3)):
        yield 'L10', k, [c[i] for c in L10 for i in tr], [c[i] for c in L10 for i in range(5) if i not in tr], False
    healthy = sorted(b for b in bearings if lab[b] == 0); art = [b for b in bearings if org[b] == 'artificial']
    real = [b for b in bearings if org[b] == 'real' and lab[b] < 3]
    yield 'A2R', 0, healthy[:3] + art, healthy[3:] + real, False


res = {'centres': {k: float(r) for k, r in CENTRES}, 'protocols': {}}; reused = 0
for proto, k, trb, teb, c0 in protocol_splits():
    def run():
        global reused
        tr, te = np.isin(B, trb), np.isin(B, teb)
        if c0: tr &= cond == COND0; te &= cond == COND0
        o = ck17(f'{proto}_{k}')
        if o is not None and len(o['y']) == te.sum(): pl, pr = o['LR'], o['RF']; reused += 1
        else: pl, pr = base_probs(tr, te)
        return {**five(pl, pr, tr, te), 'y': y[te]}
    r = ck(f'{proto}_{k}', run)
    for m in M5:
        p = r[m].argmax(1); yt = r['y']; h = yt == 0
        res['protocols'].setdefault(proto, {}).setdefault(m, []).append({'acc': float((p == yt).mean()), 'fa': float((p[h] != 0).mean()) if h.any() else None})
print(f'protocols done ({time.time() - t0:.0f} s)', flush=True)

# ------------------------------------------------------------------ LOBO with inner LOBO
OUT = {m: np.full((len(y), 3), np.nan) for m in M5}; INNER = {}
for ib, b in enumerate(single):
    def run():
        tr = [x for x in single if x != b]; te = B == b; o = ck17(f'lobo_{b}')
        if o is not None:
            pl, pr = o['outer_LR'], o['outer_RF']
        else:
            pl, pr = base_probs(np.isin(B, tr), te)
        r = {f'outer_{m}': v for m, v in five(pl, pr, np.isin(B, tr), te).items()}
        pools = {m: [] for m in M5}; start = 0
        for v in tr:
            itr, ite = np.isin(B, [x for x in tr if x != v]), B == v; nv = ite.sum()
            if o is not None:
                il, ir = o['inner_LR'][start:start + nv], o['inner_RF'][start:start + nv]
            else:
                il, ir = base_probs(itr, ite)
            start += nv
            for m, P in five(il, ir, itr, ite).items(): pools[m].append(P)
        if o is not None: assert start == len(o['inner_LR']), 'phase 17 inner pool has a different size'
        for m in pools: r[f'inner_{m}'] = np.concatenate(pools[m])
        return r
    r = ck(f'lobo_{b}', run)
    for m in M5: OUT[m][B == b] = r[f'outer_{m}']; INNER[(b, m)] = r[f'inner_{m}']
    print(f'  LOBO {ib + 1}/{len(single)} {b}  ({time.time() - t0:.0f} s)', flush=True)
keep = np.isin(B, single); res['protocols']['LOBO'] = {}
for m in M5:
    p = OUT[m][keep].argmax(1); yt = y[keep]
    res['protocols']['LOBO'][m] = [{'acc': float((p == yt).mean()), 'fa': float((p[yt == 0] != 0).mean())}]
    # per-bearing accuracy, for the paired comparison of the extended and the original branch
    res.setdefault('lobo_per_bearing', {})[m] = {b: float((OUT[m][B == b].argmax(1) == lab[b]).mean()) for b in single}


def selective(m, kappa):
    P = OUT[m][keep]; yt = y[keep]; Bk = B[keep]; conf = P.max(1); pred = P.argmax(1); acc = np.zeros(len(yt), bool)
    for b in single:
        ic = INNER[(b, m)].max(1); acc[Bk == b] = conf[Bk == b] >= np.quantile(ic, 1 - kappa)
    ok = pred == yt; h = yt == 0; dmg = ~h
    return {'coverage': float(acc.mean()), 'sel_acc': float(ok[acc].mean()), 'false_alarm': float((acc & h & (pred != 0)).sum() / h.sum()),
            'missed_fault': float((acc & dmg & (pred == 0)).sum() / dmg.sum())}


res['abstention'] = {m: {f'pooled_{k}': selective(m, k) for k in KAP} for m in M5}
res['reused_phase17_protocol_splits'] = reused
json.dump(res, open(ROOT + '/phase19_results.json', 'w'), indent=1)

print('\nRecording accuracy (%) / healthy false alarms (%)  [L10: mean of 10 splits]')
print(f"{'protocol':8}" + ''.join(f'{m:>15}' for m in M5))
for proto in ('L8_c0', 'L8_all', 'L10', 'A2R', 'LOBO'):
    row = f'{proto:8}'
    for m in M5:
        R = res['protocols'][proto][m]; ac = 100 * np.mean([x['acc'] for x in R]); fa = [x['fa'] for x in R if x['fa'] is not None]
        row += f'{ac:8.1f} / {100 * np.mean(fa):4.1f}' if fa else f'{ac:8.1f} /   - '
    print(row)
print('\nLOBO abstention (pooled threshold): coverage / selective accuracy / false alarms / missed faults (%)')
for m in M5:
    for k in KAP:
        r = res['abstention'][m][f'pooled_{k}']
        print(f'  {m:5} kappa {k}  cov {100 * r["coverage"]:5.1f}  acc {100 * r["sel_acc"]:5.1f}  FA {100 * r["false_alarm"]:5.1f}  missed {100 * r["missed_fault"]:5.1f}')
pb = res['lobo_per_bearing']; dlt = np.array([pb['LRX'][b] - pb['LR'][b] for b in single])
print(f'\nLRX - LR per bearing: better on {(dlt > 0.005).sum()}, worse on {(dlt < -0.005).sum()}, equal on {(np.abs(dlt) <= 0.005).sum()} of {len(single)}')
print(f'written {ROOT}/phase19_results.json  ({time.time() - t0:.0f} s)')
