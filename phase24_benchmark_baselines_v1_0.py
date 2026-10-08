"""Phase 24 (v1.0): optimised classical baselines and domain-generalization networks under the bearing-wise Paderborn
protocols, for Paper B (benchmark of representations for unseen bearings). Fixed before any result is seen; every
configuration is reported.

Part 1, tuned classical baselines (CPU). Representations: 32 band energies (S), time + band statistics (T+S), the
order spectrum up to 11.8 orders (590 bins) and the 15-feature kinematic core (feat_fixed). Classifiers and grids:
  lr   logistic regression, C in {0.01, 0.1, 1, 10}
  svm  RBF-SVM, C in {1, 10, 100} x gamma in {0.1, 1} x 'scale'
  rf   random forest (300 trees), max_depth in {None, 10} x max_features in {sqrt, 0.3}
  hgb  gradient boosting (200 iterations), learning_rate in {0.05, 0.1} x max_depth in {3, None}
Hyperparameters, and in the 'best' row also the classifier, are chosen by inner stratified group K-fold over the
TRAINING bearings only (groups = bearings, K = min(4, smallest class count)); the chosen model is refitted on all training
bearings and tested once. Protocols L10 (10 splits), A2R and LOBO (29 folds); L8 is skipped because it has a single
healthy training bearing, so no inner fold can hold out a healthy bearing.

Part 2, domain generalization (GPU). Domains = training bearings. Methods (DomainBed-style losses):
  erm     class-weighted cross-entropy
  dann    domain-adversarial training, gradient reversal, bearing classifier, lambda in {0.1, 1}
  coral   Deep CORAL, mean and covariance alignment between the bearings of a batch, lambda in {0.1, 1}
  gdro    group DRO over bearings, eta in {0.01, 0.1}
Inputs: 'env' = 4096-sample 2-12-kHz envelope windows with a WDCNN backbone (wide first kernel), recording probability
= mean of its windows; 'ord' = order spectrum up to 11.8 orders with a small 1-D CNN, one input per recording.
Batches are domain-balanced (8 bearings x 16 samples). Training as phase 23: AdamW lr 1e-3, weight decay 1e-4,
20 epochs (env) / 60 epochs (ord), last epoch used. Model selection (lambda/eta) by one inner bearing-wise validation split:
one training bearing per class held out (chosen by seed), the others train; the setting with the higher validation
accuracy is retrained on all training bearings and tested once (ERM has no hyperparameter and is trained once).
Protocols: L8 cond. 0, L8 all, L10 (10 splits), A2R, LOBO (29 folds); seed 0.

Every fit is checkpointed in <root>/checkpoints/phase24/ and reused on a second run (resume after a disconnect).
Output: <root>/phase24_results_part1.json and _part2.json (accuracy, false alarms, per-bearing accuracy, chosen settings).
Colab:
  !python /content/drive/MyDrive/ReliSense_study/phase24_benchmark_baselines.py --root /content/drive/MyDrive/ReliSense_study --part 1
  !python /content/drive/MyDrive/ReliSense_study/phase24_benchmark_baselines.py --root /content/drive/MyDrive/ReliSense_study --part 2   (GPU)
  (--no_lobo for a quick first pass; --inputs env or ord, --methods erm,dann to run a subset)
"""
import argparse, itertools, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--root', default='/content/drive/MyDrive/ReliSense_study')
ap.add_argument('--part', type=int, default=1); ap.add_argument('--no_lobo', action='store_true')
ap.add_argument('--inputs', default='env,ord'); ap.add_argument('--methods', default='erm,dann,coral,gdro')
ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--epochs_order', type=int, default=60)
a = ap.parse_args(); R = a.root; CK = R + '/checkpoints/phase24'; os.makedirs(CK, exist_ok=True); t0 = time.time()
sys.path.insert(0, R + '/phase1')
COND0 = 'N15_M07_F10'
L8_TRAIN = ['K002', 'KA01', 'KA05', 'KA07', 'KI01', 'KI05', 'KI07']
L8_TEST = ['K001', 'KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']
L10 = [['K001', 'K002', 'K003', 'K004', 'K005'], ['KA04', 'KA15', 'KA16', 'KA22', 'KA30'], ['KI04', 'KI14', 'KI16', 'KI18', 'KI21']]

ph = dict(np.load(R + '/pb32/pb32_physics.npz', allow_pickle=True))
keep = ph['label'] < 3; PB, Py, Pc = ph['bearing'][keep], ph['label'][keep].astype(int), ph['condition'][keep]
org = {b: str(o) for b, o in zip(ph['bearing'], ph['origin'])}; lab = {b: int(l) for b, l in zip(PB, Py)}
single = sorted(set(PB)); healthy = sorted(b for b in single if lab[b] == 0)


def protocols(with_l8):
    if with_l8:
        yield 'L8_c0', 0, L8_TRAIN, L8_TEST, True
        yield 'L8_all', 0, L8_TRAIN, L8_TEST, False
    for k, tr in enumerate(itertools.combinations(range(5), 3)):
        yield 'L10', k, [c[i] for c in L10 for i in tr], [c[i] for c in L10 for i in range(5) if i not in tr], False
    yield 'A2R', 0, healthy[:3] + [b for b in single if org[b] == 'artificial'], healthy[3:] + [b for b in single if org[b] == 'real'], False
    if not a.no_lobo:
        for k, b in enumerate(single): yield 'LOBO', k, [x for x in single if x != b], [b], False


def summary(P, y, b):
    ok = P.argmax(1) == y; h = y == 0
    return {'acc': float(ok.mean()), 'fa': float((P[h].argmax(1) != 0).mean()) if h.any() else None, 'n': int(len(y)),
            'per_bearing': {str(x): float(ok[b == x].mean()) for x in sorted(set(b))}}


RES_PATH = R + f'/phase24_results_part{a.part}.json'
RES = {}   # rebuilt from the checkpoints on every run, so a resumed run never counts a fold twice

# ===================================================================================== part 1
if a.part == 1:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    fp = CK + '/generic_features.npz'
    if os.path.exists(fp):
        Fz = dict(np.load(fp)); S, T = Fz['S'], Fz['T']
    else:
        from phase3b_ladder import build_features
        win = np.load(R + '/pb32/pb32_4096.npz'); env = np.load(R + '/pb32/pb32_4096_envelope.npz')
        feats, _ = build_features(ph, {'X': win['X'], 'recording': win['recording']}, {'X': env['X']})
        S = np.nan_to_num(feats['S'][keep]); T = np.nan_to_num(feats['T'][keep]); del win, env
        np.savez_compressed(fp, S=S, T=T); print('generic features computed', S.shape, T.shape, f'({time.time() - t0:.0f} s)', flush=True)
    g = ph['order_grid'].astype(float)
    REP = {'band': S, 'timeband': np.concatenate([T, S], 1), 'order': np.nan_to_num(ph['order_fixed'][keep][:, g <= 11.8].astype(float)),
           'core': ph['feat_fixed'][keep].astype(float)}

    def grid(m, nf):
        if m == 'lr': return [dict(C=c) for c in (0.01, 0.1, 1, 10)]
        if m == 'svm': return [dict(C=c, gm=gm) for c in (1, 10, 100) for gm in (0.1, 1.0)]
        if m == 'rf': return [dict(d=d, mf=mf) for d in (None, 10) for mf in ('sqrt', 0.3)]
        return [dict(lr=lr, d=d) for lr in (0.05, 0.1) for d in (3, None)]

    def model(m, p, nf):
        if m == 'lr': return make_pipeline(StandardScaler(), LogisticRegression(C=p['C'], max_iter=5000))
        if m == 'svm': return make_pipeline(StandardScaler(), SVC(C=p['C'], gamma=p['gm'] / nf))
        if m == 'rf': return RandomForestClassifier(300, max_depth=p['d'], max_features=p['mf'], class_weight='balanced', random_state=0, n_jobs=-1)
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=p['lr'], max_depth=p['d'], random_state=0)

    def inner_score(X, y, grp, m, p):
        k = int(min(4, min(len(set(grp[y == c])) for c in range(3))))
        if k < 2: return np.nan
        sc = []
        for itr, iva in StratifiedGroupKFold(k, shuffle=True, random_state=0).split(X, y, grp):
            if len(set(y[itr])) < 3: continue
            sc.append((model(m, p, X.shape[1]).fit(X[itr], y[itr]).predict(X[iva]) == y[iva]).mean())
        return float(np.mean(sc)) if sc else np.nan

    MODELS = ('lr', 'svm', 'rf', 'hgb')
    for rep, X in REP.items():
        for proto, fold, trb, teb, c0 in protocols(False):
            key = f'{rep}|{proto}|{fold}'
            p = f'{CK}/p1_{rep}_{proto}_{fold}.json'
            if os.path.exists(p): out = json.load(open(p))
            else:
                tr, te = np.isin(PB, trb), np.isin(PB, teb); out = {}
                cand = []
                for m in MODELS:
                    scores = [(inner_score(X[tr], Py[tr], PB[tr], m, q), i, q) for i, q in enumerate(grid(m, X.shape[1]))]
                    s_best, _, q_best = max(scores, key=lambda z: (-1 if np.isnan(z[0]) else z[0], -z[1]))
                    pred = model(m, q_best, X.shape[1]).fit(X[tr], Py[tr]).predict(X[te]); P = np.eye(3)[pred]
                    out[m] = dict(summary(P, Py[te], PB[te]), inner=s_best, params={k_: str(v) for k_, v in q_best.items()})
                    cand.append((s_best, MODELS.index(m), m))
                best = max(cand, key=lambda z: (-1 if np.isnan(z[0]) else z[0], -z[1]))[2]
                out['best'] = dict(out[best], chosen=best)
                json.dump(out, open(p, 'w'))
            for m, r in out.items(): RES.setdefault(rep, {}).setdefault(m, {}).setdefault(proto, []).append(dict(r, fold=fold))
        for m in list(MODELS) + ['best']:
            row = f'  {rep:8s} {m:4s}'
            for proto in ('L10', 'A2R', 'LOBO'):
                if proto in RES[rep][m]:
                    rr = RES[rep][m][proto]; ys = sum(x['n'] for x in rr)
                    row += f' | {proto} {100 * sum(x["acc"] * x["n"] for x in rr) / ys:5.1f}'
            print(row, f'({time.time() - t0:.0f} s)', flush=True)
        json.dump(RES, open(RES_PATH, 'w'), indent=1)

# ===================================================================================== part 2
else:
    import torch
    from torch import nn
    import torch.nn.functional as F
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; print('device', dev, flush=True)

    class WDCNN(nn.Module):
        def __init__(s, n_cls=3):
            super().__init__()
            def blk(i, o, k=3, p=1): return [nn.Conv1d(i, o, k, 1, p), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2)]
            s.f = nn.Sequential(nn.Conv1d(1, 16, 64, 16, 24), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
                                *blk(16, 32), *blk(32, 64), *blk(64, 64), *blk(64, 64, 3, 0), nn.AdaptiveAvgPool1d(4), nn.Flatten(),
                                nn.Linear(256, 100), nn.ReLU())
            s.h = nn.Linear(100, n_cls); s.dim = 100

        def forward(s, x): return s.h(s.f(x))

    class OrdCNN(nn.Module):
        def __init__(s, n_cls=3):
            super().__init__()
            def blk(i, o): return [nn.Conv1d(i, o, 5, 1, 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2)]
            s.f = nn.Sequential(*blk(1, 16), *blk(16, 32), *blk(32, 64), *blk(64, 64), nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Linear(256, 64), nn.ReLU())
            s.h = nn.Linear(64, n_cls); s.dim = 64

        def forward(s, x): return s.h(s.f(x))

    class GRL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, l): ctx.l = l; return x.view_as(x)

        @staticmethod
        def backward(ctx, g): return -ctx.l * g, None

    def coral(fa, fb):
        ma, mb = fa.mean(0, keepdim=True), fb.mean(0, keepdim=True); ca, cb = fa - ma, fb - mb
        va = ca.t() @ ca / max(len(fa) - 1, 1); vb = cb.t() @ cb / max(len(fb) - 1, 1)
        return (ma - mb).pow(2).mean() + (va - vb).pow(2).mean()

    def fit_predict(inp, method, hp, Xtr, ytr, dtr, Xte, epochs, seed=0):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        m = (WDCNN() if inp == 'env' else OrdCNN()).to(dev); doms = np.unique(dtr); nd = len(doms); dix = np.searchsorted(doms, dtr)
        dh = nn.Sequential(nn.Linear(m.dim, 64), nn.ReLU(), nn.Linear(64, nd)).to(dev) if method == 'dann' else None
        params = list(m.parameters()) + (list(dh.parameters()) if dh is not None else [])
        opt = torch.optim.AdamW(params, 1e-3, weight_decay=1e-4)
        w = torch.tensor(len(ytr) / (3 * np.maximum(np.bincount(ytr, minlength=3), 1)), dtype=torch.float32, device=dev)
        q = torch.ones(nd, device=dev) / nd
        X_ = torch.from_numpy(Xtr[:, None].astype(np.float32)); Y_ = torch.from_numpy(ytr.astype('int64')); D_ = torch.from_numpy(dix.astype('int64'))
        by_dom = [np.where(dix == k)[0] for k in range(nd)]; steps = max(1, len(ytr) // 128)
        for _ in range(epochs):
            m.train()
            for _ in range(steps):
                ds = rng.choice(nd, size=min(8, nd), replace=False)
                idx = np.concatenate([rng.choice(by_dom[k], size=16, replace=len(by_dom[k]) < 16) for k in ds])
                x, yb, db = X_[idx].to(dev), Y_[idx].to(dev), D_[idx].to(dev)
                f = m.f(x); logit = m.h(f); ce = F.cross_entropy(logit, yb, weight=w, reduction='none')
                if method == 'gdro':
                    lg = torch.stack([ce[db == k].mean() for k in ds])
                    with torch.no_grad(): q[ds] *= torch.exp(hp * lg); q /= q.sum()
                    loss = (q[ds] / q[ds].sum() * lg).sum()
                else:
                    loss = ce.mean()
                    if method == 'dann': loss = loss + F.cross_entropy(dh(GRL.apply(f, hp)), db)
                    if method == 'coral':
                        fs = [f[db == k] for k in ds]; pen = [coral(fs[i], fs[j]) for i in range(len(fs)) for j in range(i + 1, len(fs))]
                        loss = loss + hp * torch.stack(pen).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); out = []; xe = torch.from_numpy(Xte[:, None].astype(np.float32))
        with torch.no_grad():
            for b in torch.arange(len(xe)).split(512): out.append(F.softmax(m(xe[b].to(dev)), 1).cpu().numpy())
        return np.concatenate(out)

    g = ph['order_grid'].astype(float); ORD = np.nan_to_num(ph['order_fixed'][keep][:, g <= 11.8].astype(np.float32))
    ORD = (ORD - ORD.mean(1, keepdims=True)) / (ORD.std(1, keepdims=True) + 1e-8)
    WIN = None

    def data(inp):
        global WIN
        if inp == 'ord': return {'X': ORD, 'B': PB, 'y': Py, 'c': Pc, 'rec': np.arange(len(PB))}
        if WIN is None:
            w = np.load(R + '/pb32/pb32_4096.npz'); e = np.load(R + '/pb32/pb32_4096_envelope.npz')
            x = e['X'][:, 0].astype(np.float32); x -= x.mean(1, keepdims=True); x /= x.std(1, keepdims=True) + 1e-8
            WIN = {'X': x, 'B': w['bearing'], 'y': w['y'].astype(int), 'c': w['condition'], 'rec': w['recording']}
        return WIN

    def to_rec(D, sel, pw):
        ids, inv = np.unique(D['rec'][sel], return_inverse=True); P = np.zeros((len(ids), 3)); np.add.at(P, inv, pw); P /= np.bincount(inv)[:, None]
        first = np.array([np.where(inv == i)[0][0] for i in range(len(ids))])
        return P, D['y'][sel][first], D['B'][sel][first]

    HP = {'erm': [None], 'dann': [0.1, 1.0], 'coral': [0.1, 1.0], 'gdro': [0.01, 0.1]}
    inputs = [x.strip() for x in a.inputs.split(',') if x.strip()]; methods = [x.strip() for x in a.methods.split(',') if x.strip()]
    for inp in inputs:
        D = data(inp); ep = a.epochs if inp == 'env' else a.epochs_order
        for method in methods:
            name = f'{inp}_{method}'
            for proto, fold, trb, teb, c0 in protocols(True):
                p = f'{CK}/p2_{name}_{proto}_{fold}.npz'
                if os.path.exists(p): r = dict(np.load(p, allow_pickle=True))
                else:
                    tr, te = np.isin(D['B'], trb), np.isin(D['B'], teb)
                    if c0: tr &= D['c'] == COND0; te &= D['c'] == COND0
                    hp_best, val = HP[method][0], np.nan
                    if len(HP[method]) > 1:
                        rng = np.random.default_rng(fold); vb = [rng.choice(sorted(b for b in trb if lab[b] == c)) for c in range(3)
                                                                 if sum(lab[b] == c for b in trb) >= 2]
                        itr, iva = tr & ~np.isin(D['B'], vb), tr & np.isin(D['B'], vb)
                        accs = []
                        for hp in HP[method]:
                            pv = fit_predict(inp, method, hp, D['X'][itr], D['y'][itr], D['B'][itr], D['X'][iva], ep)
                            Pv, yv, _ = to_rec(D, iva, pv); accs.append(float((Pv.argmax(1) == yv).mean()))
                        hp_best, val = HP[method][int(np.argmax(accs))], max(accs)
                    pw = fit_predict(inp, method, hp_best, D['X'][tr], D['y'][tr], D['B'][tr], D['X'][te], ep)
                    P, yr, br = to_rec(D, te, pw)
                    r = {'P': P.astype(np.float32), 'y': yr, 'b': br, 'hp': np.array(-1.0 if hp_best is None else hp_best), 'val': np.array(val)}
                    np.savez_compressed(p, **r)
                RES.setdefault(name, {}).setdefault(proto, []).append(dict(summary(r['P'], r['y'], r['b']), fold=fold, hp=float(r['hp']), val=float(r['val'])))
                if proto != 'LOBO' or fold == len(single) - 1:
                    rr = RES[name][proto]; ys = sum(x['n'] for x in rr)
                    print(f'  {name:10s} {proto:6s}: {100 * sum(x["acc"] * x["n"] for x in rr) / ys:5.1f}  ({time.time() - t0:.0f} s)', flush=True)
            json.dump(RES, open(RES_PATH, 'w'), indent=1)

# ===================================================================================== table
print('\nRecording accuracy (%), pooled over folds; false alarms on healthy recordings in brackets')
for name, row in RES.items():
    for m, protos in (row.items() if a.part == 1 else [('', row)]):
        line = f'{name:10s} {m:4s}'
        for proto in ('L8_c0', 'L8_all', 'L10', 'A2R', 'LOBO'):
            if proto not in protos: continue
            rr = protos[proto]; n = sum(x['n'] for x in rr); acc = sum(x['acc'] * x['n'] for x in rr) / n
            fa = [x['fa'] for x in rr if x['fa'] is not None]
            line += f' | {proto} {100 * acc:5.1f} ({100 * np.mean(fa):4.1f})'
        print(line)
print(f'written {RES_PATH}  ({time.time() - t0:.0f} s)')
