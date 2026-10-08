"""Phase 3 analysis (v1.0) on pb32_physics.npz. Primary model fixed in advance: lr_fixed.

A  Lessmeier protocols L8 and L10 restricted to operating condition 0 (N15_M07_F10), as in the paper.
B  Confusion matrices (rows = true, columns = predicted; H/OR/IR) of lr_fixed for L8, A2R and LOBO.
C  LOBO accuracy of lr_fixed by origin and damage level (labels_32.csv, Lessmeier 2016 Tables 4-5).
D  Selective prediction (LOBO, lr_fixed): accuracy on the most confident recordings at fixed coverages.
   The confidence threshold of each fold is set on the TRAINING bearings only (inner leave-one-bearing-out),
   so no test information is used. Also reported: the risk-coverage curve of the pooled test predictions.
"""
from __future__ import annotations
import argparse, csv, itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from phase2_physics import L8_TRAIN, L8_TEST, L10, splits, rule, VARIANTS

COND0 = 'N15_M07_F10'
NAMES = ['H', 'OR', 'IR']


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))


def acc_line(d, trm, tem, name):
    X, y = d['feat_fixed'], d['label']
    p_lr = lr().fit(X[trm], y[trm]).predict(X[tem])
    Xa = np.concatenate([d[f'feat_{v}'] for v in VARIANTS] +
                        [np.log(d['stat_kurtosis'])[:, None], np.log(d['stat_crest'])[:, None]], 1)
    p_rf = RandomForestClassifier(500, random_state=0, n_jobs=-1).fit(Xa[trm], y[trm]).predict(Xa[tem])
    p_ru = rule(d, trm, tem)
    return {m: float((p == y[tem]).mean()) for m, p in (('rule', p_ru), ('lr_fixed', p_lr), ('rf_all', p_rf))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--physics', default='pb32_physics.npz')
    ap.add_argument('--labels', default='labels_32.csv')
    a = ap.parse_args()
    d = dict(np.load(a.physics)); B = d['bearing']; y = d['label']
    meta = {r['bearing']: r for r in csv.DictReader(open(a.labels))}

    print('A. Lessmeier protocols on condition 0 only (N15_M07_F10), recording accuracy')
    c0 = d['condition'] == COND0
    r = acc_line(d, np.isin(B, L8_TRAIN) & c0, np.isin(B, L8_TEST) & c0, 'L8')
    print(f"   L8  : rule {r['rule']:.3f}  lr_fixed {r['lr_fixed']:.3f} *primary  rf_all {r['rf_all']:.3f}"
          f"   (n test = {int((np.isin(B, L8_TEST) & c0).sum())}; published 0.623-0.659)")
    res = []
    for tr in itertools.combinations(range(5), 3):
        trb = [c[i] for c in L10 for i in tr]; teb = [c[i] for c in L10 for i in range(5) if i not in tr]
        res.append(acc_line(d, np.isin(B, trb) & c0, np.isin(B, teb) & c0, 'L10'))
    m = {k: (np.mean([x[k] for x in res]), np.std([x[k] for x in res])) for k in res[0]}
    print(f"   L10 : rule {m['rule'][0]:.3f}+-{m['rule'][1]:.3f}  lr_fixed {m['lr_fixed'][0]:.3f}+-{m['lr_fixed'][1]:.3f}"
          f" *primary  rf_all {m['rf_all'][0]:.3f}+-{m['rf_all'][1]:.3f}   (published up to 0.985)\n")

    print('B. Confusion matrices of lr_fixed (rows true H/OR/IR, columns predicted H/OR/IR), recordings')
    X = d['feat_fixed']; pooled = {}; probs = np.full((len(y), 3), np.nan); lobo_thr = {}
    for proto, fold, trb, teb in splits(d):
        if proto == 'L10':
            continue
        trm, tem = np.isin(B, trb), np.isin(B, teb)
        clf = lr().fit(X[trm], y[trm]); p = clf.predict(X[tem])
        cm = pooled.setdefault(proto, np.zeros((3, 3), int))
        for t, q in zip(y[tem], p):
            cm[t, q] += 1
        if proto == 'LOBO':
            probs[tem] = clf.predict_proba(X[tem])
    for proto, cm in pooled.items():
        print(f'   {proto}:')
        for i in range(3):
            print(f'     {NAMES[i]:>2} ' + ' '.join(f'{v:5d}' for v in cm[i]))
    single = [b for b in dict.fromkeys(B) if int(meta[b]['label']) < 3]
    print('   LOBO predictions of the healthy bearings (H/OR/IR):')
    pred = probs.argmax(1)
    for b in [x for x in single if int(meta[x]['label']) == 0]:
        mb = B == b; print(f'     {b}: ' + ' '.join(f'{NAMES[k]} {int((pred[mb] == k).sum())}' for k in range(3)))

    print('\nC. LOBO accuracy of lr_fixed by origin and damage level (recordings, bearings)')
    groups = {}
    for b in single:
        o = meta[b]['origin']; lev = meta[b]['extent']
        key = 'healthy' if o == 'healthy' else f'{o} level {lev}'
        groups.setdefault(key, []).append(b)
    for key in sorted(groups):
        mb = np.isin(B, groups[key]); acc = float((pred[mb] == y[mb]).mean())
        print(f'   {key:20s} acc {acc:.3f}   bearings: {", ".join(groups[key])}')

    print('\nD. Selective prediction, LOBO lr_fixed (threshold from training bearings only)')
    keep = np.isin(B, single); conf = probs.max(1)
    inner_conf = {}
    for b in single:  # inner leave-one-bearing-out confidences on the training bearings of each fold
        trb = [x for x in single if x != b]; inner = []
        for v in trb:
            tri, tei = np.isin(B, [x for x in trb if x != v]), B == v
            inner.append(lr().fit(X[tri], y[tri]).predict_proba(X[tei]).max(1))
        inner_conf[b] = np.concatenate(inner)
    for target in (0.9, 0.8, 0.7):
        accept = np.zeros(len(y), bool)
        for b in single:
            thr = np.quantile(inner_conf[b], 1 - target)
            mb = B == b; accept[mb] = conf[mb] >= thr
        cov = accept[keep].mean(); acc = (pred[accept] == y[accept]).mean() if accept.any() else float('nan')
        print(f'   target coverage {target:.0%}: realised coverage {cov:.3f}, accuracy on accepted {acc:.3f}'
              f'  (all recordings: {(pred[keep] == y[keep]).mean():.3f})')
    order = np.argsort(-conf[keep]); correct = (pred[keep] == y[keep])[order]
    print('   risk-coverage curve (pooled test confidences, descriptive only):')
    for c in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = max(1, int(round(c * len(order)))); print(f'     coverage {c:.0%}: accuracy {correct[:k].mean():.3f}')


if __name__ == '__main__':
    main()
