"""Phase 3c (v1.0): uncertainty and bearing-level results for the primary model (lr_fixed; fixed in advance).

1. 95 % bootstrap confidence intervals over TEST BEARINGS (stratified by class, 2000 resamples) for
   L8 (condition 0 and all conditions), A2R and LOBO recording accuracy.
2. Bearing-level diagnosis: each test bearing gets the class with the highest mean probability over its
   recordings; reported as correct bearings / test bearings.
3. Abstention per bearing (LOBO): share of recordings accepted at a 70 % target coverage (threshold set on the
   training bearings only, inner leave-one-bearing-out) and accuracy on the accepted ones.
"""
from __future__ import annotations
import argparse, csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from phase2_physics import L8_TRAIN, L8_TEST

COND0 = 'N15_M07_F10'


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))


def boot_ci(B, y, correct, teb, reps=2000, seed=0):
    rng = np.random.default_rng(seed); by_class = {}
    for b in teb:
        by_class.setdefault(int(y[B == b][0]), []).append(b)
    per = {b: (correct[B == b].sum(), (B == b).sum()) for b in teb}
    stats = []
    for _ in range(reps):
        c = n = 0
        for bs in by_class.values():
            for b in rng.choice(bs, len(bs), replace=True):
                c += per[b][0]; n += per[b][1]
        stats.append(c / n)
    return np.percentile(stats, [2.5, 97.5])


def run(d, trm, tem, X, y, B):
    clf = lr().fit(X[trm], y[trm]); p = np.full((len(y), 3), np.nan); p[tem] = clf.predict_proba(X[tem])
    return p


def bearing_vote(p, y, B, teb):
    ok = [int(np.nanmean(p[B == b], 0).argmax() == y[B == b][0]) for b in teb]
    return sum(ok), len(ok), [b for b, o in zip(teb, ok) if not o]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--physics', default='pb32_physics.npz')
    ap.add_argument('--labels', default='labels_32.csv')
    a = ap.parse_args()
    d = dict(np.load(a.physics)); B, y, X = d['bearing'], d['label'], d['feat_fixed']
    meta = {r['bearing']: r for r in csv.DictReader(open(a.labels))}
    single = [b for b in dict.fromkeys(B) if int(meta[b]['label']) < 3]
    healthy = sorted(b for b in single if meta[b]['origin'] == 'healthy')
    a2r_tr = healthy[:3] + [b for b in single if meta[b]['origin'] == 'artificial']
    a2r_te = healthy[3:] + [b for b in single if meta[b]['origin'] == 'real']
    c0 = d['condition'] == COND0; allc = np.ones(len(y), bool)

    print('1-2. Primary model lr_fixed: recording accuracy [95 % bootstrap CI over bearings]; bearing-level vote')
    for name, trb, teb, cm in (('L8 condition 0', L8_TRAIN, L8_TEST, c0), ('L8 all conditions', L8_TRAIN, L8_TEST, allc),
                               ('A2R all conditions', a2r_tr, a2r_te, allc)):
        trm, tem = np.isin(B, trb) & cm, np.isin(B, teb) & cm
        p = run(d, trm, tem, X, y, B); correct = np.zeros(len(y), bool); correct[tem] = p[tem].argmax(1) == y[tem]
        lo, hi = boot_ci(B[tem], y[tem], correct[tem], teb)
        k, n, wrong = bearing_vote(p, y, B, teb)
        print(f'   {name:20s} acc {correct[tem].mean():.3f} [{lo:.3f}, {hi:.3f}]   bearings correct {k}/{n}'
              f'   wrong: {", ".join(wrong) or "none"}')

    probs = np.full((len(y), 3), np.nan)
    for b in single:
        trm, tem = np.isin(B, [x for x in single if x != b]), B == b
        probs[tem] = lr().fit(X[trm], y[trm]).predict_proba(X[tem])
    keep = np.isin(B, single); pred = np.where(keep, np.nan_to_num(probs).argmax(1), -1); correct = pred == y
    lo, hi = boot_ci(B[keep], y[keep], correct[keep], single)
    k, n, wrong = bearing_vote(probs, y, B, single)
    print(f'   {"LOBO (29 bearings)":20s} acc {correct[keep].mean():.3f} [{lo:.3f}, {hi:.3f}]   bearings correct {k}/{n}'
          f'   wrong: {", ".join(wrong) or "none"}')

    print('\n3. LOBO abstention per bearing at 70 % target coverage (threshold from training bearings only)')
    conf = np.nan_to_num(probs).max(1); accept = np.zeros(len(y), bool)
    for b in single:
        trb = [x for x in single if x != b]; inner = []
        for v in trb:
            tri, tei = np.isin(B, [x for x in trb if x != v]), B == v
            inner.append(lr().fit(X[tri], y[tri]).predict_proba(X[tei]).max(1))
        thr = np.quantile(np.concatenate(inner), 0.3); accept[B == b] = conf[B == b] >= thr
    print(f"   {'bearing':8}{'origin':12}{'level':>6}{'accepted':>10}{'acc(all)':>10}{'acc(accepted)':>15}")
    for b in single:
        mb = B == b; acc_all = correct[mb].mean(); ma = mb & accept
        acc_acc = correct[ma].mean() if ma.any() else float('nan')
        print(f"   {b:8}{meta[b]['origin']:12}{meta[b]['extent']:>6}{accept[mb].mean():10.2f}{acc_all:10.2f}{acc_acc:15.2f}")
    print(f'   overall: coverage {accept[keep].mean():.3f}, accuracy on accepted {correct[accept & keep].mean():.3f}')


if __name__ == '__main__':
    main()
