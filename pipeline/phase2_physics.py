"""Phase 2 (v1.0): bearing-wise protocols on the 32-bearing physics features (pb32_physics.npz).

Every split is by bearing: no recording of a test bearing is ever used for training or scaling.
Unit of evaluation: one recording (4 s). Metrics: accuracy, balanced accuracy, macro F1, per-bearing accuracy.

Protocols
  L8   Lessmeier et al. 2016, Table 8: train K002 + KA01/05/07 + KI01/05/07, test K001 + KA04/15/16/22/30
       + KI14/16/17/18/21 (published, vibration features: 62-66 %, Table 9).
  L10  Lessmeier et al. 2016, Table 10: real damages only, K001-K005 / KA04,15,16,22,30 / KI04,14,16,18,21;
       10 combinations, 3 bearings per class train, 2 test (published, vibration: 98.5 %, Table 11).
  A2R  all artificial: train K001-K003 + 7 artificial OR + 5 artificial IR, test K004-K006 + all 11 real OR/IR.
  LOBO leave one bearing out over all 29 single-damage bearings (29 folds).
Models (recording level)
  rule        no fitted classifier: healthy if neither BPFO_h1 nor BPFI_h1 exceeds the 95th percentile of the
              training healthy bearings, else the larger excess (BPFO -> outer, BPFI -> inner)
  lr_<v>      standardise + logistic regression (C=0.3) on the 15 features of variant v
  lr_all      the same on all 60 features
  rf_all      random forest (500 trees) on all 60 features + kurtosis and crest factor
The primary model is fixed in advance: lr_fixed (as in study v1.4). All other rows are reported, none is
selected on test results.
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VARIANTS = ('fixed', 'sk', 'cpw', 'cpw_sk')
L8_TRAIN = ['K002', 'KA01', 'KA05', 'KA07', 'KI01', 'KI05', 'KI07']
L8_TEST = ['K001', 'KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']
L10 = [['K001', 'K002', 'K003', 'K004', 'K005'], ['KA04', 'KA15', 'KA16', 'KA22', 'KA30'],
       ['KI04', 'KI14', 'KI16', 'KI18', 'KI21']]


def splits(d):
    bearings = list(dict.fromkeys(d['bearing'])); lab = {b: int(d['label'][d['bearing'] == b][0]) for b in bearings}
    org = {b: str(d['origin'][d['bearing'] == b][0]) for b in bearings}
    yield 'L8', 0, L8_TRAIN, L8_TEST
    for k, tr in enumerate(itertools.combinations(range(5), 3)):
        yield 'L10', k, [c[i] for c in L10 for i in tr], [c[i] for c in L10 for i in range(5) if i not in tr]
    healthy = sorted(b for b in bearings if lab[b] == 0)
    art = [b for b in bearings if org[b] == 'artificial']; real = [b for b in bearings if org[b] == 'real' and lab[b] < 3]
    yield 'A2R', 0, healthy[:3] + art, healthy[3:] + real
    single = [b for b in bearings if lab[b] < 3]
    for k, b in enumerate(single):
        yield 'LOBO', k, [x for x in single if x != b], [b]


def features(d, model):
    if model.startswith('lr_') and model != 'lr_all':
        return d[f'feat_{model[3:]}']
    X = np.concatenate([d[f'feat_{v}'] for v in VARIANTS], 1)
    if model == 'rf_all':
        X = np.concatenate([X, np.log(d['stat_kurtosis'])[:, None], np.log(d['stat_crest'])[:, None]], 1)
    return X


def rule(d, tr, te):
    names = list(d['feature_names']); F = d['feat_fixed']
    o, i = F[:, names.index('BPFO_h1')], F[:, names.index('BPFI_h1')]
    h = tr & (d['label'] == 0)
    to, ti = np.percentile(o[h], 95), np.percentile(i[h], 95)
    eo, ei = o[te] - to, i[te] - ti
    return np.where((eo <= 0) & (ei <= 0), 0, np.where(eo >= ei, 1, 2))


def fit_predict(d, model, tr, te, seed):
    if model == 'rule':
        return rule(d, tr, te)
    X, y = features(d, model), d['label']
    clf = (RandomForestClassifier(500, random_state=seed, n_jobs=-1) if model == 'rf_all'
           else make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)))
    clf.fit(X[tr], y[tr])
    return clf.predict(X[te])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--physics', default='pb32_physics.npz')
    ap.add_argument('--out', default='phase2_results')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    d = dict(np.load(args.physics)); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    models = ['rule'] + [f'lr_{v}' for v in VARIANTS] + ['lr_all', 'rf_all']
    rows, per_bearing = [], {}
    for proto, fold, trb, teb in splits(d):
        tr, te = np.isin(d['bearing'], trb), np.isin(d['bearing'], teb)
        assert not set(trb) & set(teb) and tr.any() and te.any(), (proto, fold)
        y = d['label'][te]
        for m in models:
            p = fit_predict(d, m, tr, te, args.seed)
            rows.append({'protocol': proto, 'fold': fold, 'model': m, 'n_test': int(te.sum()),
                         'correct': int((p == y).sum()), 'acc': accuracy_score(y, p),
                         'bacc': balanced_accuracy_score(y, p) if len(set(y)) > 1 else float('nan'),
                         'f1': f1_score(y, p, average='macro') if len(set(y)) > 1 else float('nan')})
            for b in teb:
                mb = d['bearing'][te] == b
                per_bearing.setdefault((proto, m, b), []).append(float((p[mb] == y[mb]).mean()))
    (out / 'phase2_folds.json').write_text(json.dumps(rows, indent=1))
    pb = [{'protocol': k[0], 'model': k[1], 'bearing': k[2], 'acc_mean': float(np.mean(v)), 'n_folds': len(v)}
          for k, v in per_bearing.items()]
    (out / 'phase2_per_bearing.json').write_text(json.dumps(pb, indent=1))
    print('Recording-level results (mean over folds; LOBO = pooled over the 29 held-out bearings)\n')
    print(f"{'protocol':9}{'model':11}{'acc':>8}{'sd':>7}{'bacc':>8}{'f1':>8}{'folds':>7}")
    for proto in ('L8', 'L10', 'A2R', 'LOBO'):
        for m in models:
            r = [x for x in rows if x['protocol'] == proto and x['model'] == m]
            if proto == 'LOBO':
                acc = sum(x['correct'] for x in r) / sum(x['n_test'] for x in r); sd = np.std([x['acc'] for x in r])
                ba = f1 = float('nan')
            else:
                acc = np.mean([x['acc'] for x in r]); sd = np.std([x['acc'] for x in r])
                ba = np.mean([x['bacc'] for x in r]); f1 = np.mean([x['f1'] for x in r])
            star = ' *primary' if m == 'lr_fixed' else ''
            print(f'{proto:9}{m:11}{acc:8.3f}{sd:7.3f}{ba:8.3f}{f1:8.3f}{len(r):7d}{star}')
        print()
    print('Published (Lessmeier et al. 2016, vibration): L8 62.3-65.9 % (Table 9); L10 up to 98.5 % (Table 11).')
    print('\nPer-bearing accuracy, primary model lr_fixed:')
    for proto in ('L8', 'A2R', 'LOBO'):
        s = [f"{x['bearing']} {x['acc_mean']:.2f}" for x in pb if x['protocol'] == proto and x['model'] == 'lr_fixed']
        print(f'  {proto}: ' + ', '.join(s))


if __name__ == '__main__':
    main()
