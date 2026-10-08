"""Phase 17 (v1.0): two follow-up experiments on the Paderborn physics file (CPU, about 45-60 minutes; checkpoints).

Experiment 1  ReliSense (LR on the 15 kinematic features) against and combined with the random forest of phase 2
              (500 trees on all 60 envelope features + log kurtosis + log crest factor; the 60 features already
              contain the 15 kinematic ones). Ensemble = mean of the two class-probability vectors.
              (a) recording accuracy and healthy false-alarm rate on L8 (cond. 0 and all), L10, A2R and LOBO;
              (b) training-only abstention under LOBO for all three models (inner LOBO over the 28 training bearings).
Experiment 2  Class-wise thresholds: tau_{b,k} = (1 - kappa) quantile of the inner-LOBO confidences of recordings
              PREDICTED as class k (fallback to the pooled threshold if fewer than 20 such recordings), against
              the pooled threshold tau_b of the paper. Reported: coverage, selective accuracy, false alarms
              (healthy recordings that are accepted and diagnosed as damaged, per all healthy recordings) and
              missed faults (damaged recordings accepted as healthy, per all damaged recordings).
Nothing is fitted on the test bearing; settings are those of phases 2 and 13 (LR C = 0.3; RF 500 trees, seed 0).
Every outer fold is saved in <root>/checkpoints/phase17/ and reused on a second run (--fresh recomputes).
Output: <root>/phase17_results.json
Colab:  !python /content/drive/MyDrive/ReliSense_study/phase17_improve.py --root /content/drive/MyDrive/ReliSense_study
"""
import argparse, itertools, json, os, sys, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser(); ap.add_argument('--root', default='/content/drive/MyDrive/ReliSense_study'); ap.add_argument('--fresh', action='store_true')
ap.add_argument('--trees', type=int, default=500); ap.add_argument('--physics', default=None)
a = ap.parse_args(); ROOT = a.root; CK = ROOT + '/checkpoints/phase17'; os.makedirs(CK, exist_ok=True); t0 = time.time()
d = dict(np.load(a.physics or ROOT + '/pb32/pb32_physics.npz', allow_pickle=True))
B, y, cond = d['bearing'], d['label'].astype(int), d['condition']
X_lr = d['feat_fixed'].astype(float)
X_rf = np.concatenate([d[f'feat_{v}'] for v in ('fixed', 'sk', 'cpw', 'cpw_sk')] + [np.log(d['stat_kurtosis'])[:, None], np.log(d['stat_crest'])[:, None]], 1).astype(float)
X_lr, X_rf = np.nan_to_num(X_lr), np.nan_to_num(X_rf)
COND0 = 'N15_M07_F10'; KAP = (0.9, 0.8, 0.7)
L8_TRAIN = ['K002', 'KA01', 'KA05', 'KA07', 'KI01', 'KI05', 'KI07']
L8_TEST = ['K001', 'KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']
L10 = [['K001', 'K002', 'K003', 'K004', 'K005'], ['KA04', 'KA15', 'KA16', 'KA22', 'KA30'], ['KI04', 'KI14', 'KI16', 'KI18', 'KI21']]
bearings = list(dict.fromkeys(B)); lab = {b: int(y[B == b][0]) for b in bearings}; org = {b: str(d['origin'][B == b][0]) for b in bearings}
single = [b for b in bearings if lab[b] < 3]


def proba(m, X):
    P = np.zeros((len(X), 3)); P[:, m.classes_] = m.predict_proba(X); return P


def fit_both(tr):
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)).fit(X_lr[tr], y[tr])
    rf = RandomForestClassifier(a.trees, random_state=0, n_jobs=-1).fit(X_rf[tr], y[tr])
    return lr, rf


def probs3(models, te):
    lr, rf = models; pl, pr = proba(lr, X_lr[te]), proba(rf, X_rf[te]); return {'LR': pl, 'RF': pr, 'ENS': (pl + pr) / 2}


def ck(name, fn):
    p = f'{CK}/{name}.npz'
    if os.path.exists(p) and not a.fresh: return dict(np.load(p))
    r = fn(); np.savez_compressed(p, **r); return r


# ------------------------------------------------------------------ Experiment 1(a): all protocols
def protocol_splits():
    yield 'L8_c0', 0, L8_TRAIN, L8_TEST, True
    yield 'L8_all', 0, L8_TRAIN, L8_TEST, False
    for k, tr in enumerate(itertools.combinations(range(5), 3)):
        yield 'L10', k, [c[i] for c in L10 for i in tr], [c[i] for c in L10 for i in range(5) if i not in tr], False
    healthy = sorted(b for b in bearings if lab[b] == 0); art = [b for b in bearings if org[b] == 'artificial']
    real = [b for b in bearings if org[b] == 'real' and lab[b] < 3]
    yield 'A2R', 0, healthy[:3] + art, healthy[3:] + real, False


res = {'protocols': {}}
for proto, k, trb, teb, c0 in protocol_splits():
    def run():
        tr, te = np.isin(B, trb), np.isin(B, teb)
        if c0: tr &= cond == COND0; te &= cond == COND0
        P = probs3(fit_both(tr), te); return {**P, 'y': y[te]}
    r = ck(f'{proto}_{k}', run)
    for m in ('LR', 'RF', 'ENS'):
        p = r[m].argmax(1); yt = r['y']; h = yt == 0
        res['protocols'].setdefault(proto, {}).setdefault(m, []).append({'acc': float((p == yt).mean()), 'fa': float((p[h] != 0).mean()) if h.any() else None})
print('Experiment 1(a) done', f'{time.time() - t0:.0f} s', flush=True)

# ------------------------------------------------------------------ LOBO with inner LOBO (Experiments 1(b) and 2)
OUT = {m: np.full((len(y), 3), np.nan) for m in ('LR', 'RF', 'ENS')}; INNER = {}
for ib, b in enumerate(single):
    def run():
        tr = [x for x in single if x != b]; te = B == b
        r = {f'outer_{m}': v for m, v in probs3(fit_both(np.isin(B, tr)), te).items()}
        pools = {m: [] for m in ('LR', 'RF', 'ENS')}
        for v in tr:
            P = probs3(fit_both(np.isin(B, [x for x in tr if x != v])), B == v)
            for m in pools: pools[m].append(P[m])
        for m in pools: r[f'inner_{m}'] = np.concatenate(pools[m])
        return r
    r = ck(f'lobo_{b}', run)
    for m in ('LR', 'RF', 'ENS'): OUT[m][B == b] = r[f'outer_{m}']; INNER[(b, m)] = r[f'inner_{m}']
    print(f'  LOBO {ib + 1}/{len(single)} {b}  ({time.time() - t0:.0f} s)', flush=True)
keep = np.isin(B, single); res['protocols']['LOBO'] = {}
for m in ('LR', 'RF', 'ENS'):
    p = OUT[m][keep].argmax(1); yt = y[keep]
    res['protocols']['LOBO'][m] = [{'acc': float((p == yt).mean()), 'fa': float((p[yt == 0] != 0).mean())}]


def selective(m, kappa, classwise):
    P = OUT[m][keep]; yt = y[keep]; Bk = B[keep]; conf = P.max(1); pred = P.argmax(1); acc = np.zeros(len(yt), bool)
    for b in single:
        I = INNER[(b, m)]; ic, ip = I.max(1), I.argmax(1); tg = np.quantile(ic, 1 - kappa); sel = Bk == b
        if classwise:
            tk = [np.quantile(ic[ip == k], 1 - kappa) if (ip == k).sum() >= 20 else tg for k in range(3)]
            acc[sel] = conf[sel] >= np.array(tk)[pred[sel]]
        else:
            acc[sel] = conf[sel] >= tg
    ok = pred == yt; h = yt == 0; dmg = ~h
    return {'coverage': float(acc.mean()), 'sel_acc': float(ok[acc].mean()), 'false_alarm': float((acc & h & (pred != 0)).sum() / h.sum()),
            'missed_fault': float((acc & dmg & (pred == 0)).sum() / dmg.sum()), 'healthy_coverage': float(acc[h].mean())}


res['abstention'] = {m: {'none': {'coverage': 1.0, 'sel_acc': res['protocols']['LOBO'][m][0]['acc'], 'false_alarm': res['protocols']['LOBO'][m][0]['fa']}} for m in ('LR', 'RF', 'ENS')}
for m in ('LR', 'RF', 'ENS'):
    for kappa in KAP:
        for cw in (False, True):
            res['abstention'][m][f'{"classwise" if cw else "pooled"}_{kappa}'] = selective(m, kappa, cw)
json.dump(res, open(ROOT + '/phase17_results.json', 'w'), indent=1)

print('\nExperiment 1(a): recording accuracy (%) / healthy false alarms (%) [L10: mean of 10 splits]')
print(f"{'protocol':8}" + ''.join(f'{m:>16}' for m in ('LR (ReliSense)', 'RF (phase 2)', 'Ensemble')))
for proto in ('L8_c0', 'L8_all', 'L10', 'A2R', 'LOBO'):
    row = f'{proto:8}'
    for m in ('LR', 'RF', 'ENS'):
        R = res['protocols'][proto][m]; ac = 100 * np.mean([x['acc'] for x in R]); fa = [x['fa'] for x in R if x['fa'] is not None]
        row += f'{ac:9.1f} / {100 * np.mean(fa):4.1f}' if fa else f'{ac:9.1f} /   - '
    print(row)
print('\nExperiments 1(b) and 2, LOBO: coverage / selective accuracy / false alarms / missed faults (%)')
for m in ('LR', 'RF', 'ENS'):
    print(f'  {m}: no abstention  acc {100 * res["abstention"][m]["none"]["sel_acc"]:.1f}, false alarms {100 * res["abstention"][m]["none"]["false_alarm"]:.1f}')
    for kappa in KAP:
        for cw in ('pooled', 'classwise'):
            r = res['abstention'][m][f'{cw}_{kappa}']
            print(f'     kappa {kappa}  {cw:9}  cov {100 * r["coverage"]:5.1f}  acc {100 * r["sel_acc"]:5.1f}  FA {100 * r["false_alarm"]:5.1f}  missed {100 * r["missed_fault"]:5.1f}  healthy cov {100 * r["healthy_coverage"]:5.1f}')
print(f'\nwritten {ROOT}/phase17_results.json  ({time.time() - t0:.0f} s)')
