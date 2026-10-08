"""Phase 26b (v1.0): statistics of the balanced decision rule (ENSX_PC) from the phase 26 checkpoints. No training; CPU, < 1 min.

Phase 26 showed that, of the pre-specified changes, only the prior correction of the averaged branches (ENSX_PC) improved
ReliSense consistently; the learned weight, the harmonic-agreement features and stacking did not. Phase 26 printed
the paired tests and intervals for the main candidate WAVH_PC only. This script adds, from the stored posteriors:
  LOBO   ENSX_PC against RF, LRX and ENSX: per-bearing Wilcoxon signed-rank test (29 bearings), bearings better / worse;
         95% bearing-bootstrap interval (2000 resamples within class, seed 0); class recalls; accuracy of every bearing
  all    class recalls and bearings correct (highest mean probability) of ENSX and ENSX_PC per Paderborn protocol
  trade  the decision rule p ~ P / prior**alpha for alpha = 0 (ReliSense v1), 0.25, 0.5, 0.75, 1 (ENSX_PC) applied to both
         branches before averaging: LOBO accuracy, false alarms and missed faults (operating points without retraining)
  ext    ENSX_PC with training-only abstention at kappa 0.7 on CWRU, HUST and Ottawa (from the stored inner posteriors)
Output: printed, and <root>/phase26b_results.json
Colab:  !python /content/drive/MyDrive/ReliSense_study/phase26b_balanced_summary.py --root /content/drive/MyDrive/ReliSense_study
"""
import argparse, glob, json, os, re
import numpy as np
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser(); ap.add_argument('--root', default='/content/drive/MyDrive/ReliSense_study')
a = ap.parse_args(); CK = a.root + '/checkpoints/phase26'
def L(p): return dict(np.load(p, allow_pickle=True))
OUT = {}

# ------------------------------------------------------------------ Paderborn
files = {re.match(r'pb_(.+)_(.+)\.npz$', os.path.basename(f)).groups(): f for f in glob.glob(CK + '/pb_*.npz')}
lobo = sorted(k for p, k in files if p == 'LOBO'); assert len(lobo) == 29, f'{len(lobo)} LOBO checkpoints found'
R = {b: L(files[('LOBO', b)]) for b in lobo}
y = np.concatenate([R[b]['y'] for b in lobo]); B = np.concatenate([R[b]['b'] for b in lobo])
P = {m: np.concatenate([R[b]['out_' + m] for b in lobo]) for m in ('LRX', 'RF', 'ENSX', 'ENSX_PC')}
lab = {b: int(R[b]['y'][0]) for b in lobo}
acc_b = {m: {b: float((P[m][B == b].argmax(1) == lab[b]).mean()) for b in lobo} for m in P}
OUT['lobo_per_bearing'] = acc_b
print('LOBO, ENSX_PC (balanced decision) against:')
for ref in ('RF', 'LRX', 'ENSX'):
    d = np.array([acc_b['ENSX_PC'][b] - acc_b[ref][b] for b in lobo]); p = float(wilcoxon(d).pvalue)
    OUT.setdefault('wilcoxon', {})[ref] = {'better': int((d > 0.005).sum()), 'worse': int((d < -0.005).sum()), 'p': p}
    print(f'  {ref:5}: better on {(d > 0.005).sum()}, worse on {(d < -0.005).sum()} of 29 bearings, Wilcoxon p = {p:.3f}')
rng = np.random.default_rng(0); groups = {c: [b for b in lobo if lab[b] == c] for c in range(3)}; idx = {b: np.where(B == b)[0] for b in lobo}
for m in ('ENSX', 'ENSX_PC'):
    ok = P[m].argmax(1) == y; bs = []
    for _ in range(2000):
        sel = np.concatenate([idx[b] for c, g in groups.items() for b in rng.choice(g, len(g), replace=True)]); bs.append(ok[sel].mean())
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]; OUT.setdefault('ci95', {})[m] = ci
    rec = [float((P[m].argmax(1)[y == c] == c).mean()) for c in range(3)]; OUT.setdefault('recall', {})[m] = rec
    print(f'  {m:7} accuracy {100 * ok.mean():.1f} (95% CI {100 * ci[0]:.1f}-{100 * ci[1]:.1f}); recall H / OR / IR {" / ".join(f"{100 * r:.1f}" for r in rec)}')
print('  accuracy of every bearing (ENSX -> ENSX_PC):')
print('   ' + '  '.join(f'{b} {100 * acc_b["ENSX"][b]:.0f}->{100 * acc_b["ENSX_PC"][b]:.0f}' for b in lobo))

print('\nAll Paderborn protocols: recall H / OR / IR (%) and bearings correct, ENSX -> ENSX_PC')
for proto in ('L8_c0', 'L8_all', 'L10', 'A2R', 'LOBO'):
    ks = sorted(k for p, k in files if p == proto); rs = [L(files[(proto, k)]) for k in ks]
    line = f'  {proto:7}'
    for m in ('ENSX', 'ENSX_PC'):
        yy = np.concatenate([r['y'] for r in rs]); pp = np.concatenate([r['out_' + m] for r in rs]).argmax(1)
        rec = [float((pp[yy == c] == c).mean()) if (yy == c).any() else float('nan') for c in range(3)]
        nb = sum(int(r['out_' + m][r['b'] == b].mean(0).argmax() == r['y'][r['b'] == b][0]) for r in rs for b in sorted(set(r['b'])))
        tb = sum(len(set(r['b'])) for r in rs)
        OUT.setdefault('protocols', {}).setdefault(proto, {})[m] = {'recall': rec, 'bearings_correct': nb, 'bearings': tb}
        line += f' | {m}: {" / ".join(f"{100 * x:.1f}" for x in rec)}; bearings {nb}/{tb}'
    print(line)

print('\nOperating points without retraining (LOBO): p ~ P / prior**alpha in both branches, then averaged')
yall = y
for al in (0.0, 0.25, 0.5, 0.75, 1.0):
    pr = []
    for b in lobo:                                  # prior = class frequencies of the 28 training bearings' recordings
        ytr = yall[B != b]; prior = np.array([(ytr == c).mean() for c in range(3)])
        q = lambda X: (X / prior ** al) / (X / prior ** al).sum(1, keepdims=True)
        pr.append(((q(R[b]['out_LRX']) + q(R[b]['out_RF'])) / 2).argmax(1))
    pr = np.concatenate(pr); h = yall == 0
    r = {'acc': float((pr == yall).mean()), 'false_alarm': float((pr[h] != 0).mean()), 'missed_fault': float((pr[~h] == 0).mean())}
    OUT.setdefault('tradeoff', {})[str(al)] = r
    print(f'  alpha {al:4.2f}: accuracy {100 * r["acc"]:.1f}, false alarms {100 * r["false_alarm"]:.1f}, missed faults {100 * r["missed_fault"]:.1f}')

# ------------------------------------------------------------------ external datasets
print('\nExternal datasets, ENSX_PC with training-only abstention at kappa 0.7: coverage / selective accuracy / false alarms / missed (%)')
for nm in ('cwru', 'hust', 'ottawa'):
    fs = sorted(glob.glob(f'{CK}/{nm}_*.npz'))
    by = {}
    for f in fs: by.setdefault(os.path.basename(f).split('_')[1], []).append(L(f))
    for proto, rs in by.items():
        yy = np.concatenate([r['y'] for r in rs]); PP = np.concatenate([r['out_ENSX_PC'] for r in rs])
        acc = np.concatenate([r['out_ENSX_PC'].max(1) >= (np.quantile(np.nanmax(r['in_ENSX_PC'], 1)[np.isfinite(np.nanmax(r['in_ENSX_PC'], 1))], 0.3)
                                                          if r['in_ENSX_PC'] is not None and r['in_ENSX_PC'].dtype != object else -1) for r in rs])
        p = PP.argmax(1); ok = p == yy; h = yy == 0
        s = {'coverage': float(acc.mean()), 'sel_acc': float(ok[acc].mean()), 'false_alarm': float((acc & h & (p != 0)).sum() / max(h.sum(), 1)),
             'missed_fault': float((acc & ~h & (p == 0)).sum() / max((~h).sum(), 1)), 'acc_all': float(ok.mean())}
        OUT.setdefault('external', {}).setdefault(nm, {})[proto] = s
        print(f'  {nm:6} {proto:5}: all {100 * s["acc_all"]:.1f} | {100 * s["coverage"]:.1f} / {100 * s["sel_acc"]:.1f} / {100 * s["false_alarm"]:.1f} / {100 * s["missed_fault"]:.1f}')
json.dump(OUT, open(a.root + '/phase26b_results.json', 'w'), indent=1)
print('\nwritten', a.root + '/phase26b_results.json')
