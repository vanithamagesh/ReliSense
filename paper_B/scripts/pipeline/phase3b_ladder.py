"""Phase 3b (v1.0): a pre-declared ladder of signal-processing features and classifiers, with honest
(nested) method selection for the Lessmeier real-damage protocol L10.

Feature groups (one row per recording; computed from pb32_physics.npz and the 8 kHz window files):
  P     15 fault-frequency peak features (fixed band; the primary model of Phase 2)
  P2    extended physics from the order spectrum: harmonics 1-5 of BPFO/BPFI/BSF/FTF (up to 11.8 orders),
        shaft harmonics 1-3 and BPFI sidebands at +-1 and +-2 shaft orders
  T     vibration time statistics (log RMS, log peak-to-peak, kurtosis, skewness, crest, shape, impulse,
        clearance factors) and envelope statistics (log RMS, kurtosis, crest), averaged over windows
  S     32 relative log band energies of the vibration spectrum (0-4 kHz), averaged over windows
  C     motor-current statistics (log RMS, kurtosis, crest of both phases)
  Groups tested: P, P2, P+P2, T, S, T+S, T+S+C (generic, close to Lessmeier et al. 2016), P+P2+T+S, all.
Models: lr (logistic regression), svm (RBF), rf (random forest, 300 trees), hgb (gradient boosting).
Condition normalisation (cn): optional z-scoring of every feature within each operating condition, using
training-set statistics only.

Selection for L10 is nested: in each of the 10 outer splits the method is chosen on the 9 training bearings
only (3 inner folds, each holding out one training bearing per class); the chosen method is then tested once.
Ties in inner accuracy go to the candidate listed first (simplest group, then lr, svm, rf, hgb).
The 'hindsight best' row (best mean test accuracy over all candidates) is reported for transparency only;
it is optimistic and must not be quoted as a result.
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from phase2_physics import L8_TRAIN, L8_TEST, L10
import physics64 as P

GROUPS = {'P': ['P'], 'P2': ['P2'], 'P+P2': ['P', 'P2'], 'T': ['T'], 'S': ['S'], 'T+S': ['T', 'S'],
          'T+S+C': ['T', 'S', 'C'], 'P+P2+T+S': ['P', 'P2', 'T', 'S'], 'all': ['P', 'P2', 'T', 'S', 'C']}
MODELS = ('lr', 'svm', 'rf', 'hgb')


def p2_features(order, grid):
    o = P.fault_orders(); row_idx = []
    def peak(c):
        w = max(0.02 * c, 0.04); return np.where(np.abs(grid - c) <= w)[0]
    for k in ('BPFO', 'BPFI', 'BSF', 'FTF'):
        for h in range(1, 6):
            if h * o[k] <= 11.8: row_idx.append(peak(h * o[k]))
    for h in (1, 2, 3): row_idx.append(peak(h))
    for s in (-2, -1, 1, 2): row_idx.append(peak(o['BPFI'] + s))
    return np.stack([order[:, ix].max(1) for ix in row_idx], 1)


def window_features(X, E):
    """X: [n, 3, L] raw 8 kHz windows (vibration, current 1, current 2); E: [n, L] envelope."""
    def stats(s):
        s = s - s.mean(1, keepdims=True); rms = s.std(1) + 1e-12; a = np.abs(s); pk = a.max(1)
        return [np.log(rms), np.log(s.max(1) - s.min(1) + 1e-12), (s ** 4).mean(1) / rms ** 4,
                (s ** 3).mean(1) / rms ** 3, pk / rms, rms / (a.mean(1) + 1e-12), pk / (a.mean(1) + 1e-12),
                pk / (np.sqrt(a).mean(1) ** 2 + 1e-12)]
    v = X[:, 0].astype(np.float64)
    T = stats(v); e = E.astype(np.float64); e = e - e.mean(1, keepdims=True); er = e.std(1) + 1e-12
    T += [np.log(er), (e ** 4).mean(1) / er ** 4, np.abs(e).max(1) / er]
    spec = np.abs(np.fft.rfft((v - v.mean(1, keepdims=True)) * np.hanning(v.shape[1]), axis=1)) ** 2
    bands = np.array_split(np.arange(1, spec.shape[1]), 32)
    S = np.stack([spec[:, b].sum(1) for b in bands], 1); S = np.log(S / S.sum(1, keepdims=True) + 1e-12)
    C = []
    for ch in (1, 2):
        c = X[:, ch].astype(np.float64); c = c - c.mean(1, keepdims=True); cr = c.std(1) + 1e-12
        C += [np.log(cr), (c ** 4).mean(1) / cr ** 4, np.abs(c).max(1) / cr]
    return np.stack(T, 1), S, np.stack(C, 1)


def build_features(phys, win, env, chunk=2000):
    rec_p = phys['recording']; pos = {r: i for i, r in enumerate(rec_p)}
    n = len(rec_p); sums = {k: None for k in 'TSC'}; cnt = np.zeros(n)
    wr = win['recording']; idx = np.array([pos[r] for r in wr])
    for a in range(0, len(wr), chunk):
        T, S, C = window_features(win['X'][a:a + chunk], env['X'][a:a + chunk, 0])
        for k, F in zip('TSC', (T, S, C)):
            if sums[k] is None: sums[k] = np.zeros((n, F.shape[1]))
            np.add.at(sums[k], idx[a:a + chunk], F)
        np.add.at(cnt, idx[a:a + chunk], 1)
    has = cnt > 0
    feats = {k: sums[k] / np.maximum(cnt, 1)[:, None] for k in 'TSC'}
    feats['P'] = phys['feat_fixed'].astype(np.float64)
    feats['P2'] = p2_features(phys['order_fixed'].astype(np.float64), phys['order_grid'])
    return feats, has


def make_model(m, seed=0):
    if m == 'lr': return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))
    if m == 'svm': return make_pipeline(StandardScaler(), SVC(C=1.0, gamma='scale'))
    if m == 'rf': return RandomForestClassifier(300, random_state=seed, n_jobs=-1)
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)


def fit_predict(cand, feats, y, cond, tr, te):
    g, m, cn = cand
    X = np.concatenate([feats[k] for k in GROUPS[g]], 1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if cn:
        X = X.copy()
        for c in np.unique(cond):
            mc = cond == c; ref = tr & mc
            if ref.sum() < 2: continue
            mu, sd = X[ref].mean(0), X[ref].std(0) + 1e-9
            X[mc] = (X[mc] - mu) / sd
    return make_model(m).fit(X[tr], y[tr]).predict(X[te])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--physics', default='pb32_physics.npz')
    ap.add_argument('--windows', default='pb32_4096.npz')
    ap.add_argument('--envelope', default='pb32_4096_envelope.npz')
    ap.add_argument('--cond0', action='store_true', help='use only operating condition N15_M07_F10')
    ap.add_argument('--out', default='phase3b_results.json')
    a = ap.parse_args()
    phys = dict(np.load(a.physics))
    win = np.load(a.windows); env = np.load(a.envelope)
    win = {'X': win['X'], 'recording': win['recording']}; env = {'X': env['X']}
    print('computing features ...', flush=True)
    feats, has = build_features(phys, win, env)
    del win, env
    B, y, cond = phys['bearing'], phys['label'], phys['condition']
    base = has & (y < 3)
    if a.cond0: base &= cond == 'N15_M07_F10'
    cands = [(g, m, cn) for g in GROUPS for m in MODELS for cn in (0, 1)]
    name = lambda c: f'{c[0]}|{c[1]}|{"cn" if c[2] else "raw"}'
    acc = lambda p, t: float((p == y[t]).mean())

    print(f'{len(cands)} candidates; protocols L8, L10 (10 splits), A2R', flush=True)
    l10 = []
    for tr_i in itertools.combinations(range(5), 3):
        trb = [[c[i] for i in tr_i] for c in L10]; teb = [c[i] for c in L10 for i in range(5) if i not in tr_i]
        l10.append((trb, teb))
    healthy = sorted(set(B[(y == 0) & base])); org = phys['origin']
    a2r_tr = healthy[:3] + sorted(set(B[(org == 'artificial') & base]))
    a2r_te = healthy[3:] + sorted(set(B[(org == 'real') & base]))
    table = {}
    for c in cands:
        r = {}
        tr, te = np.isin(B, L8_TRAIN) & base, np.isin(B, L8_TEST) & base
        r['L8'] = acc(fit_predict(c, feats, y, cond, tr, te), te)
        accs = []
        for trb, teb in l10:
            tr = np.isin(B, sum(trb, [])) & base; te = np.isin(B, teb) & base
            accs.append(acc(fit_predict(c, feats, y, cond, tr, te), te))
        r['L10'] = float(np.mean(accs)); r['L10_folds'] = accs
        tr, te = np.isin(B, a2r_tr) & base, np.isin(B, a2r_te) & base
        r['A2R'] = acc(fit_predict(c, feats, y, cond, tr, te), te)
        table[name(c)] = r
        print(f"  {name(c):24s} L8 {r['L8']:.3f}  L10 {r['L10']:.3f}  A2R {r['A2R']:.3f}", flush=True)

    print('\nNested selection for L10 (method chosen on training bearings only):', flush=True)
    nested = []
    for k, (trb, teb) in enumerate(l10):
        inner = {}
        for c in cands:
            s = []
            for j in range(3):
                itr = np.isin(B, [x for cl in trb for i, x in enumerate(cl) if i != j]) & base
                ite = np.isin(B, [cl[j] for cl in trb]) & base
                s.append(acc(fit_predict(c, feats, y, cond, itr, ite), ite))
            inner[name(c)] = float(np.mean(s))
        best = max(inner, key=inner.get)
        test_acc = table[best]['L10_folds'][k]
        nested.append({'fold': k, 'chosen': best, 'inner_acc': inner[best], 'test_acc': test_acc})
        print(f'  split {k}: chosen {best:24s} inner {inner[best]:.3f}  test {test_acc:.3f}', flush=True)
    nm = np.mean([x['test_acc'] for x in nested]); ns = np.std([x['test_acc'] for x in nested])

    order = sorted(table, key=lambda k: -table[k]['L10'])
    print('\nSUMMARY' + (' (condition 0 only)' if a.cond0 else ' (all conditions)'))
    print(f"  primary P|lr|raw           : L8 {table['P|lr|raw']['L8']:.3f}  L10 {table['P|lr|raw']['L10']:.3f}"
          f"  A2R {table['P|lr|raw']['A2R']:.3f}")
    print(f'  NESTED-SELECTED (honest)   : L10 {nm:.3f} +- {ns:.3f}')
    print(f"  hindsight best on L10      : {order[0]}  L10 {table[order[0]]['L10']:.3f}  "
          f"(L8 {table[order[0]]['L8']:.3f}, A2R {table[order[0]]['A2R']:.3f})  <- optimistic, do not quote")
    print('  top 8 candidates by L10 (with their L8 and A2R):')
    for k in order[:8]:
        print(f"    {k:24s} L10 {table[k]['L10']:.3f}  L8 {table[k]['L8']:.3f}  A2R {table[k]['A2R']:.3f}")
    print('  best candidate per protocol:')
    for p in ('L8', 'A2R'):
        k = max(table, key=lambda q: table[q][p]); print(f"    {p}: {k}  {table[k][p]:.3f}")
    Path(a.out).write_text(json.dumps({'table': table, 'nested_L10': nested, 'nested_mean': nm,
                                       'nested_sd': ns, 'cond0': a.cond0}, indent=1))


if __name__ == '__main__':
    main()
