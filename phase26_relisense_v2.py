"""Phase 26 (v1.0): ReliSense-2 - learned fusion weight, class-balanced decisions and harmonic-agreement features.
CPU runtime is enough (no GPU). About 20-40 minutes; every split is checkpointed, so a rerun resumes.

Fixed before any result is seen; every model below is reported, none is dropped afterwards.
Motivation (weak points of ReliSense v1 in Paper A v2.4):
  (a) the two branches are averaged with fixed equal weights, so ReliSense falls between them whenever one branch is
      clearly better (CWRU: kinematic branch 84.5 vs 83.2; Ottawa 1PROF: envelope branch 97.5 vs 95.9; LOBO: not
      significantly better than the envelope branch, p = 0.73);
  (b) healthy bearings are the minority class (6 of 29 bearings), and the unweighted fit favours "damaged"
      (LOBO false alarms 32.1 % before abstention; K004 / K005 diagnosed at 25 % / 31 %);
  (c) a single spurious line near a fault order raises the first-harmonic feature although a localized defect
      produces a comb of harmonics.
Changes (each is evaluated alone and combined):
  prior        class-balanced decisions by prior correction: every branch posterior is divided by the class frequencies
               of the training recordings and renormalized (Saerens, Latinne and Decaestecker 2002), i.e. uniform priors.
  harmonic     6 harmonic-agreement features: median and minimum of the three harmonic peak ratios of BPFO, BPFI and
               BSF (columns 0-8 of the 15 standard features). A defect comb raises all harmonics, a stray line one only
               (cf. harmonic product spectrum; weighted cyclic harmonic-to-noise ratio).
  weight       learned fusion with ONE parameter: P = w * P_kin + (1 - w) * P_env, w in {0, 0.1, ..., 1} minimizing the
               class-balanced log-loss of the out-of-fold posteriors of the inner loop over the training groups (ties:
               the w closest to 0.5). This inner loop is the one ReliSense already runs for its abstention threshold;
               the threshold uses cross-fitted posteriors (w re-chosen without each inner group). One parameter is used
               because estimated combination weights add variance when few groups are available ("forecast combination
               puzzle", Smith and Wallis 2009; Claeskens et al. 2016).
  stacking     (alternative fusion, reported) multinomial logistic regression (C = 1, class-balanced) on the branch
               log-probabilities (clipped at 1e-3), fitted on the same out-of-fold posteriors (Wolpert 1992).
Inner groups: Paderborn bearing; CWRU fault size (healthy: load); HUST bearing type; Ottawa speed profile. If a class
has fewer than two inner groups in the training set, the secondary key is used (Paderborn and CWRU / HUST: operating
condition; Ottawa: recording). If a class still has fewer than two, the weight is not identifiable and w = 0.5 (this
happens on Paderborn L8 at condition 0, whose only healthy training bearing is K002).
Models (branches): LRX   31 kinematic features, LR C = 0.3 (ReliSense v1 branch)
                   LRXH  31 + 6 harmonic-agreement features, LR C = 0.3
                   RF    envelope branch, 62 features, 500 trees, seed 0 (ReliSense v1 branch, unchanged)
                   OSP   (Paderborn only, exploratory) log order spectrum max-pooled to 0.1 order up to 11.8, LR C = 0.3;
                         a learned linear view of the same order axis
       (fusions):  ENSX     mean of LRX and RF = ReliSense v1 (reference; must reproduce Paper A)
                   ENSX_PC  the same with prior correction                      (effect of balancing)
                   WAV_PC   learned weight of LRX and RF, prior correction        (effect of the learned weight)
                   WAVH_PC  learned weight of LRXH and RF, prior correction       = ReliSense-2 (main candidate)
                   STKH     stacking of LRXH and RF                               (alternative fusion)
                   STKH3    stacking of LRXH, RF and OSP                          (Paderborn only, exploratory)
Reported: accuracy, balanced accuracy, macro F1, false alarms, missed faults per protocol; LOBO abstention at kappa
0.9 / 0.8 / 0.7; LOBO per-bearing accuracy with the Wilcoxon signed-rank test of WAVH_PC and STKH against RF, LRX and
ENSX; bearing-bootstrap interval; mean learned weight w; external datasets with abstention at kappa 0.7 (new: Paper A
evaluated abstention on Paderborn only).
Inputs (all produced by earlier phases): <root>/pb32/pb32_physics.npz, <root>/checkpoints/phase19/*.npz (reused when
present, otherwise refitted), <bench>/phase10_results/phase20_<ds>_variants.npz and phase21_<ds>_extended.npz.
Output: <root>/phase26_results.json, checkpoints in <root>/checkpoints/phase26/.
Colab (CPU runtime):
  from google.colab import drive; drive.mount('/content/drive')
  !python /content/drive/MyDrive/ReliSense_study/phase26_relisense_v2.py --root /content/drive/MyDrive/ReliSense_study
Options: --part pb | ext | all (default all); --no-osp (skip the exploratory OSP branch, faster); --trees 500.
"""
import argparse, itertools, json, os, time, warnings
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=UserWarning)
ap = argparse.ArgumentParser(); ap.add_argument('--root', default='/content/drive/MyDrive/ReliSense_study')
ap.add_argument('--bench', default=None, help='default: <root>/benchmarks'); ap.add_argument('--part', default='all', choices=('pb', 'ext', 'all'))
ap.add_argument('--trees', type=int, default=500); ap.add_argument('--no-osp', action='store_true'); ap.add_argument('--fresh', action='store_true')
ap.add_argument('--datasets', default='cwru,hust,ottawa')
a = ap.parse_args(); ROOT = a.root; BENCH = a.bench or ROOT + '/benchmarks'
CK = ROOT + '/checkpoints/phase26'; CK19 = ROOT + '/checkpoints/phase19'; os.makedirs(CK, exist_ok=True); t0 = time.time()
KAP = (0.9, 0.8, 0.7); EPS = 1e-3; USE_OSP = not a.no_osp


def el(): return f'{time.time() - t0:.0f} s'


# ------------------------------------------------------------------ common pieces
def lr(balanced, C=0.3):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight='balanced' if balanced else None))


def proba(m, X):
    P = np.zeros((len(X), 3)); P[:, m.classes_] = m.predict_proba(X); return P


def harmonic_agreement(P15):
    """median and minimum of the three harmonic peak ratios of BPFO (cols 0-2), BPFI (3-5) and BSF (6-8)."""
    out = []
    for j in (0, 3, 6):
        h = P15[:, j:j + 3]; out += [np.median(h, 1), h.min(1)]
    return np.stack(out, 1)


def zlog(Ps): return np.concatenate([np.log(np.clip(P, EPS, 1.0)) for P in Ps], 1)


def meta(): return LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced')


def stack(inner_Ps, y_in, g_in, outer_Ps):
    """Fit the meta model on the inner out-of-fold posteriors; return outer posteriors, cross-fitted inner posteriors and
    the meta coefficients. inner_Ps / outer_Ps: lists of (n, 3) posteriors, one per branch."""
    Z = zlog(inner_Ps); m = meta().fit(Z, y_in); P_out = proba(m, zlog(outer_Ps))
    P_cf = np.full((len(y_in), 3), np.nan)
    for v in np.unique(g_in):
        tr = g_in != v
        if len(set(y_in[tr])) < 3: continue
        P_cf[g_in == v] = proba(meta().fit(Z[tr], y_in[tr]), Z[g_in == v])
    return P_out, P_cf, m.coef_.copy()


def tau_from(P_inner, kappa):
    c = np.nanmax(P_inner, 1); c = c[np.isfinite(c)]; return float(np.quantile(c, 1 - kappa))


def summary(P, yt, bt=None):
    p = P.argmax(1); h = yt == 0; r = {'n': int(len(yt)), 'acc': float((p == yt).mean()), 'bacc': float(balanced_accuracy_score(yt, p)),
                                        'f1': float(f1_score(yt, p, average='macro', labels=[0, 1, 2]))}
    r['false_alarm'] = float((p[h] != 0).mean()) if h.any() else None; r['missed_fault'] = float((p[~h] == 0).mean())
    if bt is not None: r['per_bearing'] = {str(b): float((p[bt == b] == yt[bt == b]).mean()) for b in sorted(set(bt))}
    return r


def selective(P, yt, acc):
    p = P.argmax(1); ok = p == yt; h = yt == 0
    return {'coverage': float(acc.mean()), 'sel_acc': float(ok[acc].mean()) if acc.any() else None,
            'false_alarm': float((acc & h & (p != 0)).sum() / max(h.sum(), 1)), 'missed_fault': float((acc & ~h & (p == 0)).sum() / max((~h).sum(), 1))}


def inner_groups(y_tr, prim, sec):
    """primary inner key unless a class has fewer than two values; then the secondary key; None if still not identifiable."""
    for key in (prim, sec):
        if key is None: continue
        if all(len(set(key[y_tr == c])) >= 2 for c in range(3)): return key
    return None


def oof(fit_predict, X, y, tr_idx, g_tr):
    """out-of-fold posteriors over the inner groups; rows of folds that miss a class stay NaN."""
    P = np.full((len(tr_idx), 3), np.nan)
    def one(v):
        itr, ite = tr_idx[g_tr != v], tr_idx[g_tr == v]
        if len(set(y[itr])) < 3: return v, None
        return v, fit_predict(X, y, itr, ite)
    for v, p in Parallel(n_jobs=-1)(delayed(one)(v) for v in np.unique(g_tr)):
        if p is not None: P[g_tr == v] = p
    return P


def fp_lr(balanced, C=0.3):
    def f(X, y, itr, ite): return proba(lr(balanced, C).fit(X[itr], y[itr]), X[ite])
    return f


def fp_rf(X, y, itr, ite):
    return proba(RandomForestClassifier(a.trees, random_state=0, n_jobs=1).fit(X[itr], y[itr]), X[ite])


def pc(P, prior):
    """prior correction to uniform class priors (Saerens et al. 2002)."""
    Q = P / prior[None, :]; return Q / Q.sum(1, keepdims=True)


def bal_logloss(P, y):
    return float(np.mean([-np.log(np.clip(P[y == c, c], EPS, 1.0)).mean() for c in range(3) if (y == c).any()]))


W_GRID = sorted(np.round(np.linspace(0, 1, 11), 1), key=lambda w: abs(w - 0.5))   # ties: closest to 0.5


def best_w(Pk, Pe, y):
    best, bw = np.inf, 0.5
    for w in W_GRID:
        L = bal_logloss(w * Pk + (1 - w) * Pe, y)
        if L < best - 1e-12: best, bw = L, w
    return bw


def fuse_all(branches_out, branches_in, y_in, g_in, names3):
    """all fusions for one split. branches_*: dict name -> posteriors (inner is None if not identifiable).
    Returns outer posteriors, inner (cross-fitted) posteriors for the thresholds, and the fusion parameters."""
    o, i = branches_out, branches_in; F_out, F_in, par = {}, {}, {}
    prior = np.array([(y_in == c).mean() for c in range(3)])
    F_out['ENSX'] = (o['LRX'] + o['RF']) / 2; F_out['ENSX_PC'] = (pc(o['LRX'], prior) + pc(o['RF'], prior)) / 2
    if i is not None:
        F_in['ENSX'] = (i['LRX'] + i['RF']) / 2; F_in['ENSX_PC'] = (pc(i['LRX'], prior) + pc(i['RF'], prior)) / 2
    for m, (bk, be) in {'WAV_PC': ('LRX', 'RF'), 'WAVH_PC': ('LRXH', 'RF')}.items():
        Ok, Oe = pc(o[bk], prior), pc(o[be], prior)
        if i is None:
            F_out[m] = (Ok + Oe) / 2; F_in[m] = None; par[m] = np.array(0.5); continue
        Ik, Ie = pc(i[bk], prior), pc(i[be], prior); ok = np.all(np.isfinite(np.c_[Ik, Ie]), 1)
        w = best_w(Ik[ok], Ie[ok], y_in[ok]); F_out[m] = w * Ok + (1 - w) * Oe; par[m] = np.array(w)
        cf = np.full((len(y_in), 3), np.nan)
        for v in np.unique(g_in):                       # cross-fitted inner posteriors for the threshold
            sel = ok & (g_in != v); tv = ok & (g_in == v)
            if not tv.any(): continue
            wv = best_w(Ik[sel], Ie[sel], y_in[sel]); cf[tv] = wv * Ik[tv] + (1 - wv) * Ie[tv]
        F_in[m] = cf
    specs = {'STKH': ('LRXH', 'RF')}
    if names3: specs['STKH3'] = ('LRXH', 'RF', 'OSP')
    for m, br in specs.items():
        if i is None:                                   # not identifiable: equal-weight average
            F_out[m] = np.mean([o[b] for b in br], 0); F_in[m] = None; par[m] = None; continue
        ok = np.all(np.isfinite(np.concatenate([i[b] for b in br], 1)), 1)
        P_out, P_cf, c = stack([i[b][ok] for b in br], y_in[ok], g_in[ok], [o[b] for b in br])
        F_out[m] = P_out; full = np.full((len(y_in), 3), np.nan); full[ok] = P_cf; F_in[m] = full; par[m] = c
    return F_out, F_in, par


def ck(name, fn):
    p = f'{CK}/{name}.npz'
    if os.path.exists(p) and not a.fresh:
        z = dict(np.load(p, allow_pickle=True)); return {k: (None if v.dtype == object and v.shape == () and v.item() is None else v) for k, v in z.items()}
    r = fn(); np.savez_compressed(p, **{k: (np.array(None, dtype=object) if v is None else v) for k, v in r.items()}); return r


BRANCHES = ['LRX', 'LRXH', 'RF'] + (['OSP'] if USE_OSP else [])
FUSIONS = ['ENSX', 'ENSX_PC', 'WAV_PC', 'WAVH_PC', 'STKH'] + (['STKH3'] if USE_OSP else [])
RES = {'settings': {'trees': a.trees, 'osp': USE_OSP, 'kappa': KAP}}


# ================================================================== Paderborn
def paderborn():
    d = dict(np.load(ROOT + '/pb32/pb32_physics.npz', allow_pickle=True))
    B, y, cond = d['bearing'], d['label'].astype(int), d['condition']
    grid = d['order_grid'].astype(float); O = np.nan_to_num(d['order_fixed'].astype(float))
    n_, d_, D_ = 8, 6.75, 28.55; q = d_ / D_
    BPFO, BPFI, BSF, FTF = n_ / 2 * (1 - q), n_ / 2 * (1 + q), D_ / (2 * d_) * (1 - q ** 2), (1 - q) / 2
    CENTRES = ([h * BPFI + s for h in (1, 2) for s in (-1, 1)] + [h * BPFO + s * FTF for h in (1, 2) for s in (-1, 1)] +
               [h * BSF + s * FTF for h in (1, 2) for s in (-1, 1)] + [4 * BSF, 5 * BSF, 4 * FTF, 5 * FTF])
    ext = []
    for r0 in CENTRES:                                   # identical to phase 19
        w = max(0.02 * r0, 0.03); win = np.abs(grid - r0) <= w; bg = (np.abs(grid - r0) <= 0.4) & ~win
        ext.append(O[:, win].max(1) - np.median(O[:, bg], 1))
    P15 = np.nan_to_num(d['feat_fixed'].astype(float))
    X = {'LRX': np.c_[P15, np.stack(ext, 1)]}
    X['LRXH'] = np.c_[X['LRX'], harmonic_agreement(P15)]
    X['RF'] = np.nan_to_num(np.concatenate([d[f'feat_{v}'] for v in ('fixed', 'sk', 'cpw', 'cpw_sk')] +
                                           [np.log(d['stat_kurtosis'])[:, None], np.log(d['stat_crest'])[:, None]], 1).astype(float))
    if USE_OSP:
        edges = np.arange(0.2, 11.8 + 1e-9, 0.1); X['OSP'] = np.stack([O[:, (grid >= lo) & (grid < lo + 0.1)].max(1) for lo in edges[:-1]], 1)
    FP = {'LRX': fp_lr(False), 'LRXH': fp_lr(False), 'RF': fp_rf, 'OSP': fp_lr(False)}
    print(f'Paderborn: {len(y)} recordings; features LRX {X["LRX"].shape[1]}, LRXH {X["LRXH"].shape[1]}, RF {X["RF"].shape[1]}'
          + (f', OSP {X["OSP"].shape[1]}' if USE_OSP else ''), flush=True)
    bearings = list(dict.fromkeys(B)); lab = {b: int(y[B == b][0]) for b in bearings}; org = {b: str(d['origin'][B == b][0]) for b in bearings}
    single = [b for b in bearings if lab[b] < 3]
    COND0 = 'N15_M07_F10'
    L8_TRAIN = ['K002', 'KA01', 'KA05', 'KA07', 'KI01', 'KI05', 'KI07']
    L8_TEST = ['K001', 'KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']
    L10 = [['K001', 'K002', 'K003', 'K004', 'K005'], ['KA04', 'KA15', 'KA16', 'KA22', 'KA30'], ['KI04', 'KI14', 'KI16', 'KI18', 'KI21']]
    healthy = sorted(b for b in bearings if lab[b] == 0); art = [b for b in bearings if org[b] == 'artificial']
    real = [b for b in bearings if org[b] == 'real' and lab[b] < 3]

    def splits():
        yield 'L8_c0', 0, L8_TRAIN, L8_TEST, True
        yield 'L8_all', 0, L8_TRAIN, L8_TEST, False
        for k, tr in enumerate(itertools.combinations(range(5), 3)):
            yield 'L10', k, [c[i] for c in L10 for i in tr], [c[i] for c in L10 for i in range(5) if i not in tr], False
        yield 'A2R', 0, healthy[:3] + art, healthy[3:] + real, False
        for b in single:
            yield 'LOBO', b, [x for x in single if x != b], [b], False

    def run(proto, k, trb, teb, c0):
        tr, te = np.isin(B, trb), np.isin(B, teb)
        if c0: tr &= cond == COND0; te &= cond == COND0
        tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
        c19 = f'{CK19}/{"lobo_" + k if proto == "LOBO" else f"{proto}_{k}"}.npz'; o19 = dict(np.load(c19)) if os.path.exists(c19) else None
        out = {}
        for m in BRANCHES:                               # outer posteriors
            if o19 is not None and m in ('LRX', 'RF'):
                key = f'outer_{m}' if proto == 'LOBO' else m; out[m] = o19[key]; assert len(out[m]) == len(te_idx)
            else:
                out[m] = FP[m](X[m], y, tr_idx, te_idx)
        g = inner_groups(y[tr_idx], B[tr_idx], cond[tr_idx]); inn = None
        if g is not None:
            inn = {}
            for m in BRANCHES:
                if o19 is not None and proto == 'LOBO' and m in ('LRX', 'RF'):
                    pool = o19[f'inner_{m}']; order = np.concatenate([np.where(B == v)[0] for v in trb])
                    assert len(pool) == len(order) and set(order) == set(tr_idx), 'phase 19 inner pool does not match'
                    P = np.full((len(tr_idx), 3), np.nan); pos = {r: j for j, r in enumerate(tr_idx)}
                    P[[pos[r] for r in order]] = pool; inn[m] = P
                else:
                    inn[m] = oof(lambda Xm, yy, itr, ite, f=FP[m]: f(Xm, yy, itr, ite), X[m], y, tr_idx, g)
        F_out, F_in, coefs = fuse_all(out, inn, y[tr_idx], g, USE_OSP)
        r = {'y': y[te_idx], 'b': B[te_idx], 'c': cond[te_idx], 'g': g if g is not None else None,
             'identifiable': np.array(g is not None), 'reused19': np.array(o19 is not None)}
        for m in BRANCHES: r['out_' + m] = out[m]
        for m in FUSIONS:
            r['out_' + m] = F_out[m]; r['in_' + m] = F_in.get(m); r['par_' + m] = coefs.get(m)
        return r

    R = {}; n_split = 0; total = 2 + 10 + 1 + len(single)
    for proto, k, trb, teb, c0 in splits():
        R[(proto, k)] = ck(f'pb_{proto}_{k}', lambda: run(proto, k, trb, teb, c0)); n_split += 1
        print(f'  {proto} {k}  ({n_split}/{total}, {el()})' + ('' if R[(proto, k)]['identifiable'] else '  [stacking not identifiable: equal weights]'), flush=True)

    out = {'protocols': {}, 'lobo': {}}
    ALL = BRANCHES + FUSIONS
    for proto in ('L8_c0', 'L8_all', 'L10', 'A2R', 'LOBO'):
        ks = [kk for (pp, kk) in R if pp == proto]
        for m in ALL:
            if proto == 'L10':                            # mean over the 10 splits, as in Paper A
                ss = [summary(R[(proto, kk)]['out_' + m], R[(proto, kk)]['y']) for kk in ks]
                out['protocols'].setdefault(proto, {})[m] = {q: float(np.mean([s[q] for s in ss])) for q in ('acc', 'bacc', 'f1', 'false_alarm', 'missed_fault')}
            else:
                P = np.concatenate([R[(proto, kk)]['out_' + m] for kk in ks]); yy = np.concatenate([R[(proto, kk)]['y'] for kk in ks])
                bb = np.concatenate([R[(proto, kk)]['b'] for kk in ks]); out['protocols'].setdefault(proto, {})[m] = summary(P, yy, bb)
    # LOBO abstention and bearing statistics
    yL = np.concatenate([R[('LOBO', b)]['y'] for b in single]); bL = np.concatenate([R[('LOBO', b)]['b'] for b in single])
    for m in FUSIONS:
        P = np.concatenate([R[('LOBO', b)]['out_' + m] for b in single]); conf = P.max(1)
        for kap in KAP:
            acc = np.concatenate([R[('LOBO', b)]['out_' + m].max(1) >= tau_from(R[('LOBO', b)]['in_' + m], kap) for b in single])
            sel = selective(P, yL, acc); cov_c = {n: float(acc[yL == c].mean()) for c, n in enumerate(('H', 'OR', 'IR'))}
            out['lobo'].setdefault(m, {})[f'kappa_{kap}'] = {**sel, 'coverage_by_class': cov_c}
        if m.startswith('WAV'):
            ws = [float(R[('LOBO', b)]['par_' + m]) for b in single]; out['lobo'][m]['w_mean'] = float(np.mean(ws)); out['lobo'][m]['w_range'] = [min(ws), max(ws)]
        if m.startswith('STK'):
            out['lobo'][m]['mean_abs_coef_by_branch'] = np.mean([np.abs(R[('LOBO', b)]['par_' + m]).reshape(3, -1, 3).sum((0, 2)) for b in single], 0).tolist()
    for proto in ('L8_c0', 'L8_all', 'L10', 'A2R'):
        out['protocols'][proto]['w_WAVH_PC'] = [float(R[(pp, kk)]['par_WAVH_PC']) for (pp, kk) in R if pp == proto]
    pb = {m: out['protocols']['LOBO'][m]['per_bearing'] for m in ALL}
    for ref in ('RF', 'LRX', 'ENSX'):
        for m in ('WAVH_PC', 'STKH') + (('STKH3',) if USE_OSP else ()):
            dlt = np.array([pb[m][b] - pb[ref][b] for b in single])
            p = float(wilcoxon(dlt).pvalue) if np.any(np.abs(dlt) > 1e-12) else 1.0
            out['lobo'].setdefault('wilcoxon', {})[f'{m}_vs_{ref}'] = {'better': int((dlt > 0.005).sum()), 'worse': int((dlt < -0.005).sum()), 'p': p}
    rng = np.random.default_rng(0); groups = {c: [b for b in single if lab[b] == c] for c in range(3)}; idx = {b: np.where(bL == b)[0] for b in single}
    for m in ('ENSX', 'WAVH_PC', 'STKH'):
        okm = np.concatenate([R[('LOBO', b)]['out_' + m] for b in single]).argmax(1) == yL; bs = []
        for _ in range(2000):
            sel = np.concatenate([idx[b] for c, gg in groups.items() for b in rng.choice(gg, len(gg), replace=True)]); bs.append(okm[sel].mean())
        out['lobo'].setdefault('ci95', {})[m] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


# ================================================================== external datasets
def protocols(name, d):                                    # = phase10_benchmarks.protocols
    groups = sorted(set(d['group'][d['cls'] != 'H'])); H = d['cls'] == 'H'
    if name == 'cwru':
        groups = [g for g in groups if g != '028']
        h_tr = H & np.isin(d['cond'], ['0', '1']); h_te = H & np.isin(d['cond'], ['2', '3']); fault = ~H & np.isin(d['group'], groups)
        for g in groups: yield 'LOSO', g, (fault & (d['group'] != g)) | h_tr, (fault & (d['group'] == g)) | h_te
        for g in groups: yield '1SIZE', g, (fault & (d['group'] == g)) | h_tr, (fault & (d['group'] != g)) | h_te
    elif name == 'hust':
        for g in sorted(set(d['group'])): yield 'LOTO', g, d['group'] != g, d['group'] == g
        for g in sorted(set(d['group'])): yield '1TYPE', g, d['group'] == g, d['group'] != g
    else:
        for g in sorted(set(d['group'])): yield 'LOPO', g, d['group'] != g, d['group'] == g
        for g in sorted(set(d['group'])): yield '1PROF', g, d['group'] == g, d['group'] != g


def external(name):
    base = f'{BENCH}/phase10_results'
    d = dict(np.load(f'{base}/phase20_{name}_variants.npz', allow_pickle=False)); e = dict(np.load(f'{base}/phase21_{name}_extended.npz', allow_pickle=False))
    assert (e['recording'] == d['recording']).all(), 'segments differ between phases 20 and 21'
    CLS3 = ['H', 'OR', 'IR']; keep = np.isin(d['cls'], CLS3); y = np.array([CLS3.index(c) if c in CLS3 else -1 for c in d['cls']])
    P15 = np.nan_to_num(d['P_fixed'].astype(float))
    X = {'LRX': np.nan_to_num(e['X31'].astype(float))}; X['LRXH'] = np.c_[X['LRX'], harmonic_agreement(P15)]
    X['RF'] = np.nan_to_num(np.c_[d['P_fixed'], d['P_sk'], d['P_cpw'], d['P_cpw_sk'], np.log(np.abs(d['T'][:, 2]) + 1e-12), np.log(np.abs(d['T'][:, 4]) + 1e-12)].astype(float))
    FP = {'LRX': fp_lr(False), 'LRXH': fp_lr(False), 'RF': fp_rf}
    BR = ['LRX', 'LRXH', 'RF']; FU = ['ENSX', 'ENSX_PC', 'WAV_PC', 'WAVH_PC', 'STKH']
    if name == 'cwru': prim = np.where(d['cls'] == 'H', np.char.add('c', d['cond'].astype(str)), np.char.add('g', d['group'].astype(str))); sec = d['cond'].astype(str)
    elif name == 'hust': prim, sec = d['group'].astype(str), d['cond'].astype(str)
    else: prim, sec = d['group'].astype(str), d['recording'].astype(str)
    res = {}
    for proto, fold, tr, te in protocols(name, d):
        tr, te = tr & keep, te & keep
        if not tr.any() or not te.any() or len(set(y[tr])) < 3: print(f'  {name} {proto} {fold}: skipped'); continue
        def run():
            tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
            out = {m: FP[m](X[m], y, tr_idx, te_idx) for m in BR}
            g = inner_groups(y[tr_idx], prim[tr_idx], sec[tr_idx]); inn = None
            if g is not None: inn = {m: oof(lambda Xm, yy, itr, ite, f=FP[m]: f(Xm, yy, itr, ite), X[m], y, tr_idx, g) for m in BR}
            F_out, F_in, coefs = fuse_all(out, inn, y[tr_idx], g, False)
            r = {'y': y[te_idx], 'u': d['unit'][te_idx], 'identifiable': np.array(g is not None)}
            for m in BR: r['out_' + m] = out[m]
            for m in FU: r['out_' + m] = F_out[m]; r['in_' + m] = F_in.get(m); r['par_' + m] = coefs.get(m)
            return r
        r = ck(f'{name}_{proto}_{fold}', run); res.setdefault(proto, []).append(r)
        print(f'  {name} {proto} {fold}  ({el()})' + ('' if r['identifiable'] else '  [stacking not identifiable: equal weights]'), flush=True)
    out = {}
    for proto, rs in res.items():
        yy = np.concatenate([r['y'] for r in rs])
        for m in BR + FU: out.setdefault(proto, {})[m] = summary(np.concatenate([r['out_' + m] for r in rs]), yy)
        for m in ('ENSX', 'WAVH_PC', 'STKH'):
            P = np.concatenate([r['out_' + m] for r in rs])
            acc = np.concatenate([r['out_' + m].max(1) >= (tau_from(r['in_' + m], 0.7) if r['in_' + m] is not None else -1) for r in rs])
            out[proto][m]['kappa_0.7'] = selective(P, yy, acc)
        out[proto]['WAVH_PC']['w_folds'] = [float(r['par_WAVH_PC']) for r in rs]
    return out


# ================================================================== run and print
def f1(x): return '   -  ' if x is None else f'{100 * x:6.1f}'


if a.part in ('pb', 'all'):
    RES['paderborn'] = paderborn(); json.dump(RES, open(ROOT + '/phase26_results.json', 'w'), indent=1)
if a.part in ('ext', 'all'):
    for nm in [s.strip() for s in a.datasets.split(',') if s.strip()]:
        try:
            RES.setdefault('external', {})[nm] = external(nm)
        except FileNotFoundError as err:
            print(f'  {nm}: cache not found ({err}); skipped')
        json.dump(RES, open(ROOT + '/phase26_results.json', 'w'), indent=1)

ALLM = BRANCHES + FUSIONS
if 'paderborn' in RES:
    Rp = RES['paderborn']
    print('\n=== Paderborn: accuracy / false alarms (%)   [L10: mean of 10 splits]   ENSX = ReliSense v1, WAVH_PC = ReliSense-2')
    print(f"{'model':8}" + ''.join(f'{p:>16}' for p in ('L8_c0', 'L8_all', 'L10', 'A2R', 'LOBO')))
    for m in ALLM:
        print(f'{m:8}' + ''.join(f"{f1(Rp['protocols'][p][m]['acc'])} / {f1(Rp['protocols'][p][m]['false_alarm'])}" for p in ('L8_c0', 'L8_all', 'L10', 'A2R', 'LOBO')))
    print('\n=== Paderborn: balanced accuracy / macro F1 / missed faults (%)')
    for m in ALLM:
        print(f'{m:8}' + ''.join(f"{f1(Rp['protocols'][p][m]['bacc'])}/{f1(Rp['protocols'][p][m]['f1'])}/{f1(Rp['protocols'][p][m]['missed_fault'])}" for p in ('L8_c0', 'L8_all', 'L10', 'A2R', 'LOBO')))
    print('\n=== LOBO abstention (training-only thresholds): coverage / selective accuracy / false alarms / missed faults (%); healthy coverage')
    for m in FUSIONS:
        for k in KAP:
            s = Rp['lobo'][m][f'kappa_{k}']
            print(f'  {m:6} kappa {k}: {f1(s["coverage"])} / {f1(s["sel_acc"])} / {f1(s["false_alarm"])} / {f1(s["missed_fault"])}   H {f1(s["coverage_by_class"]["H"])}')
    print('\n=== LOBO per bearing (%):')
    pbm = {m: Rp['protocols']['LOBO'][m]['per_bearing'] for m in ALLM}
    for b in ('K004', 'K005', 'K001', 'K002', 'K003', 'K006', 'KI17', 'KA15', 'KA22', 'KA30'):
        print(f'  {b}: ' + '  '.join(f'{m} {100 * pbm[m][b]:5.1f}' for m in ('RF', 'LRX', 'LRXH', 'ENSX', 'ENSX_PC', 'WAVH_PC', 'STKH') + (('STKH3',) if USE_OSP else ())))
    print('Wilcoxon (per-bearing accuracy, 29 bearings):', {k: f"better {v['better']}, worse {v['worse']}, p = {v['p']:.3f}" for k, v in Rp['lobo']['wilcoxon'].items()})
    print('LOBO 95% bearing-bootstrap interval:', {k: f'{100 * v[0]:.1f}-{100 * v[1]:.1f}' for k, v in Rp['lobo']['ci95'].items()})
    print('learned weight of the kinematic branch under LOBO (mean, range):', {m: (round(Rp['lobo'][m]['w_mean'], 2), Rp['lobo'][m]['w_range']) for m in ('WAV_PC', 'WAVH_PC')})
    print('learned weight (WAVH_PC) in the other protocols:', {p: Rp['protocols'][p]['w_WAVH_PC'] for p in ('L8_c0', 'L8_all', 'L10', 'A2R')})
    print('mean |meta coefficient| by branch (STKH: LRXH, RF; STKH3: LRXH, RF, OSP):',
          {m: [round(x, 2) for x in Rp['lobo'][m]['mean_abs_coef_by_branch']] for m in FUSIONS if m.startswith('STK')})
    print('check: ENSX must reproduce Paper A (LOBO 83.7 / 32.1; kappa 0.7: 68.6 / 97.5 / 2.7):',
          f"{100 * Rp['protocols']['LOBO']['ENSX']['acc']:.1f} / {100 * Rp['protocols']['LOBO']['ENSX']['false_alarm']:.1f};",
          ' / '.join(f"{100 * Rp['lobo']['ENSX']['kappa_0.7'][q]:.1f}" for q in ('coverage', 'sel_acc', 'false_alarm')))
if 'external' in RES:
    print('\n=== External datasets: accuracy / false alarms / missed faults (%); last columns: kappa 0.7 coverage / selective accuracy / false alarms')
    for nm, rr in RES['external'].items():
        for proto, ms in rr.items():
            print(f'  {nm} {proto}')
            for m in ('LRX', 'LRXH', 'RF', 'ENSX', 'ENSX_PC', 'WAV_PC', 'WAVH_PC', 'STKH'):
                s = ms[m]; line = f'    {m:6} {f1(s["acc"])} / {f1(s["false_alarm"])} / {f1(s["missed_fault"])}   bacc {f1(s["bacc"])}'
                if 'kappa_0.7' in s: k7 = s['kappa_0.7']; line += f'   | k0.7 {f1(k7["coverage"])} / {f1(k7["sel_acc"])} / {f1(k7["false_alarm"])}'
                print(line)
            print('    learned weight of the kinematic branch per fold (WAVH_PC):', ms['WAVH_PC']['w_folds'])
print(f'\nwritten {ROOT}/phase26_results.json  ({el()})')
