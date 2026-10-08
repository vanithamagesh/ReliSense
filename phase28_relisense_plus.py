"""Phase 28 (v1.1): ReliSense+ - condition-aware, comb-separated and slip-consistent fault features, sequential decisions.
v1.1: fusions are rebuilt from the stored branch posteriors whenever a checkpoint is loaded, and a checkpoint that lacks a
branch needed by the current settings is recomputed; checkpoints written with and without --no-rf-cw can therefore be mixed.
Run only one session at a time on the same folder.
Contains phase 27 v1.1 unchanged (run this script instead of phase 27) and adds:
  SC   slip-consistent comb features. All harmonics of a localized defect share one slip s: the lines lie at h*F*(1 - s),
       whereas shaft-locked lines lie at integer orders. For each fault family F (BPFO, BPFI, BSF) and slip s in
       [-1 %, 3 %] (step 0.25 %) the comb evidence C(s) is the mean over the harmonics (h = 1-3, orders <= 11.6) of the
       log line height (local maximum within one bin) above the median of 0.1-0.4 orders around it (harmonic summation,
       cf. Christensen and Jakobsson). Four features per family: best comb evidence max_s C(s); best slip; comb
       specificity = best comb evidence minus the mean evidence at the integer orders <= 11.6 that are not part of the
       best comb (a shaft-harmonic family raises all integers, a defect comb only its own lines); line sharpness = mean
       over the comb lines of peak minus log mean power within +-0.06 orders (random slip broadens bearing lines).
       12 features, appended to the comb-separated branch NF: LRNS = 40 + 12 = 52 features, LR C = 0.3.
  SEQ  decisions from several consecutive recordings of one bearing (sum of log posteriors over k = 1, 2, 5, 10
       recordings, chunks in stored order within a bearing; Paderborn stores 20 recordings per condition in sequence)
       and a sequential rule (Wald 1945) that adds recordings until the fused confidence reaches 0.95 (at most 10).
       Applied identically to every fusion and to the stored phase 23 / 24 network posteriors (LOBO).
  CT   condition-specific abstention thresholds: tau_{b,c} = (1 - kappa) quantile of the inner confidences of operating
       condition c (Paderborn LOBO), against the pooled threshold.
Everything else (CW, MB, NF, PC, protocols, inner loops, adoption rule over all fusions except V1) as phase 27 v1.1:

  CW   condition-wise standardization: every feature is standardized with the mean and SD of the TRAINING recordings of
       the same operating condition (inner loops: of the inner training recordings). It is used only when every class of
       the training set occurs in every training condition and every test condition occurs in training (Paderborn, HUST);
       otherwise global standardization is used, i.e. ReliSense v1 (CWRU: the healthy bearing has two loads in training;
       Ottawa: the condition is the held-out speed profile).
  MB   multiband kinematic branch: LR (C = 0.3) on the 15 kinematic features in all four demodulation variants (fixed,
       kurtogram, pre-whitened, pre-whitened + kurtogram) plus the 16 modulation features = 76 features.
  NF   comb-separated kinematic branch. For the Paderborn 6203, BPFO = 3 + 0.054 and BPFI = 5 - 0.054 orders, and every
       fault-order search window (+-max(2 %, 0.03) orders) of harmonics 1-3 contains an integer shaft order (2, 3, 4, 5, 6,
       9, 10, 15). Shaft-locked lines sit exactly at integer orders, whereas bearing lines are shifted by slip (Randall and
       Antoni 2011), and the separation 0.054 h orders is 1.35 Hz at 1500 rpm but 0.81 Hz at 900 rpm with 0.25-Hz bins.
       In phase 22, 41 % of the LOBO errors were healthy recordings diagnosed as damaged and 27 % outer/inner-race
       confusions, 42 % of all errors fell at 900 rpm, and the healthy K001 shows strong lines at 4.0 and 6.1 orders.
       NF reads every fault-order and modulation window WITHOUT the bins within 0.03 orders of an integer order (notch),
       and adds, for every fault-order window, the strength of the integer-order line inside it as a separate feature:
       9 notched fault features (BPFO, BPFI, BSF, h = 1-3) + 9 integer-line features + FTF and shaft (6, unchanged) +
       16 notched modulation features = 40 features, LR C = 0.3. Paderborn: from the stored order spectrum (0.02 orders,
       up to 12); the third BPFI harmonic (14.84) lies beyond it and keeps its standard feature. Further datasets: from
       50-revolution segments (0.02 orders) recomputed with the phase 10 code, cached with SC as phase28_<ds>_comb.npz.
  PC   decision with equal class priors (phase 26).
Branches: LRX (v1 kinematic, 31), LRX_C, LRM (76), LRM_C, LRN (40), LRN_C, LRNS (52), LRNS_C, RF (v1 envelope, 62), RF_C.
Fusions (mean of posteriors): V1 = LRX+RF (ReliSense v1), CW = LRX_C+RF_C, MB = LRM+RF, MBCW = LRM_C+RF_C, NF = LRN+RF,
NFCW = LRN_C+RF_C, NFS = LRNS+RF, NFSCW = LRNS_C+RF_C; each also _PC.
Adoption rule (fixed in advance): among all fusions except V1, take the fusion with the highest
mean balanced accuracy over the 11 bearing-wise settings (Paderborn L8 c0, L8 all, L10, A2R, LOBO; CWRU LOSO, 1SIZE; HUST
LOTO, 1TYPE; Ottawa LOPO, 1PROF). It replaces ReliSense v1 only if (i) its mean balanced accuracy is higher than that of V1,
(ii) its LOBO accuracy is not lower, and (iii) its LOBO selective accuracy at kappa 0.7 is at most 0.5 points lower.
Inputs: <root>/pb32/pb32_physics.npz, <root>/checkpoints/phase19/ (LRX and RF reused), <root>/checkpoints/phase23,
phase24 (network posteriors, optional), <bench>/phase10_results/phase20_<ds>_variants.npz, phase21_<ds>_extended.npz and
the raw further datasets (for the comb features, cached as phase28_<ds>_comb.npz). Output: <root>/phase28_results.json,
<root>/checkpoints/phase28/ (branch posteriors, inner and outer, are stored for later reuse).
CPU runtime, about 60-80 minutes (--no-rf-cw: about 30 minutes); every split is checkpointed, so a rerun resumes.
Colab (CPU runtime):
  !python /content/drive/MyDrive/ReliSense_study/phase28_relisense_plus.py --root /content/drive/MyDrive/ReliSense_study
"""
import argparse, itertools, json, os, time, warnings
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings('ignore')
ap = argparse.ArgumentParser(); ap.add_argument('--root', default='/content/drive/MyDrive/ReliSense_study')
ap.add_argument('--bench', default=None); ap.add_argument('--trees', type=int, default=500); ap.add_argument('--no-rf-cw', action='store_true')
ap.add_argument('--fresh', action='store_true'); ap.add_argument('--part', default='all', choices=('pb', 'ext', 'all'))
a = ap.parse_args(); ROOT = a.root; BENCH = a.bench or ROOT + '/benchmarks'
CK = ROOT + '/checkpoints/phase28'; CK19 = ROOT + '/checkpoints/phase19'; os.makedirs(CK, exist_ok=True); t0 = time.time()
KAP = (0.9, 0.8, 0.7); RFCW = not a.no_rf_cw
KIN = ['LRX', 'LRX_C', 'LRM', 'LRM_C', 'LRN', 'LRN_C', 'LRNS', 'LRNS_C']; NOTCH = 0.03; ORD_MAX = 11.6; ENV = ['RF'] + (['RF_C'] if RFCW else [])
FUS = {'V1': ('LRX', 'RF'), 'CW': ('LRX_C', 'RF_C' if RFCW else 'RF'), 'MB': ('LRM', 'RF'), 'MBCW': ('LRM_C', 'RF_C' if RFCW else 'RF'),
       'NF': ('LRN', 'RF'), 'NFCW': ('LRN_C', 'RF_C' if RFCW else 'RF'), 'NFS': ('LRNS', 'RF'), 'NFSCW': ('LRNS_C', 'RF_C' if RFCW else 'RF')}
FUSIONS = [f + s for f in FUS for s in ('', '_PC')]
def el(): return f'{time.time() - t0:.0f} s'


# ------------------------------------------------------------------ pieces
def standardize(X, cond, itr, ite, cw):
    """global or condition-wise z-scores from the training rows itr; returns (Xtr, Xte)."""
    def z(rows, mu, sd): return (X[rows] - mu) / sd
    mu, sd = X[itr].mean(0), X[itr].std(0) + 1e-9
    if not cw: return z(itr, mu, sd), z(ite, mu, sd)
    Xtr, Xte = np.empty((len(itr), X.shape[1])), np.empty((len(ite), X.shape[1]))
    for c in np.unique(cond[itr]):
        r = cond[itr] == c; m, s = X[itr][r].mean(0), X[itr][r].std(0) + 1e-9
        Xtr[r] = (X[itr][r] - m) / s
        q = cond[ite] == c
        if q.any(): Xte[q] = (X[ite][q] - m) / s
    return Xtr, Xte


def cw_valid(y, cond, itr, ite):
    """condition-wise standardization only if every class occurs in every training condition and the test conditions occur in training."""
    ct = set(cond[itr])
    if not set(cond[ite]) <= ct: return False
    return all(set(y[itr][cond[itr] == c]) == set(y[itr]) for c in ct)


def comb_separated(S, g, fo, log, beyond=None):
    """NF features of one spectrum: notched fault (BPFO, BPFI, BSF x h 1-3), integer-line strength in the same windows,
    FTF and shaft (h 1-3, standard), notched modulation centres. beyond: standard values for centres outside the grid."""
    L = S if log else np.log(S + 1e-12); dO = g[1] - g[0]; notch = np.abs(g - np.round(g)) <= NOTCH + 1e-9
    def win(c): return np.abs(g - c) <= max(0.02 * c, 0.03, 1.5 * dO)
    def ref(c, w): bg = (np.abs(g - c) <= 0.4) & ~w & ~notch; return np.median(L[bg]) if bg.any() else 0.0
    fault, integ, std, mod = [], [], [], []
    for k in ('BPFO', 'BPFI', 'BSF'):
        for h in (1, 2, 3):
            c = h * fo[k]
            if c + 0.4 > g[-1]:
                fault.append(beyond.get((k, h), 0.0) if beyond else 0.0); integ.append(0.0); continue
            w = win(c); r = ref(c, w); a = w & ~notch; b = w & notch
            fault.append(L[a].max() - r if a.any() else 0.0); integ.append(L[b].max() - r if b.any() else 0.0)
    for k in ('FTF', 'shaft'):
        for h in (1, 2, 3):
            c = h * fo[k]; w = win(c); bg = (np.abs(g - c) <= 0.4) & ~w; std.append(L[w].max() - np.median(L[bg]))
    cen = ([h * fo['BPFI'] + s_ for h in (1, 2) for s_ in (-1, 1)] + [h * fo['BPFO'] + s_ * fo['FTF'] for h in (1, 2) for s_ in (-1, 1)] +
           [h * fo['BSF'] + s_ * fo['FTF'] for h in (1, 2) for s_ in (-1, 1)] + [4 * fo['BSF'], 5 * fo['BSF'], 4 * fo['FTF'], 5 * fo['FTF']])
    for c in cen:
        w = win(c); r = ref(c, w); a = w & ~notch; mod.append(L[a].max() - r if a.any() else 0.0)
    return np.array(fault + integ + std + mod, float)


SLIPS = np.arange(-0.01, 0.03 + 1e-9, 0.0025)


def comb_slip(S, g, fo, log):
    """SC features: per family (BPFO, BPFI, BSF) best comb evidence, best slip, comb specificity, line sharpness."""
    L = S if log else np.log(S + 1e-12); dO = g[1] - g[0]; top = min(g[-1] - 0.4, ORD_MAX)
    def ev(o):
        w = np.abs(g - o) <= dO + 1e-9; bg = (np.abs(g - o) > 0.1) & (np.abs(g - o) <= 0.4)
        return L[w].max() - np.median(L[bg])
    out = []
    for k in ('BPFO', 'BPFI', 'BSF'):
        F = fo[k]; H = [h for h in (1, 2, 3) if h * F * 1.01 <= top]
        C = np.array([np.mean([ev(h * F * (1 - sl)) for h in H]) for sl in SLIPS]); j = int(np.argmax(C)); sb = SLIPS[j]
        pos = [h * F * (1 - sb) for h in H]
        others = [kk for kk in range(1, int(top) + 1) if min(abs(kk - p) for p in pos) > 0.06]
        spec = C[j] - (np.mean([ev(kk) for kk in others]) if others else 0.0)
        sharp = np.mean([L[np.abs(g - p) <= dO + 1e-9].max() - np.log(np.mean(np.exp(L[np.abs(g - p) <= 0.06]))) for p in pos])
        out += [C[j], sb, spec, sharp]
    return np.array(out, float)


def proba(m, X):
    P = np.zeros((len(X), 3)); P[:, m.classes_] = m.predict_proba(X); return P


def make_fp(kind, cw):
    def f(X, y, cond, itr, ite):
        use = cw and cw_valid(y, cond, itr, ite); Xtr, Xte = standardize(X, cond, itr, ite, use)
        if kind == 'lr': m = LogisticRegression(C=0.3, max_iter=5000).fit(Xtr, y[itr])
        else: m = RandomForestClassifier(a.trees, random_state=0, n_jobs=1).fit(Xtr, y[itr])
        return proba(m, Xte)
    return f


def oof(fp, X, y, cond, tr_idx, g):
    P = np.full((len(tr_idx), 3), np.nan)
    def one(v):
        itr, ite = tr_idx[g != v], tr_idx[g == v]
        return v, (fp(X, y, cond, itr, ite) if len(set(y[itr])) == 3 else None)
    for v, p in Parallel(n_jobs=-1)(delayed(one)(v) for v in np.unique(g)):
        if p is not None: P[g == v] = p
    return P


def pc(P, prior): Q = P / prior[None, :]; return Q / Q.sum(1, keepdims=True)


def fuse(out, inn, y_in):
    prior = np.array([(y_in == c).mean() for c in range(3)]); Fo, Fi = {}, {}
    for f, (k, e) in FUS.items():
        Fo[f] = (out[k] + out[e]) / 2; Fo[f + '_PC'] = (pc(out[k], prior) + pc(out[e], prior)) / 2
        if inn is not None:
            Fi[f] = (inn[k] + inn[e]) / 2; Fi[f + '_PC'] = (pc(inn[k], prior) + pc(inn[e], prior)) / 2
    return Fo, Fi


def tau(Pin, k): c = np.nanmax(Pin, 1); c = c[np.isfinite(c)]; return float(np.quantile(c, 1 - k))


def summ(P, y):
    p = P.argmax(1); h = y == 0
    return {'acc': float((p == y).mean()), 'bacc': float(balanced_accuracy_score(y, p)),
            'fa': float((p[h] != 0).mean()) if h.any() else None, 'missed': float((p[~h] == 0).mean())}


def seq_eval(parts, ks=(1, 2, 5, 10), tau_s=0.95, kmax=10):
    """parts: list of (posteriors, labels) of one bearing in stored order. Chunks of k consecutive recordings (sum of log
    posteriors); sequential rule: add recordings until the fused confidence reaches tau_s (at most kmax)."""
    res = {}
    for k in ks + ('seq',):
        pr, yt, used = [], [], []
        for P, y in parts:
            n = len(y); step = kmax if k == 'seq' else k
            for s0 in range(0, n - (0 if k == 'seq' else k - 1), step):
                g = np.arange(s0, min(s0 + step, n))
                if len(g) == 0: continue
                if k == 'seq':
                    Lc = np.zeros(3)
                    for j, i in enumerate(g):
                        Lc += np.log(np.clip(P[i], 1e-6, 1)); q = np.exp(Lc - Lc.max()); q /= q.sum()
                        if q.max() >= tau_s: break
                    used.append(j + 1); pr.append(int(q.argmax()))
                else:
                    pr.append(int(np.log(np.clip(P[g], 1e-6, 1)).sum(0).argmax()))
                yt.append(int(y[g[0]]))
        pr, yt = np.array(pr), np.array(yt); h = yt == 0
        res[str(k)] = {'acc': float((pr == yt).mean()), 'fa': float((pr[h] != 0).mean()), 'missed': float((pr[~h] == 0).mean()), 'n': int(len(pr))}
        if k == 'seq': res['seq']['mean_recordings'] = float(np.mean(used))
    return res


def ck(name, fn, need=()):
    """load a checkpoint; recompute it if it is missing or lacks any key in need."""
    p = f'{CK}/{name}.npz'
    if os.path.exists(p) and not a.fresh:
        r = dict(np.load(p, allow_pickle=True))
        if all(k in r for k in need): return r
        print(f'    {name}: checkpoint lacks {[k for k in need if k not in r]}; recomputed', flush=True)
    r = fn(); np.savez_compressed(p, **r); return r


def refuse(r, y_tr, inner):
    """rebuild all fusions of the current settings from the stored branch posteriors."""
    outs = {m: r['out_' + m] for m in KIN + ENV}; inns = {m: r['inb_' + m] for m in KIN + ENV} if inner else None
    Fo, Fi = fuse(outs, inns, y_tr)
    for f in FUSIONS:
        r['out_' + f] = Fo[f]
        if inner: r['in_' + f] = Fi[f]
    return r


RES = {'settings': {'rf_cw': RFCW, 'trees': a.trees}}


# ------------------------------------------------------------------ Paderborn
def paderborn():
    d = dict(np.load(ROOT + '/pb32/pb32_physics.npz', allow_pickle=True))
    B, y, cond = d['bearing'], d['label'].astype(int), d['condition'].astype(str)
    grid = d['order_grid'].astype(float); O = np.nan_to_num(d['order_fixed'].astype(float))
    n_, d_, D_ = 8, 6.75, 28.55; q = d_ / D_
    BPFO, BPFI, BSF, FTF = n_ / 2 * (1 - q), n_ / 2 * (1 + q), D_ / (2 * d_) * (1 - q ** 2), (1 - q) / 2
    CEN = ([h * BPFI + s for h in (1, 2) for s in (-1, 1)] + [h * BPFO + s * FTF for h in (1, 2) for s in (-1, 1)] +
           [h * BSF + s * FTF for h in (1, 2) for s in (-1, 1)] + [4 * BSF, 5 * BSF, 4 * FTF, 5 * FTF])
    ext = []
    for r0 in CEN:                                        # identical to phase 19
        w = max(0.02 * r0, 0.03); win = np.abs(grid - r0) <= w; bg = (np.abs(grid - r0) <= 0.4) & ~win
        ext.append(O[:, win].max(1) - np.median(O[:, bg], 1))
    ext = np.stack(ext, 1); V = {v: np.nan_to_num(d[f'feat_{v}'].astype(float)) for v in ('fixed', 'sk', 'cpw', 'cpw_sk')}
    X = {'LRX': np.c_[V['fixed'], ext], 'LRM': np.c_[V['fixed'], V['sk'], V['cpw'], V['cpw_sk'], ext]}
    fo = {'BPFO': BPFO, 'BPFI': BPFI, 'BSF': BSF, 'FTF': FTF, 'shaft': 1.0}
    X['LRN'] = np.nan_to_num(np.stack([comb_separated(O[i], grid, fo, True, {('BPFI', 3): V['fixed'][i, 5]}) for i in range(len(O))]))
    XS = np.nan_to_num(np.stack([comb_slip(O[i], grid, fo, True) for i in range(len(O))])); X['LRNS'] = np.c_[X['LRN'], XS]
    X['LRX_C'], X['LRM_C'], X['LRN_C'], X['LRNS_C'] = X['LRX'], X['LRM'], X['LRN'], X['LRNS']
    print(f'Paderborn: {len(y)} recordings; features LRX {X["LRX"].shape[1]}, LRM {X["LRM"].shape[1]}, LRN {X["LRN"].shape[1]}, LRNS {X["LRNS"].shape[1]}', flush=True)
    X['RF'] = np.nan_to_num(np.c_[V['fixed'], V['sk'], V['cpw'], V['cpw_sk'], np.log(d['stat_kurtosis']), np.log(d['stat_crest'])].astype(float))
    X['RF_C'] = X['RF']
    FP = {'LRX': make_fp('lr', False), 'LRX_C': make_fp('lr', True), 'LRM': make_fp('lr', False), 'LRM_C': make_fp('lr', True),
          'LRN': make_fp('lr', False), 'LRN_C': make_fp('lr', True), 'LRNS': make_fp('lr', False), 'LRNS_C': make_fp('lr', True),
          'RF': make_fp('rf', False), 'RF_C': make_fp('rf', True)}
    bearings = list(dict.fromkeys(B)); lab = {b: int(y[B == b][0]) for b in bearings}; org = {b: str(d['origin'][B == b][0]) for b in bearings}
    single = [b for b in bearings if lab[b] < 3]; COND0 = 'N15_M07_F10'
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
        for b in single: yield 'LOBO', b, [x for x in single if x != b], [b], False

    def run(proto, k, trb, teb, c0):
        tr, te = np.isin(B, trb), np.isin(B, teb)
        if c0: tr &= cond == COND0; te &= cond == COND0
        tri, tei = np.where(tr)[0], np.where(te)[0]
        c19 = f'{CK19}/{"lobo_" + k if proto == "LOBO" else f"{proto}_{k}"}.npz'; o = dict(np.load(c19)) if os.path.exists(c19) else None
        out = {}
        for m in KIN + ENV:
            if o is not None and m in ('LRX', 'RF'): out[m] = o[f'outer_{m}' if proto == 'LOBO' else m]
            else: out[m] = FP[m](X[m], y, cond, tri, tei)
        inn = None
        if proto == 'LOBO':                               # inner loop only for the abstention thresholds
            g = B[tri]; inn = {}
            for m in KIN + ENV:
                if o is not None and m in ('LRX', 'RF'):
                    pool = o[f'inner_{m}']; order = np.concatenate([np.where(B == v)[0] for v in trb])
                    P = np.full((len(tri), 3), np.nan); pos = {r: j for j, r in enumerate(tri)}; P[[pos[r] for r in order]] = pool; inn[m] = P
                else: inn[m] = oof(FP[m], X[m], y, cond, tri, g)
        Fo, Fi = fuse(out, inn, y[tri])
        r = {'y': y[tei], 'b': B[tei], 'c': cond[tei], 'cw_used': np.array(cw_valid(y, cond, tri, tei))}
        for m in KIN + ENV:
            r['out_' + m] = out[m]
            if inn is not None: r['inb_' + m] = inn[m]
        for f in FUSIONS:
            r['out_' + f] = Fo[f]
            if inn is not None: r['in_' + f] = Fi[f]
        if inn is not None: r['inc'] = cond[tri]
        return r

    R = {}; total = 2 + 10 + 1 + len(single); n = 0
    for proto, k, trb, teb, c0 in splits():
        need = ['out_' + m for m in KIN + ENV] + (['inb_' + m for m in KIN + ENV] + ['inc'] if proto == 'LOBO' else [])
        r_ = ck(f'pb_{proto}_{k}', lambda: run(proto, k, trb, teb, c0), need)
        tr_ = np.isin(B, trb) & ((cond == COND0) if c0 else True)
        R[(proto, k)] = refuse(r_, y[tr_], proto == 'LOBO'); n += 1
        print(f'  {proto} {k}  ({n}/{total}, {el()})', flush=True)
    out = {'settings': {}, 'lobo': {}}
    ALL = KIN + ENV + FUSIONS
    for proto in ('L8_c0', 'L8_all', 'L10', 'A2R', 'LOBO'):
        ks = [kk for (pp, kk) in R if pp == proto]
        for m in ALL:
            if proto == 'L10':
                ss = [summ(R[(proto, kk)]['out_' + m], R[(proto, kk)]['y']) for kk in ks]
                out['settings'].setdefault('PB ' + proto, {})[m] = {q: float(np.mean([s[q] for s in ss])) for q in ('acc', 'bacc', 'fa', 'missed')}
            else:
                P = np.concatenate([R[(proto, kk)]['out_' + m] for kk in ks]); yy = np.concatenate([R[(proto, kk)]['y'] for kk in ks])
                out['settings'].setdefault('PB ' + proto, {})[m] = summ(P, yy)
    yL = np.concatenate([R[('LOBO', b)]['y'] for b in single]); bL = np.concatenate([R[('LOBO', b)]['b'] for b in single])
    cL = np.concatenate([R[('LOBO', b)]['c'] for b in single])
    for f in FUSIONS:
        P = np.concatenate([R[('LOBO', b)]['out_' + f] for b in single]); ok = P.argmax(1) == yL; pr = P.argmax(1); h = yL == 0
        for kap in KAP:
            acc = np.concatenate([R[('LOBO', b)]['out_' + f].max(1) >= tau(R[('LOBO', b)]['in_' + f], kap) for b in single])
            out['lobo'].setdefault(f, {})[f'kappa_{kap}'] = {'coverage': float(acc.mean()), 'sel_acc': float(ok[acc].mean()),
                'fa': float((acc & h & (pr != 0)).sum() / h.sum()), 'missed': float((acc & ~h & (pr == 0)).sum() / (~h).sum()), 'cov_H': float(acc[h].mean())}
        out['lobo'][f]['by_condition'] = {c: float(ok[cL == c].mean()) for c in sorted(set(cL))}
        out['lobo'][f]['errors'] = {'H_to_damaged': int(((yL == 0) & (pr != 0)).sum()), 'damaged_to_H': int(((yL > 0) & (pr == 0)).sum()),
                                    'OR_IR': int(((yL > 0) & (pr > 0) & (pr != yL)).sum())}
        out['lobo'][f]['per_bearing'] = {b: float(ok[bL == b].mean()) for b in single}
    pbm = out['lobo']
    for f in FUSIONS:
        if f == 'V1': continue
        for ref in ('V1',):
            dlt = np.array([pbm[f]['per_bearing'][b] - pbm[ref]['per_bearing'][b] for b in single])
            p = float(wilcoxon(dlt).pvalue) if np.any(np.abs(dlt) > 1e-12) else 1.0
            out['lobo'][f]['vs_V1'] = {'better': int((dlt > 0.005).sum()), 'worse': int((dlt < -0.005).sum()), 'p': p}
    for f in FUSIONS:                                     # per-bearing test against the envelope branch alone
        rfb = {b: float((R[('LOBO', b)]['out_RF'].argmax(1) == lab[b]).mean()) for b in single}
        dlt = np.array([pbm[f]['per_bearing'][b] - rfb[b] for b in single])
        out['lobo'][f]['vs_RF'] = {'better': int((dlt > 0.005).sum()), 'worse': int((dlt < -0.005).sum()),
                                   'p': float(wilcoxon(dlt).pvalue) if np.any(np.abs(dlt) > 1e-12) else 1.0}
    out['cw_used'] = {f'{p} {k}': bool(R[(p, k)]['cw_used']) for (p, k) in R if p != 'LOBO'}
    # CT: condition-specific thresholds at kappa 0.7
    for f in FUSIONS:
        acc = []
        for b in single:
            Rb = R[('LOBO', b)]; ci = np.nanmax(Rb['in_' + f], 1); cc = Rb['inc']; co = Rb['out_' + f].max(1)
            t = {c: float(np.quantile(ci[(cc == c) & np.isfinite(ci)], 0.3)) for c in np.unique(cc)}
            acc.append(np.array([co[i] >= t.get(Rb['c'][i], np.nan) for i in range(len(co))]))
        acc = np.concatenate(acc); P = np.concatenate([R[('LOBO', b)]['out_' + f] for b in single]); pr = P.argmax(1); ok = pr == yL; h = yL == 0
        out['lobo'][f]['kappa_0.7_condition_thresholds'] = {'coverage': float(acc.mean()), 'sel_acc': float(ok[acc].mean()),
            'fa': float((acc & h & (pr != 0)).sum() / h.sum()), 'missed': float((acc & ~h & (pr == 0)).sum() / (~h).sum())}
    # SEQ: several consecutive recordings of one bearing
    for f in FUSIONS:
        out['lobo'][f]['seq'] = seq_eval([(R[('LOBO', b)]['out_' + f], R[('LOBO', b)]['y']) for b in single])
    nets = {}
    for pat, tag in ((ROOT + '/checkpoints/phase23/*_LOBO_*_s0.npz', 'p23_'), (ROOT + '/checkpoints/phase24/p2_*_LOBO_*.npz', 'p24_')):
        import glob, re
        for fp in sorted(glob.glob(pat)):
            m_ = re.match(r'(?:p2_)?(.+)_LOBO_(\d+)(?:_s0)?\.npz$', os.path.basename(fp)); nets.setdefault(tag + m_.group(1), []).append(fp)
    out['networks_seq'] = {}
    for nm, fs in nets.items():
        if len(fs) != 29: continue
        parts = []
        for fp in fs:
            z = np.load(fp, allow_pickle=True); P_, y_, b_ = z['P'].astype(float), z['y'].astype(int), z['b'].astype(str)
            for bb in np.unique(b_): parts.append((P_[b_ == bb], y_[b_ == bb]))
        out['networks_seq'][nm] = seq_eval(parts)
    return out


# ------------------------------------------------------------------ external datasets
def protocols(name, d):
    H = d['cls'] == 'H'
    if name == 'cwru':
        groups = [g for g in sorted(set(d['group'][~H])) if g != '028']
        h_tr = H & np.isin(d['cond'], ['0', '1']); h_te = H & np.isin(d['cond'], ['2', '3']); fault = ~H & np.isin(d['group'], groups)
        for g in groups: yield 'LOSO', g, (fault & (d['group'] != g)) | h_tr, (fault & (d['group'] == g)) | h_te
        for g in groups: yield '1SIZE', g, (fault & (d['group'] == g)) | h_tr, (fault & (d['group'] != g)) | h_te
    else:
        tag = ('LOTO', '1TYPE') if name == 'hust' else ('LOPO', '1PROF')
        for g in sorted(set(d['group'])): yield tag[0], g, d['group'] != g, d['group'] == g
        for g in sorted(set(d['group'])): yield tag[1], g, d['group'] == g, d['group'] != g


def notched_cache(name, base):
    path = f'{base}/phase28_{name}_comb.npz'
    if os.path.exists(path) and not a.fresh: return dict(np.load(path, allow_pickle=False))
    import importlib.util
    spec = importlib.util.spec_from_file_location('p10', ROOT + '/phase1/phase10_benchmarks.py'); P10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P10)
    TH = 'Thuan & Hong 2023, Table 1; D = (bore + outer diameter) / 2'          # HUST geometry as phases 20 and 21
    P10.GEOM.update({'HUST_6204': dict(n=8, d=7.6, D=33.5, src=TH), 'HUST_6205': dict(n=9, d=7.8, D=38.5, src=TH),
                     'HUST_6206': dict(n=9, d=9.0, D=46.0, src=TH), 'HUST_6207': dict(n=9, d=11.0, D=53.5, src=TH),
                     'HUST_6208': dict(n=9, d=12.0, D=60.0, src=TH)})
    std = P10.peak_features; P10.peak_features = lambda S, orders, fo: np.r_[comb_separated(S, orders, fo, False), comb_slip(S, orders, fo, False)].astype('float32')
    Xn, rec = [], []
    try:
        for r in P10.LOADERS[name](BENCH):
            segs = P10.recording_features(r['x'], r['fs'], *r['knots'], r['fo']); Xn += [q_[0] for q_ in segs]; rec += [r['file']] * len(segs)
    finally:
        P10.peak_features = std
    out = {'XN': np.array(Xn), 'recording': np.array(rec)}; np.savez_compressed(path, **out)
    print(f'  {name}: comb-separated and slip-consistent features of {len(rec)} segments cached ({el()})', flush=True); return out


def external(name):
    base = f'{BENCH}/phase10_results'
    d = dict(np.load(f'{base}/phase20_{name}_variants.npz', allow_pickle=False)); e = dict(np.load(f'{base}/phase21_{name}_extended.npz', allow_pickle=False))
    assert (e['recording'] == d['recording']).all()
    CLS3 = ['H', 'OR', 'IR']; keep = np.isin(d['cls'], CLS3); y = np.array([CLS3.index(c) if c in CLS3 else -1 for c in d['cls']])
    cond = d['cond'].astype(str); X31 = np.nan_to_num(e['X31'].astype(float))
    Pv = [np.nan_to_num(d[k].astype(float)) for k in ('P_fixed', 'P_sk', 'P_cpw', 'P_cpw_sk')]
    X = {'LRX': X31, 'LRX_C': X31, 'LRM': np.c_[Pv[0], Pv[1], Pv[2], Pv[3], X31[:, 15:]]}; X['LRM_C'] = X['LRM']
    nc = notched_cache(name, base); assert (nc['recording'] == d['recording']).all(), 'segments differ from phase 20'
    X['LRN'] = np.nan_to_num(nc['XN'][:, :40].astype(float)); X['LRN_C'] = X['LRN']; X['LRNS'] = np.nan_to_num(nc['XN'].astype(float)); X['LRNS_C'] = X['LRNS']
    X['RF'] = np.nan_to_num(np.c_[Pv[0], Pv[1], Pv[2], Pv[3], np.log(np.abs(d['T'][:, 2]) + 1e-12), np.log(np.abs(d['T'][:, 4]) + 1e-12)].astype(float)); X['RF_C'] = X['RF']
    FP = {'LRX': make_fp('lr', False), 'LRX_C': make_fp('lr', True), 'LRM': make_fp('lr', False), 'LRM_C': make_fp('lr', True),
          'LRN': make_fp('lr', False), 'LRN_C': make_fp('lr', True), 'LRNS': make_fp('lr', False), 'LRNS_C': make_fp('lr', True),
          'RF': make_fp('rf', False), 'RF_C': make_fp('rf', True)}
    if name == 'cwru': prim = np.where(d['cls'] == 'H', np.char.add('c', cond), np.char.add('g', d['group'].astype(str))); sec = cond
    elif name == 'hust': prim, sec = d['group'].astype(str), cond
    else: prim, sec = d['group'].astype(str), d['recording'].astype(str)
    res = {}; used = {}
    for proto, fold, tr, te in protocols(name, d):
        tr, te = tr & keep, te & keep
        if not tr.any() or not te.any() or len(set(y[tr])) < 3: continue
        def run():
            tri, tei = np.where(tr)[0], np.where(te)[0]
            out = {m: FP[m](X[m], y, cond, tri, tei) for m in KIN + ENV}
            g = None
            for key in (prim[tri], sec[tri]):
                if all(len(set(key[y[tri] == c])) >= 2 for c in range(3)): g = key; break
            inn = {m: oof(FP[m], X[m], y, cond, tri, g) for m in KIN + ENV} if g is not None else None
            Fo, Fi = fuse(out, inn, y[tri])
            r = {'y': y[tei], 'cw_used': np.array(cw_valid(y, cond, tri, tei)), 'has_inner': np.array(inn is not None)}
            for f in FUSIONS:
                r['out_' + f] = Fo[f]
                if inn is not None: r['in_' + f] = Fi[f]
            for m in KIN + ENV:
                r['out_' + m] = out[m]
                if inn is not None: r['inb_' + m] = inn[m]
            return r
        need = ['out_' + m for m in KIN + ENV]
        r = ck(f'{name}_{proto}_{fold}', run, need)
        if bool(r['has_inner']) and not all('inb_' + m in r for m in KIN + ENV):
            print(f'    {name}_{proto}_{fold}: inner branch posteriors missing; recomputed', flush=True)
            r = run(); np.savez_compressed(f'{CK}/{name}_{proto}_{fold}.npz', **r)
        r = refuse(r, y[np.where(tr)[0]], bool(r['has_inner'])); res.setdefault(proto, []).append(r); used.setdefault(proto, []).append(bool(r['cw_used']))
        print(f'  {name} {proto} {fold}  ({el()})', flush=True)
    out = {}
    for proto, rs in res.items():
        yy = np.concatenate([r['y'] for r in rs]); key = f'{name.upper()} {proto}'; out[key] = {}
        for m in KIN + ENV + FUSIONS: out[key][m] = summ(np.concatenate([r['out_' + m] for r in rs]), yy)
        for f in FUSIONS:
            P = np.concatenate([r['out_' + f] for r in rs]); ok = P.argmax(1) == yy
            acc = np.concatenate([r['out_' + f].max(1) >= (tau(r['in_' + f], 0.7) if bool(r['has_inner']) else -1) for r in rs])
            out[key][f]['k07'] = {'coverage': float(acc.mean()), 'sel_acc': float(ok[acc].mean()) if acc.any() else None}
        out[key]['cw_used_folds'] = used[proto]
    return out


# ------------------------------------------------------------------ run, print, adoption rule
def f1(x): return '  -  ' if x is None else f'{100 * x:5.1f}'


if a.part in ('pb', 'all'):
    RES['paderborn'] = paderborn(); json.dump(RES, open(ROOT + '/phase28_results.json', 'w'), indent=1)
if a.part in ('ext', 'all'):
    RES['external'] = {}
    for nm in ('cwru', 'hust', 'ottawa'):
        try: RES['external'].update(external(nm))
        except FileNotFoundError as err: print(f'  {nm}: cache not found ({err}); skipped')
        json.dump(RES, open(ROOT + '/phase28_results.json', 'w'), indent=1)

SET = {}
if 'paderborn' in RES: SET.update(RES['paderborn']['settings'])
if 'external' in RES: SET.update({k: v for k, v in RES['external'].items()})
names = list(SET)
print('\n=== accuracy / false alarms / missed faults / balanced accuracy (%) per setting   V1 = ReliSense v1')
for s in names:
    print(f'  {s}' + ('' if s.startswith('PB') else f"   [condition-wise used in folds: {SET[s].get('cw_used_folds')}]"))
    for m in KIN + ENV + FUSIONS:
        r = SET[s][m]; line = f'    {m:8} {f1(r["acc"])} / {f1(r["fa"])} / {f1(r["missed"])} / {f1(r["bacc"])}'
        if 'k07' in r and r['k07']['sel_acc'] is not None: line += f'   | k0.7 cov {f1(r["k07"]["coverage"])} sel {f1(r["k07"]["sel_acc"])}'
        print(line)
if 'paderborn' in RES:
    L = RES['paderborn']['lobo']
    print('\n=== LOBO abstention: coverage / selective accuracy / false alarms / missed / healthy coverage (%)')
    for f in FUSIONS:
        print(f'  {f:8} ' + ' | '.join(f'k{k}: {f1(L[f][f"kappa_{k}"]["coverage"])} / {f1(L[f][f"kappa_{k}"]["sel_acc"])} / {f1(L[f][f"kappa_{k}"]["fa"])} / {f1(L[f][f"kappa_{k}"]["missed"])} / {f1(L[f][f"kappa_{k}"]["cov_H"])}' for k in KAP))
    print('\n=== LOBO accuracy by operating condition (%)')
    for f in FUSIONS: print(f'  {f:8} ' + '  '.join(f'{c} {f1(v)}' for c, v in L[f]['by_condition'].items()) + f"   errors H->dmg / dmg->H / OR<->IR: {L[f]['errors']['H_to_damaged']} / {L[f]['errors']['damaged_to_H']} / {L[f]['errors']['OR_IR']}")
    print('\n=== LOBO per bearing (%), weakest bearings of v1')
    for b in ('K001', 'K002', 'K004', 'K005', 'KA09', 'KA15', 'KA22', 'KA30', 'KI05', 'KI07', 'KI14', 'KI17'):
        print(f'  {b}: ' + '  '.join(f'{f} {f1(L[f]["per_bearing"][b])}' for f in FUSIONS))
    print('\n=== LOBO kappa 0.7, pooled vs condition-specific thresholds: coverage / selective accuracy / false alarms / missed (%)')
    for f in FUSIONS:
        a_, b_ = L[f]['kappa_0.7'], L[f]['kappa_0.7_condition_thresholds']
        print(f'  {f:8} pooled {f1(a_["coverage"])} / {f1(a_["sel_acc"])} / {f1(a_["fa"])} / {f1(a_["missed"])}   condition-specific {f1(b_["coverage"])} / {f1(b_["sel_acc"])} / {f1(b_["fa"])} / {f1(b_["missed"])}')
    print('\n=== LOBO decisions from k consecutive recordings (accuracy / false alarms / missed, %) and sequential rule (mean recordings)')
    def sline(d): return '  '.join(f'k{k}: {f1(d[k]["acc"])}/{f1(d[k]["fa"])}/{f1(d[k]["missed"])}' for k in ('1', '2', '5', '10')) + f'  seq: {f1(d["seq"]["acc"])}/{f1(d["seq"]["fa"])}/{f1(d["seq"]["missed"])} ({d["seq"]["mean_recordings"]:.1f})'
    for f in FUSIONS: print(f'  {f:8} ' + sline(L[f]['seq']))
    for nm, d in RES['paderborn'].get('networks_seq', {}).items(): print(f'  {nm:16} ' + sline(d))
    print('\nper-bearing Wilcoxon against V1 and against the envelope branch RF (29 bearings):')
    for f in FUSIONS:
        s = f'  {f:8} vs RF: better {L[f]["vs_RF"]["better"]}, worse {L[f]["vs_RF"]["worse"]}, p = {L[f]["vs_RF"]["p"]:.3f}'
        if 'vs_V1' in L[f]: s += f'   | vs V1: better {L[f]["vs_V1"]["better"]}, worse {L[f]["vs_V1"]["worse"]}, p = {L[f]["vs_V1"]["p"]:.3f}'
        print(s)
if len(SET) == 11 and 'paderborn' in RES:
    mb = {f: float(np.mean([SET[s][f]['bacc'] for s in names])) for f in FUSIONS}
    print('\n=== mean balanced accuracy over the 11 settings (%):', {f: round(100 * v, 2) for f, v in mb.items()})
    cand = [f for f in FUSIONS if f != 'V1']; best = max(cand, key=lambda f: mb[f]); L = RES['paderborn']['lobo']
    c1 = mb[best] > mb['V1']; c2 = SET['PB LOBO'][best]['acc'] >= SET['PB LOBO']['V1']['acc']
    c3 = L[best]['kappa_0.7']['sel_acc'] >= L['V1']['kappa_0.7']['sel_acc'] - 0.005
    RES['adoption'] = {'best': best, 'mean_bacc': mb, 'criteria': [c1, c2, c3], 'adopted': bool(c1 and c2 and c3)}
    print(f'adoption rule: best candidate {best}; (i) mean balanced accuracy higher than V1: {c1}; (ii) LOBO accuracy not lower: {c2}; '
          f'(iii) LOBO selective accuracy at kappa 0.7 at most 0.5 points lower: {c3}  ->  ADOPTED: {c1 and c2 and c3}')
    json.dump(RES, open(ROOT + '/phase28_results.json', 'w'), indent=1)
print(f'\nwritten {ROOT}/phase28_results.json  ({el()})')
