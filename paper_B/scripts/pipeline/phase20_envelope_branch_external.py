"""Phase 20 (v1.0): the envelope branch of ReliSense on the three further rigs (CWRU, HUST, Ottawa), CPU, about 10-30 minutes.

Purpose: phase 17 combined, on Paderborn, the kinematic branch (15 features, fixed 2-12 kHz band, logistic regression)
with an envelope branch (random forest on the same 15 kinematic features read under four demodulation variants plus
log kurtosis and log crest factor). Phase 18 could only pair the kinematic branch with the time + band random forest on
the further rigs, because their feature files hold the fixed-band features only. This script computes the four variants
on the further rigs, so that the two-branch method is defined identically on every dataset.

Pre-specified before any result is seen; every model is reported.
  variants   fixed   band 2-12 kHz (upper edge 0.45 fs), as phase 10
             sk      band of largest envelope kurtosis (dyadic kurtogram, levels 1-4, 500 Hz to min(12 kHz, 0.45 fs))
             cpw     cepstral pre-whitening, then the fixed band
             cpw_sk  cepstral pre-whitening, then the kurtogram band          (all as physics64.py of phase 1)
  segments   exactly those of phase 10 (50 revolutions, 64 samples per revolution, fault orders of each bearing type);
             the segmentation code of phase 10 is reused unchanged, only the band of the envelope differs.
  models     LR     kinematic branch: fixed-band 15 features, standardize + logistic regression C = 0.3   (= phase 10)
             RFenv  envelope branch: 4 x 15 features + log kurtosis + log crest, random forest 500 trees, seed 0 (= phase 17)
             ENS    ReliSense two-branch: mean of the LR and RFenv class probabilities
             also, for reference, the phase 10 baselines: band energies + LR, time + band + RF (300 trees)
  protocols  those of phase 10 (CWRU LOSO / 1SIZE, HUST LOTO / 1TYPE, Ottawa LOPO / 1PROF), three-class H / OR / IR.
Check: the fixed-variant features recomputed here are compared with the phase 10 cache (<dataset>_features.npz);
the maximum difference is printed. The recomputed features are used throughout.
Uses the loaders and segmentation of phase10_benchmarks.py (default: <study>/phase1/phase10_benchmarks.py). The HUST
geometry is set here to the values of Thuan & Hong (2023), Table 1, as in phase 10 v1.1, whatever copy is imported.
Output: <out>/phase20_results.json, <out>/phase20_<dataset>_variants.npz (cache; kept for later use)
Colab:
  !python /content/drive/MyDrive/ReliSense_study/phase20_envelope_branch_external.py \
      --root /content/drive/MyDrive/ReliSense_study/benchmarks
"""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True, help='folder holding CWRU/, HUST/, Ottawa/')
ap.add_argument('--out', default=None, help='default: <root>/phase10_results'); ap.add_argument('--datasets', default='cwru,hust,ottawa')
ap.add_argument('--phase10', default=os.path.join(HERE, 'phase1', 'phase10_benchmarks.py')); ap.add_argument('--trees', type=int, default=500)
ap.add_argument('--fresh', action='store_true')
a = ap.parse_args(); OUT = Path(a.out or os.path.join(a.root, 'phase10_results')); OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
spec = importlib.util.spec_from_file_location('p10', a.phase10); P10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P10)
SRC_TH = 'Thuan & Hong 2023, Table 1; D = (bore + outer diameter) / 2'
P10.GEOM.update({'HUST_6204': dict(n=8, d=7.6, D=33.5, src=SRC_TH), 'HUST_6205': dict(n=9, d=7.8, D=38.5, src=SRC_TH),
                 'HUST_6206': dict(n=9, d=9.0, D=46.0, src=SRC_TH), 'HUST_6207': dict(n=9, d=11.0, D=53.5, src=SRC_TH),
                 'HUST_6208': dict(n=9, d=12.0, D=60.0, src=SRC_TH)})
VARIANTS = ('fixed', 'sk', 'cpw', 'cpw_sk'); CLS3 = ['H', 'OR', 'IR']


def cepstral_prewhiten(x):
    X = np.fft.rfft(x - x.mean()); return np.fft.irfft(X / (np.abs(X) + 1e-12), n=len(x))


def band_envelope(X, f, lo, hi, n):
    full = np.zeros(n, dtype=complex); keep = (f >= lo) & (f < hi); full[:len(f)][keep] = 2 * X[keep]; return np.fft.ifft(full)


def kurtogram_band(X, f, n, fmax, fmin=500.0, levels=(1, 2, 3, 4)):
    best = (-np.inf, (P10.BAND[0], fmax))
    for lev in levels:
        bw = fmax / 2 ** lev
        for k in range(2 ** lev):
            lo, hi = k * bw, (k + 1) * bw
            if lo < fmin: continue
            p = np.abs(band_envelope(X, f, lo, hi, n)) ** 2; kur = np.mean(p ** 2) / (np.mean(p) ** 2 + 1e-30) - 2
            if kur > best[0]: best = (kur, (lo, hi))
    return best[1]


def variant_envelopes(x, fs):
    """Squared envelope of each variant on the full recording (same brick-wall filter as phase 10)."""
    x = np.asarray(x, float); n = len(x); f = np.fft.rfftfreq(n, 1 / fs); fmax = min(12000.0, 0.45 * fs)
    out, bands = {}, {}
    for v in VARIANTS:
        X = np.fft.rfft(cepstral_prewhiten(x)) if v.startswith('cpw') else np.fft.rfft(x - x.mean())
        lo, hi = kurtogram_band(X, f, n, fmax) if v.endswith('sk') else (P10.BAND[0], min(P10.BAND[1], 0.45 * fs))
        out[v] = np.abs(band_envelope(X, f, lo, hi, n)) ** 2; bands[v] = (lo, hi)
    return out, bands


def build(name):
    path = OUT / f'phase20_{name}_variants.npz'
    if path.exists() and not a.fresh: return dict(np.load(path, allow_pickle=False))
    recs = P10.LOADERS[name](a.root); cols = {k: [] for k in ('P_fixed', 'P_sk', 'P_cpw', 'P_cpw_sk', 'T', 'S', 'cls', 'unit', 'group', 'cond', 'recording', 'band_sk', 'band_cpw_sk')}
    orig = P10.sq_envelope
    try:
        for r in recs:
            env, bands = variant_envelopes(r['x'], r['fs']); per = {}
            for v in VARIANTS:
                P10.sq_envelope = lambda x, fs, _e=env[v]: _e          # reuse the phase 10 segmentation unchanged
                per[v] = P10.recording_features(r['x'], r['fs'], *r['knots'], r['fo'])
            n = len(per['fixed']); assert all(len(per[v]) == n for v in VARIANTS)
            for s in range(n):
                for v in VARIANTS: cols[f'P_{v}'].append(per[v][s][0])
                cols['T'].append(per['fixed'][s][1]); cols['S'].append(per['fixed'][s][2])
                for k in ('cls', 'unit', 'group', 'cond'): cols[k].append(r[k])
                cols['recording'].append(r['file']); cols['band_sk'].append(bands['sk']); cols['band_cpw_sk'].append(bands['cpw_sk'])
            print(f'  {name} {r["file"]}: {n} segments; kurtogram band {bands["sk"][0]:.0f}-{bands["sk"][1]:.0f} Hz, '
                  f'after pre-whitening {bands["cpw_sk"][0]:.0f}-{bands["cpw_sk"][1]:.0f} Hz  ({time.time() - t0:.0f} s)', flush=True)
    finally:
        P10.sq_envelope = orig
    d = {k: np.array(v) for k, v in cols.items()}; np.savez_compressed(path, **d); return d


def lr(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))


def proba(m, X):
    P = np.zeros((len(X), 3)); P[:, m.classes_] = m.predict_proba(X); return P


def evaluate(name, d):
    keep = np.isin(d['cls'], CLS3); y = np.array([CLS3.index(c) if c in CLS3 else -1 for c in d['cls']])
    X_lr = d['P_fixed'].astype(float)
    X_rf = np.c_[d['P_fixed'], d['P_sk'], d['P_cpw'], d['P_cpw_sk'], np.log(np.abs(d['T'][:, 2]) + 1e-12), np.log(np.abs(d['T'][:, 4]) + 1e-12)].astype(float)
    X_be, X_tb = d['S'].astype(float), np.c_[d['T'], d['S']].astype(float)
    X_lr, X_rf, X_tb = np.nan_to_num(X_lr), np.nan_to_num(X_rf), np.nan_to_num(X_tb)
    res = {}
    for proto, fold, tr, te in P10.protocols(name, d):
        tr, te = tr & keep, te & keep
        if not tr.any() or not te.any() or len(set(y[tr])) < 3: print(f'  {name} {proto} {fold}: skipped'); continue
        pl = proba(lr().fit(X_lr[tr], y[tr]), X_lr[te])
        pr = proba(RandomForestClassifier(a.trees, random_state=0, n_jobs=-1).fit(X_rf[tr], y[tr]), X_rf[te])
        P = {'LR': pl, 'RFenv': pr, 'ENS': (pl + pr) / 2,
             'band energies + LR': proba(lr().fit(X_be[tr], y[tr]), X_be[te]),
             'time + band + RF': proba(RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X_tb[tr], y[tr]), X_tb[te])}
        for m, p in P.items():
            r = res.setdefault(proto, {}).setdefault(m, {'y': [], 'p': [], 'unit': [], 'fold': []})
            r['y'] += y[te].tolist(); r['p'] += p.argmax(1).tolist(); r['unit'] += d['unit'][te].tolist(); r['fold'] += [str(fold)] * int(te.sum())
    out = {}
    for proto, ms in res.items():
        for m, r in ms.items():
            yy, pp, uu, ff = map(np.array, (r['y'], r['p'], r['unit'], r['fold'])); h = yy == 0
            out.setdefault(proto, {})[m] = {'acc': float((yy == pp).mean()), 'false_alarm': float((pp[h] != 0).mean()) if h.any() else None,
                                            'missed_fault': float((pp[~h] == 0).mean()),
                                            'fold_acc': {f: float((yy[ff == f] == pp[ff == f]).mean()) for f in sorted(set(ff))},
                                            'per_unit': {u: float((yy[uu == u] == pp[uu == u]).mean()) for u in sorted(set(uu))}}
    return out


R = {}
for name in [s.strip().lower() for s in a.datasets.split(',') if s.strip()]:
    print(f'\n=== {name}: variants', flush=True); d = build(name)
    c = OUT / f'{name}_features.npz'
    if c.exists():
        old = np.load(c, allow_pickle=False)
        if old['P'].shape == d['P_fixed'].shape:
            diff = float(np.abs(old['P'] - d['P_fixed']).max()); print(f'  check against the phase 10 cache: max |difference| of the fixed-band features = {diff:.2e}')
        else:
            diff = None; print(f'  check: phase 10 cache has {old["P"].shape[0]} segments, recomputed {d["P_fixed"].shape[0]}')
    else:
        diff = None; print('  check: no phase 10 cache found')
    print(f'=== {name}: evaluation', flush=True)
    R[name] = {'check_max_abs_diff_fixed': diff, 'results': evaluate(name, d)}
    json.dump(R, open(OUT / 'phase20_results.json', 'w'), indent=1)

print('\nThree-class segment accuracy (%) / false alarms on healthy segments (%)')
M = ('LR', 'RFenv', 'ENS', 'band energies + LR', 'time + band + RF')
print(f"{'':14}" + ''.join(f'{m:>22}' for m in M))
for name, rr in R.items():
    for proto, ms in rr['results'].items():
        row = f'{name + " " + proto:14}'
        for m in M:
            r = ms[m]; fa = '  -  ' if r['false_alarm'] is None else f'{100 * r["false_alarm"]:5.1f}'
            row += f'{100 * r["acc"]:15.1f} / {fa}'
        print(row)
print(f'\nwritten {OUT}/phase20_results.json  ({time.time() - t0:.0f} s)')
