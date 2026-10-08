"""Phase 21 (v1.0): the extended kinematic branch of phase 19 on the three further rigs (CWRU, HUST, Ottawa), CPU, about 5-10 minutes.

Fixed before any result is seen. The final method chosen after phases 17-20 is
  ReliSense = mean of (a) the extended kinematic branch LRX and (b) the envelope branch RFenv, with training-only abstention.
Phase 19 tested it on Paderborn; this script tests the same method, unchanged, on the further rigs. Every model is reported.
  LRX    15 fixed-band kinematic features + 16 extended features, standardize + logistic regression C = 0.3
         extended = h*BPFI +- 1 (h = 1, 2), h*BPFO +- FTF (h = 1, 2), h*BSF +- FTF (h = 1, 2), 4*BSF, 5*BSF, 4*FTF, 5*FTF,
         read as in phase 10: peak in +-max(2 %, 1.5 bins), background = median within +-0.4 order (window excluded)
  RFenv  envelope branch of phase 20 (read from <out>/phase20_<dataset>_variants.npz; same segments)
  ENSX   mean of LRX and RFenv;  also LR, ENS (phase 20) for reference
Segments, envelope (fixed band) and protocols are those of phase 10, reused unchanged; the fault orders are those of each
bearing type (HUST geometry of Thuan & Hong 2023, as phase 20). Check: the 15 standard features must equal P_fixed of phase 20.
Output: <out>/phase21_results.json, <out>/phase21_<dataset>_extended.npz
Colab:
  !python /content/drive/MyDrive/ReliSense_study/phase21_extended_external.py --root /content/drive/MyDrive/ReliSense_study/benchmarks
"""
import argparse, importlib.util, json, os, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--out', default=None)
ap.add_argument('--datasets', default='cwru,hust,ottawa'); ap.add_argument('--phase10', default=os.path.join(HERE, 'phase1', 'phase10_benchmarks.py'))
ap.add_argument('--trees', type=int, default=500); ap.add_argument('--fresh', action='store_true')
a = ap.parse_args(); OUT = Path(a.out or os.path.join(a.root, 'phase10_results')); t0 = time.time()
spec = importlib.util.spec_from_file_location('p10', a.phase10); P10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P10)
SRC_TH = 'Thuan & Hong 2023, Table 1; D = (bore + outer diameter) / 2'
P10.GEOM.update({'HUST_6204': dict(n=8, d=7.6, D=33.5, src=SRC_TH), 'HUST_6205': dict(n=9, d=7.8, D=38.5, src=SRC_TH),
                 'HUST_6206': dict(n=9, d=9.0, D=46.0, src=SRC_TH), 'HUST_6207': dict(n=9, d=11.0, D=53.5, src=SRC_TH),
                 'HUST_6208': dict(n=9, d=12.0, D=60.0, src=SRC_TH)})
CLS3 = ['H', 'OR', 'IR']; std_peaks = P10.peak_features


def centres(fo):
    c = [h * fo['BPFI'] + s * fo['shaft'] for h in (1, 2) for s in (-1, 1)]
    c += [h * fo['BPFO'] + s * fo['FTF'] for h in (1, 2) for s in (-1, 1)]
    c += [h * fo['BSF'] + s * fo['FTF'] for h in (1, 2) for s in (-1, 1)]
    return c + [4 * fo['BSF'], 5 * fo['BSF'], 4 * fo['FTF'], 5 * fo['FTF']]


def peaks_31(S, orders, fo):
    dO = orders[1] - orders[0]; row = []
    for c in centres(fo):
        w = max(0.02 * c, 1.5 * dO); band = np.abs(orders - c) <= w; bg = (np.abs(orders - c) <= 0.4) & ~band
        row.append(0.0 if (not band.any() or not bg.any() or c > orders[-1]) else np.log((S[band].max() + 1e-12) / (np.median(S[bg]) + 1e-12)))
    return np.r_[std_peaks(S, orders, fo), np.array(row, 'float32')].astype('float32')


def build(name):
    path = OUT / f'phase21_{name}_extended.npz'
    if path.exists() and not a.fresh: return dict(np.load(path, allow_pickle=False))
    recs = P10.LOADERS[name](a.root); X, rec = [], []
    P10.peak_features = peaks_31
    try:
        for r in recs:
            segs = P10.recording_features(r['x'], r['fs'], *r['knots'], r['fo'])
            X += [s[0] for s in segs]; rec += [r['file']] * len(segs)
            print(f'  {name} {r["file"]}: {len(segs)} segments  ({time.time() - t0:.0f} s)', flush=True)
    finally:
        P10.peak_features = std_peaks
    d = {'X31': np.array(X), 'recording': np.array(rec)}; np.savez_compressed(path, **d); return d


def lr(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))


def proba(m, X):
    P = np.zeros((len(X), 3)); P[:, m.classes_] = m.predict_proba(X); return P


R = {}
for name in [s.strip().lower() for s in a.datasets.split(',') if s.strip()]:
    print(f'\n=== {name}', flush=True); e = build(name); d = dict(np.load(OUT / f'phase20_{name}_variants.npz', allow_pickle=False))
    assert (e['recording'] == d['recording']).all(), 'segments differ from phase 20'
    diff = float(np.abs(e['X31'][:, :15] - d['P_fixed']).max()); print(f'  check: max |difference| of the 15 standard features against phase 20 = {diff:.2e}')
    keep = np.isin(d['cls'], CLS3); y = np.array([CLS3.index(c) if c in CLS3 else -1 for c in d['cls']])
    X_lr = np.nan_to_num(d['P_fixed'].astype(float)); X_x = np.nan_to_num(e['X31'].astype(float))
    X_rf = np.nan_to_num(np.c_[d['P_fixed'], d['P_sk'], d['P_cpw'], d['P_cpw_sk'], np.log(np.abs(d['T'][:, 2]) + 1e-12), np.log(np.abs(d['T'][:, 4]) + 1e-12)].astype(float))
    res = {}
    for proto, fold, tr, te in P10.protocols(name, d):
        tr, te = tr & keep, te & keep
        if not tr.any() or not te.any() or len(set(y[tr])) < 3: continue
        pl = proba(lr().fit(X_lr[tr], y[tr]), X_lr[te]); px = proba(lr().fit(X_x[tr], y[tr]), X_x[te])
        pr = proba(RandomForestClassifier(a.trees, random_state=0, n_jobs=-1).fit(X_rf[tr], y[tr]), X_rf[te])
        for m, p in {'LR': pl, 'LRX': px, 'RFenv': pr, 'ENS': (pl + pr) / 2, 'ENSX': (px + pr) / 2}.items():
            r = res.setdefault(proto, {}).setdefault(m, {'y': [], 'p': [], 'unit': []})
            r['y'] += y[te].tolist(); r['p'] += p.argmax(1).tolist(); r['unit'] += d['unit'][te].tolist()
    out = {}
    for proto, ms in res.items():
        for m, r in ms.items():
            yy, pp, uu = map(np.array, (r['y'], r['p'], r['unit'])); h = yy == 0
            out.setdefault(proto, {})[m] = {'acc': float((yy == pp).mean()), 'false_alarm': float((pp[h] != 0).mean()) if h.any() else None,
                                            'missed_fault': float((pp[~h] == 0).mean()),
                                            'per_unit': {u: float((yy[uu == u] == pp[uu == u]).mean()) for u in sorted(set(uu))}}
    R[name] = {'check_max_abs_diff_standard': diff, 'results': out}
    json.dump(R, open(OUT / 'phase21_results.json', 'w'), indent=1)

M = ('LR', 'LRX', 'RFenv', 'ENS', 'ENSX')
print('\nThree-class segment accuracy (%) / false alarms on healthy segments (%)')
print(f"{'':14}" + ''.join(f'{m:>16}' for m in M))
for name, rr in R.items():
    for proto, ms in rr['results'].items():
        row = f'{name + " " + proto:14}'
        for m in M:
            r = ms[m]; fa = '  -  ' if r['false_alarm'] is None else f'{100 * r["false_alarm"]:5.1f}'
            row += f'{100 * r["acc"]:9.1f} / {fa}'
        print(row)
print(f'\nwritten {OUT}/phase21_results.json  ({time.time() - t0:.0f} s)')
