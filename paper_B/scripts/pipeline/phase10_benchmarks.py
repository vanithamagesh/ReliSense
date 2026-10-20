"""Phase 10 (v1.0): the ReliSense method on three more bearing datasets (CWRU, HUST, Ottawa), CPU only.

Everything below is fixed before any benchmark result is seen ("locked" external test):
  * segment      50 shaft revolutions (order resolution 0.02, as the 0.02-order grid of Paderborn)
  * envelope     band 2-12 kHz (upper edge capped at 0.45 fs), squared envelope, block mean to about 8 kHz,
                 resampled to 64 samples per shaft revolution (computed order tracking; for constant speed the
                 shaft angle is speed x time, for Ottawa it comes from the encoder)
  * features P   15 peak-to-background ratios: harmonics 1-3 of BPFO, BPFI, BSF, FTF and shaft, search band
                 +-2 % (at least 1.5 bins), background = median within +-0.4 order (= +-10 Hz at 25 Hz, as on
                 Paderborn); fault orders from the geometry of each bearing type
  * features T   8 vibration and 3 envelope statistics;  S  32 relative log band energies 0-4 kHz
  * methods      ReliSense = P + standardise + logistic regression (C = 0.3)
                 band energies = S + same logistic regression;  time+band RF = T+S + random forest (300 trees)
  * classes      primary 3-class H / OR / IR (as Paderborn); secondary 4-class adds ball (B). Combined faults
                 are left out. If a dataset has no healthy files, the task becomes OR / IR / B.
  * split rule   every split is by physical bearing where the dataset allows it (see each loader); scaling and
                 fitting use training data only; nothing is tuned on test data.

Protocols (each dataset says honestly what it can and cannot test):
  CWRU   one physical bearing per fault type and size; ONE healthy bearing.
           LOSO   leave one fault size out (train 2 sizes, test the third) -> unseen faulty bearings
           1SIZE  train on 1 size only (1 faulty bearing per class), test the other 2 sizes
           Healthy: loads 0-1 train, loads 2-3 test in every fold (same bearing -> its false-alarm rate is
           measured on a SEEN bearing at unseen loads; reported, but not an unseen-bearing figure).
  HUST   one physical bearing per (defect, bearing type 6204-6208).
           LOTO   leave one bearing type out (train 4 types, test the 5th) -> unseen bearings AND unseen geometry
           1TYPE  train on 1 bearing type, test the other 4
  Ottawa (Huang & Baddour 2018, time-varying speed) one bearing per health condition.
           LOPO   leave one speed profile out (A-D) -> unseen speed profile, NOT an unseen bearing
           1PROF  train on 1 speed profile, test the other 3

Run (Colab, CPU is enough):
  python phase10_benchmarks.py --root /content/drive/MyDrive/ReliSense_study/benchmarks --datasets cwru,hust,ottawa
  --check  only reads a few files per dataset and prints the speed / encoder / geometry checks, then stops.
Outputs in --out: <dataset>_features.npz (cache; delete to rebuild), phase10_results.json, phase10_summary.txt.
"""
from __future__ import annotations
import argparse, glob, json, os, re, time
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEG_REVS, PER_REV, BAND = 50, 64, (2000.0, 12000.0)
HARMS = (1, 2, 3); ORDER_KEYS = ('BPFO', 'BPFI', 'BSF', 'FTF', 'shaft')
FEATURE_NAMES = [f'{k}_h{h}' for k in ORDER_KEYS for h in HARMS]
CLS3 = ['H', 'OR', 'IR']; CLS4 = ['H', 'OR', 'IR', 'B']

# ----------------------------------------------------------------------------------------- geometry
GEOM = {  # n balls, ball diameter d, pitch diameter D (same unit), contact angle 0
    'CWRU_6205': dict(n=9, d=0.3126, D=1.537, src='CWRU Bearing Data Center, drive-end SKF 6205-2RS JEM (inches)'),
    'OTTAWA_ER16K': dict(n=9, d=7.94, D=38.52, src='Huang & Baddour 2018, ER16K (mm); BPFO 3.57, BPFI 5.43'),
    # HUST: catalogue values of standard 62xx deep-groove bearings. UNVERIFIED for the HUST rig:
    # replace with the table of Thuan & Hong (2023) if it gives different values.
    'HUST_6204': dict(n=8, d=7.938, D=33.5, src='catalogue, UNVERIFIED'),
    'HUST_6205': dict(n=9, d=7.938, D=39.04, src='catalogue, UNVERIFIED'),
    'HUST_6206': dict(n=9, d=9.525, D=46.5, src='catalogue, UNVERIFIED'),
    'HUST_6207': dict(n=9, d=11.112, D=53.5, src='catalogue, UNVERIFIED'),
    'HUST_6208': dict(n=9, d=12.7, D=60.0, src='catalogue, UNVERIFIED'),
}


def fault_orders(g):
    q = g['d'] / g['D']; n = g['n']
    return {'BPFO': n / 2 * (1 - q), 'BPFI': n / 2 * (1 + q), 'BSF': g['D'] / (2 * g['d']) * (1 - q ** 2),
            'FTF': 0.5 * (1 - q), 'shaft': 1.0}


# ----------------------------------------------------------------------------------------- signal processing
def sq_envelope(x, fs):
    """Squared envelope of the band BAND (brick-wall filter in the frequency domain, as physics64.py)."""
    x = np.asarray(x, float); n = len(x); X = np.fft.rfft(x - x.mean()); f = np.fft.rfftfreq(n, 1 / fs)
    full = np.zeros(n, complex); keep = (f >= BAND[0]) & (f < min(BAND[1], 0.45 * fs))
    full[:len(f)][keep] = 2 * X[keep]
    return np.abs(np.fft.ifft(full)) ** 2


def block_mean(e, k):
    m = len(e) // k
    return e[:m * k].reshape(m, k).mean(1)


def peak_features(S, orders, fo):
    dO = orders[1] - orders[0]; row = []
    for k in ORDER_KEYS:
        for h in HARMS:
            c = h * fo[k]; w = max(0.02 * c, 1.5 * dO)
            band = np.abs(orders - c) <= w; bg = (np.abs(orders - c) <= 0.4) & ~band
            if not band.any() or not bg.any() or c > orders[-1]:
                row.append(0.0); continue
            row.append(np.log((S[band].max() + 1e-12) / (np.median(S[bg]) + 1e-12)))
    return np.array(row, 'float32')


def time_stats(s, env):
    s = s - s.mean(); rms = s.std() + 1e-12; a = np.abs(s); pk = a.max()
    e = env - env.mean(); er = e.std() + 1e-12
    return np.array([np.log(rms), np.log(s.max() - s.min() + 1e-12), (s ** 4).mean() / rms ** 4,
                     (s ** 3).mean() / rms ** 3, pk / rms, rms / (a.mean() + 1e-12), pk / (a.mean() + 1e-12),
                     pk / (np.sqrt(a).mean() ** 2 + 1e-12),
                     np.log(np.sqrt(np.mean(env)) + 1e-12), (e ** 4).mean() / er ** 4, np.abs(e).max() / er], 'float32')


def band_energies(s, fs, nb=32, fmax=4000.0):
    P = np.abs(np.fft.rfft(s - s.mean())) ** 2; f = np.fft.rfftfreq(len(s), 1 / fs)
    edges = np.linspace(0, fmax, nb + 1)
    e = np.array([P[(f >= edges[i]) & (f < edges[i + 1])].sum() for i in range(nb)]) + 1e-20
    return np.log(e / e.sum()).astype('float32')


def recording_features(x, fs, t_knots, rev_knots, fo):
    """All segments of one recording. t_knots/rev_knots: shaft angle (revolutions) at given times (s)."""
    x = np.asarray(x, float)
    e = sq_envelope(x, fs); dec = max(1, int(round(fs / 8000))); e = block_mean(e, dec); fe = fs / dec
    te = (np.arange(len(e)) + 0.5) / fe
    rev_e = np.interp(te, t_knots, rev_knots)
    ok = (te >= t_knots[0]) & (te <= t_knots[-1]); te, e, rev_e = te[ok], e[ok], rev_e[ok]
    n_seg = int((rev_e[-1] - rev_e[0]) // SEG_REVS)
    out = []
    for s in range(n_seg):
        r0 = rev_e[0] + s * SEG_REVS
        grid = r0 + np.arange(SEG_REVS * PER_REV) / PER_REV
        ea = np.interp(grid, rev_e, e); ea = (ea - ea.mean()) * np.hanning(len(ea))
        S = np.abs(np.fft.rfft(ea)); orders = np.fft.rfftfreq(len(ea), 1 / PER_REV)
        t0, t1 = np.interp([r0, r0 + SEG_REVS], rev_knots, t_knots)
        i0, i1 = int(t0 * fs), int(t1 * fs)
        seg = x[i0:i1]; envseg = e[(te >= t0) & (te < t1)]
        out.append((peak_features(S, orders, fo), time_stats(seg, envseg), band_energies(seg, fs),
                    float(SEG_REVS / (t1 - t0))))
    return out


def const_speed_knots(n, fs, fr):
    T = n / fs
    return np.array([0.0, T]), np.array([0.0, fr * T])


# ----------------------------------------------------------------------------------------- loaders
def _mat(path):
    return loadmat(path)


CWRU_MAP = {97: ('H', '000', 0), 98: ('H', '000', 1), 99: ('H', '000', 2), 100: ('H', '000', 3)}
for base, cls, size in ((105, 'IR', '007'), (118, 'B', '007'), (130, 'OR', '007'), (169, 'IR', '014'), (185, 'B', '014'),
                        (197, 'OR', '014'), (209, 'IR', '021'), (222, 'B', '021'), (234, 'OR', '021'),
                        (3001, 'IR', '028'), (3005, 'B', '028')):
    for load in range(4):
        CWRU_MAP[base + load] = (cls, size, load)
CWRU_NOMINAL_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}


def load_cwru(root, check=False):
    files = sorted(glob.glob(os.path.join(root, 'CWRU', '**', '*.mat'), recursive=True))
    g = fault_orders(GEOM['CWRU_6205']); recs = []; unknown = []
    for p in files:
        m = re.match(r'^(\d+)\.mat$', os.path.basename(p))
        if not m or int(m.group(1)) not in CWRU_MAP:
            unknown.append(os.path.basename(p)); continue
        num = int(m.group(1)); cls, size, load = CWRU_MAP[num]; d = _mat(p)
        key = f'X{num:03d}_DE_time'
        if key not in d:
            raise SystemExit(f'{p}: {key} missing; keys {[k for k in d if not k.startswith("__")]}')
        x = d[key].ravel().astype(float)
        rk = f'X{num:03d}RPM'; rpm = float(np.ravel(d[rk])[0]) if rk in d else float(CWRU_NOMINAL_RPM[load])
        fs = 12000
        if cls == 'H':  # normal baseline is recorded at 48 kHz
            x = resample_poly(x, 1, 4)
        unit = 'H' if cls == 'H' else f'{cls}{size}'
        recs.append(dict(file=os.path.basename(p), x=x, fs=fs, knots=const_speed_knots(len(x), fs, rpm / 60), fo=g,
                         cls=cls, unit=unit, group=size, cond=str(load), speed_note=f'{rpm:.0f} rpm' + ('' if rk in d else ' (nominal)')))
        if check and len(recs) >= 6: break
    if unknown: print('CWRU: files not in the 12k drive-end map (skipped):', unknown)
    return recs


HUST_RE = re.compile(r'^(IB|IO|OB|[A-Z])(\d)(\d{2})$')
HUST_CLASS = {'N': 'H', 'H': 'H', 'I': 'IR', 'O': 'OR', 'B': 'B', 'IB': 'C', 'IO': 'C', 'OB': 'C'}


def hust_speed(d, x, fs):
    """Shaft frequency: the scalar 'fs' field of the file (values 22-25 Hz look like the shaft rate, not the
    sampling rate). Cross-checked against the largest spectral line of the vibration between 10 and 40 Hz."""
    v = float(np.ravel(d['fs'])[0]) if 'fs' in d else float('nan')
    P = np.abs(np.fft.rfft(x - x.mean())); f = np.fft.rfftfreq(len(x), 1 / fs); m = (f > 10) & (f < 40)
    return v, float(f[m][np.argmax(P[m])])


def load_hust(root, check=False):
    files = sorted(glob.glob(os.path.join(root, 'HUST', '**', '*.mat'), recursive=True)); recs = []; skipped = []
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]; m = HUST_RE.match(name)
        if not m or m.group(1) not in HUST_CLASS:
            skipped.append(name); continue
        letters, typ, load = m.groups(); cls = HUST_CLASS[letters]
        if cls == 'C':
            continue
        d = _mat(p); x = d['data'].ravel().astype(float); fs = 51200
        fr_file, fr_spec = hust_speed(d, x, fs)
        if not np.isfinite(fr_file):
            raise SystemExit(f'{p}: no speed field')
        gkey = f'HUST_620{typ}'
        recs.append(dict(file=name, x=x, fs=fs, knots=const_speed_knots(len(x), fs, fr_file), fo=fault_orders(GEOM[gkey]),
                         cls=cls, unit=f'{letters}{typ}', group=f'620{typ}', cond=f'{int(load) * 100} W',
                         speed_note=f'shaft {fr_file:.2f} Hz (file); spectrum line {fr_spec:.2f} Hz'))
        if check and len(recs) >= 6: break
    if skipped: print('HUST: names not understood (skipped):', skipped)
    return recs


OTT_PPR = 1024  # encoder cycles per revolution (Huang & Baddour 2018); checked below from the vibration


def encoder_knots(c, fs, ppr=OTT_PPR):
    c = np.asarray(c, float); lo, hi = np.percentile(c, [5, 95]); b = c > (lo + hi) / 2
    edges = np.flatnonzero(~b[:-1] & b[1:]) + 1
    gaps = np.diff(edges); keep = np.concatenate([[True], gaps > 0.3 * np.median(gaps)])
    edges = edges[keep]
    return edges / fs, np.arange(len(edges)) / ppr


def shaft_check(x, fs, t_knots, rev_knots):
    """Order of the largest line of the low-passed vibration between 0.5 and 3 orders (1.0 expected)."""
    xd = resample_poly(x, 1, 40); fsd = fs / 40; t = np.arange(len(xd)) / fsd
    ok = (t >= t_knots[0]) & (t <= t_knots[-1]); r = np.interp(t[ok], t_knots, rev_knots)
    grid = np.arange(r[0], r[-1], 1 / 32); a = np.interp(grid, r, xd[ok]); a = (a - a.mean()) * np.hanning(len(a))
    S = np.abs(np.fft.rfft(a)); o = np.fft.rfftfreq(len(a), 1 / 32); m = (o > 0.5) & (o < 3)
    return float(o[m][np.argmax(S[m])])


def load_ottawa(root, check=False):
    files = sorted(glob.glob(os.path.join(root, 'Ottawa', '**', '*.mat'), recursive=True)); recs = []
    cmap = {'H': 'H', 'I': 'IR', 'O': 'OR', 'B': 'B', 'C': 'C'}; g = fault_orders(GEOM['OTTAWA_ER16K'])
    for p in files:
        m = re.match(r'^([HIOBC])-([A-D])-(\d)$', os.path.splitext(os.path.basename(p))[0])
        if not m or cmap[m.group(1)] == 'C':
            continue
        d = _mat(p); x = d['Channel_1'].ravel().astype(float); fs = 200000
        tk, rk = encoder_knots(d['Channel_2'].ravel(), fs)
        sp = np.diff(rk[::OTT_PPR]) / np.diff(tk[::OTT_PPR]) if len(rk) > 2 * OTT_PPR else np.array([np.nan])
        note = f'shaft {np.nanmin(sp):.1f}-{np.nanmax(sp):.1f} Hz from encoder'
        if check:
            note += f'; strongest low-order line at {shaft_check(x, fs, tk, rk):.2f} orders (1.00 expected)'
        cls = cmap[m.group(1)]
        recs.append(dict(file=os.path.basename(p), x=x, fs=fs, knots=(tk, rk), fo=g, cls=cls, unit=cls,
                         group=m.group(2), cond=m.group(2), speed_note=note))
        if check and len(recs) >= 6: break
    return recs


LOADERS = {'cwru': load_cwru, 'hust': load_hust, 'ottawa': load_ottawa}


# ----------------------------------------------------------------------------------------- feature cache
def build(name, root, out):
    path = out / f'{name}_features.npz'
    if path.exists():
        return dict(np.load(path, allow_pickle=False))
    t0 = time.time(); recs = LOADERS[name](root); cols = {k: [] for k in ('P', 'T', 'S', 'fr', 'cls', 'unit', 'group', 'cond', 'recording')}
    if not recs:
        raise SystemExit(f'{name}: no recordings found under {root}')
    for r in recs:
        segs = recording_features(r['x'], r['fs'], *r['knots'], r['fo'])
        for P, T, S, fr in segs:
            for k, v in (('P', P), ('T', T), ('S', S), ('fr', fr), ('cls', r['cls']), ('unit', r['unit']),
                         ('group', r['group']), ('cond', r['cond']), ('recording', r['file'])):
                cols[k].append(v)
        print(f'  {name} {r["file"]}: {len(segs)} segments, {r["speed_note"]}', flush=True)
    d = {k: np.array(v) for k, v in cols.items()}; d['feature_names'] = np.array(FEATURE_NAMES)
    np.savez_compressed(path, **d); print(f'{name}: {len(d["cls"])} segments from {len(recs)} recordings, {time.time() - t0:.0f} s')
    return d


# ----------------------------------------------------------------------------------------- protocols and models
def protocols(name, d):
    groups = sorted(set(d['group'][d['cls'] != 'H'])); H = d['cls'] == 'H'
    if name == 'cwru':
        groups = [g for g in groups if g != '028']   # 0.028 in. exists only for IR and B
        h_tr = H & np.isin(d['cond'], ['0', '1']); h_te = H & np.isin(d['cond'], ['2', '3'])
        fault = ~H & np.isin(d['group'], groups)
        for g in groups:
            yield 'LOSO', g, (fault & (d['group'] != g)) | h_tr, (fault & (d['group'] == g)) | h_te
        for g in groups:
            yield '1SIZE', g, (fault & (d['group'] == g)) | h_tr, (fault & (d['group'] != g)) | h_te
    elif name == 'hust':
        types = sorted(set(d['group']))
        for g in types:
            yield 'LOTO', g, d['group'] != g, d['group'] == g
        for g in types:
            yield '1TYPE', g, d['group'] == g, d['group'] != g
    else:
        profs = sorted(set(d['group']))
        for g in profs:
            yield 'LOPO', g, d['group'] != g, d['group'] == g
        for g in profs:
            yield '1PROF', g, d['group'] == g, d['group'] != g


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))


METHODS = {'ReliSense (P + LR)': (('P',), lr),
           'band energies + LR': (('S',), lr),
           'time + band + RF': (('T', 'S'), lambda: RandomForestClassifier(300, random_state=0, n_jobs=-1))}


def metrics(y, p, names):
    r = {'n': int(len(y)), 'acc': float((y == p).mean()), 'bacc': float(balanced_accuracy_score(y, p)),
         'f1': float(f1_score(y, p, average='macro'))}
    for k, c in enumerate(names):
        if (y == k).any(): r[f'recall_{c}'] = float((p[y == k] == k).mean())
    if 'H' in names:
        h = y == names.index('H'); f = ~h
        if h.any(): r['false_alarm'] = float((p[h] != names.index('H')).mean())
        if f.any(): r['missed_fault'] = float((p[f] == names.index('H')).mean())
    return r


def run_task(name, d, names):
    keep = np.isin(d['cls'], names); y_all = np.array([names.index(c) if c in names else -1 for c in d['cls']])
    res = {}
    for proto, fold, tr, te in protocols(name, d):
        tr, te = tr & keep, te & keep
        assert not set(d['unit'][tr]) & set(d['unit'][te]) or name in ('cwru', 'ottawa'), (proto, fold)
        if name == 'cwru':  # only the single healthy bearing may appear on both sides
            assert set(d['unit'][tr]) & set(d['unit'][te]) <= {'H'}, (proto, fold)
        if not tr.any() or not te.any() or len(set(y_all[tr])) < len(names):
            print(f'  {name} {proto} {fold}: skipped (a class is missing from training)'); continue
        for mname, (blocks, mk) in METHODS.items():
            X = np.concatenate([d[b] for b in blocks], 1)
            p = mk().fit(X[tr], y_all[tr]).predict(X[te])
            r = res.setdefault(proto, {}).setdefault(mname, {'folds': {}, 'y': [], 'p': [], 'unit': []})
            r['folds'][fold] = metrics(y_all[te], p, names)
            r['y'] += y_all[te].tolist(); r['p'] += p.tolist(); r['unit'] += d['unit'][te].tolist()
    out = {}
    for proto, ms in res.items():
        for mname, r in ms.items():
            y, p, u = np.array(r['y']), np.array(r['p']), np.array(r['unit'])
            accs = [f['acc'] for f in r['folds'].values()]
            out.setdefault(proto, {})[mname] = {
                'pooled': metrics(y, p, names), 'fold_acc_mean': float(np.mean(accs)), 'fold_acc_sd': float(np.std(accs)),
                'folds': r['folds'], 'per_unit': {k: float((p[u == k] == y[u == k]).mean()) for k in sorted(set(u))}}
    return out


def paired(out, ref='ReliSense (P + LR)'):
    from scipy.stats import wilcoxon
    lines = []
    for proto, ms in out.items():
        a = ms[ref]['per_unit']
        for mname, r in ms.items():
            if mname == ref: continue
            b = r['per_unit']; ks = sorted(set(a) & set(b)); diff = np.array([a[k] - b[k] for k in ks]) * 100
            try: pw = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            except ValueError: pw = float('nan')
            lines.append(f'    {proto:6s} ReliSense - {mname:20s}: mean per-bearing difference {diff.mean():+6.1f} points over '
                         f'{len(ks)} test units (better {int((diff > 0).sum())}, equal {int((diff == 0).sum())}, worse {int((diff < 0).sum())}; Wilcoxon p = {pw:.3f})')
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, help='folder holding CWRU/, HUST/, Ottawa/')
    ap.add_argument('--datasets', default='cwru,hust,ottawa')
    ap.add_argument('--out', default=None, help='default: <root>/phase10_results')
    ap.add_argument('--check', action='store_true', help='read a few files per dataset, print the checks, stop')
    a = ap.parse_args()
    names = [s.strip().lower() for s in a.datasets.split(',') if s.strip()]
    for n in names:
        if n not in LOADERS:
            raise SystemExit(f'unknown dataset {n}; XJTU-SY is not included: the Kaggle copy holds preprocessed '
                             'train/test arrays, not the raw recordings of each bearing')
    if a.check:
        for n in names:
            print(f'\n=== {n}')
            for k, g in GEOM.items():
                if k.lower().startswith(n[:4]):
                    fo = fault_orders(g); print(f'  {k}: ' + ', '.join(f'{q} {v:.3f}' for q, v in fo.items() if q != 'shaft') + f'   [{g["src"]}]')
            for r in LOADERS[n](a.root, check=True):
                print(f'  {r["file"]:10s} class {r["cls"]:2s} unit {r["unit"]:6s} group {r["group"]:5s} cond {r["cond"]:6s} '
                      f'{len(r["x"]) / r["fs"]:.1f} s at {r["fs"]} Hz; {r["speed_note"]}')
        return
    out = Path(a.out or os.path.join(a.root, 'phase10_results')); out.mkdir(parents=True, exist_ok=True)
    R, txt = {}, []
    for n in names:
        print(f'\n=== {n}: features', flush=True); d = build(n, a.root, out)
        present = set(d['cls'])
        tasks = {'3-class': CLS3, '4-class': CLS4} if 'H' in present else {'fault type (no healthy files)': ['OR', 'IR', 'B']}
        for tname, cl in tasks.items():
            if not set(cl) <= present:
                txt.append(f'{n} {tname}: skipped, classes present {sorted(present)}'); continue
            print(f'=== {n}: {tname}', flush=True); o = run_task(n, d, cl); R.setdefault(n, {})[tname] = o
            txt.append(f'\n{n.upper()} {tname}  (segments of {SEG_REVS} revolutions; pooled over folds)')
            for proto, ms in o.items():
                for mname, r in ms.items():
                    pm = r['pooled']; fa = f"  false alarms {pm['false_alarm'] * 100:5.1f}%" if 'false_alarm' in pm else ''
                    txt.append(f"  {proto:6s} {mname:20s} acc {pm['acc'] * 100:5.1f}%  (folds {r['fold_acc_mean'] * 100:5.1f} +- {r['fold_acc_sd'] * 100:4.1f})"
                               f"  bacc {pm['bacc'] * 100:5.1f}%  F1 {pm['f1'] * 100:5.1f}%{fa}")
            txt += paired(o)
    (out / 'phase10_results.json').write_text(json.dumps(R, indent=1))
    (out / 'phase10_summary.txt').write_text('\n'.join(txt)); print('\n'.join(txt)); print('\nwritten', out)


if __name__ == '__main__':
    main()
