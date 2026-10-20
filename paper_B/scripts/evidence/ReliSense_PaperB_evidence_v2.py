"""ReliSense Paper B v8: evidence analyses requested in the critical review (single file, Colab).

Run in Colab:
  1) mount Google Drive and set DATA_DIR below to the folder holding the Paderborn .mat files
     (e.g. .../K001/N15_M07_F10_K001_1.mat; sub-folders are searched),
  2) check LOG_BASE (10 if your feature code uses np.log10, 'e' if it uses np.log),
  3) !python ReliSense_PaperB_evidence_v2.py
Runtime: about 15-30 min on a Colab CPU (feature extraction is cached in OUT_DIR/features_v2.npz).
Send back OUT_DIR/summary_evidence_v2.txt (and the CSV files). The keys [E2-0] ... [E2-10] match the highlighted
placeholders in ReliSense_PaperB_v8_0.docx.

Analyses
  E2-0  reproduction check (LOBO accuracy, false alarms, per-bearing accuracy, class-difference weights) and log-base check
  E2-1  order-grid audit: truncated windows on the 590-bin grid; specificity margins on the original and an extended grid
  E2-2  paired per-bearing differences, true orders vs best wrong set and vs mean random set, bootstrap 95% intervals
  E2-3  capture check: which wrong sets place a window on a true BPFO/BPFI line (h = 1-3)
  E2-4  shaft-harmonic audit: location of the band maximum relative to shaft harmonics; probe with shaft bins removed
  E2-5  Paderborn contrast: condition-matched Gamma against each of the six healthy bearings; speed pairing
  E2-6  healthy state: log-odds margins of every healthy bearing, threshold-rule joint false-alarm rate,
        probe recall at the threshold rule's false-alarm rate (matched operating point)
  E2-7  bearing identity overall, within each class and after reduction to 15 principal components
  E2-8  matched training comparison: artificial vs real damage with equal numbers of training bearings;
        effect of adding K004/K005 to the healthy class
  E2-9  external contrast association (16 bearings): tie-corrected Spearman, permutation tests, within-dataset
  E2-10 folding: synchronous-average peak factor at the nominal, the estimated and a 10%-offset period
"""
import os, glob, re, sys, csv, json, warnings, itertools, time
import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon, rankdata, kurtosis, skew
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------ settings
DATA_DIR = os.environ.get('PB_DATA_DIR', '/content/drive/MyDrive/Paderborn')
OUT_DIR = os.environ.get('PB_OUT_DIR', './evidence_v2_out')
LOG_BASE = os.environ.get('PB_LOG_BASE', '10')     # '10' -> np.log10, 'e' -> np.log ; must match the manuscript code
EXTERNAL_CSV = os.environ.get('PB_EXTERNAL_CSV', '')  # optional: dataset,race,gamma,accuracy for the 16 external bearings
J_EXTRA = int(os.environ.get('PB_J_EXTRA', '100'))  # extra random order sets for a finer rank statistic (0 = off)
N_BOOT = 10000
N_PERM = 100000
SEED = 0
FS = 64000
BAND = (2000.0, 12000.0)
DECIM = 16
C_LR = 0.3
LOGF = np.log10 if LOG_BASE == '10' else np.log

COND = {'C0': 'N15_M07_F10', '900rpm': 'N09_M07_F10', 'lowtorque': 'N15_M01_F10', 'lowforce': 'N15_M07_F04'}
NOMINAL_RPM = {'C0': 1500, '900rpm': 900, 'lowtorque': 1500, 'lowforce': 1500}
ORDERS = {'BPFO': 3.0543, 'BPFI': 4.9457, 'BSF': 1.9966, 'FTF': 0.38178, 'shaft': 1.0}   # 6203: n=8, d=6.75, D=28.55
FAMS = ['BPFO', 'BPFI', 'BSF', 'FTF', 'shaft']
FEAT_NAMES = [f'{k}{h}' for k in FAMS for h in (1, 2, 3)]
CLASS_ORDER = {'OR': 'BPFO', 'IR': 'BPFI'}

BEARINGS = {
    'K001': ('H', 'healthy', 0, '-', '-'), 'K002': ('H', 'healthy', 0, '-', '-'), 'K003': ('H', 'healthy', 0, '-', '-'),
    'K004': ('H', 'healthy', 0, '-', '-'), 'K005': ('H', 'healthy', 0, '-', '-'), 'K006': ('H', 'healthy', 0, '-', '-'),
    'KA01': ('OR', 'artificial', 1, 'EDM', 'S'), 'KA03': ('OR', 'artificial', 2, 'engraver', 'S'),
    'KA05': ('OR', 'artificial', 1, 'engraver', 'S'), 'KA06': ('OR', 'artificial', 2, 'engraver', 'S'),
    'KA07': ('OR', 'artificial', 1, 'drilling', 'S'), 'KA08': ('OR', 'artificial', 2, 'drilling', 'S'),
    'KA09': ('OR', 'artificial', 2, 'drilling', 'S'),
    'KI01': ('IR', 'artificial', 1, 'EDM', 'S'), 'KI03': ('IR', 'artificial', 1, 'engraver', 'S'),
    'KI05': ('IR', 'artificial', 1, 'engraver', 'S'), 'KI07': ('IR', 'artificial', 2, 'engraver', 'S'),
    'KI08': ('IR', 'artificial', 2, 'engraver', 'S'),
    'KA04': ('OR', 'real', 1, 'pitting', 'S'), 'KA15': ('OR', 'real', 1, 'indentation', 'S'),
    'KA16': ('OR', 'real', 2, 'pitting', 'R'), 'KA22': ('OR', 'real', 1, 'pitting', 'S'),
    'KA30': ('OR', 'real', 1, 'indentation', 'R/d'),
    'KI04': ('IR', 'real', 1, 'pitting', 'M'), 'KI14': ('IR', 'real', 1, 'pitting', 'M'),
    'KI16': ('IR', 'real', 3, 'pitting', 'S'), 'KI17': ('IR', 'real', 1, 'pitting', 'R'),
    'KI18': ('IR', 'real', 2, 'pitting', 'S'), 'KI21': ('IR', 'real', 1, 'pitting', 'S'),
}
HEALTHY = [b for b in BEARINGS if BEARINGS[b][0] == 'H']
L8_TRAIN = ['K002', 'KA01', 'KA05', 'KA07', 'KI01', 'KI05', 'KI07']
L8_TEST = ['K001', 'KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']
REAL_OR = ['KA04', 'KA15', 'KA16', 'KA22', 'KA30']
REAL_IR = ['KI04', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']
PAPER_LOBO = {'K001': 80, 'K002': 85, 'K003': 94, 'K004': 15, 'K005': 8, 'K006': 72, 'KA01': 100, 'KA03': 100,
              'KA05': 100, 'KA06': 100, 'KA07': 100, 'KA08': 87, 'KA09': 75, 'KI01': 100, 'KI03': 94, 'KI05': 75,
              'KI07': 70, 'KI08': 98, 'KA04': 99, 'KA15': 61, 'KA16': 100, 'KA22': 62, 'KA30': 46, 'KI04': 100,
              'KI14': 80, 'KI17': 39, 'KI21': 86, 'KI18': 100, 'KI16': 98}
# External bearings reconstructed from Fig. 10(b) of the manuscript (APPROXIMATE): replace with exact values via EXTERNAL_CSV.
EXTERNAL_DEFAULT = [('HUST', 'OR', 0.37, 8.3), ('HUST', 'OR', 0.91, 100), ('HUST', 'OR', 2.18, 100), ('HUST', 'OR', 2.47, 100),
                    ('HUST', 'OR', 2.82, 100), ('HUST', 'IR', 1.03, 0.0), ('HUST', 'IR', 0.90, 58.3), ('HUST', 'IR', 1.12, 100),
                    ('HUST', 'IR', 2.18, 100), ('HUST', 'IR', 2.47, 100), ('CWRU', 'OR', 0.39, 0.0), ('CWRU', 'OR', 1.82, 100),
                    ('CWRU', 'OR', 3.50, 100), ('CWRU', 'IR', 1.67, 100), ('CWRU', 'IR', 2.60, 100), ('CWRU', 'IR', 3.07, 100)]

SCALES = [0.80, 0.85, 0.90, 0.95, 0.98, 1.02, 1.05, 1.10, 1.15, 1.20]
DR = 0.02
R_ORIG = np.arange(1, 591) * DR       # 590 bins, 0.02 ... 11.80 (as stated in the manuscript; check your code)
R_EXT = np.arange(1, 631) * DR        # extended to 12.60
FOLD_PERIODS = 120

OUT = []
def say(*a):
    s = ' '.join(str(x) for x in a); OUT.append(s); print(s, flush=True)

# ------------------------------------------------------------------ reading
def _channels(rec):
    out = {}
    Y = rec.Y if hasattr(rec, 'Y') else rec['Y']
    Y = np.atleast_1d(Y).ravel()
    if Y.dtype.names:
        for row in Y:
            out[str(np.squeeze(row['Name']))] = np.asarray(np.squeeze(row['Data']), float)
    else:
        for ch in Y:
            out[str(ch.Name)] = np.asarray(ch.Data, float).ravel()
    return out

def load_recording(path, ck):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    key = [k for k in m if not k.startswith('__')][0]
    ch = _channels(m[key])
    vib = ch['vibration_1']
    rpm = float(np.median(ch['speed'])) if 'speed' in ch and ch['speed'].size else np.nan
    nom = NOMINAL_RPM[ck]
    if not (0.7 * nom < rpm < 1.3 * nom):
        rpm = float(nom)
    return vib, rpm / 60.0

def find_files(bearing, code):
    files = glob.glob(os.path.join(DATA_DIR, '**', f'{code}_{bearing}_*.mat'), recursive=True)
    return sorted(set(files), key=lambda p: int(re.findall(r'_(\d+)\.mat$', p)[0]))

# ------------------------------------------------------------------ signal processing
SOS = butter(4, [BAND[0] / (FS / 2), BAND[1] / (FS / 2)], btype='band', output='sos')

def envelope(vib):
    e = np.abs(hilbert(sosfiltfilt(SOS, vib))) ** 2
    n = (e.size // DECIM) * DECIM
    return e[:n].reshape(-1, DECIM).mean(1)

def spectrum(e):
    x = e - e.mean()
    S = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    f = np.fft.rfftfreq(x.size, DECIM / FS)
    return f, S

def bands(f, fr, k, h):
    df = f[1] - f[0]
    fk = h * ORDERS[k] * fr
    half = max(0.02 * fk, 1.5 * df)
    srch = np.abs(f - fk) <= half
    bg = (np.abs(f - fk) <= 10.0) & ~srch
    return fk, srch, bg

def core_features(f, S, fr, notch=False):
    out, peaks = [], {}
    for k in FAMS:
        for h in (1, 2, 3):
            fk, srch, bg = bands(f, fr, k, h)
            s = srch.copy()
            if notch and k in ('BPFO', 'BPFI', 'BSF'):
                m = np.round(f / fr)
                near = (m >= 1) & (np.abs(f - m * fr) <= 0.5)
                if (s & ~near).any():
                    s = s & ~near
            idx = np.where(s)[0]
            j = idx[np.argmax(S[idx])]
            out.append(LOGF(S[j] / max(np.median(S[bg]), 1e-300)))
            if not notch and h == 1 and k in ('BPFO', 'BPFI'):
                peaks[k] = f[j]
    return np.array(out), peaks

def order_spectrum(f, S, fr, grid):
    So = np.interp(grid * fr, f, S)
    return LOGF(So / np.median(So))

def generic_features(vib, e):
    P = np.abs(np.fft.rfft(vib * np.hanning(vib.size))) ** 2
    fv = np.fft.rfftfreq(vib.size, 1 / FS)
    edges = np.linspace(0, 4000, 33)
    E = np.array([P[(fv >= a) & (fv < b)].sum() for a, b in zip(edges[:-1], edges[1:])]) + 1e-30
    be = np.log10(E / E.sum())
    x = vib - vib.mean(); rms = np.sqrt(np.mean(x ** 2)); pk = np.max(np.abs(x)); ma = np.mean(np.abs(x))
    tstat = [rms, x.std(), skew(x), kurtosis(x, fisher=False), pk / rms, np.ptp(x), rms / ma, pk / ma]
    env = np.sqrt(e)
    estat = [env.mean(), env.std(), kurtosis(env, fisher=False)]
    return be, np.r_[np.log10(np.abs(tstat) + 1e-12), np.log10(np.abs(estat) + 1e-12), be]

def fold_index(e, fs_e, period):
    """synchronous-average peak factor: max/median of the envelope averaged over FOLD_PERIODS periods (100 phase bins)"""
    n_per = min(FOLD_PERIODS, int(e.size / (period * fs_e)) - 1)
    if n_per < 10:
        return np.nan
    t = (np.arange(n_per)[:, None] + np.linspace(0, 1, 100, endpoint=False)[None, :]) * period
    prof = np.interp(t * fs_e, np.arange(e.size), e).mean(0)
    return prof.max() / np.median(prof)

# ------------------------------------------------------------------ 1) extraction (cached)
def extract():
    cache = os.path.join(OUT_DIR, f'features_v2_log{LOG_BASE}.npz')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        say(f'Loaded cached features from {cache}')
        return {k: z[k] for k in z.files}
    rows, X, Xn, O, BE, TB, PK, FD = [], [], [], [], [], [], [], []
    skipped = []
    t0 = time.time()
    for b in BEARINGS:
        for ck, code in COND.items():
            for p in find_files(b, code):
                try:
                    vib, fr = load_recording(p, ck)
                    e = envelope(vib); f, S = spectrum(e)
                    x, pk = core_features(f, S, fr); xn, _ = core_features(f, S, fr, notch=True)
                    be, tb = generic_features(vib, e)
                    fs_e = FS / DECIM
                    k = CLASS_ORDER.get(BEARINGS[b][0], 'BPFO')
                    fnom = ORDERS[k] * fr
                    srch = np.abs(f - fnom) <= 0.02 * fnom
                    fstar = f[np.where(srch)[0][np.argmax(S[srch])]]
                    fd = [fold_index(e, fs_e, 1 / fnom), fold_index(e, fs_e, 1 / fstar), fold_index(e, fs_e, 1.1 / fstar)] \
                        if ck == 'C0' else [np.nan] * 3
                    rows.append((b, ck, os.path.basename(p), fr)); X.append(x); Xn.append(xn)
                    O.append(order_spectrum(f, S, fr, R_EXT)); BE.append(be); TB.append(tb)
                    PK.append([pk['BPFO'], pk['BPFI']]); FD.append(fd)
                except Exception as ex:
                    skipped.append((os.path.basename(p), str(ex)[:60]))
        say(f'  {b}: {sum(r[0] == b for r in rows)} recordings ({time.time() - t0:.0f} s)')
    if not rows:
        sys.exit(f'No .mat files found under {DATA_DIR}')
    d = dict(bear=np.array([r[0] for r in rows]), cond=np.array([r[1] for r in rows]),
             fname=np.array([r[2] for r in rows]), fr=np.array([r[3] for r in rows]),
             X=np.array(X), Xn=np.array(Xn), O=np.array(O, dtype=np.float32), BE=np.array(BE), TB=np.array(TB),
             PK=np.array(PK), FD=np.array(FD), skipped=np.array(skipped, dtype=object))
    np.savez_compressed(cache, **d)
    return d

# ------------------------------------------------------------------ helpers
def fit_lr(Xtr, ytr, C=C_LR):
    return make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000)).fit(Xtr, ytr)

def lobo_predict(Xf, y, bear, extra_mask=None):
    """out-of-fold predictions and probabilities; extra_mask(b) -> boolean mask of extra rows excluded from training"""
    pred = np.empty(len(y), dtype=object); prob = np.zeros((len(y), 3)); classes = None
    for b in BEARINGS:
        te = bear == b
        if not te.any():
            continue
        tr = ~te if extra_mask is None else (~te & ~extra_mask)
        m = fit_lr(Xf[tr], y[tr])
        pred[te] = m.predict(Xf[te]); prob[te] = m.predict_proba(Xf[te]); classes = m.classes_
    return pred, prob, list(classes)

def per_bearing(correct, bear, cond=None, c=None):
    sel = np.ones(len(bear), bool) if c is None else cond == c
    return {b: 100 * correct[sel & (bear == b)].mean() for b in BEARINGS if (sel & (bear == b)).any()}

def boot_ci(v, n=N_BOOT, seed=SEED):
    v = np.asarray(v, float); rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, v.size, (n, v.size))].mean(1)
    return np.percentile(bs, [2.5, 97.5])

def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 4 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan, np.nan, int(ok.sum())
    r, p = spearmanr(a[ok], b[ok]); return r, p, int(ok.sum())

def perm_spearman(a, b, strata=None, n=N_PERM, seed=SEED, alternative='two-sided'):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = rankdata(a); rb = rankdata(b)
    ra = (ra - ra.mean()) / np.linalg.norm(ra - ra.mean()); rb = (rb - rb.mean()) / np.linalg.norm(rb - rb.mean())
    obs = float(ra @ rb); rng = np.random.default_rng(seed); cnt = 0; done = 0
    idx = np.arange(a.size)
    while done < n:
        m = min(20000, n - done)
        if strata is None:
            P = np.argsort(rng.random((m, a.size)), axis=1)
        else:
            P = np.tile(idx, (m, 1))
            for s in np.unique(strata):
                w = np.where(strata == s)[0]
                P[:, w] = w[np.argsort(rng.random((m, w.size)), axis=1)]
        null = (ra[P] * rb).sum(1)
        cnt += int(((np.abs(null) >= abs(obs) - 1e-12) if alternative == 'two-sided' else (null >= obs - 1e-12)).sum())
        done += m
    return obs, (1 + cnt) / (1 + n)

# ------------------------------------------------------------------ specificity variant (11 order features)
def line_height(O, grid, r):
    w = max(0.02 * r, 0.04)
    inside = np.abs(grid - r) <= w
    ring = (np.abs(grid - r) <= 0.4) & ~inside
    if not inside.any() or not ring.any():
        return np.full(O.shape[0], np.nan), 'upper' if r > grid[-1] else 'lower'
    trunc = 'upper' if r + 0.4 > grid[-1] + 1e-9 else ('lower' if r - 0.4 < grid[0] - 1e-9 else '')
    return O[:, inside].max(1) - np.median(O[:, ring], axis=1), trunc

def variant_features(O, grid, orders):
    cols, trunc = [], []
    for k in ('BPFO', 'BPFI', 'BSF', 'FTF'):
        for h in (1, 2):
            v, t = line_height(O, grid, h * orders[k]); cols.append(v); trunc.append(t)
    for h in (1, 2, 3):
        v, t = line_height(O, grid, float(h)); cols.append(v); trunc.append(t)
    return np.column_stack(cols), ','.join(sorted(set(t for t in trunc if t)))

def captures(orders):
    hits = []
    for k in ('BPFO', 'BPFI', 'BSF', 'FTF'):
        for h in (1, 2):
            r = h * orders[k]; w = max(0.02 * r, 0.04)
            for kt in ('BPFO', 'BPFI'):
                for ht in (1, 2, 3):
                    if abs(ht * ORDERS[kt] - r) <= w + DR:
                        hits.append(f'{k}{h}~{kt}{ht}')
    return hits

# ------------------------------------------------------------------ analyses
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d = extract()
    bear, cond, X, Xn, O = d['bear'], d['cond'], d['X'], d['Xn'], d['O'].astype(float)
    y = np.array([BEARINGS[b][0] for b in bear]); healthy = y == 'H'
    damaged_b = [b for b in BEARINGS if BEARINGS[b][0] != 'H']
    say(f'Recordings: {len(y)}; skipped: {len(d["skipped"])} {list(d["skipped"])[:2]}; log base: {LOG_BASE}')

    # ---------------- E2-0 reproduction and log base
    say('\n[E2-0] Reproduction check (manuscript: LOBO 80.1%, FAR 41.0%; C0 88.6, 900 rpm 65.2)')
    pred, prob, classes = lobo_predict(X, y, bear)
    corr = pred == y
    pb = per_bearing(corr, bear)
    say(f'  LOBO accuracy {100 * corr.mean():.1f}%; FAR {100 * (pred[healthy] != "H").mean():.1f}%; '
        + ', '.join(f'{c} {100 * corr[cond == c].mean():.1f}%' for c in COND if (cond == c).any()))
    dif = sorted(pb, key=lambda b: -abs(pb[b] - PAPER_LOBO[b]))[:8]
    say('  largest per-bearing differences (this run / manuscript): ' + ', '.join(f'{b} {pb[b]:.0f}/{PAPER_LOBO[b]}' for b in dif))
    full = fit_lr(X, y); lr = full[-1]; ci = {c: i for i, c in enumerate(lr.classes_)}
    wd = {c: lr.coef_[ci[c]] - lr.coef_[ci['H']] for c in ('OR', 'IR')}
    for c in ('OR', 'IR'):
        top = np.argsort(-np.abs(wd[c]))[:6]
        say(f'  class-difference weights {c} - H (all 29 bearings): ' + ', '.join(f'{FEAT_NAMES[j]} {wd[c][j]:+.2f}' for j in top))
    say('  (manuscript Fig. 8: OR-H BPFO1 +2.60, BPFO3 +1.76, BSF3 -1.81, BPFO2 +1.35; IR-H BPFI1 +3.29, BPFO2 -1.56, BPFI2 +1.40, shaft1 +1.38)')
    rng = np.random.default_rng(SEED); n = 16000; fsim = np.fft.rfftfreq(n, DECIM / FS); sims = {}
    for k in ('BPFO', 'BPFI'):
        v = []
        for _ in range(300):
            ee = rng.standard_normal(n) ** 2; ee -= ee.mean()
            S = np.abs(np.fft.rfft(ee * np.hanning(n))) ** 2
            _, s, bg = bands(fsim, 25.0, k, 1); v.append(S[s].max() / np.median(S[bg]))
        sims[k] = (np.median(np.log(v)), np.median(np.log10(v)))
    for k, j in (('BPFO', 0), ('BPFI', 3)):
        hm = np.median(X[healthy & (cond == 'C0'), j])
        say(f'  log-base check {k}1 at C0: healthy median {hm:.2f} (this run, base {LOG_BASE}); pure-noise median '
            f'ln {sims[k][0]:.2f}, log10 {sims[k][1]:.2f}. Healthy values below the noise floor indicate the wrong base.')

    # ---------------- E2-1/2/3 specificity
    say('\n[E2-1/2/3] Specificity test (11-feature variant)')
    rngo = np.random.default_rng(SEED)
    rand_sets = [dict(zip(('BPFO', 'BPFI', 'BSF', 'FTF'), rngo.uniform(0.3, 5.9, 4))) for _ in range(20 + J_EXTRA)]
    sets = [('true', dict(ORDERS))] + [(f's={s:.2f}', {k: s * ORDERS[k] for k in ('BPFO', 'BPFI', 'BSF', 'FTF')}) for s in SCALES] \
        + [(f'rand{j + 1}', r) for j, r in enumerate(rand_sets)]
    l8_tr = np.isin(bear, L8_TRAIN); l8_te = np.isin(bear, L8_TEST) & (cond == 'C0')
    res = {}
    for gname, grid in (('orig', R_ORIG), ('ext', R_EXT)):
        Og = O[:, :grid.size]
        for name, orders in (sets[:31] if gname == 'orig' else sets):
            F, tr = variant_features(Og, grid, orders)
            p8 = fit_lr(F[l8_tr], y[l8_tr]).predict(F[l8_te]); c8 = p8 == y[l8_te]
            pl, _, _ = lobo_predict(F, y, bear); cl = pl == y
            res[(gname, name)] = dict(trunc=tr, cap=captures(orders), l8=100 * c8.mean(), lobo=100 * cl.mean(),
                                      l8_b={b: 100 * c8[bear[l8_te] == b].mean() for b in L8_TEST if (bear[l8_te] == b).any()},
                                      lobo_b=per_bearing(cl, bear), orders=orders)
        say(f'  grid {gname} ({grid[0]:.2f}-{grid[-1]:.2f}):')
        for proto in ('l8', 'lobo'):
            a0 = res[(gname, 'true')][proto]
            wrong = [nm for nm, _ in sets[1:31]]
            best = max(wrong, key=lambda nm: res[(gname, nm)][proto])
            r20 = [res[(gname, nm)][proto] for nm, _ in sets[11:31]]
            rall = [res[(gname, nm)][proto] for nm, _ in sets[11:] if (gname, nm) in res]
            prank = (1 + sum(a >= a0 for a in r20)) / 21
            prank_x = (1 + sum(a >= a0 for a in rall)) / (len(rall) + 1)
            say(f'    {"L8-C0" if proto == "l8" else "LOBO "}: true {a0:.1f}%; best wrong {best} {res[(gname, best)][proto]:.1f}% '
                f'(captures {res[(gname, best)]["cap"] or "none"}); [E2-1] margin {a0 - res[(gname, best)][proto]:.1f} pp; '
                f'mean of 20 random {np.mean(r20):.1f}% (Delta_rand {a0 - np.mean(r20):.1f} pp); P_rank(J=20) {prank:.3f}; '
                f'P_rank(J={len(rall)}) {prank_x:.4f}')
            key = 'l8_b' if proto == 'l8' else 'lobo_b'
            bs = [b for b in res[(gname, 'true')][key] if (proto == 'lobo' or b in L8_TEST)]
            dB = np.array([res[(gname, 'true')][key][b] - res[(gname, best)][key][b] for b in bs])
            dR = np.array([res[(gname, 'true')][key][b] - np.mean([res[(gname, nm)][key][b] for nm, _ in sets[11:31]]) for b in bs])
            lo, hi = boot_ci(dB); lo2, hi2 = boot_ci(dR)
            say(f'      [E2-2] per-bearing true - best wrong: mean {dB.mean():.1f} pp (95% bootstrap {lo:.1f} to {hi:.1f}), '
                f'better/equal/worse {np.sum(dB > 0)}/{np.sum(dB == 0)}/{np.sum(dB < 0)} of {len(bs)}; '
                f'true - mean random: {dR.mean():.1f} pp ({lo2:.1f} to {hi2:.1f})')
        tu = [nm for nm, _ in sets[:31] if 'upper' in res[(gname, nm)]['trunc']]
        tl = [nm for nm, _ in sets[:31] if 'lower' in res[(gname, nm)]['trunc']]
        say(f'    [E2-1] sets truncated at the upper grid end (window or ring beyond {grid[-1]:.2f}): {tu}; '
            f'at the lower end (FTF ring below {grid[0]:.2f}, minor): {len(tl)} sets')
        capd = [nm for nm, _ in sets[11:31] if res[(gname, nm)]['cap']]
        say(f'    [E2-3] random sets (of 20) that capture a true line: {len(capd)} {capd}; mean LOBO accuracy captured '
            f'{np.mean([res[(gname, nm)]["lobo"] for nm in capd]) if capd else float("nan"):.1f}% vs others '
            f'{np.mean([res[(gname, nm)]["lobo"] for nm, _ in sets[11:31] if nm not in capd]):.1f}%')
    with open(os.path.join(OUT_DIR, 'perturbation_v2.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['grid', 'set', 'BPFO', 'BPFI', 'BSF', 'FTF', 'truncated', 'captures', 'acc_L8C0', 'acc_LOBO'])
        for (g, nm), r in res.items():
            w.writerow([g, nm] + [f'{r["orders"][k]:.4f}' for k in ('BPFO', 'BPFI', 'BSF', 'FTF')]
                       + [r['trunc'], ';'.join(r['cap']), f'{r["l8"]:.2f}', f'{r["lobo"]:.2f}'])

    # ---------------- E2-4 shaft-harmonic audit
    say('\n[E2-4] Shaft-harmonic audit')
    fr = d['fr']; PK = d['PK']
    off_o = np.abs(PK[:, 0] - 3 * fr); off_i = np.abs(PK[:, 1] - 5 * fr)
    for c in COND:
        m = cond == c
        if not m.any():
            continue
        parts = []
        for grp, gm in (('K004+K005', np.isin(bear, ['K004', 'K005'])), ('other healthy', healthy & ~np.isin(bear, ['K004', 'K005'])),
                        ('outer race', y == 'OR'), ('inner race', y == 'IR')):
            mm = m & gm
            if mm.any():
                parts.append(f'{grp}: BPFO1 max within 0.5 Hz of 3fr {100 * (off_o[mm] <= 0.5).mean():.0f}%, '
                             f'BPFI1 max within 0.5 Hz of 5fr {100 * (off_i[mm] <= 0.5).mean():.0f}%')
        say(f'  {c}: ' + '; '.join(parts))
    pn, _, _ = lobo_predict(Xn, y, bear); cn = pn == y; pbn = per_bearing(cn, bear)
    p8n = fit_lr(Xn[l8_tr], y[l8_tr]).predict(Xn[l8_te])
    say(f'  probe with shaft-harmonic bins (+-0.5 Hz) removed from BPFO/BPFI/BSF bands: LOBO {100 * cn.mean():.1f}% '
        f'(FAR {100 * (pn[healthy] != "H").mean():.1f}%), L8-C0 {100 * (p8n == y[l8_te]).mean():.1f}%; '
        f'K004 {pbn.get("K004", np.nan):.0f}%, K005 {pbn.get("K005", np.nan):.0f}% (unnotched {pb.get("K004", np.nan):.0f}%, {pb.get("K005", np.nan):.0f}%)')

    # ---------------- E2-5 Paderborn contrast
    say('\n[E2-5] Paderborn condition-matched contrast Gamma (decimal or natural log per LOG_BASE)')
    fi = {n: i for i, n in enumerate(FEAT_NAMES)}
    mean_phi = {(b, c): X[(bear == b) & (cond == c)].mean(0) for b in BEARINGS for c in COND if ((bear == b) & (cond == c)).any()}
    G = {}
    for b in damaged_b:
        k = CLASS_ORDER[BEARINGS[b][0]]
        for c in COND:
            if (b, c) not in mean_phi:
                continue
            g = [np.mean([mean_phi[(b, c)][fi[f'{k}{h}']] - mean_phi[(r, c)][fi[f'{k}{h}']] for h in (1, 2, 3)])
                 for r in HEALTHY if (r, c) in mean_phi]
            G[(b, c)] = (np.median(g), np.min(g), np.max(g), g)
    pbc = {c: per_bearing(corr, bear, cond, c) for c in COND}
    with open(os.path.join(OUT_DIR, 'contrast_paderborn_v2.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['bearing', 'condition', 'class', 'origin', 'level', 'method', 'arrangement',
                                        'gamma_median', 'gamma_min', 'gamma_max'] + [f'gamma_vs_{r}' for r in HEALTHY]
                                       + ['lobo_acc_condition', 'lobo_acc_all'])
        for (b, c), (gm, gl, gh, g) in G.items():
            w.writerow([b, c, *BEARINGS[b], f'{gm:.3f}', f'{gl:.3f}', f'{gh:.3f}'] + [f'{v:.3f}' for v in g]
                       + [f'{pbc[c].get(b, np.nan):.1f}', f'{pb.get(b, np.nan):.1f}'])
    for c in ('C0', '900rpm'):
        bs = [b for b in damaged_b if (b, c) in G]
        r, p, nn = sp([G[(b, c)][0] for b in bs], [pbc[c][b] for b in bs])
        r2, p2, _ = sp([G[(b, c)][0] for b in bs], [pb[b] for b in bs])
        rr = [sp([G[(b, c)][3][i] for b in bs], [pbc[c][b] for b in bs])[0] for i in range(len(G[(bs[0], c)][3]))]
        spread = np.median([G[(b, c)][2] - G[(b, c)][1] for b in bs])
        say(f'  {c}: rho(Gamma median ref, LOBO acc at {c}) = {r:.2f} (p = {p:.3f}, n = {nn}); vs all-condition acc {r2:.2f}; '
            f'rho over the single references {np.nanmin(rr):.2f} to {np.nanmax(rr):.2f}; median spread of Gamma over references {spread:.2f}')
    for name, sel in (('all damaged', lambda b: True), ('real level 1', lambda b: BEARINGS[b][1] == 'real' and BEARINGS[b][2] == 1),
                      ('real level 2-3', lambda b: BEARINGS[b][1] == 'real' and BEARINGS[b][2] >= 2),
                      ('artificial', lambda b: BEARINGS[b][1] == 'artificial')):
        bs = [b for b in damaged_b if sel(b) and (b, 'C0') in G and (b, '900rpm') in G]
        if len(bs) < 2:
            continue
        g0 = np.array([G[(b, 'C0')][0] for b in bs]); g9 = np.array([G[(b, '900rpm')][0] for b in bs])
        p = wilcoxon(g0, g9).pvalue if len(bs) >= 5 and np.any(g0 != g9) else np.nan
        say(f'  {name} (n = {len(bs)}): Gamma 1500 -> 900 rpm median {np.median(g0):.2f} -> {np.median(g9):.2f}, '
            f'median paired change {np.median(g9 - g0):+.2f} (Wilcoxon p = {p:.3f}); accuracy '
            f'{np.mean([pbc["C0"][b] for b in bs]):.1f}% -> {np.mean([pbc["900rpm"][b] for b in bs]):.1f}%')
    say('  per bearing at C0 / 900 rpm (Gamma median, LOBO acc): ' + '; '.join(
        f'{b} {G[(b, "C0")][0]:.2f}/{G.get((b, "900rpm"), (np.nan,))[0]:.2f} ({pbc["C0"].get(b, np.nan):.0f}/{pbc["900rpm"].get(b, np.nan):.0f}%)'
        for b in damaged_b if (b, 'C0') in G))

    # ---------------- E2-6 healthy state
    say('\n[E2-6] Healthy state')
    rows = []
    for hb in HEALTHY:
        te = bear == hb
        if not te.any():
            continue
        m = fit_lr(X[~te], y[~te]); sc = m[:2].transform(X[te]); lrm = m[-1]; cix = {c: i for i, c in enumerate(lrm.classes_)}
        out = {}
        for c in ('OR', 'IR'):
            dw = lrm.coef_[cix[c]] - lrm.coef_[cix['H']]; db = lrm.intercept_[cix[c]] - lrm.intercept_[cix['H']]
            lo = sc @ dw + db; contrib = np.median(sc * dw, axis=0)
            top = np.argsort(-contrib)[:3]
            out[c] = (np.median(lo), 100 * np.mean(lo > 0), ', '.join(f'{FEAT_NAMES[j]} {contrib[j]:+.2f}' for j in top))
        rows.append((hb, out))
        say(f'  {hb}: median log-odds OR:H {out["OR"][0]:+.2f} (>0 in {out["OR"][1]:.0f}%), top {out["OR"][2]}; '
            f'IR:H {out["IR"][0]:+.2f} (>0 in {out["IR"][1]:.0f}%), top {out["IR"][2]}')
    # threshold rule under LOBO
    jo, ji = fi['BPFO1'], fi['BPFI1']
    tpred = np.empty(len(y), dtype=object)
    for b in BEARINGS:
        te = bear == b
        if not te.any():
            continue
        htr = (~te) & healthy
        to, ti = np.percentile(X[htr, jo], 95), np.percentile(X[htr, ji], 95)
        eo, ei = X[te, jo] - to, X[te, ji] - ti
        tpred[te] = np.where((eo <= 0) & (ei <= 0), 'H', np.where(eo - ei >= 0, 'OR', 'IR'))
    tcorr = tpred == y; tpb = per_bearing(tcorr, bear)
    far_t = 100 * (tpred[healthy] != 'H').mean()
    say(f'  threshold rule: pooled FAR {far_t:.1f}% (bearing mean {100 - np.mean([tpb[b] for b in HEALTHY if b in tpb]):.1f}%; '
        f'independence 9.75%); per healthy bearing recall ' + ', '.join(f'{b} {tpb[b]:.0f}%' for b in HEALTHY if b in tpb)
        + f'; damaged mean per-bearing accuracy {np.mean([tpb[b] for b in damaged_b if b in tpb]):.1f}% '
          f'(probe {np.mean([pb[b] for b in damaged_b if b in pb]):.1f}%)')
    pH = prob[:, classes.index('H')]
    t_star = np.quantile(pH[healthy], far_t / 100)          # flag damaged if P(H) < t_star -> FAR equals far_t
    flag = pH < t_star
    dmg = ~healthy
    sub = prob[:, [classes.index('OR'), classes.index('IR')]]
    cls = np.where(sub[:, 0] >= sub[:, 1], 'OR', 'IR')
    say(f'  [matched operating point] probe flagged {100 * flag[healthy].mean():.1f}% of healthy recordings; detected '
        f'{100 * flag[dmg].mean():.1f}% of damaged recordings and assigned the correct race to {100 * (flag & (cls == y))[dmg].mean():.1f}%; '
        f'threshold rule detected {100 * (tpred[dmg] != "H").mean():.1f}% and assigned the correct race to {100 * tcorr[dmg].mean():.1f}%')

    # ---------------- E2-7 identity
    say('\n[E2-7] Bearing identity (5-fold stratified CV over recordings; 29-class LR, C = 1)')
    reps = {'core (15)': X, 'band energies (32)': d['BE'], 'time + band (43)': d['TB'], 'order spectrum (590)': O[:, :590]}
    def ident(F, labels, pca=False):
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED); ok = 0
        for tr, te in skf.split(F, labels):
            steps = [SimpleImputer(), StandardScaler()] + ([PCA(15, random_state=SEED)] if pca else []) + \
                    [LogisticRegression(C=1.0, max_iter=3000)]
            m = make_pipeline(*steps).fit(F[tr], labels[tr]); ok += (m.predict(F[te]) == labels[te]).sum()
        return 100 * ok / len(labels)
    for name, F in reps.items():
        allid = ident(F, bear); pcaid = ident(F, bear, pca=True) if F.shape[1] > 15 else allid
        within = []
        for c in ('H', 'OR', 'IR'):
            m = y == c
            within.append(f'{c} {ident(F[m], bear[m]):.1f}% (chance {100 / len(np.unique(bear[m])):.1f}%)')
        say(f'  {name}: all {allid:.1f}% (manuscript 70.6/97.7/98.2/82.3); 15 PCs {pcaid:.1f}%; within class ' + ', '.join(within))

    # ---------------- E2-8 matched training
    say('\n[E2-8] Matched training comparison (test: the ten real-damage L8 bearings and K001)')
    rngm = np.random.default_rng(SEED)
    def acc_on(train_b, test_b, c=None):
        tr = np.isin(bear, train_b); m = fit_lr(X[tr], y[tr]); te = (bear == test_b) & ((cond == c) if c else True)
        return 100 * (m.predict(X[te]) == y[te]).mean()
    real_test = [b for b in L8_TEST if b != 'K001']
    A = {b: (acc_on(L8_TRAIN, b, 'C0'), acc_on(L8_TRAIN, b)) for b in L8_TEST}
    R = {}
    for b in real_test:
        v0, va = [], []
        for _ in range(30):
            ors = [x for x in REAL_OR if x != b]; irs = [x for x in REAL_IR if x != b]
            tb = ['K002'] + list(rngm.choice(ors, 3, replace=False)) + list(rngm.choice(irs, 3, replace=False))
            v0.append(acc_on(tb, b, 'C0')); va.append(acc_on(tb, b))
        R[b] = (np.mean(v0), np.mean(va))
    K = {b: (acc_on(L8_TRAIN + ['K004', 'K005'], b, 'C0'), acc_on(L8_TRAIN + ['K004', 'K005'], b)) for b in L8_TEST}
    say(f'  mean over ten real-damage bearings, C0 / all conditions: artificial (L8) {np.mean([A[b][0] for b in real_test]):.1f}/'
        f'{np.mean([A[b][1] for b in real_test]):.1f}%; real, same numbers (30 draws) {np.mean([R[b][0] for b in real_test]):.1f}/'
        f'{np.mean([R[b][1] for b in real_test]):.1f}%; L8 + K004 + K005 {np.mean([K[b][0] for b in real_test]):.1f}/'
        f'{np.mean([K[b][1] for b in real_test]):.1f}%')
    d8 = np.array([A[b][0] - R[b][0] for b in real_test]); lo, hi = boot_ci(d8)
    say(f'  per bearing artificial - real at C0: mean {d8.mean():.1f} pp (95% bootstrap {lo:.1f} to {hi:.1f}); '
        f'K001 recall at C0: L8 {A["K001"][0]:.0f}%, L8 + K004/K005 {K["K001"][0]:.0f}%')
    excl = np.isin(bear, ['K004', 'K005'])
    pk_, _, _ = lobo_predict(X, y, bear, extra_mask=excl); ck_ = pk_ == y
    say(f'  LOBO with K004/K005 removed from every training set: real damage {np.mean([per_bearing(ck_, bear)[b] for b in REAL_OR + REAL_IR]):.1f}% '
        f'(standard LOBO {np.mean([pb[b] for b in REAL_OR + REAL_IR]):.1f}%)')

    # ---------------- E2-9 external association
    say('\n[E2-9] External contrast association')
    ext = EXTERNAL_DEFAULT
    if EXTERNAL_CSV and os.path.exists(EXTERNAL_CSV):
        with open(EXTERNAL_CSV) as fh:
            ext = [(r['dataset'], r['race'], float(r['gamma']), float(r['accuracy'])) for r in csv.DictReader(fh)]
        say(f'  values from {EXTERNAL_CSV}')
    else:
        say('  values reconstructed from Fig. 10(b) (approximate); set PB_EXTERNAL_CSV for the exact values')
    ds = np.array([e[0] for e in ext]); g = np.array([e[2] for e in ext]); a = np.array([e[3] for e in ext])
    r, p, nn = sp(g, a)
    _, pp = perm_spearman(g, a); _, pst = perm_spearman(g, a, strata=ds); _, p1 = perm_spearman(g, a, alternative='greater')
    loo = [spearmanr(np.delete(g, i), np.delete(a, i)).statistic for i in range(len(g))]
    say(f'  pooled rho {r:.3f} (t-approx p {p:.4f}, n {nn}); permutation p {pp:.4f} (two-sided), {p1:.4f} (one-sided); '
        f'within-dataset permutation p {pst:.4f}; leave-one-out rho {min(loo):.2f} to {max(loo):.2f}')
    for dn in np.unique(ds):
        m = ds == dn; rr, _, n2 = sp(g[m], a[m])
        _, pw = perm_spearman(g[m], a[m], alternative='greater') if not np.isnan(rr) else (np.nan, np.nan)
        say(f'  within {dn}: rho {rr:.2f} (n {n2}), one-sided permutation p {pw:.3f}')
    hi_ok = a[g > 1.1]; lo_ok = a[g <= 1.1]
    say(f'  Gamma > 1.1: {np.sum(hi_ok == 100)}/{hi_ok.size} without error; Gamma <= 1.1: {np.sum(lo_ok < 100)}/{lo_ok.size} with errors')

    # ---------------- E2-10 folding
    say('\n[E2-10] Folding at C0: synchronous-average peak factor (median over recordings)')
    FD = d['FD']; m0 = cond == 'C0'
    for grp, gm in (('healthy (at BPFO)', healthy), ('outer race (BPFO)', y == 'OR'), ('inner race (BPFI)', y == 'IR')):
        mm = m0 & gm
        if mm.any():
            say(f'  {grp}: nominal period {np.nanmedian(FD[mm, 0]):.2f}, estimated period {np.nanmedian(FD[mm, 1]):.2f}, '
                f'estimated +10% {np.nanmedian(FD[mm, 2]):.2f}')
    for b in ('K001', 'KA16', 'KI18'):
        mm = m0 & (bear == b)
        if mm.any():
            say(f'  {b}: nominal {np.nanmedian(FD[mm, 0]):.2f}, estimated {np.nanmedian(FD[mm, 1]):.2f}, +10% {np.nanmedian(FD[mm, 2]):.2f}')

    open(os.path.join(OUT_DIR, 'summary_evidence_v2.txt'), 'w').write('\n'.join(OUT))
    try:
        make_figure(G, pbc, damaged_b)
    except Exception as ex:
        say(f'figure skipped: {ex}')
    print(f'\nOutputs written to {os.path.abspath(OUT_DIR)}')

# ------------------------------------------------------------------ figure: paired contrast at 1500 and 900 rpm
def make_figure(G, pbc, damaged_b):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
                         'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                         'text.color': 'black', 'axes.labelcolor': 'black', 'axes.linewidth': 0.6, 'mathtext.fontset': 'stix'})
    bs = [b for b in damaged_b if (b, 'C0') in G and (b, '900rpm') in G]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw=dict(wspace=0.32))
    col = {('artificial', 1): '#1f5a96', ('artificial', 2): '#6fa0d0', ('real', 1): '#d9822b', ('real', 2): '#3a8f5c', ('real', 3): '#3a8f5c'}
    ax = axs[0]
    for b in bs:
        o, lv = BEARINGS[b][1], BEARINGS[b][2]
        ax.plot([0, 1], [G[(b, 'C0')][0], G[(b, '900rpm')][0]], '-o', ms=3, lw=0.7, color=col[(o, min(lv, 2) if o == 'artificial' else lv)])
    ax.set_xticks([0, 1]); ax.set_xticklabels(['1500 rpm', '900 rpm']); ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel('Contrast Γ (median over references)'); ax.set_title('(a) Paired contrast', fontsize=8)
    hs = [plt.Line2D([], [], color=c, marker='o', ms=3, lw=0.7) for c in ('#1f5a96', '#6fa0d0', '#d9822b', '#3a8f5c')]
    ax.legend(hs, ['artificial, level 1', 'artificial, level 2', 'real, level 1', 'real, levels 2–3'], frameon=False, loc='best')
    ax = axs[1]
    for c, mk, cc in (('C0', 'o', '#1f5a96'), ('900rpm', '^', '#d9822b')):
        ax.scatter([G[(b, c)][0] for b in bs], [pbc[c][b] for b in bs], marker=mk, s=14, color=cc, lw=0,
                   label='1500 rpm' if c == 'C0' else '900 rpm')
    ax.set_xlabel('Contrast Γ'); ax.set_ylabel('LOBO accuracy at that condition (%)'); ax.set_title('(b) Contrast and accuracy', fontsize=8)
    ax.legend(frameon=False, loc='lower right')
    for a_ in axs:
        a_.tick_params(direction='out', length=2)
    fig.savefig(os.path.join(OUT_DIR, 'fig_speed_contrast_v2.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()
