"""ReliSense Paper B: checks requested in the second review of v11.0 (single file, Colab CPU).  Version 3.1.

Run in Colab:
  1) mount Google Drive and set PB_DATA_DIR to the folder that holds the Paderborn .mat files (sub-folders searched)
  2) !PB_DATA_DIR=/content/paderborn PB_OUT_DIR=/content/drive/MyDrive/evidence_v3_out python ReliSense_PaperB_evidence_v3_1.py
  optional: PB_WDCNN_CSV=<csv with columns bearing,acc> (per-bearing LOBO accuracy of WDCNN, in %, from phase 4)
Runtime: about 30-60 min on a Colab CPU; spectra are cached in OUT_DIR/spectra_v3_1.npz (about 150 MB).
Send back OUT_DIR/summary_evidence_v3.txt, the CSV files and fig_perturbation_v3.png.

v3.1 (fix of v3): v3 cut every recording to 4.0 s and dropped recordings shorter than 3.96 s; recordings of slightly
different length then gave spectra of different length and the stacking failed. v3.1 follows build_32.py exactly:
the WHOLE recording is used, a recording is skipped only if vibration_1, phase_current_1 or phase_current_2 is
missing, unequal in length or non-finite (2319 recordings expected), and every spectrum keeps its own frequency grid.

All features follow the ORIGINAL implementation (physics64.py): ideal band-pass 2-12 kHz in the frequency domain,
squared envelope, block mean of 16 samples, Hann window, AMPLITUDE spectrum, natural logarithm, search band
+-max(2 %, 1.5 bins), background median within +-10 Hz, measured speed if within 20 % of the set point.

  E3-0  implementation comparison: original filter vs 4th-order zero-phase Butterworth (the second implementation),
        per real-damage L8-C0 bearing and under LOBO -> does the filter explain 93.0 % vs 86.5 %?
  E3-1  coordinate perturbation on the FULL 15-feature probe (harmonics 1-3, Hz-domain bands, shaft fixed):
        10 scaled + the same 120 random sets as the v2 repetition; margins, P_rank, per-bearing bootstrap, capture
  E3-2  capture ablation: wrong sets that capture a true BPFO/BPFI line are refitted with the true-line bins removed
  E3-3  fitted-model intervention (no refit): BPFO (OR bearings) or BPFI (IR bearings) features of the test recordings
        replaced by the median of the healthy training recordings of the same condition; FTF as a control
  E3-4  contrast beyond the probe: Gamma vs detection, vs race assignment, within OR and IR, vs WDCNN (optional CSV)
  E3-5  K004/K005 variants with class contributions held fixed (class_weight='balanced') next to the unweighted ones
  E3-6  calibrated alarm threshold: P(H) threshold set by inner LOBO on the healthy TRAINING bearings for a target
        false-alarm rate (10 %), evaluated on the held-out bearing; threshold rule alongside
  E3-7  contrast over a fixed number of revolutions: C0 recordings cut to 60 revolutions (as at 900 rpm)
  E3-8  window-matched probe: features on seven 0.512-s windows per recording, mean probability (as WDCNN)
"""
import os, glob, re, sys, csv, json, warnings, time
import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')
DATA_DIR = os.environ.get('PB_DATA_DIR', '/content/drive/MyDrive/Paderborn')
OUT_DIR = os.environ.get('PB_OUT_DIR', './evidence_v3_out')
WDCNN_CSV = os.environ.get('PB_WDCNN_CSV', '')
FS, BAND, DECIM, C_LR, SEED, N_BOOT = 64000, (2000.0, 12000.0), 16, 0.3, 0, 10000
N_RAND = int(os.environ.get('PB_N_RAND', '120'))
os.makedirs(OUT_DIR, exist_ok=True)

COND = {'C0': 'N15_M07_F10', '900rpm': 'N09_M07_F10', 'lowtorque': 'N15_M01_F10', 'lowforce': 'N15_M07_F04'}
NOMINAL_RPM = {'C0': 1500, '900rpm': 900, 'lowtorque': 1500, 'lowforce': 1500}
ORDERS = {'BPFO': 3.0543, 'BPFI': 4.9457, 'BSF': 1.9966, 'FTF': 0.38178, 'shaft': 1.0}
FAMS = ['BPFO', 'BPFI', 'BSF', 'FTF', 'shaft']
FEAT = [f'{k}{h}' for k in FAMS for h in (1, 2, 3)]
FI = {n: i for i, n in enumerate(FEAT)}
CLASS_ORDER = {'OR': 'BPFO', 'IR': 'BPFI'}
BEARINGS = {
    'K001': ('H', 'healthy', 0), 'K002': ('H', 'healthy', 0), 'K003': ('H', 'healthy', 0), 'K004': ('H', 'healthy', 0),
    'K005': ('H', 'healthy', 0), 'K006': ('H', 'healthy', 0),
    'KA01': ('OR', 'artificial', 1), 'KA03': ('OR', 'artificial', 2), 'KA05': ('OR', 'artificial', 1),
    'KA06': ('OR', 'artificial', 2), 'KA07': ('OR', 'artificial', 1), 'KA08': ('OR', 'artificial', 2),
    'KA09': ('OR', 'artificial', 2), 'KI01': ('IR', 'artificial', 1), 'KI03': ('IR', 'artificial', 1),
    'KI05': ('IR', 'artificial', 1), 'KI07': ('IR', 'artificial', 2), 'KI08': ('IR', 'artificial', 2),
    'KA04': ('OR', 'real', 1), 'KA15': ('OR', 'real', 1), 'KA16': ('OR', 'real', 2), 'KA22': ('OR', 'real', 1),
    'KA30': ('OR', 'real', 1), 'KI04': ('IR', 'real', 1), 'KI14': ('IR', 'real', 1), 'KI16': ('IR', 'real', 3),
    'KI17': ('IR', 'real', 1), 'KI18': ('IR', 'real', 2), 'KI21': ('IR', 'real', 1)}
HEALTHY = [b for b in BEARINGS if BEARINGS[b][0] == 'H']
DAMAGED = [b for b in BEARINGS if BEARINGS[b][0] != 'H']
L8_TRAIN = ['K002', 'KA01', 'KA05', 'KA07', 'KI01', 'KI05', 'KI07']
L8_TEST = ['K001', 'KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']
REAL_TEST = [b for b in L8_TEST if b != 'K001']
REAL = [b for b in BEARINGS if BEARINGS[b][1] == 'real']
SCALES = [0.80, 0.85, 0.90, 0.95, 0.98, 1.02, 1.05, 1.10, 1.15, 1.20]
OUT = []


def say(*a):
    s = ' '.join(str(x) for x in a); OUT.append(s); print(s, flush=True)


# ------------------------------------------------------------------ reading
CHANNELS = ['vibration_1', 'phase_current_1', 'phase_current_2']


def find_channels(obj, found):
    """Collect every {Name, Data} pair in the nested MATLAB struct (same rule as build_32.py)."""
    if isinstance(obj, dict):
        lower = {k.lower(): k for k in obj}
        if 'name' in lower and 'data' in lower:
            arr = np.asarray(obj[lower['data']]).squeeze()
            if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                found[str(obj[lower['name']]).strip().lower()] = arr
        for v in obj.values():
            find_channels(v, found)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            find_channels(v, found)
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        for v in obj.flat:
            find_channels(v, found)


def load_recording(path, ck):
    """Same acceptance rule and speed rule as build_32.py; the whole recording is returned."""
    found = {}
    find_channels(sio.loadmat(path, simplify_cells=True), found)
    missing = [c for c in CHANNELS if c not in found]
    if missing:
        raise ValueError(f'missing {missing}')
    sig = [np.asarray(found[c], dtype=float) for c in CHANNELS]
    if len({len(s) for s in sig}) != 1 or not all(np.isfinite(s).all() for s in sig):
        raise ValueError('unequal lengths or non-finite values')
    nom = float(NOMINAL_RPM[ck]); rpm = nom
    if 'speed' in found and np.isfinite(found['speed']).all():
        med = float(np.median(np.abs(found['speed'])))
        if abs(med - nom) <= 0.2 * nom:
            rpm = med
    return sig[0], rpm / 60.0


def find_files(bearing, code):
    files = glob.glob(os.path.join(DATA_DIR, '**', f'{code}_{bearing}_*.mat'), recursive=True)
    return sorted(set(files), key=lambda p: int(re.findall(r'_(\d+)\.mat$', p)[0]))


# ------------------------------------------------------------------ signal processing (physics64.py conventions)
SOS = butter(4, [BAND[0] / (FS / 2), BAND[1] / (FS / 2)], btype='band', output='sos')


def sq_env_ideal(x):
    x = np.asarray(x, float); n = len(x); X = np.fft.rfft(x - x.mean()); f = np.fft.rfftfreq(n, 1 / FS)
    full = np.zeros(n, complex); keep = (f >= BAND[0]) & (f < BAND[1]); full[:len(f)][keep] = 2 * X[keep]
    return np.abs(np.fft.ifft(full)) ** 2


def sq_env_butter(x):
    return np.abs(hilbert(sosfiltfilt(SOS, np.asarray(x, float)))) ** 2


def amp_spectrum(e):
    m = len(e) // DECIM; e = e[:m * DECIM].reshape(m, DECIM).mean(1); e = e - e.mean()
    return np.abs(np.fft.rfft(e * np.hanning(m))).astype(np.float32), np.fft.rfftfreq(m, DECIM / FS)


def bands_idx(f, f0):
    df = f[1] - f[0]; w = max(0.02 * f0, 1.5 * df)
    s = np.abs(f - f0) <= w; g = (np.abs(f - f0) <= 10.0) & ~s
    return s, g


def feat_from_spectrum(S, f, fr, orders=None, exclude=None):
    """15 features (ln of amplitude peak-to-background) for one spectrum; orders: dict for the four families.
    exclude: list of frequencies whose +-max(2 %, 1.5 bins) neighbourhood is removed from every search band."""
    orders = orders or ORDERS; row = []
    df = f[1] - f[0]
    ex = None
    if exclude:
        ex = np.zeros(len(f), bool)
        for fx in exclude:
            ex |= np.abs(f - fx) <= max(0.02 * fx, 1.5 * df)
    for k in FAMS:
        o = ORDERS['shaft'] if k == 'shaft' else orders[k]
        for h in (1, 2, 3):
            f0 = h * o * fr
            if f0 + 10 > f[-1]:
                row.append(0.0); continue
            w = max(0.02 * f0, 1.5 * df)
            s0, s1 = max(0, int(np.ceil((f0 - w) / df - 1e-9))), int(np.floor((f0 + w) / df + 1e-9))
            g0, g1 = max(0, int(np.ceil((f0 - 10.0) / df - 1e-9))), int(np.floor((f0 + 10.0) / df + 1e-9))
            bg = np.concatenate([S[g0:s0], S[s1 + 1:g1 + 1]])
            if ex is not None and k != 'shaft':
                idx = np.arange(s0, s1 + 1); idx = idx[~ex[idx]]
                peak = S[idx].max() if idx.size else None
            else:
                peak = S[s0:s1 + 1].max() if s1 >= s0 else None
            if peak is None or bg.size == 0:
                row.append(0.0); continue
            row.append(np.log((peak + 1e-12) / (np.median(bg) + 1e-12)))
    return np.array(row, np.float32)


def window_features(x, fr, n_win=7, L=32768):
    out = []
    for i in range(n_win):
        seg = x[i * L:(i + 1) * L]
        if len(seg) < L:
            break
        S, f = amp_spectrum(sq_env_ideal(seg)); out.append(feat_from_spectrum(S, f, fr))
    return np.array(out, np.float32)


def extract():
    cache = os.path.join(OUT_DIR, 'spectra_v3_1.npz')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True); say(f'Loaded {cache}')
        return {k: z[k] for k in z.files}
    rows, SI, SB, X60, W = [], [], [], [], []
    skipped = []; t0 = time.time()
    for b in BEARINGS:
        for ck, code in COND.items():
            for p in find_files(b, code):
                try:
                    vib, fr = load_recording(p, ck)                  # whole recording, as in build_32.py
                    Si, _ = amp_spectrum(sq_env_ideal(vib)); Sb, _ = amp_spectrum(sq_env_butter(vib))
                    n60 = int(round(60 / fr * FS))                   # 60 shaft revolutions
                    S6, f6 = amp_spectrum(sq_env_ideal(vib[:min(n60, len(vib))]))
                    X60.append(feat_from_spectrum(S6, f6, fr))
                    W.append(window_features(vib, fr))
                    SI.append(Si); SB.append(Sb); rows.append((b, ck, os.path.basename(p), fr, len(vib)))
                except Exception as ex:
                    skipped.append((os.path.basename(p), str(ex)[:60]))
        say(f'  {b}: {sum(r[0] == b for r in rows)} recordings ({time.time() - t0:.0f} s)')
    if not rows:
        sys.exit(f'No .mat files found under {DATA_DIR}')

    def obj(lst):                                                   # 1-D object array of arrays of any length
        a = np.empty(len(lst), dtype=object)
        for i, v in enumerate(lst):
            a[i] = v
        return a
    d = dict(bear=np.array([r[0] for r in rows]), cond=np.array([r[1] for r in rows]), fname=np.array([r[2] for r in rows]),
             fr=np.array([r[3] for r in rows]), nsamp=np.array([r[4] for r in rows]), SI=obj(SI), SB=obj(SB),
             X60=np.array(X60), W=obj(W), skipped=obj(skipped))
    np.savez_compressed(cache, **d)
    return d


def fgrids(nsamp):
    """Frequency grid of the envelope spectrum of each recording (block mean of DECIM samples)."""
    cache = {}
    for n in np.unique(nsamp):
        cache[n] = np.fft.rfftfreq(int(n) // DECIM, DECIM / FS)
    return [cache[n] for n in nsamp]


# ------------------------------------------------------------------ models and helpers
def lr(balanced=False):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C_LR, max_iter=5000,
                                                              class_weight='balanced' if balanced else None))


def feats(S, F, fr, orders=None, exclude=None):
    """F: list with the frequency grid of each spectrum."""
    return np.array([feat_from_spectrum(S[i], F[i], fr[i], orders, exclude) for i in range(len(S))], np.float32)


def lobo(X, y, bear, keep=None, balanced=False, extra_out=None):
    pred = np.empty(len(y), dtype=object); prob = np.zeros((len(y), 3)); cls = None
    for b in BEARINGS:
        te = bear == b
        if not te.any():
            continue
        tr = ~te if keep is None else (~te & keep)
        if extra_out is not None:
            tr &= ~extra_out
        m = lr(balanced).fit(X[tr], y[tr]); pred[te] = m.predict(X[te]); prob[te] = m.predict_proba(X[te]); cls = list(m.classes_)
    return pred, prob, cls


def per_bearing(ok, bear, sel=None):
    sel = np.ones(len(bear), bool) if sel is None else sel
    return {b: 100 * ok[sel & (bear == b)].mean() for b in BEARINGS if (sel & (bear == b)).any()}


def boot(v, n=N_BOOT, seed=SEED):
    v = np.asarray(v, float); rng = np.random.default_rng(seed)
    return np.percentile(v[rng.integers(0, v.size, (n, v.size))].mean(1), [2.5, 97.5])


def l8(X, y, bear, cond, c0=True, balanced=False, extra_train=()):
    tr = np.isin(bear, L8_TRAIN + list(extra_train)); te = np.isin(bear, L8_TEST)
    if c0:
        tr &= cond == 'C0'; te &= cond == 'C0'
    m = lr(balanced).fit(X[tr], y[tr]); p = m.predict(X[te]); ok = p == y[te]
    return {b: 100 * ok[bear[te] == b].mean() for b in L8_TEST if (bear[te] == b).any()}, 100 * ok.mean()


def captures(orders, fr_ref=25.0, df=0.25):
    hits = []
    for k in ('BPFO', 'BPFI', 'BSF', 'FTF'):
        for h in (1, 2, 3):
            r = h * orders[k]; w = max(0.02 * r, 1.5 * df / fr_ref)
            for kt in ('BPFO', 'BPFI'):
                for ht in (1, 2, 3):
                    if abs(ht * ORDERS[kt] - r) <= w:
                        hits.append((k, h, kt, ht))
    return hits


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 4 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan, np.nan, int(ok.sum())
    r, p = spearmanr(a[ok], b[ok]); return r, p, int(ok.sum())


# ------------------------------------------------------------------ analyses
def main():
    d = extract()
    bear, cond, fr = d['bear'], d['cond'], d['fr']
    f = fgrids(d['nsamp'])                                          # one grid per recording
    y = np.array([BEARINGS[b][0] for b in bear]); healthy = y == 'H'
    say(f'Recordings: {len(bear)} (expected 2319); length {d["nsamp"].min()} to {d["nsamp"].max()} samples; '
        f'skipped: {len(d["skipped"])} {[tuple(s) for s in d["skipped"]]}')
    X = feats(d['SI'], f, fr); XB = feats(d['SB'], f, fr)

    # ---------------- E3-0 implementation comparison
    say('\n[E3-0] Original (ideal filter) vs Butterworth filter, same units (ln amplitude)')
    for name, Z in (('ideal filter', X), ('Butterworth', XB)):
        pb8, a8 = l8(Z, y, bear, cond)
        pl, _, _ = lobo(Z, y, bear); okl = pl == y
        say(f'  {name}: L8-C0 real-damage mean {np.mean([pb8[b] for b in REAL_TEST]):.1f}% (all L8-C0 {a8:.1f}%), '
            f'LOBO {100 * okl.mean():.1f}%, FAR {100 * (pl[healthy] != "H").mean():.1f}%')
        say('    L8-C0 per bearing: ' + ', '.join(f'{b} {pb8[b]:.0f}' for b in L8_TEST))
    rel = np.median(np.abs(X - XB), axis=0)
    say('  median |ideal - Butterworth| per feature: ' + ', '.join(f'{n} {v:.2f}' for n, v in zip(FEAT, rel)))

    # ---------------- E3-1 perturbation on the full probe
    say('\n[E3-1] Coordinate perturbation on the full 15-feature probe (Hz-domain bands, harmonics 1-3, shaft fixed)')
    rngo = np.random.default_rng(SEED)
    rand = [dict(zip(('BPFO', 'BPFI', 'BSF', 'FTF'), rngo.uniform(0.3, 5.9, 4))) for _ in range(N_RAND)]
    sets = [('true', dict(ORDERS))] + [(f's={s:.2f}', {k: s * ORDERS[k] for k in ('BPFO', 'BPFI', 'BSF', 'FTF')}) for s in SCALES] \
        + [(f'rand{j + 1}', r) for j, r in enumerate(rand)]
    l8tr = np.isin(bear, L8_TRAIN) & (cond == 'C0'); l8te = np.isin(bear, L8_TEST) & (cond == 'C0')
    res = {}; t0 = time.time()
    for name, o in sets:
        Z = X if name == 'true' else feats(d['SI'], f, fr, o)
        m = lr().fit(Z[l8tr], y[l8tr]); ok8 = m.predict(Z[l8te]) == y[l8te]
        pl, _, _ = lobo(Z, y, bear); okl = pl == y
        res[name] = dict(o=o, l8=100 * ok8.mean(), lobo=100 * okl.mean(), cap=captures(o),
                         l8b={b: 100 * ok8[bear[l8te] == b].mean() for b in L8_TEST}, lobob=per_bearing(okl, bear))
        if len(res) % 20 == 0:
            say(f'    {len(res)}/{len(sets)} sets ({time.time() - t0:.0f} s)')
    wrong = [n for n, _ in sets[1:31]]; r20 = [n for n, _ in sets[11:31]]; rall = [n for n, _ in sets[11:]]
    for proto, key in (('L8-C0', 'l8'), ('LOBO', 'lobo')):
        a0 = res['true'][key]; best = max(wrong, key=lambda n: res[n][key])
        k_all = sum(res[n][key] >= a0 for n in rall)
        bkey = key + 'b'; bs = list(res['true'][bkey])
        dR = np.array([res['true'][bkey][b] - np.mean([res[n][bkey][b] for n in r20]) for b in bs])
        dB = np.array([res['true'][bkey][b] - res[best][bkey][b] for b in bs])
        say(f'  {proto}: true {a0:.1f}%; best of 30 wrong {best} {res[best][key]:.1f}% (captures {len(res[best]["cap"])} lines); '
            f'mean of 20 random {np.mean([res[n][key] for n in r20]):.1f}%; random sets >= true: {k_all} of {len(rall)} '
            f'(P_rank {(1 + k_all) / (len(rall) + 1):.3f})')
        say(f'    per bearing: true - mean random {dR.mean():.1f} pp ({boot(dR)[0]:.1f} to {boot(dR)[1]:.1f}); '
            f'true - best wrong {dB.mean():.1f} pp ({boot(dB)[0]:.1f} to {boot(dB)[1]:.1f})')
    cap = [n for n in rall if res[n]['cap']]
    say(f'  random sets that capture a true BPFO/BPFI line: {len(cap)} of {len(rall)}; mean LOBO {np.mean([res[n]["lobo"] for n in cap]) if cap else np.nan:.1f}% '
        f'vs {np.mean([res[n]["lobo"] for n in rall if n not in cap]):.1f}%')
    with open(os.path.join(OUT_DIR, 'perturbation_full_v3.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['set', 'BPFO', 'BPFI', 'BSF', 'FTF', 'acc_L8C0', 'acc_LOBO', 'captures'])
        for n, r in res.items():
            w.writerow([n] + [f'{r["o"][k]:.4f}' for k in ('BPFO', 'BPFI', 'BSF', 'FTF')] + [f'{r["l8"]:.2f}', f'{r["lobo"]:.2f}',
                       ';'.join(f'{a}{b}~{c}{e}' for a, b, c, e in r['cap'])])
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
        for ax, key, lab in ((axs[0], 'l8', '(a) L8-C0'), (axs[1], 'lobo', '(b) LOBO')):
            rv = np.array([res[n][key] for n in rall]); cv = np.array([bool(res[n]['cap']) for n in rall])
            jit = np.random.default_rng(1).uniform(-0.15, 0.15, len(rv))
            ax.scatter(2 + jit[~cv], rv[~cv], s=8, c='#d07840', label='random')
            ax.scatter(2 + jit[cv], rv[cv], s=10, c='#7b5aa6', marker='D', label='random, captures a true line')
            ax.plot(SCALES, [res[f's={s:.2f}'][key] for s in SCALES], 'o', mfc='white', mec='k', ms=4, label='scaled')
            ax.plot(1.0, res['true'][key], 'o', c='#083868', ms=7, label='true orders')
            ax.set_xticks([0.8, 1.0, 1.2, 2]); ax.set_xticklabels(['0.8', '1.0', '1.2', 'random']); ax.set_title(lab, loc='left', fontsize=9)
            ax.set_ylabel('Accuracy (%)', fontsize=8); ax.tick_params(labelsize=7)
        axs[1].legend(fontsize=6, loc='lower right'); fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, 'fig_perturbation_v3.png'), dpi=600); plt.close(fig)
    except Exception as ex:
        say('  (figure skipped:', ex, ')')

    # ---------------- E3-2 capture ablation
    say('\n[E3-2] Capture ablation: capturing sets refitted with the true BPFO/BPFI harmonic bins removed')
    for n in sorted(cap, key=lambda n: -res[n]['lobo'])[:8]:
        exl = [[ht * ORDERS[kt] * fr[i] for kt in ('BPFO', 'BPFI') for ht in (1, 2, 3)] for i in range(len(bear))]
        Z = np.array([feat_from_spectrum(d['SI'][i], f[i], fr[i], res[n]['o'], exl[i]) for i in range(len(bear))], np.float32)
        m = lr().fit(Z[l8tr], y[l8tr]); a8 = 100 * (m.predict(Z[l8te]) == y[l8te]).mean()
        pl, _, _ = lobo(Z, y, bear); al = 100 * (pl == y).mean()
        say(f'  {n}: L8-C0 {res[n]["l8"]:.1f} -> {a8:.1f}%, LOBO {res[n]["lobo"]:.1f} -> {al:.1f}% (captures {res[n]["cap"]})')

    # ---------------- E3-3 fitted-model intervention
    say('\n[E3-3] Fitted-model intervention (LOBO models fitted once, test features replaced, no refit)')
    acc = {'none': [], 'fault order -> healthy': [], 'FTF -> healthy': []}
    for b in DAMAGED:
        te = bear == b; tr = ~te; m = lr().fit(X[tr], y[tr]); k = CLASS_ORDER[BEARINGS[b][0]]
        Xi = X[te].copy(); Xf = X[te].copy()
        for c in np.unique(cond[te]):
            hm = np.median(X[tr & healthy & (cond == c)], axis=0); sel = cond[te] == c
            for h in (1, 2, 3):
                Xi[sel, FI[f'{k}{h}']] = hm[FI[f'{k}{h}']]; Xf[sel, FI[f'FTF{h}']] = hm[FI[f'FTF{h}']]
        acc['none'].append(100 * (m.predict(X[te]) == y[te]).mean())
        acc['fault order -> healthy'].append(100 * (m.predict(Xi) == y[te]).mean())
        acc['FTF -> healthy'].append(100 * (m.predict(Xf) == y[te]).mean())
    for kname, v in acc.items():
        say(f'  {kname}: mean accuracy of the 23 damaged bearings {np.mean(v):.1f}%')
    dI = np.array(acc['none']) - np.array(acc['fault order -> healthy'])
    say(f'  per-bearing drop when the fault-order features are set to healthy values: {dI.mean():.1f} pp '
        f'({boot(dI)[0]:.1f} to {boot(dI)[1]:.1f}); bearings with a drop > 20 pp: '
        + ', '.join(b for b, v in zip(DAMAGED, dI) if v > 20))

    # ---------------- E3-4 contrast beyond the probe
    say('\n[E3-4] Contrast (original units, median over the six healthy bearings, C0) vs detection and race')
    pl, _, _ = lobo(X, y, bear)
    G = {}
    for b in DAMAGED:
        k = CLASS_ORDER[BEARINGS[b][0]]; sel = (bear == b) & (cond == 'C0')
        g = [np.mean([X[sel, FI[f'{k}{h}']].mean() - X[(bear == r) & (cond == 'C0'), FI[f'{k}{h}']].mean() for h in (1, 2, 3)]) for r in HEALTHY]
        det = 100 * (pl[sel] != 'H').mean(); race = 100 * (pl[sel] == y[sel]).sum() / max((pl[sel] != 'H').sum(), 1)
        G[b] = (np.median(g), det, race, 100 * (pl[sel] == y[sel]).mean())
    bs = list(G)
    for j, lab in ((3, 'accuracy'), (1, 'detection'), (2, 'race given detection')):
        r, p, n = sp([G[b][0] for b in bs], [G[b][j] for b in bs]); say(f'  Gamma vs {lab}: rho {r:.2f} (p {p:.3f}, n {n})')
    for c in ('OR', 'IR'):
        bc = [b for b in bs if BEARINGS[b][0] == c]; r, p, n = sp([G[b][0] for b in bc], [G[b][3] for b in bc])
        say(f'  within {c}: rho(Gamma, accuracy) {r:.2f} (p {p:.3f}, n {n})')
    if WDCNN_CSV and os.path.exists(WDCNN_CSV):
        wd = {row['bearing']: float(row['acc']) for row in csv.DictReader(open(WDCNN_CSV))}
        bw = [b for b in bs if b in wd]; r, p, n = sp([G[b][0] for b in bw], [wd[b] for b in bw])
        say(f'  Gamma vs WDCNN accuracy: rho {r:.2f} (p {p:.3f}, n {n})')
    with open(os.path.join(OUT_DIR, 'contrast_original_units_v3.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['bearing', 'gamma_C0', 'detection_C0', 'race_given_detection_C0', 'accuracy_C0'])
        for b in bs:
            w.writerow([b] + [f'{v:.3f}' for v in G[b]])

    # ---------------- E3-5 class-balanced K004/K005 variants
    say('\n[E3-5] K004/K005 variants, unweighted and with class contributions held fixed (balanced)')
    for bal in (False, True):
        tag = 'balanced' if bal else 'unweighted'
        p0, _ = l8(X, y, bear, cond, balanced=bal); p1, _ = l8(X, y, bear, cond, balanced=bal, extra_train=('K004', 'K005'))
        pa, _, _ = lobo(X, y, bear, balanced=bal); pbx, _, _ = lobo(X, y, bear, balanced=bal, extra_out=np.isin(bear, ['K004', 'K005']))
        ra = np.mean([per_bearing(pa == y, bear)[b] for b in REAL]); rb = np.mean([per_bearing(pbx == y, bear)[b] for b in REAL])
        say(f'  {tag}: L8-C0 real damage {np.mean([p0[b] for b in REAL_TEST]):.1f} -> {np.mean([p1[b] for b in REAL_TEST]):.1f}% '
            f'with K004/K005; K001 recall {p0["K001"]:.0f} -> {p1["K001"]:.0f}%; LOBO real damage {ra:.1f} -> {rb:.1f}% '
            f'without K004/K005; LOBO FAR {100 * (pa[healthy] != "H").mean():.1f}%')

    # ---------------- E3-6 calibrated alarm threshold
    say('\n[E3-6] Alarm threshold on P(H) set by inner LOBO on the healthy training bearings (target FAR 10 %)')
    alpha = 0.10; flag = np.zeros(len(y), bool)
    for b in BEARINGS:
        te = bear == b; tr = ~te
        inner = np.zeros(tr.sum()); bt = bear[tr]; yt = y[tr]; Xt = X[tr]
        for b2 in np.unique(bt):
            t2 = bt == b2; m2 = lr().fit(Xt[~t2], yt[~t2]); ci = list(m2.classes_).index('H')
            inner[t2] = m2.predict_proba(Xt[t2])[:, ci]
        thr = np.quantile(inner[yt == 'H'], alpha)                    # P(H) below thr -> alarm
        m = lr().fit(X[tr], y[tr]); ph = m.predict_proba(X[te])[:, list(m.classes_).index('H')]
        flag[te] = ph < thr
    far = 100 * flag[healthy].mean(); det = 100 * flag[~healthy].mean()
    say(f'  calibrated probe: FAR on held-out healthy bearings {far:.1f}% (target 10 %); per bearing '
        + ', '.join(f'{b} {100 * flag[bear == b].mean():.0f}%' for b in HEALTHY) + f'; damaged recordings detected {det:.1f}%')
    tflag = np.zeros(len(y), bool)
    for b in BEARINGS:
        te = bear == b; htr = (~te) & healthy
        to, ti = np.percentile(X[htr, FI['BPFO1']], 95), np.percentile(X[htr, FI['BPFI1']], 95)
        tflag[te] = (X[te, FI['BPFO1']] > to) | (X[te, FI['BPFI1']] > ti)
    say(f'  threshold rule (95th percentiles): FAR {100 * tflag[healthy].mean():.1f}%, damaged detected {100 * tflag[~healthy].mean():.1f}%')

    # ---------------- E3-7 contrast over a fixed number of revolutions
    say('\n[E3-7] Contrast with 60 revolutions at both speeds (C0 recordings cut to 60 revolutions)')
    X60 = d['X60']
    def gamma(Z, c):
        out = {}
        for b in DAMAGED:
            k = CLASS_ORDER[BEARINGS[b][0]]; sel = (bear == b) & (cond == c)
            if not sel.any():
                continue
            out[b] = np.median([np.mean([Z[sel, FI[f'{k}{h}']].mean() - Z[(bear == r) & (cond == c), FI[f'{k}{h}']].mean()
                                         for h in (1, 2, 3)]) for r in HEALTHY if ((bear == r) & (cond == c)).any()])
        return out
    g15, g15c, g9 = gamma(X, 'C0'), gamma(X60, 'C0'), gamma(X, '900rpm')
    for name, sel in (('all damaged', lambda b: True), ('artificial', lambda b: BEARINGS[b][1] == 'artificial'),
                      ('real level 1', lambda b: BEARINGS[b][1] == 'real' and BEARINGS[b][2] == 1)):
        bb = [b for b in g9 if sel(b) and b in g15c]
        a, c_, z9 = np.array([g15[b] for b in bb]), np.array([g15c[b] for b in bb]), np.array([g9[b] for b in bb])
        p = wilcoxon(c_, z9).pvalue if len(bb) >= 5 else np.nan
        say(f'  {name} (n {len(bb)}): median Gamma 1500 rpm 4 s {np.median(a):.2f}, 1500 rpm 60 rev {np.median(c_):.2f}, '
            f'900 rpm 60 rev {np.median(z9):.2f}; median paired change at 60 rev {np.median(z9 - c_):+.2f} (Wilcoxon p {p:.3f})')

    # ---------------- E3-8 window-matched probe
    say('\n[E3-8] Probe on seven 0.512-s windows per recording, mean window probability (as WDCNN)')
    nw = np.array([len(w) for w in d['W']]); has = nw > 0
    Xw = np.concatenate([np.asarray(w, np.float32).reshape(-1, len(FEAT)) for w in d['W']])
    yw = np.repeat(y, nw); bw_ = np.repeat(bear, nw); rid = np.repeat(np.arange(len(y)), nw)
    okw = np.zeros(len(y), bool)
    for b in BEARINGS:
        tew = bw_ == b
        if not tew.any():
            continue
        m = lr().fit(Xw[~tew], yw[~tew]); P = m.predict_proba(Xw[tew]); rr = rid[tew]
        for r in np.unique(rr):
            okw[r] = m.classes_[P[rr == r].mean(0).argmax()] == y[r]
    say(f'  windows per recording {nw.min()} to {nw.max()}; LOBO accuracy {100 * okw[has].mean():.1f}% '
        f'(full-record probe {100 * (lobo(X, y, bear)[0] == y).mean():.1f}%)')

    with open(os.path.join(OUT_DIR, 'summary_evidence_v3.txt'), 'w') as fh:
        fh.write('\n'.join(OUT) + '\n')
    say(f'\nWritten: {OUT_DIR}/summary_evidence_v3.txt')


if __name__ == '__main__':
    main()
