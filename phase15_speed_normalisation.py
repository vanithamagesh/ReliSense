"""Phase 15 (v1.0): speed normalisation on real Paderborn recordings (CPU, a few minutes per bearing).

For KA16 (outer race), KI18 (inner race) and K001 (healthy), all 20 recordings of the four operating conditions
N09_M07_F10 (900 rpm), N15_M07_F10, N15_M01_F10 and N15_M07_F04 (1500 rpm) are processed with the functions of
physics64.py (phase 1), unchanged. For every recording it stores
  - the measured speed (median rpm) and the 15 features with the measured shaft frequency (as in ReliSense),
  - the 15 features read at FIXED frequencies of the nominal 1500 rpm (no speed normalisation),
  - for recording _1 only: the squared-envelope spectrum up to 500 Hz and the order spectrum.
The archives <bearing>.rar are reused from --archives (kept there by phase 14; downloaded once if missing).
Output: <root>/checkpoints/phase15/<bearing>_speed.npz (reused on a second run; --fresh recomputes).
Colab:  !pip -q install libarchive-c
        !python /content/drive/MyDrive/ReliSense_study/phase15_speed_normalisation.py --root /content/drive/MyDrive/ReliSense_study
"""
import argparse, io, os, subprocess
from pathlib import Path
import numpy as np
from scipy.io import loadmat

BASE = 'https://groups.uni-paderborn.de/kat/BearingDataCenter/'; FS = 64000
COND = ['N09_M07_F10', 'N15_M07_F10', 'N15_M01_F10', 'N15_M07_F04']; NOMINAL_RPM = {'N09': 900.0, 'N15': 1500.0}
ap = argparse.ArgumentParser(); ap.add_argument('--root', default='/content/drive/MyDrive/ReliSense_study')
ap.add_argument('--bearings', default='KA16,KI18,K001'); ap.add_argument('--archives', default=None); ap.add_argument('--fresh', action='store_true')
ap.add_argument('--selftest', action='store_true', help='run on a synthetic signal only (no data needed)')
a = ap.parse_args()
ARCH = a.archives or a.root + '/paderborn_archives'; CK = a.root + '/checkpoints/phase15'

# ------------------------------------------------------------------ physics64.py (phase 1), unchanged
GEOMETRY_6203 = {'n': 8, 'd': 6.75, 'D': 28.55}
ORDER_GRID = np.round(np.arange(0.2, 12.0 + 1e-9, 0.02), 4)
def fault_orders(geom=GEOMETRY_6203):
    q = geom['d'] / geom['D']; n = geom['n']
    return {'BPFO': n / 2 * (1 - q), 'BPFI': n / 2 * (1 + q), 'BSF': geom['D'] / (2 * geom['d']) * (1 - q ** 2), 'FTF': 0.5 * (1 - q), 'shaft': 1.0}
def band_envelope(X, f, lo, hi, n):
    full = np.zeros(n, dtype=complex); keep = (f >= lo) & (f < hi); full[:len(f)][keep] = 2 * X[keep]; return np.fft.ifft(full)
def peak_features(S, f, fr, orders=None):
    orders = orders or fault_orders(); df = f[1] - f[0]; row = []
    for o in orders.values():
        for h in (1, 2, 3):
            f0 = h * o * fr; w = max(0.02 * f0, 1.5 * df); band = np.abs(f - f0) <= w; bg = (np.abs(f - f0) <= 10) & ~band
            row.append(np.log((S[band].max() + 1e-12) / (np.median(S[bg]) + 1e-12)))
    return np.array(row, dtype='float32')
def order_spectrum(S, f, fr):
    s = np.interp(ORDER_GRID * fr, f, S); return np.log((s + 1e-12) / (np.median(s) + 1e-12)).astype('float32')


def ses(x):
    n = len(x); f = np.fft.rfftfreq(n, 1 / FS); X = np.fft.rfft(x - x.mean())
    e = np.abs(band_envelope(X, f, 2000.0, 12000.0, n)) ** 2
    m = len(e) // 16; eb = e[:m * 16].reshape(m, 16).mean(1); eb = eb - eb.mean()
    return np.abs(np.fft.rfft(eb * np.hanning(m))), np.fft.rfftfreq(m, 16 / FS)


def process(x, spd, cond):
    nominal = NOMINAL_RPM[cond[:3]]
    rpm = float(np.median(np.abs(spd))) if spd is not None and np.isfinite(spd).all() and abs(np.median(np.abs(spd)) - nominal) <= 0.2 * nominal else nominal
    S, fe = ses(x)
    return rpm, S, fe, peak_features(S, fe, rpm / 60.0), peak_features(S, fe, 1500.0 / 60.0)


def find_channels(obj, found):
    if isinstance(obj, dict):
        lower = {k.lower(): k for k in obj}
        if 'name' in lower and 'data' in lower:
            arr = np.asarray(obj[lower['data']]).squeeze()
            if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number): found[str(obj[lower['name']]).strip().lower()] = arr
        for v in obj.values(): find_channels(v, found)
    elif isinstance(obj, (list, tuple)):
        for v in obj: find_channels(v, found)
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        for v in obj.flat: find_channels(v, found)


if a.selftest:     # synthetic outer-race impacts at BPFO of 900 and 1500 rpm: the normalised feature must stay high
    rng = np.random.default_rng(0); t = np.arange(4 * FS) / FS; hres = np.exp(-800 * t[:200]) * np.sin(2 * np.pi * 5000 * t[:200])
    for cond in ('N09_M07_F10', 'N15_M07_F10'):
        rpm0 = NOMINAL_RPM[cond[:3]]; fd = fault_orders()['BPFO'] * rpm0 / 60; x = rng.normal(0, 0.3, t.size)
        for ti in np.arange(0.01, 3.99, 1 / fd): i = int(ti * FS); x[i:i + 200] += hres
        rpm, S, fe, Fn, Ff = process(x, np.full(16000, rpm0), cond)
        print(f'selftest {cond}: rpm {rpm:.0f}, BPFO h1 normalised {Fn[0]:.2f}, fixed-1500 {Ff[0]:.2f}')
    raise SystemExit
os.makedirs(CK, exist_ok=True); os.makedirs(ARCH, exist_ok=True)
for bearing in [b.strip() for b in a.bearings.split(',') if b.strip()]:
    out = f'{CK}/{bearing}_speed.npz'
    if os.path.exists(out) and not a.fresh: print(f'{bearing}: already saved in {out}, reused', flush=True); continue
    archive = Path(ARCH) / f'{bearing}.rar'
    if not archive.exists():
        print(f'{bearing}: downloading archive', flush=True)
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--retry', '3', BASE + archive.name, '--output', str(archive)], check=True)
    import libarchive
    rows = []; spec = {}
    with libarchive.file_reader(str(archive)) as entries:
        for e in entries:
            name = Path(e.pathname.replace('\\', '/')).name
            if not name.endswith('.mat') or not any(name.startswith(c + '_') for c in COND): continue
            cond = name[:11]; rec = int(name.rsplit('_', 1)[1].split('.')[0])
            found = {}; find_channels(loadmat(io.BytesIO(b''.join(e.get_blocks())), simplify_cells=True), found)
            x = np.asarray(found['vibration_1'], float); spd = found.get('speed'); spd = None if spd is None else np.asarray(spd, float)
            rpm, S, fe, Fn, Ff = process(x, spd, cond); rows.append((cond, rec, rpm, Fn, Ff))
            if rec == 1:
                m = fe <= 500; spec[cond] = dict(f=fe[m].astype(np.float32), S=S[m].astype(np.float32), O=order_spectrum(S, fe, rpm / 60.0))
            print(f'  {bearing} {cond}_{rec}: rpm {rpm:.1f}, BPFO h1 {Fn[0]:.2f} (fixed 1500 rpm: {Ff[0]:.2f}), BPFI h1 {Fn[3]:.2f} ({Ff[3]:.2f})', flush=True)
    rows.sort(key=lambda r: (r[0], r[1]))
    dct = dict(bearing=bearing, cond=np.array([r[0] for r in rows]), rec=np.array([r[1] for r in rows]), rpm=np.array([r[2] for r in rows]),
               feat_norm=np.stack([r[3] for r in rows]), feat_fixed=np.stack([r[4] for r in rows]), order_grid=ORDER_GRID.astype(np.float32),
               fault_orders=np.array(list(fault_orders().values()), np.float32))
    for c, v in spec.items():
        for k, arr in v.items(): dct[f'{k}_{c}'] = arr
    np.savez_compressed(out, **dct); print(f'{bearing}: {len(rows)} recordings, saved {out} ({os.path.getsize(out) / 1e6:.2f} MB)', flush=True)
print('done; archives kept in', ARCH)
