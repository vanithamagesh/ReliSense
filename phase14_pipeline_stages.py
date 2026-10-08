"""Phase 14 (v1.0): real output of every stage of Module I for one recording (CPU, about 2 minutes).

Recording: N15_M07_F10_<bearing>_1.mat (default KA16, real outer-race damage, condition 0; KI18 and K001 optional).
The archive <bearing>.rar is downloaded once from the Paderborn KAt-DataCenter and KEPT in --archives, and the
stage outputs are saved to <root>/checkpoints/phase14/<bearing>_stages.npz; a second run reuses both.
The stages use exactly the functions of physics64.py (phase 1), copied below unchanged:
  1 raw vibration (64 kHz)                 5 measured shaft speed n(t) and f_r
  2 spectrum of the raw signal, 2-12 kHz   6 squared-envelope spectrum on the order axis
  3 squared envelope |x_a(t)|^2            7 peak and background windows at h*k*f_r
  4 block average of 16 samples, Hann     8 the 15 features
Colab:  !pip -q install libarchive-c   (if it fails: the script falls back to 7z, which Colab provides)
        !python phase14_pipeline_stages.py --root /content/drive/MyDrive/ReliSense_study --bearings KA16,KI18,K001
"""
import argparse, io, os, re, subprocess, tempfile
from pathlib import Path
import numpy as np
from scipy.io import loadmat

BASE = 'https://groups.uni-paderborn.de/kat/BearingDataCenter/'; FS = 64000
ap = argparse.ArgumentParser(); ap.add_argument('--root', default='/content/drive/MyDrive/ReliSense_study')
ap.add_argument('--bearings', default='KA16'); ap.add_argument('--archives', default=None); ap.add_argument('--fresh', action='store_true')
a = ap.parse_args()
ARCH = a.archives or a.root + '/paderborn_archives'; CK = a.root + '/checkpoints/phase14'
os.makedirs(ARCH, exist_ok=True); os.makedirs(CK, exist_ok=True)

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


def read_mat(bearing, stem):
    archive = Path(ARCH) / f'{bearing}.rar'
    if not archive.exists():
        print(f'{bearing}: downloading archive (kept in {ARCH})', flush=True)
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--retry', '3', BASE + archive.name, '--output', str(archive)], check=True)
    else:
        print(f'{bearing}: archive already present, not downloaded', flush=True)
    try:
        import libarchive
        with libarchive.file_reader(str(archive)) as entries:
            for e in entries:
                if Path(e.pathname.replace('\\', '/')).name.lower() == stem.lower() + '.mat': return b''.join(e.get_blocks())
    except ImportError:
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(['7z', 'e', str(archive), f'-o{td}', f'*{stem}.mat', '-r', '-y'], check=True, stdout=subprocess.DEVNULL)
            return Path(td, stem + '.mat').read_bytes()
    raise FileNotFoundError(stem)


for bearing in [b.strip() for b in a.bearings.split(',') if b.strip()]:
    out_path = f'{CK}/{bearing}_stages.npz'
    if os.path.exists(out_path) and not a.fresh:
        print(f'{bearing}: stages already saved in {out_path}, reused', flush=True); continue
    stem = f'N15_M07_F10_{bearing}_1'
    found = {}; find_channels(loadmat(io.BytesIO(read_mat(bearing, stem)), simplify_cells=True), found)
    x = np.asarray(found['vibration_1'], float); n = len(x)
    spd = np.asarray(found.get('speed', np.full(16000, np.nan)), float)
    rpm = float(np.median(np.abs(spd))) if np.isfinite(spd).all() and abs(np.median(np.abs(spd)) - 1500) <= 300 else 1500.0
    fr = rpm / 60.0
    # 1-2: raw signal and its spectrum, band-limited
    f = np.fft.rfftfreq(n, 1 / FS); X = np.fft.rfft(x - x.mean())
    mag = np.abs(X); nb = 2000; edges = np.linspace(0, len(f), nb + 1).astype(int)
    spec_f = np.array([f[edges[i]:edges[i + 1]].mean() for i in range(nb)]); spec_m = np.array([mag[edges[i]:edges[i + 1]].max() for i in range(nb)])
    # 3: squared envelope of the 2-12 kHz band
    c = band_envelope(X, f, 2000.0, 12000.0, n); e = np.abs(c) ** 2
    # 4: block average of 16 samples, Hann window, spectrum (envelope_spectrum of physics64)
    m = len(e) // 16; eb = e[:m * 16].reshape(m, 16).mean(1); eb0 = eb - eb.mean(); hann = np.hanning(m)
    S = np.abs(np.fft.rfft(eb0 * hann)); fe = np.fft.rfftfreq(m, 16 / FS)
    # 6-8
    orders = ORDER_GRID; O = order_spectrum(S, fe, fr); F = peak_features(S, fe, fr)
    fo = fault_orders(); f0 = fo['BPFO'] * fr; w = max(0.02 * f0, 1.5 * (fe[1] - fe[0]))
    seg = int(0.1 * FS)
    np.savez_compressed(out_path, bearing=bearing, recording=stem, fs=FS, rpm=rpm, fr=fr,
        raw=x[:seg].astype(np.float32), raw_full_rms=np.float32(x.std()),
        spec_f=spec_f.astype(np.float32), spec_mag=spec_m.astype(np.float32),
        bandpassed=np.real(c[:seg]).astype(np.float32), env_sq=e[:seg].astype(np.float32),
        env_block=eb[:int(0.1 * FS / 16)].astype(np.float32), hann=hann.astype(np.float32),
        ses_f=fe.astype(np.float32), ses=S.astype(np.float32),
        speed=spd.astype(np.float32), order_grid=orders.astype(np.float32), order_spec=O,
        bpfo_f0=np.float32(f0), bpfo_halfwidth=np.float32(w), features=F,
        feature_names=np.array([f'{k}_h{h}' for k in fo for h in (1, 2, 3)]), fault_orders=np.array([fo[k] for k in fo], np.float32))
    print(f'{bearing}: rpm {rpm:.1f}, BPFO h1 feature {F[0]:.3f}; saved {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB)', flush=True)
print('done; archives kept in', ARCH)
