"""Physics features from the raw 64 kHz vibration of one Paderborn recording (Phase 1, v1.0).

Four envelope variants are computed for every recording:
  fixed   - fixed band 2-12 kHz on the raw signal (same band as the v1.3/v1.4 envelope data)
  sk      - band with the largest envelope kurtosis (dyadic-band kurtogram, levels 1-4, up to fmax)
  cpw     - cepstral pre-whitening (unit spectral magnitude), then the fixed band
  cpw_sk  - cepstral pre-whitening, then the kurtogram band
For each variant we keep
  15 features: log peak-to-background ratio of the squared-envelope spectrum at harmonics 1-3 of
               BPFO, BPFI, BSF, FTF and the shaft frequency (search +-2 %, background = median within +-10 Hz);
  an order spectrum: log squared-envelope spectrum on a fixed order grid, divided by its median.
All quantities are invariant to the scale and offset of the signal.
"""
from __future__ import annotations
import numpy as np

GEOMETRY_6203 = {'n': 8, 'd': 6.75, 'D': 28.55}
VARIANTS = ('fixed', 'sk', 'cpw', 'cpw_sk')
ORDER_GRID = np.round(np.arange(0.2, 12.0 + 1e-9, 0.02), 4)  # 590 bins


def fault_orders(geom=GEOMETRY_6203):
    q = geom['d'] / geom['D']; n = geom['n']
    return {'BPFO': n / 2 * (1 - q), 'BPFI': n / 2 * (1 + q), 'BSF': geom['D'] / (2 * geom['d']) * (1 - q ** 2),
            'FTF': 0.5 * (1 - q), 'shaft': 1.0}


FEATURE_NAMES = [f'{k}_h{h}' for k in fault_orders() for h in (1, 2, 3)]


def cepstral_prewhiten(x):
    """Keep the phase, set every spectral magnitude to one (Randall and Sawalhi, 2011)."""
    X = np.fft.rfft(x - x.mean())
    return np.fft.irfft(X / (np.abs(X) + 1e-12), n=len(x))


def band_envelope(X, f, lo, hi, n):
    """Complex envelope of the band [lo, hi] from the one-sided spectrum X (brick-wall filter)."""
    full = np.zeros(n, dtype=complex)
    keep = (f >= lo) & (f < hi)
    full[:len(f)][keep] = 2 * X[keep]
    return np.fft.ifft(full)


def kurtogram_band(X, f, n, fmax, fmin=500.0, levels=(1, 2, 3, 4)):
    """Band with the largest kurtosis of the complex envelope. Bands below fmin are skipped
    (they hold shaft and electrical lines, not bearing resonances)."""
    best = (-np.inf, (2000.0, min(12000.0, fmax)))
    for lev in levels:
        bw = fmax / 2 ** lev
        for k in range(2 ** lev):
            lo, hi = k * bw, (k + 1) * bw
            if lo < fmin:
                continue
            c = band_envelope(X, f, lo, hi, n)
            p = np.abs(c) ** 2
            kur = np.mean(p ** 2) / (np.mean(p) ** 2 + 1e-30) - 2
            if kur > best[0]:
                best = (kur, (lo, hi))
    return best[1], float(best[0])


def envelope_spectrum(c, fs, decim):
    """Spectrum of the squared envelope, after averaging blocks of `decim` samples (envelope is low-pass)."""
    e = np.abs(c) ** 2
    m = len(e) // decim
    e = e[:m * decim].reshape(m, decim).mean(1)
    e = e - e.mean()
    S = np.abs(np.fft.rfft(e * np.hanning(m)))
    return S, np.fft.rfftfreq(m, decim / fs)


def peak_features(S, f, fr, orders=None):
    orders = orders or fault_orders()
    df = f[1] - f[0]; row = []
    for o in orders.values():
        for h in (1, 2, 3):
            f0 = h * o * fr; w = max(0.02 * f0, 1.5 * df)
            band = np.abs(f - f0) <= w; bg = (np.abs(f - f0) <= 10) & ~band
            row.append(np.log((S[band].max() + 1e-12) / (np.median(S[bg]) + 1e-12)))
    return np.array(row, dtype='float32')


def order_spectrum(S, f, fr):
    s = np.interp(ORDER_GRID * fr, f, S)
    return np.log((s + 1e-12) / (np.median(s) + 1e-12)).astype('float32')


def recording_physics(vib, fs, rpm, fmax=12000.0, fixed=(2000.0, 12000.0), decim=16):
    """All physics outputs of one recording. vib: 1-D raw vibration at fs (64 kHz)."""
    x = np.asarray(vib, dtype=float); n = len(x); fr = rpm / 60.0
    fmax = min(fmax, fs / 2); fixed = (fixed[0], min(fixed[1], fmax))
    f = np.fft.rfftfreq(n, 1 / fs)
    out = {'feat': {}, 'order': {}, 'band': {}, 'kurt': {}}
    if not np.any(x - x.mean()):
        for v in VARIANTS:
            out['feat'][v] = np.zeros(len(FEATURE_NAMES), 'float32')
            out['order'][v] = np.zeros(len(ORDER_GRID), 'float32')
            out['band'][v] = (0.0, 0.0); out['kurt'][v] = 0.0
        return out
    spectra = {'raw': np.fft.rfft(x - x.mean()), 'cpw': np.fft.rfft(cepstral_prewhiten(x))}
    for v in VARIANTS:
        X = spectra['cpw' if v.startswith('cpw') else 'raw']
        if v.endswith('sk'):
            (lo, hi), kur = kurtogram_band(X, f, n, fmax)
        else:
            lo, hi = fixed; kur = float('nan')
        c = band_envelope(X, f, lo, hi, n)
        S, fe = envelope_spectrum(c, fs, decim)
        out['feat'][v] = peak_features(S, fe, fr)
        out['order'][v] = order_spectrum(S, fe, fr)
        out['band'][v] = (lo, hi); out['kurt'][v] = kur
    return out


def stats(x):
    x = np.asarray(x, dtype=float); x = x - x.mean(); sd = x.std() + 1e-12
    return {'kurtosis': float(np.mean(x ** 4) / sd ** 4), 'crest': float(np.abs(x).max() / sd), 'rms': float(sd)}
