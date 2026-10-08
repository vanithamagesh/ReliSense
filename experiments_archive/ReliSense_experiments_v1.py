#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReliSense: additional experiments for manuscript v3.4 (script version 1)
=======================================================================

Runs the experiments that the v3.4 revision lists as future work, on the Paderborn
bearing dataset, with bearing-wise partitions throughout. Nothing here is tuned on test
bearings: every threshold comes from the training bearings of each fold.

  ex0  Baseline reproduction of ReliSense (L8, L10, A2R, LOBO), class-wise recall,
       false-alarm and missed-fault rates, and LOBO accuracy split into the 15
       development bearings and the 14 untouched bearings.
  ex1  Same 15 features with LR, RBF-SVM and RF; LR-fixed representation ladder
       (time statistics, band energies, order spectrum up to 16 orders, 15 features).
  ex2  Networks with matched information: WDCNN on raw vibration in a 2 x 2 design
       (bandwidth 4 or 16 kHz, 0.512-s windows or the full 4-s recording), WDCNN on the
       envelope (window and full), LR and CNN on the order spectrum up to 16 orders,
       and a PICNN-style feature-weighting network re-implemented after Lu et al. (2023).
  ex3  E1 repeated with the 15 deployed features: 10 scaled sets and many random sets
       that are kept away from true fault harmonics; fault-band versus equal-width
       non-fault-band masking of the order spectrum.
  ex4  Physical-evidence verifier and the four-arm reliability comparison:
       no rejection, confidence only, verifier only, confidence + verifier, with
       class-wise risk-coverage curves, acceptance per class and per bearing.
  ex5  Robustness: speed error, geometry (fault-order) error, and synthetic
       interference near the fault orders, with and without the verifier.
  ex6  Bearing-diversity learning curves (1-5 bearings per class) with the number of
       training recordings held constant, for ReliSense and the strongest baselines.

All method comparisons on the same test bearings are paired (per-bearing differences,
class-stratified bootstrap and Wilcoxon signed-rank test).

How to run on Google Colab
--------------------------
1. Runtime > Change runtime type > GPU (needed for ex2 only; the rest runs on CPU).
2. Put the extracted Paderborn folders (K001/, KA01/, ..., each holding the .mat files)
   under one folder, for example /content/drive/MyDrive/paderborn.
3. In a cell:
       from google.colab import drive; drive.mount('/content/drive')
       !python ReliSense_experiments_v1.py --data /content/drive/MyDrive/paderborn \
               --out /content/drive/MyDrive/relisense_results_v1 --exp ex0,ex1,ex3,ex4
   then, on the GPU runtime, --exp ex2; then --exp ex5,ex6.
   The first run builds a feature cache (about 20-40 min); later runs reuse it.
   Use --quick for a short smoke test (few folds, few epochs) before the full run.

Outputs go to --out: one CSV per table, risk-coverage figures (PNG), and summary.txt
with the numbers the manuscript needs.

Run time (rough): cache 20-40 min; ex0, ex1, ex6 a few minutes each; ex3 10-20 min
(200 random order sets); ex4 10-20 min; ex5 about 1 h; ex2 several hours on a T4 GPU
for all four protocols (start with --ex2_protocols L8_c0,A2R if time is short).

L10 here uses the 10 splits that hold out the same two list positions in each class
(K001-K005, KA04/15/16/22/30, KI04/14/16/18/21). If the original L10 splits differed,
L10 numbers will differ slightly; the other protocols are fully determined.

Settings that are not stated in the manuscript are marked "ASSUMPTION" below (for example
the band-pass filter order). If the original pipeline used different values, change the
constant and re-run ex0 to confirm that the published ReliSense numbers are reproduced
(93.6% L8 condition 0, 82.8% L8 all, 73.0% A2R, 80.1% LOBO) before trusting the rest.
"""
import argparse
import glob
import itertools
import json
import math
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.io import loadmat
from scipy.ndimage import maximum_filter1d, median_filter

from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================================================
# Constants from the manuscript
# =====================================================================================
FS = 64000                      # vibration sampling rate (Hz)
N64 = 256000                    # 4 s at 64 kHz
BAND = (2000.0, 12000.0)        # demodulation band (Hz)
BP_ORDER = 4                    # ASSUMPTION: 4th-order Butterworth, zero phase
ENV_FS = 4000                   # squared envelope averaged over 16 samples -> 4 kHz
ORDERS = {"BPFO": 3.054, "BPFI": 4.946, "BSF": 1.997, "FTF": 0.382, "shaft": 1.0}
FEAT_ORDERS = ["BPFO", "BPFI", "BSF", "FTF", "shaft"]
HARMS = (1, 2, 3)
FEAT_NAMES = [f"{o}_h{h}" for o in FEAT_ORDERS for h in HARMS]
REL_DELTA = 0.02                # search half-width: 2% of f_kh
MIN_BINS = 1.5                  # ... at least 1.5 bins
BG_HZ = 10.0                    # background ring up to 10 Hz
LR_C = 0.3

CONDITIONS = {"N15_M07_F10": 1500, "N09_M07_F10": 900, "N15_M01_F10": 1500, "N15_M07_F04": 1500}
COND0 = "N15_M07_F10"
CLASSES = ["H", "OR", "IR"]

HEALTHY = ["K001", "K002", "K003", "K004", "K005", "K006"]
ART_OR = ["KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09"]
ART_IR = ["KI01", "KI03", "KI05", "KI07", "KI08"]
REAL_OR = ["KA04", "KA15", "KA16", "KA22", "KA30"]
REAL_IR = ["KI04", "KI14", "KI16", "KI17", "KI18", "KI21"]
BOTH = ["KB23", "KB24", "KB27"]
SINGLE = HEALTHY + ART_OR + ART_IR + REAL_OR + REAL_IR          # 29 bearings
DEV = ["K001", "K002", "K003", "K004", "K005", "KA01", "KA04", "KA05", "KA07", "KA15",
       "KI01", "KI04", "KI05", "KI07", "KI14"]                  # preceding 15-bearing study
UNTOUCHED = [b for b in SINGLE if b not in DEV]                  # 14 bearings

L8_TRAIN = ["K002", "KA01", "KA05", "KA07", "KI01", "KI05", "KI07"]
L8_TEST = ["K001", "KA04", "KA15", "KA16", "KA22", "KA30", "KI14", "KI16", "KI17", "KI18", "KI21"]
A2R_TRAIN = ["K001", "K002", "K003"] + ART_OR + ART_IR
A2R_TEST = ["K004", "K005", "K006"] + REAL_OR + REAL_IR
L10_SETS = {"H": ["K001", "K002", "K003", "K004", "K005"],
            "OR": ["KA04", "KA15", "KA16", "KA22", "KA30"],
            "IR": ["KI04", "KI14", "KI16", "KI18", "KI21"]}
PUBLISHED_REPRO = {"L8_c0": 93.6, "L8_all": 82.8, "A2R": 73.0, "LOBO": 80.1, "L10_all": 74.6}


def bearing_class(b):
    if b.startswith("KA"):
        return "OR"
    if b.startswith("KI"):
        return "IR"
    if b.startswith("KB"):
        return "BOTH"
    return "H"


# =====================================================================================
# Utilities
# =====================================================================================
class Log:
    def __init__(self, out):
        self.path = os.path.join(out, "summary.txt")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(s + "\n")


def set_plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
        "font.weight": "bold", "font.size": 8, "axes.labelweight": "bold",
        "axes.titleweight": "bold", "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black"})
    return plt


def acc(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p))) * 100 if len(y) else float("nan")


def per_bearing_acc(y, p, bear):
    out = {}
    for b in np.unique(bear):
        m = bear == b
        out[b] = acc(y[m], p[m])
    return out


def class_rates(y, p):
    """Recall per class, false alarms (H called faulty) and missed faults (faulty called H)."""
    y = np.asarray(y); p = np.asarray(p)
    r = {}
    for k in range(3):
        m = y == k
        r[f"recall_{CLASSES[k]}"] = acc(y[m], p[m])
    h = y == 0
    f = y > 0
    r["false_alarm_H"] = float(np.mean(p[h] > 0)) * 100 if h.any() else float("nan")
    r["missed_fault"] = float(np.mean(p[f] == 0)) * 100 if f.any() else float("nan")
    r["accuracy"] = acc(y, p)
    return r


def paired_compare(a, b, B=2000, seed=0):
    """Paired per-bearing comparison of two {bearing: accuracy} dicts (a minus b).
    Bootstrap resamples bearings within each class; Wilcoxon signed-rank on differences."""
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    cls = np.array([bearing_class(k) for k in keys])
    rng = np.random.default_rng(seed)
    boots = []
    groups = [np.where(cls == c)[0] for c in np.unique(cls)]
    for _ in range(B):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        boots.append(d[idx].mean())
    try:
        p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = float("nan")
    return dict(n=len(d), mean_diff=float(d.mean()), ci_lo=float(np.percentile(boots, 2.5)),
                ci_hi=float(np.percentile(boots, 97.5)), wilcoxon_p=p,
                n_better=int((d > 0).sum()), n_equal=int((d == 0).sum()), n_worse=int((d < 0).sum()))


# =====================================================================================
# Loading the Paderborn .mat files
# =====================================================================================
FNAME_RE = re.compile(r"(N\d\d_M\d\d_F\d\d)_(K\d{3}|K[AIB]\d{2})_(\d+)\.mat$")


def load_channels(path):
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    key = [k for k in m if not k.startswith("__")][0]
    s = m[key]
    ch = {}
    for grp in ("X", "Y"):
        entries = getattr(s, grp, None)
        if entries is None:
            continue
        for e in np.atleast_1d(entries):
            name = str(getattr(e, "Name", "")).strip()
            data = getattr(e, "Data", None)
            if name and data is not None:
                ch[name] = np.asarray(data, dtype=np.float64).ravel()
    return ch


def list_files(root):
    files = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*.mat"), recursive=True)):
        m = FNAME_RE.search(os.path.basename(p))
        if m and m.group(1) in CONDITIONS:
            files.append((p, m.group(1), m.group(2), int(m.group(3))))
    return files


_SOS64 = signal.butter(BP_ORDER, BAND, btype="bandpass", fs=FS, output="sos")
_SOS32 = signal.butter(BP_ORDER, BAND, btype="bandpass", fs=FS // 2, output="sos")


def squared_envelope_4k(x, fs):
    """2-12 kHz band-pass, analytic signal, squared envelope, block-averaged to 4 kHz."""
    sos = _SOS64 if fs == FS else _SOS32
    xb = signal.sosfiltfilt(sos, x)
    xa = signal.hilbert(xb)
    e = np.abs(xa) ** 2
    blk = fs // ENV_FS
    n = len(e) // blk * blk
    return e[:n].reshape(-1, blk).mean(1), np.abs(xa)


def time_stats(x, env):
    """Eight vibration statistics and three envelope statistics.
    ASSUMPTION: the manuscript names the counts but not the statistics."""
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    pk = np.max(np.abs(x))
    erms = np.sqrt(np.mean(env ** 2)) + 1e-12
    return np.array([np.log(rms), np.log(pk), np.log(np.ptp(x) + 1e-12), pk / rms,
                     stats.kurtosis(x), stats.skew(x), rms / (np.mean(np.abs(x)) + 1e-12),
                     pk / (np.mean(np.abs(x)) + 1e-12),
                     np.log(erms), stats.kurtosis(env), np.max(env) / erms])


def band_energies(x8k, nb=32):
    f, P = signal.welch(x8k, fs=8000, nperseg=1024)
    edges = np.linspace(0, 4000, nb + 1)
    e = np.array([P[(f >= edges[i]) & (f < edges[i + 1])].sum() for i in range(nb)]) + 1e-20
    return np.log(e / e.sum())


def build_cache(data_root, cache_dir, log, need_32k=True):
    os.makedirs(cache_dir, exist_ok=True)
    meta_p = os.path.join(cache_dir, "meta.csv")
    if os.path.exists(meta_p):
        log("cache found:", cache_dir)
        return
    files = list_files(data_root)
    if not files:
        raise SystemExit(f"No Paderborn .mat files found under {data_root}")
    log(f"building cache from {len(files)} files ...")
    N = len(files)
    env4k = np.lib.format.open_memmap(os.path.join(cache_dir, "env4k.npy"), "w+", np.float32, (N, 16000))
    env8k = np.lib.format.open_memmap(os.path.join(cache_dir, "env8k.npy"), "w+", np.float32, (N, 32000))
    vib8k = np.lib.format.open_memmap(os.path.join(cache_dir, "vib8k.npy"), "w+", np.float32, (N, 32000))
    vib32k = (np.lib.format.open_memmap(os.path.join(cache_dir, "vib32k.npy"), "w+", np.float16, (N, 128000))
              if need_32k else None)
    tstat = np.zeros((N, 11), np.float32)
    bande = np.zeros((N, 32), np.float32)
    rows, bad = [], []
    t0 = time.time()
    for i, (p, cond, b, rec) in enumerate(files):
        ok = True
        try:
            ch = load_channels(p)
            x = ch["vibration_1"]
        except Exception as ex:  # unreadable file (the manuscript reports one)
            bad.append((os.path.basename(p), str(ex)[:80]))
            ok = False
        if ok and len(x) < N64:
            x = np.pad(x, (0, N64 - len(x)))
        if ok:
            x = x[:N64]
            rpm_set = CONDITIONS[cond]
            sp = ch.get("speed")
            rpm = float(np.median(sp)) if sp is not None and len(sp) else float("nan")
            speed_ok = np.isfinite(rpm) and abs(rpm - rpm_set) <= 0.2 * rpm_set
            fr = (rpm if speed_ok else rpm_set) / 60.0
            e4, envmag = squared_envelope_4k(x, FS)
            env4k[i] = e4[:16000]
            n8 = len(envmag) // 8 * 8
            env8k[i] = envmag[:n8].reshape(-1, 8).mean(1)[:32000]
            x8 = signal.resample_poly(x, 1, 8)
            vib8k[i] = x8[:32000]
            if vib32k is not None:
                vib32k[i] = signal.resample_poly(x, 1, 2)[:128000].astype(np.float16)
            tstat[i] = time_stats(x, envmag)
            bande[i] = band_energies(x8)
        else:
            rpm, speed_ok, fr = float("nan"), False, CONDITIONS[cond] / 60.0
        rows.append(dict(file=os.path.basename(p), cond=cond, bearing=b, rec=rec, ok=ok,
                         rpm=rpm, speed_ok=speed_ok, fr=fr, cls=bearing_class(b)))
        if (i + 1) % 200 == 0:
            log(f"  {i + 1}/{N} files, {time.time() - t0:.0f} s")
    for a in (env4k, env8k, vib8k, vib32k):
        if a is not None:
            a.flush()
    np.save(os.path.join(cache_dir, "tstat.npy"), tstat)
    np.save(os.path.join(cache_dir, "bande.npy"), bande)
    pd.DataFrame(rows).to_csv(meta_p, index=False)
    log(f"cache built: {N} files, unreadable: {bad}")
    log(f"speed signal outside 20% of set point (set point used instead): "
        f"{int((~pd.DataFrame(rows)['speed_ok']).sum())} recordings")


class Data:
    """Read-only view of the cache, restricted to readable recordings."""

    def __init__(self, cache_dir):
        meta = pd.read_csv(os.path.join(cache_dir, "meta.csv"))
        self.keep = np.where(meta["ok"].values)[0]
        self.meta = meta.iloc[self.keep].reset_index(drop=True)
        self.dir = cache_dir
        self.bear = self.meta["bearing"].values
        self.cond = self.meta["cond"].values
        self.fr = self.meta["fr"].values.astype(float)
        self.cls = self.meta["cls"].values
        self.y = np.array([CLASSES.index(c) if c in CLASSES else -1 for c in self.cls])
        self._mm = {}

    def arr(self, name):
        if name not in self._mm:
            self._mm[name] = np.load(os.path.join(self.dir, name + ".npy"), mmap_mode="r")
        return self._mm[name]

    def rows(self, name):
        """All readable rows of a cached array (loads it into memory)."""
        return np.asarray(self.arr(name)[self.keep])

    def get(self, name, idx):
        """Rows of a cached array for positions idx in the filtered index."""
        return np.asarray(self.arr(name)[self.keep[idx]], dtype=np.float32)

    def idx(self, bearings, cond=None):
        m = np.isin(self.bear, bearings)
        if cond is not None:
            m &= self.cond == cond
        return np.where(m)[0]


# =====================================================================================
# Module I: squared envelope spectrum and the 15 peak-to-background features
# =====================================================================================
def ses_batch(E):
    """Hann-windowed magnitude spectrum of mean-removed squared envelopes (rows of E)."""
    E = np.asarray(E, dtype=np.float64)
    E = E - E.mean(1, keepdims=True)
    w = np.hanning(E.shape[1])
    S = np.abs(np.fft.rfft(E * w, axis=1))
    df = ENV_FS / E.shape[1]
    return S, df


def pb_features(S, df, fr, orders, harmonics=HARMS, rel=REL_DELTA, min_bins=MIN_BINS, bg_hz=BG_HZ):
    """phi_kh = log( max_{|f-f_kh|<=Delta} S / median_{Delta<|f-f_kh|<=10 Hz} S ), eq. (6).
    orders: list of order values (one per row allowed: array (N,) per order)."""
    N, F = S.shape
    out = []
    maxoff = int(np.ceil((bg_hz + 15.0) / df)) + 2
    offs = np.arange(-maxoff, maxoff + 1)
    for o in orders:
        o = np.broadcast_to(np.asarray(o, dtype=float), (N,))
        for h in harmonics:
            fk = h * o * fr
            delta = np.maximum(rel * fk, min_bins * df)
            bg = np.maximum(bg_hz, delta + 2 * df)
            idx = np.round(fk / df).astype(int)[:, None] + offs[None, :]
            valid = (idx >= 0) & (idx < F)
            vals = np.take_along_axis(S, np.clip(idx, 0, F - 1), 1)
            dist = np.abs(idx * df - fk[:, None])
            pk = np.where((dist <= delta[:, None]) & valid, vals, -np.inf).max(1)
            bgv = np.where((dist > delta[:, None]) & (dist <= bg[:, None]) & valid, vals, np.nan)
            med = np.nanmedian(bgv, 1)
            out.append(np.log(np.maximum(pk, 1e-30) / np.maximum(med, 1e-30)))
    return np.stack(out, 1)


def kin_features(S, df, fr, order_scale=1.0, fr_scale=1.0, fault_orders=None):
    """The 15 ReliSense features. order_scale perturbs the four fault orders (not the shaft),
    fr_scale perturbs the measured shaft speed, fault_orders replaces the four fault orders."""
    fo = [ORDERS[k] * order_scale for k in ["BPFO", "BPFI", "BSF", "FTF"]] if fault_orders is None else list(fault_orders)
    return pb_features(S, df, fr * fr_scale, fo + [ORDERS["shaft"]])


def order_spectrum(S, df, fr, rmax=16.0, rmin=0.2, step=0.02):
    """O(r) = log[S(r f_r) / median S], eq. (5), on r = rmin, rmin+step, ... < rmax."""
    r = np.arange(rmin, rmax - 1e-9, step)
    f = np.arange(S.shape[1]) * df
    med = np.median(S, 1)
    O = np.empty((S.shape[0], len(r)), np.float32)
    for i in range(S.shape[0]):
        O[i] = np.log(np.maximum(np.interp(r * fr[i], f, S[i]), 1e-30) / med[i])
    return O, r


# =====================================================================================
# Classifiers and protocols
# =====================================================================================
def make_model(name, seed=0):
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(C=LR_C, max_iter=5000))
    if name == "SVM":   # ASSUMPTION: fixed in advance, not tuned
        return make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale", probability=True, random_state=seed))
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    raise ValueError(name)


def fit_predict(model_name, X, y, tr, te, seed=0):
    m = make_model(model_name, seed)
    m.fit(X[tr], y[tr])
    P = np.zeros((len(te), 3))
    P[:, m.classes_] = m.predict_proba(X[te]) if hasattr(m, "predict_proba") else np.eye(3)[m.predict(X[te])]
    return P


def protocols(D, quick=False):
    """List of (name, [(train_idx, test_idx), ...])."""
    out = []
    out.append(("L8_c0", [(D.idx(L8_TRAIN, COND0), D.idx(L8_TEST, COND0))]))
    out.append(("L8_all", [(D.idx(L8_TRAIN), D.idx(L8_TEST))]))
    out.append(("A2R", [(D.idx(A2R_TRAIN), D.idx(A2R_TEST))]))
    l10 = []
    for comb in itertools.combinations(range(5), 2):
        te_b = [L10_SETS[c][j] for c in CLASSES for j in comb]
        tr_b = [b for c in CLASSES for j, b in enumerate(L10_SETS[c]) if j not in comb]
        l10.append((D.idx(tr_b), D.idx(te_b)))
    out.append(("L10_all", l10[:2] if quick else l10))
    out.append(("LOBO", lobo_folds(D, quick)))
    return out


def lobo_bearings(quick=False):
    return ["K001", "K004", "KA01", "KA15", "KI01", "KI17"] if quick else SINGLE


def lobo_folds(D, quick=False):
    return [(D.idx([b2 for b2 in SINGLE if b2 != b]), D.idx([b])) for b in lobo_bearings(quick)]


def run_folds(model_name, X, y, folds, seed=0):
    preds, probs, idxs = [], [], []
    for tr, te in folds:
        P = fit_predict(model_name, X, y, tr, te, seed)
        probs.append(P); preds.append(P.argmax(1)); idxs.append(te)
    return np.concatenate(idxs), np.concatenate(preds), np.concatenate(probs)


def mean_over_splits(model_name, X, y, folds, seed=0):
    """Pooled accuracy (one fold) or mean +- SD over splits (L10)."""
    accs = []
    for tr, te in folds:
        P = fit_predict(model_name, X, y, tr, te, seed)
        accs.append(acc(y[te], P.argmax(1)))
    return float(np.mean(accs)), float(np.std(accs)) if len(accs) > 1 else 0.0


# =====================================================================================
# Physical-evidence verifier
# =====================================================================================
VER_H = 4               # harmonics used by the verifier
VER_DELTA = 0.02        # bounded joint frequency correction (+-2%)
VER_STEP = 0.0005
VER_NB = 1              # narrow peak window: +-1 bin around h * alpha * f0
VER_BG_HZ = 10.0
N_CTRL = 30             # matched incorrect base orders


def control_orders(n, seed=0, H=VER_H, guard=0.025, shaft_guard=0.08, spacing=0.01):
    """Matched incorrect base orders in [1.2, 6.0]. Their harmonics 1..H stay more than 3%
    away from harmonics 1..H+1 of BPFO, BPFI and BSF and more than 0.08 order away from
    every shaft harmonic, so a control cannot borrow a true fault line or a shaft line
    within the +-2% joint frequency correction. Controls differ from each other by >1%."""
    rng = np.random.default_rng(seed)
    fault_lines = np.array([h * ORDERS[k] for k in ["BPFO", "BPFI", "BSF"] for h in range(1, H + 2)])
    out = []
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > 200000:
            raise RuntimeError("could not draw enough control orders; relax the guards")
        c = rng.uniform(1.2, 6.0)
        lines = c * np.arange(1, H + 1)
        far_fault = np.all(np.abs(lines[:, None] / fault_lines[None, :] - 1) > guard)
        far_shaft = np.all(np.abs(lines - np.round(lines)) > shaft_guard)
        distinct = all(abs(c - o) / o > spacing for o in out)
        if far_fault and far_shaft and distinct:
            out.append(c)
    return np.array(out)


def _ratio_profile(S, df):
    """log(local max over +-VER_NB bins / running median over +-VER_BG_HZ) for every bin."""
    mx = maximum_filter1d(S, size=2 * VER_NB + 1, axis=-1)
    w = int(round(VER_BG_HZ / df)) * 2 + 1
    md = median_filter(S, size=(1, w), mode="nearest")
    return np.log(np.maximum(mx, 1e-30) / np.maximum(md, 1e-30))


def joint_harmonic_curve(R, df, f0, alphas, H=VER_H):
    """E(alpha) = mean_h R(h * alpha * f0) for one recording (R: ratio profile)."""
    F = R.shape[-1]
    fh = np.outer(alphas, np.arange(1, H + 1)) * f0
    idx = np.round(fh / df).astype(int)
    ok = idx < F - int(VER_BG_HZ / df)
    vals = np.where(ok, R[np.clip(idx, 0, F - 1)], np.nan)
    return np.nanmean(vals, 1)


def persistence_evidence(env, fr, base_orders, H=VER_H):
    """Split-half persistence of harmonic evidence for each base order.
    alpha* is fitted jointly over H harmonics on one half and the evidence is read on the
    other half at that alpha*; the two directions are averaged."""
    n = len(env) // 2
    halves = np.stack([env[:n], env[n:2 * n]])
    S, df = ses_batch(halves)
    R = _ratio_profile(S, df)
    alphas = np.arange(1 - VER_DELTA, 1 + VER_DELTA + 1e-12, VER_STEP)
    ev = np.empty(len(base_orders))
    for j, o in enumerate(base_orders):
        f0 = o * fr
        cA = joint_harmonic_curve(R[0], df, f0, alphas, H)
        cB = joint_harmonic_curve(R[1], df, f0, alphas, H)
        aA, aB = np.nanargmax(cA), np.nanargmax(cB)
        ev[j] = 0.5 * (cB[aA] + cA[aB])
    return ev


def evidence_table(E4, fr, ctrl, sideband=True):
    """Robust z-scores of BPFO and BPFI persistence evidence against matched controls.
    Optional IR sideband term: evidence at h*BPFI +- f_r (load-zone modulation)."""
    bases = np.concatenate([[ORDERS["BPFO"], ORDERS["BPFI"]], ctrl])
    out = np.zeros((len(E4), 4))
    for i in range(len(E4)):
        ev = persistence_evidence(np.asarray(E4[i], dtype=np.float64), fr[i], bases)
        c = ev[2:]
        med = np.median(c)
        mad = 1.4826 * np.median(np.abs(c - med)) + 1e-9
        out[i, 0] = (ev[0] - med) / mad
        out[i, 1] = (ev[1] - med) / mad
        out[i, 2] = ev[0]
        out[i, 3] = ev[1]
    return pd.DataFrame(out, columns=["z_OR", "z_IR", "ev_OR", "ev_IR"])


class HealthyModel:
    """Compatibility with healthy training bearings: shrinkage Mahalanobis distance on the
    standardized 15 features, plus a signal-quality range check (log RMS, kurtosis)."""

    def __init__(self, Xk, Q, y):
        h = y == 0
        self.mu = Xk[h].mean(0)
        self.sd = Xk[h].std(0) + 1e-9
        Z = (Xk[h] - self.mu) / self.sd
        self.lw = LedoitWolf().fit(Z)
        self.q_lo = np.percentile(Q, 0.5, axis=0)
        self.q_hi = np.percentile(Q, 99.5, axis=0)

    def dist(self, Xk):
        return np.sqrt(np.maximum(self.lw.mahalanobis((Xk - self.mu) / self.sd), 0))

    def quality_ok(self, Q):
        return np.all((Q >= self.q_lo) & (Q <= self.q_hi), axis=1)


def _ecdf_rank(ref, x):
    ref = np.sort(np.asarray(ref))
    if len(ref) == 0:
        return np.full(len(x), 0.5)
    return np.searchsorted(ref, x, side="right") / len(ref)


def arm_scores(pred, conf, zOR, zIR, dist, qok, ref):
    """Scores of each arm, transformed to percentiles of the inner (held-out training
    bearing) recordings with the same predicted class. ref: dict with the inner arrays."""
    sc = {"conf": np.zeros(len(pred)), "ver": np.zeros(len(pred)), "both": np.zeros(len(pred))}
    for k in range(3):
        m = pred == k
        if not m.any():
            continue
        rm = ref["pred"] == k
        uc = _ecdf_rank(ref["conf"][rm], conf[m])
        if k == 0:
            u1 = _ecdf_rank(-np.maximum(ref["zOR"][rm], ref["zIR"][rm]), -np.maximum(zOR[m], zIR[m]))
            u2 = _ecdf_rank(-ref["dist"][rm], -dist[m])
            uv = np.minimum(u1, u2)
        else:
            z = zOR if k == 1 else zIR
            rz = ref["zOR"] if k == 1 else ref["zIR"]
            uv = _ecdf_rank(rz[rm], z[m])
        uv = np.where(qok[m], uv, -1.0)
        sc["conf"][m] = uc
        sc["ver"][m] = uv
        sc["both"][m] = np.minimum(uc, uv)
    return sc


def per_class_threshold(ref_scores, ref_pred, kappa):
    return {k: (np.quantile(ref_scores[ref_pred == k], 1 - kappa) if np.any(ref_pred == k) else 0.0)
            for k in range(3)}


# =====================================================================================
# ex0: baseline reproduction
# =====================================================================================
def get_kin(D, cache):
    p = os.path.join(cache, "kin15.npy")
    if os.path.exists(p):
        return np.load(p)
    E4 = D.rows("env4k")
    X = np.zeros((len(D.bear), 15))
    for s in range(0, len(X), 256):
        S, df = ses_batch(E4[s:s + 256])
        X[s:s + 256] = kin_features(S, df, D.fr[s:s + 256])
    np.save(p, X)
    return X


def get_orderspec(D, cache, rmax=16.0):
    p = os.path.join(cache, f"order{int(rmax)}.npy")
    if os.path.exists(p):
        return np.load(p)
    E4 = D.rows("env4k")
    Os = []
    for s in range(0, len(E4), 256):
        S, df = ses_batch(E4[s:s + 256])
        O, _ = order_spectrum(S, df, D.fr[s:s + 256], rmax=rmax)
        Os.append(O)
    O = np.concatenate(Os)
    np.save(p, O)
    return O


def ex0(D, X, args, log, out):
    log("\n=== ex0: baseline reproduction of ReliSense (15 features + LR) ===")
    m = D.y >= 0
    rows, rates = [], []
    for name, folds in protocols(D, args.quick):
        folds = [(tr[m[tr]], te[m[te]]) for tr, te in folds]
        if name == "L10_all":
            mu, sd = mean_over_splits("LR", X, D.y, folds)
            mu0, sd0 = mean_over_splits("LR", X, D.y, [(tr[D.cond[tr] == COND0], te[D.cond[te] == COND0]) for tr, te in folds])
            rows.append(dict(protocol=name, accuracy=mu, sd=sd, published=PUBLISHED_REPRO.get(name)))
            rows.append(dict(protocol="L10_c0", accuracy=mu0, sd=sd0, published=73.5))
            continue
        te, pr, P = run_folds("LR", X, D.y, folds)
        rows.append(dict(protocol=name, accuracy=acc(D.y[te], pr), sd=0.0, published=PUBLISHED_REPRO.get(name)))
        r = class_rates(D.y[te], pr); r["protocol"] = name; rates.append(r)
        if name == "LOBO":
            pb = per_bearing_acc(D.y[te], pr, D.bear[te])
            pd.Series(pb).to_csv(os.path.join(out, "ex0_lobo_per_bearing.csv"), header=["accuracy"])
            dev = np.isin(D.bear[te], DEV)
            log(f"  LOBO development bearings (15): {acc(D.y[te][dev], pr[dev]):.1f}%   "
                f"untouched bearings (14): {acc(D.y[te][~dev], pr[~dev]):.1f}%")
            for grp, bl in [("development", DEV), ("untouched", UNTOUCHED)]:
                mm = np.isin(D.bear[te], bl)
                rr = class_rates(D.y[te][mm], pr[mm]); rr["protocol"] = f"LOBO_{grp}"; rates.append(rr)
        if name in ("L8_c0", "L8_all", "A2R"):
            dev = np.isin(D.bear[te], DEV)
            log(f"  {name}: development test bearings {acc(D.y[te][dev], pr[dev]):.1f}% "
                f"({sorted(set(D.bear[te][dev]))}); untouched {acc(D.y[te][~dev], pr[~dev]):.1f}%")
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "ex0_reproduction.csv"), index=False)
    log(df.round(1).to_string(index=False))
    rt = pd.DataFrame(rates); rt.to_csv(os.path.join(out, "ex0_class_rates.csv"), index=False)
    log(rt.round(1).to_string(index=False))
    bad = [r for r in rows if r["published"] and abs(r["accuracy"] - r["published"]) > 1.0]
    if bad and not args.quick:
        log("  WARNING: reproduction differs from the manuscript by more than 1 point for "
            + ", ".join(r["protocol"] for r in bad) + ". Check BP_ORDER and the feature settings first.")


# =====================================================================================
# ex1: same features, three classifiers; LR-fixed representation ladder
# =====================================================================================
def ex1(D, X, args, log, out):
    log("\n=== ex1: identical 15 features with LR / SVM / RF, and LR-fixed ladder ===")
    m = D.y >= 0
    T = D.rows("tstat").astype(float)
    Sb = D.rows("bande").astype(float)
    O16 = get_orderspec(D, args.cache, 16.0)
    reps = {"T (time stats)": T, "S (band energies)": Sb, "T+S": np.hstack([T, Sb]),
            "order spectrum <=16": O16, "15 kinematic": X,
            "14 kinematic (no BPFI h3)": np.delete(X, FEAT_NAMES.index("BPFI_h3"), axis=1)}
    rows, perb = [], {}
    for pname, folds in protocols(D, args.quick):
        if pname == "L10_all":
            continue
        folds = [(tr[m[tr]], te[m[te]]) for tr, te in folds]
        for model in ["LR", "SVM", "RF"]:
            te, pr, _ = run_folds(model, X, D.y, folds)
            rows.append(dict(protocol=pname, input="15 kinematic", model=model, accuracy=acc(D.y[te], pr),
                             **{k: v for k, v in class_rates(D.y[te], pr).items() if k != "accuracy"}))
            if pname == "LOBO":
                perb[f"15 kinematic/{model}"] = per_bearing_acc(D.y[te], pr, D.bear[te])
        for rname, R in reps.items():
            if rname == "15 kinematic":
                continue
            te, pr, _ = run_folds("LR", R, D.y, folds)
            rows.append(dict(protocol=pname, input=rname, model="LR", accuracy=acc(D.y[te], pr),
                             **{k: v for k, v in class_rates(D.y[te], pr).items() if k != "accuracy"}))
            if pname == "LOBO":
                perb[f"{rname}/LR"] = per_bearing_acc(D.y[te], pr, D.bear[te])
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "ex1_classifiers_and_ladder.csv"), index=False)
    log(df.pivot_table(index=["input", "model"], columns="protocol", values="accuracy").round(1).to_string())
    pd.DataFrame(perb).to_csv(os.path.join(out, "ex1_lobo_per_bearing.csv"))
    base = perb["15 kinematic/LR"]
    comp = [dict(comparison=f"15 kinematic/LR minus {k}", **paired_compare(base, v)) for k, v in perb.items()
            if k != "15 kinematic/LR"]
    cp = pd.DataFrame(comp); cp.to_csv(os.path.join(out, "ex1_paired_LOBO.csv"), index=False)
    log(cp.round(3).to_string(index=False))


# =====================================================================================
# ex2: networks with matched information
# =====================================================================================
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_wdcnn(nn):
    """WDCNN of Table S1 (85 603 parameters); adaptive pooling allows any input length."""
    def blk(ci, co, k, s=1, p=1):
        return [nn.Conv1d(ci, co, k, s, p), nn.BatchNorm1d(co), nn.ReLU(), nn.MaxPool1d(2)]
    return nn.Sequential(*blk(1, 16, 64, 16, 24), *blk(16, 32, 3), *blk(32, 64, 3), *blk(64, 64, 3),
                         *blk(64, 64, 3), nn.AdaptiveAvgPool1d(8), nn.Flatten(),
                         nn.Linear(512, 100), nn.ReLU(), nn.Dropout(0.3), nn.Linear(100, 3))


def build_ordercnn(nn, n_in, picnn_centers=None):
    """1-D CNN on the order spectrum (three conv layers 16, 32, 32; dropout 0.3).
    ASSUMPTION: kernel sizes 7/5/5 and pooling to 8; the manuscript gives the widths only.
    With picnn_centers, a learnable Gaussian feature-weighting layer is placed in front
    (PICNN-style, re-implemented after Lu et al., 2023; not the authors' code)."""
    torch, _ = _torch()

    class Weighting(nn.Module):
        def __init__(self, centers, r):
            super().__init__()
            self.register_buffer("c", torch.tensor(centers, dtype=torch.float32)[:, None])
            self.register_buffer("r", torch.tensor(r, dtype=torch.float32)[None, :])
            self.log_sig = nn.Parameter(torch.full((len(centers), 1), math.log(0.05)))
            self.amp = nn.Parameter(torch.zeros(len(centers), 1))

        def forward(self, x):
            g = torch.exp(-0.5 * ((self.r - self.c) / torch.exp(self.log_sig)) ** 2)
            w = 1 + (torch.nn.functional.softplus(self.amp) * g).sum(0)
            return x * w

    layers = []
    if picnn_centers is not None:
        layers.append(Weighting(picnn_centers[0], picnn_centers[1]))
    layers += [nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
               nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
               nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(8), nn.Flatten(),
               nn.Dropout(0.3), nn.Linear(256, 3)]
    return nn.Sequential(*layers)


def windows(x, wlen):
    n = x.shape[1] // wlen
    return x[:, :n * wlen].reshape(x.shape[0] * n, wlen), n


def train_net(net, Xtr, ytr, epochs, lr, wd, bs, device, shift=0, seed=0):
    torch, nn = _torch()
    torch.manual_seed(seed); np.random.seed(seed)
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    cw = np.bincount(ytr, minlength=3).astype(float)
    cw = torch.tensor(cw.sum() / (3 * np.maximum(cw, 1)), dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=cw)
    n = len(ytr)
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(n)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            xb = Xtr[b]
            if shift:
                xb = np.stack([np.roll(r, np.random.randint(-shift, shift + 1)) for r in xb])
            xb = torch.tensor(xb, dtype=torch.float32, device=device)[:, None, :]
            yb = torch.tensor(ytr[b], dtype=torch.long, device=device)
            opt.zero_grad()
            lossf(net(xb), yb).backward()
            opt.step()
    return net


def predict_net(net, X, device, bs=256):
    torch, _ = _torch()
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            xb = torch.tensor(X[s:s + bs], dtype=torch.float32, device=device)[:, None, :]
            out.append(torch.softmax(net(xb), 1).cpu().numpy())
    return np.concatenate(out)


def zscore_rows(A):
    A = A.astype(np.float32)
    return (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-8)


def ex2(D, X, args, log, out):
    torch, nn = _torch()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"\n=== ex2: networks with matched information (device {device}) ===")
    m = D.y >= 0
    ep_w = 2 if args.quick else 20
    ep_o = 2 if args.quick else 60
    # input variants: (cache array, samples per window or None for full 4 s, batch size)
    variants = {
        "WDCNN raw 8 kHz, 0.512-s windows (original)": ("vib8k", 4096, 128),
        "WDCNN raw 32 kHz, 0.512-s windows (bandwidth matched)": ("vib32k", 16384, 64),
        "WDCNN raw 8 kHz, full 4 s (duration matched)": ("vib8k", None, 32),
        "WDCNN raw 32 kHz, full 4 s (bandwidth and duration matched)": ("vib32k", None, 16),
        "WDCNN envelope 8 kHz, 0.512-s windows (original)": ("env8k", 4096, 128),
        "WDCNN envelope 8 kHz, full 4 s (duration matched)": ("env8k", None, 32),
    }
    wd = build_wdcnn(nn)
    log(f"  WDCNN parameters: {sum(p.numel() for p in wd.parameters())}")
    O16 = get_orderspec(D, args.cache, 16.0)
    r16 = np.arange(0.2, 16.0 - 1e-9, 0.02)
    O12 = O16[:, r16 < 12.0]
    centers = sorted({h * ORDERS[k] for k in ["BPFO", "BPFI", "BSF", "FTF", "shaft"] for h in range(1, 6)
                      if 0.2 < h * ORDERS[k] < 16})
    log(f"  order CNN parameters: {sum(p.numel() for p in build_ordercnn(nn, O16.shape[1]).parameters())}")
    proto = [(n, f) for n, f in protocols(D, args.quick) if n in args.ex2_protocols.split(",")]
    rows, perb = [], {}
    seeds = [0] if args.quick else [0, 1, 2]
    for pname, folds in proto:
        folds = [(tr[m[tr]], te[m[te]]) for tr, te in folds]
        for vname, (arr, wlen, bs) in variants.items():
            accs, allp, allt = [], [], []
            for si, (tr, te) in enumerate(folds):
                seed_list = [si % 3] if pname == "LOBO" else seeds   # LOBO: one seed per fold
                for seed in seed_list:
                    Xtr = D.get(arr, tr); Xte = D.get(arr, te)
                    if wlen:
                        Wtr, ntr = windows(Xtr, wlen); ytr = np.repeat(D.y[tr], ntr)
                        Wte, nte = windows(Xte, wlen)
                    else:
                        Wtr, ytr, Wte, nte = Xtr, D.y[tr], Xte, 1
                    net = train_net(build_wdcnn(nn), zscore_rows(Wtr), ytr, ep_w, 1e-3, 1e-4, bs, device, seed=seed)
                    P = predict_net(net, zscore_rows(Wte), device).reshape(len(te), nte, 3).mean(1)
                    accs.append(acc(D.y[te], P.argmax(1)))
                    allp.append(P.argmax(1)); allt.append(te)
            te_all = np.concatenate(allt); pr_all = np.concatenate(allp)
            rows.append(dict(protocol=pname, model=vname, accuracy=acc(D.y[te_all], pr_all),
                             sd=float(np.std(accs)), **{k: v for k, v in class_rates(D.y[te_all], pr_all).items()
                                                        if k != "accuracy"}))
            if pname == "LOBO":
                perb[vname] = per_bearing_acc(D.y[te_all], pr_all, D.bear[te_all])
            log(f"  {pname:7s} {vname}: {rows[-1]['accuracy']:.1f}%")
        for oname, Oin, pic in [("order spectrum <=12 + CNN", O12, None), ("order spectrum <=16 + CNN", O16, None),
                                ("PICNN-style weighting + CNN, <=16", O16, (centers, r16))]:
            allp, allt = [], []
            for si, (tr, te) in enumerate(folds):
                seed = si % 3 if pname == "LOBO" else 0
                mu, sd = Oin[tr].mean(0), Oin[tr].std(0) + 1e-6
                net = build_ordercnn(nn, Oin.shape[1], pic)
                net = train_net(net, (Oin[tr] - mu) / sd, D.y[tr], ep_o, 3e-3, 1e-3, 64, device, shift=2, seed=seed)
                P = predict_net(net, (Oin[te] - mu) / sd, device)
                allp.append(P.argmax(1)); allt.append(te)
            te_all = np.concatenate(allt); pr_all = np.concatenate(allp)
            rows.append(dict(protocol=pname, model=oname, accuracy=acc(D.y[te_all], pr_all), sd=0.0,
                             **{k: v for k, v in class_rates(D.y[te_all], pr_all).items() if k != "accuracy"}))
            if pname == "LOBO":
                perb[oname] = per_bearing_acc(D.y[te_all], pr_all, D.bear[te_all])
            log(f"  {pname:7s} {oname}: {rows[-1]['accuracy']:.1f}%")
        te_all, pr_all, _ = run_folds("LR", X, D.y, folds)
        rows.append(dict(protocol=pname, model="ReliSense (15 features + LR)", accuracy=acc(D.y[te_all], pr_all), sd=0.0))
        if pname == "LOBO":
            perb["ReliSense"] = per_bearing_acc(D.y[te_all], pr_all, D.bear[te_all])
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "ex2_networks.csv"), index=False)
    log(df.pivot_table(index="model", columns="protocol", values="accuracy").round(1).to_string())
    if perb:
        pd.DataFrame(perb).to_csv(os.path.join(out, "ex2_lobo_per_bearing.csv"))
        comp = [dict(comparison=f"ReliSense minus {k}", **paired_compare(perb["ReliSense"], v))
                for k, v in perb.items() if k != "ReliSense"]
        cp = pd.DataFrame(comp); cp.to_csv(os.path.join(out, "ex2_paired_LOBO.csv"), index=False)
        log(cp.round(3).to_string(index=False))


# =====================================================================================
# ex3: E1 with the 15 deployed features; fault-band masking
# =====================================================================================
def random_order_sets(n, seed, guard=0.02):
    """Sets of four random orders in [0.3, 5.9] whose harmonics 1-3 stay outside the +-2%
    search band of every harmonic 1-3 of the true BPFO, BPFI, BSF and FTF."""
    rng = np.random.default_rng(seed)
    true_lines = np.array([h * ORDERS[k] for k in ["BPFO", "BPFI", "BSF", "FTF"] for h in HARMS])
    sets = []
    while len(sets) < n:
        s = rng.uniform(0.3, 5.9, 4)
        lines = np.outer(s, HARMS).ravel()
        if np.all(np.abs(lines[:, None] / true_lines[None, :] - 1) > 2 * guard):
            sets.append(s)
    return sets


def ex3(D, X, args, log, out):
    log("\n=== ex3: E1 with the 15 deployed features; fault-band versus non-fault-band masking ===")
    m = D.y >= 0
    E4 = D.rows("env4k")
    S, df = ses_batch(E4)
    scales = [0.80, 0.85, 0.90, 0.95, 0.98, 1.02, 1.05, 1.10, 1.15, 1.20]
    n_rand = 10 if args.quick else args.n_random
    rsets = random_order_sets(n_rand, seed=1)
    prot = {n: [(tr[m[tr]], te[m[te]]) for tr, te in f] for n, f in protocols(D, args.quick) if n in ("L8_c0", "LOBO")}
    rows = []

    def score(Xs):
        res = {}
        for n, folds in prot.items():
            te, pr, _ = run_folds("LR", Xs, D.y, folds)
            res[n] = acc(D.y[te], pr)
        return res

    true = score(X)
    rows.append(dict(kind="true", label="6203 kinematics", **true))
    for s in scales:
        rows.append(dict(kind="scaled", label=f"scale {s}", **score(kin_features(S, df, D.fr, order_scale=s))))
    for j, rs in enumerate(rsets):
        rows.append(dict(kind="random", label="random " + ",".join(f"{v:.3f}" for v in rs),
                         **score(kin_features(S, df, D.fr, fault_orders=rs))))
        if (j + 1) % 25 == 0:
            log(f"  random sets: {j + 1}/{len(rsets)}")
    dfr = pd.DataFrame(rows); dfr.to_csv(os.path.join(out, "ex3_E1_15features.csv"), index=False)
    for n in prot:
        wrong = dfr[dfr.kind != "true"][n].values
        rnd = dfr[dfr.kind == "random"][n].values
        rank = 1 + int(np.sum(wrong >= true[n]))
        p_emp = (1 + np.sum(rnd >= true[n])) / (len(rnd) + 1)
        log(f"  {n}: true {true[n]:.1f}% | best scaled {dfr[dfr.kind == 'scaled'][n].max():.1f}% | "
            f"random mean {rnd.mean():.1f}%, max {rnd.max():.1f}% | rank of true orders {rank}/{len(wrong) + 1} | "
            f"empirical p vs random sets {p_emp:.4f}")
    # masking on the order spectrum (LR), fault bands versus equal-width random bands
    O16 = get_orderspec(D, args.cache, 16.0)
    r = np.arange(0.2, 16.0 - 1e-9, 0.02)
    fault_lines = [h * ORDERS[k] for k in ["BPFO", "BPFI", "BSF", "FTF"] for h in HARMS]
    widths = [max(0.02 * c, 0.04) for c in fault_lines]

    def mask(lines, ws):
        M = np.zeros(len(r), bool)
        for c, w in zip(lines, ws):
            M |= np.abs(r - c) <= w
        return M

    Mf = mask(fault_lines, widths)
    rng = np.random.default_rng(2)
    shaft_lines = np.arange(1, 17)
    mrows = []
    base = score(O16)
    mrows.append(dict(mask="none", n_bins=0, **base))
    Om = O16.copy(); Om[:, Mf] = 0.0                       # 0 = log(median): band set to background level
    mrows.append(dict(mask="fault bands", n_bins=int(Mf.sum()), **score(Om)))
    n_ctrl = 5 if args.quick else args.n_mask
    for j in range(n_ctrl):
        while True:
            cand = rng.uniform(0.3, 15.5, len(fault_lines))
            Mc = mask(cand, widths)
            near_true = mask(fault_lines + list(shaft_lines), [w * 2 for w in widths] + [0.06] * len(shaft_lines))
            if not np.any(Mc & near_true):
                break
        Om = O16.copy(); Om[:, Mc] = 0.0
        mrows.append(dict(mask=f"non-fault bands {j}", n_bins=int(Mc.sum()), **score(Om)))
    dm = pd.DataFrame(mrows); dm.to_csv(os.path.join(out, "ex3_masking.csv"), index=False)
    for n in prot:
        ctrl = dm[dm["mask"].str.startswith("non-fault")][n].values
        log(f"  masking, {n}: none {base[n]:.1f}% | fault bands {dm.iloc[1][n]:.1f}% | "
            f"equal-width non-fault bands mean {ctrl.mean():.1f}% (min {ctrl.min():.1f}%, max {ctrl.max():.1f}%)")


# =====================================================================================
# ex4: physical-evidence verifier, four-arm comparison
# =====================================================================================
def get_evidence(D, args, log, E4=None, fr=None, tag="clean"):
    p = os.path.join(args.cache, f"evidence_{tag}_{N_CTRL}.csv")
    if tag.startswith("clean") and os.path.exists(p):
        return pd.read_csv(p)
    E4 = D.rows("env4k") if E4 is None else E4
    fr = D.fr if fr is None else fr
    ctrl = control_orders(N_CTRL, seed=3)
    t0 = time.time()
    parts = []
    for s in range(0, len(E4), 200):
        parts.append(evidence_table(E4[s:s + 200], fr[s:s + 200], ctrl))
        if tag.startswith("clean"):
            log(f"  evidence {min(s + 200, len(E4))}/{len(E4)} ({time.time() - t0:.0f} s)")
    ev = pd.concat(parts, ignore_index=True)
    if tag.startswith("clean"):
        ev.to_csv(p, index=False)
    return ev


def selective_lobo(D, X, Q, ev, bearings, kappas, model="LR", Xtest=None, Qtest=None, evtest=None):
    """LOBO with inner calibration; returns a per-recording frame with predictions and the
    acceptance decision of every arm at every target coverage."""
    Xtest = X if Xtest is None else Xtest
    Qtest = Q if Qtest is None else Qtest
    evtest = ev if evtest is None else evtest
    zOR, zIR = ev["z_OR"].values, ev["z_IR"].values
    tzOR, tzIR = evtest["z_OR"].values, evtest["z_IR"].values
    recs = []
    for b in bearings:
        te = D.idx([b])
        trb = [v for v in SINGLE if v != b]
        # inner loop over training bearings
        ref = {k: [] for k in ["pred", "conf", "zOR", "zIR", "dist", "true"]}
        for v in trb:
            tr_in = D.idx([u for u in trb if u != v]); te_in = D.idx([v])
            P = fit_predict(model, X, D.y, tr_in, te_in)
            hm = HealthyModel(X[tr_in], Q[tr_in], D.y[tr_in])
            ref["pred"].append(P.argmax(1)); ref["conf"].append(P.max(1))
            ref["zOR"].append(zOR[te_in]); ref["zIR"].append(zIR[te_in])
            ref["dist"].append(hm.dist(X[te_in])); ref["true"].append(D.y[te_in])
        ref = {k: np.concatenate(v) for k, v in ref.items()}
        ref_q = np.ones(len(ref["pred"]), bool)
        ref_sc = arm_scores(ref["pred"], ref["conf"], ref["zOR"], ref["zIR"], ref["dist"], ref_q, ref)
        tr = D.idx(trb)
        P = fit_predict(model, X, D.y, tr, te) if Xtest is X else _fit_predict_shifted(model, X, Xtest, D.y, tr, te)
        pred, conf = P.argmax(1), P.max(1)
        hm = HealthyModel(X[tr], Q[tr], D.y[tr])
        dist = hm.dist(Xtest[te]); qok = hm.quality_ok(Qtest[te])
        sc = arm_scores(pred, conf, tzOR[te], tzIR[te], dist, qok, ref)
        rec = pd.DataFrame(dict(bearing=b, true=D.y[te], pred=pred, conf=conf, z_OR=tzOR[te], z_IR=tzIR[te],
                                dist=dist, quality_ok=qok, s_conf=sc["conf"], s_ver=sc["ver"], s_both=sc["both"]))
        for kap in kappas:
            tglob = np.quantile(ref["conf"], 1 - kap)                    # eq. (8), as in the manuscript
            rec[f"acc_confglobal_{kap}"] = conf >= tglob
            for arm in ["conf", "ver", "both"]:
                th = per_class_threshold(ref_sc[arm], ref["pred"], kap)
                rec[f"acc_{arm}_{kap}"] = sc[arm] >= np.array([th[k] for k in pred])
        recs.append(rec)
    return pd.concat(recs, ignore_index=True)


def _fit_predict_shifted(model, Xtrain_all, Xtest_all, y, tr, te):
    m = make_model(model)
    m.fit(Xtrain_all[tr], y[tr])
    P = np.zeros((len(te), 3))
    P[:, m.classes_] = m.predict_proba(Xtest_all[te])
    return P


ARMS = [("none", None), ("confidence, global threshold (eq. 8)", "confglobal"), ("confidence, per class", "conf"),
        ("verifier only", "ver"), ("confidence + verifier", "both")]


def summarize_arms(R, kappas):
    rows = []
    y, p = R["true"].values, R["pred"].values
    for kap in kappas:
        for aname, key in ARMS:
            a = np.ones(len(R), bool) if key is None else R[f"acc_{key}_{kap}"].values
            err = a & (y != p)
            row = dict(target_coverage=kap if key else 1.0, arm=aname, coverage=100 * a.mean(),
                       selective_accuracy=acc(y[a], p[a]), accepted_errors=int(err.sum()),
                       accepted_false_alarms=int((a & (y == 0) & (p > 0)).sum()),
                       accepted_missed_faults=int((a & (y > 0) & (p == 0)).sum()),
                       accepted_wrong_fault_type=int((a & (y > 0) & (p > 0) & (y != p)).sum()))
            for k in range(3):
                mk = y == k
                row[f"acceptance_{CLASSES[k]}"] = 100 * a[mk].mean()
                bears = R.loc[mk, "bearing"].values
                row[f"bearings_with_acceptance_{CLASSES[k]}"] = f"{len(set(bears[a[mk]]))}/{len(set(bears))}"
            rows.append(row)
            if key is None and kap != kappas[0]:
                rows.pop()
    return pd.DataFrame(rows)


def risk_coverage(R, score_col):
    s = R[score_col].values
    err = (R["true"].values != R["pred"].values).astype(float)
    order = np.argsort(-s, kind="stable")
    e = err[order]
    cov = np.arange(1, len(e) + 1) / len(e)
    risk = np.cumsum(e) / np.arange(1, len(e) + 1)
    return cov, risk, float(np.trapezoid(risk, cov) if hasattr(np, "trapezoid") else np.trapz(risk, cov))


def plot_rc(R, path, title):
    plt = set_plot_style()
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.3), sharey=True)
    cols = {"s_conf": ("Confidence", "#1f5fa8"), "s_ver": ("Verifier", "#2e8b57"), "s_both": ("Confidence + verifier", "#b22222")}
    for ax, (lab, sel) in zip(axs, [("All classes", None), ("Healthy", 0), ("Outer race", 1), ("Inner race", 2)]):
        RR = R if sel is None else R[R["true"] == sel]
        for c, (nm, colr) in cols.items():
            cov, risk, aurc = risk_coverage(RR, c)
            ax.plot(100 * cov, 100 * risk, color=colr, lw=1.2, label=nm if sel is None else None)
        ax.set_title(lab); ax.set_xlabel("Coverage (%)"); ax.set_xlim(100, 0); ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Risk among accepted (%)")
    fig.legend(loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title, fontsize=8, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=400)
    plt.close(fig)


def ex4(D, X, args, log, out):
    log("\n=== ex4: physical-evidence verifier, four-arm reliability comparison (LOBO) ===")
    ev = get_evidence(D, args, log)
    Q = D.rows("tstat")[:, [0, 4]].astype(float)          # log RMS and kurtosis of vibration
    kappas = [0.9, 0.8, 0.7, 0.6]
    bearings = lobo_bearings(args.quick)
    R = selective_lobo(D, X, Q, ev, bearings, kappas)
    R.to_csv(os.path.join(out, "ex4_lobo_records.csv"), index=False)
    sm = summarize_arms(R, kappas)
    sm.to_csv(os.path.join(out, "ex4_lobo_arms.csv"), index=False)
    log(sm.round(1).to_string(index=False))
    for c in ["s_conf", "s_ver", "s_both"]:
        log(f"  AURC {c}: {100 * risk_coverage(R, c)[2]:.2f}%  "
            + " ".join(f"{CLASSES[k]}: {100 * risk_coverage(R[R['true'] == k], c)[2]:.2f}%" for k in range(3)))
    plot_rc(R, os.path.join(out, "ex4_risk_coverage_LOBO.png"), "LOBO: risk-coverage by class")
    # per-bearing acceptance at kappa = 0.7, and paired comparison of accepted-error rates
    rows = []
    for b, g in R.groupby("bearing"):
        row = dict(bearing=b, accuracy=acc(g["true"], g["pred"]))
        for key in ["confglobal", "conf", "ver", "both"]:
            a = g[f"acc_{key}_0.7"].values
            row[f"accept_{key}"] = 100 * a.mean()
            row[f"accepted_error_rate_{key}"] = 100 * np.mean(a & (g["true"].values != g["pred"].values))
        rows.append(row)
    pb = pd.DataFrame(rows); pb.to_csv(os.path.join(out, "ex4_per_bearing_kappa0.7.csv"), index=False)
    d1 = {r.bearing: -r.accepted_error_rate_both for r in pb.itertuples()}
    d0 = {r.bearing: -r.accepted_error_rate_conf for r in pb.itertuples()}
    d2 = {r.bearing: -r.accepted_error_rate_confglobal for r in pb.itertuples()}
    log("  paired (kappa 0.7), fewer accepted errors per bearing, confidence+verifier vs confidence per class:",
        json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in paired_compare(d1, d0).items()}))
    log("  paired (kappa 0.7), confidence+verifier vs confidence global (eq. 8):",
        json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in paired_compare(d1, d2).items()}))
    # A2R and L8 with the same machinery (training bearings of the protocol only)
    for pname, trb, teb, cond in [("A2R", A2R_TRAIN, A2R_TEST, None), ("L8_all", L8_TRAIN, L8_TEST, None)]:
        R2 = selective_split(D, X, Q, ev, trb, teb, kappas)
        s2 = summarize_arms(R2, kappas); s2.insert(0, "protocol", pname)
        s2.to_csv(os.path.join(out, f"ex4_{pname}_arms.csv"), index=False)
        log(f"  {pname}:"); log(s2.round(1).to_string(index=False))
        plot_rc(R2, os.path.join(out, f"ex4_risk_coverage_{pname}.png"), f"{pname}: risk-coverage by class")


def selective_split(D, X, Q, ev, trb, teb, kappas, model="LR"):
    """Fixed split: thresholds from inner leave-one-training-bearing-out."""
    zOR, zIR = ev["z_OR"].values, ev["z_IR"].values
    ref = {k: [] for k in ["pred", "conf", "zOR", "zIR", "dist"]}
    for v in trb:
        tr_in = D.idx([u for u in trb if u != v]); te_in = D.idx([v])
        if len(set(D.y[tr_in])) < 3:
            continue                                     # cannot train without a class
        P = fit_predict(model, X, D.y, tr_in, te_in)
        hm = HealthyModel(X[tr_in], Q[tr_in], D.y[tr_in])
        ref["pred"].append(P.argmax(1)); ref["conf"].append(P.max(1))
        ref["zOR"].append(zOR[te_in]); ref["zIR"].append(zIR[te_in]); ref["dist"].append(hm.dist(X[te_in]))
    ref = {k: np.concatenate(v) for k, v in ref.items()}
    ref_sc = arm_scores(ref["pred"], ref["conf"], ref["zOR"], ref["zIR"], ref["dist"], np.ones(len(ref["pred"]), bool), ref)
    tr, te = D.idx(trb), D.idx(teb)
    P = fit_predict(model, X, D.y, tr, te)
    pred, conf = P.argmax(1), P.max(1)
    hm = HealthyModel(X[tr], Q[tr], D.y[tr])
    dist, qok = hm.dist(X[te]), hm.quality_ok(Q[te])
    sc = arm_scores(pred, conf, zOR[te], zIR[te], dist, qok, ref)
    R = pd.DataFrame(dict(bearing=D.bear[te], true=D.y[te], pred=pred, conf=conf, s_conf=sc["conf"],
                          s_ver=sc["ver"], s_both=sc["both"]))
    for kap in kappas:
        R[f"acc_confglobal_{kap}"] = conf >= np.quantile(ref["conf"], 1 - kap)
        for arm in ["conf", "ver", "both"]:
            th = per_class_threshold(ref_sc[arm], ref["pred"], kap)
            R[f"acc_{arm}_{kap}"] = sc[arm] >= np.array([th[k] for k in pred])
    return R


# =====================================================================================
# ex5: robustness to speed and geometry errors and to interference
# =====================================================================================
def impulse_train(n, fs, rate, rms_target, seed, f_res=6000.0, tau=4e-4, jitter=0.005):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    t_imp = np.arange(0, n / fs, 1.0 / rate)
    t_imp = t_imp + rng.normal(0, jitter / rate, len(t_imp))
    L = int(8 * tau * fs)
    tt = np.arange(L) / fs
    pulse = np.exp(-tt / tau) * np.sin(2 * np.pi * f_res * tt)
    for t0 in t_imp:
        i0 = int(t0 * fs)
        if 0 <= i0 < n:
            seg = min(L, n - i0)
            x[i0:i0 + seg] += pulse[:seg]
    return x * rms_target / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def env4k_from_32k(x32):
    e, _ = squared_envelope_4k(np.asarray(x32, dtype=np.float64), FS // 2)
    return e[:16000]


def ex5(D, X, args, log, out):
    log("\n=== ex5: robustness to speed and geometry errors and to interference (LOBO, LR) ===")
    bearings = lobo_bearings(args.quick)
    E4 = D.rows("env4k")
    S, df = ses_batch(E4)
    rows = []
    folds = [(D.idx([b2 for b2 in SINGLE if b2 != b]), D.idx([b])) for b in bearings]

    def eval_shifted(Xt, label):
        allp, allt = [], []
        for tr, te in folds:
            P = _fit_predict_shifted("LR", X, Xt, D.y, tr, te)
            allp.append(P.argmax(1)); allt.append(te)
        te, pr = np.concatenate(allt), np.concatenate(allp)
        rows.append(dict(condition=label, **class_rates(D.y[te], pr)))
        log(f"  {label}: accuracy {rows[-1]['accuracy']:.1f}%, false alarms {rows[-1]['false_alarm_H']:.1f}%, "
            f"missed faults {rows[-1]['missed_fault']:.1f}%")

    eval_shifted(X, "nominal")
    for e in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]:
        eval_shifted(kin_features(S, df, D.fr, fr_scale=1 + e), f"speed error {100 * e:+.1f}%")
    for e in [-0.02, -0.01, 0.01, 0.02]:
        eval_shifted(kin_features(S, df, D.fr, order_scale=1 + e), f"fault-order (geometry) error {100 * e:+.1f}%")
    # interference: impulse trains added to the 32 kHz vibration. Clean and perturbed
    # features, evidence and quality statistics all come from the same 32 kHz pipeline.
    sub = D.idx(bearings)
    allrows = np.arange(len(D.bear))
    log("  building the clean 32 kHz envelopes ...")
    E4c = np.zeros((len(D.bear), 16000)); Q32 = np.zeros((len(D.bear), 2))
    for i in allrows:
        x = D.get("vib32k", [i])[0].astype(np.float64)
        E4c[i] = env4k_from_32k(x)
        Q32[i] = [np.log(np.sqrt(np.mean(x ** 2)) + 1e-12), stats.kurtosis(x)]
    Sc, _ = ses_batch(E4c)
    X32 = kin_features(Sc, df, D.fr)
    ev32 = get_evidence(D, args, log, E4=E4c, fr=D.fr, tag="clean32")
    folds32 = folds

    def eval32(Xt, label):
        allp, allt = [], []
        for tr, te in folds32:
            P = _fit_predict_shifted("LR", X32, Xt, D.y, tr, te)
            allp.append(P.argmax(1)); allt.append(te)
        te, pr = np.concatenate(allt), np.concatenate(allp)
        rows.append(dict(condition=label, **class_rates(D.y[te], pr)))
        log(f"  {label}: accuracy {rows[-1]['accuracy']:.1f}%, false alarms {rows[-1]['false_alarm_H']:.1f}%, "
            f"missed faults {rows[-1]['missed_fault']:.1f}%")

    eval32(X32, "nominal, 32 kHz pipeline")
    R0 = selective_lobo(D, X32, Q32, ev32, bearings, [0.8])
    for r in summarize_arms(R0, [0.8]).itertuples():
        rows.append(dict(condition=f"nominal, 32 kHz pipeline | {r.arm} (target 0.8)", accuracy=r.selective_accuracy,
                         coverage=r.coverage, accepted_false_alarms=r.accepted_false_alarms,
                         accepted_missed_faults=r.accepted_missed_faults))
    rates = {"1.04 x BPFO": 1.04 * ORDERS["BPFO"], "1.015 x BPFO (inside search band)": 1.015 * ORDERS["BPFO"],
             "1.04 x BPFI": 1.04 * ORDERS["BPFI"], "2.1 orders (rig-like)": 2.1}
    levels = [0.3] if args.quick else [0.1, 0.3, 0.6]
    for rname, rorder in rates.items():
        for lev in levels:
            E4p = E4c.copy(); Qp = Q32.copy()
            for i in sub:
                x = D.get("vib32k", [i])[0].astype(np.float64)
                x = x + impulse_train(len(x), FS // 2, rorder * D.fr[i], lev * np.sqrt(np.mean(x ** 2)), seed=i)
                E4p[i] = env4k_from_32k(x)
                Qp[i] = [np.log(np.sqrt(np.mean(x ** 2)) + 1e-12), stats.kurtosis(x)]
            Sp, _ = ses_batch(E4p)
            Xp = kin_features(Sp, df, D.fr)
            label = f"interference at {rname}, RMS ratio {lev}"
            eval32(Xp, label)
            evp = ev32.copy()
            evp.iloc[sub] = get_evidence(D, args, log, E4=E4p[sub], fr=D.fr[sub], tag="pert").values
            R = selective_lobo(D, X32, Q32, ev32, bearings, [0.8], Xtest=Xp, Qtest=Qp, evtest=evp)
            sm = summarize_arms(R, [0.8])
            for r in sm.itertuples():
                rows.append(dict(condition=label + f" | {r.arm} (target 0.8)", accuracy=r.selective_accuracy,
                                 coverage=r.coverage, accepted_false_alarms=r.accepted_false_alarms,
                                 accepted_missed_faults=r.accepted_missed_faults))
    dfo = pd.DataFrame(rows); dfo.to_csv(os.path.join(out, "ex5_robustness.csv"), index=False)
    log(dfo.round(1).to_string(index=False))


# =====================================================================================
# ex6: bearing-diversity learning curves with controlled recording counts
# =====================================================================================
def ex6(D, X, args, log, out):
    log("\n=== ex6: learning curves, 1-5 training bearings per class, fixed recording count ===")
    T = D.rows("tstat").astype(float); Sb = D.rows("bande").astype(float)
    O16 = get_orderspec(D, args.cache, 16.0)
    inputs = {"15 kinematic + LR": (X, "LR"), "15 kinematic + RF": (X, "RF"), "15 kinematic + SVM": (X, "SVM"),
              "order spectrum <=16 + LR": (O16, "LR"), "time stats + band energies + RF": (np.hstack([T, Sb]), "RF")}
    pools = {c: [b for b in SINGLE if bearing_class(b) == c] for c in CLASSES}
    draws = 3 if args.quick else 30
    R_PER_CLASS = 80                                   # training recordings per class, every k
    rng = np.random.default_rng(4)
    rows = []
    for k in range(1, 6):
        for d in range(draws):
            tr_idx, te_b = [], []
            for c in CLASSES:
                pick = rng.permutation(pools[c])
                te_b.append(pick[0])
                trb = pick[1:1 + k]
                per = int(np.ceil(R_PER_CLASS / k))
                for b in trb:
                    ib = D.idx([b])
                    tr_idx.extend(rng.choice(ib, min(per, len(ib)), replace=False))
            tr_idx = np.array(tr_idx); te = D.idx(te_b)
            for name, (Xin, mdl) in inputs.items():
                P = fit_predict(mdl, Xin, D.y, tr_idx, te, seed=d)
                rows.append(dict(k=k, draw=d, input=name, accuracy=acc(D.y[te], P.argmax(1)), n_train=len(tr_idx)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "ex6_learning_curves.csv"), index=False)
    log(df.groupby(["input", "k"])["accuracy"].agg(["mean", "std"]).unstack("k").round(1).to_string())


# =====================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with the Paderborn bearing folders (.mat files)")
    ap.add_argument("--out", required=True, help="results folder")
    ap.add_argument("--cache", default=None, help="feature cache folder (default: <out>/cache)")
    ap.add_argument("--exp", default="ex0,ex1,ex3,ex4", help="comma list of ex0..ex6, or 'all'")
    ap.add_argument("--ex2_protocols", default="L8_c0,L8_all,A2R,LOBO")
    ap.add_argument("--n_random", type=int, default=200, help="random order sets in ex3")
    ap.add_argument("--n_mask", type=int, default=50, help="non-fault masking controls in ex3")
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true", help="short smoke test")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    args.cache = args.cache or os.path.join(args.out, "cache")
    log = Log(args.out)
    exps = ["ex0", "ex1", "ex2", "ex3", "ex4", "ex5", "ex6"] if args.exp == "all" else args.exp.split(",")
    log(f"\n##### ReliSense experiments v1, {time.strftime('%Y-%m-%d %H:%M')}, exp={exps}, quick={args.quick}")
    build_cache(args.data, args.cache, log, need_32k=True)
    D = Data(args.cache)
    log(f"recordings: {len(D.bear)} readable; single-damage bearings present: "
        f"{len(set(D.bear) & set(SINGLE))}/29")
    X = get_kin(D, args.cache)
    for e in exps:
        t0 = time.time()
        {"ex0": ex0, "ex1": ex1, "ex2": ex2, "ex3": ex3, "ex4": ex4, "ex5": ex5, "ex6": ex6}[e](D, X, args, log, args.out)
        log(f"  [{e} finished in {time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
