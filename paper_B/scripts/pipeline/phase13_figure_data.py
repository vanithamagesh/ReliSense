"""Phase 13 (v1.1): every step and every fitted model is saved as a checkpoint and reused on the next run.
Phase 13 (v1.0): export compact real-data arrays for the redrawn manuscript figures (CPU, a few minutes).

Writes phase13_figdata.npz (well below 10 MB) next to the pb32 folder:
  signature map      median log order spectrum of every bearing (32 x 590)
  signal chain       one raw 8-kHz window, its 2-12-kHz envelope window, the order spectrum and the 15 features
                     of the first condition-0 recording of KA16 (outer race), KI18 (inner race) and K001 (healthy)
  folding fields     K001 and KA16 at BPFO, KI18 at BPFI, measured and 10%-offset period (120 x 64 each),
                     and shaft-angle fields of KI18 and K001 (80 x 180)
  LOBO abstention    class probabilities of the primary model under LOBO for every recording, and the
                     training-only thresholds tau_b for target coverages 0.9, 0.8 and 0.7 (inner LOBO over the
                     28 training bearings), plus the pooled inner confidences of one example test bearing
Nothing is trained on test bearings; the model is the fixed primary model (standardization + LR, C = 0.3).
Run in Colab after mounting Drive:   !python phase13_figure_data.py --root /content/drive/MyDrive/ReliSense_study
"""
import argparse, csv, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser(); ap.add_argument('--root', default='/content/drive/MyDrive/ReliSense_study')
ap.add_argument('--labels', default=None); ap.add_argument('--fresh', action='store_true', help='ignore saved checkpoints')
a = ap.parse_args()
ROOT = a.root; PB = ROOT + '/pb32'; LABELS = a.labels or ROOT + '/phase1/labels_32.csv'
import os, joblib
CK = ROOT + '/checkpoints/phase13'; os.makedirs(CK + '/models', exist_ok=True)
def ck_load(name):
    p = f'{CK}/{name}.npz'
    if not a.fresh and os.path.exists(p):
        print(f'  reused checkpoint {name}', flush=True); return dict(np.load(p, allow_pickle=True))
    return None
def ck_save(name, dct): np.savez_compressed(f'{CK}/{name}.npz', **dct); print(f'  saved checkpoint {name}', flush=True)
def fit_cached(tag, Xtr, ytr):
    p = f'{CK}/models/{tag}.joblib'
    if not a.fresh and os.path.exists(p): return joblib.load(p)
    m = lr().fit(Xtr, ytr); joblib.dump(m, p); return m
t0 = time.time(); FS = 8000.0; COND0 = 'N15_M07_F10'
n, dd, D = 8, 6.75, 28.55
FO = {'BPFO': n / 2 * (1 - dd / D), 'BPFI': n / 2 * (1 + dd / D), 'BSF': D / (2 * dd) * (1 - (dd / D) ** 2), 'FTF': 0.5 * (1 - dd / D)}

d = dict(np.load(PB + '/pb32_physics.npz', allow_pickle=True))
B, y, cond, rec, rpm = d['bearing'], d['label'].astype(int), d['condition'], d['recording'], d['rpm'].astype(float)
grid = d['order_grid']; O = d['order_fixed'].astype(np.float32); F = d['feat_fixed'].astype(float)
meta = {r['bearing']: r for r in csv.DictReader(open(LABELS))}
order_b = [b for b in meta if b in set(B)]
out = {'grid': grid.astype(np.float32), 'sig_bearings': np.array(order_b), 'sig_labels': np.array([int(meta[b]['label']) for b in order_b]),
       'feature_names': np.array(d['feature_names'])}
c = ck_load('sig_map')
if c is None: c = {'sig_map': np.stack([np.median(O[B == b], 0) for b in order_b]).astype(np.float16)}; ck_save('sig_map', c)
out.update(c)
print('signature map', out['sig_map'].shape, f'{time.time() - t0:.0f} s', flush=True)

# ------------------------------------------------------------------ signal chain (raw window, envelope window, order spectrum, features)
c = ck_load('signal_chain')
if c is None:
  c = {}
  w = np.load(PB + '/pb32_4096.npz'); WB, WR, WC = w['bearing'], w['recording'], w['condition']; WX = w['X']
  e = np.load(PB + '/pb32_4096_envelope.npz'); EX, ER = e['X'], e['recording']
  for b in ('KA16', 'KI18', 'K001'):
    i = np.flatnonzero((B == b) & (cond == COND0))[0]; r = rec[i]
    iw = np.flatnonzero(WR == r)[0]; ie = np.flatnonzero(ER == r)[0]
    c[f'raw_{b}'] = WX[iw, 0].astype(np.float32); c[f'env_{b}'] = EX[ie, 0].astype(np.float32)
    c[f'order_{b}'] = O[i].astype(np.float32); c[f'feat_{b}'] = F[i].astype(np.float32); c[f'rpm_{b}'] = np.float32(rpm[i])
  del WX, w
  for b in ('K001', 'KA16', 'KI18'):           # full-recording envelopes for the folding fields
    i = np.flatnonzero((B == b) & (cond == COND0))[0]; c[f'recenv_{b}'] = EX[ER == rec[i], 0].reshape(-1).astype(np.float32)
  del EX
  ck_save('signal_chain', c)
out.update({k: v for k, v in c.items() if not k.startswith('recenv_')})
print('signal chain done', f'{time.time() - t0:.0f} s', flush=True)


# ------------------------------------------------------------------ folding fields (as phase 7)


def exact_rate(x, f0, tol=0.02):
    x = x - x.mean(); m_ = 16 * len(x); S = np.abs(np.fft.rfft(x * np.hanning(len(x)), m_)); f = np.fft.rfftfreq(m_, 1 / FS)
    m = np.abs(f - f0) <= tol * f0
    return float(f[m][np.argmax(S[m])]) if m.any() else f0


def fold(x, rate, n_cycles=120, n_phase=64):
    T = 1.0 / rate; nn = min(n_cycles, int(len(x) / FS / T) - 1)
    t = (np.arange(nn)[:, None] + (np.arange(n_phase)[None, :] + 0.5) / n_phase) * T
    Fm = np.interp(t * FS, np.arange(len(x)), x); return Fm / (np.median(Fm) + 1e-12)


rates = {}
for b, k in (('K001', 'BPFO'), ('KA16', 'BPFO'), ('KI18', 'BPFI')):
    i = np.flatnonzero((B == b) & (cond == COND0))[0]; x = c[f'recenv_{b}'].astype(float); f0 = FO[k] * rpm[i] / 60
    fx = exact_rate(x, f0); rates[b] = (f0, fx)
    out[f'fold_{b}'] = fold(x, fx).astype(np.float16); out[f'fold10_{b}'] = fold(x, fx * 1.10).astype(np.float16)
    if b in ('KI18', 'K001'):
        fr = exact_rate(x, rpm[i] / 60, 0.03); out[f'shaft_{b}'] = fold(x, fr, 80, 180).astype(np.float16)
out['fold_rates'] = np.array([[rates[b][0], rates[b][1]] for b in ('K001', 'KA16', 'KI18')], dtype=np.float32)
print('folding fields', {b: (round(v[0], 3), round(v[1], 3)) for b, v in rates.items()}, f'{time.time() - t0:.0f} s', flush=True)

# ------------------------------------------------------------------ LOBO probabilities and training-only abstention thresholds
lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))
single = [b for b in order_b if int(meta[b]['label']) < 3]
keep = np.isin(B, single); prob = np.full((len(y), 3), np.nan, np.float32)
KAP = (0.9, 0.8, 0.7); tau = np.zeros((len(single), len(KAP)), np.float32); example = 'KA22'
for ib, b in enumerate(single):
    tr = [x for x in single if x != b]; te = B == b
    cb = ck_load(f'lobo_{b}')
    if cb is None:
        pb_ = fit_cached(f'lobo_{b}', F[np.isin(B, tr)], y[np.isin(B, tr)]).predict_proba(F[te])
        pool = []
        for v in tr:
            tr2 = [x for x in tr if x != v]
            pool.append(fit_cached(f'inner_{b}_{v}', F[np.isin(B, tr2)], y[np.isin(B, tr2)]).predict_proba(F[B == v]).max(1))
        cb = {'prob': pb_, 'pool': np.concatenate(pool)}; ck_save(f'lobo_{b}', cb)
    prob[te] = cb['prob']; pool = cb['pool']; tau[ib] = [np.quantile(pool, 1 - k) for k in KAP]
    if b == example: out['inner_conf_example'] = pool.astype(np.float32)
    print(f'  {b}: tau {np.round(tau[ib], 3)}', flush=True)
conf = np.nan_to_num(prob).max(1); pred = np.nan_to_num(prob).argmax(1); ok = (pred == y) & keep
tb = {b: tau[i] for i, b in enumerate(single)}
for j, k in enumerate(KAP):
    acc_m = np.array([conf[i] >= tb[B[i]][j] if keep[i] else False for i in range(len(y))])
    print(f'target {k:.1f}: coverage {acc_m[keep].mean() * 100:.1f}%, selective accuracy {ok[acc_m].mean() * 100:.1f}%', flush=True)
print(f'LOBO accuracy without abstention {ok[keep].mean() * 100:.1f}%', flush=True)
out.update({'lobo_prob': prob, 'lobo_bearing': B, 'lobo_label': y, 'lobo_condition': cond, 'tau_bearings': np.array(single), 'tau': tau,
            'tau_kappa': np.array(KAP, np.float32), 'example_bearing': np.array(example)})
np.savez_compressed(ROOT + '/phase13_figdata.npz', **out)
import os
print('written', ROOT + '/phase13_figdata.npz', f'{os.path.getsize(ROOT + "/phase13_figdata.npz") / 1e6:.2f} MB', f'{time.time() - t0:.0f} s')
