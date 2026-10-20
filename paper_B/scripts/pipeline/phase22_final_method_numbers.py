"""Phase 22 (v1.1): numbers and figure data of the final ReliSense (ENSX of phase 19) on Paderborn. CPU, about 1-2 minutes.

No model is trained: everything is read from the phase 19 checkpoints (<root>/checkpoints/phase19/) and the physics file.
For the final method (ENSX: mean of the extended kinematic branch LRX and the envelope branch RF) and, for reference, for
LR, LRX and RF, it computes per protocol (L8 cond. 0, L8 all, A2R, LOBO):
  recording accuracy with a 95% interval that resamples test bearings within each class (2000 resamples, seed 0),
  bearings correct (class of highest mean probability), class recalls, false alarms, missed faults, confusion matrix,
  macro AUC (one-vs-rest);
and under LOBO: accuracy and class recall per operating condition, accuracy of every bearing per condition, the
training-only thresholds tau_b (kappa 0.9 / 0.8 / 0.7), acceptance per bearing, median confidence of correct and wrong
recordings, accepted / wrong / referred counts, and the inner-LOBO confidences of the example bearing KA22.
v1.1 also exports, for redrawing two figures in one style: feature box statistics (first BPFO and BPFI harmonic) and
the first two principal components of the 15 standardized features of all 32 bearings; the envelope order spectra
(orders <= 10.5) of the 20 condition-0 recordings of K001, KA16, KA30 and KI18; and spectrograms of 0.5 s of the
8-kHz vibration (pb32_4096.npz) of K001, KA16 and KI18 at condition 0.
Output: <root>/phase22_numbers.json, <root>/phase22_figdata.npz and <root>/phase22_views.npz (small; for the figures)
Colab:  !python /content/drive/MyDrive/ReliSense_study/phase22_final_method_numbers.py --root /content/drive/MyDrive/ReliSense_study
"""
import argparse, itertools, json, os
import numpy as np
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser(); ap.add_argument('--root', default='/content/drive/MyDrive/ReliSense_study'); ap.add_argument('--physics', default=None)
a = ap.parse_args(); CK = a.root + '/checkpoints/phase19'
d = dict(np.load(a.physics or a.root + '/pb32/pb32_physics.npz', allow_pickle=True))
B, y, cond = d['bearing'], d['label'].astype(int), d['condition']
COND0 = 'N15_M07_F10'; CONDS = ['N15_M07_F10', 'N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04']; KAP = (0.9, 0.8, 0.7); MODELS = ('LR', 'LRX', 'RF', 'ENSX')
L8_TRAIN = ['K002', 'KA01', 'KA05', 'KA07', 'KI01', 'KI05', 'KI07']
L8_TEST = ['K001', 'KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']
bearings = list(dict.fromkeys(B)); lab = {b: int(y[B == b][0]) for b in bearings}; org = {b: str(d['origin'][B == b][0]) for b in bearings}
single = [b for b in bearings if lab[b] < 3]
healthy = sorted(b for b in bearings if lab[b] == 0); art = [b for b in bearings if org[b] == 'artificial']
real = [b for b in bearings if org[b] == 'real' and lab[b] < 3]
rng = np.random.default_rng(0)


def summary(P, yt, bt):
    p = P.argmax(1); ok = p == yt; h = yt == 0; out = {'n': int(len(yt)), 'acc': float(ok.mean())}
    bs = sorted(set(bt)); cls = {b: int(yt[bt == b][0]) for b in bs}
    out['bearings_correct'] = int(sum(P[bt == b].mean(0).argmax() == cls[b] for b in bs)); out['bearings'] = len(bs)
    out['wrong_bearings'] = [str(b) for b in bs if P[bt == b].mean(0).argmax() != cls[b]]
    boots = []
    groups = {k: [b for b in bs if cls[b] == k] for k in set(cls.values())}
    idx = {b: np.where(bt == b)[0] for b in bs}
    for _ in range(2000):
        sel = np.concatenate([idx[b] for k, g in groups.items() for b in rng.choice(g, len(g), replace=True)])
        boots.append(ok[sel].mean())
    out['ci95'] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    for k, n in enumerate(('H', 'OR', 'IR')):
        if (yt == k).any(): out[f'recall_{n}'] = float((p[yt == k] == k).mean())
    if h.any(): out['false_alarm'] = float((p[h] != 0).mean())
    out['missed_fault'] = float((p[~h] == 0).mean())
    out['confusion'] = [[int(((yt == i) & (p == j)).sum()) for j in range(3)] for i in range(3)]
    try: out['auc_macro'] = float(roc_auc_score(yt, P, multi_class='ovr', average='macro'))
    except ValueError: out['auc_macro'] = None
    out['per_bearing'] = {b: float(ok[bt == b].mean()) for b in bs}
    return out


R = {'protocols': {}}
for proto, trb, teb, c0 in (('L8_c0', L8_TRAIN, L8_TEST, True), ('L8_all', L8_TRAIN, L8_TEST, False), ('A2R', healthy[:3] + art, healthy[3:] + real, False)):
    r = dict(np.load(f'{CK}/{proto}_0.npz')); te = np.isin(B, teb)
    if c0: te &= cond == COND0
    assert len(r['y']) == te.sum() and (r['y'] == y[te]).all()
    R['protocols'][proto] = {m: summary(r[m], y[te], B[te]) for m in MODELS}

OUT = {m: np.full((len(y), 3), np.nan) for m in MODELS}; INNER = {}
for b in single:
    r = dict(np.load(f'{CK}/lobo_{b}.npz'))
    for m in MODELS: OUT[m][B == b] = r[f'outer_{m}']; INNER[(b, m)] = r[f'inner_{m}']
keep = np.isin(B, single); R['protocols']['LOBO'] = {m: summary(OUT[m][keep], y[keep], B[keep]) for m in MODELS}

# LOBO by operating condition (final method)
P = OUT['ENSX'][keep]; yt, bt, ct = y[keep], B[keep], cond[keep]; pr = P.argmax(1)
R['lobo_by_condition'] = {c: {'acc': float((pr[ct == c] == yt[ct == c]).mean()), **{f'recall_{n}': float((pr[(ct == c) & (yt == k)] == k).mean()) for k, n in enumerate(('H', 'OR', 'IR'))}} for c in CONDS}
R['lobo_bearing_by_condition'] = {b: [float((pr[(bt == b) & (ct == c)] == lab[b]).mean()) for c in CONDS] for b in single}

# abstention details (final method)
conf = P.max(1); tau = {b: [float(np.quantile(INNER[(b, 'ENSX')].max(1), 1 - k)) for k in KAP] for b in single}
acc07 = np.array([conf[i] >= tau[bt[i]][2] for i in range(len(bt))]); ok = pr == yt
R['abstention'] = {'tau': tau, 'tau_mean': [float(np.mean([tau[b][j] for b in single])) for j in range(3)],
                   'acceptance_k07_per_bearing': {b: float(acc07[bt == b].mean()) for b in single},
                   'accuracy_accepted_k07_per_bearing': {b: (float(ok[(bt == b) & acc07].mean()) if ((bt == b) & acc07).any() else None) for b in single},
                   'median_conf_correct': float(np.median(conf[ok])), 'median_conf_wrong': float(np.median(conf[~ok])),
                   'k07_accepted_correct': int((acc07 & ok).sum()), 'k07_accepted_wrong': int((acc07 & ~ok).sum()), 'k07_referred': int((~acc07).sum())}
for j, k in enumerate(KAP):
    a_ = np.array([conf[i] >= tau[bt[i]][j] for i in range(len(bt))])
    R['abstention'][f'kappa_{k}'] = {'coverage': float(a_.mean()), 'sel_acc': float(ok[a_].mean()),
                                     'false_alarm': float((a_ & (yt == 0) & (pr != 0)).sum() / (yt == 0).sum()),
                                     'missed_fault': float((a_ & (yt > 0) & (pr == 0)).sum() / (yt > 0).sum())}
json.dump(R, open(a.root + '/phase22_numbers.json', 'w'), indent=1)
np.savez_compressed(a.root + '/phase22_figdata.npz', prob=P.astype(np.float32), label=yt, bearing=bt, condition=ct,
                    tau_bearings=np.array(single), tau=np.array([tau[b] for b in single], np.float32),
                    inner_conf_KA22=INNER[('KA22', 'ENSX')].max(1).astype(np.float32))

# ------------------------------------------------------------------ v1.1: data of the redrawn views
from scipy.signal import spectrogram
F = np.nan_to_num(d['feat_fixed'].astype(float)); allb = list(dict.fromkeys(B))
def box(v):
    q1, med, q3 = np.percentile(v, [25, 50, 75]); iqr = q3 - q1
    return [float(v[v >= q1 - 1.5 * iqr].min()), float(q1), float(med), float(q3), float(v[v <= q3 + 1.5 * iqr].max())]
Z = (F - F.mean(0)) / (F.std(0) + 1e-12); U, Sv, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
pcs = (Z - Z.mean(0)) @ Vt[:2].T; evr = Sv ** 2 / (Sv ** 2).sum()
g = d['order_grid'].astype(float); gm = g <= 10.5; W = {}
for b in ('K001', 'KA16', 'KA30', 'KI18'):
    O = d['order_fixed'][(B == b) & (cond == COND0)][:, gm].astype(float); lo, hi = np.percentile(O, [1, 99.5])
    W[b] = (np.clip((O - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8); W[b + '_lohi'] = np.array([lo, hi])
views = dict(box_bearings=np.array(allb), box_label=np.array([lab[b] for b in allb]), box_origin=np.array([org[b] for b in allb]),
             box_bpfo1=np.array([box(F[B == b, 0]) for b in allb], np.float32), box_bpfi1=np.array([box(F[B == b, 3]) for b in allb], np.float32),
             pca=pcs.astype(np.float16), pca_evr=evr[:2], pca_bearing=B, pca_label=y, pca_origin=d['origin'], wf_orders=g[gm].astype(np.float32),
             **{'wf_' + k: v for k, v in W.items()})
try:
    win = np.load(a.root + '/pb32/pb32_4096.npz'); rec = np.array([str(r) for r in win['recording']]); X = win['X']
    for b in ('K001', 'KA16', 'KI18'):
        i = [j for j, r in enumerate(rec) if b in r and COND0 in r][0]; x = np.asarray(X[i])
        x = x[0] if x.ndim == 2 else x
        f_, t_, Sxx = spectrogram(x - x.mean(), fs=8000, nperseg=128, noverlap=96); L = 10 * np.log10(Sxx + 1e-20); lo, hi = np.percentile(L, [1, 99.9])
        views['sg_' + b] = (np.clip((L - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8); views['sg_' + b + '_lohi'] = np.array([lo, hi]); views['sg_' + b + '_rec'] = np.array(rec[i])
    views['sg_f'] = f_.astype(np.float32); views['sg_t'] = t_.astype(np.float32)
    print('spectrograms from', [str(views['sg_' + b + '_rec']) for b in ('K001', 'KA16', 'KI18')])
except Exception as e:
    print('spectrograms skipped:', e)
np.savez_compressed(a.root + '/phase22_views.npz', **views)

print('Final method (ENSX)   accuracy (95% CI) | bearings correct | recall H / OR / IR | false alarms | missed | macro AUC')
for proto in ('L8_c0', 'L8_all', 'A2R', 'LOBO'):
    s = R['protocols'][proto]['ENSX']
    rec = ' / '.join(f'{100 * s[k]:.1f}' if k in s else '-' for k in ('recall_H', 'recall_OR', 'recall_IR'))
    print(f"  {proto:7} {100 * s['acc']:.1f} ({100 * s['ci95'][0]:.1f}-{100 * s['ci95'][1]:.1f}) | {s['bearings_correct']}/{s['bearings']} {s['wrong_bearings']} | {rec} | "
          f"{100 * s.get('false_alarm', float('nan')):.1f} | {100 * s['missed_fault']:.1f} | {s['auc_macro'] if s['auc_macro'] is None else round(100 * s['auc_macro'], 1)}")
ab = R['abstention']
print(f"abstention: median confidence correct {ab['median_conf_correct']:.2f}, wrong {ab['median_conf_wrong']:.2f}; mean tau {[round(t, 2) for t in ab['tau_mean']]}")
for k in KAP: s = ab[f'kappa_{k}']; print(f"  kappa {k}: coverage {100 * s['coverage']:.1f}  acc {100 * s['sel_acc']:.1f}  FA {100 * s['false_alarm']:.1f}  missed {100 * s['missed_fault']:.1f}")
print('by condition:', {c: round(100 * v['acc'], 1) for c, v in R['lobo_by_condition'].items()})
print('written', a.root + '/phase22_numbers.json, phase22_figdata.npz and phase22_views.npz')
