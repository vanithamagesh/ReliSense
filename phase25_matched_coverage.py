"""Phase 25 (v1.0): abstention at matched coverage under LOBO, from stored predictions only (no training, CPU, minutes).

For ReliSense (phase22_figdata.npz: LOBO posteriors of all 2319 recordings and the training-only thresholds tau_b) and for
every network whose LOBO predictions were checkpointed in phase 23 (checkpoints/phase23/<name>_LOBO_<fold>_s0.npz) or
phase 24 part 2 (checkpoints/phase24/p2_<name>_LOBO_<fold>.npz), the recordings are pooled and the most confident ones are
accepted until the coverage of ReliSense at kappa = 0.9, 0.8 and 0.7 is reached (confidence = largest class probability).
This global ranking uses the test recordings and is therefore descriptive and slightly favourable to the networks; ReliSense
is reported both with its training-only thresholds and with the same global ranking. Reported per method and coverage:
selective accuracy, error among accepted recordings, false alarms (healthy recordings accepted and diagnosed as damaged, as a
share of all healthy recordings) and missed faults (damaged recordings accepted and diagnosed as healthy, as a share of all
damaged recordings), and coverage of each true class.
Output: <root>/phase25_matched_coverage.json and the printed table.
Colab (CPU runtime is enough):
  !python /content/drive/MyDrive/ReliSense_study/phase25_matched_coverage.py --root /content/drive/MyDrive/ReliSense_study
"""
import argparse, glob, json, os, re
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--root', default='/content/drive/MyDrive/ReliSense_study')
ap.add_argument('--relisense', default=None, help='path of phase22_figdata.npz (default: <root>/phase22_figdata.npz)')
a = ap.parse_args(); R = a.root
fp = a.relisense or R + '/phase22_figdata.npz'
if not os.path.exists(fp):
    hits = glob.glob(R + '/**/phase22_figdata.npz', recursive=True)
    if not hits: raise SystemExit('phase22_figdata.npz not found; pass --relisense <path>')
    fp = hits[0]
F = dict(np.load(fp, allow_pickle=True))
RS = {'P': F['prob'].astype(float), 'y': F['label'].astype(int), 'b': F['bearing'].astype(str)}
tb = [str(b) for b in F['tau_bearings']]; KAP = (0.9, 0.8, 0.7)
tau = np.array([[F['tau'][tb.index(b), j] for j in range(3)] for b in RS['b']])
print('ReliSense posteriors:', fp, len(RS['y']), 'recordings')


def lobo_runs():
    out = {}
    for f in sorted(glob.glob(R + '/checkpoints/phase23/*_LOBO_*_s0.npz')):
        m = re.match(r'(.+)_LOBO_(\d+)_s0\.npz$', os.path.basename(f)); out.setdefault('p23_' + m.group(1), []).append(f)
    for f in sorted(glob.glob(R + '/checkpoints/phase24/p2_*_LOBO_*.npz')):
        m = re.match(r'p2_(.+)_LOBO_(\d+)\.npz$', os.path.basename(f)); out.setdefault('p24_' + m.group(1), []).append(f)
    return out


def load(files):
    P, y, b = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True); P.append(d['P'].astype(float)); y.append(d['y'].astype(int)); b.append(d['b'].astype(str))
    return {'P': np.concatenate(P), 'y': np.concatenate(y), 'b': np.concatenate(b)}


def stats(P, y, acc):
    pred = P.argmax(1); h = y == 0; d = ~h
    r = {'coverage': 100 * acc.mean(), 'sel_acc': 100 * (pred[acc] == y[acc]).mean() if acc.any() else float('nan'),
         'err_accepted': 100 * (pred[acc] != y[acc]).mean() if acc.any() else float('nan'),
         'false_alarms': 100 * (acc & h & (pred != 0)).sum() / max(h.sum(), 1), 'missed': 100 * (acc & d & (pred == 0)).sum() / max(d.sum(), 1)}
    for c, n in enumerate(('H', 'OR', 'IR')): r[f'cov_{n}'] = 100 * acc[y == c].mean()
    return r


def at_coverage(P, q):
    conf = P.max(1); k = int(round(q * len(conf)))
    acc = np.zeros(len(conf), bool); acc[np.argsort(-conf, kind='stable')[:k]] = True
    return acc


RES = {'relisense_training_thresholds': {}, 'relisense_ranked': {}}
targets = {}
for j, k in enumerate(KAP):
    accT = RS['P'].max(1) >= tau[:, j]; q = accT.mean(); targets[k] = q
    RES['relisense_training_thresholds'][str(k)] = stats(RS['P'], RS['y'], accT)
    RES['relisense_ranked'][str(k)] = stats(RS['P'], RS['y'], at_coverage(RS['P'], q))
RES['relisense_training_thresholds']['all'] = stats(RS['P'], RS['y'], np.ones(len(RS['y']), bool))
RES['relisense_ranked']['all'] = RES['relisense_training_thresholds']['all']
for name, files in lobo_runs().items():
    D = load(files)
    if len(files) != 29: print(f'  {name}: {len(files)} of 29 LOBO folds found; skipped'); continue
    RES[name] = {str(k): stats(D['P'], D['y'], at_coverage(D['P'], targets[k])) for k in KAP}
    RES[name]['all'] = stats(D['P'], D['y'], np.ones(len(D['y']), bool)); RES[name]['n'] = int(len(D['y']))
json.dump(RES, open(R + '/phase25_matched_coverage.json', 'w'), indent=1)

print('\nLOBO, pooled recordings. Coverage targets = ReliSense coverage at kappa 0.9 / 0.8 / 0.7:',
      ' / '.join(f'{100 * targets[k]:.1f}%' for k in KAP))
print('Columns per coverage: selective accuracy / false alarms / missed faults (%); "all" = no abstention')
for name, r in RES.items():
    line = f'{name:32s} | all {r["all"]["sel_acc"]:5.1f} / {r["all"]["false_alarms"]:4.1f} / {r["all"]["missed"]:4.1f}'
    for k in KAP:
        s = r[str(k)]; line += f' | {s["coverage"]:4.1f}%: {s["sel_acc"]:5.1f} / {s["false_alarms"]:4.1f} / {s["missed"]:4.1f}'
    print(line)
print('\nCoverage of each true class at the lowest coverage (H / OR / IR):')
for name, r in RES.items():
    s = r[str(KAP[-1])]; print(f'{name:32s} | {s["cov_H"]:5.1f} / {s["cov_OR"]:5.1f} / {s["cov_IR"]:5.1f}')
print('written', R + '/phase25_matched_coverage.json')
