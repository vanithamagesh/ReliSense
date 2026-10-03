"""Draw the manuscript figures from the finished study jobs (600 dpi PNG + PDF).

Usage: python make_figures.py --runs study_runs --data paderborn_15bearings_all.npz --out figures
Every figure is computed from metrics.json / preds.npz of the jobs; nothing is typed in by hand.
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, roc_auc_score

COL = {'relisense': '#2a78d6', 'transformer': '#eb6834', 'wdcnn': '#1baf7a', 'tcn': '#eda100', 'envelope_rf': '#e87ba4',
       'relisense_uniform': '#008300', 'relisense_norec': '#4a3aa7', 'relisense_noq': '#e34948',
       'relisense_p': '#0b3d91', 'physics_lr': '#8a6d00', 'relisense_wn': '#7a7a7a'}
NAMES = {'relisense': 'ReliSense', 'transformer': 'Temporal Transformer', 'wdcnn': 'WDCNN', 'tcn': 'TCN',
         'envelope_rf': 'Envelope + RF', 'relisense_uniform': 'Uniform weights', 'relisense_norec': 'No reconstruction',
         'relisense_noq': 'No quality supervision', 'relisense_p': 'ReliSense-P', 'physics_lr': 'Physics + LR',
         'relisense_wn': 'ReliSense-WN'}
MAIN = ['relisense_p', 'physics_lr', 'relisense', 'transformer', 'wdcnn', 'tcn', 'envelope_rf']
SENSORS = ['Vibration', 'Current 1', 'Current 2']
CLASSES = ['Healthy', 'Outer ring', 'Inner ring']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'legend.fontsize': 7, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6,
                     'font.family': 'DejaVu Sans', 'savefig.dpi': 600, 'figure.dpi': 120})


def save(fig, out, name):
    fig.savefig(out / f'{name}.png', bbox_inches='tight'); fig.savefig(out / f'{name}.pdf', bbox_inches='tight'); plt.close(fig)
    print('wrote', name)


def load(runs):
    jobs = []
    for f in sorted(Path(runs).glob('*/metrics.json')):
        m = json.loads(f.read_text()); m['sc'] = {s['scenario']: s for s in m['scenarios']}; m['dir'] = f.parent; jobs.append(m)
    return jobs


def pick(jobs, m, joint=False):
    return [j for j in jobs if j['model'] == m and (j['heldout'] is not None) == joint]


def fig_confusion(jobs, out):
    ms = [m for m in MAIN if pick(jobs, m)]
    fig, axs = plt.subplots(1, len(ms), figsize=(2.1 * len(ms), 2.3), squeeze=False)
    for ax, m in zip(axs[0], ms):
        cm = sum(np.array(j['sc']['clean']['confusion_matrix']) for j in pick(jobs, m))
        pct = 100 * cm / cm.sum(1, keepdims=True).clip(1)
        ax.imshow(pct, cmap='Blues', vmin=0, vmax=100); ax.grid(False)
        for i in range(3):
            for k in range(3):
                ax.text(k, i, f'{pct[i, k]:.0f}', ha='center', va='center', fontsize=7, color='white' if pct[i, k] > 60 else INK)
        ax.set_xticks(range(3), ['H', 'OR', 'IR']); ax.set_yticks(range(3), ['H', 'OR', 'IR'])
        ax.set_title(NAMES[m]); ax.set_xlabel('Predicted')
    axs[0][0].set_ylabel('True')
    fig.suptitle('Clean test recordings, summed over runs (row %)', fontsize=9, y=1.03)
    save(fig, out, 'fig_confusion_clean')


def mean_f1(job, pred):
    v = [s['macro_f1'] for s in job['scenarios'] if pred(s)]
    return np.mean(v) if v else np.nan


def fig_robustness(jobs, out):
    kinds = ['noise', 'bias', 'gain', 'drift', 'clip', 'spike', 'quant']
    titles = {'noise': 'Noise', 'bias': 'Bias', 'gain': 'Gain', 'drift': 'Drift', 'clip': 'Clipping',
              'spike': 'Spikes (test-only)', 'quant': 'Quantisation (test-only)'}
    ms = [m for m in MAIN if pick(jobs, m)]
    fig, axs = plt.subplots(2, 4, figsize=(9, 4.4), sharey=True); axs = axs.ravel()
    for ax, kd in zip(axs, kinds):
        for m in ms:
            levels = sorted({s['level'] for j in pick(jobs, m) for s in j['scenarios'] if s['kind'] == kd})
            ys = [[mean_f1(j, lambda s: s['kind'] == kd and s['level'] == lv) for j in pick(jobs, m)] for lv in levels]
            mu = [100 * np.nanmean(y) for y in ys]; sd = [100 * np.nanstd(y) for y in ys]
            ax.errorbar(levels, mu, yerr=sd, color=COL[m], lw=2, marker='o', ms=4, capsize=2, label=NAMES[m])
        clean = [100 * np.mean([j['sc']['clean']['macro_f1'] for j in pick(jobs, 'relisense')])] if pick(jobs, 'relisense') else []
        if clean: ax.axhline(clean[0], color=COL['relisense'], ls=':', lw=1)
        ax.set_title(titles[kd]); ax.set_xlabel('Severity (training SD)'); ax.set_ylim(0, 100)
    ax = axs[7]; ax.axis('off'); h, l = axs[0].get_legend_handles_labels(); ax.legend(h, l, loc='center', frameon=False)
    axs[0].set_ylabel('Macro F1 (%)'); axs[4].set_ylabel('Macro F1 (%)')
    fig.suptitle('Robustness to simulated sensor faults (mean ± SD over runs; mean over the three sensors). '
                 'Dotted: ReliSense clean.', fontsize=9)
    fig.tight_layout(); save(fig, out, 'fig_robustness')


def fig_sensor_sets(jobs, out):
    sets = [('clean', 'All three'), ('set_vib_cur1', 'Vibration +\ncurrent 1'), ('set_currents_only', 'Currents\nonly'),
            ('set_vib_only', 'Vibration\nonly'), ('missing_0', 'Vibration\nlost'), ('missing_1', 'Current 1\nlost'),
            ('missing_2', 'Current 2\nlost')]
    ms = [m for m in MAIN if pick(jobs, m)]
    fig, ax = plt.subplots(figsize=(7.2, 2.8)); w = .8 / len(ms); x = np.arange(len(sets))
    for i, m in enumerate(ms):
        v = np.array([[j['sc'][k]['macro_f1'] for j in pick(jobs, m)] for k, _ in sets]) * 100
        ax.bar(x + (i - (len(ms) - 1) / 2) * w, v.mean(1), w * .92, yerr=v.std(1), color=COL[m], label=NAMES[m],
               error_kw=dict(lw=.8, capsize=1.5, ecolor=MUTED))
    ax.set_xticks(x, [s[1] for s in sets]); ax.set_ylabel('Macro F1 (%)'); ax.set_ylim(0, 105); ax.grid(axis='x', visible=False)
    ax.legend(ncol=len(ms), loc='upper center', bbox_to_anchor=(.5, 1.18), frameon=False)
    save(fig, out, 'fig_sensor_sets')


def fig_weights(jobs, out):
    b = pick(jobs, 'relisense_p') or pick(jobs, 'relisense')
    if not b: return
    rows = [('clean', 'Clean')] + [(f'missing_{s}', f'{SENSORS[s]} lost') for s in range(3)] + \
           [(f'{kd}_{s}_1', f'{kd.capitalize()} on {SENSORS[s]}') for kd in ('noise', 'bias', 'drift') for s in range(3)]
    W = np.array([np.mean([j['sc'][k]['mean_sensor_weights'] for j in b], 0) for k, _ in rows])
    fig, ax = plt.subplots(figsize=(3.6, 4.6))
    im = ax.imshow(W, cmap='Blues', vmin=0, vmax=1, aspect='auto'); ax.grid(False)
    for i in range(len(rows)):
        for k in range(3):
            ax.text(k, i, f'{W[i, k]:.2f}', ha='center', va='center', fontsize=7, color='white' if W[i, k] > .6 else INK)
    ax.set_xticks(range(3), SENSORS); ax.set_yticks(range(len(rows)), [r[1] for r in rows]); ax.xaxis.tick_top()
    fig.colorbar(im, ax=ax, fraction=.05, pad=.03, label='Mean fusion weight')
    ax.set_title('ReliSense fusion weights (severity 1.0)', pad=22)
    save(fig, out, 'fig_sensor_weights')


def pooled(jobs, m, scen_pred):
    P, Y = [], []
    for j in pick(jobs, m):
        z = np.load(j['dir'] / 'preds.npz')
        for s in j['scenarios']:
            if scen_pred(s): P.append(z[f'p__{s["scenario"]}']); Y.append(z['y_rec'])
    return (np.concatenate(P), np.concatenate(Y)) if P else (None, None)


def fig_reliability(jobs, out):
    ms = [m for m in MAIN if pick(jobs, m)]
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3), sharey=True)
    for ax, (title, pred) in zip(axs, [('Clean test', lambda s: s['kind'] == 'clean'),
                                       ('Corrupted tests (pooled)', lambda s: s['kind'] not in ('clean', 'set'))]):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls='--')
        for m in ms:
            p, y = pooled(jobs, m, pred); conf = p.max(1); ok = p.argmax(1) == y
            edges = np.linspace(1 / 3, 1, 9); xs, ys = [], []
            for a, b in zip(edges[:-1], edges[1:]):
                k = (conf > a) & (conf <= b)
                if k.sum() >= 5: xs.append(conf[k].mean()); ys.append(ok[k].mean())
            ax.plot(xs, ys, color=COL[m], lw=2, marker='o', ms=3.5, label=NAMES[m])
        ax.set_title(title); ax.set_xlabel('Confidence (recording)'); ax.set_xlim(.3, 1.01); ax.set_ylim(0, 1.01)
    axs[0].set_ylabel('Observed accuracy'); axs[1].legend(frameon=False, loc='lower right')
    fig.tight_layout(); save(fig, out, 'fig_reliability')


def fig_risk_coverage(jobs, out):
    ms = [m for m in MAIN if pick(jobs, m)]
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    for ax, (title, pred) in zip(axs, [('Clean test', lambda s: s['kind'] == 'clean'),
                                       ('Corrupted tests (pooled)', lambda s: s['kind'] not in ('clean', 'set'))]):
        for m in ms:
            p, y = pooled(jobs, m, pred); o = np.argsort(-p.max(1)); err = (p.argmax(1)[o] != y[o]).astype(float)
            cov = np.arange(1, len(err) + 1) / len(err)
            ax.plot(cov * 100, 100 * np.cumsum(err) / np.arange(1, len(err) + 1), color=COL[m], lw=2, label=NAMES[m])
        ax.set_title(title); ax.set_xlabel('Coverage (% recordings accepted)')
    axs[0].set_ylabel('Risk (% error among accepted)'); axs[1].legend(frameon=False)
    fig.tight_layout(); save(fig, out, 'fig_risk_coverage')


def fig_quality_roc(jobs, out):
    b = pick(jobs, 'relisense_p') or pick(jobs, 'relisense')
    if not b: return
    fams = [('noise', 'Noise'), ('bias', 'Bias'), ('gain', 'Gain'), ('drift', 'Drift'), ('clip', 'Clipping'),
            ('spike', 'Spikes (test-only)'), ('quant', 'Quantisation (test-only)')]
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
    fig, ax = plt.subplots(figsize=(3.6, 3.4)); ax.plot([0, 1], [0, 1], color=MUTED, ls='--', lw=1)
    for (kd, label), c in zip(fams, colors):
        S, Yb = [], []
        for j in b:
            z = np.load(j['dir'] / 'preds.npz')
            for s in j['scenarios']:
                if s['kind'] != kd: continue
                q = z[f'q__{s["scenario"]}'].astype(float); bad = np.zeros_like(q, bool); bad[:, s['sensor']] = True
                S.append((1 - q).ravel()); Yb.append(bad.ravel())
        if not S: continue
        S, Yb = np.concatenate(S), np.concatenate(Yb); fpr, tpr, _ = roc_curve(Yb, S)
        ax.plot(fpr, tpr, color=c, lw=2, label=f'{label} (AUROC {roc_auc_score(Yb, S):.2f})')
    ax.set_xlabel('False-alarm rate'); ax.set_ylabel('Detection rate'); ax.legend(frameon=False, fontsize=6.5, loc='lower right')
    ax.set_title('Sensor-fault isolation by the quality head')
    save(fig, out, 'fig_quality_roc')


def fig_training(jobs, out):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for m in [x for x in MAIN if x != 'envelope_rf' and pick(jobs, x)]:
        for i, j in enumerate(pick(jobs, m)):
            h = [r for r in json.loads((j['dir'] / 'history.json').read_text()) if r['stage'] == 'supervised']
            ax.plot([r['epoch'] for r in h], [100 * r['val_macro_f1'] for r in h], color=COL[m], lw=.8, alpha=.35,
                    label=NAMES[m] if i == 0 else None)
    ax.set_xlabel('Supervised epoch'); ax.set_ylabel('Validation macro F1 (%, windows)'); ax.legend(frameon=False)
    save(fig, out, 'fig_training_curves')


def fig_signals(data_path, out):
    import torch
    from study import corrupt_eval
    d = np.load(data_path); X = d['X']; y = d['y']; fs = float(d['fs'])
    i = int(np.flatnonzero(y == 1)[0]); mu = X.mean((0, 2), keepdims=True)[0]; sd = X.std((0, 2), keepdims=True)[0]
    x = torch.tensor(((X[i] - mu) / sd)[None].astype('float32'))
    cases = [('clean', None, 0, 'Clean'), ('noise', 0, 1., 'Noise on vibration'), ('drift', 1, 1., 'Drift on current 1'),
             ('clip', 0, .5, 'Clipping on vibration'), ('missing', 2, 0, 'Current 2 lost')]
    n = 1024; t = np.arange(n) / fs * 1000
    fig, axs = plt.subplots(len(cases), 3, figsize=(7.2, 6), sharex=True)
    for r, (kd, s, lv, label) in enumerate(cases):
        z = corrupt_eval(x, kd, s, lv, seed=1)[0][0].numpy()
        for c in range(3):
            ax = axs[r, c]; ax.plot(t, z[c, :n], color=COL['relisense'] if (s != c or kd == 'clean') else '#e34948', lw=.6)
            ax.grid(False)
            if r == 0: ax.set_title(SENSORS[c])
            if c == 0: ax.set_ylabel(label, rotation=0, ha='right', va='center')
    for c in range(3): axs[-1, c].set_xlabel('Time (ms)')
    fig.suptitle('One outer-ring test window (normalised units); red = corrupted channel', fontsize=9)
    fig.tight_layout(); save(fig, out, 'fig_example_signals')


def fig_architecture(out):
    fig, ax = plt.subplots(figsize=(9.5, 3.9)); ax.axis('off'); ax.set_xlim(-1.5, 121.5); ax.set_ylim(0, 50)
    def box(x, y, w, h, text, fc='#eef4fc', ec='#2a78d6'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1', fc=fc, ec=ec, lw=.9))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=6, color=INK, linespacing=1.25)
    def arr(x0, y0, x1, y1):
        ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='->', color=MUTED, lw=.8))
    for s, y in zip(SENSORS, (36, 22, 8)):
        box(0, y, 13, 7, f'{s}\n4,096 samples')
        box(17, y, 15, 7, 'Conv stem + BN\n(shared)')
        box(36, y, 17, 7, 'Transformer, 2 layers\n+ sensor embedding')
        arr(13.5, y + 3.5, 16.5, y + 3.5); arr(32.5, y + 3.5, 35.5, y + 3.5); arr(53.5, y + 3.5, 57.5, 25.5)
    box(58, 19, 16, 13, 'Cross-sensor\nattention\n(availability\nmask)')
    box(58, 38, 16, 8, 'Amplitude statistics\nof each input', fc='#fdf0ea', ec='#eb6834')
    box(79, 34, 16, 10, 'Reliability head\nquality q per sensor', fc='#fdf0ea', ec='#eb6834')
    box(79, 19, 16, 10, 'Quality-weighted\nfusion  \u03a3 w\u00b7h')
    box(100, 27, 16, 8, 'Classifier\nhealthy / OR / IR')
    box(100, 15, 16, 8, 'Decoder\n(reconstruction loss)', fc='#f4f3f0', ec=MUTED)
    box(79, 1, 37, 8, 'Temperature scaling \u00b7 recording averaging\nconformal sets \u00b7 rejection', fc='#f4f3f0', ec=MUTED)
    arr(74.5, 42, 78.5, 40); arr(74.5, 28, 78.5, 37); arr(74.5, 24, 78.5, 24); arr(87, 33.5, 87, 29.5)
    arr(95.5, 25, 99.5, 30); arr(95.5, 23, 99.5, 19); ax.plot([116.5, 119.5, 119.5], [31, 31, 5], color=MUTED, lw=.8); arr(119.5, 5, 116.7, 5)
    save(fig, out, 'fig_architecture')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', default='study_runs'); ap.add_argument('--data')
    ap.add_argument('--out', default='figures'); a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True); jobs = load(a.runs)
    print(f'{len(jobs)} finished jobs')
    fig_architecture(out); fig_confusion(jobs, out); fig_robustness(jobs, out); fig_sensor_sets(jobs, out)
    fig_weights(jobs, out); fig_reliability(jobs, out); fig_risk_coverage(jobs, out); fig_quality_roc(jobs, out)
    fig_training(jobs, out)
    if a.data: fig_signals(a.data, out)


if __name__ == '__main__':
    main()
