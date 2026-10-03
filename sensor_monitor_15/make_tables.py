"""Build manuscript Tables V-X (mean ± SD over runs) from the finished study jobs.

Usage: python make_tables.py --runs study_runs --out tables
Only finished jobs (metrics.json present) are read. Nothing is typed in by hand.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

NAMES = {'relisense': 'ReliSense (compact)', 'relisense_uniform': 'Uniform sensor weights',
         'relisense_norec': 'No reconstruction objective', 'relisense_noq': 'No quality supervision',
         'transformer': 'Temporal Transformer', 'wdcnn': 'Compact CNN (WDCNN)', 'tcn': 'TCN',
         'envelope_rf': 'Envelope descriptors + random forest', 'relisense+phys': 'ReliSense + physics loss',
         'relisense+moment': 'ReliSense + frozen MOMENT', 'relisense_p': 'ReliSense-P (physics + reliability gate)',
         'physics_lr': 'Kinematic physics features + logistic regression', 'relisense_wn': 'ReliSense, window-normalised'}
SETS = [('clean', 'All three channels'), ('set_currents_only', 'Two current channels only'),
        ('set_vib_cur1', 'Vibration and one current phase'), ('set_vib_only', 'Vibration only')]
FAMILIES = [('Noise severity sweep', lambda s: s['kind'] == 'noise'),
            ('Bias and gain sweep', lambda s: s['kind'] in ('bias', 'gain')),
            ('Short-record drift', lambda s: s['kind'] == 'drift'),
            ('Clipping', lambda s: s['kind'] == 'clip'),
            ('Single-channel loss', lambda s: s['kind'] == 'missing'),
            ('Test-only family (spikes, quantisation)', lambda s: s['kind'] in ('spike', 'quant'))]
CORRUPTED = lambda s: s['kind'] not in ('clean', 'set')


def load(runs):
    jobs = []
    for f in sorted(Path(runs).glob('*/metrics.json')):
        m = json.loads(f.read_text()); m['sc'] = {s['scenario']: s for s in m['scenarios']}; jobs.append(m)
    return jobs


def ms(values, pct=True, nd=1):
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) == 0: return 'n/a'
    k = 100 if pct else 1
    return f'{k * v.mean():.{nd}f} ± {k * v.std(ddof=1) if len(v) > 1 else 0:.{nd}f}'


def pick(jobs, model, joint=False):
    return [j for j in jobs if j['model'] == model and (j['heldout'] is not None) == joint]


def mean_over(job, pred, key):
    v = [s[key] for s in job['scenarios'] if pred(s) and s.get(key) is not None]
    return float(np.mean(v)) if v else None


def md(title, header, rows, note=''):
    out = [f'### {title}', '', '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]
    return '\n'.join(out + ([''] + [note] if note else []) + [''])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', default='study_runs'); ap.add_argument('--out', default='tables')
    a = ap.parse_args(); jobs = load(a.runs); out = Path(a.out); out.mkdir(exist_ok=True)
    models = [m for m in NAMES if pick(jobs, m) or pick(jobs, m, True)]
    doc = [f'# ReliSense result tables\n\nGenerated from {len(jobs)} finished jobs in `{a.runs}`. '
           'Values are mean ± SD over runs (bearing rotations × seeds), recording level, in %. '
           'n = number of runs.\n']
    T = {}

    # Table V main comparison
    rows = []
    for m in ['envelope_rf', 'wdcnn', 'tcn', 'transformer', 'physics_lr', 'relisense', 'relisense_p', 'relisense+moment', 'relisense+phys']:
        b, jt = pick(jobs, m), pick(jobs, m, True)
        if not b and not jt: continue
        rows.append([NAMES[m], ms([j['sc']['clean']['macro_f1'] for j in b]), ms([j['sc']['clean']['accuracy'] for j in b]),
                     ms([j['sc']['clean']['balanced_accuracy'] for j in b]),
                     ms([mean_over(j, CORRUPTED, 'macro_f1') for j in b]),
                     ms([j['sc']['clean']['macro_f1'] for j in jt]) if jt else 'not run', f'{len(b)}/{len(jt)}'])
    T['V'] = rows
    doc.append(md('Table V. Main diagnostic comparison',
                  ['Method', 'Bearing holdout macro F1', 'Accuracy', 'Balanced accuracy',
                   'Mean macro F1 over corrupted tests', 'Joint holdout macro F1', 'n (bearing/joint)'], rows))

    # paired tests against ReliSense (same rotation and seed)
    REF = 'relisense_p' if pick(jobs, 'relisense_p') else 'relisense'
    ref = {(j['rotation'], j['seed']): j for j in pick(jobs, REF)}
    rows = []
    for m in [x for x in models if x != REF]:
        pairs = [(ref[(j['rotation'], j['seed'])], j) for j in pick(jobs, m) if (j['rotation'], j['seed']) in ref]
        if len(pairs) < 2: continue
        for label, f in [('clean', lambda j: j['sc']['clean']['macro_f1']), ('corrupted', lambda j: mean_over(j, CORRUPTED, 'macro_f1'))]:
            d = np.array([f(r) - f(o) for r, o in pairs])
            p = wilcoxon(d).pvalue if np.any(d != 0) and len(d) >= 5 else float('nan')
            rows.append([NAMES[m], label, f'{100 * d.mean():+.1f}', f'{(d > 0).sum()}/{(d < 0).sum()}/{(d == 0).sum()}',
                         f'{p:.3g}', len(d)])
    T['paired'] = rows
    doc.append(md(f'Paired difference, {NAMES[REF]} minus comparator (macro F1, points)',
                  ['Comparator', 'Test', 'Mean difference', 'ReliSense better/worse/equal', 'Wilcoxon p', 'pairs'], rows))

    # Table VI available measurement set
    rows = []
    for m in [x for x in ['relisense_p', 'relisense', 'physics_lr', 'transformer', 'wdcnn', 'tcn', 'envelope_rf'] if pick(jobs, x)]:
        for key, label in SETS:
            b = pick(jobs, m)
            rows.append([NAMES[m], label, ms([j['sc'][key]['macro_f1'] for j in b]),
                         ms([j['sc'][key]['accepted_error'] for j in b]), ms([j['sc'][key]['accepted_fraction'] for j in b])])
    T['VI'] = rows
    doc.append(md('Table VI. Available measurement set (no retraining; missing channels masked)',
                  ['Model', 'Measurement set', 'Macro F1', 'Accepted-case error', 'Acceptance fraction'], rows,
                  'No channels: abstain by definition (the model raises an error instead of classifying).'))

    # Table VII corruption outcome (ReliSense quality head) + baselines macro F1
    rows = []
    for fam, pred in FAMILIES:
        row = [fam]
        for m in [x for x in ['relisense_p', 'relisense', 'physics_lr', 'transformer', 'wdcnn', 'tcn', 'envelope_rf'] if pick(jobs, x)]:
            row.append(ms([mean_over(j, pred, 'macro_f1') for j in pick(jobs, m)]))
        b = pick(jobs, 'relisense_p') or pick(jobs, 'relisense')
        row += [ms([mean_over(j, pred, 'sensor_isolation_auroc') for j in b], nd=1),
                ms([mean_over(j, pred, 'quality_detection') for j in b]), ms([mean_over(j, pred, 'quality_false_alarm') for j in b])]
        rows.append(row)
    b = pick(jobs, 'relisense')
    rows.append(['Clean (false alarms only)'] + ['' for x in ['relisense_p', 'relisense', 'physics_lr', 'transformer', 'wdcnn', 'tcn', 'envelope_rf'] if pick(jobs, x)]
                + ['n/a', 'n/a', ms([j['sc']['clean'].get('quality_false_alarm') for j in b])])
    hdr = ['Corruption family'] + [f'Macro F1 {NAMES[x]}' for x in ['relisense_p', 'relisense', 'physics_lr', 'transformer', 'wdcnn', 'tcn', 'envelope_rf'] if pick(jobs, x)] \
          + ['Sensor isolation AUROC (ReliSense)', 'Detection rate', 'False-alarm rate']
    T['VII'] = rows
    doc.append(md('Table VII. Measurement corruption outcome (mean over sensors and severities in each family)', hdr, rows,
                  'Isolation AUROC uses the unnormalised quality output (1 − sigmoid). Detection and false alarm use the 0.5 threshold.'))

    # Table VIII ablation (paired to the ablation rotations/seeds)
    abl = [j for j in jobs if j['model'] in ('relisense_uniform', 'relisense_norec', 'relisense_noq')]
    keys = {(j['rotation'], j['seed']) for j in abl}
    rows = []
    for m in ['relisense', 'relisense_uniform', 'relisense_norec', 'relisense_noq']:
        b = [j for j in pick(jobs, m) if (j['rotation'], j['seed']) in keys]
        jt = [j for j in pick(jobs, m, True) if (j['rotation'], j['seed']) in keys]
        if not b: continue
        rows.append([NAMES[m], ms([j['sc']['clean']['macro_f1'] for j in b]), ms([mean_over(j, CORRUPTED, 'macro_f1') for j in b]),
                     ms([mean_over(j, lambda s: s['kind'] == 'missing', 'macro_f1') for j in b]),
                     ms([j['sc']['clean']['macro_f1'] for j in jt]) if jt else 'not run', len(b)])
    T['VIII'] = rows
    doc.append(md('Table VIII. Ablation (same rotations and seeds for every row)',
                  ['Configuration', 'Clean macro F1', 'Corrupted macro F1', 'Single-channel-loss macro F1', 'Joint holdout macro F1', 'n'], rows))

    # Table IX reliability (ReliSense)
    rows = []
    for m in [x for x in ['relisense_p', 'relisense', 'physics_lr', 'transformer', 'wdcnn', 'tcn', 'envelope_rf'] if pick(jobs, x)]:
        b, jt = pick(jobs, m), pick(jobs, m, True)
        for label, getter, nd in [('NLL', 'nll', 3), ('Brier', 'brier', 3), ('ECE (10 bins) %', 'ece', 1),
                                  ('Coverage of 90% sets %', 'coverage', 1), ('Mean set size', 'set_size', 2),
                                  ('Risk at 80% acceptance %', 'risk_at_80', 1), ('Risk at 90% acceptance %', 'risk_at_90', 1),
                                  ('AURC %', 'aurc', 1)]:
            pct = getter in ('ece', 'coverage', 'risk_at_80', 'risk_at_90', 'aurc')
            rows.append([NAMES[m], label, ms([j['sc']['clean'][getter] for j in b], pct, nd),
                         ms([mean_over(j, CORRUPTED, getter) for j in b], pct, nd),
                         ms([j['sc']['clean'][getter] for j in jt], pct, nd) if jt else 'not run'])
        rec = [[j['sc']['clean']['accepted_recall'][c] for j in b] for c in range(3)]
        rows.append([NAMES[m], 'Accepted recall H/OR/IR %', ' / '.join(ms(r) for r in rec), '', ''])
    T['IX'] = rows
    doc.append(md('Table IX. Reliability outcome', ['Model', 'Measure', 'Clean test', 'Corrupted test (mean)', 'Joint condition shift'], rows))

    # Table X computation
    rows = []
    for m in models:
        b = pick(jobs, m) or pick(jobs, m, True)
        c = [j['computation'] for j in b]; p = c[0]['params']
        tr = p.get('trainable')
        rows.append([NAMES[m], (f"{tr:,} ({tr - p.get('decoder', 0):,} used at inference)" if p.get('decoder') else f'{tr:,}')
                     if tr is not None else p.get('note', ''), p.get('frozen') if p.get('frozen') is not None else '',
                     ms([x['latency_ms_per_recording'] for x in c], False, 1), ms([x.get('peak_gpu_mb') for x in c], False, 0),
                     f"{c[0]['mc_samples']} / {c[0]['windows_per_recording']}", ms([x['train_seconds'] / 60 for x in c], False, 1),
                     c[0]['device']])
    T['X'] = rows
    doc.append(md('Table X. Computational outcome',
                  ['Model', 'Trainable parameters', 'Frozen parameters', 'Latency per recording (ms)', 'Peak GPU memory (MB)',
                   'MC samples / windows', 'Training time per run (min)', 'Device'], rows,
                  'Latency includes all windows of one recording, MC-dropout samples and softmax averaging; warmed runs, batch = one recording.'))

    (out / 'tables.md').write_text('\n'.join(doc)); (out / 'tables.json').write_text(json.dumps(T, indent=2))
    # long per-run, per-scenario table for the supplement
    import csv
    with open(out / 'all_scenarios_per_run.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['model', 'rotation', 'seed', 'heldout', 'scenario', 'accuracy', 'macro_f1', 'balanced_accuracy',
                                       'ece', 'nll', 'brier', 'coverage', 'set_size', 'accepted_fraction', 'accepted_error',
                                       'sensor_isolation_auroc', 'quality_false_alarm', 'test_bearings'])
        for j in jobs:
            for s in j['scenarios']:
                w.writerow([j['model'], j['rotation'], j['seed'], j['heldout'], s['scenario'], s['accuracy'], s['macro_f1'],
                            s['balanced_accuracy'], s['ece'], s['nll'], s['brier'], s['coverage'], s['set_size'],
                            s['accepted_fraction'], s['accepted_error'], s.get('sensor_isolation_auroc'),
                            s.get('quality_false_alarm'), ' '.join(j['split']['test']['bearings'])])
    print('\n'.join(doc))


if __name__ == '__main__':
    main()
