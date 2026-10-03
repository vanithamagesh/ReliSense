"""Phase 1 (v1.0): join the per-bearing files of build_32.py.

Writes to --out:
  pb32_physics.npz          one row per recording: labels, origin, speed, the 4 x 15 physics features,
                            the 4 order spectra, kurtogram bands and signal statistics (all bearings, KB included)
  pb32_4096.npz             windows in the study.py format (vibration, current 1, current 2)
  pb32_4096_envelope.npz    same, with the vibration replaced by its 2-12 kHz envelope
The window files keep only labels 0-2 (healthy, outer, inner); combined-damage bearings (label 3) are
left out of them because study.py is a 3-class study. They stay in pb32_physics.npz.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from build_32 import read_labels, CHANNELS
import physics64 as P


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', default='labels_32.csv')
    ap.add_argument('--bearings-dir', default='pb32_bearings')
    ap.add_argument('--out', default='.')
    ap.add_argument('--no-windows', action='store_true', help='write only pb32_physics.npz')
    args = ap.parse_args()
    rows = read_labels(args.labels); src = Path(args.bearings_dir); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    missing = [r['bearing'] for r in rows if not (src / f"{r['bearing']}.npz").exists()]
    if missing:
        raise SystemExit(f'Not built yet: {missing}. Run build_32.py first.')
    phys = {}; win = {'X': [], 'E': [], 'y': [], 'bearing': [], 'condition': [], 'recording': [], 'rpm': []}
    summary = []
    for r in rows:
        b = r['bearing']; lab = int(r['label']); d = np.load(src / f'{b}.npz')
        n = len(d['recording'])
        cols = {'recording': d['recording'], 'bearing': np.array([b] * n), 'label': np.full(n, lab),
                'origin': np.array([r.get('origin', '')] * n), 'condition': d['condition'],
                'rpm_set': d['rpm_set'], 'rpm': d['rpm'], 'rpm_source': d['rpm_source']}
        for k in d.files:
            if k.startswith(('feat_', 'order_', 'band_', 'kurt_', 'stat_')):
                cols[k] = d[k]
        for k, v in cols.items():
            phys.setdefault(k, []).append(v)
        summary.append({'bearing': b, 'label': lab, 'origin': r.get('origin', ''), 'recordings': n,
                        'windows': int(len(d['windows']))})
        if args.no_windows or lab > 2:
            continue
        W = d['windows']; wr = d['window_recording']
        pos = {s: i for i, s in enumerate(d['recording'])}; idx = np.array([pos[s] for s in wr])
        win['X'].append(W[:, :3]); win['E'].append(np.concatenate([W[:, 3:4], W[:, 1:3]], 1))
        win['y'].append(np.full(len(W), lab)); win['bearing'].append(np.array([b] * len(W)))
        win['condition'].append(d['condition'][idx]); win['recording'].append(wr)
        win['rpm'].append(d['rpm_set'][idx])
    phys = {k: np.concatenate(v) for k, v in phys.items()}
    phys['order_grid'] = P.ORDER_GRID; phys['feature_names'] = np.array(P.FEATURE_NAMES)
    np.savez_compressed(out / 'pb32_physics.npz', **phys)
    print(f'pb32_physics.npz: {len(phys["recording"])} recordings, {len(rows)} bearings')
    if not args.no_windows:
        common = {k: np.concatenate(win[k]) for k in ('y', 'bearing', 'condition', 'recording', 'rpm')}
        common['rpm'] = common['rpm'].astype('float32'); common['fs'] = np.array(8000)
        for key, name, mode in (('X', 'pb32_4096.npz', 'raw'), ('E', 'pb32_4096_envelope.npz', 'envelope')):
            np.savez_compressed(out / name, X=np.concatenate(win[key]), **common,
                                channel_names=np.array(CHANNELS), signal_mode=np.array(mode),
                                provenance=np.array('PADERBORN REAL RECORDINGS; 32-BEARING BUILD v1.0; USER AUDITED LABELS'))
            print(f'{name}: {len(common["y"])} windows')
    (out / 'pb32_summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
