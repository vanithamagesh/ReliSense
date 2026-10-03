"""Phase 1 (v1.1): build the 32-bearing Paderborn data, one bearing at a time.

For every bearing in labels_32.csv (include=1):
  1. download the official archive <bearing>.rar (skipped if it is already there);
  2. read every .mat recording of the four operating conditions straight from the archive;
  3. save, per recording, the 8 kHz windows (vibration, current 1, current 2, plus the 2-12 kHz envelope
     of the vibration) and the 64 kHz physics outputs of physics64.py;
  4. write <out>/<bearing>.npz and <out>/<bearing>.json, then delete the archive (unless --keep-archives).
A recording that cannot be read is skipped and listed in <bearing>.json (the bearing fails if more than 10 %
are unreadable). A bearing whose .npz already exists is skipped, so the script can be re-run after a disconnect.
Then run merge_32.py to join the bearings.

Data: Lessmeier et al., KAt-DataCenter, Paderborn University, CC BY-NC 4.0.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt
import physics64 as P

BASE = 'https://groups.uni-paderborn.de/kat/BearingDataCenter/'
CONDITIONS = ['N15_M07_F04', 'N15_M01_F10', 'N15_M07_F10', 'N09_M07_F10']
CHANNELS = ['vibration_1', 'phase_current_1', 'phase_current_2']
FS = 64000


def read_labels(path):
    with open(path, newline='') as f:
        return [r for r in csv.DictReader(f) if r.get('include', '1') == '1']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def find_channels(obj, found):
    """Collect every {Name, Data} pair in the nested MATLAB struct (same rule as relisense.prepare)."""
    if isinstance(obj, dict):
        lower = {k.lower(): k for k in obj}
        if 'name' in lower and 'data' in lower:
            arr = np.asarray(obj[lower['data']]).squeeze()
            if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                found[str(obj[lower['name']]).strip().lower()] = arr
        for v in obj.values():
            find_channels(v, found)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            find_channels(v, found)
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        for v in obj.flat:
            find_channels(v, found)


def download(bearing, archives):
    archive = Path(archives) / f'{bearing}.rar'
    if not archive.exists():
        part = archive.with_suffix('.rar.part')
        print(f'{bearing}: downloading', flush=True)
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--retry', '3',
                        '--connect-timeout', '60', '--max-time', '3600', BASE + archive.name,
                        '--output', str(part)], check=True)
        part.replace(archive)
    return archive


def process_recording(raw, name, bearing, args, sos):
    found = {}
    find_channels(loadmat(io.BytesIO(raw), simplify_cells=True), found)
    missing = [c for c in CHANNELS if c not in found]
    if missing:
        raise ValueError(f'{name}: missing {missing}; found {sorted(found)}')
    sig = [np.asarray(found[c], dtype=float) for c in CHANNELS]
    if len({len(s) for s in sig}) != 1 or not all(np.isfinite(s).all() for s in sig):
        raise ValueError(f'{name}: unequal lengths or non-finite values')
    cond = name.rsplit('_', 2)[0]
    rpm_set = int(cond.split('_')[0][1:]) * 100.0
    # Measured speed if the speed channel is plausible (within 20 % of the set point), else the set point.
    rpm, src = rpm_set, 'setpoint'
    if 'speed' in found and np.isfinite(found['speed']).all():
        m = float(np.median(np.abs(found['speed'])))
        if abs(m - rpm_set) <= 0.2 * rpm_set:
            rpm, src = m, 'measured'
    env = np.abs(hilbert(sosfiltfilt(sos, sig[0])))
    arr = np.stack([resample_poly(s, 1, args.downsample) for s in sig + [env]]).astype('float32')
    L = args.length
    starts = range(0, arr.shape[1] - L + 1, L)
    win = np.stack([arr[:, s:s + L] for s in starts]) if len(starts) else np.zeros((0, 4, L), 'float32')
    phys = P.recording_physics(sig[0], FS, rpm, fmax=args.fmax)
    return {'stem': Path(name).stem, 'condition': cond, 'rpm_set': rpm_set, 'rpm': rpm, 'rpm_source': src,
            'samples': len(sig[0]), 'windows': win, 'phys': phys, 'stats': P.stats(sig[0])}


def build_bearing(row, args):
    bearing = row['bearing']; out = Path(args.out)
    target = out / f'{bearing}.npz'
    if target.exists():
        print(f'{bearing}: already built, skipped', flush=True)
        return bearing, 'skipped'
    t0 = time.time()
    archive = download(bearing, args.archives)
    asha = sha256(archive); asize = archive.stat().st_size
    sos = butter(4, [2000, 12000], btype='bandpass', fs=FS, output='sos')
    pattern = re.compile(rf'({"|".join(CONDITIONS)})_{bearing}_(\d+)\.mat$', re.I)
    import libarchive
    recs, skipped = [], []
    with libarchive.file_reader(str(archive)) as entries:
        for entry in entries:
            name = Path(entry.pathname.replace('\\', '/')).name
            m = pattern.search(name)
            if not m or (args.records_per_condition and int(m.group(2)) > args.records_per_condition):
                continue
            raw = b''.join(entry.get_blocks())
            if len(raw) != entry.size:
                raise ValueError(f'{bearing}: incomplete read of {name}')
            try:
                recs.append(process_recording(raw, name, bearing, args, sos))
            except Exception as exc:  # unreadable file in the official archive: skip it and record why
                skipped.append({'file': name, 'error': f'{type(exc).__name__}: {exc}'})
                print(f'{bearing}: SKIPPED unreadable recording {name} ({exc})', flush=True)
    recs.sort(key=lambda r: (CONDITIONS.index(r['condition']), int(r['stem'].rsplit('_', 1)[1])))
    counts = {c: sum(r['condition'] == c for r in recs) for c in CONDITIONS}
    if not recs or min(counts.values()) == 0:
        raise ValueError(f'{bearing}: recordings per condition {counts}; skipped {skipped}')
    if len(skipped) > 0.1 * (len(recs) + len(skipped)):
        raise ValueError(f'{bearing}: more than 10 % of recordings unreadable: {skipped}')
    W = np.concatenate([r['windows'] for r in recs])
    wrec = np.concatenate([[r['stem']] * len(r['windows']) for r in recs])
    save = {'windows': W, 'window_recording': wrec,
            'recording': np.array([r['stem'] for r in recs]),
            'condition': np.array([r['condition'] for r in recs]),
            'rpm_set': np.array([r['rpm_set'] for r in recs], 'float32'),
            'rpm': np.array([r['rpm'] for r in recs], 'float32'),
            'rpm_source': np.array([r['rpm_source'] for r in recs]),
            'samples': np.array([r['samples'] for r in recs]),
            'order_grid': P.ORDER_GRID, 'feature_names': np.array(P.FEATURE_NAMES),
            'window_channels': np.array(CHANNELS + ['vibration_1_envelope_2_12kHz'])}
    for k in ('kurtosis', 'crest', 'rms'):
        save[f'stat_{k}'] = np.array([r['stats'][k] for r in recs], 'float32')
    for v in P.VARIANTS:
        save[f'feat_{v}'] = np.stack([r['phys']['feat'][v] for r in recs])
        save[f'order_{v}'] = np.stack([r['phys']['order'][v] for r in recs]).astype('float16')
        save[f'band_{v}'] = np.array([r['phys']['band'][v] for r in recs], 'float32')
        save[f'kurt_{v}'] = np.array([r['phys']['kurt'][v] for r in recs], 'float32')
    part = out / f'{bearing}.part.npz'
    np.savez_compressed(part, **save)
    part.replace(target)
    meta = {'bearing': bearing, 'label': int(row['label']), 'origin': row.get('origin', ''),
            'location': row.get('location', ''), 'archive_url': BASE + archive.name,
            'archive_bytes': asize, 'archive_sha256': asha, 'recordings_per_condition': counts,
            'skipped_recordings': skipped,
            'windows': int(len(W)), 'rpm_measured_share': float(np.mean(save['rpm_source'] == 'measured')),
            'seconds': round(time.time() - t0, 1)}
    (out / f'{bearing}.json').write_text(json.dumps(meta, indent=2))
    if not args.keep_archives:
        archive.unlink()
    print(f'{bearing}: {len(recs)} recordings {counts}, {len(skipped)} skipped, {len(W)} windows, {meta["seconds"]} s', flush=True)
    return bearing, 'built'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', default='labels_32.csv')
    ap.add_argument('--out', default='pb32_bearings')
    ap.add_argument('--archives', default='paderborn_archives')
    ap.add_argument('--bearings', default='', help='comma list; empty = all included bearings')
    ap.add_argument('--records-per-condition', type=int, default=0, help='0 = all (20)')
    ap.add_argument('--length', type=int, default=4096)
    ap.add_argument('--downsample', type=int, default=8)
    ap.add_argument('--fmax', type=float, default=12000.0, help='upper limit of the kurtogram search (Hz)')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--keep-archives', action='store_true')
    args = ap.parse_args()
    rows = read_labels(args.labels)
    if args.bearings:
        want = set(args.bearings.split(',')); rows = [r for r in rows if r['bearing'] in want]
    Path(args.out).mkdir(parents=True, exist_ok=True); Path(args.archives).mkdir(parents=True, exist_ok=True)
    errors = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(build_bearing, r, args): r['bearing'] for r in rows}
        for job in as_completed(jobs):
            try:
                job.result()
            except Exception as exc:
                errors[jobs[job]] = str(exc); print(f'FAILED {jobs[job]}: {exc}', file=sys.stderr, flush=True)
    if errors:
        (Path(args.out) / 'errors.json').write_text(json.dumps(errors, indent=2))
        raise SystemExit(f'{len(errors)} bearings failed (see errors.json). Re-run to retry only those.')
    print('All bearings built in ' + args.out, flush=True)


if __name__ == '__main__':
    main()
