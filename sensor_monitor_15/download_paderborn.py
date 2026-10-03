"""Download official Paderborn archives and extract measured MATLAB recordings.

Data: Christian Lessmeier et al., Paderborn University KAt-DataCenter,
CC BY-NC 4.0. Only noncommercial use is covered by the data licence.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import libarchive

BASE = 'https://groups.uni-paderborn.de/kat/BearingDataCenter/'
GROUPS = {
    0: ['K001', 'K002', 'K003', 'K004', 'K005'],
    1: ['KA01', 'KA04', 'KA05', 'KA07', 'KA15'],
    2: ['KI01', 'KI04', 'KI05', 'KI07', 'KI14'],
}
CONDITIONS = ['N15_M07_F04', 'N15_M01_F10', 'N15_M07_F10', 'N09_M07_F10']
REFERENCE = ('Lessmeier, C., Kimotho, J. K., Zimmer, D., & Sextro, W. (2016). '
    'Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by '
    'Using Motor Current Signals of Electric Motors: A Benchmark Data Set for '
    'Data-Driven Classification. European Conference of the PHM Society.')
PAPER = 'https://mb.uni-paderborn.de/fileadmin-mb/kat/PDF/Veroeffentlichungen/20160703_PHME16_CM_bearing.pdf'

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def acquire(bearing, label, args):
    dest = Path(args.raw) / bearing
    dest.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archives) / f'{bearing}.rar'
    url = BASE + archive.name
    if not archive.exists():
        temporary = archive.with_suffix('.rar.part')
        print(f'Downloading {bearing} from official source', flush=True)
        # A .part file is never reused as a completed archive.
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
            '--retry', '3', '--connect-timeout', '60', '--max-time', '1800',
            url, '--output', str(temporary)], check=True)
        temporary.replace(archive)
    # Index the archive before selecting records; selection uses sorted original names.
    names = []
    with libarchive.file_reader(str(archive)) as entries:
        for entry in entries:
            if entry.pathname.lower().endswith('.mat'):
                names.append(Path(entry.pathname.replace('\\', '/')).name)
    chosen = set()
    for condition in CONDITIONS:
        candidates = sorted(name for name in names
            if re.fullmatch(rf'{condition}_{bearing}_\d+\.mat', name))
        if len(candidates) != 20:
            raise ValueError(f'{bearing}: expected 20 recordings for {condition}, found {len(candidates)}')
        # Numeric recording order avoids lexical 1,10,11,... ordering.
        candidates.sort(key=lambda name: int(Path(name).stem.rsplit('_', 1)[1]))
        chosen.update(candidates if args.records_per_condition == 0
            else candidates[:args.records_per_condition])
    rows = []
    with libarchive.file_reader(str(archive)) as entries:
        for entry in entries:
            name = Path(entry.pathname.replace('\\', '/')).name
            if name not in chosen:
                continue
            target = dest / name
            temporary = target.with_suffix('.mat.part')
            with temporary.open('wb') as f:
                for block in entry.get_blocks():
                    f.write(block)
            if temporary.stat().st_size != entry.size:
                raise ValueError(f'Incomplete extraction: {name}')
            temporary.replace(target)
            rows.append({'bearing': bearing, 'label': label,
                'condition': name.rsplit('_', 2)[0], 'recording': target.stem,
                'file_name': name, 'relative_path': f'{bearing}/{name}',
                'raw_bytes': target.stat().st_size, 'raw_sha256': sha256(target),
                'archive_url': url})
    if len(rows) != len(chosen):
        raise ValueError(f'{bearing}: missing selected recordings')
    result = {'bearing': bearing, 'label': label, 'archive_url': url,
        'archive_bytes': archive.stat().st_size, 'archive_sha256': sha256(archive),
        'recordings': sorted(rows, key=lambda row: row['recording'])}
    (dest / 'source_manifest.json').write_text(json.dumps(result, indent=2))
    if args.remove_archives:
        archive.unlink()
    print(f'{bearing}: extracted {len(rows)} real recordings', flush=True)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', default='raw_paderborn')
    parser.add_argument('--archives', default='paderborn_archives')
    parser.add_argument('--records-per-condition', type=int, default=2,
        help='1..20 per condition; 0 extracts all 20. Always uses all four conditions.')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--remove-archives', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.records_per_condition <= 20 or args.workers < 1:
        parser.error('records-per-condition must be 0..20; workers must be positive')
    if shutil.which('curl') is None:
        parser.error('curl is required (available on Colab/Linux; install it locally first).')
    Path(args.raw).mkdir(parents=True, exist_ok=True)
    Path(args.archives).mkdir(parents=True, exist_ok=True)
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(acquire, bearing, label, args): bearing
            for label, bearings in GROUPS.items() for bearing in bearings}
        for future in as_completed(tasks):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({'bearing': tasks[future], 'error': str(exc)})
                print(f'FAILED {tasks[future]}: {exc}', file=sys.stderr, flush=True)
    manifest = {'dataset': 'Paderborn University KAt measured bearing data',
        'source_page': 'https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter',
        'license': 'CC BY-NC 4.0', 'license_url': 'https://creativecommons.org/licenses/by-nc/4.0/',
        'attribution': 'Christian Lessmeier et al.; Chair of Design and Drive Technology, Paderborn University',
        'reference': REFERENCE, 'label_reference': PAPER,
        'label_tables': 'Healthy: Table 6; outer/inner damage: Tables 4 and 5.',
        'selection': 'First numbered recordings per operating condition; no performance-based selection.',
        'records_per_condition': args.records_per_condition or 20,
        'bearings': sorted(results, key=lambda row: row['bearing']), 'errors': errors}
    (Path(args.raw) / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    if errors:
        raise SystemExit('Some downloads failed. Correct the errors and rerun before training.')
    print('Completed: real data available in ' + args.raw, flush=True)

if __name__ == '__main__':
    main()
