import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("""# ReliSense Phase 1, v1.0: 32-bearing Paderborn build
Runs on a **CPU** runtime (no GPU needed). It downloads the 32 bearings one at a time (about 5–6 GB in total, never all on disk together), saves one small file per bearing to Drive, and deletes each archive.
If Colab disconnects, run the cells again: finished bearings are skipped.

Before Cell 1: put `physics64.py`, `build_32.py`, `merge_32.py`, `check_labels.py` and `labels_32.csv` in `MyDrive/ReliSense_study/phase1/`.""")
code("""# Cell 1: setup
from google.colab import drive; drive.mount('/content/drive')
import os, shutil, subprocess, sys
from pathlib import Path
DRIVE = Path('/content/drive/MyDrive/ReliSense_study'); SRC = DRIVE/'phase1'; OUT = DRIVE/'pb32'
need = ['physics64.py','build_32.py','merge_32.py','check_labels.py','labels_32.csv']
miss = [f for f in need if not (SRC/f).exists()]
assert not miss, f'Missing in {SRC}: {miss}'
WORK = Path('/content/p1'); WORK.mkdir(exist_ok=True)
for f in need: shutil.copy(SRC/f, WORK/f)
os.chdir(WORK); OUT.mkdir(exist_ok=True)
subprocess.run([sys.executable,'-m','pip','install','-q','libarchive-c'],check=True)
print('ready; outputs go to', OUT)""")
code("""# Cell 2: build all 32 bearings (about 30-60 min, mostly download). The log goes to Drive, not the page.
LOG = OUT/'build_log.txt'
with open(LOG,'a') as log:
    r = subprocess.run([sys.executable,'build_32.py','--labels','labels_32.csv','--out',str(OUT/'bearings'),
                        '--archives','/content/archives','--workers','3'], stdout=log, stderr=subprocess.STDOUT)
print('exit code', r.returncode); print(''.join(open(LOG).readlines()[-40:]))
built = sorted(p.stem for p in (OUT/'bearings').glob('K*.npz') if not p.stem.endswith('.part'))
print(len(built), 'of 32 built:', built)""")
code("""# Cell 3: merge into the study files
r = subprocess.run([sys.executable,'merge_32.py','--labels','labels_32.csv','--bearings-dir',str(OUT/'bearings'),
                    '--out',str(OUT)], capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:])
for f in sorted(OUT.glob('pb32_*')): print(f.name, round(f.stat().st_size/1e6,1),'MB')""")
code("""# Cell 4: label check. Copy the full output back to Claude.
for v in ['fixed','sk','cpw','cpw_sk']:
    r = subprocess.run([sys.executable,'check_labels.py','--physics',str(OUT/'pb32_physics.npz'),'--variant',v],
                       capture_output=True, text=True); print(r.stdout, r.stderr)
import json; print(json.dumps([json.load(open(p)) | {'archive_sha256':'..'} for p in sorted((OUT/'bearings').glob('K*.json'))][:3], indent=1))""")
nb['cells']=C; nbf.write(nb,'ReliSense_Phase1_build32_v1_0.ipynb'); print('ok')
