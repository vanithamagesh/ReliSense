import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("""# ReliSense, 32 bearings: all steps (v1.1)
Runtime: **CPU** is enough. Run the cells in order. Every result is saved to `MyDrive/ReliSense_study/pb32/`.
If Colab disconnects: run Step 1 again, then continue where you stopped. Work that is already finished is skipped.""")
code("""# STEP 1: setup (always run this first)
from google.colab import drive; drive.mount('/content/drive')
import os, shutil, subprocess, sys
from pathlib import Path
DRIVE = Path('/content/drive/MyDrive/ReliSense_study'); SRC = DRIVE/'phase1'; OUT = DRIVE/'pb32'
need = ['physics64.py','build_32.py','merge_32.py','check_labels.py','labels_32.csv','phase2_physics.py']
miss = [f for f in need if not (SRC/f).exists()]
assert not miss, f'Missing in {SRC}: {miss}'
WORK = Path('/content/p1'); WORK.mkdir(exist_ok=True)
for f in need: shutil.copy(SRC/f, WORK/f)
os.chdir(WORK); (OUT/'bearings').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable,'-m','pip','install','-q','libarchive-c'], check=True)
def count(): return sorted(p.name[:-4] for p in (OUT/'bearings').glob('K*.npz') if '.part' not in p.name)
print('STEP 1 done.', len(count()), 'of 32 bearings already built')""")
code("""# STEP 2: download + build the bearings (30-60 min). Progress: pb32/build_log_<GROUP>.txt on Drive.
# One Colab: keep GROUP = 'ALL'.  Faster: open 3 Colabs, run STEP 1 in each, then GROUP 'A', 'B', 'C'.
GROUP = 'ALL'
GROUPS = {'ALL': '',
 'A': 'K001,K002,K003,K004,K005,K006,KA01,KA03,KA05,KA06,KA07',
 'B': 'KA08,KA09,KI01,KI03,KI05,KI07,KI08,KA04,KA15,KA16,KA22',
 'C': 'KA30,KI04,KI14,KI16,KI17,KI18,KI21,KB23,KB24,KB27'}
LOG = OUT/f'build_log_{GROUP}.txt'
cmd = [sys.executable,'build_32.py','--labels','labels_32.csv','--out',str(OUT/'bearings'),
       '--archives','/content/archives','--workers','4']
if GROUPS[GROUP]: cmd += ['--bearings', GROUPS[GROUP]]
with open(LOG,'a') as log: r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
print('exit code', r.returncode, '(0 = fine)'); print(''.join(open(LOG).readlines()[-15:]))
print('STEP 2:', len(count()), 'of 32 built. If less than 32: run STEP 2 again.')""")
code("""# STEP 3: check that all 32 are there (run any time)
b = count(); print(len(b), 'of 32 built'); import csv; allb = [x['bearing'] for x in csv.DictReader(open('labels_32.csv'))]
print('missing:', [x for x in allb if x not in b] or 'none')""")
code("""# STEP 4: merge the 32 bearings (a few minutes). Only when STEP 3 says missing: none.
r = subprocess.run([sys.executable,'merge_32.py','--labels','labels_32.csv','--bearings-dir',str(OUT/'bearings'),
                    '--out',str(OUT)], capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:])
for f in sorted(OUT.glob('pb32_*')): print(f.name, round(f.stat().st_size/1e6,1), 'MB')""")
code("""# STEP 5: label check -> COPY THIS OUTPUT TO CLAUDE
assert (OUT/'pb32_physics.npz').exists(), 'Run STEP 4 first.'
for v in ['fixed','sk','cpw','cpw_sk']:
    r = subprocess.run([sys.executable,'check_labels.py','--physics',str(OUT/'pb32_physics.npz'),'--variant',v],
                       capture_output=True, text=True); print(r.stdout, r.stderr[-1000:])""")
code("""# STEP 6: Phase 2 results (bearing-wise protocols, ~5 min) -> COPY THIS OUTPUT TO CLAUDE
assert (OUT/'pb32_physics.npz').exists(), 'Run STEP 4 first.'
r = subprocess.run([sys.executable,'phase2_physics.py','--physics',str(OUT/'pb32_physics.npz'),
                    '--out',str(OUT/'phase2_results')], capture_output=True, text=True)
print(r.stdout, r.stderr[-2000:])
open(OUT/'phase2_results'/'phase2_output.txt','w').write(r.stdout)""")
nb['cells']=C; nbf.write(nb,'ReliSense_32bearings_v1_1.ipynb'); print('ok')
