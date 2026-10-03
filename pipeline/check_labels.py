"""Phase 1 (v1.0): per-bearing physics summary to help check labels_32.csv against Lessmeier et al. 2016.
For every bearing: median over its recordings of the BPFO and BPFI peak ratios (harmonic 1) per variant.
A bearing labelled outer ring should show a clear BPFO peak, inner ring a clear BPFI peak, healthy neither.
This is a check, not a label source: a weak peak can also mean a small or smeared defect (for example KA15)."""
import argparse, numpy as np
ap=argparse.ArgumentParser(); ap.add_argument('--physics',default='pb32_physics.npz'); ap.add_argument('--variant',default='fixed')
a=ap.parse_args(); d=np.load(a.physics); names=list(d['feature_names']); F=d[f'feat_{a.variant}']
o=F[:,names.index('BPFO_h1')]; i=F[:,names.index('BPFI_h1')]
h=d['label']==0; ref_o=np.median(o[h]); ref_i=np.median(i[h])
print(f'variant {a.variant}; healthy medians BPFO {ref_o:.2f} BPFI {ref_i:.2f}\n')
print(f"{'bearing':8}{'label':>6} {'origin':11}{'BPFO':>7}{'BPFI':>7}  physics says")
L={0:'healthy',1:'outer',2:'inner',3:'both'}
for b in dict.fromkeys(d['bearing']):
    m=d['bearing']==b; lab=int(d['label'][m][0]); po=np.median(o[m]); pi=np.median(i[m])
    so=po-ref_o>0.7; si=pi-ref_i>0.7
    says={(0,0):'healthy',(1,0):'outer',(0,1):'inner',(1,1):'both'}[(int(so),int(si))]
    flag='' if says==L[lab] else '   <-- check'
    print(f"{b:8}{L[lab]:>6} {str(d['origin'][m][0]):11}{po:7.2f}{pi:7.2f}  {says}{flag}")
