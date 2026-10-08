"""Synthetic test archives with the Paderborn .mat layout (test only; never used for results)."""
import io, sys, zipfile
from pathlib import Path
import numpy as np
from scipy.io import savemat
import physics64 as P
CONDS=['N15_M07_F04','N15_M01_F10','N15_M07_F10','N09_M07_F10']
def signal(label, rpm, rng, n=64000*4//4, fs=64000):
    t=np.arange(n)/fs; fr=rpm/60; o=P.fault_orders()
    x=0.3*np.sin(2*np.pi*fr*t)+rng.normal(0,1,n)
    if label in (1,2,3):
        ff=(o['BPFO'] if label in (1,3) else o['BPFI'])*fr*(1+rng.normal(0,0.005))
        imp=np.zeros(n); imp[(np.arange(0,t[-1],1/ff)*fs).astype(int)]=1
        if label==2: imp*=1+0.8*np.sin(2*np.pi*fr*t)
        res=np.exp(-800*np.arange(200)/fs)*np.sin(2*np.pi*(4000+2000*rng.random())*np.arange(200)/fs)
        x+=4*np.convolve(imp,res)[:n]
        if label==3:
            imp2=np.zeros(n); imp2[(np.arange(0,t[-1],1/(o['BPFI']*fr))*fs).astype(int)]=1
            x+=3*np.convolve(imp2,res)[:n]
    x*=rng.uniform(0.5,3); x+=rng.normal(0,2)  # scale/offset differ per bearing
    cur=lambda ph: np.sin(2*np.pi*50*t+ph)+0.05*rng.normal(0,1,n)
    return x, cur(0), cur(2.1), np.full(n//16, rpm*(1+rng.normal(0,0.003)))
def mat_bytes(stem,label,rpm,rng):
    v,c1,c2,sp=signal(label,rpm,rng)
    Y=np.zeros((1,4),dtype=[('Name','O'),('Data','O'),('Raster','O')])
    for i,(nm,d,r) in enumerate([('vibration_1',v,'HostService'),('phase_current_1',c1,'HostService'),
                                 ('phase_current_2',c2,'HostService'),('speed',sp,'Mech_4kHz')]):
        Y[0,i]=(nm,d[None,:],r)
    b=io.BytesIO(); savemat(b,{stem:{'Y':Y,'Info':'fake'}}); return b.getvalue()
if __name__=='__main__':
    out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True); per=int(sys.argv[2])
    import csv
    for row in csv.DictReader(open('labels_32.csv')):
        b=row['bearing']; lab=int(row['label']); rng=np.random.default_rng(abs(hash(b))%2**32)
        with zipfile.ZipFile(out/f'{b}.rar','w') as z:
            for c in CONDS:
                rpm=int(c[1:3])*100
                for k in range(1,per+1):
                    stem=f'{c}_{b}_{k}'; z.writestr(f'{b}/{stem}.mat',mat_bytes(stem,lab,rpm,rng))
    print('fake archives in',out)
