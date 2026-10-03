"""Meaningful leakage, sensor mask, physics gradient and MATLAB converter checks."""
import argparse, tempfile
from pathlib import Path
import numpy as np
import torch
from scipy.io import savemat
from relisense import ReliSense, synthetic, group_split, physics_loss, prepare, seed_all

def main():
    seed_all(7)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);synthetic(root/'demo.npz')
        data=dict(np.load(root/'demo.npz'));ids=group_split(data,7,'N09_M07_F10')
        for j,idx in enumerate(ids):
            assert ((data['condition'][idx]=='N09_M07_F10') if j==3 else (data['condition'][idx]!='N09_M07_F10')).all()
        model=ReliSense(3,512,3).eval();x=torch.randn(2,3,512)
        with torch.no_grad():
            a=model(x,torch.tensor([[1.,0.,1.],[0.,1.,1.]]));w=a[-1]
            assert torch.allclose(w.sum(1),torch.ones(2)) and w[0,1]==0 and w[1,0]==0
            altered=x.clone();altered[0,1]=999;altered[1,0]=-999
            assert torch.allclose(a[0],model(altered,torch.tensor([[1.,0.,1.],[0.,1.,1.]]))[0])
        try:model(x,torch.zeros(2,3))
        except ValueError:pass
        else:raise AssertionError('All-missing input must abstain')
        rec=torch.randn(2,3,512,requires_grad=True)
        geo={'n':8,'d':6.75,'D':28.55,'theta_deg':0}
        loss=physics_loss(rec,x,torch.tensor([900.,1500.]),2000,geo,'raw')
        loss.backward();assert torch.isfinite(loss) and torch.isfinite(rec.grad).all() and rec.grad.abs().sum()>0
        # Test the expected Name/Data MAT structure without claiming real dataset validation.
        mat=root/'N15_M07_F10_K001_1.mat'
        payload={'Y':np.array([{'Name':name,'Data':np.arange(1024,dtype=float)} for name in ['vibration_1','phase_current_1','phase_current_2']],dtype=object)}
        savemat(mat,{'fixture':payload});(root/'labels.csv').write_text('bearing,label,include\nK001,0,1\n')
        prepare(argparse.Namespace(raw=str(root),labels=str(root/'labels.csv'),output=str(root/'converted.npz'),
            channels='vibration_1,phase_current_1,phase_current_2',fs=64000,downsample=8,length=64,vibration_mode='raw',envelope_band='2000,12000'))
        converted=dict(np.load(root/'converted.npz'));assert converted['X'].shape==(2,3,64)
        assert converted['fs']==8000 and (converted['bearing']=='K001').all()
    print('PASS: bearing and recording isolation, target-condition isolation, mask invariance, all-missing abstention, differentiable physics objective, MATLAB fixture conversion')

if __name__=='__main__':main()
