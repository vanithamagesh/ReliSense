"""ReliSense research prototype. Demo data are synthetic, never publication evidence.
Python 3.10+; torch, numpy, scipy, scikit-learn, matplotlib.
See README.md for group splits, checkpoint status, and implementation boundaries.
"""
from __future__ import annotations
import argparse, csv, json, math, random, re, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.signal import resample_poly, butter, sosfiltfilt, hilbert
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score

CHANNELS = ['vibration_1', 'phase_current_1', 'phase_current_2']
CONDITIONS = ['N15_M07_F10', 'N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04']

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))

def synthetic(path, length=512, seed=7):
    """Independent simulated bearing identities; no claim of realistic motor dynamics."""
    rng=np.random.default_rng(seed); fs=2000.; t=np.arange(length)/fs
    xs=[]; ys=[]; groups=[]; cond=[]; rec=[]; rpm=[]
    for y in range(3):
        for g in range(8):
            gid=f'sim_{y}_{g}'
            offset=rng.uniform(-.15,.15)
            for c in range(4):
                for r in range(5):
                    speed=900. if c==1 else 1500.
                    phase=rng.uniform(0,2*np.pi)
                    carrier=4*speed/60
                    fault=(y>0)*(.3+.15*y)*np.sin(2*np.pi*(80+65*y)*t+phase)
                    v=.25*np.sin(2*np.pi*speed/60*t)+fault+.14*rng.normal(size=length)
                    i1=np.sin(2*np.pi*carrier*t+phase)+.3*fault+.08*rng.normal(size=length)
                    i2=np.sin(2*np.pi*carrier*t+phase-2*np.pi/3)+.25*fault+.08*rng.normal(size=length)
                    xs.append(np.stack([v,i1,i2])+offset);ys.append(y);groups.append(gid)
                    cond.append(CONDITIONS[c]);rec.append(f'{gid}_{c}_{r}');rpm.append(speed)
    np.savez_compressed(path,X=np.array(xs,dtype='float32'),y=np.array(ys),
        bearing=np.array(groups),condition=np.array(cond),recording=np.array(rec),
        rpm=np.array(rpm,dtype='float32'),fs=np.array(fs),channel_names=np.array(CHANNELS),
        provenance=np.array('SYNTHETIC PIPELINE DEMO ONLY'),signal_mode=np.array('raw'))

def _find_channels(obj, found):
    if isinstance(obj,dict):
        lower={k.lower():k for k in obj}
        if 'name' in lower and 'data' in lower:
            name=str(obj[lower['name']]).strip().lower()
            arr=np.asarray(obj[lower['data']]).squeeze()
            if arr.ndim==1 and np.issubdtype(arr.dtype,np.number):found[name]=arr
        for v in obj.values():_find_channels(v,found)
    elif isinstance(obj,(list,tuple)):
        for v in obj:_find_channels(v,found)
    elif isinstance(obj,np.ndarray) and obj.dtype==object:
        for v in obj.flat:_find_channels(v,found)

def prepare(args):
    """Strict metadata parsing; labels must be verified against original fact sheets."""
    labels={}
    with open(args.labels,newline='') as f:
        for row in csv.DictReader(f):
            if row.get('include','1')=='1':labels[row['bearing']]=int(row['label'])
    if not labels:raise ValueError('No included labels. Audit labels.csv first.')
    channels=args.channels.split(','); xs=[];ys=[];gs=[];cs=[];rs=[];speeds=[]; audit=[]
    fs=args.fs//args.downsample
    if args.fs%args.downsample:raise ValueError('Use an integer sample rate ratio.')
    for p in sorted(Path(args.raw).rglob('*.mat')):
        match=re.search(r'(N\d+_M\d+_F\d+)_(K[A-Z]?\d+)_(\d+)',p.stem)
        if not match:raise ValueError(f'Unrecognized recording filename {p.name}')
        c,g,r=match.groups()
        if g not in labels:continue
        found={};_find_channels(loadmat(p,simplify_cells=True),found)
        missing=[ch for ch in channels if ch.lower() not in found]
        if missing:raise ValueError(f'{p}: missing {missing}; found {sorted(found)}')
        signals=[np.asarray(found[ch.lower()],dtype=float) for ch in channels]
        if len(set(map(len,signals)))!=1:raise ValueError(f'Unequal main-channel lengths: {p}')
        if not all(np.isfinite(s).all() for s in signals):raise ValueError(f'Nonfinite data: {p}')
        if args.vibration_mode=='envelope':
            if channels[0]!='vibration_1':raise ValueError('Envelope mode expects vibration_1 first')
            low,high=map(float,args.envelope_band.split(','))
            if not 0<low<high<args.fs/2:raise ValueError('Invalid envelope band')
            sos=butter(4,[low,high],btype='bandpass',fs=args.fs,output='sos')
            signals[0]=np.abs(hilbert(sosfiltfilt(sos,signals[0])))
        arr=np.stack([resample_poly(s,1,args.downsample) for s in signals]).astype('float32')
        speed=int(c.split('_')[0][1:])*100.
        # The filename supplies the setpoint, not an independently verified measured speed.
        count=0
        for start in range(0,arr.shape[1]-args.length+1,args.length):
            xs.append(arr[:,start:start+args.length]);ys.append(labels[g]);gs.append(g)
            cs.append(c);rs.append(p.stem);speeds.append(speed);count+=1
        audit.append({'file':str(p),'recording':p.stem,'bearing':g,'condition':c,
            'source_samples':len(signals[0]),'windows':count,'label':labels[g]})
    if not xs:raise ValueError('No usable windows found. Check path, audited labels, and length.')
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,X=np.array(xs),y=np.array(ys),bearing=np.array(gs),
        condition=np.array(cs),recording=np.array(rs),rpm=np.array(speeds,dtype='float32'),
        fs=np.array(fs),channel_names=np.array(channels),
        provenance=np.array('PADERBORN REAL RECORDINGS; USER AUDITED LABELS'),
        signal_mode=np.array(args.vibration_mode))
    dest.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2))
    print(f'Converted {len(audit)} recordings into {len(xs)} windows: {dest}')

def group_split(data,seed,heldout=None):
    """Per-class bearing-disjoint partitions; target condition is test only."""
    rng=np.random.default_rng(seed); groups=data['bearing'];labels=data['y']
    parts=[[],[],[],[]] # training, validation (temperature), calibration, test
    for label in sorted(np.unique(labels)):
        gids=np.unique(groups[labels==label]);rng.shuffle(gids)
        if len(gids)<5:raise ValueError(f'Class {label} needs >=5 physical bearings for four partitions; got {len(gids)}')
        ntest=max(1,len(gids)//5);ncal=max(1,len(gids)//5);nval=max(1,len(gids)//5)
        chunks=[gids[ntest+ncal+nval:],gids[ntest+ncal:ntest+ncal+nval],gids[ntest:ntest+ncal],gids[:ntest]]
        for part,chunk in zip(parts,chunks):part.extend(chunk.tolist())
    indices=[]
    for j,part in enumerate(parts):
        keep=np.isin(groups,part)
        if heldout:keep &= (data['condition']==heldout) if j==3 else (data['condition']!=heldout)
        idx=np.flatnonzero(keep)
        if len(idx)==0:raise ValueError('Empty partition. Verify condition and bearing coverage.')
        indices.append(idx)
    for a in range(4):
        for b in range(a):
            assert not set(groups[indices[a]])&set(groups[indices[b]])
            assert not set(data['recording'][indices[a]])&set(data['recording'][indices[b]])
    return indices

class ReliSense(nn.Module):
    def __init__(self,k,length,nclasses,backbone='compact',checkpoint='AutonLab/MOMENT-1-small',uniform=False,stem_norm=False):
        super().__init__();self.k=k;self.length=length;self.uniform=uniform;self.backbone=backbone;d=48
        if stem_norm:
            # BatchNorm keeps the stem activations at unit scale; without it the time-averaged
            # features of a zero-mean signal are close to zero and training stalls at chance level.
            self.local=nn.Sequential(nn.Conv1d(1,24,15,4,7),nn.BatchNorm1d(24),nn.GELU(),nn.Conv1d(24,d,9,4,4),nn.BatchNorm1d(d),nn.GELU())
        else:
            self.local=nn.Sequential(nn.Conv1d(1,24,15,4,7),nn.GELU(),nn.Conv1d(24,d,9,4,4),nn.GELU())
        layer=nn.TransformerEncoderLayer(d,4,128,.15,batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.sensor_id=nn.Parameter(torch.randn(k,d)*.02)
        self.cross=nn.MultiheadAttention(d,4,dropout=.15,batch_first=True)
        self.reliability=nn.Sequential(nn.Linear(2*d+4,48),nn.GELU(),nn.Dropout(.15),nn.Linear(48,1))
        self.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Dropout(.2),nn.Linear(d,nclasses))
        self.decode=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Linear(128,k*length))
        if backbone=='moment':
            from momentfm import MOMENTPipeline
            self.fm=MOMENTPipeline.from_pretrained(checkpoint,model_kwargs={'task_name':'embedding'})
            self.fm.init();self.fm.eval()
            for p in self.fm.parameters():p.requires_grad=False
            self.fm_adapter=nn.Linear(int(self.fm.config.d_model),d)
            # Frozen FM features complement the local signal path, preserving amplitude statistics.
    def train(self,mode=True):
        super().train(mode)
        if hasattr(self,'fm'):self.fm.eval()
        return self
    def forward(self,x,available):
        if (available.sum(1)==0).any():raise ValueError('No sensors available: abstain, do not classify.')
        b,k,l=x.shape;masked=x*available[:,:,None]
        tokens=self.local(masked.reshape(b*k,1,l)).transpose(1,2)
        # Deterministic sinusoidal temporal position encoding.
        pos=torch.arange(tokens.shape[1],device=x.device)[:,None]
        freq=torch.exp(torch.arange(0,48,2,device=x.device)*(-math.log(10000.)/48))
        pe=torch.zeros(tokens.shape[1],48,device=x.device)
        pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq)
        features=self.temporal(tokens+pe[None]).mean(1).reshape(b,k,48)
        if self.backbone=='moment':
            # Split into 512-point chunks; no sample skipping and no time-axis stretching.
            pad=(-l)%512;xx=F.pad(masked,(0,pad))
            chunks=xx.unfold(-1,512,512).reshape(-1,1,512)
            masks=torch.ones((b,k,xx.shape[-1]),device=x.device)
            if pad:masks[:,:,-pad:]=0
            masks=masks.unfold(-1,512,512).reshape(-1,512)
            with torch.no_grad():
                embed=self.fm(x_enc=chunks,input_mask=masks).embeddings
            embed=embed.reshape(b,k,-1,embed.shape[-1]).mean(2)
            features=features+self.fm_adapter(embed)
        features=features+self.sensor_id[None]
        joint,_=self.cross(features,features,features,key_padding_mask=~available.bool())
        stats=torch.stack([masked.mean(-1),masked.std(-1),masked.square().mean(-1).sqrt(),masked.abs().amax(-1)],-1)
        quality_logits=self.reliability(torch.cat([features,joint,stats],-1)).squeeze(-1)
        reliability=torch.sigmoid(quality_logits)*available
        if self.uniform:reliability=available
        weights=reliability/reliability.sum(1,keepdim=True).clamp_min(1e-8)
        fused=((features+joint)*weights[:,:,None]).sum(1)
        return self.head(fused),quality_logits,self.decode(fused).reshape(b,k,l),weights

def corrupt(x,kind='mixed',sensor=None,level=.5):
    """All units are source-training standard deviations; paired clean targets are retained."""
    out=x.clone();b,k,l=x.shape;avail=torch.ones((b,k),device=x.device);good=avail.clone()
    if kind=='clean':return out,avail,good
    selected=torch.randint(k,(b,),device=x.device) if sensor is None else torch.full((b,),sensor,device=x.device)
    for n in range(b):
        s=int(selected[n]);op=random.choice(['noise','bias','gain','drift','missing','clip']) if kind=='mixed' else kind
        if op=='noise':out[n,s]+=level*torch.randn(l,device=x.device)
        elif op=='bias':out[n,s]+=level
        elif op=='gain':out[n,s]*=(1+level)
        elif op=='drift':out[n,s]+=torch.linspace(0,level,l,device=x.device)
        elif op=='clip':out[n,s]=out[n,s].clamp(-level,level)
        elif op=='missing':out[n,s]=0;avail[n,s]=0
        else:raise ValueError(op)
        good[n,s]=0
    return out,avail,good

def envelope_torch(x):
    """Analytic-signal magnitude for already selected/filtered vibration band."""
    n=x.shape[-1];h=torch.zeros(n,device=x.device);h[0]=1
    if n%2==0:h[1:n//2]=2;h[n//2]=1
    else:h[1:(n+1)//2]=2
    return torch.fft.ifft(torch.fft.fft(x)*h).abs()

def physics_loss(reconstructed,clean,rpm,fs,geometry,mode):
    """Optional kinematic envelope-band preservation. No fabricated energy-balance equation."""
    if geometry is None:return reconstructed.sum()*0
    def descriptor(z):
        v=z[:,0]
        if mode!='envelope':v=envelope_torch(v)
        v=v-v.mean(-1,keepdim=True)
        power=torch.fft.rfft(v*torch.hann_window(v.shape[-1],device=v.device)).abs().square()
        freq=torch.fft.rfftfreq(v.shape[-1],1/fs,device=v.device)
        n,d,D,theta=[geometry[q] for q in ['n','d','D','theta_deg']]
        a=d/D*math.cos(math.radians(theta)); shaft=rpm/60
        outer=n/2*(1-a)*shaft;inner=n/2*(1+a)*shaft
        features=[]
        # At least two FFT bins to accommodate finite resolution and limited slip.
        for cf in [outer,inner,2*outer,2*inner]:
            width=torch.maximum(.03*cf,torch.full_like(cf,2*fs/v.shape[-1]))
            mask=(freq[None]-cf[:,None]).abs()<=width[:,None]
            if (mask.sum(1)==0).any():raise ValueError('Physics frequency outside Nyquist range')
            features.append((power*mask).sum(-1)/power.sum(-1).clamp_min(1e-8))
        return torch.stack(features,-1)
    return F.mse_loss(descriptor(reconstructed),descriptor(clean).detach())

def get_batches(x,y,rpm,batch=32,shuffle=False):
    order=torch.randperm(len(x)) if shuffle else torch.arange(len(x))
    for sl in order.split(batch):yield x[sl],y[sl],rpm[sl]

@torch.no_grad()
def predict(model,x,batch=32,kind='clean',sensor=None,level=.5,temperature=1.,mc=1):
    model.eval();allp=[];allw=[];allu=[]
    for chunk in x.split(batch):
        z,avail,_=corrupt(chunk,kind,sensor,level);ps=[];ws=[]
        for m in range(mc):
            model.train(mc>1)
            logits,_,_,weights=model(z,avail)
            ps.append(F.softmax(logits/temperature,dim=-1));ws.append(weights)
        p=torch.stack(ps);mean=p.mean(0)
        h=lambda q:-(q*q.clamp_min(1e-9).log()).sum(-1)
        mi=(h(mean)-h(p).mean(0)).clamp_min(0)
        allp.append(mean.cpu().numpy());allw.append(torch.stack(ws).mean(0).cpu().numpy());allu.append(mi.cpu().numpy())
    model.eval();return np.concatenate(allp),np.concatenate(allw),np.concatenate(allu)

def temperature_fit(model,x,y):
    model.eval()
    with torch.no_grad():
        logits=torch.cat([model(z,torch.ones(z.shape[:2],device=z.device))[0] for z in x.split(32)])
    # Deterministic grid fitted ONLY to validation-bearing predictions.
    grid=np.geomspace(.25,4,61)
    return float(min(grid,key=lambda t:F.cross_entropy(logits/t,y).item()))

def score(y,p):
    pred=p.argmax(1);conf=p.max(1);correct=pred==y;ece=0.
    for a,b in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        ids=(conf>a)&(conf<=b)
        if ids.any():ece+=ids.mean()*abs(correct[ids].mean()-conf[ids].mean())
    return dict(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),ece=float(ece),
        nll=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-9,1)).mean()),
        brier=float(((p-np.eye(p.shape[1])[y])**2).sum(1).mean()),
        confusion_matrix=confusion_matrix(y,pred,labels=np.arange(p.shape[1])).tolist())

def aggregate_recordings(prob,y,recordings):
    ids=np.unique(recordings);ps=[];ys=[]
    for r in ids:
        mask=recordings==r
        if len(np.unique(y[mask]))!=1:raise ValueError('Inconsistent recording labels')
        ps.append(prob[mask].mean(0));ys.append(y[mask][0])
    return np.array(ps),np.array(ys),ids

def train_run(args):
    seed_all(args.seed);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.command=='demo':
        path=out/'synthetic.npz';synthetic(path,args.length,args.seed)
    else:path=Path(args.data)
    data=dict(np.load(path,allow_pickle=False));X=data['X'];y=data['y'].astype('int64')
    if not np.array_equal(np.unique(y),np.arange(len(np.unique(y)))):raise ValueError('Labels must be consecutive integers from zero')
    if X.ndim!=3 or not np.isfinite(X).all():raise ValueError('X must be finite [windows, sensors, points]')
    parts=group_split(data,args.seed,args.heldout);names=['train','validation','calibration','test']
    split={name:{'bearings':np.unique(data['bearing'][idx]).tolist(),'recordings':np.unique(data['recording'][idx]).tolist(),'windows':len(idx)} for name,idx in zip(names,parts)}
    (out/'split.json').write_text(json.dumps(split,indent=2))
    mean=X[parts[0]].mean((0,2),keepdims=True);std=X[parts[0]].std((0,2),keepdims=True).clip(1e-6)
    X=(X-mean)/std;device=torch.device(args.device)
    sets=[(torch.tensor(X[idx],device=device),torch.tensor(y[idx],device=device),torch.tensor(data['rpm'][idx],device=device)) for idx in parts]
    model=ReliSense(X.shape[1],X.shape[2],len(np.unique(y)),args.backbone,args.checkpoint,args.uniform).to(device)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-4)
    geometry=json.loads(Path(args.geometry).read_text()) if args.geometry else None
    history=[];best=float('inf');best_state=None
    for stage,epochs in [('pretrain',args.pretrain),('supervised',args.epochs)]:
        for ep in range(epochs):
            model.train();losses=[]
            for clean,target,rpm in get_batches(*sets[0],args.batch,True):
                z,avail,good=corrupt(clean,'mixed',level=random.uniform(.15,1.))
                if stage=='pretrain':
                    # Independent temporal masking, on top of whole-channel corruption.
                    z=z*(torch.rand_like(z)>.15)
                logits,q,rec,w=model(z,avail)
                reconstruction=F.mse_loss(rec,clean)
                quality=F.binary_cross_entropy_with_logits(q,good)
                phys=physics_loss(rec,clean,rpm,float(data['fs']),geometry,str(data.get('signal_mode','raw')))
                loss=.3*reconstruction+.2*quality+args.physics_weight*phys
                if stage=='supervised':loss=loss+F.cross_entropy(logits,target)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                losses.append(float(loss.detach()))
            p,_,_=predict(model,sets[1][0],args.batch);metric=score(sets[1][1].cpu().numpy(),p)
            history.append({'stage':stage,'epoch':ep+1,'loss':float(np.mean(losses)),**{k:v for k,v in metric.items() if k!='confusion_matrix'}})
            if stage=='supervised' and metric['nll']<best:
                best=metric['nll'];best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(f'{stage} {ep+1}/{epochs} loss={np.mean(losses):.4f} val_f1={metric["macro_f1"]:.4f}',flush=True)
    if best_state is None:raise ValueError('Need at least one supervised epoch')
    model.load_state_dict(best_state);temperature=temperature_fit(model,sets[1][0],sets[1][1])
    calp,_,_=predict(model,sets[2][0],args.batch,temperature=temperature,mc=args.mc)
    calp,caly,_=aggregate_recordings(calp,y[parts[2]],data['recording'][parts[2]])
    nonconformity=1-calp[np.arange(len(caly)),caly];n=len(caly)
    rank=math.ceil((n+1)*(1-args.alpha))
    qhat=float(np.sort(nonconformity)[rank-1]) if rank<=n else float('inf')
    results=[];testx=sets[3][0];testy=y[parts[3]];records=data['recording'][parts[3]]
    scenarios=[('clean',None,0.)]+[(kind,k,.5) for kind in ['missing','noise','bias','gain','drift','clip'] for k in range(X.shape[1])]
    # Separate from conformal calibration: fixed rejection policy fit to validation confidence.
    vp,_,_=predict(model,sets[1][0],args.batch,temperature=temperature,mc=args.mc)
    reject_threshold=float(np.quantile(vp.max(1),.1))
    for kind,sensor,level in scenarios:
        # Reset corruption random numbers so ablations can share exactly the same cases.
        seed_all(args.seed+1000+len(results))
        p,w,mi=predict(model,testx,args.batch,kind,sensor,level,temperature,args.mc)
        ap,ay,rids=aggregate_recordings(p,testy,records)
        sets_pred=(1-ap)<=qhat;accepted=ap.max(1)>=reject_threshold
        row={'scenario':kind,'sensor':sensor,'level':level,**score(ay,ap),
            'recordings':len(ay),'conformal_empirical_coverage':float(sets_pred[np.arange(len(ay)),ay].mean()),
            'conformal_mean_set_size':float(sets_pred.sum(1).mean()),
            'accepted_fraction':float(accepted.mean()),
            'accepted_error':float((ap.argmax(1)[accepted]!=ay[accepted]).mean()) if accepted.any() else None,
            'mean_sensor_weights':w.mean(0).tolist(),'mean_mc_disagreement':float(mi.mean())}
        results.append(row)
        filename=f'{kind}_{sensor if sensor is not None else "all"}.npz'
        np.savez_compressed(out/filename,probabilities=p,labels=testy,recording=records,
            bearing=data['bearing'][parts[3]],sensor_weights=w,mc_disagreement=mi,
            recording_probabilities=ap,recording_labels=ay,recording_ids=rids)
    # Measure gate isolation across clean and random-channel corrupted windows.
    seed_all(args.seed+999);mixed,avail,good=corrupt(testx,'mixed')
    model.eval()
    with torch.no_grad():
        qs=torch.cat([model(z,a)[1] for z,a in zip(mixed.split(args.batch),avail.split(args.batch))]).sigmoid().cpu().numpy()
    gate_auroc=float(roc_auc_score((1-good.cpu().numpy()).ravel(),(1-qs).ravel()))
    summary={'status':'SYNTHETIC DEMO ONLY' if args.command=='demo' else 'REAL DATA PROTOTYPE RUN',
        'backbone':args.backbone,'pretrained_checkpoint':args.checkpoint if args.backbone=='moment' else None,
        'physics_enabled':geometry is not None,'geometry':geometry,'temperature':temperature,
        'alpha':args.alpha,'conformal_qhat':qhat if math.isfinite(qhat) else None,
        'conformal_calibration_recordings':n,'sensor_isolation_auroc_synthetic_corruption':gate_auroc,
        'conformal_note':'Empirical recording coverage only; correlated recordings and domain shift invalidate naive exchangeable guarantees.',
        'args':vars(args),'scenarios':results}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    (out/'history.json').write_text(json.dumps(history,indent=2))
    torch.save({'state_dict':model.state_dict(),'normalizer_mean':mean,'normalizer_std':std,
        'temperature':temperature,'args':vars(args),'length':X.shape[2],'channels':X.shape[1],
        'nclasses':len(np.unique(y)),'geometry':geometry},out/'checkpoint.pt')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4));ax.plot([h['loss'] for h in history]);ax.set(xlabel='Epoch including pretraining',ylabel='Training objective',title=summary['status']);fig.tight_layout();fig.savefig(out/'loss.png',dpi=180);plt.close(fig)
    print(json.dumps({'status':summary['status'],'clean':results[0],'sensor_isolation_auroc':gate_auroc},indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--raw',required=True);p.add_argument('--labels',required=True);p.add_argument('--output',default='paderborn.npz')
    p.add_argument('--channels',default=','.join(CHANNELS));p.add_argument('--fs',type=int,default=64000);p.add_argument('--downsample',type=int,default=8)
    p.add_argument('--length',type=int,default=4096);p.add_argument('--vibration-mode',choices=['raw','envelope'],default='raw');p.add_argument('--envelope-band',default='2000,12000')
    for command in ['demo','train']:
        p=sub.add_parser(command);p.add_argument('--output',default=f'{command}_output');p.add_argument('--seed',type=int,default=7)
        p.add_argument('--epochs',type=int,default=8);p.add_argument('--pretrain',type=int,default=2);p.add_argument('--batch',type=int,default=32)
        p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--device',default='cpu');p.add_argument('--mc',type=int,default=1)
        p.add_argument('--heldout',choices=CONDITIONS);p.add_argument('--geometry');p.add_argument('--physics-weight',type=float,default=.1)
        p.add_argument('--backbone',choices=['compact','moment'],default='compact');p.add_argument('--checkpoint',default='AutonLab/MOMENT-1-small')
        p.add_argument('--uniform',action='store_true');p.add_argument('--alpha',type=float,default=.1)
        if command=='train':p.add_argument('--data',required=True)
        else:p.add_argument('--length',type=int,default=512)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args)
    else:train_run(args)

if __name__=='__main__':main()
