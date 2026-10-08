"""Phase 8 (v1.0): modern learned baselines under the same bearing-wise protocols as Phase 4 (WDCNN).

Models (all settings fixed here, before any run; nothing is tuned on test bearings):
  resnet1d  1-D residual network on one 4096-sample window (stem conv k=15 s=2 + max-pool, four stages of two
            basic residual blocks with 32-64-128-128 channels, global average pooling, dropout 0.3, linear head)
  stft2d    2-D residual network on the log-magnitude STFT of the window (n_fft 256, hop 64, Hann; 129 x 65 image,
            standardised per image; three stages of two basic blocks with 32-64-128 channels, GAP, dropout 0.3)
Inputs: 'raw' = 8 kHz vibration window, 'env' = 2-12 kHz envelope window, each standardised by its own mean and SD
(same files and same standardisation as Phase 4).
Training recipe (modern, fixed): AdamW lr 1e-3, weight decay 1e-2, cosine schedule with 1 warm-up epoch, 20 epochs,
batch 128, class-weighted cross-entropy with label smoothing 0.1, random circular time shift of each window
(augmentation), mixed precision on GPU, final epoch used. Recording prediction = mean window probability.
Protocols: L8 (condition 0 and all), L10 (10 splits), A2R, LOBO (--lobo). Seeds 0,1,2 (one seed per LOBO fold).
Results are written after every fold (resumable): re-running the same command skips finished folds.
"""
from __future__ import annotations
import argparse, itertools, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from phase2_physics import L8_TRAIN, L8_TEST, L10
from phase4_learned import standardise, rec_probs

COND0 = 'N15_M07_F10'
# ReliSense per-bearing LOBO accuracy (%, manuscript table) for the paired comparison printed at the end
RELISENSE_LOBO = {'K001': 80, 'K002': 85, 'K003': 94, 'K004': 15, 'K005': 7, 'K006': 72, 'KA01': 100, 'KA03': 100, 'KA05': 100,
                  'KA06': 100, 'KA07': 100, 'KA08': 87, 'KA09': 75, 'KI01': 100, 'KI03': 94, 'KI05': 75, 'KI07': 70, 'KI08': 97,
                  'KA04': 99, 'KA15': 61, 'KA16': 100, 'KA22': 62, 'KA30': 46, 'KI04': 100, 'KI14': 80, 'KI16': 97, 'KI17': 39,
                  'KI18': 100, 'KI21': 86}


# ----------------------------------------------------------------------------------------------- models
class Block1d(nn.Module):
    def __init__(s, i, o, st):
        super().__init__()
        s.c1 = nn.Conv1d(i, o, 7, st, 3, bias=False); s.b1 = nn.BatchNorm1d(o)
        s.c2 = nn.Conv1d(o, o, 7, 1, 3, bias=False); s.b2 = nn.BatchNorm1d(o)
        s.sk = nn.Sequential() if st == 1 and i == o else nn.Sequential(nn.Conv1d(i, o, 1, st, bias=False), nn.BatchNorm1d(o))

    def forward(s, x):
        return F.relu(s.b2(s.c2(F.relu(s.b1(s.c1(x))))) + s.sk(x))


class ResNet1D(nn.Module):
    def __init__(s, n_cls=3, w=(32, 64, 128, 128)):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv1d(1, w[0], 15, 2, 7, bias=False), nn.BatchNorm1d(w[0]), nn.ReLU(), nn.MaxPool1d(2))
        layers, i = [], w[0]
        for k, o in enumerate(w):
            layers += [Block1d(i, o, 1 if k == 0 else 2), Block1d(o, o, 1)]; i = o
        s.body = nn.Sequential(*layers); s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(i, n_cls))

    def forward(s, x):
        return s.head(s.body(s.stem(x)).mean(-1))


class Block2d(nn.Module):
    def __init__(s, i, o, st):
        super().__init__()
        s.c1 = nn.Conv2d(i, o, 3, st, 1, bias=False); s.b1 = nn.BatchNorm2d(o)
        s.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(o)
        s.sk = nn.Sequential() if st == 1 and i == o else nn.Sequential(nn.Conv2d(i, o, 1, st, bias=False), nn.BatchNorm2d(o))

    def forward(s, x):
        return F.relu(s.b2(s.c2(F.relu(s.b1(s.c1(x))))) + s.sk(x))


class STFTResNet2D(nn.Module):
    def __init__(s, n_cls=3, w=(32, 64, 128), n_fft=256, hop=64):
        super().__init__()
        s.n_fft, s.hop = n_fft, hop; s.register_buffer('win', torch.hann_window(n_fft), persistent=False)
        s.stem = nn.Sequential(nn.Conv2d(1, w[0], 3, 1, 1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU())
        layers, i = [], w[0]
        for k, o in enumerate(w):
            layers += [Block2d(i, o, 1 if k == 0 else 2), Block2d(o, o, 1)]; i = o
        s.body = nn.Sequential(*layers); s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(i, n_cls))

    def tf(s, x):                                   # x: [B, 1, L] -> log-magnitude STFT image [B, 1, 129, 65]
        S = torch.stft(x[:, 0].float(), s.n_fft, s.hop, window=s.win, return_complex=True).abs()
        S = torch.log1p(S); S = (S - S.mean((1, 2), keepdim=True)) / (S.std((1, 2), keepdim=True) + 1e-6)
        return S.unsqueeze(1)

    def forward(s, x):
        return s.head(s.body(s.stem(s.tf(x))).mean((-2, -1)))


MODELS = {'resnet1d': ResNet1D, 'stft2d': STFTResNet2D}


# ----------------------------------------------------------------------------------------------- training
def train_predict(model_name, Xtr, ytr, Xte, seed, dev, epochs=20, bs=128):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MODELS[model_name]().to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    steps = epochs * int(np.ceil(len(ytr) / bs)); warm = int(np.ceil(len(ytr) / bs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(t, steps) / steps)))
    w = torch.tensor(len(ytr) / (3 * np.maximum(np.bincount(ytr, minlength=3), 1)), dtype=torch.float32, device=dev)
    use_amp = dev == 'cuda'; scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    xt = torch.from_numpy(Xtr).unsqueeze(1); yt = torch.from_numpy(ytr.astype('int64')); L = xt.shape[-1]
    for ep in range(epochs):
        m.train()
        for b in torch.randperm(len(yt)).split(bs):
            xb = xt[b].to(dev, non_blocking=True)
            xb = torch.roll(xb, int(np.random.randint(0, L)), dims=-1)          # random circular time shift
            with torch.autocast('cuda', enabled=use_amp):
                loss = F.cross_entropy(m(xb), yt[b].to(dev), weight=w, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    m.eval(); out = []
    with torch.no_grad(), torch.autocast('cuda', enabled=use_amp):
        for b in torch.arange(len(Xte)).split(512):
            out.append(F.softmax(m(torch.from_numpy(Xte[b.numpy()]).unsqueeze(1).to(dev)).float(), 1).cpu().numpy())
    return np.concatenate(out), sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--windows', default='pb32_4096.npz')
    ap.add_argument('--envelope', default='pb32_4096_envelope.npz')
    ap.add_argument('--physics', default='pb32_physics.npz')
    ap.add_argument('--models', default='resnet1d,stft2d')
    ap.add_argument('--inputs', default='raw,env')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--protocols', default='L8_cond0,L8_all,L10,A2R', help='add LOBO, or use --lobo')
    ap.add_argument('--lobo', action='store_true')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--out', default='phase8_results.json')
    ap.add_argument('--summary-only', action='store_true')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; print('device', dev, flush=True)
    out = Path(a.out); res = json.loads(out.read_text()) if out.exists() else []
    key = lambda r: (r['model'], r['input'], r['protocol'], r['fold'], r['seed'])
    done = {key(r) for r in res}
    w = np.load(a.windows); B, y, cond, recs = w['bearing'], w['y'], w['condition'], w['recording']
    inputs = [i for i in a.inputs.split(',') if i]
    data = {}
    if not a.summary_only:
        if 'raw' in inputs: data['raw'] = standardise(w['X'][:, 0])
        if 'env' in inputs: data['env'] = standardise(np.load(a.envelope)['X'][:, 0])
    del w
    ph = np.load(a.physics); org = {b: o for b, o in zip(ph['bearing'], ph['origin'])}
    single = sorted(set(B)); healthy = [b for b in single if org[b] == 'healthy']
    want = set(a.protocols.split(',')) | ({'LOBO'} if a.lobo else set())
    prot = []
    if 'L8_cond0' in want: prot.append(('L8_cond0', 0, L8_TRAIN, L8_TEST, cond == COND0))
    if 'L8_all' in want: prot.append(('L8_all', 0, L8_TRAIN, L8_TEST, None))
    if 'L10' in want:
        for k, tr in enumerate(itertools.combinations(range(5), 3)):
            prot.append(('L10', k, [c[i] for c in L10 for i in tr], [c[i] for c in L10 for i in range(5) if i not in tr], None))
    if 'A2R' in want:
        prot.append(('A2R', 0, healthy[:3] + [b for b in single if org[b] == 'artificial'], healthy[3:] + [b for b in single if org[b] == 'real'], None))
    if 'LOBO' in want:
        prot += [('LOBO', k, [x for x in single if x != b], [b], None) for k, b in enumerate(single)]
    seeds = [int(s) for s in a.seeds.split(',')]; t0 = time.time()
    for mn in [m for m in a.models.split(',') if m] if not a.summary_only else []:
        for inp, X in data.items():
            for name, fold, trb, teb, cm in prot:
                for s in (seeds if name != 'LOBO' else seeds[:1]):
                    if (mn, inp, name, fold, s) in done: continue
                    assert not set(trb) & set(teb), 'bearing in both partitions'
                    trm = np.isin(B, trb); tem = np.isin(B, teb)
                    if cm is not None: trm &= cm; tem &= cm
                    t1 = time.time(); pw, npar = train_predict(mn, X[trm], y[trm], X[tem], s, dev, a.epochs)
                    ids, P = rec_probs(pw, recs[tem]); yr = np.array([y[recs == r][0] for r in ids])
                    br = np.array([B[recs == r][0] for r in ids]); ok = P.argmax(1) == yr
                    res.append({'model': mn, 'input': inp, 'protocol': name, 'fold': fold, 'seed': s, 'acc': float(ok.mean()),
                                'per_bearing': {b: float(ok[br == b].mean()) for b in teb}, 'params': int(npar),
                                'train_seconds': round(time.time() - t1, 1)})
                    out.write_text(json.dumps(res, indent=1))                      # save after every fold
                    print(f'  {mn} {inp} {name} fold {fold} seed {s}: acc {ok.mean():.3f}  ({time.time() - t0:.0f} s)', flush=True)
    # ------------------------------------------------------------------ summary
    print('\nSUMMARY (recording accuracy, %; mean ± SD over folds and seeds)')
    for mn in sorted({r['model'] for r in res}):
        for inp in sorted({r['input'] for r in res if r['model'] == mn}):
            npar = next(r['params'] for r in res if r['model'] == mn)
            for name in ['L8_cond0', 'L8_all', 'L10', 'A2R', 'LOBO']:
                rr = [r for r in res if r['model'] == mn and r['input'] == inp and r['protocol'] == name]
                if not rr: continue
                if name == 'LOBO':
                    pb = {}
                    for r in rr: pb.update(r['per_bearing'])
                    line = f'mean of {len(pb)} bearings {100 * np.mean(list(pb.values())):.1f}'
                    common = [b for b in pb if b in RELISENSE_LOBO]
                    if len(common) >= 10:
                        d = np.array([RELISENSE_LOBO[b] - 100 * pb[b] for b in common])
                        rng = np.random.default_rng(0); bs_ = [rng.choice(d, len(d)).mean() for _ in range(20000)]
                        lo, hi = np.percentile(bs_, [2.5, 97.5])
                        try:
                            from scipy.stats import wilcoxon; p = wilcoxon(d).pvalue
                        except Exception:
                            p = float('nan')
                        line += (f' | ReliSense minus model: mean {d.mean():+.1f} [{lo:+.1f}, {hi:+.1f}], better/equal/worse '
                                 f'{(d > 0.5).sum()}/{(abs(d) <= 0.5).sum()}/{(d < -0.5).sum()}, Wilcoxon p = {p:.3f}')
                    print(f'  {mn:9s} {inp:3s} {name:9s} {line}')
                    print('      per bearing: ' + ', '.join(f'{b} {100 * v:.0f}' for b, v in sorted(pb.items())))
                else:
                    acc = 100 * np.array([r['acc'] for r in rr])
                    print(f'  {mn:9s} {inp:3s} {name:9s} {acc.mean():.1f} ± {acc.std():.1f}  (n = {len(acc)})')
            print(f'  {mn:9s} {inp:3s} trainable parameters: {npar:,}')
    print('\nFor comparison: ReliSense L8 cond0 93.6, L8 all 82.8, L10 74.6 ± 11.8, A2R 73.0, LOBO 80.1;'
          ' WDCNN-env 70.8, 59.9, 59.6, 69.0, 68.1; WDCNN-raw 46.4, 47.5, 58.1, 66.0, 61.9')


if __name__ == '__main__':
    main()
