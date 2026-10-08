"""Phase 23 (v1.0): modern deep and physics-informed networks under the bearing-wise Paderborn protocols (GPU recommended).

Fixed before any result is seen; every model is reported. Training as for WDCNN in phase 4: AdamW, lr 1e-3, weight decay
1e-4, batch 128, class-weighted cross-entropy, last epoch used, nothing tuned on test bearings; window models: 20 epochs,
recording probability = mean of its window probabilities; order-domain model: 60 epochs on one input per recording.
Models
  resnet_raw   1-D ResNet (stem + 4 stages of 2 basic blocks, 32-64-128-128 channels) on 8-kHz vibration windows (4096)
  resnet_env   the same ResNet on 2-12-kHz envelope windows (4096)
  irk_raw      physics-informed impulse-response kernel network: first layer = 32 learnable damped-resonance kernels
               h(t) = exp(-xi*w*t) sin(w*t), t in [0, 8 ms) (the impulse response of the signal model), learnable
               resonance frequency w and damping xi, followed by a WDCNN-type convolutional stack (in the spirit of
               WaveletKernelNet); raw 8-kHz windows
  fo_resnet    physics-informed fault-order-guided ResNet: input = envelope order spectrum O(r) (orders 0.2-11.8 of
               pb32_physics.npz) and a second channel that marks the 15 search windows of the fault orders
               (+-max(2 %, 0.04) at h = 1-3 of BPFO, BPFI, BSF, FTF, shaft); ResNet 16-32-64-64
Windows are standardized by their own mean and SD (as in phase 4). Protocols: L8 cond. 0, L8 all and A2R with seeds 0, 1, 2;
L10 (10 splits) and LOBO (29 folds) with seed 0 (--l10_seeds 0,1,2 for three seeds on L10). Mixed precision on GPU.
Every training is checkpointed in <root>/checkpoints/phase23/ (recording probabilities included) and reused on a second run.
Output: <root>/phase23_results.json (accuracy, false alarms on healthy recordings, per-bearing accuracy)
Colab (GPU runtime):
  !python /content/drive/MyDrive/ReliSense_study/phase23_deep_baselines.py --root /content/drive/MyDrive/ReliSense_study
  (--models resnet_raw,irk_raw to run a subset; run again to resume after a disconnect)
"""
import argparse, itertools, json, os, time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ap = argparse.ArgumentParser(); ap.add_argument('--root', default='/content/drive/MyDrive/ReliSense_study')
ap.add_argument('--models', default='resnet_raw,resnet_env,irk_raw,fo_resnet'); ap.add_argument('--seeds', default='0,1,2')
ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--epochs_order', type=int, default=60)
ap.add_argument('--windows', default=None); ap.add_argument('--envelope', default=None); ap.add_argument('--physics', default=None)
ap.add_argument('--no_lobo', action='store_true'); ap.add_argument('--l10_seeds', default='0')
a = ap.parse_args(); R = a.root; CK = R + '/checkpoints/phase23'; os.makedirs(CK, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; print('device', dev, flush=True); t0 = time.time()
COND0 = 'N15_M07_F10'
L8_TRAIN = ['K002', 'KA01', 'KA05', 'KA07', 'KI01', 'KI05', 'KI07']
L8_TEST = ['K001', 'KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']
L10 = [['K001', 'K002', 'K003', 'K004', 'K005'], ['KA04', 'KA15', 'KA16', 'KA22', 'KA30'], ['KI04', 'KI14', 'KI16', 'KI18', 'KI21']]


# ------------------------------------------------------------------ models
class Basic(nn.Module):
    def __init__(s, i, o, st, k=7):
        super().__init__()
        s.c1 = nn.Conv1d(i, o, k, st, k // 2, bias=False); s.b1 = nn.BatchNorm1d(o)
        s.c2 = nn.Conv1d(o, o, k, 1, k // 2, bias=False); s.b2 = nn.BatchNorm1d(o)
        s.sc = nn.Sequential() if (st == 1 and i == o) else nn.Sequential(nn.Conv1d(i, o, 1, st, bias=False), nn.BatchNorm1d(o))

    def forward(s, x):
        return F.relu(s.b2(s.c2(F.relu(s.b1(s.c1(x))))) + s.sc(x))


class ResNet1D(nn.Module):
    def __init__(s, cin=1, ch=(32, 64, 128, 128), stem_k=15, stem_s=2, pool=True, n_cls=3):
        super().__init__()
        L = [nn.Conv1d(cin, ch[0], stem_k, stem_s, stem_k // 2, bias=False), nn.BatchNorm1d(ch[0]), nn.ReLU()] + ([nn.MaxPool1d(2)] if pool else [])
        prev = ch[0]
        for j, c in enumerate(ch):
            L += [Basic(prev, c, 1 if j == 0 else 2), Basic(c, c, 1)]; prev = c
        s.f = nn.Sequential(*L, nn.AdaptiveAvgPool1d(1), nn.Flatten()); s.h = nn.Sequential(nn.Dropout(0.3), nn.Linear(prev, n_cls))

    def forward(s, x):
        return s.h(s.f(x))


class IRKLayer(nn.Module):
    """Learnable damped-resonance kernels h(t) = exp(-xi*w*t) sin(w*t) (impulse response of the signal model)."""
    def __init__(s, n=32, K=64, fs=8000.0):
        super().__init__()
        s.register_buffer('t', torch.arange(K, dtype=torch.float32) / fs)
        s.logw = nn.Parameter(torch.log(2 * np.pi * torch.logspace(np.log10(200.0), np.log10(3800.0), n)))
        s.logxi = nn.Parameter(torch.log(torch.full((n,), 0.05)))

    def forward(s, x):
        w = torch.exp(s.logw)[:, None]; xi = torch.exp(s.logxi).clamp(1e-3, 0.9)[:, None]
        k = torch.exp(-xi * w * s.t) * torch.sin(w * s.t); k = k / (k.norm(dim=1, keepdim=True) + 1e-8)
        return F.conv1d(x, k[:, None, :], padding=k.shape[1] // 2)


class IRKNet(nn.Module):
    def __init__(s, n_cls=3):
        super().__init__()
        def blk(i, o, k=3):
            return [nn.Conv1d(i, o, k, 1, 1), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2)]
        s.f = nn.Sequential(IRKLayer(), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(8), *blk(32, 32), *blk(32, 64), *blk(64, 64), *blk(64, 64),
                            nn.AdaptiveAvgPool1d(8), nn.Flatten())
        s.h = nn.Sequential(nn.Linear(512, 100), nn.ReLU(), nn.Dropout(0.3), nn.Linear(100, n_cls))

    def forward(s, x):
        return s.h(s.f(x))


def make(name):
    if name in ('resnet_raw', 'resnet_env'): return ResNet1D()
    if name == 'irk_raw': return IRKNet()
    return ResNet1D(cin=2, ch=(16, 32, 64, 64), stem_k=7, stem_s=1, pool=False)


# ------------------------------------------------------------------ data
ph = dict(np.load(a.physics or R + '/pb32/pb32_physics.npz', allow_pickle=True))
keep_r = ph['label'] < 3; PB, Py, Pc = ph['bearing'][keep_r], ph['label'][keep_r].astype(int), ph['condition'][keep_r]
org = {b: str(o) for b, o in zip(ph['bearing'], ph['origin'])}; lab = {b: int(l) for b, l in zip(PB, Py)}
single = sorted(set(PB)); healthy = sorted(b for b in single if lab[b] == 0)
n_, d_, D_ = 8, 6.75, 28.55; q = d_ / D_
FO = [n_ / 2 * (1 - q), n_ / 2 * (1 + q), D_ / (2 * d_) * (1 - q ** 2), 0.5 * (1 - q), 1.0]
g = ph['order_grid'].astype(float); gm = g <= 11.8
mask = np.zeros(gm.sum(), np.float32)
for o in FO:
    for h in (1, 2, 3):
        c = h * o; mask[np.abs(g[gm] - c) <= max(0.02 * c, 0.04)] = 1.0
ORD = np.nan_to_num(ph['order_fixed'][keep_r][:, gm].astype(np.float32))
ORD = np.stack([ORD, np.broadcast_to(mask, ORD.shape)], 1).astype(np.float32)
WIN = None


def windows(kind):
    global WIN
    if WIN is None:
        w = np.load(a.windows or R + '/pb32/pb32_4096.npz'); WIN = {'B': w['bearing'], 'y': w['y'].astype(int), 'cond': w['condition'], 'rec': w['recording']}
        WIN['raw'] = None; WIN['_w'] = w
    if WIN.get(kind) is None:
        src = WIN['_w'] if kind == 'raw' else np.load(a.envelope or R + '/pb32/pb32_4096_envelope.npz')
        x = src['X'][:, 0].astype(np.float32); x -= x.mean(1, keepdims=True); x /= x.std(1, keepdims=True) + 1e-8; WIN[kind] = x
    return WIN


# ------------------------------------------------------------------ training
def fit_predict(name, Xtr, ytr, Xte, seed, epochs):
    torch.manual_seed(seed); np.random.seed(seed)
    m = make(name).to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4); scaler = torch.amp.GradScaler('cuda', enabled=(dev == 'cuda'))
    w = torch.tensor(len(ytr) / (3 * np.maximum(np.bincount(ytr, minlength=3), 1)), dtype=torch.float32, device=dev)
    xt = torch.from_numpy(Xtr if Xtr.ndim == 3 else Xtr[:, None]); yt = torch.from_numpy(ytr.astype('int64'))
    for _ in range(epochs):
        m.train()
        for b in torch.randperm(len(yt)).split(128):
            if len(b) < 2: continue
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
                loss = F.cross_entropy(m(xt[b].to(dev)), yt[b].to(dev), weight=w)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    m.eval(); out = []; xe = torch.from_numpy(Xte if Xte.ndim == 3 else Xte[:, None])
    with torch.no_grad():
        for b in torch.arange(len(xe)).split(512): out.append(F.softmax(m(xe[b].to(dev)), 1).cpu().numpy())
    return np.concatenate(out), sum(p.numel() for p in m.parameters())


def protocols(lobo):
    yield 'L8_c0', 0, L8_TRAIN, L8_TEST, True
    yield 'L8_all', 0, L8_TRAIN, L8_TEST, False
    for k, tr in enumerate(itertools.combinations(range(5), 3)):
        yield 'L10', k, [c[i] for c in L10 for i in tr], [c[i] for c in L10 for i in range(5) if i not in tr], False
    yield 'A2R', 0, healthy[:3] + [b for b in single if org[b] == 'artificial'], healthy[3:] + [b for b in single if org[b] == 'real'], False
    if lobo:
        for k, b in enumerate(single): yield 'LOBO', k, [x for x in single if x != b], [b], False


def run(name, proto, fold, trb, teb, c0, seed):
    p = f'{CK}/{name}_{proto}_{fold}_s{seed}.npz'
    if os.path.exists(p): return dict(np.load(p, allow_pickle=True))
    if name == 'fo_resnet':
        tr, te = np.isin(PB, trb), np.isin(PB, teb)
        if c0: tr &= Pc == COND0; te &= Pc == COND0
        P, npar = fit_predict(name, ORD[tr], Py[tr], ORD[te], seed, a.epochs_order); yr, br = Py[te], PB[te]
    else:
        W = windows('raw' if name.endswith('raw') else 'env'); X = W['raw' if name.endswith('raw') else 'env']
        tr, te = np.isin(W['B'], trb), np.isin(W['B'], teb)
        if c0: tr &= W['cond'] == COND0; te &= W['cond'] == COND0
        pw, npar = fit_predict(name, X[tr], W['y'][tr], X[te], seed, a.epochs)
        ids, inv = np.unique(W['rec'][te], return_inverse=True); P = np.zeros((len(ids), 3)); np.add.at(P, inv, pw); P /= np.bincount(inv)[:, None]
        yr = np.array([W['y'][te][inv == i][0] for i in range(len(ids))]); br = np.array([W['B'][te][inv == i][0] for i in range(len(ids))])
    r = {'P': P.astype(np.float32), 'y': yr, 'b': br, 'npar': npar}; np.savez_compressed(p, **r); return r


RES = {}; models = [m.strip() for m in a.models.split(',') if m.strip()]; seeds = [int(x) for x in a.seeds.split(',')]; l10_seeds = [int(x) for x in a.l10_seeds.split(',')]
for name in models:
    for proto, fold, trb, teb, c0 in protocols(not a.no_lobo):
        for s in (seeds if proto in ('L8_c0', 'L8_all', 'A2R') else (l10_seeds if proto == 'L10' else seeds[:1])):
            r = run(name, proto, fold, trb, teb, c0, s); ok = r['P'].argmax(1) == r['y']; h = r['y'] == 0
            RES.setdefault(name, {}).setdefault(proto, []).append({'fold': fold, 'seed': s, 'acc': float(ok.mean()), 'fa': float((r['P'][h].argmax(1) != 0).mean()) if h.any() else None,
                                                                  'per_bearing': {str(b): float(ok[r['b'] == b].mean()) for b in set(r['b'])}, 'npar': int(r['npar'])})
        accs = [x['acc'] for x in RES[name][proto]]
        print(f'  {name:10s} {proto:6s}: {100 * np.mean(accs):5.1f}  (n = {len(accs)}, {time.time() - t0:.0f} s)', flush=True)
    json.dump(RES, open(R + '/phase23_results.json', 'w'), indent=1)

print('\nRecording accuracy (%) / healthy false alarms (%); L8, A2R: mean +- SD over 3 seeds; L10: over the 10 splits; LOBO: pooled recordings, seed 0')
for name in models:
    row = f'{name:10s} ({RES[name]["L8_c0"][0]["npar"]:>7d} par.)'
    for proto in ('L8_c0', 'L8_all', 'L10', 'A2R', 'LOBO'):
        if proto not in RES[name]: continue
        R_ = RES[name][proto]; acc = [x['acc'] for x in R_]; fa = [x['fa'] for x in R_ if x['fa'] is not None]
        if proto == 'LOBO':
            pb = {}
            for x in R_: pb.update(x['per_bearing'])
            n_rec = {b: (PB == b).sum() for b in pb}; pooled = sum(pb[b] * n_rec[b] for b in pb) / sum(n_rec.values())
            row += f' | LOBO {100 * pooled:5.1f} (bearing mean {100 * np.mean(list(pb.values())):5.1f})'
        else:
            row += f' | {proto} {100 * np.mean(acc):5.1f} +- {100 * np.std(acc):4.1f} / {100 * np.mean(fa):4.1f}'
    print(row)
print('ReliSense (phase 19): L8_c0 95.9, L8_all 87.0, L10 78.7 +- 11.8, A2R 77.7, LOBO 83.7;  WDCNN env (phase 4): 70.8, 59.9, 59.6, 69.0, 68.1')
print(f'written {R}/phase23_results.json  ({time.time() - t0:.0f} s)')
