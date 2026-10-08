"""Phase 4 (v1.0): learned models on the 32-bearing windows, under the same bearing-wise protocols.

Question: with 32 bearings, do learned models generalise to new bearings as well as the physics model?
Model: WDCNN (Zhang et al., 2017; wide first-layer kernel 1-D CNN), one vibration channel per window.
Inputs: 'raw' = 8 kHz vibration window; 'env' = 2-12 kHz envelope window. Every window is standardised by
its own mean and standard deviation (removes gain and offset, a known bearing fingerprint).
Training (fixed in advance, nothing tuned on test bearings): AdamW lr 1e-3, weight decay 1e-4, batch 128,
20 epochs, class-weighted cross-entropy, last epoch is used. Recording prediction = mean window probability.
Protocols: L8 (condition 0 and all conditions), L10 (10 splits), A2R, LOBO (29 folds; --lobo to include).
Output: recording accuracy per protocol (mean over folds/seeds) and per-bearing accuracy; per-fold results saved as JSON.
"""
from __future__ import annotations
import argparse, itertools, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from phase2_physics import L8_TRAIN, L8_TEST, L10

COND0 = 'N15_M07_F10'


class WDCNN(nn.Module):
    def __init__(s, n_cls=3):
        super().__init__()
        def blk(i, o, k, st=1, p=1):
            return [nn.Conv1d(i, o, k, st, p), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2)]
        s.f = nn.Sequential(*blk(1, 16, 64, 16, 24), *blk(16, 32, 3), *blk(32, 64, 3), *blk(64, 64, 3),
                            *blk(64, 64, 3), nn.AdaptiveAvgPool1d(8), nn.Flatten())
        s.h = nn.Sequential(nn.Linear(512, 100), nn.ReLU(), nn.Dropout(0.3), nn.Linear(100, n_cls))

    def forward(s, x):
        return s.h(s.f(x))


def standardise(x):
    x = x - x.mean(1, keepdims=True); return (x / (x.std(1, keepdims=True) + 1e-8)).astype('float32')


def train_predict(Xtr, ytr, Xte, seed, dev, epochs=20, bs=128):
    torch.manual_seed(seed); np.random.seed(seed)
    m = WDCNN().to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    w = torch.tensor(len(ytr) / (3 * np.maximum(np.bincount(ytr, minlength=3), 1)), dtype=torch.float32, device=dev)
    xt = torch.from_numpy(Xtr).unsqueeze(1); yt = torch.from_numpy(ytr.astype('int64'))
    for ep in range(epochs):
        m.train()
        for b in torch.randperm(len(yt)).split(bs):
            loss = F.cross_entropy(m(xt[b].to(dev)), yt[b].to(dev), weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); out = []
    with torch.no_grad():
        for b in torch.arange(len(Xte)).split(1024):
            out.append(F.softmax(m(torch.from_numpy(Xte[b.numpy()]).unsqueeze(1).to(dev)), 1).cpu().numpy())
    return np.concatenate(out)


def rec_probs(pw, recs):
    ids, inv = np.unique(recs, return_inverse=True); P = np.zeros((len(ids), 3)); np.add.at(P, inv, pw)
    return ids, P / np.bincount(inv)[:, None]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--windows', default='pb32_4096.npz')
    ap.add_argument('--envelope', default='pb32_4096_envelope.npz')
    ap.add_argument('--physics', default='pb32_physics.npz')
    ap.add_argument('--inputs', default='raw,env')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--lobo', action='store_true')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--out', default='phase4_results.json')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; print('device', dev, flush=True)
    w = np.load(a.windows); B, y, cond, recs = w['bearing'], w['y'], w['condition'], w['recording']
    data = {'raw': standardise(w['X'][:, 0])} if 'raw' in a.inputs else {}
    if 'env' in a.inputs: data['env'] = standardise(np.load(a.envelope)['X'][:, 0])
    del w
    ph = np.load(a.physics); org = {b: o for b, o in zip(ph['bearing'], ph['origin'])}
    single = sorted(set(B)); healthy = [b for b in single if org[b] == 'healthy']
    prot = [('L8_cond0', 0, L8_TRAIN, L8_TEST, cond == COND0), ('L8_all', 0, L8_TRAIN, L8_TEST, None)]
    for k, tr in enumerate(itertools.combinations(range(5), 3)):
        prot.append(('L10', k, [c[i] for c in L10 for i in tr], [c[i] for c in L10 for i in range(5) if i not in tr], None))
    prot.append(('A2R', 0, healthy[:3] + [b for b in single if org[b] == 'artificial'],
                 healthy[3:] + [b for b in single if org[b] == 'real'], None))
    if a.lobo:
        prot += [('LOBO', k, [x for x in single if x != b], [b], None) for k, b in enumerate(single)]
    seeds = [int(s) for s in a.seeds.split(',')]; res = []; t0 = time.time()
    for inp, X in data.items():
        for name, fold, trb, teb, cm in prot:
            for s in (seeds if name != 'LOBO' else seeds[:1]):
                trm = np.isin(B, trb); tem = np.isin(B, teb)
                if cm is not None: trm &= cm; tem &= cm
                pw = train_predict(X[trm], y[trm], X[tem], s, dev, a.epochs)
                ids, P = rec_probs(pw, recs[tem]); yr = np.array([y[recs == r][0] for r in ids])
                br = np.array([B[recs == r][0] for r in ids]); ok = P.argmax(1) == yr
                res.append({'input': inp, 'protocol': name, 'fold': fold, 'seed': s, 'acc': float(ok.mean()),
                            'per_bearing': {b: float(ok[br == b].mean()) for b in teb}})
            print(f'  {inp} {name} fold {fold}: acc {np.mean([r["acc"] for r in res if r["input"] == inp and r["protocol"] == name and r["fold"] == fold]):.3f}'
                  f'  ({time.time() - t0:.0f} s)', flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print('\nSUMMARY (recording accuracy; mean over folds and seeds)')
    for inp in data:
        for name in dict.fromkeys(p[0] for p in prot):
            r = [x['acc'] for x in res if x['input'] == inp and x['protocol'] == name]
            if name == 'LOBO':
                pb = {}
                for x in res:
                    if x['input'] == inp and x['protocol'] == 'LOBO': pb.update(x['per_bearing'])
                print(f'  WDCNN-{inp:3s} {name:9s} mean of bearings {np.mean(list(pb.values())):.3f}   per bearing: '
                      + ', '.join(f'{b} {v:.2f}' for b, v in pb.items()))
            else:
                print(f'  WDCNN-{inp:3s} {name:9s} {np.mean(r):.3f} +- {np.std(r):.3f}  (n = {len(r)})')
    print('Physics primary (Phase 2/3, for comparison): L8 cond0 0.936, L8 all 0.828, L10 0.746, A2R 0.730, LOBO 0.801')


if __name__ == '__main__':
    main()
