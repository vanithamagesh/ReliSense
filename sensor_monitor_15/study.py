"""ReliSense full study: bearing-rotation protocol, baselines, ablations, evaluation.

One job = one model trained and tested on one bearing rotation with one seed
(optionally with one operating condition held out). Every job writes its own
folder, so an interrupted Colab session resumes where it stopped.

Split (per class, bearings sorted by name, rotation r = 0..4), v1.3 default:
    test bearing        = b[r]   (never seen in training, validation or calibration)
    the other 4 bearings: recordings split 3:1:1 per condition into training / validation / calibration
(--val-mode bearing restores v1.2: one separate bearing for validation + calibration, three for training)
--protocol art2real (v1.4): train on healthy K001-K003 and the artificial damages, test on K004, K005 and the
    real damages (Lessmeier et al. 2016); validation/calibration from held-out recordings of the training bearings.
v1.4 adds kinematic physics features (envelope-spectrum peaks at the 6203 fault frequencies), the physics_lr
baseline and ReliSense-P (ReliSense + physics evidence gated by the learned vibration-quality score).
With --heldout COND the training/validation/calibration parts drop COND and the
test part keeps only COND (joint bearing + operating-condition holdout).
"""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, recall_score, roc_auc_score)
import relisense as R

CLASSES = ['healthy', 'outer_ring', 'inner_ring']
SENSOR_SETS = {'currents_only': [0, 1, 1], 'vib_cur1': [1, 1, 0], 'vib_only': [1, 0, 0]}
TRAIN_KINDS = ['noise', 'bias', 'gain', 'drift', 'clip']
TEST_ONLY_KINDS = ['spike', 'quant']          # never used during training
LEVELS = [0.25, 0.5, 1.0]
ART2REAL_TEST = ['K004', 'K005', 'KA04', 'KA15', 'KI04', 'KI14']
GEOMETRY_6203 = dict(n=8, d=6.75, D=28.55)
MODELS = ['relisense', 'relisense_p', 'physics_lr', 'relisense_wn', 'relisense_uniform', 'relisense_norec', 'relisense_noq',
          'transformer', 'wdcnn', 'tcn', 'envelope_rf']


# ----------------------------------------------------------------------------- split
def rotation_split(data, rotation, heldout=None, val_mode='recordings', protocol='rotation'):
    """Window indices of (train, validation, calibration, test) for one bearing rotation.

    val_mode='recordings' (default, v1.3): per class the test bearing is b[r]; the other four bearings
        are used for training, and within each of them and each condition the recordings are split
        3:1:1 (numeric order j: j % 5 == 3 -> validation, j % 5 == 4 -> calibration, else training).
    val_mode='bearing' (v1.2): test b[r]; one bearing b[r+1] for validation + calibration; three for training.
    """
    groups, labels, cond, rec = data['bearing'], data['y'], data['condition'], data['recording']
    num = lambda s: int(str(s).rsplit('_', 1)[1])
    parts = [[], [], [], []]                     # train, validation, calibration, test (window indices)
    for label in sorted(np.unique(labels)):
        gids = sorted(np.unique(groups[labels == label]).tolist())
        n = len(gids)
        if n < 3:
            raise ValueError(f'class {label} needs >=3 bearings, got {n}')
        if protocol == 'art2real':
            tests = [g for g in gids if g in ART2REAL_TEST]
        else:
            tests = [gids[rotation % n]]
        test = tests[0]
        parts[3] += np.flatnonzero(np.isin(groups, tests)).tolist()
        if val_mode == 'bearing':
            if protocol != 'rotation':
                raise ValueError("val_mode 'bearing' needs protocol 'rotation'")
            vc = gids[(rotation + 1) % n]
            train = [g for g in gids if g not in (test, vc)]
            parts[0] += np.flatnonzero(np.isin(groups, train)).tolist()
            for c in np.unique(cond):
                recs = sorted(np.unique(rec[(groups == vc) & (cond == c)]), key=num)
                for j, r in enumerate(recs):
                    parts[1 if j % 2 == 0 else 2] += np.flatnonzero(rec == r).tolist()
        else:
            for g in [g for g in gids if g not in tests]:
                for c in np.unique(cond):
                    recs = sorted(np.unique(rec[(groups == g) & (cond == c)]), key=num)
                    for j, r in enumerate(recs):
                        k = j % 5 if len(recs) >= 5 else {len(recs) - 2: 3, len(recs) - 1: 4}.get(j, 0)
                        parts[{3: 1, 4: 2}.get(k, 0)] += np.flatnonzero(rec == r).tolist()
    out = []
    for j, idx in enumerate(parts):
        idx = np.array(sorted(idx))
        if heldout:
            idx = idx[cond[idx] == heldout] if j == 3 else idx[cond[idx] != heldout]
        if len(idx) == 0:
            raise ValueError('empty partition')
        out.append(idx)
    # leakage checks: the test bearing appears nowhere else; no recording is in two parts
    for a in range(4):
        for b in range(a):
            assert not set(rec[out[a]]) & set(rec[out[b]])
            if 3 in (a, b):
                assert not set(groups[out[a]]) & set(groups[out[b]])
    return out


# ----------------------------------------------------------------------------- baselines
class Base(nn.Module):
    """Common interface with ReliSense: forward(x, available) -> (logits, None, None, None)."""
    def forward(self, x, available):
        return self.net(x * available[:, :, None]), None, None, None


class WDCNN(Base):
    """Wide first-layer kernel 1D CNN (Zhang et al., Sensors 2017), early fusion of the 3 channels."""
    def __init__(self, k, nclasses):
        super().__init__()
        def blk(i, o, kk, s=1, p=1):
            return [nn.Conv1d(i, o, kk, s, p), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2)]
        self.net = nn.Sequential(*blk(k, 16, 64, 16, 24), *blk(16, 32, 3), *blk(32, 64, 3), *blk(64, 64, 3),
                                 *blk(64, 64, 3), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                 nn.Linear(64, 100), nn.ReLU(), nn.Dropout(.2), nn.Linear(100, nclasses))


class TCNBlock(nn.Module):
    def __init__(self, i, o, d):
        super().__init__()
        self.c1 = nn.Conv1d(i, o, 3, padding=d, dilation=d); self.c2 = nn.Conv1d(o, o, 3, padding=d, dilation=d)
        self.n1 = nn.BatchNorm1d(o); self.n2 = nn.BatchNorm1d(o); self.drop = nn.Dropout(.15)
        self.skip = nn.Conv1d(i, o, 1) if i != o else nn.Identity()
    def forward(self, x):
        h = self.drop(F.relu(self.n1(self.c1(x))))
        return F.relu(self.n2(self.c2(h)) + self.skip(x))


class TCN(Base):
    def __init__(self, k, nclasses):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(k, 32, 16, 8, 4), nn.BatchNorm1d(32), nn.ReLU(),
                                 TCNBlock(32, 32, 1), TCNBlock(32, 64, 2), TCNBlock(64, 64, 4), TCNBlock(64, 64, 8),
                                 nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(.2), nn.Linear(64, nclasses))


class TemporalTransformer(Base):
    """Same stem and encoder size as ReliSense, but early fusion and no reliability weighting."""
    def __init__(self, k, nclasses, d=48):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(k, 24, 15, 4, 7), nn.BatchNorm1d(24), nn.GELU(),
                                  nn.Conv1d(24, d, 9, 4, 4), nn.BatchNorm1d(d), nn.GELU())
        layer = nn.TransformerEncoderLayer(d, 4, 128, .15, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(.2), nn.Linear(d, nclasses))
        self.d = d
    def net(self, x):
        t = self.stem(x).transpose(1, 2)
        pos = torch.arange(t.shape[1], device=x.device)[:, None]
        freq = torch.exp(torch.arange(0, self.d, 2, device=x.device) * (-math.log(10000.) / self.d))
        pe = torch.zeros(t.shape[1], self.d, device=x.device)
        pe[:, 0::2] = torch.sin(pos * freq); pe[:, 1::2] = torch.cos(pos * freq)
        return self.head(self.enc(t + pe[None]).mean(1))


def build(name, k, length, nclasses, backbone='compact'):
    name = name.split('+')[0]
    if name == 'relisense_p':
        name = 'relisense'
    if name.startswith('relisense'):
        model = R.ReliSense(k, length, nclasses, backbone, uniform=name.startswith('relisense_uniform'))
        # BatchNorm in the stem keeps activations at unit scale; without it the time-averaged
        # features of a zero-mean signal are close to zero and training stalls at chance level.
        stem = [nn.Conv1d(1, 24, 15, 4, 7), nn.BatchNorm1d(24), nn.GELU(),
                nn.Conv1d(24, 48, 9, 4, 4), nn.BatchNorm1d(48), nn.GELU()]
        if name == 'relisense_wn':
            # per-window standardisation of the classification path only; the reliability head still
            # receives the raw amplitude statistics, so offsets and gain changes stay detectable
            stem = [nn.InstanceNorm1d(1)] + stem
        model.local = nn.Sequential(*stem)
        return model
    return {'wdcnn': WDCNN, 'tcn': TCN, 'transformer': TemporalTransformer}[name](k, nclasses)


# ----------------------------------------------------------------------------- envelope + RF
def envelope_features(x, fs):
    """x: [n, k, L] numpy. Time statistics, log band energies of the spectrum and of the envelope spectrum."""
    n, k, L = x.shape
    freqs = np.fft.rfftfreq(L, 1 / fs)
    spec_edges = np.geomspace(10, fs / 2, 25)
    env_edges = np.linspace(2, 500, 25)
    feats = []
    for c in range(k):
        s = x[:, c]
        rms = np.sqrt((s ** 2).mean(1))
        stats = [s.mean(1), s.std(1), rms, np.abs(s).mean(1), s.max(1) - s.min(1),
                 np.abs(s).max(1) / np.maximum(rms, 1e-8),
                 np.nan_to_num(kurtosis(s, axis=1)), np.nan_to_num(skew(s, axis=1))]
        P = np.abs(np.fft.rfft(s - s.mean(1, keepdims=True), axis=1)) ** 2
        env = np.abs(hilbert(s - s.mean(1, keepdims=True), axis=1))
        E = np.abs(np.fft.rfft(env - env.mean(1, keepdims=True), axis=1)) ** 2
        bands = [np.log1p(P[:, (freqs >= a) & (freqs < b)].sum(1)) for a, b in zip(spec_edges[:-1], spec_edges[1:])]
        ebands = [np.log1p(E[:, (freqs >= a) & (freqs < b)].sum(1)) for a, b in zip(env_edges[:-1], env_edges[1:])]
        feats += stats + bands + ebands
    return np.stack(feats, 1).astype('float32')


# ----------------------------------------------------------------------------- physics (kinematic) features
def fault_orders(geom=GEOMETRY_6203):
    q = geom['d'] / geom['D']; n = geom['n']
    return {'BPFO': n / 2 * (1 - q), 'BPFI': n / 2 * (1 + q), 'BSF': geom['D'] / (2 * geom['d']) * (1 - q ** 2),
            'FTF': 0.5 * (1 - q), 'shaft': 1.0}


def physics_features(vib, rpm, fs, orders=None):
    """15 features of one recording: log peak-to-background ratio of the envelope spectrum at harmonics 1-3 of
    BPFO, BPFI, BSF, FTF and the shaft frequency (+-2 %, at least 1.5 bins; background = median within +-10 Hz).
    vib: [windows, L] consecutive windows of the vibration (envelope) channel. Scale- and offset-invariant."""
    orders = orders or fault_orders()
    s = vib.reshape(-1).astype(float); s = s - s.mean()
    if not np.any(s):
        return np.zeros(3 * len(orders))
    S = np.abs(np.fft.rfft(s * np.hanning(len(s)))); f = np.fft.rfftfreq(len(s), 1 / fs); fr = rpm / 60
    row = []
    for o in orders.values():
        for h in (1, 2, 3):
            f0 = h * o * fr; w = max(0.02 * f0, 1.5 * (f[1] - f[0])); band = np.abs(f - f0) <= w
            bg = (np.abs(f - f0) <= 10) & ~band
            row.append(np.log((S[band].max() + 1e-12) / (np.median(S[bg]) + 1e-12)))
    return np.array(row)


def recording_physics(xv, recs, rpm, fs):
    """Physics features per recording. xv: [windows, L] vibration; returns (unique recording ids, features)."""
    ids = np.unique(recs)
    return ids, np.stack([physics_features(xv[recs == r], float(rpm[recs == r][0]), fs) for r in ids])


def fit_temperature_np(p, y):
    """Temperature on recording-level probabilities (validation NLL, log grid 0.05-20)."""
    lp = np.log(np.clip(p, 1e-9, 1)); grid = np.geomspace(.05, 20, 241); best = (np.inf, 1.)
    for t in grid:
        z = lp / t; z = z - z.max(1, keepdims=True); q = np.exp(z); q /= q.sum(1, keepdims=True)
        nll = -np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1)).mean()
        if nll < best[0]: best = (nll, t)
    return float(best[1])


def apply_temperature_np(p, t):
    z = np.log(np.clip(p, 1e-9, 1)) / t; z = z - z.max(1, keepdims=True); q = np.exp(z)
    return q / q.sum(1, keepdims=True)


def to_windows(p_rec, ids, recs):
    """Copy recording-level probabilities to every window of the recording."""
    pos = {r: i for i, r in enumerate(ids)}
    return p_rec[[pos[r] for r in recs]]


# ----------------------------------------------------------------------------- corruption for evaluation
def corrupt_eval(x, kind, sensor, level, seed):
    """Deterministic corruption of every window on one sensor (or a sensor-set mask). Units: training SDs."""
    g = torch.Generator().manual_seed(seed)
    out = x.clone(); b, k, l = x.shape
    avail = torch.ones(b, k); good = torch.ones(b, k)
    if kind == 'clean':
        return out, avail, good
    if kind == 'set':
        m = torch.tensor(sensor, dtype=torch.float32)
        out = out * m[None, :, None]; avail = avail * m; good = good * m
        return out, avail, good
    s = sensor
    if kind == 'noise': out[:, s] += level * torch.randn(b, l, generator=g)
    elif kind == 'bias': out[:, s] += level
    elif kind == 'gain': out[:, s] *= (1 + level)
    elif kind == 'drift': out[:, s] += torch.linspace(0, level, l)
    elif kind == 'clip': out[:, s] = out[:, s].clamp(-level, level)
    elif kind == 'missing': out[:, s] = 0; avail[:, s] = 0
    elif kind == 'spike':                         # test-only: sparse large impulses
        hit = (torch.rand(b, l, generator=g) < .002).float()
        sign = torch.where(torch.rand(b, l, generator=g) < .5, -1., 1.)
        out[:, s] += 8 * level * hit * sign
    elif kind == 'quant':                         # test-only: coarse ADC quantisation
        out[:, s] = torch.round(out[:, s] / level) * level
    else:
        raise ValueError(kind)
    good[:, s] = 0
    return out, avail, good


def scenarios():
    sc = [('clean', None, 0.)]
    sc += [('set', name, 0.) for name in SENSOR_SETS]
    sc += [('missing', s, 0.) for s in range(3)]
    sc += [(kd, s, lv) for kd in TRAIN_KINDS for s in range(3) for lv in LEVELS]
    sc += [(kd, s, lv) for kd in TEST_ONLY_KINDS for s in range(3) for lv in (.5, 1.)]
    return sc


def scen_name(kind, sensor, level):
    if kind == 'clean': return 'clean'
    if kind == 'set': return f'set_{sensor}'
    if kind == 'missing': return f'missing_{sensor}'
    return f'{kind}_{sensor}_{level:g}'


# ----------------------------------------------------------------------------- metrics
def ece_score(p, y, bins=10):
    conf = p.max(1); correct = p.argmax(1) == y; e = 0.
    for a, b in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > a) & (conf <= b)
        if m.any(): e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def cls_metrics(p, y, nclasses=3):
    pred = p.argmax(1)
    return dict(accuracy=float(accuracy_score(y, pred)), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average='macro', labels=range(nclasses), zero_division=0)),
                recall=recall_score(y, pred, average=None, labels=range(nclasses), zero_division=0).tolist(),
                ece=ece_score(p, y), nll=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean()),
                brier=float(((p - np.eye(nclasses)[y]) ** 2).sum(1).mean()),
                confusion_matrix=confusion_matrix(y, pred, labels=range(nclasses)).tolist())


def risk_at(p, y, cov):
    order = np.argsort(-p.max(1)); n = max(1, int(round(cov * len(y))))
    return float((p.argmax(1)[order[:n]] != y[order[:n]]).mean())


def aurc(p, y):
    order = np.argsort(-p.max(1)); err = (p.argmax(1)[order] != y[order]).astype(float)
    return float((np.cumsum(err) / np.arange(1, len(err) + 1)).mean())


def aggregate(prob, y, recs):
    ids = np.unique(recs); ps = []; ys = []
    for r in ids:
        m = recs == r; ps.append(prob[m].mean(0)); ys.append(y[m][0])
    return np.array(ps), np.array(ys), ids


# ----------------------------------------------------------------------------- torch training
def batches(n, bs, shuffle, device):
    order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    return order.split(bs)


@torch.no_grad()
def predict_torch(model, x, avail, bs=128, mc=1, temperature=1.):
    """Window probabilities; MC dropout switches on Dropout layers only (BatchNorm stays in eval)."""
    model.eval()
    if mc > 1:
        for m in model.modules():
            if isinstance(m, nn.Dropout): m.train()
    P, W, Q, MI = [], [], [], []
    for sl in torch.arange(len(x)).split(bs):
        z = x[sl]; a = avail[sl]; ps = []; ws = []; qs = []
        for _ in range(mc):
            logits, q, _, w = model(z, a)
            ps.append(F.softmax(logits / temperature, -1))
            if w is not None: ws.append(w); qs.append(torch.sigmoid(q))
        p = torch.stack(ps); mean = p.mean(0)
        h = lambda t: -(t * t.clamp_min(1e-9).log()).sum(-1)
        MI.append((h(mean) - h(p).mean(0)).clamp_min(0).cpu())
        P.append(mean.cpu())
        if ws: W.append(torch.stack(ws).mean(0).cpu()); Q.append(torch.stack(qs).mean(0).cpu())
    model.eval()
    cat = lambda L: torch.cat(L).numpy() if L else None
    return cat(P), cat(W), cat(Q), cat(MI)


@torch.no_grad()
def val_logits(model, x, bs=128):
    model.eval(); one = torch.ones(x.shape[:2], device=x.device)
    return torch.cat([model(x[s], one[s])[0] for s in torch.arange(len(x)).split(bs)])


def fit_temperature(logits, y):
    """Single temperature minimising validation NLL over a wide log grid (0.05-20, 241 values)."""
    grid = np.geomspace(.05, 20, 241)
    nll = [float(F.cross_entropy(logits / t, y)) for t in grid]
    return float(grid[int(np.nanargmin(nll))])


def train_torch(name, model, sets, cfg, fs, geometry):
    name = name.split('+')[0]; name = 'relisense' if name == 'relisense_p' else name; is_reli = name.startswith('relisense')
    w_rec = 0. if name == 'relisense_norec' else .3
    w_q = 0. if name == 'relisense_noq' else .2
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-4)
    stages = ([('pretrain', cfg.pretrain)] if is_reli else []) + [('supervised', cfg.epochs)]
    total = sum(n for _, n in stages) * math.ceil(len(sets[0][0]) / cfg.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=total, pct_start=.1)
    xtr, ytr, rpm = sets[0]; xva, yva = sets[1][0], sets[1][1]
    history = []; best = (-1., float('inf')); best_state = None; bad = 0
    for stage, n_ep in stages:
        for ep in range(n_ep):
            model.train(); losses = []; t0 = time.time()
            for sl in batches(len(xtr), cfg.batch, True, xtr.device):
                clean, target = xtr[sl], ytr[sl]
                z, avail, good = R.corrupt(clean, 'mixed', level=random.uniform(.15, 1.))
                keep = torch.rand(len(z), device=z.device) < cfg.p_clean      # some windows stay clean
                z[keep] = clean[keep]; avail[keep] = 1; good[keep] = 1
                if stage == 'pretrain':
                    z = z * (torch.rand_like(z) > .15)
                logits, q, rec, _ = model(z, avail)
                loss = torch.zeros((), device=z.device)
                if is_reli:
                    loss = loss + w_rec * F.mse_loss(rec, clean) + w_q * F.binary_cross_entropy_with_logits(q, good)
                    if geometry is not None:
                        loss = loss + cfg.physics_weight * R.physics_loss(rec, clean, rpm[sl], fs, geometry, 'raw')
                if stage == 'supervised':
                    loss = loss + F.cross_entropy(logits, target)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.); opt.step(); sched.step()
                losses.append(float(loss.detach()))
            lg = val_logits(model, xva); p = F.softmax(lg, -1).cpu().numpy()
            m = cls_metrics(p, yva.cpu().numpy())
            history.append(dict(stage=stage, epoch=ep + 1, loss=float(np.mean(losses)), seconds=time.time() - t0,
                                val_macro_f1=m['macro_f1'], val_accuracy=m['accuracy'], val_nll=m['nll']))
            print(f'  {stage} {ep + 1}/{n_ep} loss={np.mean(losses):.4f} val_f1={m["macro_f1"]:.4f} '
                  f'val_nll={m["nll"]:.4f} ({time.time() - t0:.1f}s)', flush=True)
            if stage == 'supervised':
                key = (m['macro_f1'], -m['nll'])
                if key > (best[0], -best[1]):
                    best = (m['macro_f1'], m['nll']); bad = 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                if bad >= cfg.patience:
                    print(f'  early stop at epoch {ep + 1}', flush=True); break
    model.load_state_dict(best_state)
    return history


# ----------------------------------------------------------------------------- one job
def run_job(cfg, data, model_name, rotation, seed, heldout, out):
    R.seed_all(seed); torch.backends.cudnn.benchmark = True
    device = torch.device(cfg.device)
    X = data['X']; y = data['y'].astype('int64'); fs = float(data['fs'])
    parts = rotation_split(data, rotation, heldout, cfg.val_mode, cfg.protocol)
    recs_all, rpm_all = data['recording'], data['rpm']
    names = ['train', 'validation', 'calibration', 'test']
    split = {n: {'bearings': np.unique(data['bearing'][i]).tolist(), 'recordings': int(len(np.unique(data['recording'][i]))),
                 'windows': int(len(i))} for n, i in zip(names, parts)}
    mean = X[parts[0]].mean((0, 2), keepdims=True); std = X[parts[0]].std((0, 2), keepdims=True).clip(1e-6)
    Xn = ((X - mean) / std).astype('float32')
    nclasses = int(y.max() + 1); k, L = X.shape[1], X.shape[2]
    t_start = time.time(); history = []; model = None; rf = None; temperature = 1.

    if model_name == 'envelope_rf':
        feats = {j: envelope_features(Xn[parts[j]], fs) for j in (0, 1, 2)}
        rf = RandomForestClassifier(500, n_jobs=-1, random_state=seed, min_samples_leaf=2)
        rf.fit(feats[0], y[parts[0]])
        def prob_fn(xc, av, recs, rpm):
            return rf.predict_proba(envelope_features(xc.numpy(), fs)), None, None, None
        val_p = rf.predict_proba(feats[1]); cal_p = rf.predict_proba(feats[2])
        params = dict(trainable=None, frozen=None, note='random forest, 500 trees')

    if model_name in ('physics_lr', 'relisense_p'):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        phys = {}
        for j in (0, 1, 2):
            ids, Fp = recording_physics(Xn[parts[j], 0], recs_all[parts[j]], rpm_all[parts[j]], fs)
            phys[j] = (ids, Fp, np.array([y[parts[j]][recs_all[parts[j]] == r][0] for r in ids]))
        plr = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)).fit(phys[0][1], phys[0][2])
        t_phys = fit_temperature_np(plr.predict_proba(phys[1][1]), phys[1][2])

        def phys_prob(xc, av, recs, rpm):
            """recording-level physics probabilities from the (possibly corrupted) vibration channel"""
            ids, Fp = recording_physics(xc[:, 0].numpy(), recs, rpm, fs)
            return ids, apply_temperature_np(plr.predict_proba(Fp), t_phys)

    if model_name == 'physics_lr':
        def prob_fn(xc, av, recs, rpm):
            ids, pp = phys_prob(xc, av, recs, rpm)
            return to_windows(pp, ids, recs), None, None, None
        temperature = t_phys
        val_p = prob_fn(torch.tensor(Xn[parts[1]]), None, recs_all[parts[1]], rpm_all[parts[1]])[0]
        cal_p = prob_fn(torch.tensor(Xn[parts[2]]), None, recs_all[parts[2]], rpm_all[parts[2]])[0]
        params = dict(trainable=int(plr[-1].coef_.size + plr[-1].intercept_.size), frozen=0,
                      note='logistic regression on 15 kinematic features')
    elif model_name != 'envelope_rf':
        model = build(model_name, k, L, nclasses, cfg.backbone).to(device)
        to = lambda idx: (torch.tensor(Xn[idx], device=device), torch.tensor(y[idx], device=device),
                          torch.tensor(data['rpm'][idx], device=device))
        sets = [to(p) for p in parts[:3]]
        geometry = json.loads(Path(cfg.geometry).read_text()) if cfg.geometry else None
        history = train_torch(model_name, model, sets, cfg, fs, geometry)
        temperature = fit_temperature(val_logits(model, sets[1][0]), sets[1][1])

        def learned_fn(xc, av):
            return predict_torch(model, xc.to(device), av.to(device), mc=cfg.mc, temperature=temperature)

        if model_name == 'relisense_p':
            def prob_fn(xc, av, recs, rpm):
                """ReliSense-P: log p = g * log p_physics + (1 - g) * log p_learned per recording, where the gate g is
                the mean learned quality of the vibration channel (0 when the channel is unavailable)."""
                p, w, q, mi = learned_fn(xc, av)
                ids, pp = phys_prob(xc, av, recs, rpm)
                pl, _, _ = aggregate(p, np.zeros(len(p), int), recs)
                qv = q[:, 0] * av[:, 0].numpy()
                g = np.array([qv[recs == r].mean() for r in ids])[:, None]
                z = g * np.log(np.clip(pp, 1e-9, 1)) + (1 - g) * np.log(np.clip(pl, 1e-9, 1))
                z = np.exp(z - z.max(1, keepdims=True)); z /= z.sum(1, keepdims=True)
                return to_windows(z, ids, recs), w, q, mi
        else:
            def prob_fn(xc, av, recs, rpm):
                return learned_fn(xc, av)
        one = lambda s: torch.ones(s[0].shape[:2])
        val_p = prob_fn(sets[1][0].cpu(), one(sets[1]), recs_all[parts[1]], rpm_all[parts[1]])[0]
        cal_p = prob_fn(sets[2][0].cpu(), one(sets[2]), recs_all[parts[2]], rpm_all[parts[2]])[0]
        params = dict(trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                      decoder=sum(p.numel() for p in model.decode.parameters()) if hasattr(model, 'decode') else 0)
        del sets
    train_seconds = time.time() - t_start

    # conformal threshold from calibration recordings; rejection threshold from validation recordings
    cp, cy, _ = aggregate(cal_p, y[parts[2]], data['recording'][parts[2]])
    nc = 1 - cp[np.arange(len(cy)), cy]; n = len(cy); rank = math.ceil((n + 1) * (1 - cfg.alpha))
    qhat = float(np.sort(nc)[rank - 1]) if rank <= n else float('inf')
    vp, _, _ = aggregate(val_p, y[parts[1]], data['recording'][parts[1]])
    reject_thr = float(np.quantile(vp.max(1), .1))

    test_idx = parts[3]; xt = torch.tensor(Xn[test_idx]); yt = y[test_idx]; rt = data['recording'][test_idx]
    rpm_t = rpm_all[test_idx]
    rows = []; store = {}; t_eval = time.time()
    for si, (kind, sensor, level) in enumerate(scenarios()):
        name = scen_name(kind, sensor, level)
        sens = SENSOR_SETS[sensor] if kind == 'set' else sensor
        xc, av, good = corrupt_eval(xt, kind, sens, level, seed=10_000 * rotation + si)  # same for every model
        p, w, q, mi = prob_fn(xc, av, rt, rpm_t)
        ap, ay, rids = aggregate(p, yt, rt)
        sets_pred = (1 - ap) <= qhat; acc = ap.max(1) >= reject_thr
        row = dict(scenario=name, kind=kind, sensor=sensor if kind != 'set' else sensor, level=level,
                   recordings=int(len(ay)), **cls_metrics(ap, ay, nclasses),
                   window_accuracy=float((p.argmax(1) == yt).mean()),
                   window_macro_f1=float(f1_score(yt, p.argmax(1), average='macro', zero_division=0)),
                   coverage=float(sets_pred[np.arange(len(ay)), ay].mean()), set_size=float(sets_pred.sum(1).mean()),
                   accepted_fraction=float(acc.mean()),
                   accepted_error=float((ap.argmax(1)[acc] != ay[acc]).mean()) if acc.any() else None,
                   accepted_recall=[float((ap.argmax(1)[acc & (ay == c)] == c).mean()) if (acc & (ay == c)).any() else None
                                    for c in range(nclasses)],
                   risk_at_80=risk_at(ap, ay, .8), risk_at_90=risk_at(ap, ay, .9), aurc=aurc(ap, ay))
        if w is not None:
            row['mean_sensor_weights'] = w.mean(0).tolist(); row['mean_mc_disagreement'] = float(mi.mean())
            bad = (1 - good.numpy()).astype(bool); okmask = av.numpy().astype(bool)
            score = 1 - q
            row['quality_false_alarm'] = float((q[okmask & ~bad] < .5).mean())
            if kind not in ('clean', 'set', 'missing'):
                row['quality_detection'] = float((q[bad] < .5).mean())
                row['sensor_isolation_auroc'] = float(roc_auc_score(bad[okmask].ravel(), score[okmask].ravel()))
            store[f'q__{name}'] = q.astype('float16')
            store[f'w__{name}'] = w.astype('float16')
        store[f'p__{name}'] = ap.astype('float32')
        rows.append(row)

    # computational cost: one recording (all its windows), warmed, mc samples included
    comp = dict(params=params, train_seconds=train_seconds, eval_seconds=time.time() - t_eval, epochs_run=len(history))
    one_rec = rt == rt[0]; xr = xt[one_rec]; ar = torch.ones(xr.shape[:2])
    for _ in range(3): prob_fn(xr, ar, rt[one_rec], rpm_t[one_rec])
    if device.type == 'cuda' and model is not None:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter(); reps = 20
    for _ in range(reps): prob_fn(xr, ar, rt[one_rec], rpm_t[one_rec])
    if device.type == 'cuda' and model is not None:
        torch.cuda.synchronize(); comp['peak_gpu_mb'] = torch.cuda.max_memory_allocated() / 2 ** 20
    comp['latency_ms_per_recording'] = 1000 * (time.perf_counter() - t0) / reps
    comp['windows_per_recording'] = int(one_rec.sum()); comp['mc_samples'] = cfg.mc if model is not None else 1
    comp['includes_physics_features'] = model_name in ('physics_lr', 'relisense_p')
    comp['device'] = torch.cuda.get_device_name() if device.type == 'cuda' and model is not None else 'cpu'

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'preds.npz', y_rec=ay, rec_ids=rids,
                        rec_bearing=np.array([data['bearing'][test_idx][rt == r][0] for r in rids]),
                        rec_condition=np.array([data['condition'][test_idx][rt == r][0] for r in rids]), **store)
    (out / 'split.json').write_text(json.dumps(split, indent=2))
    (out / 'history.json').write_text(json.dumps(history, indent=2))
    if model is not None:
        torch.save({'state_dict': model.state_dict(), 'mean': mean, 'std': std, 'temperature': temperature,
                    'model': model_name}, out / 'checkpoint.pt')
    meta = dict(model=model_name, protocol=cfg.protocol, rotation=rotation, seed=seed, heldout=heldout, temperature=temperature,
                conformal_qhat=qhat if math.isfinite(qhat) else None, reject_threshold=reject_thr,
                computation=comp, config=vars(cfg), split=split, scenarios=rows)
    (out / 'metrics.json').write_text(json.dumps(meta, indent=2))   # written last = job finished
    clean = rows[0]
    print(f'DONE {out.name}: clean acc={clean["accuracy"]:.3f} macroF1={clean["macro_f1"]:.3f} '
          f'T={temperature:.2f} train={train_seconds / 60:.1f} min eval={comp["eval_seconds"] / 60:.1f} min', flush=True)
    return meta


def job_tag(model, rotation, seed, heldout, protocol='rotation'):
    split = 'a2r' if protocol == 'art2real' else f'rot{rotation}'
    return f'{model}__{split}__seed{seed}__{heldout or "none"}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True); ap.add_argument('--out', default='study_runs')
    ap.add_argument('--models', default='relisense,transformer,wdcnn,tcn,envelope_rf')
    ap.add_argument('--rotations', default='0,1,2,3,4'); ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--heldout', default='none', help="'none' or a comma list of conditions, e.g. N09_M07_F10")
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--pretrain', type=int, default=10)
    ap.add_argument('--patience', type=int, default=10); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--p-clean', type=float, default=.3)
    ap.add_argument('--mc', type=int, default=5); ap.add_argument('--alpha', type=float, default=.1)
    ap.add_argument('--backbone', default='compact', choices=['compact', 'moment'])
    ap.add_argument('--geometry'); ap.add_argument('--physics-weight', type=float, default=.1)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--max-jobs', type=int, default=0, help='stop after this many new jobs (0 = all)')
    ap.add_argument('--val-mode', default='recordings', choices=['recordings', 'bearing'],
                    help="recordings (v1.3): validation/calibration from held-out recordings of the training bearings")
    ap.add_argument('--protocol', default='rotation', choices=['rotation', 'art2real'],
                    help='rotation: one new bearing per class (5 rotations); art2real: artificial -> real damage (one split)')
    ap.add_argument('--suffix', default='', help="added to the model name of ReliSense jobs, e.g. '+phys' or '+moment'")
    cfg = ap.parse_args()
    data = dict(np.load(cfg.data, allow_pickle=False))
    print(f'data {data["X"].shape} fs={int(data["fs"])} bearings={len(np.unique(data["bearing"]))} '
          f'recordings={len(np.unique(data["recording"]))} device={cfg.device}', flush=True)
    held = [None] if cfg.heldout == 'none' else cfg.heldout.split(',')
    done = 0
    for h in held:
        for seed in map(int, cfg.seeds.split(',')):
            for rot in ([0] if cfg.protocol == 'art2real' else map(int, cfg.rotations.split(','))):
                for m in cfg.models.split(','):
                    m = m + cfg.suffix if m.startswith('relisense') else m
                    out = Path(cfg.out) / job_tag(m, rot, seed, h, cfg.protocol)
                    if (out / 'metrics.json').exists():
                        continue
                    print(f'=== {out.name}', flush=True)
                    run_job(cfg, data, m, rot, seed, h, out)
                    done += 1
                    if cfg.max_jobs and done >= cfg.max_jobs:
                        return


if __name__ == '__main__':
    main()
