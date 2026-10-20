"""Results of the final ReliSense (manuscript v3.26) for the figures: phases 17, 19, 20, 21 and 22 (JSON/NPZ in this folder).
Model keys: LR = kinematic core (15 features, stage 1), LRX = kinematic branch (31 features), RF = envelope branch (62 features),
ENSX = ReliSense (mean of the two branches). External rigs: RFenv is the envelope branch."""
import json
import numpy as np

P19 = json.load(open('phase19_results.json')); P20 = json.load(open('phase20_results.json'))
P21 = json.load(open('phase21_results.json')); P22 = json.load(open('phase22_numbers.json'))
FD = dict(np.load('phase22_figdata.npz', allow_pickle=True)); VW = dict(np.load('phase22_views.npz', allow_pickle=True))
PROTOS = ['L8_c0', 'L8_all', 'L10', 'A2R', 'LOBO']


def pad(model, proto):
    """Recording accuracy (%) and SD over the L10 splits (0 elsewhere) of a Paderborn model (phase 19)."""
    a = [100 * r['acc'] for r in P19['protocols'][proto][model]]
    return float(np.mean(a)), (float(np.std(a)) if len(a) > 1 else 0.0)


def pfa(model, proto):
    f = [100 * r['fa'] for r in P19['protocols'][proto][model] if r['fa'] is not None]
    return float(np.mean(f))


def ext(ds, proto, model):
    """External rig result (phase 21 for LR, LRX, RFenv, ENS, ENSX; phase 20 for the baselines)."""
    src = P21 if model in ('LR', 'LRX', 'RFenv', 'ENS', 'ENSX') else P20
    return src[ds]['results'][proto][model]


if __name__ == '__main__':
    for m in ('LR', 'LRX', 'RF', 'ENSX'):
        print(m, [f'{pad(m, p)[0]:.1f}±{pad(m, p)[1]:.1f}' for p in PROTOS], [f'{pfa(m, p):.1f}' for p in PROTOS])
