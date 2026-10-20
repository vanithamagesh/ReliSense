"""Phase 11 (v1.1): analyses in the style of an IEEE TIM experimental section, on the Paderborn physics file (CPU).

A  Ablation study. The 15-feature model with one element changed at a time: one fault family removed
   (BPFO, BPFI, BSF, FTF, shaft), harmonics limited to h1 or h1-h2, the demodulation band replaced
   (kurtogram band, cepstral pre-whitening, both), the regularization C varied (0.03-10), and the
   classifier replaced (RBF-SVM, random forest). Protocols L8 (condition 0 and all), L10, A2R, LOBO.
   Metrics: accuracy, macro F1, macro AUC (one-vs-rest), healthy false-alarm rate.
B  Small-sample cases. k = 1..5 training bearings per class, one test bearing per class, 30 random draws
   (the draws of E5, same seed), for kinematic features + LR, band energies + LR and time + band + RF:
   accuracy, macro F1, macro AUC, false alarms; mean, SD and half peak-to-peak range over draws.
C  Distribution gap between artificial and real damage (the A2R split): maximum mean discrepancy (MMD,
   Gaussian kernel, median bandwidth) between training-domain and test-domain recordings of the same class,
   divided by the MMD between classes within the test domain (lower = the domain gap is small relative to
   the class gap). Representations: 15 kinematic features, order spectrum, 32 band energies, time + band.
   Also t-SNE maps of the four representations (descriptive).
D  Bearing-identity information. How well a representation identifies the individual bearing: 29-way
   logistic regression over recordings, 5-fold cross-validation stratified by bearing
   (identity accuracy), next to its accuracy for the damage class under LOBO. A representation that
   identifies bearings far better than chance but transfers poorly carries bearing identity.
Every number comes from pb32_physics.npz (and the two window files for band energies and time statistics);
nothing is tuned on test bearings; all settings are fixed here.

Run:  python phase11_analysis.py --physics pb32_physics.npz --windows pb32_4096.npz --envelope pb32_4096_envelope.npz \
          --labels labels_32.csv --out phase11_out
"""
from __future__ import annotations
import argparse, csv, json, shutil, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from phase2_physics import L8_TRAIN, L8_TEST, L10, splits

COND0 = 'N15_M07_F10'; NAMES = ['H', 'OR', 'IR']
FAM = ['BPFO', 'BPFI', 'BSF', 'FTF', 'shaft']
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'DejaVu Serif'], 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})


def lr(C=0.3):
    return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def svm():
    return make_pipeline(StandardScaler(), SVC(C=1.0, gamma='scale', probability=True, random_state=0))


def rf():
    return RandomForestClassifier(300, random_state=0, n_jobs=-1)


def metrics(y, p, P):
    r = {'acc': float((y == p).mean()), 'f1': float(f1_score(y, p, average='macro', labels=[0, 1, 2], zero_division=0))}
    present = np.unique(y)
    if len(present) == 3:
        r['auc'] = float(roc_auc_score(y, P, multi_class='ovr', average='macro', labels=[0, 1, 2]))
    h = y == 0
    if h.any(): r['false_alarm'] = float((p[h] != 0).mean())
    return r


def proba3(model, X):
    P = np.zeros((len(X), 3)); P[:, model.classes_] = model.predict_proba(X); return P


def run_protocols(X, y, B, cond, mk, d):
    """Accuracy etc. on L8 (cond 0 and all), L10 (mean of 10 splits), A2R, LOBO (pooled)."""
    out = {}
    def fit(tr, te):
        m = mk().fit(X[tr], y[tr]); return m.predict(X[te]), proba3(m, X[te])
    for name, c0 in (('L8_c0', True), ('L8_all', False)):
        tr, te = np.isin(B, L8_TRAIN), np.isin(B, L8_TEST)
        if c0: tr &= cond == COND0; te &= cond == COND0
        p, P = fit(tr, te); out[name] = metrics(y[te], p, P)
    l10, ys, ps, Ps = [], [], [], []
    lobo = {'y': [], 'p': [], 'P': []}
    for proto, fold, trb, teb in splits(d):
        tr, te = np.isin(B, trb), np.isin(B, teb)
        if proto == 'L8': continue
        p, P = fit(tr, te)
        if proto == 'L10': l10.append(metrics(y[te], p, P))
        elif proto == 'A2R': out['A2R'] = metrics(y[te], p, P)
        else: lobo['y'].append(y[te]); lobo['p'].append(p); lobo['P'].append(P)
    out['L10'] = {k: float(np.mean([m[k] for m in l10 if k in m])) for k in l10[0]}
    out['L10']['acc_sd'] = float(np.std([m['acc'] for m in l10]))
    out['LOBO'] = metrics(np.concatenate(lobo['y']), np.concatenate(lobo['p']), np.concatenate(lobo['P']))
    return out


def mmd2(A, B, gamma):
    def k(X, Y):
        d = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None] - 2 * X @ Y.T
        return np.exp(-gamma * np.maximum(d, 0))
    return float(k(A, A).mean() + k(B, B).mean() - 2 * k(A, B).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--physics', default='pb32_physics.npz'); ap.add_argument('--windows', default='pb32_4096.npz')
    ap.add_argument('--envelope', default='pb32_4096_envelope.npz'); ap.add_argument('--labels', default='labels_32.csv')
    ap.add_argument('--out', default='phase11_out'); ap.add_argument('--draws', type=int, default=30)
    ap.add_argument('--parts', default='A,B,C,D')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); parts = a.parts.split(',')
    resf = out / 'phase11_numbers.json'
    R = json.loads(resf.read_text()) if resf.exists() else {}
    save = lambda: resf.write_text(json.dumps(R, indent=1))
    full = dict(np.load(a.physics)); keep = full['label'] < 3
    d = {k: (v[keep] if hasattr(v, 'shape') and v.shape[:1] == keep.shape else v) for k, v in full.items()}
    B, y, cond = d['bearing'], d['label'].astype(int), d['condition']
    meta = {r['bearing']: r for r in csv.DictReader(open(a.labels))}
    single = [b for b in meta if b in set(B)]
    names = list(d['feature_names']); P15 = d['feat_fixed'].astype(float)
    O = d['order_fixed'].astype(float); O = O[:, d['order_grid'] <= 11.8]
    S = T = None
    if Path(a.windows).exists() and Path(a.envelope).exists():
        from phase3b_ladder import build_features
        win = np.load(a.windows); env = np.load(a.envelope)
        feats, _ = build_features(full, {'X': win['X'], 'recording': win['recording']}, {'X': env['X']})
        S = np.nan_to_num(feats['S'][keep]); T = np.nan_to_num(feats['T'][keep]); del win, env
    else:
        print('window files not found: parts B-D run without band energies and time statistics')
    t0 = time.time()

    # ------------------------------------------------------------------ A ablation
    if 'A' in parts:
        print('A ablation ...', flush=True)
        cols = lambda pred: [i for i, n in enumerate(names) if pred(n)]
        cfg = [('Full model (15 features, fixed band, LR C = 0.3)', P15, lr(0.3))]
        for f in FAM:
            cfg.append((f'without {f} (12 features)', P15[:, cols(lambda n: not n.startswith(f + '_'))], lr(0.3)))
        cfg.append(('harmonic 1 only (5 features)', P15[:, cols(lambda n: n.endswith('_h1'))], lr(0.3)))
        cfg.append(('harmonics 1-2 (10 features)', P15[:, cols(lambda n: not n.endswith('_h3'))], lr(0.3)))
        for v, lab in (('sk', 'kurtogram band'), ('cpw', 'cepstral pre-whitening'), ('cpw_sk', 'pre-whitening + kurtogram band')):
            cfg.append((f'demodulation: {lab}', d[f'feat_{v}'].astype(float), lr(0.3)))
        for C in (0.03, 0.1, 1.0, 3.0, 10.0):
            cfg.append((f'regularization C = {C:g}', P15, lr(C)))
        cfg.append(('classifier: RBF-SVM', P15, svm))
        cfg.append(('classifier: random forest', P15, rf))
        R['A_ablation'] = {}
        for name, X, mk in cfg:
            R['A_ablation'][name] = run_protocols(np.nan_to_num(X), y, B, cond, mk, d); save()
            r = R['A_ablation'][name]
            print(f"  {name:48s} L8c0 {r['L8_c0']['acc']*100:5.1f} L8 {r['L8_all']['acc']*100:5.1f} L10 {r['L10']['acc']*100:5.1f} "
                  f"A2R {r['A2R']['acc']*100:5.1f} LOBO {r['LOBO']['acc']*100:5.1f} F1 {r['LOBO']['f1']*100:5.1f} "
                  f"AUC {r['LOBO'].get('auc', float('nan'))*100:5.1f} FA {r['LOBO']['false_alarm']*100:5.1f}", flush=True)

    # ------------------------------------------------------------------ B small-sample cases (E5 draws)
    if 'B' in parts:
        print('B small-sample cases ...', flush=True)
        rng = np.random.default_rng(1); cls_b = {k: [b for b in single if int(meta[b]['label']) == k] for k in range(3)}
        sets = {'kinematic features + LR': (P15, lr(0.3))}
        if S is not None:
            sets['band energies + LR'] = (S, lr(0.3)); sets['time + band + RF'] = (np.concatenate([T, S], 1), rf)
        else:
            sets['order spectrum + LR'] = (O, lr(0.3))
        res = {k: {} for k in sets}
        for kk in (1, 2, 3, 4, 5):
            acc = {k: [] for k in sets}
            for _ in range(a.draws):
                te_b = [rng.choice(cls_b[c]) for c in range(3)]
                tr_b = sum([list(rng.choice([b for b in cls_b[c] if b not in te_b], kk, replace=False)) for c in range(3)], [])
                tr, te = np.isin(B, tr_b), np.isin(B, te_b)
                for k, (X, mk) in sets.items():
                    m = mk().fit(X[tr], y[tr]); acc[k].append(metrics(y[te], m.predict(X[te]), proba3(m, X[te])))
            for k in sets:
                res[k][kk] = {m: {'mean': float(np.mean([r[m] for r in acc[k] if m in r])), 'sd': float(np.std([r[m] for r in acc[k] if m in r])),
                                  'half_range': float((np.max([r[m] for r in acc[k] if m in r]) - np.min([r[m] for r in acc[k] if m in r])) / 2)}
                              for m in ('acc', 'f1', 'auc', 'false_alarm')}
                print(f"  k={kk} {k:26s} acc {res[k][kk]['acc']['mean']*100:5.1f} F1 {res[k][kk]['f1']['mean']*100:5.1f} "
                      f"AUC {res[k][kk]['auc']['mean']*100:5.1f} FA {res[k][kk]['false_alarm']['mean']*100:5.1f}", flush=True)
        R['B_cases'] = res; save()
        fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={'wspace': 0.32})
        cols_ = ['#2a6fb5', '#2a9d75', '#e0702f']
        for ax, m, lab in zip(axs, ('acc', 'f1', 'auc'), ('Accuracy (%)', 'Macro F1 (%)', 'Macro AUC (%)')):
            for (k, v), c in zip(res.items(), cols_):
                ks = sorted(v); mu = np.array([v[q][m]['mean'] for q in ks]) * 100; hr = np.array([v[q][m]['half_range'] for q in ks]) * 100
                ax.errorbar(np.array(ks) + (0.08 * list(res).index(k) - 0.08), mu, yerr=hr, color=c, marker='o', ms=3.5, lw=1.4, capsize=2, label=k)
            ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xlabel('Training bearings per class'); ax.set_ylabel(lab); ax.set_ylim(0, 105)
        axs[0].legend(loc='upper center', bbox_to_anchor=(1.75, -0.25), ncol=3, fontsize=7.5)
        fig.savefig(out / 'F26_cases.png'); plt.close(fig)

    # ------------------------------------------------------------------ C distribution gap (A2R) and t-SNE
    if 'C' in parts:
        print('C distribution gap ...', flush=True)
        a2r = next((trb, teb) for proto, fold, trb, teb in splits(d) if proto == 'A2R')
        src, tgt = np.isin(B, a2r[0]), np.isin(B, a2r[1])
        reps = {'15 kinematic features': P15, 'order spectrum': O}
        if S is not None:
            reps['32 band energies'] = S; reps['time + band statistics'] = np.concatenate([T, S], 1)
        R['C_gap'] = {}
        for name, X in reps.items():
            sc = StandardScaler().fit(X[src]); Z = sc.transform(X); Z = np.nan_to_num(Z)
            sub = np.random.default_rng(0).choice(len(Z), min(len(Z), 1500), replace=False)
            Zs = Z[sub]; n2 = (Zs ** 2).sum(1); dd = n2[:, None] + n2[None] - 2 * Zs @ Zs.T   # squared distances, no 3-D array
            gamma = 1.0 / np.median(dd[dd > 1e-9])
            dom = {c: mmd2(Z[src & (y == c)], Z[tgt & (y == c)], gamma) for c in range(3)}
            cls = [mmd2(Z[tgt & (y == i)], Z[tgt & (y == j)], gamma) for i in range(3) for j in range(i + 1, 3)]
            R['C_gap'][name] = {'mmd_domain': {NAMES[c]: v for c, v in dom.items()}, 'mmd_class_target': float(np.mean(cls)),
                                'ratio': float(np.mean(list(dom.values())) / np.mean(cls))}
            print(f"  {name:26s} domain MMD H {dom[0]:.3f} OR {dom[1]:.3f} IR {dom[2]:.3f}  class MMD {np.mean(cls):.3f}  ratio {R['C_gap'][name]['ratio']:.2f}", flush=True)
        save()
        from sklearn.manifold import TSNE
        fig, axs = plt.subplots(1, len(reps), figsize=(1.8 * len(reps) + 0.4, 2.2))
        axs = np.atleast_1d(axs); CC = ['#555555', '#2a6fb5', '#e0702f']
        sel = src | tgt
        for ax, (name, X) in zip(axs, reps.items()):
            Z = np.nan_to_num(StandardScaler().fit(X[src]).transform(X[sel]))
            E = TSNE(2, perplexity=30, init='pca', random_state=0).fit_transform(Z); ys = y[sel]; ss = src[sel]
            for c in range(3):
                ax.scatter(E[ss & (ys == c), 0], E[ss & (ys == c), 1], s=3, color=CC[c], lw=0)
                ax.scatter(E[~ss & (ys == c), 0], E[~ss & (ys == c), 1], s=5, facecolor='none', edgecolor=CC[c], lw=0.4)
            ax.set_title(f"{name}\nratio {R['C_gap'][name]['ratio']:.2f}", fontsize=7.5); ax.set_xticks([]); ax.set_yticks([])
        fig.savefig(out / 'F27_tsne_a2r.png'); plt.close(fig)

    # ------------------------------------------------------------------ D bearing identity
    if 'D' in parts:
        print('D bearing identity ...', flush=True)
        reps = {'15 kinematic features': P15, 'order spectrum': O}
        if S is not None:
            reps['32 band energies'] = S; reps['time + band statistics'] = np.concatenate([T, S], 1)
        bid = np.array([single.index(b) for b in B]); strat = np.array([f'{b}|{c}' for b, c in zip(B, cond)])
        skf = StratifiedKFold(5, shuffle=True, random_state=0); R['D_identity'] = {}
        lobo_acc = {}
        if 'A_ablation' in R:
            lobo_acc['15 kinematic features'] = R['A_ablation']['Full model (15 features, fixed band, LR C = 0.3)']['LOBO']['acc']
        for name, X in reps.items():
            X = np.nan_to_num(X); ok = n = 0
            for tr, te in skf.split(X, bid):
                m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(X[tr], bid[tr])
                ok += (m.predict(X[te]) == bid[te]).sum(); n += len(te)
            if name not in lobo_acc:
                lobo_acc[name] = run_protocols(X, y, B, cond, lr(0.3), d)['LOBO']['acc']
            R['D_identity'][name] = {'identity_acc': ok / n, 'chance': 1 / len(single), 'lobo_damage_acc': float(lobo_acc[name])}
            print(f"  {name:26s} bearing identity {ok / n * 100:5.1f}% (chance {100 / len(single):.1f}%)  LOBO damage accuracy {lobo_acc[name] * 100:5.1f}%", flush=True)
        save()
    shutil.make_archive(str(out), 'zip', out); print(f'written {out} and {out}.zip ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
