"""ReliSense phase 12 (v1.0): damage-wise ("360-degree") analysis of results that already exist.

No model is trained here. Inputs (all real results of earlier phases):
  phase2_per_bearing.json   per-bearing LOBO accuracy of ReliSense and four feature baselines (phase 2)
  wdcnn_env_lobo.json       per-bearing LOBO accuracy of WDCNN with envelope input (phase 4, seed 0)
  phase5_numbers.json       per-bearing LOBO accuracy of ReliSense under each operating condition (phase 5)
  labels_32.csv             damage attributes from Lessmeier et al. 2016, Tables 4, 5 and 7
  phase10_results.json, cwru_features.npz, hust_features.npz   external benchmarks (phase 10)
Outputs: phase12_numbers.json and figures F28-F30.
Usage: python phase12_damage_analysis.py <dir with inputs> <out dir>
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'FreeSerif'], 'font.size': 8.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})

# ------------------------------------------------------------------ Paderborn: per-bearing table
LAB = {r['bearing']: r for r in csv.DictReader(open(IN / 'labels_32.csv'))}
P2 = json.load(open(IN / 'phase2_per_bearing.json'))
WD = json.load(open(IN / 'wdcnn_env_lobo.json'))
P5 = json.load(open(IN / 'phase5_numbers.json'))['lobo_accuracy_bearing_by_condition']
MODELS = {'lr_fixed': 'ReliSense', 'rule': 'Threshold rule', 'lr_cpw': 'CPW + LR', 'lr_all': '60 env. feat. + LR',
          'rf_all': '60 env. feat. + RF', 'wdcnn': 'WDCNN (envelope)'}
acc = {m: {} for m in MODELS}
for r in P2:
    if r['protocol'] == 'LOBO' and r['model'] in acc:
        acc[r['model']][r['bearing']] = r['acc_mean'] * 100
acc['wdcnn'] = {b: v * 100 for b, v in WD.items()}
BEAR = sorted(acc['lr_fixed'])
assert len(BEAR) == 29 and all(set(acc[m]) == set(BEAR) for m in acc)

SIZE = {'0': 'none', '1': '≤ 2 mm', '2': '2–4.5 mm', '3': '4.5–13.5 mm'}
METHOD = {'EDM': 'EDM trench (0.25 mm)', 'drilling': 'Drilling (Ø 0.9–3 mm)',
          'electric engraver': 'Electric engraver (1–4 mm)', 'fatigue: pitting': 'Fatigue pitting',
          'plastic deformation: indentations': 'Indentations'}
ARR = {'S': 'single', 'R': 'repeated', 'M': 'multiple'}


def attrs(b):
    l = LAB[b]
    if l['origin'] == 'healthy':
        return dict(group='Healthy', origin='healthy', loc='none', level='0', method='none', arr='none', char='none',
                    runin=float(l['mode'].split()[1].lstrip('>')))
    return dict(group=f"{l['origin']} {l['location']}", origin=l['origin'], loc=l['location'], level=l['extent'],
                method=l['mode'], arr=l['arrangement'], char=l['characteristic'])


A = {b: attrs(b) for b in BEAR}


def group_table(key, order, label, sel=lambda b: True):
    rows = []
    for g in order:
        bs = [b for b in BEAR if sel(b) and key(b) == g]
        if not bs:
            continue
        rows.append({'group': label(g), 'bearings': bs, 'n': len(bs),
                     **{MODELS[m]: round(float(np.mean([acc[m][b] for b in bs])), 1) for m in MODELS}})
    return rows


dam = lambda b: A[b]['origin'] != 'healthy'
T = {}
T['size_origin'] = group_table(lambda b: (A[b]['origin'], A[b]['level']),
                               [('artificial', '1'), ('artificial', '2'), ('real', '1'), ('real', '2'), ('real', '3'), ('healthy', '0')],
                               lambda g: f"{g[0]}, level {g[1]} ({SIZE[g[1]]})" if g[1] != '0' else 'healthy')
T['method'] = group_table(lambda b: A[b]['method'], list(METHOD), lambda g: METHOD[g], dam)
T['arrangement_real'] = group_table(lambda b: A[b]['arr'], ['S', 'R', 'M'], lambda g: ARR[g], lambda b: A[b]['origin'] == 'real')
T['characteristic_real'] = group_table(lambda b: A[b]['char'], ['single point', 'distributed'], lambda g: g, lambda b: A[b]['origin'] == 'real')
T['location'] = group_table(lambda b: (A[b]['origin'], A[b]['loc']), [('artificial', 'OR'), ('artificial', 'IR'), ('real', 'OR'), ('real', 'IR')],
                            lambda g: f'{g[0]} {g[1]}', dam)

# ------------------------------------------------------------------ statistics on ReliSense per-bearing accuracy
rs = acc['lr_fixed']
S = {}
real = [b for b in BEAR if A[b]['origin'] == 'real']
art = [b for b in BEAR if A[b]['origin'] == 'artificial']
damaged = real + art
rho, p = stats.spearmanr([int(A[b]['level']) for b in damaged], [rs[b] for b in damaged])
S['spearman_level_all_damaged'] = {'rho': round(rho, 3), 'p': round(p, 4), 'n': len(damaged)}
rho, p = stats.spearmanr([int(A[b]['level']) for b in real], [rs[b] for b in real])
S['spearman_level_real'] = {'rho': round(rho, 3), 'p': round(p, 4), 'n': len(real)}
a1 = [rs[b] for b in art if A[b]['level'] == '1']; r1 = [rs[b] for b in real if A[b]['level'] == '1']
u = stats.mannwhitneyu(a1, r1, alternative='two-sided')
S['mwu_level1_artificial_vs_real'] = {'median_art': float(np.median(a1)), 'median_real': float(np.median(r1)),
                                      'U': float(u.statistic), 'p': round(u.pvalue, 4), 'n': [len(a1), len(r1)]}
sp = [rs[b] for b in real if A[b]['arr'] == 'S' and A[b]['char'] == 'single point']
ot = [rs[b] for b in real if not (A[b]['arr'] == 'S' and A[b]['char'] == 'single point')]
u = stats.mannwhitneyu(sp, ot, alternative='two-sided')
S['mwu_real_single_vs_repeated_or_distributed'] = {'median_single': float(np.median(sp)), 'median_other': float(np.median(ot)),
                                                    'U': float(u.statistic), 'p': round(u.pvalue, 4), 'n': [len(sp), len(ot)]}
pit = [rs[b] for b in real if 'pitting' in A[b]['method']]; ind = [rs[b] for b in real if 'indent' in A[b]['method']]
u = stats.mannwhitneyu(pit, ind, alternative='two-sided')
S['mwu_real_pitting_vs_indentation'] = {'median_pitting': float(np.median(pit)), 'median_indent': float(np.median(ind)),
                                        'U': float(u.statistic), 'p': round(u.pvalue, 4), 'n': [len(pit), len(ind)]}
groups = [[rs[b] for b in damaged if A[b]['method'] == m] for m in METHOD]
h = stats.kruskal(*groups)
S['kruskal_method'] = {'H': round(h.statistic, 3), 'p': round(h.pvalue, 4), 'n': [len(g) for g in groups]}
hb = [b for b in BEAR if A[b]['origin'] == 'healthy']
rho, p = stats.spearmanr([A[b]['runin'] for b in hb], [rs[b] for b in hb])
S['spearman_runin_healthy'] = {'rho': round(rho, 3), 'p': round(p, 4), 'n': len(hb),
                               'runin_h': {b: A[b]['runin'] for b in hb}, 'acc': {b: rs[b] for b in hb}}
S['smallest_defect'] = {b: rs[b] for b in ('KA01', 'KI01')}

# per-model comparisons per bearing
best = {}
for b in BEAR:
    v = {MODELS[m]: acc[m][b] for m in MODELS}
    mx = max(v.values()); best[b] = [k for k, x in v.items() if x == mx]
S['bearings_where_model_is_best_or_tied'] = {MODELS[m]: sum(MODELS[m] in best[b] for b in BEAR) for m in MODELS}
S['oracle_mean_best_model_per_bearing'] = round(float(np.mean([max(acc[m][b] for m in MODELS) for b in BEAR])), 1)
S['all_models_below_60'] = [b for b in BEAR if max(acc[m][b] for m in MODELS) < 60]
S['all_models_at_least_95'] = [b for b in BEAR if min(acc[m][b] for m in MODELS) >= 95]
pair = {}
for m in MODELS:
    if m == 'lr_fixed':
        continue
    d = np.array([rs[b] - acc[m][b] for b in BEAR])
    w = stats.wilcoxon(d) if np.any(d != 0) else None
    pair[MODELS[m]] = {'mean_diff': round(float(d.mean()), 1), 'wins': int((d > 0).sum()), 'ties': int((d == 0).sum()),
                       'losses': int((d < 0).sum()), 'p_wilcoxon': round(float(w.pvalue), 4) if w else None}
S['paired_vs_ReliSense'] = pair
# model agreement: Spearman between models across the 29 bearings
S['spearman_between_models'] = {f'{MODELS[a]} | {MODELS[c]}': round(stats.spearmanr([acc[a][b] for b in BEAR], [acc[c][b] for b in BEAR])[0], 2)
                                for a in MODELS for c in MODELS if a < c}

# operating conditions (phase 5): 1500 rpm (N15_M07_F10) vs 900 rpm (N09_M07_F10) vs low radial force (N15_M07_F04)
cond = {}
for name, sel in (('artificial', lambda b: A[b]['origin'] == 'artificial'), ('real, level 1', lambda b: A[b]['origin'] == 'real' and A[b]['level'] == '1'),
                  ('real, level 2-3', lambda b: A[b]['origin'] == 'real' and A[b]['level'] in '23'), ('healthy', lambda b: A[b]['origin'] == 'healthy')):
    bs = [b for b in BEAR if sel(b)]
    cond[name] = [round(float(np.mean([P5[b][k] for b in bs])) * 100, 1) for k in range(4)]
S['condition_accuracy_by_group'] = {'order': ['N15_M07_F10', 'N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04'], **cond}

# ------------------------------------------------------------------ external datasets: bearing size and fault size
J = json.load(open(IN / 'phase10_results.json'))
RSM = 'ReliSense (P + LR)'; OTH = ['time + band + RF', 'band energies + LR']
GEO = {'6204': (33.5, 7.6), '6205': (38.5, 7.8), '6206': (46.0, 9.0), '6207': (53.5, 11.0), '6208': (60.0, 12.0)}


def contrast(npz, unit_of, healthy_of):
    """Mean kinematic peak-to-background (3 harmonics) at the fault order of the class, minus the same value on the
    healthy bearing of the same group: how much the defect raises its own line above a healthy bearing."""
    d = np.load(npz); P = d['P'].astype(float); names = list(d['feature_names'])
    key = {'OR': 'BPFO', 'IR': 'BPFI', 'B': 'BSF'}
    out = {}
    for u in np.unique(d['unit']):
        sel = d['unit'] == u; c = d['cls'][sel][0]
        if c == 'H':
            continue
        cols = [names.index(f'{key[c]}_h{h}') for h in (1, 2, 3)]
        hs = healthy_of(d, u)
        out[str(u)] = round(float(P[sel][:, cols].mean() - P[hs][:, cols].mean()), 3)
    return out


hc = contrast(IN / 'hust_features.npz', None, lambda d, u: d['unit'] == 'N' + u[1])
cc = contrast(IN / 'cwru_features.npz', None, lambda d, u: d['unit'] == 'H')
E = {'hust_contrast': hc, 'cwru_contrast': cc}
hu = J['hust']['3-class']['LOTO']; cw = J['cwru']['3-class']['LOSO']
E['hust_by_type'] = {t: {'pitch_mm': GEO[t][0], 'ball_mm': GEO[t][1],
                         **{m: round(float(np.mean([hu[m]['per_unit'][c + t[-1]] for c in 'NOI'])) * 100, 1) for m in [RSM] + OTH}}
                     for t in GEO}
E['hust_by_class'] = {c: {m: round(float(np.mean([hu[m]['per_unit'][c + t[-1]] for t in GEO])) * 100, 1) for m in [RSM] + OTH} for c in 'NOI'}
rho, p = stats.spearmanr([GEO[t][0] for t in GEO], [E['hust_by_type'][t][RSM] for t in GEO])
E['spearman_pitch_vs_relisense_hust'] = {'rho': round(rho, 3), 'p': round(p, 4), 'n': 5}
fu = [u for u in hc if u[0] in 'OI']
rho, p = stats.spearmanr([hc[u] for u in fu], [hu[RSM]['per_unit'][u] for u in fu])
E['spearman_contrast_vs_acc_hust_faulty'] = {'rho': round(rho, 3), 'p': round(p, 4), 'n': len(fu)}
cu = [u for u in cc if u[0] in 'IO']
xs_c = [hc[u] for u in fu] + [cc[u] for u in cu]; ys_c = [hu[RSM]['per_unit'][u] for u in fu] + [cw[RSM]['per_unit'][u] for u in cu]
rho, p = stats.spearmanr(xs_c, ys_c)
E['spearman_contrast_vs_acc_hust_cwru_faulty'] = {'rho': round(rho, 3), 'p': round(p, 4), 'n': len(xs_c)}
E['cwru_by_size'] = {s: {'diameter_mm': round(int(s) * 0.0254, 3),
                         **{m: round(float(np.mean([cw[m]['per_unit'][c + s] for c in ('IR', 'OR')])) * 100, 1) for m in [RSM] + OTH}}
                     for s in ('007', '014', '021')}
json.dump({'tables': T, 'stats': S, 'external': E}, open(OUT / 'phase12_numbers.json', 'w'), indent=1, ensure_ascii=False)

# ------------------------------------------------------------------ F28: 360-degree damage matrix
order = (sorted(hb, key=lambda b: A[b]['runin'])
         + sorted(art, key=lambda b: (A[b]['loc'] != 'OR', A[b]['level'], A[b]['method'], b))
         + sorted(real, key=lambda b: (A[b]['loc'] != 'OR', A[b]['level'], A[b]['method'], b)))
ML = list(MODELS)
M = np.array([[acc[m][b] for m in ML] for b in order])
fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.04)
ax = fig.add_subplot(gs[1]); axt = fig.add_subplot(gs[0])
im = ax.imshow(M, cmap='Blues', vmin=0, vmax=100, aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=6, color='white' if M[i, j] > 60 else '#222222')
SHORT = {'lr_fixed': 'Reli-\nSense', 'rule': 'Thresh.\nrule', 'lr_cpw': 'CPW\n+ LR', 'lr_all': '60 feat.\n+ LR', 'rf_all': '60 feat.\n+ RF', 'wdcnn': 'WDCNN\n(env.)'}
ax.set_xticks(range(len(ML))); ax.set_xticklabels([SHORT[m] for m in ML], fontsize=6.6)
ax.xaxis.tick_top()
ax.set_yticks([]); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
cols = ['Bearing', 'Origin', 'Race', 'Size', 'Method', 'Arr.']
xs = [0.0, 0.14, 0.32, 0.42, 0.62, 0.93]
axt.set_xlim(0, 1); axt.set_ylim(len(order) - 0.5, -0.5); axt.axis('off')
for x, c in zip(xs, cols):
    axt.text(x, -1.2, c, fontsize=7, fontweight='bold', va='center')
short = {'EDM': 'EDM', 'drilling': 'drilling', 'electric engraver': 'engraver', 'fatigue: pitting': 'pitting',
         'plastic deformation: indentations': 'indentation'}
for i, b in enumerate(order):
    a = A[b]
    if a['origin'] == 'healthy':
        vals = [b, 'healthy', '–', '–', f"run-in {LAB[b]['mode'].split()[1]} h", '–']
    else:
        vals = [b, a['origin'], a['loc'], SIZE[a['level']], short[a['method']],
                a['arr'] + ('/d' if a['char'] == 'distributed' else '')]
    for x, v in zip(xs, vals):
        axt.text(x, i, v, fontsize=6.6, va='center')
for yy in (len(hb) - 0.5, len(hb) + len(art) - 0.5):
    ax.axhline(yy, color='#d0d0d0' if False else '#555555', lw=0.8); axt.axhline(yy, color='#555555', lw=0.8)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label('LOBO accuracy of the bearing (%)', fontsize=7)
fig.savefig(OUT / 'F28_damage_matrix.png'); plt.close(fig)

# ------------------------------------------------------------------ F29: accuracy by damage attribute
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True, gridspec_kw={'wspace': 0.08, 'width_ratios': [6, 5, 4]})
SHOW = ['lr_fixed', 'rf_all', 'wdcnn']; COL = {'lr_fixed': '#2a6fb5', 'rf_all': '#e0702f', 'wdcnn': '#7a4fb0'}; MK = dict(zip(SHOW, 'os^'))
panels = [('(a) Origin and damage size', T['size_origin'], lambda g: g.replace('artificial', 'Art.').replace('real', 'Real').replace(', level', ' L').split(' (')[0]),
          ('(b) Damage method', T['method'], lambda g: g.split(' (')[0].replace('Electric engraver', 'Engraver').replace('Fatigue pitting', 'Pitting').replace('EDM trench', 'EDM')),
          ('(c) Real damage arrangement', T['arrangement_real'] + T['characteristic_real'][1:], lambda g: g)]
for ax, (t, rows, lab) in zip(axs, panels):
    x = np.arange(len(rows))
    for k, m in enumerate(SHOW):
        ax.scatter(x + (k - 1) * 0.2, [r[MODELS[m]] for r in rows], color=COL[m], marker=MK[m], s=22, zorder=3, label=MODELS[m])
        # individual bearings of ReliSense
    for i, r in enumerate(rows):
        ax.scatter(np.full(r['n'], i - 0.2) + np.linspace(-0.05, 0.05, r['n']), [acc['lr_fixed'][b] for b in r['bearings']],
                   s=5, color='#9bbbe0', zorder=2, lw=0)
    ax.set_xticks(x); ax.set_xticklabels([f"{lab(r['group'])}\n(n={r['n']})" for r in rows], fontsize=6.4)
    ax.set_title(t, fontsize=8, loc='left'); ax.set_ylim(-3, 105); ax.grid(axis='y', color='#e6e6e6', lw=0.5)
axs[0].set_ylabel('Mean LOBO accuracy per bearing (%)')
axs[0].legend(loc='upper center', bbox_to_anchor=(1.6, -0.3), ncol=3, fontsize=7)
fig.savefig(OUT / 'F29_damage_attributes.png'); plt.close(fig)

# ------------------------------------------------------------------ F30: bearing size (HUST) and fault size (CWRU)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={'wspace': 0.42})
MC = {RSM: '#2a6fb5', 'time + band + RF': '#e0702f', 'band energies + LR': '#2a9d75'}
ML2 = {RSM: 'ReliSense', 'time + band + RF': 'Time + band, RF', 'band energies + LR': 'Band energies, LR'}
pd_ = [GEO[t][0] for t in GEO]
for m in [RSM] + OTH:
    axs[0].plot(pd_, [E['hust_by_type'][t][m] for t in GEO], marker='o', ms=4, lw=1.2, color=MC[m], label=ML2[m])
for t, x in zip(GEO, pd_):
    axs[0].text(x, 3, t, ha='center', fontsize=6, color='#555555')
axs[0].set_xlabel('Pitch diameter of the unseen type (mm)'); axs[0].set_ylabel('Accuracy (%)'); axs[0].set_ylim(0, 105)
axs[0].set_title('(a) HUST, by bearing type', fontsize=8, loc='left')
for u in fu:
    y = hu[RSM]['per_unit'][u] * 100
    axs[1].scatter(hc[u], y, color='#2a6fb5', marker='o' if u[0] == 'O' else '^', s=18, zorder=3)
    if y < 99: axs[1].annotate(f'HUST {u[0]} 620{u[1]}', (hc[u], y), fontsize=5.5, xytext=(4, 2), textcoords='offset points')
for u in cu:
    y = cw[RSM]['per_unit'][u] * 100
    axs[1].scatter(cc[u], y, facecolor='white', edgecolor='#2a6fb5', marker='o' if u[0] == 'O' else '^', s=18, zorder=3)
    if y < 99: axs[1].annotate(f'CWRU {u[:2]} {u[2:]}', (cc[u], y), fontsize=5.5, xytext=(-6, -11), textcoords='offset points')
r_ = E['spearman_contrast_vs_acc_hust_cwru_faulty']
axs[1].text(0.98, 0.42, f"Spearman \u03c1 = {r_['rho']:.2f}\np = {r_['p']:.3f}, n = {r_['n']}", transform=axs[1].transAxes, ha='right', fontsize=6.3)
axs[1].text(0.98, 0.25, 'filled: HUST, open: CWRU\ncircle: outer race, triangle: inner race', transform=axs[1].transAxes, ha='right', fontsize=6)
axs[1].set_xlabel('Signature contrast (log units)'); axs[1].set_ylabel('ReliSense accuracy (%)'); axs[1].set_ylim(-16, 108)
axs[1].set_title('(b) Defect line vs accuracy', fontsize=8, loc='left')
xs3 = np.arange(3)
for k, m in enumerate([RSM] + OTH):
    axs[2].bar(xs3 + (k - 1) * 0.26, [E['cwru_by_size'][s][m] for s in ('007', '014', '021')], 0.24, color=MC[m])
axs[2].set_xticks(xs3); axs[2].set_xticklabels(['0.18', '0.36', '0.53'])
axs[2].set_xlabel('Unseen fault diameter (mm)'); axs[2].set_ylabel('Accuracy, IR and OR (%)'); axs[2].set_ylim(0, 105)
axs[2].set_title('(c) CWRU, by fault size', fontsize=8, loc='left')
axs[0].legend(loc='upper center', bbox_to_anchor=(1.9, -0.3), ncol=3, fontsize=7)
fig.savefig(OUT / 'F30_bearing_fault_size.png'); plt.close(fig)
print(json.dumps({'stats': S, 'external': E}, indent=1, ensure_ascii=False))
print(json.dumps(T, indent=1, ensure_ascii=False)[:6000])
