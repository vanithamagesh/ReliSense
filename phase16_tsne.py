"""Phase 16 (v1.0): t-SNE coordinates and separation scores of four representations (CPU, about 5-10 minutes).

Representations exactly as in phase 11: the 15 kinematic features (feat_fixed of pb32_physics.npz), the order spectrum
(order_fixed, orders <= 11.8), 32 band energies and time + band statistics (phase3b_ladder.build_features on the
window files). All 2559 recordings of the 29 single-damage bearings are standardized together and embedded with t-SNE
(perplexity 30, PCA initialisation, random_state 0); the embedding is descriptive only and is not used by any model.
Separation scores in the standardized space (not in t-SNE): silhouette by damage class and by bearing.
Every representation is saved as a checkpoint (checkpoints/phase16/<name>.npz) and reused on a second run.
Output: <root>/phase16_tsne.npz (small).
Colab:  !python /content/drive/MyDrive/ReliSense_study/phase16_tsne.py --root /content/drive/MyDrive/ReliSense_study
"""
import argparse, csv, os, sys, time
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser(); ap.add_argument('--root', default='/content/drive/MyDrive/ReliSense_study'); ap.add_argument('--fresh', action='store_true')
a = ap.parse_args(); ROOT = a.root; PB = ROOT + '/pb32'; CK = ROOT + '/checkpoints/phase16'; os.makedirs(CK, exist_ok=True)
sys.path.insert(0, ROOT + '/phase1'); t0 = time.time()
full = dict(np.load(PB + '/pb32_physics.npz', allow_pickle=True)); keep = full['label'] < 3
d = {k: (v[keep] if hasattr(v, 'shape') and v.shape[:1] == keep.shape else v) for k, v in full.items()}
B, y, cond = d['bearing'], d['label'].astype(int), d['condition']
meta = {r['bearing']: r for r in csv.DictReader(open(ROOT + '/phase1/labels_32.csv'))}
reps = {'15 kinematic features': d['feat_fixed'].astype(float),
        'order spectrum': d['order_fixed'].astype(float)[:, d['order_grid'] <= 11.8]}
need_s = [n for n in ('32 band energies', 'time + band statistics') if not os.path.exists(f'{CK}/{n}.npz') or a.fresh]
if need_s:
    from phase3b_ladder import build_features
    win = np.load(PB + '/pb32_4096.npz'); env = np.load(PB + '/pb32_4096_envelope.npz')
    feats, _ = build_features(full, {'X': win['X'], 'recording': win['recording']}, {'X': env['X']})
    S = np.nan_to_num(feats['S'][keep]); T = np.nan_to_num(feats['T'][keep]); del win, env
    reps['32 band energies'] = S; reps['time + band statistics'] = np.concatenate([T, S], 1)
else:
    reps['32 band energies'] = reps['time + band statistics'] = None
out = {'bearing': B, 'label': y, 'condition': cond, 'names': np.array(list(reps))}
for col in next(iter(meta.values())).keys():
    out[f'meta_{col}'] = np.array([meta[b][col] for b in B])
for i, (name, X) in enumerate(reps.items()):
    p = f'{CK}/{name}.npz'
    if os.path.exists(p) and not a.fresh:
        c = dict(np.load(p)); print(f'{name}: reused checkpoint', flush=True)
    else:
        Z = np.nan_to_num(StandardScaler().fit_transform(np.nan_to_num(X)))
        E = TSNE(2, perplexity=30, init='pca', random_state=0).fit_transform(Z)
        c = {'E': E.astype(np.float32), 'sil_class': silhouette_score(Z, y), 'sil_bearing': silhouette_score(Z, B), 'dim': Z.shape[1]}
        np.savez_compressed(p, **c); print(f'{name}: dim {Z.shape[1]}, silhouette class {c["sil_class"]:.3f}, bearing {c["sil_bearing"]:.3f} ({time.time() - t0:.0f} s)', flush=True)
    out[f'E{i}'] = c['E']; out[f'sil_class{i}'] = c['sil_class']; out[f'sil_bearing{i}'] = c['sil_bearing']
np.savez_compressed(ROOT + '/phase16_tsne.npz', **out)
print('written', ROOT + '/phase16_tsne.npz', f'{os.path.getsize(ROOT + "/phase16_tsne.npz") / 1e6:.2f} MB', f'{time.time() - t0:.0f} s')
