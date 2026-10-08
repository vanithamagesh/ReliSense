"""Overall architecture of ReliSense for Paper A (v1.0, from fig_arch3.py v3.0 without the verification module): two-branch data-flow diagram drawn with real outputs, no text boxes.
Data: KA16_stages.npz (raw, envelope, speed, order spectrum and 15 features of N15_M07_F10_KA16_1, phase 14); the 16
modulation features of the same recording are computed here from its order spectrum with the definition of phase 19;
phase22_figdata.npz (LOBO posteriors and thresholds of the final ReliSense); phase6_numbers.json (fault-order perturbation);
phase13_figdata.npz (folding field); phase16_tsne.npz (embedding); cut-away render of the 6203 (Figures v6.0).
The envelope branch input (4 demodulation variants x 15 features + 2 statistics) is drawn as its structure, without values,
because those values of this recording are not stored locally."""
import json
import numpy as np
from PIL import Image
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.lines import Line2D
from ieee_style import plt, C, COL2, save
from v326 import FD

S = dict(np.load('KA16_stages.npz', allow_pickle=True)); Z = dict(np.load('phase13_figdata.npz', allow_pickle=True))
P6 = json.load(open('phase6_numbers.json')); T = dict(np.load('phase16_tsne.npz', allow_pickle=True))
FO = S['fault_orders'].astype(float); fs = int(S['fs']); g = S['order_grid'].astype(float); O = S['order_spec'].astype(float)
BPFO, BPFI, BSF, FTF = FO[:4]
cent = ([h * BPFI + s for h in (1, 2) for s in (-1, 1)] + [h * BPFO + s * FTF for h in (1, 2) for s in (-1, 1)] +
        [h * BSF + s * FTF for h in (1, 2) for s in (-1, 1)] + [4 * BSF, 5 * BSF, 4 * FTF, 5 * FTF])
ext = []
for r0 in cent:                                                     # phase 19 definition on the order axis
    w = max(0.02 * r0, 0.03); win = np.abs(g - r0) <= w; bg = (np.abs(g - r0) <= 0.4) & ~win
    ext.append(O[win].max() - np.median(O[bg]))
z31 = np.r_[S['features'].astype(float), ext]

fig = plt.figure(figsize=(COL2, 3.75)); MOD = {'I': '#2b6aa6', 'II': '#c8641e', 'III': '#3f7f3a'}


def ax_at(x, y, w, h, label=None, dy=0.016):
    a = fig.add_axes([x, y, w, h]); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(False)
    if label: fig.text(x + w / 2, y - dy, label, ha='center', va='top', linespacing=1.05)
    return a


def flow(p0, p1, col='#7d7d7d', rad=0.0, ls='-', lw=0.8):
    fig.add_artist(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle='-|>', mutation_scale=7, lw=lw, color=col, ls=ls,
                                   connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))


def bracket(x0, x1, y, col, name):
    fig.add_artist(Line2D([x0, x0, x1, x1], [y - 0.014, y, y, y - 0.014], transform=fig.transFigure, color=col, lw=1.0))
    fig.text((x0 + x1) / 2, y + 0.01, name, ha='center', va='bottom', fontsize=8, color=col, style='italic')


Y1, H1 = 0.63, 0.19                 # kinematic path
Y2 = 0.36                           # envelope path
a = ax_at(0.0, 0.47, 0.09, 0.26, 'bearing\nunder test'); a.imshow(Image.open('../v60figs/cutaway.png')); a.set_aspect('equal')
t = np.arange(len(S['raw'])) / fs * 1e3
a = ax_at(0.11, Y1, 0.1, H1, r'vibration $x(t)$'); a.plot(t, S['raw'], lw=0.25, color=C['OR']); a.margins(x=0)
a = ax_at(0.235, Y1, 0.1, H1, 'envelope\n2–12 kHz'); e = S['env_sq'].astype(float); a.plot(t, e / e.max(), lw=0.3, color=C['rs']); a.margins(x=0)
a = ax_at(0.36, Y1, 0.13, H1, r'order spectrum $O(r)$'); a.plot(g, O, lw=0.35, color=C['OR']); a.set_xlim(0.2, 10.5); a.set_ylim(-1.5, 5.8)
for h in (1, 2, 3): a.axvspan(h * BPFO * 0.97, h * BPFO * 1.03, color=C['IR'], alpha=0.3, lw=0)
for r0 in cent[:4]: a.axvspan(r0 - 0.08, r0 + 0.08, color=C['B'], alpha=0.35, lw=0)
a = ax_at(0.515, Y1 + 0.01, 0.026, H1 - 0.02, r'$\mathbf{z}$, 31', dy=0.012)
a.imshow(z31[:, None], cmap='Blues', vmin=0, vmax=4, aspect='auto'); a.axhline(14.5, color='white', lw=1.2)
# envelope path: structure of the 62-dimensional input (4 variants x 15 features + 2 statistics)
a = ax_at(0.36, Y2, 0.13, 0.12, '4 demodulation bands\n(fixed, kurtogram, pre-whitened)')
f_ = S['spec_f'].astype(float); m_ = S['spec_mag'].astype(float); a.plot(f_ / 1000, m_ / m_.max(), lw=0.3, color='#888888'); a.set_xlim(0, f_[-1] / 1000)
a.axvspan(2, 12, color=C['be'], alpha=0.2, lw=0)
a = ax_at(0.515, Y2 - 0.005, 0.026, 0.13, r'$\mathbf{e}$, 62', dy=0.012); a.set_xlim(0, 1); a.set_ylim(62, 0)
for i in range(62): a.add_patch(Rectangle((0.08, i + 0.12), 0.84, 0.76, fc='white', ec=C['be'], lw=0.25))
for yy in (15, 30, 45, 60): a.axhline(yy, color=C['be'], lw=0.9)
fig.add_artist(Line2D([0.16, 0.16, 0.34], [0.585, 0.395, 0.395], transform=fig.transFigure, color='#7d7d7d', lw=0.8)); flow((0.34, 0.395), (0.357, 0.395))
for x0, x1 in ((0.09, 0.108), (0.212, 0.232), (0.337, 0.357), (0.492, 0.512)): flow((x0, Y1 + H1 / 2), (x1, Y1 + H1 / 2))
flow((0.492, Y2 + 0.06), (0.512, Y2 + 0.06))
# speed and geometry feed the order axis
sp = np.abs(S['speed'].astype(float)); a = ax_at(0.245, 0.465, 0.08, 0.06, r'speed $n(t)$', dy=0.008); a.plot(sp, lw=0.5, color=C['ink']); a.set_ylim(1440, 1560)
flow((0.33, 0.5), (0.375, Y1 - 0.005), rad=-0.2, col='#9a9a9a')
bracket(0.11, 0.545, 0.93, MOD['I'], 'Module I: kinematic signature and envelope features')
# Module II: two branches -> averaged posterior -> confidence vs threshold
fig.text(0.6, Y1 + H1 / 2, 'kinematic branch\nLR, $C$ = 0.3', ha='left', va='center', color=C['rs'], linespacing=1.1)
fig.text(0.6, Y2 + 0.06, 'envelope branch\nRF, 500 trees', ha='left', va='center', color=C['be'], linespacing=1.1)
flow((0.545, Y1 + H1 / 2), (0.595, Y1 + H1 / 2), col=C['rs']); flow((0.545, Y2 + 0.06), (0.595, Y2 + 0.06), col=C['be'])
flow((0.715, Y1 + H1 / 2), (0.75, 0.73), col=C['rs'], rad=0.0); flow((0.715, Y2 + 0.06), (0.75, 0.67), col=C['be'], rad=0.2)
P = FD['prob'].astype(float); y = FD['label'].astype(int); tau = float(np.mean(FD['tau'][:, 2]))
V = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]]); XY = P @ V; acc = P.max(1) >= tau; CC = np.array([C['H'], C['OR'], C['IR']])
a = ax_at(0.745, 0.62, 0.11, 0.24, 'averaged posterior\n' + r'$\frac{1}{2}(p_{\mathrm{LR}} + p_{\mathrm{RF}})$'); a.set_aspect('equal'); a.add_patch(Polygon(V, closed=True, fill=False, ec='#555555', lw=0.5))
a.scatter(XY[~acc, 0], XY[~acc, 1], s=0.6, color='#c9c9c9', lw=0, rasterized=True); a.scatter(XY[acc, 0], XY[acc, 1], s=0.8, c=CC[y[acc]], lw=0, rasterized=True)
for (vx, vy), t_ in zip(V, ('H', 'OR', 'IR')): a.text(vx + {'H': -0.09, 'OR': 0.1, 'IR': 0}[t_], vy + (0.08 if t_ == 'IR' else -0.06), t_, ha='center', va='center')
a.set_xlim(-0.18, 1.18); a.set_ylim(-0.14, 1.0)
conf = P.max(1); a = ax_at(0.745, 0.29, 0.12, 0.14, r'confidence $c$ vs. $\tau_b$')
bins = np.linspace(1 / 3, 1, 25); hc, be = np.histogram(conf, bins)
a.bar(be[:-1], hc, width=np.diff(be), align='edge', color=np.where(be[:-1] >= tau, '#2f8f7f', '#cdb38a'), lw=0); a.axvline(tau, color=C['ink'], lw=0.7)
a.set_yscale('log'); a.set_xlim(1 / 3, 1); a.yaxis.set_minor_locator(plt.NullLocator()); a.set_yticks([]); a.spines['bottom'].set_visible(True); a.spines['bottom'].set_linewidth(0.4)
flow((0.8, 0.515), (0.8, 0.445), col='#7d7d7d')
fig.text(0.955, 0.43, 'accept\nH / OR / IR', ha='center', va='center', color='#2f8f7f', linespacing=1.1)
fig.text(0.955, 0.27, 'refer to\ninspection', ha='center', va='center', color='#a5824c', linespacing=1.1)
flow((0.87, 0.39), (0.91, 0.43), col='#2f8f7f'); flow((0.87, 0.34), (0.91, 0.29), col='#a5824c')
bracket(0.6, 0.99, 0.93, MOD['II'], 'Module II: two branches and abstention')
save(fig, 'fig_arch3a.png')
print('modulation features of KA16:', np.round(ext, 2))
