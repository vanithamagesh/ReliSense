"""Paper A variant of fig_flow (v1.0, from fig_method.py v1.0): the verification controls are omitted."""
import json
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Wedge
from matplotlib.lines import Line2D
from ieee_style import plt, C, COL1, COL2, save, panel

# =============================================================== flowchart: training, calibration, deployment (two branches)
fig = plt.figure(figsize=(COL2, 3.25)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis('off')
def node(x, y, w, h, t, fc='#f4f7fb', ec='#7d93ad'):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.25,rounding_size=0.9', fc=fc, ec=ec, lw=0.5))
    ax.text(x, y, t, ha='center', va='center', fontsize=7, linespacing=1.2)
def arr(p0, p1, rad=0.0, col='#555555', ls='-'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=6, lw=0.6, color=col, ls=ls, connectionstyle=f'arc3,rad={rad}', shrinkA=1, shrinkB=1))
for x0, x1, t, fc in ((0.5, 61, 'Offline: training bearings $\\mathcal{T}_b$ only', '#eaf1f8'), (62.5, 99.5, 'Online: new bearing $b$', '#fbefe4')):
    ax.add_patch(FancyBboxPatch((x0, 0.8), x1 - x0, 44, boxstyle='round,pad=0,rounding_size=1', fc=fc, ec='#9a9a9a', lw=0.5, ls=(0, (3, 2)), zorder=-1))
    ax.text((x0 + x1) / 2, 42.8, t, ha='center', va='center', fontsize=8, style='italic')
node(9.5, 33, 16, 7, 'recordings of $\\mathcal{T}_b$\nwith labels')
node(29.5, 33, 18, 8.5, 'Module I\nkinematic $\\mathbf{z}$ (31)\nenvelope $\\mathbf{e}$ (62)')
node(50.5, 37.5, 18, 6, 'kinematic branch:\nstandardize, LR, $C$ = 0.3', fc='#e6eef8')
node(50.5, 28.5, 18, 6, 'envelope branch:\nRF, 500 trees', fc='#e6eef8')
node(29.5, 17, 18, 9, 'for each $v \\in \\mathcal{T}_b$:\nfit both branches on\n$\\mathcal{T}_b\\setminus\\{v\\}$, score $v$')
node(50.5, 17, 18, 7, 'pool confidences\n$\\tau_b = Q_{1-\\kappa}$')
arr((17.6, 33), (20.4, 33)); arr((38.7, 34.5), (41.4, 37)); arr((38.7, 31.5), (41.4, 29)); arr((29.5, 28.7), (29.5, 21.6)); arr((38.7, 17), (41.4, 17))
node(70, 33, 13, 7, 'recording of $b$,\nspeed, geometry')
node(88, 33, 19, 8.5, 'Module I, both branches\n$p = \\frac{1}{2}(p_{\\mathrm{LR}} + p_{\\mathrm{RF}})$')
node(88, 18.5, 15, 6.5, '$c = \\max_k\\, p(k)$\n$c \\geq \\tau_b$ ?', fc='#fff7ec', ec='#c99a6e')
node(73, 5.8, 13, 6, 'diagnosis\nH / OR / IR'); node(91, 5.8, 13, 6, 'refer to\ninspection')
arr((76.6, 33), (78.4, 33)); arr((88, 28.7), (88, 21.8)); arr((84, 15.2), (76.5, 8.9)); ax.text(78.8, 12.6, 'yes', fontsize=7); arr((90, 15.2), (91, 8.9)); ax.text(91.6, 12.6, 'no', fontsize=7)
ax.plot([59.6, 61.7, 61.7], [37.5, 37.5, 33], color=C['rs'], lw=0.6); ax.plot([59.6, 61.7], [28.5, 28.5], color=C['rs'], lw=0.6)
ax.plot([61.7, 61.7], [33, 39.8], color=C['rs'], lw=0.6); ax.plot([61.7, 88, 88], [39.8, 39.8, 37.3], color=C['rs'], lw=0.6); arr((88, 38.6), (88, 37.4), col=C['rs'])
ax.text(75, 40.3, 'trained branches', fontsize=7, color=C['rs'], ha='center', va='bottom')
arr((59.6, 17), (80.4, 17.8), col=C['rs']); ax.text(70, 18.9, r'threshold $\tau_b$', fontsize=7, color=C['rs'], ha='center')
save(fig, 'fig_flow_A.png')
