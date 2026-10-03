"""Schematic of the 6203 bearing geometry and of the fault-impact mechanism (drawing, no data)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import Circle, Wedge, FancyArrowPatch
BLUE, ORANGE, GREY, INK, MUTED = '#2a78d6', '#eb6834', '#9a9a93', '#222222', '#5f5e58'
n, d, D = 8, 6.75, 28.55; Ro_out, Ro_in = 20.0, D/2 + d/2; Ri_out, Ri_in = D/2 - d/2, 8.5
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw=dict(width_ratios=[1, 1.25]))
a.set_aspect('equal'); a.axis('off')
a.add_patch(Wedge((0, 0), Ro_out, 0, 360, width=Ro_out-Ro_in, fc='#d9d8d2', ec=INK, lw=0.8))
a.add_patch(Wedge((0, 0), Ri_out, 0, 360, width=Ri_out-Ri_in, fc='#d9d8d2', ec=INK, lw=0.8))
for k in range(n):
    t = 2*np.pi*k/n + np.pi/2
    a.add_patch(Circle((D/2*np.cos(t), D/2*np.sin(t)), d/2, fc='#f4f3ee', ec=INK, lw=0.8))
a.add_patch(Circle((0, 0), D/2, fill=False, ls='--', ec=MUTED, lw=0.7))
a.add_patch(Wedge((0, 0), Ro_in+0.9, 80, 88, width=0.9, fc=ORANGE, ec=ORANGE))        # outer-race defect
a.add_patch(Wedge((0, 0), Ri_out, 200, 208, width=0.9, fc=ORANGE, ec=ORANGE))       # inner-race defect
a.annotate('', xy=(D/2*np.cos(np.radians(-30)), D/2*np.sin(np.radians(-30))), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=BLUE, lw=1))
a.text(2.0, -3.6, 'D/2', color=BLUE, fontsize=8)
bx, by = 0, D/2; a.annotate('', xy=(bx+d/2, by+0.0), xytext=(bx-d/2, by), arrowprops=dict(arrowstyle='<->', color=BLUE, lw=0.9))
a.text(bx-1.2, by+1.0, 'd', color=BLUE, fontsize=8)
a.text(2.2, 19.6, 'outer-race\ndefect', color=ORANGE, fontsize=7, ha='left')
a.annotate('inner-race\ndefect', xy=(-6.6, -2.6), xytext=(-21, -15), color=ORANGE, fontsize=7, arrowprops=dict(arrowstyle='-', color=ORANGE, lw=0.6))
a.text(0, -23.5, '6203: n = 8, d = 6.75 mm, D = 28.55 mm, φ = 0°', ha='center', fontsize=7.5, color=INK)
a.set_xlim(-24, 24); a.set_ylim(-25, 23); a.set_title('(a) bearing geometry', fontsize=8.5, loc='left')
# (b) impact train and its envelope
from scipy.signal import hilbert
fr = 25.0; fs = 50000; t = np.arange(int(0.06*fs))/fs; bpfo = 3.054*fr
x = np.zeros_like(t); rng = np.random.default_rng(1)
for t0 in np.arange(0.004, 0.06, 1/bpfo):
    m = t >= t0; x[m] += np.exp(-500*(t[m]-t0))*np.sin(2*np.pi*3000*(t[m]-t0))
x += 0.05*rng.standard_normal(len(t)); env = np.abs(hilbert(x))
b.plot(t*1000, x, color=BLUE, lw=0.5); b.plot(t*1000, env, color=ORANGE, lw=1.1)
t1, t2 = 4, 4 + 1000/bpfo
b.annotate('', xy=(t2, 1.2), xytext=(t1, 1.2), arrowprops=dict(arrowstyle='<->', color=INK, lw=0.8))
b.text((t1+t2)/2, 1.28, '1 / BPFO', ha='center', fontsize=7.5)
b.set_xlabel('time (ms)'); b.set_yticks([]); b.set_ylim(-1.1, 1.45); b.spines[['top', 'right', 'left']].set_visible(False)
b.set_title('(b) impacts excite a resonance; the envelope repeats at the fault rate', fontsize=8.5, loc='left')
b.text(45, -1.0, 'blue: vibration   orange: envelope', fontsize=7, ha='center', color=MUTED)
fig.tight_layout(); fig.savefig('rs_figs/fig0_geometry.png', dpi=600, bbox_inches='tight')
