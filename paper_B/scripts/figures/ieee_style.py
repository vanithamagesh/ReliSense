"""Common style for all manuscript figures (v2.0, manuscript v3.26). Serif, 8 pt labels, 7 pt minimum for every text
element (enforced in save), thin lines, colour-blind-safe colours. Figures are drawn at the size at which they are printed:
COL2 = 7.0 in (text width of the manuscript page) or COL1 = 3.4 in, and the builder inserts them at that size."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COL1, COL2 = 3.4, 7.0                       # printed widths (in): single column, full text width
C = dict(rs='#0b3c6f', rf='#c8641e', wd='#7a7a7a', be='#3f7f3a', rule='#a0522d',
         H='#5b5b5b', OR='#0072b2', IR='#d55e00', B='#009e73', grid='#e4e4e4', ink='#1a1a1a', light='#bdbdbd')
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': False, 'ytick.right': False,
    'axes.spines.top': True, 'axes.spines.right': True, 'axes.edgecolor': '#1a1a1a', 'axes.labelcolor': '#1a1a1a',
    'xtick.color': '#1a1a1a', 'ytick.color': '#1a1a1a', 'lines.linewidth': 0.9, 'lines.markersize': 3.5,
    'legend.frameon': False, 'legend.handlelength': 1.6, 'axes.titleweight': 'normal', 'axes.titlepad': 3,
    'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'figure.dpi': 150})


def panel(ax, s, x=-0.02, y=1.02):
    """Panel label, IEEE style: (a) in roman at the upper left."""
    ax.text(x, y, s, transform=ax.transAxes, ha='right', va='bottom', fontsize=8)


MIN_PT = 7.0


def save(fig, name, min_pt=MIN_PT):
    """Raise every text element below min_pt to min_pt (ticks, legends, annotations, colour bars), then save."""
    import matplotlib.text as mtext
    n = 0
    for t in fig.findobj(mtext.Text):
        if t.get_text() and t.get_fontsize() < min_pt - 1e-6: t.set_fontsize(min_pt); n += 1
    fig.savefig(name); plt.close(fig)
    from PIL import Image
    w, h = Image.open(name).size
    print(f'written {name}: {w / 600:.2f} x {h / 600:.2f} in' + (f' ({n} text elements raised to {min_pt:g} pt)' if n else ''))
