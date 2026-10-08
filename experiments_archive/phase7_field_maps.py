"""Phase 7 (v1.0): 2-D field maps of the bearing signatures (real data, CPU, a few minutes).

F1 signature map      median log squared-envelope spectrum of every bearing (rows) over order (columns)
F2 folding fields     envelope folded at the fault period: rows = successive impacts, columns = phase within one
                      period. Correct kinematics -> impacts line up as a vertical ridge; a period 10 % off -> they
                      drift into diagonals. The exact period is the envelope-spectrum peak within +-2 % of the
                      theoretical fault frequency (rolling-element slip makes the true rate slightly lower).
F3 shaft-angle field  envelope over shaft angle (0-360 deg) for successive revolutions (inner-race load-zone modulation)
F4 condition maps     mean signature per operating condition (rows) and class (panels)
"""
from __future__ import annotations
import argparse, csv, json, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import physics64 as P

INK, MUTED = '#222222', '#5f5e58'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
COND0 = 'N15_M07_F10'; CONDS = ['N15_M07_F10', 'N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04']
CMAP = 'viridis'; FS = 8000.0


def recording_envelope(env, wrec, rec):
    """Concatenate the consecutive 8-kHz envelope windows of one recording (windows are stored in time order)."""
    return env[wrec == rec, 0].reshape(-1).astype(float)


def exact_rate(e, f0, tol=0.02):
    """Envelope-spectrum peak within +-tol of the theoretical rate f0 (Hz)."""
    e = e - e.mean(); n = 16 * len(e)                       # zero padding: finer frequency grid for the peak
    S = np.abs(np.fft.rfft(e * np.hanning(len(e)), n)); f = np.fft.rfftfreq(n, 1 / FS)
    m = np.abs(f - f0) <= tol * f0
    return float(f[m][np.argmax(S[m])]) if m.any() else f0


def fold(e, rate, n_cycles=120, n_phase=64):
    T = 1.0 / rate; n = min(n_cycles, int(len(e) / FS / T) - 1)
    t = (np.arange(n)[:, None] + (np.arange(n_phase)[None, :] + 0.5) / n_phase) * T
    F = np.interp(t * FS, np.arange(len(e)), e)
    return F / (np.median(F) + 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--physics', default='pb32_physics.npz')
    ap.add_argument('--envelope', default='pb32_4096_envelope.npz')
    ap.add_argument('--labels', default='labels_32.csv')
    ap.add_argument('--out', default='phase7_fields')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); R = {}
    d = dict(np.load(a.physics)); meta = {r['bearing']: r for r in csv.DictReader(open(a.labels))}
    B, y, cond, rec, rpm = d['bearing'], d['label'], d['condition'], d['recording'], d['rpm']
    grid = d['order_grid']; O = d['order_fixed'].astype(float); fo = P.fault_orders()
    save = lambda fig, n: (fig.savefig(out / f'{n}.png'), plt.close(fig))
    order_b = [b for b in meta if b in set(B)]

    # ---------------- F1 signature map (median over each bearing's recordings)
    M = np.stack([np.median(O[B == b], 0) for b in order_b])
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    im = ax.imshow(np.clip(M, -0.5, 3.0), aspect='auto', cmap=CMAP, extent=[grid[0], grid[-1], len(order_b) - 0.5, -0.5], interpolation='nearest')
    ax.set_yticks(range(len(order_b))); ax.set_yticklabels([f"{b} ({ {'0': 'H', '1': 'OR', '2': 'IR', '3': 'OR+IR'}[meta[b]['label']] })" for b in order_b], fontsize=6)
    for k, ls, c in (('BPFO', '--', 'white'), ('BPFI', ':', '#ffb000')):
        for h in (1, 2, 3):
            if h * fo[k] <= grid[-1]: ax.axvline(h * fo[k], color=c, lw=0.8, ls=ls)
    for yv in (5.5, 17.5, 28.5): ax.axhline(yv, color='white', lw=0.8)
    ax.set_xlabel('order (multiples of shaft frequency)'); cb = fig.colorbar(im, ax=ax, shrink=0.9); cb.set_label('log SES / median')
    ax.set_title('dashed white: BPFO harmonics    dotted yellow: BPFI harmonics', fontsize=7.5, loc='left')
    save(fig, 'F1_signature_map')

    # ---------------- F2 folding fields
    e_npz = np.load(a.envelope); env, wrec = e_npz['X'], e_npz['recording']
    def first_rec(b):
        m = np.flatnonzero((B == b) & (cond == COND0)); return (rec[m[0]], float(rpm[m[0]])) if len(m) else (None, None)
    cases = [('K001', 'BPFO', 'healthy K001, folded at BPFO'), ('KA16', 'BPFO', 'outer race KA16, folded at BPFO'),
             ('KI18', 'BPFI', 'inner race KI18, folded at BPFI')]
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 5.6)); R['F2'] = {}
    for j, (b, k, title) in enumerate(cases):
        r_, rp = first_rec(b)
        if r_ is None: continue
        e = recording_envelope(env, wrec, r_); f0 = fo[k] * rp / 60.0; fx = exact_rate(e, f0)
        R['F2'][b] = {'theoretical_Hz': round(f0, 3), 'exact_Hz': round(fx, 3), 'ratio': round(fx / f0, 4)}
        for i, (rate, lab) in enumerate(((fx, 'correct period'), (fx * 1.10, 'period 10% off'))):
            Fm = fold(e, rate); ax = axs[i, j]
            im = ax.imshow(np.clip(Fm, 0, np.percentile(Fm, 99)), aspect='auto', cmap=CMAP, extent=[0, 1, Fm.shape[0], 0])
            ax.set_title(f'{title}\n{lab}' if i == 0 else lab, fontsize=7)
            ax.set_xlabel('phase within one fault period' if i == 1 else ''); ax.set_ylabel('impact number' if j == 0 else '')
            fig.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(labelsize=6)
    fig.tight_layout(); save(fig, 'F2_folding_fields')

    # ---------------- F3 shaft-angle field
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, b in zip(axs, ('KI18', 'K001')):
        r_, rp = first_rec(b)
        if r_ is None: continue
        e = recording_envelope(env, wrec, r_); fr = exact_rate(e, rp / 60.0, 0.03)
        Fm = fold(e, fr, n_cycles=80, n_phase=180)
        im = ax.imshow(np.clip(Fm, 0, np.percentile(Fm, 99)), aspect='auto', cmap=CMAP, extent=[0, 360, Fm.shape[0], 0])
        ax.set_xlabel('shaft angle (deg)'); ax.set_ylabel('revolution'); ax.set_title(f'{b} ({meta[b]["origin"]})', fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.85, label='envelope / median')
    fig.tight_layout(); save(fig, 'F3_shaft_angle_field')

    # ---------------- F4 condition maps per class
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True)
    for k, (ax, nm) in enumerate(zip(axs, ('healthy', 'outer race', 'inner race'))):
        Mc = np.stack([O[(y == k) & (cond == c)].mean(0) if ((y == k) & (cond == c)).any() else np.zeros(len(grid)) for c in CONDS])
        im = ax.imshow(np.clip(Mc, -0.3, 2.0), aspect='auto', cmap=CMAP, extent=[grid[0], grid[-1], 3.5, -0.5], interpolation='nearest')
        ax.set_yticks(range(4)); ax.set_yticklabels(CONDS, fontsize=6.5); ax.set_title(nm, fontsize=8, loc='left')
        kk = 'BPFO' if k == 1 else 'BPFI' if k == 2 else None
        if kk:
            for h in (1, 2):
                if h * fo[kk] <= grid[-1]: ax.axvline(h * fo[kk], color='white', lw=0.8, ls='--')
        fig.colorbar(im, ax=ax, shrink=0.9).ax.tick_params(labelsize=6)
    axs[-1].set_xlabel('order (multiples of shaft frequency)'); fig.tight_layout(); save(fig, 'F4_condition_maps')

    (out / 'phase7_numbers.json').write_text(json.dumps(R, indent=1))
    shutil.make_archive(str(out), 'zip', out)
    print(json.dumps(R, indent=1)); print('written', out, str(out) + '.zip')


if __name__ == '__main__':
    main()
