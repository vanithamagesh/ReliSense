# ReliSense: physics-guided bearing fault diagnosis on unseen bearings

Code for the manuscript *"What transfers to an unseen bearing? Kinematic signatures, not waveforms: physics-guided
fault diagnosis with predictable failure modes on the Paderborn dataset"* (manuscript version 2.4).

Every experiment is split **by bearing**: no recording of a test bearing is ever used for training, feature scaling,
threshold setting or model selection. The primary model (15 kinematic envelope features + logistic regression,
C = 0.3) was fixed before testing and is used unchanged in every script.

## Repository layout

| Folder | Content |
|---|---|
| `pipeline/` | The 32-bearing study (all results of the manuscript except the sensor monitor). |
| `sensor_monitor_15/` | The earlier 15-bearing study: sensor-health monitor, sensor-fault injection, motor-current baselines. |
| `manuscript_figures/` | Scripts that draw the schematic and summary figures of the manuscript. |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
sudo apt-get install libarchive-dev        # system library needed by libarchive-c (Linux / Colab)
pip install -r requirements.txt
```

CPU is enough for everything except `phase4_learned.py` (WDCNN) and `sensor_monitor_15/study.py`, which should run on a GPU.
All scripts were run on Google Colab (Python 3, CPU runtime; GPU runtime for the networks).

## Data

The Paderborn bearing data (Lessmeier et al., 2016; KAt-DataCenter, Paderborn University) are licensed under
**CC BY-NC 4.0** and are not included. `pipeline/build_32.py` downloads the official archives one bearing at a time
from `https://groups.uni-paderborn.de/kat/BearingDataCenter/` and deletes each archive after processing.
Labels, damage levels and damage descriptions (`pipeline/labels_32.csv`) were taken from Tables 4, 5 and 7 of the
dataset paper. One recording, `N15_M01_F10_KA08_2.mat`, cannot be read and is skipped automatically (2,559 recordings).

## Reproducing the 32-bearing study

Run from inside `pipeline/` (the scripts import each other by module name). Each step writes its outputs to the
current folder; finished bearings and jobs are skipped, so a step can be restarted after a disconnect.

| Step | Command | Output | Manuscript |
|---|---|---|---|
| 1 | `python build_32.py --workers 3` | `pb32_bearings/<bearing>.npz` (windows + 64-kHz physics features) | Section 3.1 |
| 2 | `python merge_32.py` | `pb32_physics.npz`, `pb32_4096.npz`, `pb32_4096_envelope.npz` | Section 3.1.3 |
| 3 | `python check_labels.py` | per-bearing BPFO/BPFI summary (label check; labels are never changed) | Section 4.2.1 |
| 4 | `python phase2_physics.py` | L8, L10, A2R, LOBO for the physics models | Table 6 |
| 5 | `python phase3_analysis.py` | condition-0 protocols, confusion matrices, damage-level accuracy, selective decisions | Tables 10–12 |
| 6 | `python phase3b_ladder.py` and `python phase3b_ladder.py --cond0 --out phase3b_results_cond0.json` | 72-combination ladder with nested selection | Table 13, Fig. 31 |
| 7 | `python phase3c_stats.py` | bootstrap intervals over bearings, bearing-level diagnosis, abstention per bearing | Table 7, Fig. 30 |
| 8 | `python phase4_learned.py --lobo` (GPU) | WDCNN on raw vibration and envelope, all protocols | Tables 6 and 9 |
| 9 | `python phase5_figures.py` | data figures D1–D14 and `phase5_numbers.json` | data figures in Sections 3–4 |
| 10 | `python phase6_physics_evidence.py` | mechanism tests E1–E5 and `phase6_numbers.json` | Table 8, Figs. 18–22 |
| 11 | `python phase7_field_maps.py` | field maps F1–F4 and `phase7_numbers.json` | Figs. 10–12 and 24 |

Table and figure numbers refer to manuscript version 2.4.

### Main scripts

- `physics64.py`: squared envelope spectrum (2–12 kHz), kurtogram and pre-whitened variants, the 15 peak-to-background
  features at harmonics 1–3 of BPFO, BPFI, BSF, FTF and the shaft frequency, and the 590-bin order spectrum.
  6203 geometry: n = 8, d = 6.75 mm, D = 28.55 mm (fault orders 3.054, 4.946, 1.997, 0.382).
- `phase2_physics.py`: protocol definitions (L8 and L10 of Lessmeier et al., A2R, LOBO) with explicit bearing-overlap checks.
- `phase6_physics_evidence.py`: E1 fault-frequency perturbation (scaled and random fault orders), E2 speed transfer
  (train 1500 rpm, test 900 rpm), E3 representation ladder (order spectrum + LR, order-spectrum 1-D CNN, kinematic features),
  E4 classifier weights, E5 learning curves (1–5 training bearings per class, 30 draws).
- `phase7_field_maps.py`: signature map, envelope folding at the measured fault period, shaft-angle field, condition maps.

## 15-bearing study (sensor-health monitor)

```bash
cd sensor_monitor_15
python download_paderborn.py --records-per-condition 0
python relisense.py prepare --raw raw_paderborn --labels labels.csv --output paderborn_15bearings_4096.npz
python study.py --data paderborn_15bearings_4096.npz --out study_runs \
    --models relisense,physics_lr,wdcnn,envelope_rf,tcn,transformer --rotations 0,1,2,3,4 --seeds 0,1,2
python study.py --data paderborn_15bearings_4096.npz --out study_runs --protocol art2real --models relisense,physics_lr
python make_tables.py --runs study_runs --out tables
python make_figures.py --runs study_runs --data paderborn_15bearings_4096.npz --out figures
python test_contracts.py            # leakage, sensor-mask and converter checks
```

`relisense.py` also contains a `demo` command that runs on synthetic signals; it is a software test only and was
never used for any reported result.

## Manuscript figures

`manuscript_figures/fig_geometry.py` draws the bearing geometry and a schematic (simulated, labelled as such) of the
fault mechanism. `rs_figs.py` draws the framework, protocol and summary figures from values copied from the study
outputs above. `fig_p6.py` redraws the E1, E2, E3 and E5 figures from `phase6_numbers.json`
(copy the Step 10 output to `manuscript_figures/p6figs/phase6_numbers.json`).

## Citation and licence

Please cite the manuscript (details to be added on publication) and the dataset:
C. Lessmeier, J.K. Kimotho, D. Zimmer, W. Sextro, *Condition monitoring of bearing damage in electromechanical drive
systems by using motor current signals of electric motors: a benchmark data set for data-driven classification*,
PHM Society European Conference, 2016.

Code licence: to be chosen by the authors before the repository is made public (for example MIT or BSD-3-Clause).
The data licence (CC BY-NC 4.0) restricts use of the data to non-commercial purposes.
