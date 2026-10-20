# Paper B: Assessing Bearing Fault Evidence Through Kinematic Specificity and Spectral Contrast

Code, result files and label sheet of Paper B (release `paperB-v1.0`). Every table and figure of the paper and its
supplement is listed below with the script that computes or draws it and the result file it reads. All result files
are the outputs of the runs reported in the paper, unchanged, except the two CSV files marked *derived*, which are
regenerated from the stored results by the scripts named.

## Contents

| Folder | What it holds |
|---|---|
| `scripts/pipeline/` | Data build and analysis phases (Paderborn features, protocols, baselines, external rigs). They import each other, so keep them in one folder, as in the original runs. |
| `scripts/evidence/` | The two evidence scripts of the paper (perturbation, contrast, intervention, healthy-state analyses) and two small scripts that write the derived CSV files. |
| `scripts/figures/` | Scripts that draw the data figures at 600 dpi from `results/`. |
| `results/` | Result files of every run used in the paper, the label sheet `labels_32.csv` and the derived CSV files. |

## Data

* **Paderborn** (Lessmeier et al., KAt-DataCenter, Paderborn University, CC BY-NC 4.0). `build_32.py` downloads each
  bearing archive and computes the features with `physics64.py`; `merge_32.py` joins them into `pb32_physics.npz`
  and the window files. These data files are not included because of their size and the dataset licence; they are
  rebuilt from the public archives by the two scripts.
* **CWRU** and **HUST**: `phase10_benchmarks.py` reads the public files and writes `phase10_results.json`; the
  segment features used by phase 12 are in `results/cwru_features.npz` and `results/hust_features.npz`.
* `labels_32.csv`: damage attributes of the 32 Paderborn bearings from Lessmeier et al. (2016), Tables 4, 5 and 7.

## Two implementations of the probe

| | Original implementation | Second implementation |
|---|---|---|
| Code | `physics64.py`, `phase2_physics.py`, `ReliSense_PaperB_evidence_v3_1.py` | `ReliSense_PaperB_evidence_v2.py` |
| Band-pass 2–12 kHz | Ideal filter in the frequency domain | Fourth-order zero-phase Butterworth |
| Spectrum, logarithm | Amplitude, natural log | Power, log10 |
| Contrast Γ | Units of (24) | Paderborn scale, (24) divided by 1.151 |

The order grid of `physics64.py` has 591 points from 0.20 to 12.00 in steps of 0.02 (the inline comment
"590 bins" in the code is a leftover; the grid itself is as stated). The paper names the implementation of every
number; Table S1 lists the differences.

## Run order

1. `scripts/pipeline/`: `build_32.py` → `merge_32.py` → `check_labels.py` → `phase2_physics.py` →
   `phase3b_ladder.py` → `phase4_learned.py` → `phase5_figures.py` → `phase6_physics_evidence.py` →
   `phase10_benchmarks.py` → `phase11_analysis.py` → `phase12_damage_analysis.py` → `phase13_figure_data.py` →
   `phase16_tsne.py` → `phase17_improve.py` → `phase19_extended_kinematics.py` →
   `phase20_envelope_branch_external.py` → `phase21_extended_external.py` → `phase22_final_method_numbers.py`.
   Each script states its inputs, outputs and command line in its header. The runs were made on Google Colab.
2. `scripts/evidence/`: `ReliSense_PaperB_evidence_v2.py` (writes `summary_evidence_v2.txt`,
   `perturbation_v2.csv`, `contrast_paderborn_v2.csv`) and `ReliSense_PaperB_evidence_v3_1.py` (writes
   `summary_evidence_v3.txt`, `perturbation_full_v3.csv`, `contrast_original_units_v3.csv`). Set `PB_DATA_DIR` to
   the folder with the Paderborn `.mat` files and `PB_OUT_DIR` to the output folder. The section keys of the
   summaries ([E2-0] … [E2-10], [E3-0] … [E3-8]) are used in the tables below.
3. Derived CSV files (seconds, no data needed), from `scripts/evidence/`:
   ```
   python make_matched_draws_csv.py ../../results/ReliSense_PaperB_matched_draws_v1.csv
   python make_external_contrast_csv.py ../../results/phase12_numbers.json ../../results/phase10_results.json ../../results/external_contrast_accuracy.csv
   ```
4. Figures: copy `scripts/figures/*.py` into a folder with the contents of `results/` and run each script there,
   e.g. `python fig_B_core.py`. `make_fig12_v11_64.py` and `make_fig13_v11_70.py` take their file names as
   arguments (see their headers).

## Tables of the paper

| Table | Content | Result file (section) | Script |
|---|---|---|---|
| I | Bearing-wise protocols | defined in code | `phase2_physics.py` (L8, L10, A2R, LOBO), `phase10_benchmarks.py` (HUST, CWRU) |
| II | Representation ladder, bearing identity, MMD ratio | `phase11_numbers.json`; order spectrum + CNN: `phase6_numbers.json` (`E3_ladder`) | `phase11_analysis.py`, `phase6_physics_evidence.py` |
| III | Healthy-state errors of probe and threshold rule | own point: `phase2_per_bearing.json`; 13.1% rows: `summary_evidence_v2.txt` [E2-6]; calibrated threshold: `summary_evidence_v3.txt` [E3-6] | `phase2_physics.py`, evidence v2, evidence v3.1 |
| IV | Healthy training bearings and accuracy on real damage | `summary_evidence_v3.txt` [E3-5] | evidence v3.1 |
| V | Findings, interpretations, reporting items | summary of the results in this list | – |

## Figures of the paper

| Figure | Content | Result file (section) | Script (output) |
|---|---|---|---|
| 1–4, 7, 18 | Schematics | – | no result data |
| 5 | Search bands on a schematic spectrum | bands as in Table S4 | `physics64.py` (band definition) |
| 6 | True, scaled and random windows on the KA16 order spectrum | `phase13_figdata.npz`, `perturbation_full_v3.csv` (sets `true`, `s=0.90`, `rand1`) | values only |
| 8 | (a), (b) perturbation of the 11-feature variant; (c) per-bearing margins of the full probe | (a), (b): `phase6_numbers.json` (`E1_scaled`, `E1_random`); (c): `summary_evidence_v3.txt` [E3-1] | `phase6_physics_evidence.py`, evidence v3.1 |
| 9 | Signature map | `phase13_figdata.npz` | `fig_real.py` (`fig_sigmap.png`) |
| 10 | Probe weights | `phase6_numbers.json` (`E4_weights`) | `fig_user.py` (`fig_weights.png`) |
| 11 | External data: HUST by type, accuracy against Γ | `phase12_numbers.json`, `phase10/20/21_results.json`, `external_contrast_accuracy.csv` | `fig_B_core.py` (`fig_line_B.png`) |
| 12 | Kinematic features and t-SNE of the 29 bearings | `phase22_views.npz`, `phase16_tsne.npz` | `make_fig12_v11_64.py` |
| 13 | LOBO accuracy per bearing and model | `phase2_per_bearing.json`, `phase19_results.json`, `wdcnn_env_lobo.json`, `labels_32.csv` | `make_fig13_v11_70.py` |
| 14 | Operating condition | `phase5_numbers.json`, `phase6_numbers.json` (`E2_speed_transfer`) | `fig_B_core.py` (`fig_conditions_B.png`) |
| 15 | Contrast at 1500 and 900 rpm | `contrast_paderborn_v2.csv` (rows C0 and 900rpm: `gamma_median`, `lobo_acc_condition`), [E2-5] | evidence v2 (values); its draft plot is `fig_speed_contrast_v2.png` |
| 16 | Data efficiency | `phase6_numbers.json` (`E5_data_efficiency`) | `fig_user.py` (`fig_efficiency.png`) |
| 17 | Accuracy on unseen bearings in five settings | `phase19/20/21_results.json`, `phase22_numbers.json`, `phase22_figdata.npz` | `fig_disc_inputs_B.py` |

For Figs. 6, 8 and 15 the files above hold every plotted value; the final layout of these three figures was
prepared outside these scripts.

## Numbers quoted in the text

| Number | Result file (section) |
|---|---|
| LOBO 80.1%, false alarms 41.0%, damaged bearings 85.6% (Fig. 13 run) | `phase2_per_bearing.json` (`lr_fixed`), `phase19_results.json` |
| LOBO 80.2%, false alarms 40.8% (original implementation, evidence run) | `summary_evidence_v3.txt` [E3-0], [E3-1], [E3-5] |
| LOBO 80.9%, false alarms 39.4% (second implementation) | `summary_evidence_v2.txt` [E2-0] |
| Perturbation margins, Δrand, P_rank = 0.017, capture analysis | `summary_evidence_v3.txt` [E3-1], [E3-2]; `perturbation_full_v3.csv` |
| Intervention: damaged-bearing accuracy 85.6% → 39.1% | `summary_evidence_v3.txt` [E3-3] |
| Γ 0.62 → 0.35 and accuracy 93.3% → 73.7% from 1500 to 900 rpm | `summary_evidence_v2.txt` [E2-5]; `contrast_paderborn_v2.csv` |
| Contrast at 60 revolutions | `summary_evidence_v3.txt` [E3-7] |
| Probe on seven 0.512-s windows: 78.6% | `summary_evidence_v3.txt` [E3-8] |
| External: Spearman ρ = 0.704, p = 0.0023 (16 bearings); all 11 bearings with Γ > 1.1 correct | `external_contrast_accuracy.csv` (*derived*), `phase12_numbers.json` |
| Log-odds and contributions of K004 and K005; shaft-harmonic audit | `summary_evidence_v2.txt` [E2-6], [E2-4] |
| Bearing identity | `summary_evidence_v2.txt` [E2-7]; `phase11_numbers.json` |

The perturbation statistics use 30 wrong sets (the ten scaled sets and the first 20 random sets) for the margin
against the best wrong set, the mean of these 20 random sets for Δrand, and all 120 random sets
(`numpy.random.default_rng(0)`, orders uniform on 0.3–5.9) for P_rank and the capture analysis.

## Supplement

| Item | Result file (section) | Script |
|---|---|---|
| Table S1 | `summary_evidence_v3.txt` [E3-0] | `physics64.py`, evidence v2 and v3.1 |
| Table S2 | `phase10_results.json` | `phase10_benchmarks.py` |
| Table S3 | defined in code | `phase2_physics.py` (L10) |
| Table S4 | defined in code | `physics64.py` |
| Fig. S1 | `phase13_figdata.npz`; peak factors `summary_evidence_v2.txt` [E2-10] | `fig_real.py` (`fig_folding.png`) |
| Table S5 | `ReliSense_PaperB_matched_draws_v1.csv` (*derived*; 300 draws), [E2-8] | `make_matched_draws_csv.py` (same seed and draw order as [E2-8] of evidence v2) |
| Table S6 | settings | `phase2_physics.py`, `phase4_learned.py`, `phase10_benchmarks.py` |
| Table S7 | `summary_evidence_v3.txt` [E3-2] | evidence v3.1 |
| Table S8 | `phase11_numbers.json` | `phase11_analysis.py` |
| Table S9 | `external_contrast_accuracy.csv` (*derived*) | `make_external_contrast_csv.py` |
| Table S10 | `phase12_numbers.json` (`tables`) | `phase12_damage_analysis.py` |
| Table S11 | `contrast_paderborn_v2.csv`, [E2-5] | evidence v2 |
| Fig. S2 | `phase16_tsne.npz` | `fig_tsne.py` |
| Fig. S3 | `phase5_numbers.json`, `phase22_numbers.json` | `fig_B_core2.py` (`fig_disc_a2r_B.png`) |

## Requirements

Python 3 with the packages in `requirements.txt`. `torch` is needed only for the networks of
`phase4_learned.py` and `phase6_physics_evidence.py`, and `libarchive-c` only for reading the Paderborn archives
in `build_32.py`.

## Citation

Please cite the paper when using this code or these results. The Paderborn data are subject to their own licence
(CC BY-NC 4.0).
