# ReliSense: code for review

Physics-informed diagnosis of unseen rolling bearings (Paper A, current version v2.9). This folder holds the code and
the small result files. The data, checkpoints, manuscripts (.docx/.pdf) and figure archives stay on Google Drive
(`MyDrive/ReliSense_study`) and are not in Git (see `../.gitignore`).

## Pipeline (run in this order on Colab, CPU unless stated)

| Step | Script | What it does |
|---|---|---|
| Data | `experiments_archive/build_32.py`, `physics64.py`, `merge_32.py`, `check_labels.py` | Download the Paderborn archives, compute the 4 x 15 kinematic features and order spectra per 4-s recording, write `pb32_physics.npz`; check labels against Lessmeier et al. 2016 |
| Early phases | `experiments_archive/phase2_physics.py` … `phase8_modern.py`, `phase12_damage_analysis.py` | Bearing-wise protocols (L8, L10, A2R, LOBO), statistics, WDCNN and other learned baselines, physics controls |
| Further rigs | `experiments_archive/phase10_benchmarks.py` | CWRU, HUST, Ottawa segments and features (settings fixed before results) |
| Figures | `phase14_pipeline_stages.py`, `phase15_speed_normalisation.py`, `phase16_tsne.py` | Real Module I outputs, speed normalization, t-SNE |
| Method | `phase17_improve.py`, `phase19_extended_kinematics.py`, `phase20_envelope_branch_external.py`, `phase21_extended_external.py` | Kinematic branch (31 features) + envelope branch (RF, 62 features); inner-LOBO abstention thresholds; checkpoints in `checkpoints/phase19` |
| Final numbers | `phase22_final_method_numbers.py` | Numbers and figure data of the final ReliSense from the phase 19 checkpoints |
| Baselines | `phase23_deep_baselines.py` (GPU), `phase24_benchmark_baselines.py` | Deep and physics-informed networks; tuned classical baselines and domain-generalization networks (ERM, DANN, Deep CORAL, group DRO) |
| Analyses | `phase25_matched_coverage.py`, `phase26_relisense_v2.py`, `phase26b_balanced_summary.py` | Matched-coverage abstention; equal-prior decision |
| Pre-specified changes | `phase27_condition_aware.py`, `phase28_relisense_plus.py` (v1.1) | Condition-wise standardization, multiband, comb-separated and slip-consistent features, decisions from several recordings; adoption rule fixed in advance (result: not adopted) |

Older versions are kept with the suffix `_v1_0`. Colab call, for example:

```
!python /content/drive/MyDrive/ReliSense_study/phase28_relisense_plus.py --root /content/drive/MyDrive/ReliSense_study --no-rf-cw
```

## Result files

`phase23_results.json`, `phase27_results.json`, `phase28_results.json` (real Colab runs) and the pasted console outputs
`phase2x_*_pasted.txt`; `phase28_summary_v1_0.txt` is computed from `phase28_results.json`.

## Manuscript builders

`make_A24.py` … `make_A29.py` build Paper A v2.4–v2.9 from the previous .docx. Every table value is read from the
result files, and `make_A28.py`/`make_A29.py` stop if a rewritten paragraph contains a number that the original
paragraph did not contain. `make_B2.py`, `make_B21.py`, `make_split.py`, `build_v31x.py` and `fig_*.py` belong to
earlier versions and Paper B.

## Review checklist (for Antigravity or any reviewer)

Report problems; do not rewrite.

1. **Leakage:** every split is by bearing; no recording, normalization statistic, threshold or fitted parameter of a
   test bearing is used in training; the inner leave-one-bearing-out loop that sets the abstention thresholds uses
   training bearings only.
2. **Labels:** K001–K006 healthy, KA outer race, KI inner race; KB23, KB24, KB27 (both rings) excluded; damage
   properties as in `labels_32.csv` (Lessmeier et al. 2016, Tables 4, 5 and 7).
3. **Checkpoints:** a resumed run cannot mix results from different settings (phase 28 v1.1 rebuilds all fusions from
   the stored branch posteriors).
4. **Metrics:** accuracy, balanced accuracy, false alarms (healthy diagnosed as damaged), missed faults (damaged
   diagnosed as healthy), coverage and selective accuracy as defined in the docstrings.
5. **Adoption rule** (phase 28): criteria (i)–(iii) implemented as stated.
6. **Reproducibility:** the printed numbers can be regenerated from the JSON and checkpoint files.
