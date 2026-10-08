"""Paper A v2.8 from v2.7: the text is condensed to about 9,800 words (all paragraph text, including references and
captions; tables not counted). No result, table, figure or reference is removed; the 7) Properties paragraph of
Module I, which repeated the introduction, is deleted. Every number in a rewritten paragraph must already occur in the
paragraph it replaces (checked below; derived values are listed explicitly), so no value can change by accident.
  python3 make_A28.py <v2_7.docx> <out_clean.docx> <out_marked.docx>
Marked copy: red strikethrough = deletion, blue = addition."""
import sys, copy, re, difflib
from docx import Document
from docx.shared import RGBColor

src, out_clean, out_marked = sys.argv[1:4]
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
RED, BLUE = RGBColor(255, 0, 0), RGBColor(0, 0, 255)

# (start of the paragraph, new text [without the italic run-in heading], derived numbers allowed in the new text)
REWRITE = [
 ('Paper A, version 2.7', 'Paper A, version 2.8, 8 October 2026', {'2.8'}),
 ('Abstract—',
  'Abstract—A bearing fault classifier in service must diagnose bearings that it has never seen, although public datasets contain few physical bearings per class. This article proposes ReliSense, a physics-informed framework that reads the squared envelope spectrum only at the fault orders fixed by bearing geometry and measured shaft speed. A 96-parameter logistic regression and a random forest on four demodulation bands classify these features, their posteriors are averaged, and low-confidence recordings are withheld with thresholds set on training bearings only. On 29 Paderborn bearings, which also served for method development, ReliSense reached 95.9% when trained on artificial and tested on real damage, against 75.0% for the published ensemble and at most 70.8% for convolutional, residual, physics-informed and domain-generalization networks. In leave-one-bearing-out evaluation, it reached 83.7%; abstention raised this to 97.5% at 69% coverage and lowered the false alarms from 32.1% to 2.7%. Equal class priors, applied without retraining, lowered the false alarms before abstention to 15.6% with 11.2% missed faults, and a decision from ten consecutive recordings to 6.2% with 8.7% missed faults. With the procedure fixed in advance and retrained on each dataset, ReliSense reached 85.0% on unseen bearing types of the HUST dataset and 83.2% on unseen fault sizes of the CWRU dataset. Healthy bearings that resemble damage remain the main source of error.', set()),
 # ---------------------------------------------------------------- I. Introduction
 ('Rolling-element bearings support',
  'Rolling-element bearings support almost every rotating shaft in industrial machinery, and their failure is a leading '
  'cause of unplanned downtime [1]. A localized defect produces a short impact each time a rolling element passes over it, '
  'at a rate fixed by the shaft speed and the bearing geometry. The impacts excite structural resonances, and demodulating '
  'the resonant band turns their repetition rate into peaks in the envelope spectrum [2]. Because these lines are located '
  'by kinematics alone, envelope analysis is the industrial reference for bearing diagnosis.', set()),
 ('Data-driven diagnosis now dominates',
  'Data-driven methods now dominate research on this task [3]–[5]. Convolutional networks such as WDCNN [6], transformers [7] and transfer learning [8] learn features directly from vibration, and reviews report accuracies close to 100% on public benchmarks [9]. These accuracies depend on how the data are split. On the Paderborn dataset [10], [11], the splitting method changed accuracy by more than 40%, and ten of 55 studies were judged likely to be affected by data leakage [12]. When training and test bearings are kept apart, the published accuracy for transfer from artificial to real damage falls to 62.3%–65.9% for single classifiers and 75.0% for an ensemble. A monitoring system in service always meets bearings that it has not seen, so this is the setting that matters.', set()),
 ('Diagnosing an unseen bearing poses',
  'Diagnosing an unseen bearing poses three difficulties. First, public datasets contain many recordings of few physical '
  'bearings, six or seven per class on Paderborn, so a model can separate its training bearings by properties unrelated to '
  'the damage, such as mounting or sensor gain [13], [14]; the number of distinct training bearings largely decides '
  'generalization [15]. Second, training data may contain only artificial damage, whereas damage in service grows by '
  'fatigue or plastic deformation. Third, small or distributed damage produces weak impacts, which lower speed or load '
  'weakens further [16], [17], and their frequency moves with speed.', set()),
 ('Existing approaches act at different stages',
  'Signal-processing methods select a demodulation band with the kurtogram [18], combined indicators [19], [20] or cepstral pre-whitening [21], [22]; they are interpretable, but the selected band is not always the one excited by the defect, and the decision usually requires an expert. Data-driven methods learn the whole chain, from fault-adapted kernels [23] to transfer learning and domain generalization [24]–[26], and transfer from artificial to natural damage has used data of the target domain [27]. Physics-informed networks add characteristic-frequency priors [28], modal properties [29], fault-frequency-guided decomposition [30] or interpretable filtering layers [31], [32], and fault orders have been used to weight envelope features [33]. Uncertainty estimates and selective prediction aim at trustworthy diagnoses [34]–[37].', set()),
 ('In this study, we propose ReliSense',
  'This study proposes ReliSense, which reads the squared envelope spectrum only at the fault orders fixed by bearing geometry and measured shaft speed. Two branches classify these features, their posteriors are averaged, and diagnoses below a threshold set on training bearings only are withheld. Physical knowledge thus enters through the input representation rather than the loss [39]. ReliSense is evaluated under bearing-wise protocols on Paderborn and, with all settings fixed in advance, on three further test rigs.', set()),
 ('Fault-order analysis is established',
  'Fault-order analysis itself is established. The novelty lies in combining fault-order features, two complementary '
  'demodulation branches and a bearing-wise abstention rule whose threshold needs no data of the new bearing. The main '
  'contributions are as follows.', set()),
 ('A fault-order representation for unseen bearings',
  'A fault-order representation for unseen bearings. Every input is read at an order computed from the geometry and the '
  'measured speed, so one feature describes one physical event on every bearing. Besides the peaks at three harmonics of '
  'five fault orders, it includes 16 sideband and higher-harmonic features predicted by the defect kinematics.', set()),
 ('A two-branch architecture',
  'A two-branch architecture. A 96-parameter logistic regression classifies the features with weights that can be traced '
  'to individual fault orders, and a random forest reads the same peaks in four demodulation bands for the case in which '
  'the fixed band misses the excited resonance. Averaging lowers the confidence when the two readings disagree.', set()),
 ('A bearing-wise abstention rule',
  'A bearing-wise abstention rule that needs no data of the new bearing. The threshold of each test bearing is set for a '
  'target coverage by an inner leave-one-bearing-out loop over the training bearings, so the balance between referrals '
  'and errors is set from training data alone.', set()),
 # ---------------------------------------------------------------- II. ReliSense (plain paragraphs only)
 ('Fig. 2. Kinematics of the 6203',
  'Fig. 2. Kinematics of the 6203 bearing. (a) Rolling element between the rings (φ = 0°). (b) Load-zone function q(θ) of '
  '(4) for ε = 0.5 and 0.3. (c) Search windows of the five orders and three harmonics on the order axis; dotted: the two '
  'overlapping pairs.', set()),
 ('Fig. 5 tests the normalization',
  'Fig. 5 tests the normalization on real recordings of KA16, KI18 and K001 at 900 and 1500 rpm. On the frequency axis, '
  'the fault lines move with speed; on the order axis, they stay at the fault orders. Read at the frequencies of 1500 rpm '
  'instead of the measured speed, the mean first-harmonic feature at 900 rpm fell from 3.30 to 1.90 for KA16 and from 2.66 '
  'to 0.65 for KI18, the level of the healthy K001 (0.48–0.63). The fault features were also lower at 900 rpm than at '
  '1500 rpm, in agreement with weaker impacts at lower speed.', set()),
 ('Fig. 4. Real output of each stage',
  'Fig. 4. Real output of Module I for recording N15_M07_F10_KA16_1 (real outer-race damage, 1500 rpm). (1) Raw '
  'vibration, first 100 ms. (2) Amplitude spectrum with the 2–12-kHz band. (3) Squared envelope. (4) Envelope spectrum '
  '(Δf = 0.25 Hz); dotted: first three BPFO harmonics. (5) Measured shaft speed. (6) Envelope spectrum on the order axis, '
  'with the healthy K001. (7) Peak and background windows around BPFO. (8) The 15 features of the recording.', set()),
 ('Fig. 5. Speed normalization',
  'Fig. 5. Speed normalization on real Paderborn recordings at 900 and 1500 rpm. Top: squared-envelope spectra; ticks: '
  'first three fault harmonics at each speed. Middle: the same spectra on the order axis; dotted: fault orders. Bottom: '
  'feature at the first harmonic of the fault order (mean ± SD of 20 recordings per condition).', set()),
 ('The branch has 3 × 31 weights',
  'The branch has 3 × 31 weights and 3 biases, that is, 96 trained parameters. A linear model was chosen deliberately: '
  'with six or seven bearings per class, every additional degree of freedom is an opportunity to learn the identity of '
  'the training bearings. The envelope branch is a random forest of 500 trees (seed 0) on e, which can combine the '
  'demodulation bands nonlinearly. The two posteriors are averaged,', set()),
 ('Fig. 7. LOBO posteriors',
  'Fig. 7. LOBO posteriors of ReliSense for the 2319 recordings in the probability simplex. (a) Coloured by true class '
  '(grey: H, blue: OR, orange: IR). (b) Decision at the mean training-only threshold for κ = 0.7; shaded: withheld '
  'region.', set()),
 ('Fig. 9 summarizes the procedure',
  'Fig. 9 summarizes the procedure: the standardization, both branches and the threshold are computed offline from the '
  'training bearings, and a new bearing needs one recording, its shaft speed and its geometry.', set()),
 # ---------------------------------------------------------------- III. Experimental setup
 ('Every split is defined by bearing identifiers',
  'Every split is defined by bearing identifiers; normalization statistics, thresholds and fitted parameters come from the training bearings only. Because the feature design was informed by Paderborn bearings, the Paderborn results are development-stage evidence. Four protocols are used (Fig. 11). L8, the artificial-to-real protocol of the dataset authors, trains on K002, KA01, KA05, KA07, KI01, KI05 and KI07 and tests on K001 and ten real-damage bearings, at condition 0 and over all conditions. L10, their real-damage protocol, splits 15 bearings into three training and two test bearings per class in 10 combinations. A2R trains on K001–K003 and the 12 artificially damaged bearings and tests on K004–K006 and the 11 real single-damage bearings. LOBO holds out each of the 29 bearings once. The rig, sensor position and bearing type are shared in all protocols.', set()),
 ('Table II lists the compared methods',
  'Table II lists the compared methods; all settings were fixed in advance. Besides ReliSense, its kinematic core, its two branches and a threshold rule are evaluated. WDCNN and a one-dimensional residual network (ResNet-1D) process standardized 4096-sample windows of raw vibration, down-sampled to 8 kHz, or of the 2–12-kHz envelope, and average their window probabilities per recording. Of two physics-informed networks, an impulse-response kernel network replaces the first layer of a WDCNN-type network by 32 learnable kernels h(t) = exp(−ξωt) sin(ωt), the impulse response of (3), and a fault-order-guided ResNet receives the envelope order spectrum (7) with a second channel that marks the fault-order windows. Two further groups had their settings selected on the training bearings only: tuned feature baselines, whose classifier and hyperparameters were chosen by stratified group K-fold (not on L8, whose single healthy training bearing cannot be held out), and domain-generalization networks, which treat every training bearing as a domain and were trained with class-weighted cross-entropy (ERM), domain-adversarial training (DANN) [41], Deep CORAL [42] or group DRO [43], on envelope windows or on the order spectrum.', set()),
 ('Recording accuracy is reported',
  'Recording accuracy is reported, with balanced accuracy, macro F1-score and the macro area under the ROC curve (AUC) '
  'where needed. Because recordings of one bearing are not independent, 95% intervals resample test bearings within each '
  'class (2000 resamples) [44], and methods are compared per bearing with the Wilcoxon signed-rank test. A bearing is '
  'correctly diagnosed when the class with the highest mean probability over its recordings is correct. The false-alarm '
  'rate is the share of healthy recordings diagnosed as damaged, and the missed-fault rate the share of damaged '
  'recordings diagnosed as healthy. Coverage is the share of accepted recordings and selective accuracy the accuracy on '
  'them. The p-values are descriptive.', set()),
 # ---------------------------------------------------------------- IV. Paderborn results
 ('Table III compares all methods',
  'Table III compares all methods. On L8, ReliSense reached 95.9% at condition 0 and 87.0% over all conditions, against 62.3%–65.9% for the published single classifiers, 75.0% for their ensemble and 70.8% for WDCNN on the envelope. At condition 0, all nine errors among the 220 test recordings were inner-race recordings diagnosed as outer-race damage (Fig. 12(a)). The bearing-level 95% interval (88.6%–100%) lies above the published ensemble value; this is not a formal test, because that value has no interval and L8 has only one healthy test bearing. On L10, the published 98.3% exceeds every result of this study; Vieira et al. [15], using bearing-wise splits, could not reproduce it.', set()),
 ('ReliSense was the most accurate method under all five',
  'ReliSense was the most accurate method under all five protocols, with 78.7% on L10, 77.7% on A2R and 83.7% under LOBO, but its margins over its own branches were small: at most 1.5 points for the kinematic branch on L8, L10 and A2R and 1.4 points for the envelope branch on A2R and LOBO. WDCNN and the four further networks were less accurate than ReliSense, its two branches and its kinematic core under every protocol. Under LOBO, ReliSense diagnosed 27 of the 29 bearings correctly (95% interval 76.5%–90.6%; macro AUC 95.7%; balanced accuracy 81.2%; macro F1-score 81.5%). Against WDCNN, it was more accurate on 16 bearings and less accurate on 6 (mean difference 15.6 points, 95% interval 6.2–25.3; Wilcoxon p = 0.009).', {'1.5'}),
 ('The residual network improved on WDCNN',
  'The residual network improved on WDCNN only on L8 over all conditions with envelope input; with raw vibration, it diagnosed every healthy L8 recording as damaged, as did the impulse-response kernel network. The fault-order-guided ResNet was the most accurate of these networks under four of the five protocols (78.8% under LOBO) but remained below the kinematic core under every protocol. Under LOBO, ReliSense was more accurate than this network on 17 bearings and less accurate on 5 (mean difference 4.9 points; p = 0.038), and than the envelope ResNet on 18 and 3 (17.5 points; p = 0.016). The envelope ResNet raised fewer false alarms than ReliSense with training priors (18.1% against 32.1%) because it missed 28.0% of the faults, against 6.6%; with equal priors (Section VI-C), ReliSense had fewer of both (15.6% and 11.2%). On L8, the envelope ResNet raised 68.3% false alarms, against none for ReliSense.', set()),
 ('Baselines whose settings were selected',
  'Baselines whose settings were selected on the training bearings did not close the gap. The tuned kinematic core reached '
  '73.2%, 73.5% and 79.7% on L10, A2R and LOBO, and the tuned generic features at most 58.9%. The domain-generalization '
  'networks reached at most 67.7% on L8 at condition 0, 65.8% on L10, 70.4% on A2R and 79.1% under LOBO. DANN gave their '
  'highest LOBO accuracy but fell to 24.1% on L8 at condition 0, and all of them raised 36.2%–47.7% false alarms under '
  'LOBO.', set()),
 ('The healthy class is the weakest part of the diagnosis outside',
  'The healthy class is the weakest part of the diagnosis outside L8 at condition 0. Under A2R, 65.8% of the healthy test recordings raised a fault alarm, under LOBO 32.1% and on L8 over all conditions 43.8%, whereas only 0.6%–6.6% of the damaged recordings were diagnosed as healthy. Most LOBO false alarms came from K004 and K005, which were correct on 25.0% and 31.3% of their recordings. Both were mostly diagnosed as outer-race damage, and their median BPFO feature (0.83 and 0.84) exceeded that of K001–K003 (0.56–0.69).', set()),
 ('Because weak excitation lowers the peaks',
  'Weak excitation lowers both the peaks and the confidence: under LOBO, the median confidence was 0.89 for correct and 0.54 for wrong recordings (Fig. 13(a)). Thresholds from training bearings only, set for target coverages of 90%, 80% and 70%, accepted 88.7%, 78.6% and 68.6% of the recordings, with selective accuracies of 89.9%, 94.1% and 97.5% against 83.7% without abstention (Fig. 13(b)). The false alarms fell from 32.1% to 17.5%, 8.1% and 2.7%, and the missed faults from 6.6% to 3.4%, 1.5% and 0.5%. At κ = 0.7, 729 of the 2319 recordings were referred, and 39 of the 1590 accepted ones were wrong. The 95% bearing-level intervals were 57.1%–79.7% for the coverage, 95.1%–99.2% for the selective accuracy and 0.4%–5.4% for the false alarms, so the target coverage holds on average over bearings, not for every bearing. Acceptance was lowest for the healthy K004 and K005 (4% and 5%) and for the weakest damage (18%–25% for KA30, KA15, KI17 and KA22), and complete for six bearings with clear signatures (Fig. 13(c)).', set()),
 ('To compare abstention at equal coverage',
  'For a comparison at equal coverage, the LOBO recordings of each network were ranked by confidence and the most confident ones accepted until the coverage of ReliSense was reached, a ranking on the test recordings that favours the networks. At 68.6% coverage, ReliSense reached 97.5% selective accuracy with 2.7% false alarms, and 96.7% with 4.6% under the same ranking, whereas the other networks reached 63.9%–90.8% with 7.5%–26.9% false alarms and the domain-generalization networks 70.2%–89.0% with 13.1%–23.1% (Section VI-C, part 2). ReliSense also referred more healthy recordings, accepting 45.6% of them against 61.5%–88.3% for the networks.', {'90.8', '63.9', '7.5', '26.9'}),
 ('Fig. 13. Selective classification',
  'Fig. 13. Selective classification of ReliSense under LOBO. (a) Confidence of correct and incorrect recordings. (b) '
  'Accuracy against coverage; markers: training-only thresholds for κ = 0.9, 0.8 and 0.7. (c) Share of recordings '
  'accepted at κ = 0.7 against the accuracy of each bearing.', set()),
 # ---------------------------------------------------------------- V. Further rigs
 ('Table V gives the three-class results',
  'Table V gives the three-class results. On CWRU, ReliSense was the most accurate method under both protocols (83.2% and 76.0%), and the 0.014-in. outer-race defect was missed by every method. On HUST, where every test bearing is new and of an unseen geometry, ReliSense reached 85.0% and, with a single training type, 73.9%, against at most 71.1% and 55.4% for the generic features. The envelope branch raised 60.0% false alarms on CWRU, whose 12-kHz sampling leaves a band of only 2–5.4 kHz, but averaging kept ReliSense at or above the kinematic core in every case. On Ottawa, the generic features classified every segment correctly, against 98.3% and 95.9% for ReliSense. Because each Ottawa class is one bearing, a held-out speed profile comes from a bearing seen in training, so features that identify the bearing rather than its fault can classify every segment.', set()),
 ('The per-bearing results on HUST',
  'The per-bearing results on HUST (Fig. 14) locate the errors. Under LOTO, 11 of the 15 bearings were diagnosed with at '
  'least 90% accuracy; the errors of ReliSense concentrated on the healthy 6205, the outer-race 6208 and the inner-race '
  '6204, whereas the generic random forest failed on several damaged bearings. With one training type, ReliSense lost '
  '11.1 points relative to LOTO and the generic random forest 15.7.', set()),
 ('Fig. 14. Accuracy on each HUST',
  'Fig. 14. Accuracy on each HUST test bearing (three classes) for ReliSense, its kinematic core and two generic '
  'baselines. (a) One bearing type held out (LOTO). (b) One bearing type for training (1TYPE).', set()),
 ('Two weaknesses recurred',
  'Two weaknesses recurred. ReliSense diagnosed 10.0%–16.7% of the healthy CWRU and HUST segments as faulty, fewer than '
  'its kinematic core (20.0%–36.7%); on HUST, with new healthy bearings, the generic methods raised up to 45.8%. Ball '
  'defects were the weakest class of the kinematic core in the four-class task, with a recall of 42.2% on CWRU under LOSO '
  'and 60.0% on Ottawa, in line with the weak ball-defect signatures of the CWRU data reported in [45].', {'45'}),
 # ---------------------------------------------------------------- VI. Discussion
 ('Table VI evaluates the components',
  'Table VI evaluates the components of ReliSense. The 16 modulation features raised the accuracy of the kinematic core under all five protocols, by 1.5–4.2 points. Averaging the kinematic and envelope branches gave the highest pooled accuracy under every protocol and, at κ = 0.7, the highest selective accuracy (97.5%, with 2.7% false alarms), equal to the core combined with the envelope branch; the modulation features thus did not improve the accepted diagnoses at this coverage.', set()),
 ('Table VII compares the results',
  'Table VII compares the results with published Paderborn studies whose test bearings were absent from training. On L8, ReliSense exceeded the published classifiers and their ensemble by more than 20 points. Its macro AUC under LOBO exceeded those of Vieira et al. [15], whose splits contained only real-damage and healthy bearings, so this comparison is indicative only. Perminov and Korzun [48] reported large differences in accuracy between folds.', {'15'}),
 ('1) What the Two Branches Contribute',
  'The two branches erred on different recordings (Fig. 15(a)). On L8 at condition 0, the kinematic core misclassified twelve inner-race recordings as outer-race damage and the envelope branch 35, whereas the averaged model misclassified nine. The main effect of averaging was on the confidence: at κ = 0.7, the averaged posterior gave 97.5% selective accuracy with 2.7% false alarms, against 95.7% and 5.0% for the kinematic branch alone. Per bearing, ReliSense was more accurate than the envelope branch on 7 bearings and less accurate on 11 (Wilcoxon p = 0.69), so its higher pooled accuracy came from fewer errors on recordings at the margin. Against the kinematic core, the improvement was more consistent (14 bearings better, 5 worse, p = 0.048; Fig. 15(b)).', set()),
 ('Table VIII compares the two branches',
  'Table VIII compares the two branches and ReliSense over the eleven bearing-wise settings. The better branch differed between datasets: the kinematic branch on L8, L10, A2R, CWRU and HUST, the envelope branch under LOBO and on Ottawa. Each branch alone fell up to 6.3 points (kinematic) or 11.4 points (envelope) below the other, whereas ReliSense was at least as accurate as the better branch in 9 of the 11 settings and at most 1.6 points below it in the other two. The benefit of averaging is therefore robustness across datasets rather than a significant gain on any one of them. Weighted and stacked combinations and harmonic-agreement features did not improve on equal weights.', set()),
 ('2) The Healthy Bearings and the Cost',
  'The healthy class was the weakest part of the diagnosis throughout (Fig. 16). The false alarms of K004 and K005 reflect a weak line at the outer-race fault order, which their run-in times do not explain (ρ = 0.23, n = 6) and which may come from a small surface irregularity, contamination or a rig resonance; the bearings may not be perfectly healthy. False alarms cost unnecessary inspections and, if frequent, the trust of the operators, whereas a missed fault can cause an unplanned stop. At κ = 0.7, ReliSense accepted only 3.8% and 5.0% of the recordings of K004 and K005 (Fig. 16(b)), and the false-alarm rate fell from 32.1% to 2.7% with 0.5% missed faults. Class-wise thresholds, calibrated in the same way, lowered the false alarms of the core with the envelope branch from 2.7% to 0.6% but raised its missed faults from 0.5% to 4.0%.', set()),
 ('The balance can also be shifted before abstention',
  'The balance can also be shifted before abstention, without retraining. The healthy class is the smallest in training (6 of 29 bearings under LOBO), so both posteriors carry the training class frequencies; dividing each posterior by these frequencies and renormalizing gives the decision for equal class priors [49] (Table IX). Under LOBO, equal priors raised the accuracy from 83.7% to 84.7% and lowered the false alarms from 32.1% to 15.6%, while the missed faults rose from 6.6% to 11.2%; K004 and K005 were then correct in 61.3% and 72.5% of their recordings. Exponents between 0 and 1 on the class frequencies give intermediate operating points. The correction moves errors between classes rather than removing them: KA15, KA22 and KA30 lost 12.5–16.3 points, and per bearing the two rules did not differ significantly (Wilcoxon p = 0.59). Unless labelled otherwise, all other results use the training priors, which gave the fewest missed faults.', {'12.5', '16.3', '0', '1'}),
 ('Table X places these operating points',
  'Table X places these operating points among all methods evaluated under LOBO. Only the envelope ResNet and the '
  'impulse-response kernel network raised fewer false alarms than ReliSense with training priors (18.1% and 30.4%), and '
  'both were less accurate and missed more faults. With equal priors, ReliSense raised fewer false alarms (15.6%) than '
  'every network. At 68.6% coverage, the networks still raised 7.5%–26.9% false alarms, against 2.7% for ReliSense.', set()),
 ('ReliSense at 68.6% coverage: training-only',
  'ReliSense: training-only thresholds (κ = 0.7); with equal priors, they accepted 69.9%. Networks: the most confident '
  'recordings accepted up to 68.6% coverage, a ranking that favours them. Missed faults: damaged recordings diagnosed as '
  'healthy. WDCNN: mean per-bearing accuracy; its missed faults and confidences were not stored.', set()),
 ('Fig. 16. Diagnosis of the six healthy',
  'Fig. 16. The six healthy Paderborn bearings under LOBO. (a) Share of the recordings diagnosed as healthy, outer-race '
  'or inner-race damage; FA: false alarms. (b) The same recordings with training-only abstention at κ = 0.7', set()),
 ('3) Relation to Learned and Physics-Informed',
  'None of the compared networks reached the accuracy of ReliSense under any protocol (Table III). These comparisons concern the implemented pipelines, which differed in input information, and do not show that physical priors in learned representations fail in general. A larger network did not help, since the residual network with 1.02 million parameters was rarely better than WDCNN with 86 thousand. Only the network that received the fault orders came close, 12.7 points above the same architecture on the envelope, consistent with the view that the order domain carries most of what transfers to a new bearing. It still remained 4.9 points below ReliSense under LOBO, with the largest gaps on KA15 and KA22 (Fig. 17). The envelope ResNet diagnosed K004, K005 and KI17 without error, so a learned representation contains information that the fault-order features miss; a learned branch restricted to the order domain is a possible extension.', set()),
 ('4) Deployment',
  'ReliSense needs the bearing geometry, the shaft speed and one recording of the monitored bearing, but no data of that bearing for training or calibration, and its computation is small: a few Fourier transforms, a 96-parameter linear model and a forest of 500 trees. The referral rate is the price of reliability: at κ = 0.7, about 31% of the recordings were referred, mostly from weak damage, low speed and the two healthy bearings that resemble damage. A referred recording asks for another measurement or a manual inspection; decisions from several recordings are evaluated in part 5.', set()),
 ('Abstention was also applied to the further datasets',
  'On the further datasets (Table XI), where the inner loop could hold out the same kind of unit as the test, the coverage stayed near its target and the accepted segments were more accurate. With a single training fault size or bearing type, the inner loop could hold out only loads, whose confidences exceed those of new bearings, so the thresholds accepted only 18.5% of the CWRU and 27.2% of the HUST segments. The calibration therefore needs at least two training bearings per class.', set()),
 ('5) Decisions From Several Recordings',
  'A monitored bearing is measured repeatedly, so a diagnosis need not rest on a single recording. Table XII combines consecutive recordings of one bearing and operating condition by summing the logarithms of their posteriors, and applies the same rule to the stored posteriors of every network. A sequential variant adds recordings until the combined confidence reaches 0.95, with at most ten. With training priors, combining recordings lowered the missed faults but left the false alarms at 30.2%–32.1%, because most recordings of K004 and K005 were diagnosed as damaged consistently. With equal priors, ten recordings, or 40 s of signal, gave 89.6% accuracy with 6.2% false alarms and 8.7% missed faults, and the sequential rule gave similar values with 3.3 recordings on average. With ten recordings, the networks reached at most 81.8%, and all except the envelope ResNet raised 29.2%–47.9% false alarms; the envelope ResNet raised 16.7% but missed 29.0% of the faults. These rules were evaluated after the main results were known, and with ten recordings the false-alarm rate rests on 48 decisions for the six healthy bearings, three of them wrong.', set()),
 ('Posteriors of consecutive recordings',
  'Posteriors of consecutive recordings of one bearing combined by the sum of their logarithms (2319, 463 and 231 '
  'decisions). Sequential rule: recordings added until the combined confidence reached 0.95, at most 10; mean 3.2 and 3.3 '
  'recordings for ReliSense, 1.5–4.3 for the networks (1 seed). WDCNN posteriors were not stored.', set()),
 ('6) Damage Type and Extent',
  'Table XIII groups the LOBO results by the damage properties documented for the dataset [10]. Artificial damage and real fatigue pitting were diagnosed with similar accuracy, 91.4% and 90.1%, so the transfer from artificial to real damage held for the fatigue damage of 9 of the 11 real bearings. Damage of extent levels 2 and 3 was diagnosed at 93.6%, against 84.1% for level 1. The two bearings with plastic deformation by indentations were the weakest damaged bearings, KA15 at 68.8% and KA30 at 42.5%. KA30 is the only single-damage bearing documented as distributed damage, which does not produce the single impact per rolling-element passage assumed by (3). Without KA30, the LOBO accuracy was 85.2%, and 98.1% at 70.4% coverage with abstention. All other results include KA30, because the damage type of a monitored bearing is not known in advance.', set()),
 ('Damage properties as documented',
  'Damage properties from the dataset documentation [10]; extent levels as in Table I. Recordings pooled within each '
  'group. Equal priors: Section VI-C, part 2. κ = 0.7: training-only thresholds, training priors.', set()),
 ('7) Changes Tested Under an Adoption Rule',
  'After the main results, four changes to the features were tested on the same splits under a rule fixed before the runs: the change with the highest mean balanced accuracy over the eleven settings would replace ReliSense only if it was also not less accurate under LOBO and lost at most 0.5 points of selective accuracy at κ = 0.7. The changes were condition-wise standardization, a multiband kinematic branch and two feature sets that separate the fault lines from integer shaft orders, which every fault window of the 6203 bearing contains up to the third harmonic (BPFO = 3 + 0.054 orders): a comb-separated branch without the bins within 0.03 orders of an integer order, and slip-consistent features that search one slip common to the harmonics of each fault order [2]. No change met the rule (Table XIV). The equal-prior decision of ReliSense itself had the highest mean, 85.5% against 83.9%, but lowered the selective accuracy by 1.1 points. The slip-consistent features helped where the test bearings differed in size or type from training but lost 5.0 points on L8 at condition 0, and thresholds set per operating condition lowered the selective accuracy of ReliSense from 97.5% to 95.9%.', {'5.0'}),
 ('Mean over Paderborn L8 (condition 0 and all)',
  'Mean over the eleven settings of Table VIII. Each change replaces the kinematic branch, except condition-wise '
  'standardization, which was applied to both branches. Last column: training-only thresholds; coverage 68.3%–69.9%.', {'8'}),
 # ---------------------------------------------------------------- VII, VIII
 ('The study has limitations.',
  'The study has limitations. The main analysis uses one rig, one bearing type and four operating conditions, and the real damage grew in accelerated lifetime tests rather than in service. The further datasets have seeded damage and few physical bearings, and no model was trained on one machine and tested on another. Fifteen of the 29 bearings informed the design of the kinematic core, and the modulation features and the envelope branch were added after its Paderborn results were known, so the Paderborn results are development-stage evidence. The compared networks did not receive the bandwidth, observation length and harmonic range of ReliSense, the physics-informed networks followed published principles rather than the original code, and the domain-generalization networks were trained with one seed. The equal-coverage comparison ranked the test recordings of the networks. The equal-prior decision, the decisions from several recordings and the changes of Section VI-C, part 7, were evaluated after the main results were known. Tests across machines, for example on the constant-speed Ottawa collection [50] with 20 bearings, calibration under distribution shift [51], [52] and conformal prediction [53] are further directions.', set()),
 ('This article addressed the diagnosis',
  'This article addressed the diagnosis of bearings that a model has never seen. ReliSense reads the squared envelope spectrum at the fault orders fixed by bearing geometry and measured speed, classifies these features in two branches, and withholds low-confidence diagnoses with thresholds set on training bearings only. On 29 Paderborn bearings, which also served for method development, it reached 95.9% when trained on artificial and tested on real damage, against 75.0% for the published ensemble and at most 70.8% for the compared networks. Under leave-one-bearing-out evaluation, it reached 83.7%, and abstention raised this to 97.5% at 69% coverage with 2.7% false alarms. Equal class priors lowered the false alarms before abstention to 15.6%, and ten consecutive recordings to 6.2%, whereas every compared network raised at least 16.7%. With the procedure fixed in advance, ReliSense was also the most accurate method on unseen bearing types of HUST and unseen fault sizes of CWRU. Healthy bearings that resemble damage, weak or distributed real damage and low speed remain its main limits.', set()),
 ('Fig. 1 shows the framework',
  'Fig. 1 shows the framework: Module I converts one 4-s vibration recording into kinematic and envelope features, and Module II classifies them in two branches and decides whether the diagnosis is accepted. The kinematic core, 15 features in a fixed band with logistic regression, was fixed in a preliminary study on 15 of the bearings (K001–K005, KA01, KA04, KA05, KA07, KA15, KI01, KI04, KI05, KI07 and KI14); the modulation features and the envelope branch were added after its Paderborn results were known, and the final configuration was fixed before the three further datasets were analyzed.', set()),
 ('Fig. 3. Local defects',
  'Fig. 3. Local defects of a deep-groove ball bearing. (a)–(c) Rendered illustrations of defects on the outer raceway, the inner raceway and a rolling element (not to scale). (d)–(f) Simulated impact trains of model (3) for the 6203 bearing at 1500 rpm.', set()),
 ('Fig. 10. Paderborn test rig',
  'Fig. 10. Paderborn test rig and test bearing. (a) Rendered rig layout after the description of the dataset authors. (b) Rendered cut-away of the 6203 bearing (n = 8, d = 6.75 mm, D = 28.55 mm).', set()),
 ('L10: mean ± SD over 10 splits',
  'L10: mean ± SD over 10 splits (WDCNN: 3 seeds; other networks: 1 seed). Networks on L8 and A2R: mean ± SD over 3 seeds. LOBO: recordings of all 29 bearings pooled (WDCNN: mean per-bearing accuracy; other networks: 1 seed). Published single classifiers on L8: 62.3–65.9. Tuned baselines and domain-generalization networks: L10 pooled over the 10 splits, 1 seed; tuned baselines not evaluated on L8; Deep CORAL on envelope input not recorded on L8; group DRO on envelope input not completed. Equal priors: Section VI-C.', set()),
 ('Each dataset was split as its structure allows',
  'On CWRU, one defect diameter was held out at a time (LOSO), so that the faulty test bearings are new, and the single healthy bearing was split by load. On HUST, one bearing type was held out (LOTO), so that every test bearing is new and of an unseen geometry. On Ottawa, one speed profile was held out (LOPO), which tests only an unseen speed profile of the same bearings. Each protocol was also run in reverse, with one diameter, type or profile for training (1SIZE, 1TYPE, 1PROF). The comparators were 32 band energies with the same logistic regression, and time and envelope statistics with the band energies in a random forest. The primary task has three classes; a four-class task adds ball defects.', set()),
 ('The Paderborn data are available',
  'The Paderborn data are available from the KAt-DataCenter of Paderborn University under the CC BY-NC 4.0 license, and the CWRU, HUST and Ottawa data from their providers. The code, label sheet, split manifests including the ten L10 combinations, the identifier of the unreadable recording and the per-fold predictions will be released at [repository link to be added].', set()),
 ('The Paderborn dataset was recorded',
  'The Paderborn dataset was recorded on a modular rig in which a synchronous motor drives the test-bearing module against a load motor (Fig. 10); the housing vibration was sampled at 64 kHz. Each bearing was run under four operating conditions with 20 recordings of 4 s each: N15_M07_F10 (condition 0: 1500 rpm, 0.7 Nm, 1000 N), N09_M07_F10 (900 rpm), N15_M01_F10 (0.1 Nm) and N15_M07_F04 (400 N). It is the only public dataset with artificial and real damage on many bearings of one type.', set()),
 ('Table I lists the bearings',
  'Table I lists the bearings. Six are healthy, twelve carry artificial damage made by electrical discharge machining (EDM), drilling or electric engraving, and fourteen carry real damage from accelerated lifetime tests; labels and damage descriptions were taken from the dataset publication. KB23, KB24 and KB27, damaged on both rings, were excluded from the three-class task, which leaves 29 bearings and 2319 recordings (one file could not be read). The unit of evaluation is the 4-s recording.', set()),
 ('LOBO, κ = 0.7: selective accuracy',
  'LOBO, κ = 0.7: selective accuracy / false alarms / missed faults with training-only thresholds; coverage 68.6% and 69.9%. Balanced accuracy under LOBO: 81.2% and 84.7%. L10, HUST and Ottawa have equal class frequencies in training, so both rules give the same results there.', set()),
 ('Evaluation units and class sets differ',
  'Evaluation units and class sets differ. The correspondence of the ten L10 combinations of this study to the published L10 evaluation has not been established, so its 98.3% is listed for reference only.', set()),
]
SENTENCE_EDITS = [   # paragraphs with formatted symbols: only plain sentences are shortened
 ('Dividing by fr gives the fault orders',
  'Fig. 2(a) shows the rolling kinematics behind these relations; without slip, the cage moves at half the surface speed '
  'of the inner ring, which gives FTF.', 'Fig. 2(a) shows the rolling kinematics.'),
 ('1) Band Selection and Demodulation', ' Each impact excites a resonance of several kilohertz that decays within milliseconds.', ''),
 ('The band covers the structural resonances',
  ' It is fixed in the kinematic branch; the envelope branch also uses the band chosen by the fast kurtogram and both bands '
  'after cepstral pre-whitening.', ''),
 ('where θi = 2π·fr·i·Td', 'Two properties of (3) underlie the method. ', ''),
 ('Table IV summarizes the datasets', ', with the shaft angle taken from an encoder', ''),
 ('Fig. 9. Training, calibration and deployment', ' Online, a new bearing b needs one recording, its shaft speed and its geometry.', ''),
 ('[33]\t', 'H. Lu, V. P. Nemani, V. Barzegar, C. Allen, C. Hu, S. Laflamme, S. Sarkar, and A. T. Zimmerman,', 'H. Lu et al.,'),
 ('[41]\t', 'Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, and V. Lempitsky,', 'Y. Ganin et al.,'),
]
DELETE = ['7) Properties:']


# ------------------------------------------------------------------ helpers
def all_pars(d):
    ps = list(d.paragraphs)
    for t in d.tables:
        for r in t.rows:
            for c in r.cells: ps += c.paragraphs
    return ps


def find(d, start):
    hits = [p for p in d.paragraphs if p.text.startswith(start)]
    assert len(hits) == 1, (start, len(hits)); return hits[0]


def nums(t): return set(re.findall(r'\d+(?:\.\d+)?', t))


def has_head(p):
    rs = p.runs
    return len(rs) > 1 and rs[0].italic and re.match(r'\d\) ', rs[0].text) is not None


def rewrite(p, new):
    """replace the text of p; keep an italic run-in heading; the body takes the format of its longest run."""
    rs = list(p.runs); head = rs[0] if has_head(p) else None
    body = [r for r in rs if r is not head]; keep = max(body, key=lambda r: len(r.text))
    for r in body:
        if r is not keep: r._r.getparent().remove(r._r)
    for tag in ('hyperlink', 'proofErr'):
        for x in p._p.findall(W + tag): p._p.remove(x)
    keep.text = new


def replace_across_runs(p, old, new):
    runs = p.runs; full = ''.join(r.text for r in runs); k = full.find(old)
    assert k >= 0 and full.count(old) == 1, (old[:70], full[:90])
    pos = 0; first = True; end = k + len(old)
    for r in runs:
        a, b = pos, pos + len(r.text); pos = b
        if b <= k or a >= end: continue
        s, e = max(k, a) - a, min(end, b) - a
        if first: r.text = r.text[:s] + new + r.text[e:]; first = False
        else: r.text = r.text[:s] + r.text[e:]


def mark(p, oldtext):
    new = p.text; runs = list(p.runs); head = runs[0] if has_head(p) and oldtext.startswith(runs[0].text) else None
    off = len(head.text) if head is not None else 0
    A = re.findall(r'\S+\s*', oldtext[off:]); B = re.findall(r'\S+\s*', new[off:])
    tmpl = copy.deepcopy(max([r for r in runs if r is not head], key=lambda r: len(r.text))._r)
    for r in runs:
        if r is not head: r._r.getparent().remove(r._r)
    def add(t, col=None, strike=False):
        r = p.add_run(t); rpr = tmpl.find(W + 'rPr')
        if rpr is not None: r._r.insert(0, copy.deepcopy(rpr))
        if col: r.font.color.rgb = col
        if strike: r.font.strike = True
    for op, a1, a2, b1, b2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        if op == 'equal': add(''.join(A[a1:a2]))
        if op in ('delete', 'replace'): add(''.join(A[a1:a2]), RED, True)
        if op in ('insert', 'replace'): add(''.join(B[b1:b2]), BLUE)


def build(marked):
    d = Document(src); old = {p._p: p.text for p in all_pars(d)}
    for start, new, allowed in REWRITE:
        p = find(d, start); o = p.text
        body_old = o[len(p.runs[0].text):] if has_head(p) else o
        extra = nums(new) - nums(body_old) - allowed
        assert not extra, (start, sorted(extra))
        rewrite(p, new)
    for start, o, n in SENTENCE_EDITS: replace_across_runs(find(d, start), o, n)
    for start in DELETE:
        p = find(d, start)
        if marked:
            for r in p.runs: r.font.color.rgb = RED; r.font.strike = True
        else: p._p.getparent().remove(p._p)
    if marked:
        for p in all_pars(d):
            if p._p in old and p.text != old[p._p] and not any(p.text.startswith(s) for s in DELETE): mark(p, old[p._p])
        P = d.paragraphs; k = [i for i, p in enumerate(P) if p.text.startswith('Revision key')]
        if k: P[k[0]]._p.getparent().remove(P[k[0]]._p)
        key = d.paragraphs[0].insert_paragraph_before('')
        key.add_run('Revision key (v2.7 → v2.8, text condensed):  ')
        x = key.add_run('red strikethrough = deletion'); x.font.color.rgb = RED; x.font.strike = True
        key.add_run('   |   '); x = key.add_run('blue = addition'); x.font.color.rgb = BLUE
    return d


build(False).save(out_clean); build(True).save(out_marked)
d = Document(out_clean); print('words (paragraphs incl. references and captions):', sum(len(p.text.split()) for p in d.paragraphs))
