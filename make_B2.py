"""Creates build_B2.py from build_B.py (Paper B v1.1): Paper B v2.0 for MSSP.
- Framed as a bearing-wise benchmark and explanation study (no new classifier), with the originality stated as engineering knowledge.
- Self-contained: every result is obtained with the kinematic core, generic or learned representations; no result of the companion
  method (ReliSense) is used. Figures redrawn with the core (fig_B_core.py, fig_B_core2.py, fig_disc_inputs_B.py v1.1);
  numbers recomputed from stored per-bearing results (phases 2, 5, 12, 19-22).
- New Section IV-G and Table: results on the bearings that informed the design against all other bearings.
- Strength of the baselines stated; tuned classical baselines and domain-generalization networks (phase 24) follow in v2.1."""
import re
s = open('build_B.py').read()


def rep(o, n, cnt=1):
    global s; assert s.count(o) == cnt, (s.count(o), o[:90]); s = s.replace(o, n)


s = s.replace('"""Paper B v1.1 (analysis;', '"""Paper B v2.0 (benchmark and explanation study; make_B2.py). Paper B v1.1 (analysis;', 1)
rep("r.text = 'Paper B, version 1.1, 5 October 2026'", "r.text = 'Paper B, version 2.0, 5 October 2026'")
TITLE = 'Why Physics-Informed Fault-Order Features Transfer to Unseen Bearings: A Bearing-Wise Benchmark of Bearing Identity, Detectability and Evaluation Bias'
s = re.sub(r"doc\.paragraphs\[0\]\.runs\[0\]\.text = '[^\n]*'\n", lambda m: f"doc.paragraphs[0].runs[0].text = {TITLE!r}\n", s, count=1)
ABS = ('Vibration-based fault classifiers often report near-perfect accuracy, yet their accuracy on bearings absent from training is rarely examined and even more rarely explained. '
       'This article is a bearing-wise benchmark and explanation study, not a new classifier. On 29 Paderborn bearings and two further test rigs, under protocols in which every choice is made on training bearings only, '
       'it compares a physics-informed representation, 15 peak-to-background features at the fault orders of the bearing, with generic spectral and time-domain representations and with learned networks, and asks why the physics-informed representation transfers and when it fails. '
       'Corrupting the fault orders showed that the true orders rank first among 31 order sets. A bearing-identity test revealed an inverse relation between identity and transfer: band energies and time and band statistics named the individual bearing of a recording in 97.7% and 98.2% of cases '
       'but diagnosed unseen bearings at only 50.1% and 55.2%, against 70.6% and 80.1% for the fault-order features. Defect size did not predict whether a defect was found, whereas the contrast of the defect line against a healthy bearing of the same type did (Spearman ρ = 0.70, n = 16). '
       'Trained on artificial damage only, the fault-order model was at least as accurate on every real-damage test bearing as when it had also seen real damage (93.0% against 84.5% at the reference condition), and the bearings used to design the features were not diagnosed significantly better than the others. '
       'Finally, a published accuracy of 98.3% could not be reproduced, and model selection with knowledge of the test results raised accuracy by up to 19.5 points. Four guidelines for evaluating diagnosis on unseen bearings are derived.')
s = re.sub(r"\nABSTRACT = '[^\n]*'\n", lambda m: f"\nABSTRACT = {ABS!r}\n", s, count=1)
assert 'bearing-wise benchmark and explanation study, not a new classifier' in s

# ------------------------------------------------------------------ introduction
a = s.index("P('A companion article {c:900} proposed ReliSense"); b = s.index("P('Section II defines the kinematic core")
s = s[:a] + r'''P('This article is a benchmark and explanation study, not a proposal for another classifier. It evaluates one physics-informed representation, 15 peak-to-background features read at the fault orders of the bearing and classified by logistic regression, here called the kinematic core, '
  'against generic spectral and time-domain representations and learned networks, under bearing-wise protocols in which every choice of features, hyperparameters and thresholds is made on training bearings only and all controls were fixed before execution. '
  'Its originality lies in what it establishes about the physics of transfer rather than in a new algorithm: why fault-order features carry over to a new bearing, which damage they cannot detect, whether artificial damage suffices for training, and how strongly the evaluation itself decides a reported accuracy. '
  'The same features form one branch of a diagnosis method described in a companion article {c:900}; no result of that method is used here, and all conclusions follow from the representations evaluated in this article. The contributions are as follows.')
LIST('A mechanism for transfer to unseen bearings. Fault-order perturbation with scaled and random orders, a representation ladder from the waveform to the fault orders, and a bearing-identity test show that the true fault orders rank first among 31 order sets and reveal an inverse relation between identity and transfer: '
     'representations that named the individual bearing in 97.7%–98.2% of cases diagnosed unseen bearings worst (50.1%–55.2%), against 80.1% for the kinematic core. In five settings with unseen bearings on three rigs, the kinematic core with logistic regression was more accurate than band energies with the same classifier and than time and band statistics with a random forest. '
     'Bearing identity, rather than the distribution gap between artificial and real damage, is therefore the main obstacle to transfer.')
LIST('A detectability criterion for new bearings. The contrast of the defect line against a healthy bearing of the same type predicted the accuracy on unseen bearings of two rigs (Spearman ρ = 0.70, n = 16), and every bearing with a contrast above 1.1 was diagnosed without error, whereas defect size and bearing size did not predict accuracy. '
     'Together with a damage-wise analysis, the criterion identifies the damage that is hardest to detect, small real indentations, repeated and distributed damage and operation at low speed, and gives the operator a measurable reason to distrust a negative diagnosis.')
LIST('Evidence that artificial damage is adequate training data for real damage when the representation is fixed by kinematics. The maximum mean discrepancy between artificial and real recordings was not smaller for the kinematic features than for band energies, '
     'yet on the ten real-damage test bearings of the protocol of the dataset authors, the kinematic core trained on artificial damage only was at least as accurate on every bearing as when it had also seen real damage (93.0% against 84.5% at condition 0). The limit lies in repeated, distributed and indented real damage, which the artificial damage does not represent.')
LIST('A strict bearing-wise benchmark with evaluation guidelines. Under protocols in which every choice is made on training bearings, the published 98.3% on the real-damage protocol could not be reproduced, model selection with knowledge of the test results raised accuracy by up to 19.5 points, a dataset with one bearing per class gave perfect accuracy for features that only recognize known bearings, '
     'and the bearings that informed the design of the features were not diagnosed significantly better than the others. These results lead to four practical guidelines for evaluating diagnosis on unseen bearings.')
''' + s[b:]
rep("P('Section II defines the kinematic core and the verification controls. Section III describes the data, the protocols and the compared representations. Section IV reports the results, Section V discusses them,",
    "P('Section II defines the kinematic core and the verification controls. Section III describes the data, the protocols and the compared representations. Section IV reports the results, including a test of the independence of the results from the bearings used in the design, Section V discusses them,")
rep("before any analysis reported here. Where results of the full two-branch ReliSense are shown for comparison, they are taken from {c:900}.')",
    "before any analysis reported here; Section IV-G tests whether this choice favoured the design bearings.')")
rep("and the spectrum is placed on the order axis with the measured shaft speed {c:900},')", "and the spectrum is placed on the order axis with the measured shaft speed,')")
# ------------------------------------------------------------------ data and baselines
rep("datasets are used with the segmentation and protocols of {c:900}: one defect diameter (CWRU) or one bearing type (HUST, 6204–6208) is held out at a time.",
    "datasets are used: every recording is cut into segments of 50 shaft revolutions, the features are read at the fault orders of each bearing type (HUST geometry from {c:52}), and one defect diameter (CWRU) or one bearing type (HUST, 6204–6208) is held out at a time, or used alone for training.")
rep("  'Accuracy is the share of correctly diagnosed 4-s recordings. Because the groups of the damage-wise analysis contain one to nine bearings, all tests are descriptive.')",
    "  'The baselines were chosen to be strong within the class of methods they represent: every generic feature group was paired with four classifiers of increasing flexibility, on L10 the feature group, classifier and per-condition standardization were selected by nested bearing-wise cross-validation, and the networks were trained with class weighting and evaluated over three seeds; the hyperparameters of the classical classifiers were set to common default values. '\n"
    "  'Accuracy is the share of correctly diagnosed 4-s recordings. Because the groups of the damage-wise analysis contain one to nine bearings, all tests are descriptive.')")
# ------------------------------------------------------------------ results
rep("in line with the false alarms reported in {c:900}.')", "in line with the false alarms of the kinematic core on K004 and K005 (Fig. {f:matrix}).')")
rep("which probably reflects its overlap with the second BPFO harmonic {c:900} rather than a ball defect.')", "which probably reflects its overlap with the second BPFO harmonic (BSF·3 ≈ 6.0 orders, 2·BPFO ≈ 6.1 orders) rather than a ball defect.')")
rep("; as additional readings in the envelope branch of ReliSense {c:900}, the same variants contributed to the best results. '", ". '")
rep("for the kinematic core, for ReliSense and its two branches {c:900} and for two further models together with the damage attributes",
    "for the kinematic core, for two variants of it with a different demodulation band, for the threshold rule and for WDCNN, together with the damage attributes")
rep("(Spearman ρ = 0.58, p = 0.059, n = 11), and level-2 and level-3 damage was diagnosed at 98.8%–100%. '", "(Spearman ρ = 0.54, p = 0.083, n = 11), and level-2 and level-3 damage was diagnosed at 97.5%–100%. '")
rep("(ρ = 0.23, p = 0.28), because artificial level-1 damage was diagnosed better than real level-1 damage (median 100% against 75.6%; Mann–Whitney p = 0.075). '",
    "(ρ = 0.19, p = 0.38), because artificial level-1 damage was diagnosed better than real level-1 damage (median 100% against 71.3%; Mann–Whitney p = 0.041). '")
rep("were diagnosed without error by all six models.", "were diagnosed without error by all five models.")
rep("(median 98.8% against 55.6%; p = 0.074, two indented bearings), and repeated damage (70.0%) and the one distributed damage (KA30, 42.5%) less often than single damage (89.4%). '",
    "(median 97.5% against 53.8%; p = 0.12, two indented bearings), and repeated damage (61.7%, including the distributed damage of KA30 at 46.3%) less often than single damage (84.4%). '")
rep("remained below 70% for all six models, '\n    'and six bearings (KA01, KA04, KA06, KA16, KI01 and KI18) were at least 95% for all of them: the hardest and the easiest damage were the same for every model. '",
    "remained below 50% for all five models, '\n    'and three bearings (KA01, KI01 and KI18) were at least 95% for all of them. '")
rep("'The feature models erred on the same bearings (Spearman correlation of per-bearing accuracy 0.90–0.94 between ReliSense, its branches and its core), whereas WDCNN erred on different ones (0.64): '\n    'it failed on artificial bearings with clear lines (KA09, KI05 and KI07) but was more accurate on real outer-race damage (87.0% against 77.0%) and on indentations (83.1% against 55.6%).')",
    "'The per-bearing accuracy of the core correlated with that of its pre-whitened variant (Spearman 0.78) and of the threshold rule (0.68) more than with WDCNN (0.49): '\n    'WDCNN failed on artificial bearings with clear lines (KA09, KI05 and KI07: 5%, 19% and 10% against 75%, 75% and 70%) but was more accurate on real outer-race damage (87.0% against 73.8%) and on indentations (83.1% against 53.8%).')")
rep("Under LOBO, ReliSense reached 90.7% at condition 0, 90.8% at low torque, 80.5% at low radial force and 72.8% at 900 rpm, and the healthy recall was lowest at 900 rpm (63.3%) and at low radial force (64.2%). '",
    "Under LOBO, the kinematic core reached 88.6% at condition 0, 89.3% at low torque, 77.4% at low radial force and 65.2% at 900 rpm, and the healthy recall was lowest at 900 rpm (36.7%) and at low radial force (56.7%). '")
rep("'For the kinematic core, accuracy at 900 rpm fell", "'Accuracy at 900 rpm fell")
rep("FIG('matrix', NEWDIR + 'fig_matrix.png', 'LOBO accuracy of every Paderborn bearing for ReliSense, its kinematic and envelope branches, its kinematic core, the threshold rule and WDCNN,",
    "FIG('matrix', NEWDIR + 'fig_matrix_B.png', 'LOBO accuracy of every Paderborn bearing for the kinematic core, the core read in the kurtogram band and after cepstral pre-whitening, the threshold rule and WDCNN with envelope input,")
rep("FIG('conditions', NEWDIR + 'fig_conditions.png', 'Effect of the operating condition. (a) LOBO accuracy and class recall of ReliSense for each condition.",
    "FIG('conditions', NEWDIR + 'fig_conditions_B.png', 'Effect of the operating condition. (a) LOBO accuracy and class recall of the kinematic core for each condition.")
rep("(c) LOBO accuracy of ReliSense for every bearing and condition.')", "(c) LOBO accuracy of the kinematic core for every bearing and condition.')")
rep("from v326_data import ENSX_PB as _E, RF_PB as _RF, np\n", "import numpy as np\n")
rep("_rows.append([lab if i == 0 else '', g, str(r['n']), f\"{np.mean([_E[b] for b in r['bearings']]):.1f}\", f\"{r['ReliSense']:.1f}\", f\"{r['Threshold rule']:.1f}\", f\"{np.mean([_RF[b] for b in r['bearings']]):.1f}\", f\"{r['WDCNN (envelope)']:.1f}\"])",
    "_rows.append([lab if i == 0 else '', g, str(r['n']), f\"{r['ReliSense']:.1f}\", f\"{r['Threshold rule']:.1f}\", f\"{r['WDCNN (envelope)']:.1f}\"])   # phase 12: 'ReliSense' = kinematic core")
rep("    ['Attribute', 'Group', 'Bearings', 'ReliSense', 'Kinematic core', 'Threshold rule', 'Envelope branch', 'WDCNN, envelope'], _rows,\n    [1600, 2500, 900, 1000, 1000, 1000, 1000, 1100],",
    "    ['Attribute', 'Group', 'Bearings', 'Kinematic core', 'Threshold rule', 'WDCNN, envelope'], _rows,\n    [1900, 3000, 1100, 1300, 1300, 1300],")
rep("On HUST, the accuracy of ReliSense {c:900} on the five unseen types (pitch diameter 33.5–60 mm) showed no monotonic trend (86.1%, 75.0%, 94.4%, 94.4% and 75.0% for 6204–6208). '",
    "On HUST, the accuracy of the kinematic core on the five unseen types (pitch diameter 33.5–60 mm) showed no monotonic trend (66.7%, 66.7%, 97.2%, 86.1% and 69.4% for 6204–6208). '")
rep("'Accuracy increased with this contrast (ρ = 0.81, p < 0.001, n = 16): all 11 bearings with a contrast above 1.1 were diagnosed without error, and the failures had the weakest lines, the CWRU 0.014-in. outer-race defect (0.39, against 3.50 and 1.82 for the other outer-race sizes; 0%) and the HUST outer-race 6208 (0.37; 25%). '\n  'The HUST inner-race 6204, with a moderate contrast (1.03), was diagnosed at 58%. ReliSense therefore diagnoses a defect when its kinematic line rises clearly above that of a healthy bearing, whatever the bearing size or the defect size.')",
    "'Accuracy increased with this contrast (ρ = 0.70, p = 0.002, n = 16): all 11 bearings with a contrast above 1.1 were diagnosed without error, and the failures had weak lines, the CWRU 0.014-in. outer-race defect (0.39, against 3.50 and 1.82 for the other outer-race sizes; 0%), the HUST outer-race 6208 (0.37; 8.3%) and the HUST inner-race 6204 (1.03; 0%). '\n  'The HUST inner-race 6207 (0.90) was diagnosed at 58.3%. A defect is therefore diagnosed when its kinematic line rises clearly above that of a healthy bearing, whatever the bearing size or the defect size.')")
rep("FIG('line', NEWDIR + 'fig_line.png',", "FIG('line', NEWDIR + 'fig_line_B.png',")
rep("(b) ReliSense accuracy of each faulty test bearing against its signature contrast;", "(b) Kinematic-core accuracy of each faulty test bearing against its signature contrast;")
rep("P('On L10, the two-branch ReliSense {c:900} reached 78.7% over all conditions and its kinematic core 74.6% over all conditions and 73.5% at condition 0,",
    "P('On L10, the kinematic core reached 74.6% over all conditions and 73.5% at condition 0,")
rep("  'No method tested here exceeded 78.7% on L10.')", "  'No configuration tested here exceeded 76.2% on L10, even when chosen with knowledge of the test results.')")
rep("    [['ReliSense', '–', '78.7', '95.9', '87.0'],\n     ['Kinematic core (fixed in advance)',", "    [['Kinematic core (fixed in advance)',")
rep("The best-of-72 row uses test results. ReliSense values from {c:900}; it was not evaluated on L10 at condition 0 alone.')", "The best-of-72 row uses test results.')")
# ------------------------------------------------------------------ new Section IV-G: design bearings
rep("\nH1('V. Discussion')", r'''
H2('G. Independence From the Design Bearings')
P('Fifteen of the 29 bearings (K001–K005, KA01, KA04, KA05, KA07, KA15, KI01, KI04, KI05, KI07 and KI14) informed the choice of the kinematic core in a preliminary study. If this choice had been tuned to these bearings, they should be diagnosed better than the other 14. '
  'No significant difference was found (Table {t:design}). Under LOBO, the design bearings reached a mean per-bearing accuracy of 77.8% and the other bearings 82.7% (Mann–Whitney p = 0.91); under A2R and L8 over all conditions the other bearings were also more accurate, and only on L8 at condition 0 were the four design bearings more accurate (98.8% against 90.7%). '
  'The lower LOBO value of the design bearings comes from the healthy K004 and K005, which are among them; among the damaged bearings alone, the design bearings were 5.1 points more accurate (88.5% against 83.4%; p = 0.48), a difference within the spread between bearings but in the direction that a tuned design would produce, and it should be kept in mind when the Paderborn accuracies are read. '
  'The external rigs played no part in the design, and on them the core reached 77.2% and 73.9% on unseen HUST bearing types and 79.5% and 67.1% on unseen CWRU fault sizes, in each case more than the generic representations (Fig. {f:discinputs}).')
TAB('design', 'Kinematic Core on the Bearings Used in Its Design and on All Other Bearings (Mean Per-Bearing Recording Accuracy, %)', ['Protocol', 'Design bearings', 'Other bearings'],
    [['L8, condition 0', '98.8 (4)', '90.7 (7)'], ['L8, all conditions', '79.1 (4)', '85.0 (7)'], ['A2R', '60.0 (6)', '82.8 (8)'], ['LOBO', '77.8 (15)', '82.7 (14)'],
     ['HUST, LOTO / 1TYPE', '–', '77.2 / 73.9'], ['CWRU, LOSO / 1SIZE', '–', '79.5 / 67.1']],
    [3500, 2800, 2800], 'Number of test bearings in parentheses. Design bearings: the 15 Paderborn bearings of the preliminary study. External rigs: pooled segment accuracy; no bearing of these rigs was used in the design.')

H1('V. Discussion')''')
# ------------------------------------------------------------------ discussion
rep("'The features of ReliSense are read only at these rates,", "'The fault-order features are read only at these rates,")
rep("'This explanation also accounts for the behaviour of the envelope branch. A random forest is flexible enough to memorize individual bearings, yet on the 62 envelope features of {c:900} it reached 82.5% under LOBO, close to the kinematic core (80.1%), because its inputs are themselves read at the fault orders and contain no generic spectral description. '\n"
    "    'Flexibility of the classifier was therefore less harmful than generality of the input: on HUST, with unseen bearing types, the same type of model reached 81.1% on the envelope features but 71.1% on time and band statistics. '\n"
    "    'Fig. {f:discinputs} shows that this holds in all five settings with unseen bearings: with the same classifier, the generic input was always less accurate than the kinematic input, by 6.8–32.2 points for logistic regression and 1.9–20.5 points for the random forest.')",
    "'Flexibility of the classifier did not compensate for generality of the input. Fig. {f:discinputs} shows this in all five settings with unseen bearings: band energies were 6.8–32.2 points less accurate than the kinematic core with the same logistic regression, '\n"
    "    'and on the external rigs time and band statistics remained 6.1–19.8 points below the linear kinematic core even when classified by a random forest, which is flexible enough to memorize individual bearings (on Paderborn, with logistic regression, 24.9 points).')")
rep("FIG('discinputs', NEWDIR + 'fig_disc_inputs_B.png', 'Accuracy on unseen bearings by input and classifier in five settings. Grey, open: generic inputs (band energies with LR; time and band statistics with RF, on Paderborn with LR); blue, filled: inputs read at the fault orders (kinematic core with LR; envelope features of {c:900} with RF). Circles: logistic regression; squares: random forest.",
    "FIG('discinputs', NEWDIR + 'fig_disc_inputs_B.png', 'Accuracy on unseen bearings by input and classifier in five settings. Grey, open: generic inputs (band energies with LR; time and band statistics with RF, on Paderborn with LR); blue, filled: the kinematic core with LR. Circles: logistic regression; squares: random forest.")
rep("the extent level and accuracy were only weakly related (ρ = 0.23; Fig. {f:discsize}(a)),", "the extent level and accuracy were only weakly related (ρ = 0.19; Fig. {f:discsize}(a)),")
rep("accuracy followed the signature contrast with ρ = 0.81 (Fig. {f:line}), every bearing with a contrast above 1.1 was diagnosed without error, and the two complete failures, the CWRU 0.014-in. outer-race defect and the HUST outer-race 6208, had the two lowest contrasts (0.39 and 0.37). '",
    "accuracy followed the signature contrast with ρ = 0.70 (Fig. {f:line}), every bearing with a contrast above 1.1 was diagnosed without error, and the two bearings with the lowest contrasts, the HUST outer-race 6208 (0.37) and the CWRU 0.014-in. outer-race defect (0.39), were almost never diagnosed (8.3% and 0%). '")
rep("(median 100% against 75.6%) and plastic indentations worse than fatigue pitting (55.6% against 98.8%; Fig. {f:discsize}(b)). '",
    "(median 100% against 71.3%) and plastic indentations worse than fatigue pitting (53.8% against 97.5%; Fig. {f:discsize}(b)). '")
rep("FIG('discsize', NEWDIR + 'fig_disc_size.png', 'Damage size and LOBO accuracy of ReliSense on", "FIG('discsize', NEWDIR + 'fig_disc_size_B.png', 'Damage size and LOBO accuracy of the kinematic core on")
rep("'On L8, which trains on artificial damage and tests on real damage, ReliSense {c:900} reached 95.9% at condition 0 and 87.0% over all conditions, higher than its accuracy under LOBO, which trains on both kinds of damage. The same holds bearing by bearing (Fig. {f:disca2r}): on the ten real-damage test bearings of L8, the model trained on artificial damage only was at least as accurate as the LOBO model, which had also seen the real damage of 28 other bearings, on every bearing at condition 0 (mean 95.5% against 90.0%) and on all but KI21 over all conditions (90.1% against 82.2%).",
    "'On L8, which trains on artificial damage and tests on real damage, the kinematic core reached 93.6% at condition 0 and 82.8% over all conditions, higher than its accuracy under LOBO (80.1%), which trains on both kinds of damage. The same holds bearing by bearing (Fig. {f:disca2r}): on the ten real-damage test bearings of L8, the core trained on artificial damage only was at least as accurate as the LOBO model, which had also seen the real damage of 28 other bearings, on every bearing at condition 0 (mean 93.0% against 84.5%) and on all but KA15 over all conditions (84.0% against 77.1%).")
rep("FIG('disca2r', NEWDIR + 'fig_disc_a2r.png', 'Accuracy of ReliSense {c:900} on", "FIG('disca2r', NEWDIR + 'fig_disc_a2r_B.png', 'Accuracy of the kinematic core on")
rep("and the 74.6% of the kinematic core (78.7% for ReliSense {c:900}) illustrates", "and the 74.6% of the kinematic core illustrates")
# ------------------------------------------------------------------ limitations and conclusion
rep("  'Fifteen of the 29 bearings informed the design of the kinematic core, including four of the 11 L8 test bearings, so that its Paderborn results are not fully independent of its design. The groups of the damage-wise analysis",
    "  'Fifteen of the 29 bearings informed the design of the kinematic core, including four of the 11 L8 test bearings; these bearings were not diagnosed significantly better than the others, but among the damaged bearings they were 5.1 points more accurate under LOBO, and a fully independent test requires a dataset that played no part in the design, which the external rigs provide only for unseen bearing types and fault sizes. '\n"
    "  'The baselines cover generic features with four classifiers and nested selection and a wide-kernel network; domain-generalization methods that align the training bearings, such as adversarial or covariance-alignment training, were not evaluated in this version. The groups of the damage-wise analysis")
rep("P('This article examined why features read at the fault orders of a bearing transfer to bearings absent from training. Controls",
    "P('This benchmark and explanation study examined why physics-informed features read at the fault orders of a bearing transfer to bearings absent from training. Controls")
open('build_B2.py', 'w').write(s)
print('build_B2.py written')
