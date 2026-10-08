"""Creates build_A.py and build_B.py from build_v332.py: the full study (manuscript v3.32) split into two articles without
overlapping results. Paper A (v1.0): the ReliSense method and its performance. Paper B (v1.0): why fault-order features transfer
to unseen bearings, detectability and evaluation. No new experiment; every number is taken from v3.32 or its stored results
(Paper B tables use the kinematic-core values of phases 19-21 where v3.32 shows ReliSense)."""
L = open('build_v332.py').read().split('\n')
R = lambda a, b: '\n'.join(L[a - 1:b]) + '\n'
HEAD, TAIL = R(1, 113), R(591, len(L))
TAIL = TAIL.replace("doc.save(OUT)\n", "for p_ in doc.paragraphs:\n    for r_ in p_.runs:\n        if '[51]–[53]' in r_.text: r_.text = r_.text.replace('[51]–[53]', cite_list([ref_order.index(o) + 1 for o in (51, 52, 53)]))\ndoc.save(OUT)\n", 1)
assert 'cite_list([ref_order.index(o) + 1 for o in (51, 52, 53)])' in TAIL


def edit(s, pairs):
    for o, n in pairs:
        assert s.count(o) == 1, (s.count(o), o[:80]); s = s.replace(o, n)
    return s


def header(title, terms, abstract, label):
    h = edit(HEAD, [("if 'Manuscript version 3.11' in r.text: r.text = re.sub(r'3\\.11, \\d+ \\w+ \\d{4}', '3.32, 5 October 2026', r.text).replace('3.11', '3.32')",
                     f"if 'Manuscript version 3.11' in r.text: r.text = '{label}, 5 October 2026'"),
                    ("    if m: REFP[int(m.group(1))] = e\n",
                     "    if m: REFP[int(m.group(1))] = e\n"
                     "def _mkref(n, text):\n"
                     "    e = copy.deepcopy(REFP[1]); ts = list(e.iter(q('t'))); ts[0].text = f'[{n}]'; ts[1].text = text\n"
                     "    for t_ in ts[2:]: t_.text = ''\n"
                     "    REFP[n] = e; REFP[1].addnext(e)\n"
                     "_mkref(900, 'Authors, “ReliSense: Physics-informed two-branch fault-order diagnosis with training-only abstention for rolling bearings never seen in training,” companion manuscript submitted to IEEE Trans. Instrum. Meas., 2026.')\n"
                     "_mkref(901, 'Authors, “Why physics-informed fault-order features transfer to unseen bearings: Bearing identity, detectability and evaluation pitfalls,” companion manuscript submitted to Mech. Syst. Signal Process., 2026.')\n")])
    a = h.index('ABSTRACT = ('); b = h.index("for p in doc.paragraphs:\n    if p.text.startswith('Abstract')")
    h = h[:a] + 'ABSTRACT = ' + repr(abstract) + '\n' + (
        f"doc.paragraphs[0].runs[0].text = {title!r}\n"
        "for p in doc.paragraphs:\n"
        "    if p.text.startswith('Index Terms') and len(p.runs) >= 2:\n"
        f"        p.runs[1].text = '—' + {terms!r}\n"
        "        for r in p.runs[2:]: r.text = ''\n") + h[b:]
    return h


# ===================================================================================== PAPER A
A_ABS = ('A bearing fault classifier in service must diagnose bearings that it has never seen, but public datasets contain few physical bearings per class, and a model may have to learn from artificial damage only. '
         'This article proposes ReliSense, a physics-informed framework that reads the squared envelope spectrum only at the fault orders fixed by bearing geometry and measured shaft speed. '
         'A kinematic branch classifies 31 order-domain features, peak-to-background ratios at the outer-race, inner-race, ball-spin, cage and shaft orders and at their modulation sidebands, with a 96-parameter logistic regression; '
         'an envelope branch classifies the same kinematic features read in four demodulation bands with a random forest. The two posteriors are averaged, and low-confidence recordings are withheld with thresholds calibrated on training bearings only. '
         'On 29 Paderborn bearings under bearing-wise protocols, ReliSense reached 95.9% when trained on artificial and tested on real damage, against 75.0% for the published vibration ensemble, 70.8% for a convolutional network and at most 58.3% for a residual network and two physics-informed networks. '
         'In leave-one-bearing-out evaluation it reached 83.7%, and abstention raised this to 97.5% at 69% coverage while the false alarms fell from 32.1% to 2.7%. '
         'With all settings fixed before testing, it reached 85.0% on unseen bearing types of the HUST dataset and 83.2% on unseen fault sizes of the CWRU dataset. Healthy bearings that resemble damage remain the main source of error.')
A = header('ReliSense: Physics-Informed Two-Branch Fault-Order Diagnosis With Training-Only Abstention for Rolling Bearings Never Seen in Training',
           'Bearing fault diagnosis, envelope analysis, generalization to unseen bearings, physics-informed learning, selective classification, vibration monitoring',
           A_ABS, 'Paper A, version 1.1')
A = A.replace('"""Manuscript v3.32 = v3.31 with', '"""Paper A v1.1 (method; v1.0 + physics-informed in title): split from manuscript v3.32 by make_split.py.\nManuscript v3.32 = v3.31 with', 1)
A += "# ================================================================== CONTENT (Paper A)\n" + R(115, 134)
A += r'''P('Despite this progress, two limitations remain. First, physics-informed models are rarely evaluated on bearings absent from training, so it is not known whether their physical prior carries over to a new bearing or to real damage, and they are seldom compared with simpler models that use the same physics. '
  'Second, confidence-based rejection is usually calibrated on data of the test domain, which is not available for a bearing that has never been seen {c:15,39}, so that a diagnosis system without such data cannot tell when to withhold its answer.')
''' + R(141, 152) + r'''LIST('A comparison with learned and physics-informed networks under the same bearing-wise protocols. A wide-kernel network, a residual network with raw and envelope input, an impulse-response kernel network and a fault-order-guided residual network were trained on the same bearings. '
     'None reached the accuracy of ReliSense under any protocol; the fault-order-guided network was the most accurate of them (78.8% under LOBO against 83.7%), which indicates that the order domain, rather than the size of the model, carries what transfers to a new bearing.')
P('Why the kinematic coordinates transfer, which kinds of damage they fail to diagnose and how bearing-wise evaluation should be designed are analyzed in a companion article {c:901}; the present article concentrates on the method and its performance. '
  'The remainder of this article is organized as follows. Section II describes ReliSense. Section III presents the dataset, the evaluation protocols and the compared methods. Section IV reports the results on the Paderborn dataset, and Section V the results on three further test rigs. '
  'Section VI analyzes the components of the method and discusses the results, Section VII states the limitations, and Section VIII concludes the article.')

H1('II. ReliSense')
P('Fig. {f:framework} shows the framework. Module I converts one 4-s vibration recording into kinematic and envelope features, and Module II classifies them in two branches and decides whether the diagnosis is accepted. '
  'The design was completed in two stages, and both are reported. The kinematic core, 15 features in a fixed band with logistic regression, was fixed in a preliminary study on 15 of the bearings (K001–K005, KA01, KA04, KA05, KA07, KA15, KI01, KI04, KI05, KI07 and KI14). '
  'The modulation features and the envelope branch were added after the results of the kinematic core under all Paderborn protocols were known; the final two-branch configuration was then fixed before it was applied to the three further datasets, which played no part in the design.')
FIG('framework', NEWDIR + 'fig_arch3a.png', 'Overall architecture of ReliSense, drawn with the real output of each step for recording N15_M07_F10_KA16_1 (outer race). Module I: the vibration is demodulated in the 2–12 kHz band, placed on the order axis with the measured shaft speed, and read at the fault orders (shaded orange) and at their modulation sidebands (shaded green) to give the 31 kinematic features z (white line: end of the 15-feature core); the envelope features e repeat the 15 core features in four demodulation bands and add two statistics. Module II: the kinematic branch (logistic regression) and the envelope branch (random forest) are averaged; the posterior of all LOBO recordings (grey: withheld) and the confidence c compared with the training-only threshold τ_b decide between a diagnosis and referral.')
'''
A += edit(R(170, 240), [("this is tested in Section IV-D", "this is tested in Section IV-C"),
                        ("For the verification controls of Module III, which need the full spectrum, S is resampled", "For the modulation features and the compared order-domain models, which need the full spectrum, S is resampled"),
                        ("and its geometry. Module III controls are run offline on the kinematic core and are not part of the decision.')", "and its geometry.')"),
                        ("FIG('flow', NEWDIR + 'fig_flow.png',", "FIG('flow', NEWDIR + 'fig_flow_A.png',")])
A += '\n' + edit(R(263, 311), [
    ("{c:29,35}. '\n  'To test whether other processing or classifiers could do better, 72 combinations of nine feature groups, four classifiers and optional per-condition standardization were also declared in advance and, in each L10 split, selected by three inner folds on the training bearings {c:45,46,47}.')",
     "{c:29,35}.')"),
    ("     ['Order-spectrum CNN', 'Order spectrum ({e:order}) up to 11.8 orders', 'Three convolutional layers, 60 epochs, shift augmentation; 8803 parameters'],\n     ['72 combinations', 'Nine feature groups', 'LR, RBF-SVM, RF, gradient boosting; nested selection on training bearings']],",
     "]")
])
A = A.replace("'ResNet-1D, 16–64 channels, 60 epochs; 256 131 parameters'],\n]", "'ResNet-1D, 16–64 channels, 60 epochs; 256 131 parameters']],")
A += '\n' + edit(R(313, 355), [("lies above the published ensemble and the envelope branch alone (84.1%).')",
                                "lies above the published ensemble and the envelope branch alone (84.1%). On L10, the published result of 98.3% {c:10} exceeds every result of this study; this gap is analyzed in {c:901}.')")])
A += edit(R(371, 377), [("H2('D. Selective Classification", "H2('C. Selective Classification")])
A += edit(R(378, 410), [("whereas the generic random forest lost 15.7. '\n  'The same pattern appeared on Paderborn for the kinematic core (Fig. {f:efficiency}): with one training bearing per class, drawn 30 times at random, the kinematic features reached 65.2% accuracy and 87.0% macro AUC, against at most 45.1% and 62.1% for the generic inputs, and with five bearings 80.2% and 95.5% against at most 57.0% and 77.1%.')",
                         "whereas the generic random forest lost 15.7.')"),
                        ]) 
A = edit(A, [("FIG('efficiency', NEWDIR + 'fig_efficiency.png', 'Accuracy of the kinematic core and of two generic inputs on unseen Paderborn bearings against the number of training bearings per class (mean ± SD over 30 random draws, slightly offset for legibility; one test bearing per class in each draw).')\n", "")])
A += "H1('VI. Analysis and Discussion')\nH2('A. Components of ReliSense')\n" + R(443, 453)
A += edit(R(503, 517), [("H2('E. Comparison", "H2('B. Comparison"), ("On L10, the published 98.3% exceeds all results of this study, for the reasons discussed in Section IV-C.')",
                                                                            "On L10, the published 98.3% exceeds all results of this study; possible reasons are analyzed in {c:901}.')")])
A += r'''H2('C. Discussion')
P('The results support one central finding: a diagnosis expressed in the coordinates fixed by the bearing kinematics carries over to bearings that were never seen in training. This section discusses what the two branches and the abstention rule add, how the method relates to learned and physics-informed networks, and what deployment requires; why the kinematic coordinates transfer is analyzed in {c:901}.')
'''
A += edit(R(529, 543), [("RUN('2) What the Two Branches", "RUN('1) What the Two Branches"), ("RUN('3) The Healthy", "RUN('2) The Healthy")])
A += edit(R(564, 578), [("RUN('7) Relation", "RUN('3) Relation"), ("RUN('8) Deployment", "RUN('4) Deployment"),
                        ("which ReliSense diagnosed at 25.0%, 31.2% and 67.5%, and the per-bearing accuracies of ReliSense and WDCNN correlated with ρ = 0.64, against 0.90–0.94 among the feature models. '",
                         "which ReliSense diagnosed at 25.0%, 31.2% and 67.5%. '")])
A += r'''H1('VII. Limitations and Future Work')
P('The study has limitations. The main analysis uses one rig, one bearing type and four operating conditions, and the real damage grew in accelerated lifetime tests rather than in service. The external datasets add three rigs and six bearing designs, but their damage was seeded, each has few physical bearings, and no model was trained on one machine and tested on another; '
  'the results therefore concern unseen bearings of a known rig, not unseen machines or sensor positions. Fifteen of the 29 bearings informed the design of the kinematic core, including four of the 11 L8 test bearings, and the modulation features and the envelope branch were added after the Paderborn results of the core were known, so that the Paderborn results of the final configuration are not fully independent of its design; the three further datasets were analyzed only after it was fixed. '
  'The envelope branch is a random forest and is less transparent than the linear core; the controls that verify the kinematic content were applied to the core and are reported in {c:901}. The compared networks did not receive the bandwidth, observation length and harmonic range of ReliSense; the two physics-informed networks were implemented following the published principles rather than the original code, were not tuned, and were trained with one seed under L10 and LOBO. '
  'Healthy bearings remain the main weakness, with 32.1% of healthy recordings diagnosed as faulty under LOBO and 43.8% on L8 over all conditions before abstention. A natural extension is to accept a diagnosis only when several harmonics of the candidate fault agree and stand out from matched incorrect orders, '
  'and to test the framework across machines, for example on the constant-speed Ottawa collection {c:57}, whose 20 bearings allow bearing-wise splits, together with calibration under distribution shift and conformal prediction {c:58,59,60}.')
H1('VIII. Conclusion')
P('This article addressed the diagnosis of bearings that a model has never seen. ReliSense reads the squared envelope spectrum at the fault orders fixed by bearing geometry and measured speed, classifies the peaks and modulation sidebands with a 96-parameter logistic regression and the same kinematic features in four demodulation bands with a random forest, averages the two, '
  'and withholds low-confidence diagnoses with thresholds calibrated on training bearings only. On 29 Paderborn bearings, it reached 95.9% when trained on artificial and tested on real damage, against 75.0% for the published ensemble, 70.8% for a convolutional network and 58.3% for the best of four residual and physics-informed networks, '
  'and 83.7% under leave-one-bearing-out evaluation, which abstention raised to 97.5% at 69% coverage with 2.7% false alarms. With settings fixed in advance, it was the most accurate method on unseen bearing types of HUST (85.0%) and unseen fault sizes of CWRU (83.2%). '
  'Among the compared networks, only the one that received the fault orders came close, which indicates that the order domain carries most of what transfers to a new bearing. '
  'Healthy bearings that resemble damage, small real indentations, repeated and distributed damage and operation at low speed remain the main limits of the method.')

'''
A += TAIL
open('build_A.py', 'w').write(A)

# ===================================================================================== PAPER B
B_ABS = ('Physics-informed features read at the fault orders of a bearing often diagnose bearings absent from training better than generic or learned representations, but why they do so, which damage they miss and how such results should be evaluated have not been examined. '
         'This article studies a kinematic core of 15 peak-to-background features at three harmonics of five fault orders, classified by logistic regression, on 29 Paderborn bearings and two further test rigs, under protocols in which every choice is made on training bearings. '
         'Fault-order perturbation showed that the true orders rank first among 31 order sets. A bearing-identity test revealed an inverse relation between identity and transfer: band energies and time and band statistics named the individual bearing of a recording in 97.7% and 98.2% of cases but diagnosed unseen bearings at only 50.1% and 55.2%, '
         'whereas the kinematic core named 70.6% and diagnosed 80.1%. Defect size did not predict whether a defect was found, whereas the contrast of the defect line against a healthy bearing of the same type did (Spearman ρ = 0.81, n = 16). '
         'A fault-order model trained on artificial damage only was at least as accurate on real-damage bearings as one that had also seen real damage, so that the gap between artificial and real damage was not the main obstacle. '
         'Finally, a published accuracy of 98.3% on the real-damage protocol could not be reproduced, and model selection with knowledge of the test results raised accuracy by up to 19.5 points. Four guidelines for evaluating diagnosis on unseen bearings are derived.')
B = header('Why Physics-Informed Fault-Order Features Transfer to Unseen Bearings: Bearing Identity, Detectability and Evaluation Pitfalls',
           'Bearing fault diagnosis, bearing-wise evaluation, data leakage, envelope analysis, fault orders, generalization to unseen bearings, interpretability, physics-informed learning',
           B_ABS, 'Paper B, version 1.1')
B = B.replace('"""Manuscript v3.32 = v3.31 with', '"""Paper B v1.1 (analysis; v1.0 + physics-informed in title, abstract, index terms): split from manuscript v3.32 by make_split.py.\nManuscript v3.32 = v3.31 with', 1)
B += r'''# ================================================================== CONTENT (Paper B)
H1('I. Introduction')
P('Vibration diagnosis of rolling-element bearings rests on a simple physical fact: a localized defect is struck at a rate fixed by the shaft speed and the bearing geometry, and envelope analysis turns these impacts into lines at characteristic fault orders {c:1,2}. '
  'Data-driven methods have since reported accuracies close to 100% on public benchmarks {c:3,4,5,9}, but on the Paderborn dataset {c:10,11} the choice of data split alone changed accuracy by more than 40%, and many published studies were judged likely to be affected by leakage {c:12}. '
  'When the test bearings are kept out of training, the reported accuracies fall sharply {c:10,15}.')
P('Two explanations are usually given for this fall. The first attributes it to a distribution gap between training and test data, for example between artificial and real damage, and addresses it with transfer learning and domain adaptation {c:24,25,26,27}. '
  'The second attributes it to the small number of physical bearings per class, which lets a model separate its training bearings by properties unrelated to the damage, such as mounting, run-in history or sensor gain {c:13,14,15}. '
  'Physics-informed features and networks are expected to help in both cases {c:29,30,31,32,33,34,35}, but it has rarely been tested whether their accuracy actually comes from the physics, which damage they fail to detect, and how much of a reported accuracy depends on the evaluation itself.')
P('A companion article {c:900} proposed ReliSense, a two-branch method that reads the squared envelope spectrum only at the fault orders and withholds uncertain diagnoses. The present article does not propose a new method. '
  'It analyzes the kinematic core on which that method is built, 15 fault-order features classified by logistic regression, together with generic and learned representations, to explain why fault-order features transfer to unseen bearings and where they fail. '
  'All analyses use bearing-wise protocols in which every choice is made on training bearings, and all controls were fixed before execution. The contributions are as follows.')
LIST('A verification protocol that tests the physical basis of a diagnosis directly. Fault-order perturbation with scaled and random orders, a representation ladder from the waveform to the fault orders, and a bearing-identity test show that the true fault orders rank first among 31 order sets and reveal an inverse relation between identity and transfer: '
     'representations that named the individual bearing in 97.7%–98.2% of cases diagnosed unseen bearings worst (50.1%–55.2%), against 80.1% for the kinematic core. In five settings on three rigs, a generic input was always less accurate than a kinematic input with the same classifier. This identifies bearing identity, rather than the distribution gap between artificial and real damage, as the main obstacle to transfer.')
LIST('A detectability criterion for new bearings. The contrast of the defect line against a healthy bearing of the same type predicted the accuracy on unseen bearings of two rigs (Spearman ρ = 0.81, n = 16), and every bearing with a contrast above 1.1 was diagnosed without error, whereas defect size and bearing size did not predict accuracy. '
     'Together with a damage-wise analysis, the criterion identifies the damage that is hardest to detect, small real indentations, repeated and distributed damage and operation at low speed, and gives the operator a measurable reason to distrust a negative diagnosis.')
LIST('Evidence that artificial damage is adequate training data for real damage when the representation is fixed by kinematics. The maximum mean discrepancy between artificial and real recordings was not smaller for the kinematic features than for band energies, '
     'yet on the ten real-damage test bearings of the protocol of the dataset authors, a fault-order model trained on artificial damage only was at least as accurate as one that had also seen real damage (95.5% against 90.0% at condition 0). The limit lies in repeated, distributed and indented real damage, which the artificial damage does not represent.')
LIST('A strict bearing-wise evaluation with guidelines. Under protocols in which every choice is made on training bearings, the published 98.3% on the real-damage protocol could not be reproduced, model selection with knowledge of the test results raised accuracy by up to 19.5 points, and a dataset with one bearing per class gave perfect accuracy for features that only recognize known bearings. '
     'These results lead to four practical guidelines for evaluating diagnosis on unseen bearings.')
P('Section II defines the kinematic core and the verification controls. Section III describes the data, the protocols and the compared representations. Section IV reports the results, Section V discusses them, Section VI states the limitations, and Section VII concludes the article.')

H1('II. Kinematic Core and Verification Controls')
H2('A. Kinematic Core')
P('For a bearing with n rolling elements of diameter d, pitch diameter D and contact angle φ, rotating at the shaft frequency f_r, a local defect is struck at the characteristic frequencies {c:1}')
EQT('orders1', 'BPFO = (n/2)·f_r·(1 − (d/D)cosφ),   BPFI = (n/2)·f_r·(1 + (d/D)cosφ)')
EQT('orders2', 'BSF = (D/2d)·f_r·(1 − ((d/D)cosφ)²),   FTF = (1/2)·f_r·(1 − (d/D)cosφ).')
P('Divided by f_r, they give fault orders that depend on the geometry only; for the 6203 bearing of the Paderborn rig, the outer- and inner-race orders are 3.05 and 4.95. Each 4-s recording, sampled at 64 kHz, is demodulated in the 2–12-kHz band, the spectrum S(f) of its squared envelope is computed, and the spectrum is placed on the order axis with the measured shaft speed {c:900},')
EQT('order', 'O(r_l) = log[ S(r_l·f_r) / median_l S(r_l·f_r) ].')
P('For each fault order k ∈ {BPFO, BPFI, BSF, FTF, shaft} and harmonic h ∈ {1, 2, 3}, the core feature compares the largest value of S in a narrow search band 𝓑_kh around the expected frequency f_kh with the median of a surrounding background band 𝓖_kh,')
EQT('feature', 'φ_kh = log[ max_{f∈𝓑_kh} S(f) / median_{f∈𝓖_kh} S(f) ],')
P('where the search band has a half-width of max(0.02·f_kh, 1.5 frequency bins) and the background band extends to ±10 Hz. The 15 features are standardized with training statistics and classified by multinomial logistic regression with L2 regularization (C = 0.3), which gives 48 trained parameters. '
  'The core was fixed in a preliminary study on 15 of the 29 bearings (K001–K005, KA01, KA04, KA05, KA07, KA15, KI01, KI04, KI05, KI07 and KI14), before any analysis reported here. Where results of the full two-branch ReliSense are shown for comparison, they are taken from {c:900}.')
H2('B. Verification Controls')
'''
B += edit(R(242, 261), [("P('Module III tests whether the accuracy", "P('Five controls test whether the accuracy"),
                        ("; the branches of ReliSense are also evaluated alone.')", ".')")])
B += r'''
H1('III. Data, Protocols and Compared Representations')
P('The main data are the Paderborn bearing dataset {c:10,11}: 6203 bearings on one modular rig, housing vibration sampled at 64 kHz, four operating conditions (condition 0: 1500 rpm, 0.7 Nm and 1000 N; the others with 900 rpm, 0.1 Nm or 400 N and the remaining values unchanged) and 20 recordings of 4 s per bearing and condition. '
  'The 29 bearings with one damaged ring are used: six healthy, twelve with artificial damage made by electrical discharge machining, drilling or engraving, and eleven with real damage from accelerated lifetime tests, 2319 recordings in total. Damage descriptions and extent levels are taken from {c:10} (level 1: damage length up to 2 mm; level 2: 2–4.5 mm; level 3: 4.5–13.5 mm). '
  'For the effect of bearing and fault size, the CWRU {c:51} and HUST {c:52} datasets are used with the segmentation and protocols of {c:900}: one defect diameter (CWRU) or one bearing type (HUST, 6204–6208) is held out at a time. The Ottawa data {c:53}, with one bearing per class, serve to illustrate the recognition of known bearings.')
P('All splits are defined by bearing, and no bearing appears in both partitions. L8 is the artificial-to-real protocol of the dataset authors {c:10}: K002, KA01, KA05, KA07, KI01, KI05 and KI07 for training, and K001 with ten real-damage bearings for testing. L10 is their real-damage protocol, with three training and two test bearings per class in 10 combinations of 15 bearings. '
  'A2R trains on three healthy and all artificially damaged bearings and tests on the remaining healthy and all real single-damage bearings, and LOBO holds out each of the 29 bearings once. Standardization, thresholds and any model choice use the training bearings only.')
P('The kinematic core is compared with generic representations of the same recordings: 32 band energies and time and band statistics (with logistic regression or a random forest), the 590-bin order spectrum O(r) of ({e:order}) with logistic regression and with a three-layer one-dimensional CNN, and a wide-kernel convolutional network (WDCNN) {c:6} on 4096-sample windows of raw vibration and of the envelope. '
  'To test whether other processing could do better on L10, 72 combinations of nine feature groups, four classifiers (logistic regression, RBF-SVM, random forest and gradient boosting) and optional per-condition standardization were declared in advance and selected in each split by three inner folds on the training bearings {c:45,46,47}. '
  'Accuracy is the share of correctly diagnosed 4-s recordings. Because the groups of the damage-wise analysis contain one to nine bearings, all tests are descriptive.')

H1('IV. Results')
'''
B += edit(R(413, 441), [("in line with the false alarms of Section IV-B.')", "in line with the false alarms reported in {c:900}.')"),
                        ("rather than a ball defect.')", "rather than a ball defect.')"),
                        ("(Fig. {f:kinematics}(c))", "{c:900}"),
                        ("80.1% for the 15 core features and 83.7% for ReliSense. '", "and 80.1% for the 15 core features. '"),
                        ("     ['15 kinematic features (core)', 'LR', '80.1', '70.6', '0.54'], ['31 kinematic and 62 envelope features (ReliSense)', 'LR and RF', '83.7', '–', '–']],",
                         "     ['15 kinematic features (core)', 'LR', '80.1', '70.6', '0.54']],")])
B += "H2('B. Ablation of the Kinematic Core')\n" + edit(R(454, 468), [("as additional readings in the envelope branch, the same variants", "as additional readings in the envelope branch of ReliSense {c:900}, the same variants")])
B += r'''H2('C. Data Efficiency')
P('The advantage of the kinematic features grew when training bearings were scarce (Fig. {f:efficiency}). With one training bearing per class, drawn 30 times at random, the kinematic features reached 65.2% accuracy and 87.0% macro AUC on unseen Paderborn bearings, against at most 45.1% and 62.1% for the generic inputs, and with five bearings 80.2% and 95.5% against at most 57.0% and 77.1%.')
FIG('efficiency', NEWDIR + 'fig_efficiency.png', 'Accuracy of the kinematic core and of two generic inputs on unseen Paderborn bearings against the number of training bearings per class (mean ± SD over 30 random draws, slightly offset for legibility; one test bearing per class in each draw).')
'''
B += edit(R(469, 495), [("H2('C. Damage-Wise Analysis')", "H2('D. Damage-Wise Analysis')"),
                        ("P('Fig. {f:matrix} shows the LOBO accuracy of every bearing for ReliSense, its branches, its kinematic core and two further models",
                         "P('Fig. {f:matrix} shows the LOBO accuracy of every bearing for the kinematic core, for ReliSense and its two branches {c:900} and for two further models"),
                        ("so that order normalization helped without making the features speed-invariant. '\n    'The run-in time of the healthy bearings did not explain their accuracy (ρ = 0.23, n = 6).')",
                         "so that order normalization helped without making the features speed-invariant.')")])
B += edit(R(496, 502), [("H2('D. Effect of Defect-Line Strength, Bearing Size and Fault Size')", "H2('E. Defect-Line Strength, Bearing Size and Fault Size')"),
                        ("On HUST, the accuracy of ReliSense on the five unseen types", "On HUST, the accuracy of ReliSense {c:900} on the five unseen types")])
B += edit(R(356, 370), [("H2('C. Real-Damage Protocol and the Published Result')", "H2('F. Real-Damage Protocol and the Published Result')"),
                        ("P('On L10, ReliSense reached 78.7% over all conditions", "P('On L10, the two-branch ReliSense {c:900} reached 78.7% over all conditions"),
                        ("ReliSense was not evaluated on L10 at condition 0 alone.')", "ReliSense values from {c:900}; it was not evaluated on L10 at condition 0 alone.')")])
B += r'''
H1('V. Discussion')
P('The results support one central finding: a diagnosis expressed in coordinates fixed by the bearing kinematics carries over to bearings never seen in training because these coordinates do not describe the individual bearing. This section discusses why this happens, why defect size is a poor guide to detectability, what artificial damage can and cannot teach, and what the results imply for evaluation.')
'''
B += edit(R(521, 528), [("(Fig. {f:motivation}(a) and Table {t:repr})", "(Table {t:repr})"),
                        ("on the 62 envelope features it reached 82.5% under LOBO, close to the kinematic branch,", "on the 62 envelope features of {c:900} it reached 82.5% under LOBO, close to the kinematic core (80.1%),"),
                        ("FIG('discinputs', NEWDIR + 'fig_disc_inputs.png', 'Accuracy on unseen bearings by input and classifier in five settings. Grey, open: generic inputs (band energies with LR; time and band statistics with RF, on Paderborn with LR); blue, filled: inputs read at the fault orders. Circles and diamonds: logistic regression; squares: random forest; star: ReliSense.",
                         "FIG('discinputs', NEWDIR + 'fig_disc_inputs_B.png', 'Accuracy on unseen bearings by input and classifier in five settings. Grey, open: generic inputs (band energies with LR; time and band statistics with RF, on Paderborn with LR); blue, filled: inputs read at the fault orders (kinematic core with LR; envelope features of {c:900} with RF). Circles: logistic regression; squares: random forest.")])
B += edit(R(544, 549), [("RUN('4) Detectability", "RUN('2) Detectability")])
B += edit(R(550, 554), [("RUN('5) Artificial", "RUN('3) Artificial"), ("On L8, which trains on artificial damage and tests on real damage, ReliSense reached", "On L8, which trains on artificial damage and tests on real damage, ReliSense {c:900} reached"),
                        ("FIG('disca2r', NEWDIR + 'fig_disc_a2r.png', 'Accuracy of ReliSense on", "FIG('disca2r', NEWDIR + 'fig_disc_a2r.png', 'Accuracy of ReliSense {c:900} on")])
B += edit(R(555, 563), [("RUN('6) What Published", "RUN('4) What Published"),
                        ("The gap between the published 98.3% on L10 and the 78.7% obtained here illustrates", "The gap between the published 98.3% on L10 and the 74.6% of the kinematic core (78.7% for ReliSense {c:900}) illustrates"),
                        ("['', 'ReliSense, all settings fixed in advance', '78.7']", "['', 'Kinematic core, fixed in advance', '74.6']"),
                        ("['', 'ReliSense', '98.3 / 95.9']", "['', 'Kinematic core', '94.2 / 89.8']"),
                        ("['', 'ReliSense', '85.0']", "['', 'Kinematic core', '77.2']")])
B += r'''H1('VI. Limitations and Future Work')
P('The study has limitations. The main analysis uses one rig, one bearing type and four operating conditions, the real damage grew in accelerated lifetime tests rather than in service, and the external datasets have few physical bearings with seeded damage. '
  'Fifteen of the 29 bearings informed the design of the kinematic core, including four of the 11 L8 test bearings, so that its Paderborn results are not fully independent of its design. The groups of the damage-wise analysis contain one to nine bearings, and the associations reported there are descriptive. '
  'The defect-line contrast requires a healthy bearing of the same type and was evaluated on 16 bearings of two rigs. The aspects of the published L10 evaluation that may explain its higher accuracy are possibilities that this study could not test. '
  'The guidelines should be tested on further datasets with many bearings per class, such as the constant-speed Ottawa collection {c:57}, and the verification controls should be applied to physics-informed networks.')
H1('VII. Conclusion')
P('This article examined why features read at the fault orders of a bearing transfer to bearings absent from training. Controls that corrupt the fault orders, a representation ladder and a bearing-identity test showed that the accuracy of the kinematic core comes from its kinematic content and that representations which identify the individual bearing transfer worst, '
  'so that bearing identity, rather than the gap between artificial and real damage, is the main obstacle to transfer. Whether a defect was found depended on the contrast of its line against a healthy bearing of the same type rather than on its size; repeated, distributed and indented real damage and operation at low speed were the hardest cases. '
  'Under strictly bearing-wise evaluation, a published accuracy of 98.3% could not be reproduced, and model selection with knowledge of the test results was worth up to 19.5 points. Diagnosis on unseen bearings should therefore be evaluated per physical bearing, with every choice made on training bearings, and reported with the false-alarm rate and the number of training bearings.')

'''
B += TAIL
open('build_B.py', 'w').write(B)
print('build_A.py and build_B.py written')
