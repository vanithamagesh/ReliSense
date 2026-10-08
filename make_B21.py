"""Creates build_B21.py from build_B2.py: Paper B v2.1 = v2.0 with the phase 24 benchmark (tuned classical baselines and
domain-generalization networks, Colab results pasted on 5 October 2026; phase24_benchmark_baselines.py v1.0/v1.1)."""
s = open('build_B2.py').read()


def rep(o, n):
    global s; assert s.count(o) == 1, (s.count(o), o[:90]); s = s.replace(o, n)


s = s.replace('"""Paper B v2.0 (benchmark', '"""Paper B v2.1 (v2.0 + phase 24 benchmark of tuned and domain-generalization baselines; make_B21.py). Paper B v2.0 (benchmark', 1)
rep("Paper B, version 2.0", "Paper B, version 2.1")
i = s.index("_mkref(901,"); j = s.index("\n", i) + 1
s = s[:j] + (
    "_mkref(902, 'Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, and V. Lempitsky, “Domain-adversarial training of neural networks,” J. Mach. Learn. Res., vol. 17, no. 59, pp. 1–35, 2016.')\n"
    "_mkref(903, 'B. Sun and K. Saenko, “Deep CORAL: Correlation alignment for deep domain adaptation,” in Proc. Eur. Conf. Comput. Vis. Workshops, 2016, pp. 443–450.')\n"
    "_mkref(904, 'S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang, “Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2020.')\n"
    "_mkref(905, 'I. Gulrajani and D. Lopez-Paz, “In search of lost domain generalization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2021.')\n") + s[j:]
# abstract
rep("Finally, a published accuracy of 98.3% could not be reproduced,",
    "Generic representations remained 14.6–27.6 points below the fault-order features when their classifiers were tuned on training bearings, and networks trained with three domain-generalization objectives did not exceed the fault-order features under any protocol. Finally, a published accuracy of 98.3% could not be reproduced,")
# contribution 1
rep("Bearing identity, rather than the distribution gap between artificial and real damage, is therefore the main obstacle to transfer.')",
    "Tuning the classifiers of the generic representations on training bearings did not close the gap, and neither did domain-adversarial, covariance-alignment or group-robust training of networks that treat each training bearing as a domain. Bearing identity, rather than the distribution gap between artificial and real damage, is therefore the main obstacle to transfer.')")
# methods
rep("the hyperparameters of the classical classifiers were set to common default values. '",
    "the hyperparameters of these classical classifiers were set to common default values. '\n"
    "  'To test whether stronger baselines change the conclusions, two further sets were evaluated. First, band energies, time and band statistics and the kinematic core were classified by logistic regression, an RBF-SVM, a random forest and gradient boosting, each with a small hyperparameter grid; the hyperparameters, and in a selected variant also the classifier, were chosen by stratified group cross-validation over the training bearings only (L8 is excluded, because its single healthy training bearing cannot be held out). '\n"
    "  'Second, networks were trained with objectives that treat each training bearing as a domain {c:905}: empirical risk minimization with bearing-balanced batches (ERM), domain-adversarial training (DANN) {c:902}, correlation alignment (CORAL) {c:903} and group distributionally robust optimization (group DRO) {c:904}, on 4096-sample envelope windows with a wide-kernel network and on the order spectrum with a small one-dimensional CNN; the strength of each objective was chosen on one bearing-wise validation split of the training bearings. '")
# results subsection H with benchmark table
rep("\nH1('V. Discussion')", r'''
H2('H. Optimized and Domain-Generalization Baselines')
P('Table {t:bench} gives the results. Tuning did not rescue the generic representations: with the classifier and its hyperparameters chosen on training bearings, band energies reached 50.5%, 58.9% and 52.1% on L10, A2R and LOBO, and time and band statistics 45.7%, 50.7% and 55.5%, 14.6–27.6 points below the kinematic core under the same selection. '
  'The kinematic core gained nothing from tuning: logistic regression was its best classifier, and the more flexible classifiers were less accurate under LOBO (75.0%–78.1% against 79.7%). '
  'The domain-generalization objectives did not close the gap either. On envelope windows, ERM with bearing-balanced batches reached 75.9% under LOBO, more than the earlier wide-kernel network (68.1%), but 10.9–25.9 points less than the core on L8, L10 and A2R, and domain-adversarial training lowered LOBO accuracy to 59.1%. '
  'On the order spectrum, whose axis is already fixed by the kinematics, the networks came closer: CORAL reached 65.8% on L10 and 70.4% on A2R, and DANN 79.1% under LOBO, 1.0 point below the core, but DANN collapsed to 24.1% on L8 at condition 0, where only seven bearings are available for training, and no objective exceeded the core under any protocol. '
  'Aligning the training bearings thus helped little and inconsistently, whereas moving the input to the fault orders helped most, which is the same conclusion that the identity test gives from a different direction.')
TAB('bench', 'Benchmark With Tuned Classical Baselines and Domain-Generalization Networks (Recording Accuracy, %)', ['Representation and training', 'L8 cond. 0', 'L8 all', 'L10', 'A2R', 'LOBO', 'LOBO false alarms'],
    [['Kinematic core, LR (fixed in advance)', '93.6', '82.8', '74.6', '73.0', '80.1', '41.0'],
     ['Kinematic core, tuned classifier', '–', '–', '73.2', '73.5', '79.7', '43.3'],
     ['Band energies, tuned classifier', '–', '–', '50.5', '58.9', '52.1', '24.2'],
     ['Time and band statistics, tuned classifier', '–', '–', '45.7', '50.7', '55.5', '26.5'],
     ['Envelope windows, ERM', '67.7', '61.1', '58.9', '62.1', '75.9', '36.9'],
     ['Envelope windows, DANN', '56.8', '44.3', '52.2', '65.5', '59.1', '41.7'],
     ['Order spectrum, ERM', '63.2', '59.0', '63.1', '67.1', '75.8', '39.0'],
     ['Order spectrum, DANN', '24.1', '60.2', '50.3', '66.2', '79.1', '36.2'],
     ['Order spectrum, CORAL', '66.4', '64.9', '65.8', '70.4', '76.8', '37.3'],
     ['Order spectrum, group DRO', '63.2', '58.5', '63.1', '63.4', '75.9', '38.8']],
    [3300, 1050, 950, 950, 950, 950, 1150],
    'Recordings pooled over folds; one seed. Tuned classifier: LR, RBF-SVM, random forest or gradient boosting with hyperparameters and classifier selected by stratified group cross-validation over training bearings; not possible on L8 (one healthy training bearing). Networks: objective strength selected on a bearing-wise validation split. False alarms: healthy recordings diagnosed as damaged. CORAL and group DRO on envelope windows were not completed.')

H1('V. Discussion')''')
# discussion part 1
rep("'and on the external rigs time and band statistics remained 6.1–19.8 points below the linear kinematic core even when classified by a random forest, which is flexible enough to memorize individual bearings (on Paderborn, with logistic regression, 24.9 points).')",
    "'and on the external rigs time and band statistics remained 6.1–19.8 points below the linear kinematic core even when classified by a random forest, which is flexible enough to memorize individual bearings (on Paderborn, with logistic regression, 24.9 points). '\n"
    "    'Nor did tuning or domain alignment change this (Table {t:bench}): objectives that make the representation of the training bearings alike act on the bearings that are available, but they cannot remove bearing-specific content that a new bearing does not share, whereas fault-order coordinates do not contain it in the first place.')")
# limitations
rep("The baselines cover generic features with four classifiers and nested selection and a wide-kernel network; domain-generalization methods that align the training bearings, such as adversarial or covariance-alignment training, were not evaluated in this version.",
    "The domain-generalization networks were trained with one seed and a small search over the strength of each objective, CORAL and group DRO on envelope windows were not completed for lack of computing resources, and the order spectrum was not included among the tuned classical baselines.")
open('build_B21.py', 'w').write(s)
print('build_B21.py written')
