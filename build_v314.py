"""Manuscript v3.14 = v3.13 with four stronger contributions. v3.13 = v3.12 rewritten in research-article form (no research-question list) with redrawn figures. v3.12: of v3.11 around three research questions (about 10,000 words).
All numbers are those of v3.11 (phases 2-12); no new experiment. Figures and equations are reused from v3.11,
Fig. 1 is redrawn without the shared-split reference line (fig01_overview.py).
Usage: python build_v312.py <v3.11.docx> <phase12_numbers.json> <fig01_overview.png> <out.docx>"""
import copy, json, re, sys
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_COLOR_INDEX
from lxml import etree

SRC, N12F, FIG1, OUT = sys.argv[1:5]
N12 = json.load(open(N12F)); T12 = N12['tables']
NEWDIR = '../figs13/'
doc = Document(SRC); body = doc.element.body
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'; q = lambda t: f'{{{W}}}{t}'
txt = lambda e: ''.join(t.text or '' for t in e.iter(q('t')))


def el_starting(s, tag='p'):
    for e in body.iterchildren():
        if e.tag == q(tag) and txt(e).startswith(s): return e
    raise KeyError(s)


# ------------------------------------------------------------------ templates and reusable elements (copied before deletion)
T_H1 = copy.deepcopy(el_starting('II. Proposed Method')); T_H2 = copy.deepcopy(el_starting('B. Module II: Diagnosis'))
T_P = copy.deepcopy(el_starting('Rolling-element bearings support')); T_RUN = copy.deepcopy(el_starting('7) Computational Cost'))
T_LIST = copy.deepcopy(el_starting('A kinematic signature module'))
_tn = el_starting('TABLE V'); T_TN = copy.deepcopy(_tn); T_TT = copy.deepcopy(_tn.getnext()); T_TBL = copy.deepcopy(_tn.getnext().getnext()); T_NOTE = copy.deepcopy(el_starting('L10: mean ± SD over 10 splits'))
T_CAP = copy.deepcopy(el_starting('Fig. 3. ')); T_FIGP_PPR = copy.deepcopy(el_starting('Fig. 3. ').getprevious().find(q('pPr')))
OLDFIG = {int(re.match(r'Fig\. (\d+)\. ', txt(e)).group(1)): copy.deepcopy(e.getprevious())
          for e in body.iterchildren() if e.tag == q('p') and re.match(r'Fig\. \d+\. ', txt(e))}
EQ = {k: copy.deepcopy(el_starting(s)) for k, s in
      [(1, 'BPFO = (n/2)'), (2, 'BSF = (D/2d)'), (3, 'xₐ(t) ='), (4, 'S(f) = |'), (5, 'O(r) = log'), (6, 'φ_kh = log'),
       (7, 'p(k | z)'), (8, 'τ_b(κ)'), (9, 'F(i, θ)')]}
REFP = {}
ref_head = el_starting('References')
for e in list(ref_head.itersiblings()):
    m = re.match(r'\[(\d+)\]', txt(e))
    if m: REFP[int(m.group(1))] = e

# ------------------------------------------------------------------ title block edits; remove the old body
for p in doc.paragraphs:
    for r in p.runs:
        if 'Manuscript version 3.11' in r.text: r.text = r.text.replace('3.11', '3.14')
start = el_starting('I. Introduction'); stop = el_starting('Acknowledgment')
e = start
while e is not stop:
    nxt = e.getnext(); body.remove(e); e = nxt
anchor = stop


# ------------------------------------------------------------------ helpers
def set_runs(e, parts):
    runs = e.findall(q('r')); proto = [copy.deepcopy(x) for x in runs] or [etree.Element(q('r'))]
    for x in runs: e.remove(x)
    for i, (t, hl) in enumerate(parts):
        x = copy.deepcopy(proto[min(i, len(proto) - 1)])
        for tt in x.findall(q('t')): x.remove(tt)
        rpr = x.find(q('rPr'))
        if rpr is None: rpr = etree.Element(q('rPr')); x.insert(0, rpr)
        for h in rpr.findall(q('highlight')): rpr.remove(h)
        if hl: h = etree.SubElement(rpr, q('highlight')); h.set(q('val'), 'yellow')
        tn = etree.SubElement(x, q('t')); tn.text = t; tn.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        e.append(x)
    return e


def put(e): anchor.addprevious(e); return e
BLOCKS = []   # (kind, ...) resolved after all keys are known
def H1(t): BLOCKS.append(('H1', t))
def H2(t): BLOCKS.append(('H2', t))
def P(t): BLOCKS.append(('P', t))
def RUN(title, t): BLOCKS.append(('RUN', title, t))
def LIST(t): BLOCKS.append(('LIST', t))
def EQN(k): BLOCKS.append(('EQ', k))
def FIG(key, src, cap): BLOCKS.append(('FIG', key, src, cap))
def TAB(key, title, header, rows, widths, note='', note_hl=None): BLOCKS.append(('TAB', key, title, header, rows, widths, note, note_hl))

# ================================================================== CONTENT
H1('I. Introduction')
P('Rolling-element bearings support almost every rotating shaft in industrial machinery, and their failure is a leading cause of unplanned downtime {c:1}. '
  'A localized defect produces a short impact each time a rolling element passes over it, at a rate fixed by the shaft speed and the bearing geometry. '
  'The impacts excite structural resonances, and envelope analysis demodulates them so that the impact rate appears as discrete lines in the envelope spectrum {c:2}. '
  'Because these lines are located by kinematics alone, envelope analysis has long been the industrial reference for bearing diagnosis.')
P('Data-driven diagnosis now dominates research on the same task {c:3,4,5}. Convolutional networks, transformers and transfer-learning methods learn features directly from vibration {c:6,7,8}, '
  'and reviews report accuracies close to 100% on public benchmarks {c:9}. How far such accuracies carry over to a bearing that the model has never seen depends on how the data are partitioned. '
  'On the widely used Paderborn dataset {c:10,11}, Wheat et al. {c:12} found that the choice of splitting method changed accuracy by more than 40%, and among 55 Paderborn studies they judged ten likely to be affected by data leakage. '
  'When training and test bearings are kept apart, as in the protocols of the dataset authors, the published accuracy for transfer from artificial to real damage is 62.3%–65.9% for single classifiers and 75.0% for an ensemble {c:10}. '
  'A monitoring system in service always meets bearings that it has not seen, so this is the setting that matters in practice.')
P('Diagnosing an unseen bearing poses three difficulties. First, public datasets contain many recordings of few physical bearings, six or seven per class on Paderborn, '
  'so that a model can separate its training bearings by properties unrelated to the damage, such as mounting, run-in history or sensor gain {c:13,14}; the number of distinct training bearings, '
  'rather than the number of recordings, largely decides generalization {c:15}. Second, training data may contain only artificial damage, whereas damage in service grows by fatigue or plastic deformation and differs in shape and extent {c:10}. '
  'Third, small or distributed damage produces weak impacts, and lower speed or load weakens them further, so that the informative part of the signal is small relative to resonances, rig components and noise, and its frequency moves with speed {c:16,17}.')
P('Existing approaches act at different stages of the diagnosis chain. Signal-processing methods select a demodulation band and read the envelope spectrum: the kurtogram selects the most impulsive band {c:18}, '
  'later indicators combine several criteria or bands {c:19,20}, and cepstral pre-whitening removes deterministic components {c:21,22}. These methods are interpretable, but the selected band is not always the one excited by the defect, and the decision usually requires an expert. '
  'Data-driven methods learn the whole chain, from wide-kernel networks and kernels adapted to the fault signal {c:6,23} to transfer learning and domain generalization {c:24,25,26}; '
  'transfer from artificial to natural damage has been addressed with meta-learning and domain adaptation that use data of the target domain during training {c:27}. '
  'Physics-informed networks add characteristic-frequency priors {c:29}, physical modal properties {c:30}, fault-frequency-guided decomposition {c:31} or interpretable filtering and enveloping layers {c:32,33}, '
  'and fault orders have been used to weight envelope features {c:34} and to guide spectral attention {c:35}. Uncertainty estimates and selective prediction have been proposed to make such diagnoses trustworthy {c:42,43,61,62}.')
P('Despite this progress, three limitations remain. First, physics-informed models are rarely evaluated on bearings absent from training, so it is not known whether their physical prior carries over to a new bearing or to real damage. '
  'Second, where accuracy on unseen bearings is reported, it is seldom verified with controls that corrupt the physics, and the damage that the method fails to diagnose is seldom identified. '
  'Third, confidence-based rejection is usually calibrated on data of the test domain, which is not available for a bearing that has never been seen {c:15,39}.')
P('To address these issues, this article proposes ReliSense, a physics-informed framework for the diagnosis of unseen bearings. ReliSense reads the squared envelope spectrum only at the fault orders fixed by bearing geometry and measured shaft speed, '
  'classifies the resulting 15 features with a 48-parameter model, and withholds diagnoses whose confidence lies below a threshold calibrated on training bearings only. Physical knowledge enters through the input representation rather than through the loss, in the broad sense of {c:40}. '
  'The method is evaluated under bearing-wise protocols on the Paderborn dataset and, with all settings fixed in advance, on three further test rigs.')
P('The main contributions of this article are summarized as follows.')
LIST('A geometry-parameterized, physics-informed diagnosis of unseen bearings. Because the 15 features are defined in fault-order coordinates computed from the bearing geometry and the measured speed, a single 48-parameter model trained on artificial damage diagnoses real damage and bearings of other geometries without redesigning the input. '
     'On the artificial-to-real protocol of the dataset authors it reached 93.6%, 18.6 points above the published ensemble (75.0%), and with one training bearing type it reached 73.9% on unseen HUST bearing types, against at most 55.4% for generic features.')
LIST('A training-only abstention rule for new bearings. The confidence threshold of each test bearing is calibrated by holding out every training bearing in turn, so that no data of the test bearing are needed; on unseen bearings the achieved coverage followed the target, and accuracy rose from 80.1% to 94.8% at 69% coverage.')
LIST('An empirical detectability criterion that explains when a defect on a new bearing can be diagnosed. Accuracy followed the contrast of the defect line against a healthy bearing (Spearman ρ = 0.70 over 16 test bearings of two rigs), and every bearing with a contrast above 1.1 was diagnosed without error, whereas defect size and bearing size did not predict accuracy. '
     'Together with a damage-wise analysis, this identifies the damage that is hardest to detect: small real indentations, repeated and distributed damage, and operation at low speed.')
LIST('Evidence that bearing identity, rather than the distribution gap between artificial and real damage, limits transfer to new bearings. Representations that identified the individual bearing almost perfectly (97.7%–98.2%) diagnosed unseen bearings worst (50.1%–55.2%), and controls that corrupt the fault orders, '
     'an ablation and a re-examination of the published 98.3% real-damage result show what bearing-wise evaluation can and cannot support.')
P('The remainder of this article is organized as follows. Section II describes ReliSense. Section III presents the dataset, the evaluation protocols and the compared methods. Section IV reports the results on the Paderborn dataset, and Section V the results on three further test rigs. '
  'Section VI analyzes which parts of the signature and which kinds of damage determine the diagnosis. Section VII states the limitations, and Section VIII concludes the article.')

H1('II. ReliSense')
P('Fig. {f:framework} shows the framework. Module I extracts the kinematic signature, Module II classifies it and decides whether to accept the diagnosis, and Module III verifies the role of the kinematic content. '
  'All settings of Modules I and II were fixed in a preceding study on 15 of the bearings (K001–K005, KA01, KA04, KA05, KA07, KA15, KI01, KI04, KI05, KI07 and KI14) and were not changed afterwards; '
  'the other 14 single-damage bearings and the three further datasets played no part in the design.')
FIG('framework', 3, 'Framework of ReliSense. Module I extracts 15 peak-to-background features at the fault orders located with the measured shaft speed. Module II classifies them and withholds recordings whose confidence lies below a threshold calibrated on training bearings only. Module III tests the role of the kinematic content.')
H2('A. Fault Kinematics')
P('For a bearing with n rolling elements of diameter d, pitch diameter D and contact angle φ, the ball-pass frequencies of the outer and inner race (BPFO, BPFI), the ball-spin frequency (BSF) and the fundamental train frequency (FTF) are proportional to the shaft frequency f_r {c:2}:')
EQN(1); EQN(2)
P('A local defect can lie on the outer raceway, the inner raceway or a rolling element (Fig. {f:defects}). An outer-race defect is fixed in the load zone and is struck at BPFO. '
  'An inner-race defect rotates with the shaft, is struck at BPFI and passes through the load zone once per revolution, so that its impacts are modulated at f_r and produce sidebands around BPFI; a rolling-element defect strikes both races at twice the BSF. '
  'For the 6203 bearing of the Paderborn rig (n = 8, d = 6.75 mm, D = 28.55 mm, φ = 0°), the fault orders BPFO/f_r, BPFI/f_r, BSF/f_r and FTF/f_r are 3.054, 4.946, 1.997 and 0.382. '
  'The location of these lines is set by geometry and speed alone, independently of how the defect was produced and of the mounting or sensor of the individual bearing; their strength, by contrast, depends on the damage, the excitation, the noise and the transmission path. '
  'A representation read at these locations is therefore expected to describe the damage more closely than the bearing, although not independently of it.')
FIG('defects', NEWDIR + 'fig_defects.png', 'Local defects of the 6203 bearing. (a) Cross-section to scale with the load zone under the radial force F_r and the three defect locations. (b)–(d) Simulated impact trains at 1500 rpm: an outer-race defect is struck at BPFO with constant strength; an inner-race defect is struck at BPFI and modulated by the load zone at f_r; a ball defect strikes both races at twice the BSF and is modulated at the cage rate (FTF).')
H2('B. Module I: Kinematic Signature Extraction')
P('Each impact excites a resonance of several kilohertz that decays within milliseconds, so the repetition rate appears only after demodulation (Fig. {f:pipeline}). The 64-kHz vibration x(t) is band-pass filtered between 2 and 12 kHz, and the analytic signal is formed as')
EQN(3)
P('where x_b is the filtered signal and ℋ the Hilbert transform. The squared envelope e(t) = |xₐ(t)|² is averaged over blocks of 16 samples, its mean ē is removed, and its spectrum is computed with a Hann window w(t):')
EQN(4)
P('with a resolution of 0.25 Hz for a 4-s recording. Because the fault frequencies scale with speed, the spectrum is read at multiples of the shaft frequency measured in each recording, as in computed order tracking {c:41}. '
  'For the controls of Module III, which use the full spectrum, the squared envelope spectrum is resampled onto an order axis and normalized by its median:')
EQN(5)
P('with 590 bins of 0.02 order. For each fault order k ∈ {BPFO, BPFI, BSF, FTF, shaft} and harmonic h ∈ {1, 2, 3}, the expected frequency is f_kh = h·k·f_r, and the feature is the logarithm of the largest spectral value near f_kh relative to the median of its neighborhood:')
EQN(6)
P('where Δ is 2% of f_kh (at least 1.5 bins). The search band absorbs small speed errors and rolling-element slip, and the ratio within one spectrum makes each feature invariant to sensor gain. Five orders with three harmonics give 15 features per recording.')
FIG('pipeline', 7, 'Feature chain of Module I for one window: band-pass filtering (2–12 kHz), squared envelope from the Hilbert transform, spectral averaging, conversion to the order axis with the measured shaft speed, and peak-to-background ratios at five fault orders and three harmonics.')
H2('C. Module II: Diagnosis and Calibrated Abstention')
P('The features are standardized with training statistics and classified by multinomial logistic regression. For a feature vector z, the posterior probability of class k ∈ {H, OR, IR} is')
EQN(7)
P('with L2 regularization (C = 0.3), which gives 48 trained parameters. A linear model was chosen deliberately: with six or seven bearings per class, every additional degree of freedom is an opportunity to learn bearing identity.')
P('A monitoring system should request another measurement rather than give a confident wrong answer. The confidence of a prediction is c(z) = max_k p(k | z), and the diagnosis is accepted when c(z) ≥ τ_b. '
  'As shown in Fig. {f:calib}, the threshold for test bearing b and target coverage κ is computed from training bearings only:')
EQN(8)
P('where Q_{1−κ} is the empirical quantile, T_b the training bearings when b is tested, ℛ_v the recordings of bearing v, and c_{−v,−b} the confidence of a model trained without v and b. '
  'The inner loop reproduces, within the training data, the situation of meeting an unseen bearing, so that the threshold is calibrated on the kind of confidence that a new bearing produces, and the test bearing never influences its own threshold {c:42,43}.')
FIG('calib', 8, 'Calibration of the abstention threshold. For a held-out bearing b, every training bearing v is held out in turn, a model trained without v and b scores the recordings of v, and the pooled confidences give the (1 − κ) quantile τ_b.')
H2('D. Module III: Verification of the Kinematic Content')
P('Module III tests whether the accuracy depends on the kinematic content rather than on an incidental property of the data. Its controls were written in one script and fixed before execution.')
RUN('1) Fault-Order Perturbation', 'Eleven order features (harmonics 1–2 of the four fault orders and shaft harmonics 1–3) are computed from (5) at the true orders, at orders scaled by s ∈ {0.80, 0.85, 0.90, 0.95, 0.98, 1.02, 1.05, 1.10, 1.15, 1.20}, '
    'and at 20 sets of four random orders between 0.3 and 5.9. If the kinematics carry the diagnosis, the true orders should outrank all 30 wrong sets.')
RUN('2) Representation Ladder', 'Five inputs lead from the raw waveform to the fault orders: a wide-kernel convolutional network (WDCNN) {c:6} on raw vibration and on the envelope, logistic regression and a one-dimensional CNN on the order spectrum (5), and the 15 kinematic features.')
RUN('3) Ablation and Bearing Identity', 'One element of ReliSense is changed at a time (fault family, harmonics, demodulation band, regularization, classifier). '
    'To measure how much bearing identity a representation carries, a 29-class logistic regression is trained to name the bearing of a recording, with five-fold cross-validation stratified by bearing.')
RUN('4) Folding Fields', 'The envelope of one recording is cut into consecutive fault periods that are stacked as rows,')
EQN(9)
P('where T* = 1/f* and f* is the envelope-spectrum peak within ±2% of the theoretical fault frequency. Impacts that repeat at the fault period form a vertical ridge, whereas a period error of 10% makes them drift diagonally, so that the field displays the signature without any classifier.')

H1('III. Experimental Setup')
H2('A. Paderborn Dataset')
P('The Paderborn bearing dataset {c:10,11} was recorded on a modular rig in which a permanent-magnet synchronous motor drives the test-bearing module against a load motor. The housing vibration was sampled at 64 kHz, together with motor currents, shaft speed, torque, radial force and temperature. '
  'Each bearing was run under four operating conditions, N15_M07_F10 (condition 0: 1500 rpm, 0.7 Nm, 1000 N), N09_M07_F10 (900 rpm), N15_M01_F10 (0.1 Nm) and N15_M07_F04 (400 N), with 20 recordings of 4 s each. '
  'The dataset was selected because it is the only public dataset with both artificial and real damage on many bearings of one type, so that transfer from artificial to real damage can be tested by bearing, and because its authors published bearing-wise protocols and results.')
P('Table {t:bearings} lists the bearings. Six are healthy, with run-in times of 1 h to more than 50 h; twelve carry artificial damage made by electrical discharge machining (EDM), drilling or electric engraving; and fourteen carry real damage from accelerated lifetime tests on the same rig. '
  'Labels, damage levels and damage descriptions were taken from the dataset publication {c:10}, whose extent level is set by the damage length in the rolling direction: level 1 up to 2 mm, level 2 from 2 to 4.5 mm and level 3 from 4.5 to 13.5 mm. '
  'KB23, KB24 and KB27, damaged on both rings, were excluded from the three-class task, which leaves 29 bearings and 2559 recordings (one file could not be read). The unit of evaluation is the 4-s recording.')
TAB('bearings', 'Paderborn Bearings Used in This Study', ['Group', 'Bearings', 'Damage method', 'Extent level'],
    [['Healthy (6)', 'K001–K006', 'run-in 1 h to >50 h', '–'],
     ['Artificial, outer race (7)', 'KA01, KA03, KA05, KA06, KA07, KA08, KA09', 'EDM, drilling, engraving', '1–2'],
     ['Artificial, inner race (5)', 'KI01, KI03, KI05, KI07, KI08', 'EDM, engraving', '1–2'],
     ['Real, outer race (5)', 'KA04, KA15, KA16, KA22, KA30', 'pitting, indentation', '1–2'],
     ['Real, inner race (6)', 'KI04, KI14, KI16, KI17, KI18, KI21', 'pitting', '1–3'],
     ['Real, both rings (3)', 'KB23, KB24, KB27', 'pitting, indentation', 'excluded']],
    [2300, 3700, 2300, 1300], 'Damage descriptions and extent levels from {c:10}. Level 1: damage length ≤ 2 mm; level 2: 2–4.5 mm; level 3: 4.5–13.5 mm.')
H2('B. Bearing-Wise Protocols')
P('Every split is defined by bearing identifiers, and the pipeline stops if a bearing appears in both partitions; normalization statistics, thresholds and any model choice come from the training bearings only. Four protocols are used (Fig. {f:protocols}). '
  'L8 is the artificial-to-real protocol of the dataset authors {c:10}: it trains on K002, KA01, KA05, KA07, KI01, KI05 and KI07 and tests on K001 and ten real-damage bearings (KA04, KA15, KA16, KA22, KA30, KI14, KI16, KI17, KI18 and KI21), at condition 0 as in the original study and over all conditions. '
  'L10 is their real-damage protocol: 15 bearings, five per class, are split into three training and two test bearings per class in 10 combinations. '
  'A2R trains on K001–K003 and the 12 artificially damaged bearings and tests on K004–K006 and the 11 real single-damage bearings, which gives more healthy test bearings than L8. LOBO holds out each of the 29 bearings once. '
  'In all protocols the test bearing is new to the model, whereas the rig, the sensor position and the bearing type are shared.')
FIG('protocols', NEWDIR + 'fig_protocols.png', 'Bearing assignment under the four Paderborn protocols. L10 rotates its 15 bearings over 10 combinations; LOBO tests each of the 29 bearings once.')
H2('C. Compared Methods')
P('Table {t:settings} lists the compared methods; all settings were fixed in advance. The kinematic threshold rule uses the first BPFO and BPFI features with the 95th percentile of healthy training recordings and no fitted classifier. '
  'The random forest uses 60 envelope features, which include the kinematic ones, together with kurtosis and crest factor. WDCNN {c:6} processes standardized 4096-sample windows of raw vibration, down-sampled to 8 kHz, or of the 2–12-kHz envelope, and its window probabilities are averaged per recording. '
  'To test whether other processing or classifiers could do better, 72 combinations of nine feature groups, four classifiers and optional per-condition standardization were also declared in advance and, in each L10 split, selected by three inner folds on the training bearings {c:45,46,47}.')
TAB('settings', 'Compared Methods and Settings', ['Method', 'Input', 'Settings'],
    [['ReliSense', '15 kinematic features per recording', 'Standardization, multinomial LR, L2, C = 0.3; 48 parameters; thresholds by inner LOBO'],
     ['Threshold rule', 'First BPFO and BPFI features', '95th percentile of healthy training recordings; no fitted classifier'],
     ['Envelope features + RF', '60 envelope features, kurtosis, crest factor', 'Random forest'],
     ['WDCNN', '4096-sample window, raw or envelope', 'AdamW, lr 10⁻³, batch 128, 20 epochs, class-weighted loss; 85 603 parameters; 3 seeds'],
     ['Order-spectrum CNN', 'Order spectrum (5) up to 11.8 orders', 'Three convolutional layers, 60 epochs, shift augmentation; 8803 parameters'],
     ['72 combinations', 'Nine feature groups', 'LR, RBF-SVM, RF, gradient boosting; nested selection on training bearings']],
    [2100, 3000, 4500])
H2('D. Evaluation Metrics')
P('Recording accuracy is reported with macro F1-score and the macro area under the ROC curve (AUC) where needed. Because recordings of one bearing are not independent, 95% intervals resample test bearings within each class (2000 resamples) {c:44}, '
  'and methods are compared per bearing with the Wilcoxon signed-rank test. A bearing counts as correctly diagnosed when the class with the highest mean probability over its recordings is correct. '
  'The false-alarm rate is the share of healthy recordings diagnosed as damaged, and the missed-fault rate the share of damaged recordings diagnosed as healthy. For abstention, coverage is the share of accepted recordings and selective accuracy the accuracy on them. '
  'p-values are descriptive and not corrected for multiplicity.')

H1('IV. Experimental Results on the Paderborn Dataset')
H2('A. Comparison With Baseline Methods')
P('Table {t:res} compares all methods. On L8, the protocol designed by the dataset authors for transfer from artificial to real damage, ReliSense reached 93.6% at condition 0 and 82.8% over all conditions, against 62.3%–65.9% for the published single classifiers and 75.0% for their ensemble {c:10}. '
  'WDCNN, trained on the same bearings, reached 70.8% with envelope input and 46.4% with raw vibration. Of the 220 L8 test recordings at condition 0, 206 were correct, and 12 of the 14 errors were inner-race recordings diagnosed as outer-race damage (Fig. {f:confusion}(a)). '
  'The bearing-level interval on L8 at condition 0 (82.7%–100%; Table {t:class}) lies above the published ensemble but contains the random forest (84.1%).')
P('The advantage was not uniform. The random forest on 60 envelope features was more accurate on A2R (76.3% against 73.0%) and under LOBO (82.5% against 80.1%), and the threshold rule on L10 (76.6% against 74.6%). WDCNN was the least accurate trained method under every protocol. '
  'Under LOBO, ReliSense diagnosed 26 of the 29 bearings correctly; against WDCNN it was more accurate on 15 bearings and less accurate on 9, with a mean per-bearing difference of 11.9 points (95% interval 1.1–23.5; Wilcoxon p = 0.11). '
  'The kinematic signature therefore transferred best where the transfer is hardest, from artificial to real damage, but did not dominate every protocol.')
TAB('res', 'Recording Accuracy (%) on Unseen Paderborn Bearings', ['Method', 'L8 cond. 0', 'L8 all', 'L10', 'A2R', 'LOBO'],
    [['ReliSense', '93.6', '82.8', '74.6 ± 11.8', '73.0', '80.1'],
     ['Threshold rule', '67.3', '66.2', '76.6 ± 8.4', '65.1', '78.3'],
     ['Envelope features + RF', '84.1', '81.7', '72.6 ± 10.9', '76.3', '82.5'],
     ['WDCNN, envelope', '70.8 ± 11.2', '59.9 ± 2.1', '59.6 ± 15.6', '69.0 ± 1.2', '68.1'],
     ['WDCNN, raw vibration', '46.4 ± 6.2', '47.5 ± 2.2', '58.1 ± 13.6', '66.0 ± 1.5', '61.9'],
     ['Published ensemble {c:10}', '75.0', '–', '98.3', '–', '–']],
    [2900, 1400, 1300, 1400, 1300, 1300],
    'L10: mean ± SD over 10 splits (WDCNN: 10 splits × 3 seeds). WDCNN on L8 and A2R: mean ± SD over 3 seeds. LOBO: recordings of all 29 bearings pooled (WDCNN: mean per-bearing accuracy). Published single classifiers on L8: 62.3–65.9.')
FIG('confusion', NEWDIR + 'fig_confusion.png', 'Confusion matrices of ReliSense (recordings and row percentages). (a) L8, condition 0. (b) A2R. (c) LOBO.')
H2('B. Healthy Bearings and False Alarms')
P('The healthy class is the weakest part of the diagnosis outside L8 (Table {t:class}). Under A2R, 73.3% of the healthy test recordings would have raised a fault alarm, and under LOBO 41.0%, whereas only 9.9% of the damaged recordings were diagnosed as healthy. '
  'Most of these false alarms came from two bearings, K004 and K005, which were correct on 15.0% and 7.5% of their recordings under LOBO and were mostly diagnosed as outer-race damage; their median BPFO feature (0.83 and 0.84) exceeded that of K001–K003 (0.56–0.69). '
  'An acceptable overall accuracy can thus coexist with too many maintenance alarms, which is why false-alarm rates are reported throughout.')
TAB('class', 'ReliSense: Bearing-Level Interval and Class-Wise Rates (%)',
    ['Protocol', 'Accuracy (95% CI)', 'Bearings correct', 'Healthy recall', 'False alarms', 'OR recall', 'IR recall', 'Missed faults'],
    [['L8, cond. 0', '93.6 (82.7–100)', '10/11', '100.0', '0.0', '98.0', '88.0', '0.5'],
     ['L8, all', '82.8 (74.0–91.9)', '11/11', '–', '–', '–', '–', '–'],
     ['A2R', '73.0 (61.3–84.9)', '12/14', '26.7', '73.3', '81.5', '89.2', '4.3'],
     ['LOBO', '80.1 (71.8–88.4)', '26/29', '59.0', '41.0', '85.9', '85.3', '9.9']],
    [1300, 1700, 1100, 1100, 1100, 1000, 1000, 1100],
    'CI: 95% bootstrap interval over test bearings. Misclassified bearings: KI17 (L8, cond. 0); K004, K005 (A2R); K004, K005, KI17 (LOBO). L8 has one healthy test bearing.')
H2('C. Real-Damage Protocol and the Published Result')
P('On L10, ReliSense reached 74.6% over all conditions and 73.5% at condition 0, far below the 98.3% that the dataset publication lists for its best vibration classifiers {c:10}. To test whether other processing could close this gap, the 72 pre-declared combinations were evaluated (Table {t:l10}). '
  'The best combination, chosen after seeing the test results and therefore an optimistic bound, reached 74.8% and 76.2%. Chosen on the training bearings only, as in deployment, accuracy fell to 55.3% and 69.3%: inner-fold accuracies of up to 96.1% were followed by test accuracies as low as 30.0%. '
  'No method tested here exceeded the 76.6% of the threshold rule.')
P('The published evaluation uses the same unit as this study: its confusion matrix contains 400 test recordings per class, that is, 10 combinations × 2 test bearings × 20 recordings {c:10}. Three aspects of that evaluation may contribute to the gap. '
  'Its 15 vibration features were selected by their separation of the classes, with a threshold chosen by classification accuracy, and the publication does not state that this selection used training bearings only. '
  'The 98.3% is the best of eight algorithms, whose accuracies range from 44.2% to 98.3% (median 79.6%). And all 16 accuracies in its table for this protocol are multiples of 1/120, the size of the test set of one combination (2 bearings × 3 classes × 20 recordings), '
  'whereas means over 10 different combinations would not generally fall on this grid. These observations do not establish the cause, but they show that the 98.3% could not be reproduced under a protocol in which every choice is made on training bearings.')
TAB('l10', 'Model Selection on the Real-Damage Protocol (Recording Accuracy, %)', ['Configuration', 'L10 cond. 0', 'L10 all', 'L8 cond. 0', 'L8 all'],
    [['ReliSense (fixed in advance)', '73.5', '74.6', '93.6', '82.8'],
     ['Nested selection among 72 combinations', '55.3 ± 14.8', '69.3 ± 11.5', '–', '–'],
     ['Best of 72 after testing (bound, not valid)', '74.8', '76.2', '93.2', '86.0'],
     ['Published best classifiers {c:10}', '98.3', '–', '75.0', '–']],
    [3800, 1500, 1500, 1400, 1400], 'Nested selection is not possible on L8, which has one healthy training bearing. The best-of-72 row uses test results.')
H2('D. Selective Classification With Training-Only Thresholds')
P('Because weak excitation lowers the peaks, it also lowers the confidence: under LOBO, the median confidence was 0.98 for correct and 0.66 for wrong recordings (Fig. {f:abstain}(a)). '
  'With thresholds from training bearings only, target coverages of 90%, 80% and 70% accepted 89.9%, 78.7% and 68.8% of the recordings, with selective accuracies of 84.5%, 89.5% and 94.8% against 80.1% without abstention (Fig. {f:abstain}(b)). '
  'The achieved coverage thus followed the target closely, which shows that the inner calibration carried over to unseen bearings. Acceptance was lowest for the weakest damage, KA30 (19%), KA15 and KA22 (20%) and KI17 (33%), and 100% for KA01, KA04, KA16, KI01, KI04 and KI18 (Fig. {f:abstain}(c)). '
  'The rule failed on two bearings, however: the accepted recordings of K004 and K005 were all wrong. Abstention thus removed errors on weak damage but not confident errors on healthy bearings that resemble damage.')
FIG('abstain', 22, 'Selective classification under LOBO. (a) Confidence of correct and incorrect recordings. (b) Accuracy against coverage; the line ranks all test recordings (descriptive), markers use training-only thresholds. (c) Acceptance at 70% target coverage against the accuracy of each bearing.')

H1('V. Generalization to Other Test Rigs')
P('To examine whether the kinematic signature transfers beyond one rig and one bearing type, the framework was applied without change to three public datasets: CWRU {c:51}, HUST {c:52} and the time-varying-speed Ottawa data {c:53}. '
  'Segmentation, features, classifier and comparators were written in one script and fixed before these data were analyzed.')
H2('A. Datasets and Protocols')
P('Table {t:ext} summarizes the datasets. In CWRU, single-point defects of 0.007, 0.014 and 0.021 in. were machined into the inner race, the outer race and a ball of the drive-end bearing; each combination of defect and diameter is one physical bearing, and all healthy recordings come from one bearing. '
  'HUST contains healthy and damaged bearings of five types (6204–6208) with pitch diameters from 33.5 to 60 mm; each combination of defect and type is one bearing, and combined defects were excluded. '
  'Ottawa contains one ER16K bearing per health condition under four speed profiles between 9.8 and 29.8 Hz, with the shaft angle taken from an encoder. '
  'Every recording was cut into segments of 50 revolutions, the squared envelope was resampled to the shaft angle {c:41}, and the 15 features were read at the fault orders of each bearing, with the ball and pitch diameters of HUST from {c:52}.')
P('Each dataset was split as its structure allows. On CWRU, one defect diameter was held out at a time (LOSO), so that the faulty test bearings are new; the single healthy bearing was split by load. On HUST, one bearing type was held out (LOTO), so that every test bearing is new and of an unseen geometry. '
  'On Ottawa, one speed profile was held out (LOPO), which tests only an unseen speed profile of the same bearings. Each protocol was also run in reverse, with one diameter, type or profile for training (1SIZE, 1TYPE, 1PROF), which leaves one training bearing per class on CWRU and HUST. '
  'The comparators were 32 band energies with the same logistic regression, and time and envelope statistics with the band energies in a random forest. The primary task has three classes, and a four-class task adds ball defects.')
TAB('ext', 'Three Further Datasets', ['Dataset', 'Bearing, sampling, speed', 'BPFO / BPFI orders', 'Segments', 'Bearings', 'Held-out partition'],
    [['CWRU {c:51}', 'SKF 6205; 12 kHz; 1721–1797 rpm', '3.585 / 5.415', '213', '1 healthy, 9 faulty', 'Defect diameter'],
     ['HUST {c:52}', '6204–6208; 51.2 kHz; 22.6–24.9 Hz', '3.093–3.620 / 4.907–5.425', '221', '19', 'Bearing type'],
     ['Ottawa {c:53}', 'ER16K; 200 kHz; 9.8–29.8 Hz', '3.572 / 5.428', '161', '4', 'Speed profile']],
    [1200, 2600, 1900, 1000, 1400, 1500], 'Segments of 50 shaft revolutions; combined defects excluded. Fault orders from (1) with φ = 0°.')
H2('B. Results and Analysis')
P('Table {t:extres} gives the three-class results. On CWRU, ReliSense was the most accurate method under both protocols (79.5% and 67.1%, against at most 73.3% and 60.3%). Under LOSO it diagnosed five of the six unseen faulty bearings without error; '
  'the exception, the 0.014-in. outer-race defect, was missed in every segment, as it was by the random forest. On HUST, where every test bearing is new and of an unseen geometry, ReliSense reached 77.2% against 71.1% and 54.4%, and 73.9% against 55.4% and 41.7% with a single training type. '
  'On Ottawa the order was reversed: the generic features classified every segment correctly, whereas ReliSense reached 94.2% and 89.8%. Because each Ottawa class is one bearing, a held-out speed profile still comes from a bearing seen in training, and features that identify the bearing suffice; such a dataset cannot measure transfer to new bearings.')
TAB('extres', 'Three-Class Segment Accuracy on the Further Datasets (%)', ['Protocol', 'ReliSense', 'Time + band, RF', 'Band energies, LR', 'ReliSense false alarms'],
    [['CWRU LOSO', '79.5', '73.3', '63.4', '36.7'], ['CWRU 1SIZE', '67.1', '47.3', '60.3', '20.0'],
     ['HUST LOTO', '77.2', '71.1', '54.4', '21.7'], ['HUST 1TYPE', '73.9', '55.4', '41.7', '20.8'],
     ['Ottawa LOPO', '94.2', '100.0', '100.0', '2.3'], ['Ottawa 1PROF', '89.8', '100.0', '100.0', '5.4']],
    [2000, 1500, 1900, 1900, 2300], 'Segments pooled over folds. False alarms: healthy segments diagnosed as faulty. On CWRU the healthy bearing also supplies training data; on Ottawa every test bearing does.')
P('The per-bearing results on HUST (Fig. {f:hust}) locate the errors. Under LOTO, 11 of the 15 bearings were diagnosed with at least 90% accuracy, and the errors of ReliSense concentrated on the inner-race 6204 (0%), the healthy 6205 (0%) and the outer-race 6208 (8%); '
  'the random forest failed completely on the inner-race 6204 and 6207 and reached 8% and 17% on the outer-race 6205 and 6206. The advantage grew when training data were scarce. With one training type, ReliSense lost 3.3 points relative to LOTO, whereas the random forest lost 15.7; '
  'in the four-class task it reached 71.7% against 42.1% and 38.3%, and it was more accurate on 15 of 19 bearings than the random forest (p = 0.010) and on 16 of 19 than band energies (p = 0.002). '
  'The same pattern appeared on Paderborn: with one training bearing per class, drawn 30 times at random, the kinematic features reached 65.2% accuracy and 87.0% macro AUC, against at most 45.1% and 62.1% for the generic inputs, and with five bearings 80.2% and 95.5% against at most 57.0% and 77.1%.')
FIG('hust', NEWDIR + 'fig_hust.png', 'Accuracy on each HUST test bearing (three-class task). (a) One bearing type held out (LOTO). (b) One bearing type for training (1TYPE); mean over the four folds in which the bearing is tested. N: healthy; O: outer race; I: inner race.')
P('Two weaknesses recurred. ReliSense diagnosed 20.0%–36.7% of the healthy CWRU and HUST segments as faulty. The generic methods raised no false alarm on CWRU, but there the only healthy bearing also supplied training data; on HUST, with new healthy bearings, band energies raised 26.7% and 45.8% false alarms and the random forest 0.0% and 30.4%. '
  'Ball defects were the weakest class of ReliSense, with a ball recall of 42.2% on CWRU under LOSO and 60.0% on Ottawa; ball defects in the CWRU data are known to give weak and inconsistent signatures {c:51}.')

H1('VI. Analysis and Discussion')
P('This section examines why the diagnosis succeeds and when it fails: whether the accuracy comes from the kinematic content, which elements of the signature carry it, and which damage, operating conditions and bearings decide the result.')
H2('A. Interpretability of the Kinematic Signature')
RUN('1) The Signature in the Data', 'The signature is visible before any classifier is trained. The signature map of all 32 bearings (Fig. {f:sigmap}) shows lines exactly at the BPFO harmonics for KA01, KA04 and KA16 (3.05, 6.11 and 9.16 orders) and at BPFI with shaft-rate sidebands for KI01 and KI18 (4.95 and 9.89 orders). '
    'The bearings without a distinct line, mostly real level-1 damage (KA15, KA22, KA30, KI14 and KI17), are those misclassified later. In the folding fields (Fig. {f:folding}), the measured fault rates were 76.695 Hz for KA16 and 123.274 Hz for KI18, within 0.5% of the theoretical 76.328 and 123.622 Hz. '
    'At the measured period, the KA16 impacts form a vertical ridge over 120 consecutive periods, and a 10% period error turns them into diagonals; only about every fifth KI18 impact is strong, because the inner-race defect is struck hard only inside the load zone. The healthy K001 shows no such structure.')
FIG('sigmap', 12, 'Signature map: median log squared envelope spectrum of every bearing over shaft order. Dashed: BPFO harmonics; dotted: BPFI harmonics.')
FIG('folding', 13, 'Folding fields: 120 consecutive fault periods (rows) of one condition-0 envelope against phase. Top: measured fault period; bottom: period offset by 10%. Left to right: K001 and KA16 at BPFO, KI18 at BPFI.')
RUN('2) Corrupting the Physics', 'With the true 6203 orders, the order features ranked first among 31 order sets under both L8 and LOBO (Fig. {f:e1}), an outcome with probability 1/21 under exchangeability with the 20 random sets alone. '
    'On L8 at condition 0 the margin was large: 91.8% against at most 82.7% for the scaled sets and 85.5% for the random sets, and a 2% shift reduced accuracy to 58.2% (scale 0.98) and 67.7% (1.02), because it moves the true peaks to the edge of the search band. '
    'Under LOBO the margin was small (78.4% against 78.1% for the best random set). These near-ties have a kinematic explanation: the second harmonics of the scaled BPFO at 0.80 and of the random order 2.478 fall inside the search band of the true BPFI.')
FIG('e1', NEWDIR + 'fig_e1.png', 'Fault-order perturbation. Recording accuracy with the true 6203 fault orders (filled circle), with the four orders scaled by a common factor (open circles), and with 20 sets of random orders (diamonds; line: mean). (a) L8, condition 0. (b) LOBO.')
RUN('3) From the Waveform to the Fault Orders', 'Along the representation ladder (Table {t:repr}), LOBO accuracy rose at every step, from 61.9% for WDCNN on raw vibration to 68.1% on the envelope, 73.9% and 77.3% for logistic regression and a CNN on the order spectrum, and 80.1% for the 15 kinematic features. '
    'The ladder does not isolate the representation, because the classifier, the bandwidth and the observation length change with the input. The bearing-identity test explains why the kinematic features transfer. '
    'Band energies and time and band statistics identified the bearing of a recording in 97.7% and 98.2% of cases (chance 3.4%) but reached only 50.1% and 55.2% damage accuracy on unseen bearings, whereas the kinematic features identified 70.6% and reached 80.1%. '
    'The two representations that encode the individual bearing almost perfectly transferred worst. A distribution-gap measure does not capture this: the maximum mean discrepancy (MMD) {c:64} between artificial and real recordings of the same class, divided by that between classes, '
    'was 0.54 for the kinematic features and 0.42 for band energies, which is consistent with band energies separating the classes through their individual bearings.')
TAB('repr', 'Representation, Damage Accuracy and Bearing Identity', ['Representation', 'Classifier', 'LOBO accuracy (%)', 'Bearing identity (%)', 'MMD ratio, A2R'],
    [['Raw vibration (4096-sample window)', 'WDCNN', '61.9', '–', '–'], ['Envelope (4096-sample window)', 'WDCNN', '68.1', '–', '–'],
     ['32 band energies', 'LR', '50.1', '97.7', '0.42'], ['Time + band statistics', 'LR', '55.2', '98.2', '0.53'],
     ['Order spectrum (590 bins)', 'LR', '73.9', '82.3', '0.79'], ['Order spectrum (590 bins)', 'CNN', '77.3', '–', '–'],
     ['15 kinematic features', 'LR', '80.1', '70.6', '0.54']],
    [3300, 1200, 1700, 1800, 1600],
    'Bearing identity: 29-class LR, five-fold cross-validation over recordings stratified by bearing; chance 3.4%. MMD ratio: within-class MMD between artificial and real recordings divided by between-class MMD (lower = smaller relative gap).')
H2('B. Ablation Study')
P('Table {t:abl} changes one element at a time. The three defect families carried the diagnosis: removing BPFI or BSF cost 14.5 and 15.4 points on L8 at condition 0, and removing BPFO cost 9.1 points and raised the healthy false alarms under LOBO from 41.0% to 52.7%. '
  'BSF features helped although Paderborn has no ball defects; their lines may describe the spectral background near the race orders, but this was not established. The FTF features did not help: without them, accuracy rose slightly under every protocol, so that a 12-feature model is the simpler choice for future work, while the model fixed before testing is kept here. '
  'The second harmonic restored most of the accuracy lost with the first harmonic alone (92.3% against 74.5% on L8 at condition 0). The demodulation band mattered most: the band selected by the kurtogram {c:18} lowered accuracy to 49.5% on L8 and 51.4% under LOBO, '
  'because it chose a narrow band centered at 11.25 kHz for healthy and damaged bearings alike, and cepstral pre-whitening {c:21} lowered it to 62.3% and 75.7%. '
  'The classifier mattered little: an RBF-SVM and a random forest were slightly less accurate, and C between 0.03 and 10 changed LOBO accuracy by less than 0.3 points.')
TAB('abl', 'Ablation of ReliSense: Recording Accuracy (%)', ['Variant', 'L8 cond. 0', 'L8 all', 'L10', 'A2R', 'LOBO'],
    [['Full model', '93.6', '82.8', '74.6', '73.0', '80.1'], ['Without BPFO', '84.5', '82.6', '73.4', '72.7', '75.6'],
     ['Without BPFI', '79.1', '75.2', '66.8', '69.4', '75.8'], ['Without BSF', '78.2', '75.0', '71.2', '68.6', '77.9'],
     ['Without FTF', '95.0', '84.5', '75.0', '76.2', '80.9'], ['Without shaft orders', '90.9', '83.9', '76.4', '72.5', '80.3'],
     ['First harmonic only', '74.5', '73.4', '76.4', '71.2', '78.1'], ['Kurtogram band', '49.5', '58.6', '56.2', '53.6', '51.4'],
     ['Cepstral pre-whitening', '62.3', '71.2', '72.7', '70.1', '75.7'], ['RBF-SVM instead of LR', '92.7', '80.3', '68.0', '70.5', '76.5'],
     ['Random forest instead of LR', '89.5', '77.0', '73.5', '71.8', '76.6']],
    [3300, 1300, 1200, 1200, 1200, 1200],
    'Full model: 15 features, 2–12 kHz band, LR with C = 0.3. Harmonics 1–2: 92.3, 83.2, 75.7, 72.1, 79.0. C from 0.03 to 10: LOBO 80.1–80.3.')
H2('C. Damage-Wise Analysis')
P('Fig. {f:matrix} shows the LOBO accuracy of every bearing for ReliSense and five other models together with the damage attributes of the dataset publication, and Table {t:damage} averages it by attribute. Because the groups contain one to nine bearings, the tests below are descriptive.')
RUN('1) Damage Size and Origin', 'Within real damage, accuracy increased with the extent level (Spearman ρ = 0.55, p = 0.083, n = 11), and level-2 and level-3 damage was diagnosed at 97.5%–100%. '
    'Across all damaged bearings, however, size did not predict accuracy (ρ = 0.19, p = 0.38), because artificial level-1 damage was diagnosed better than real level-1 damage (median 100% against 71.2%; Mann–Whitney p = 0.041). '
    'The smallest defects of the dataset, the 0.25-mm EDM trenches of KA01 and KI01, were diagnosed without error by all six models. A sharp-edged artificial defect produces short impacts that are easy to demodulate, whereas small real damage may be shallower or less regular.')
RUN('2) Damage Type and Arrangement', 'Fatigue pitting was diagnosed more often than plastic indentation (median 97.5% against 53.8%; p = 0.12, two indented bearings), and repeated damage (61.7%) and the one distributed damage (KA30, 46.2%) less often than single damage (84.4%). '
    'Repeated or distributed damage weakens the periodicity at one fault order, and an indentation produces a smoother contact than a pit. One bearing, KI17 (real inner-race pitting, level 1, repeated), was below 60% for all six models, '
    'and five (KA01, KA04, KA16, KI01 and KI18) were at least 95% for all of them: the hardest and the easiest damage were the same for every model. '
    'The feature models erred on the same bearings (Spearman correlation of per-bearing accuracy 0.89–0.92 between ReliSense and the 60-feature models), whereas WDCNN erred on different ones (0.49): '
    'it failed on artificial bearings with clear lines (KA09, KI05 and KI07) but was more accurate on real outer-race damage (87.0% against 73.8%) and on indentations (83.1% against 53.8%).')
RUN('3) Operating Condition and Healthy Bearings', 'Speed mattered by damage size. At 900 rpm, accuracy fell from 81.2% to 53.8% for real level-1 damage and from 70.8% to 36.7% for healthy bearings, whereas real level-2 and level-3 damage stayed at 98.3%: lower speed weakens the impacts, which removes the line of small damage first. '
    'Trained at 1500 rpm and tested at 900 rpm, the kinematic features remained the most accurate input (62.1% against at most 45.2% for features defined in hertz) but lost 23.6 points, so that order normalization helped without making the features speed-invariant. '
    'The run-in time of the healthy bearings did not explain their accuracy (ρ = 0.03, n = 6).')
FIG('matrix', NEWDIR + 'fig_matrix.png', 'LOBO accuracy of every Paderborn bearing for six models, with the damage attributes of the dataset publication. Size: extent level (≤2 mm, 2–4.5 mm, 4.5–13.5 mm damage length). Arr.: S single, R repeated, M multiple; /d: distributed.')
_cols = ['ReliSense', 'Threshold rule', '60 env. feat. + RF', 'WDCNN (envelope)']
_grp = [('size_origin', 'Origin, extent'), ('method', 'Damage method'), ('arrangement_real', 'Arrangement (real)')]
_rows = []
for key, lab in _grp:
    for i, r in enumerate(T12[key]):
        g = r['group'].replace('artificial', 'Artificial').replace('real,', 'Real,').replace('healthy', 'Healthy').replace('single', 'Single').replace('repeated', 'Repeated').replace('multiple', 'Multiple')
        _rows.append([lab if i == 0 else '', g, str(r['n'])] + [f'{r[c]:.1f}' for c in _cols])
TAB('damage', 'LOBO Accuracy by Damage Attribute: Mean Over the Bearings of Each Group (%)',
    ['Attribute', 'Group', 'Bearings', 'ReliSense', 'Threshold rule', 'Envelope features, RF', 'WDCNN, envelope'], _rows,
    [1700, 2700, 900, 1100, 1100, 1300, 1300], 'Extent levels and damage methods from {c:10}. Repeated and multiple damage occur only in real bearings.')
H2('D. Effect of Defect-Line Strength, Bearing Size and Fault Size')
P('The external datasets show what decides the result when bearing size and fault size vary (Fig. {f:line}). On HUST, accuracy on the five unseen types (pitch diameter 33.5–60 mm) showed no monotonic trend (66.7%, 66.7%, 97.2%, 86.1% and 69.4% for 6204–6208; ρ = 0.56, p = 0.32). '
  'On CWRU, the inner- and outer-race defects of 0.18 and 0.53 mm were all diagnosed, but those of 0.36 mm only at 50.0%, because the 0.014-in. outer-race defect failed. '
  'For each faulty test bearing, the signature contrast was defined as the mean kinematic feature of its fault order (three harmonics) minus the same value on the healthy bearing of the same type. '
  'Accuracy increased with this contrast (ρ = 0.70, p = 0.002, n = 16): all 11 bearings with a contrast above 1.1 were diagnosed without error, and the failures had the weakest lines, the CWRU 0.014-in. outer-race defect (0.39, against 3.50 and 1.82 for the other outer-race sizes) and the HUST outer-race 6208 (0.37). '
  'The HUST inner-race 6204 was an exception, with a moderate contrast (1.03) but no correct segment. ReliSense therefore diagnoses a defect when its kinematic line rises clearly above that of a healthy bearing, whatever the bearing size or the defect size.')
FIG('line', NEWDIR + 'fig_line.png', 'Bearing size, fault size and defect-line strength on the external datasets (three-class task). (a) HUST: accuracy on each unseen bearing type (LOTO). (b) ReliSense accuracy of each faulty test bearing against its signature contrast; filled: HUST LOTO, open: CWRU LOSO; circles: outer race, triangles: inner race. (c) CWRU: accuracy on the inner- and outer-race defects of each unseen diameter (LOSO).')
H2('E. Comparison With Published Bearing-Wise Results')
P('Table {t:pub} compares the results only with published Paderborn studies whose test bearings were absent from training. On L8, the published classifiers reached 62.3%–65.9% and their ensemble 75.0% {c:10}, against 93.6% for ReliSense. '
  'Vieira et al. {c:15} evaluated bearing-wise splits on the real-damage and healthy bearings and reported a macro AUROC of 79.6% for WDCNN with envelope input and 69.8% for a random forest; ReliSense reached a macro AUC of 93.7% under LOBO, but that protocol also contains the artificially damaged bearings, so the comparison is indicative only. '
  'Perminov and Korzun {c:66} and Wheat et al. {c:12} also tested on bearings absent from training. On L10, the published 98.3% exceeds all results of this study, for the reasons discussed in Section IV-C.')
TAB('pub', 'Published Paderborn Results With Test Bearings Absent From Training', ['Study', 'Method', 'Protocol', 'Result (%)'],
    [['Lessmeier et al. {c:10}', 'Vibration features; single classifiers, ensemble', 'Artificial → real (L8)', 'Accuracy 62.3–65.9; 75.0'],
     ['Lessmeier et al. {c:10}', 'Vibration features, best classifiers', 'Real → real (L10)', 'Accuracy 98.3'],
     ['Perminov and Korzun {c:66}', 'Convolutional neural network', 'Test bearings absent from training', ('[to be added]', True)],
     ['Wheat et al. {c:12}', 'PCA, SPCA, LDA on spectral and envelope features', 'Bearing-wise split', ('[to be added]', True)],
     ['Vieira et al. {c:15}', 'WDCNN (envelope); RF; SVM', 'Bearing-wise, real damage and healthy', ('Macro AUROC 79.6; 69.8; 64.4', True)],
     ['This study', 'ReliSense', 'L8 cond. 0 / L8 all / LOBO', 'Accuracy 93.6 / 82.8 / 80.1; LOBO macro AUC 93.7'],
     ['This study', 'Envelope features + RF', 'L8 cond. 0 / L8 all / LOBO', 'Accuracy 84.1 / 81.7 / 82.5'],
     ['This study', 'WDCNN, envelope', 'L8 cond. 0 / L8 all / LOBO', 'Accuracy 70.8 / 59.9 / 68.1']],
    [2100, 2900, 2400, 2400], 'Evaluation units and class sets differ: 4-s recordings and 29 bearings (LOBO) in this study.',
    'Highlighted values to be inserted or checked against the cited papers.')

H2('F. Discussion')
P('The results show that a 48-parameter model on 15 kinematic features transfers to unseen bearings on three rigs and is most clearly ahead where the transfer is hardest: from artificial to real damage, to bearing types of unseen geometry, and with one training bearing per class. '
  'It was not the most accurate method under every protocol; a random forest on a larger envelope feature set was more accurate under LOBO and A2R. Corrupting the fault orders, removing fault families and moving the input from the waveform to the fault orders all point to the kinematic content, '
  'and the strength of the defect line, not the size of the defect or the bearing, decides whether a defect is found. Thresholds calibrated on training bearings achieved the target coverage on new bearings and raised the accuracy of the accepted recordings, but did not withhold the confident errors on two healthy bearings.')
P('A plausible explanation is that a defect on a given race produces impacts at a rate set by geometry and speed, whether it was machined or grew by fatigue, whereas the waveform also carries resonances, rig components and properties of the individual bearing, which a flexible model can use to separate its few training bearings. '
  'The bearing-identity test supports this explanation: the inputs that identified the individual bearing almost perfectly transferred worst. For practice, the results suggest that the diagnosis of a new bearing should rest on evidence whose location is fixed by physics, that results should be reported per bearing and with false-alarm rates, '
  'and that model selection must use training bearings only, since selection with knowledge of the test results raised L10 accuracy by up to 19.5 points over nested selection.')
H1('VII. Limitations and Future Work')
P('The study has limitations. The main analysis uses one rig, one bearing type and four operating conditions, and the real damage grew in accelerated lifetime tests rather than in service. The external datasets add three rigs and six bearing designs, but their damage was seeded, each has few physical bearings, and no model was trained on one machine and tested on another; '
  'the results therefore concern unseen bearings of a known rig, not unseen machines or sensor positions. Fifteen of the 29 bearings informed the design, including four of the 11 L8 test bearings. The compared networks did not receive the bandwidth, observation length and harmonic range of ReliSense, and no physics-informed network was evaluated under the same protocols. '
  'Healthy bearings remain the main weakness, with 41.0% of healthy recordings diagnosed as faulty under LOBO. A natural extension is to accept a diagnosis only when several harmonics of the candidate fault agree and stand out from matched incorrect orders, '
  'and to test the framework across machines, for example on the constant-speed Ottawa collection {c:57}, whose 20 bearings allow bearing-wise splits, together with calibration under distribution shift and conformal prediction {c:58,59,60}.')

H1('VIII. Conclusion')
P('This article addressed the diagnosis of bearings that a model has never seen. ReliSense reads the squared envelope spectrum at the fault orders fixed by bearing geometry and measured speed, classifies 15 peak-to-background features with a 48-parameter model fixed before testing, '
  'and withholds low-confidence diagnoses with thresholds calibrated on training bearings only. On 29 Paderborn bearings, it reached 93.6% when trained on artificial and tested on real damage, against 75.0% for the published ensemble and 70.8% for a convolutional network, '
  'and 80.1% under leave-one-bearing-out evaluation, which abstention raised to 94.8% at 69% coverage. With settings fixed in advance, it was the most accurate of three methods on unseen bearing types of HUST and unseen fault sizes of CWRU, with the largest margins when few training bearings were available. '
  'Controls that corrupt the fault orders, an ablation and a bearing-identity test indicated that the accuracy comes from the kinematic content, and a damage-wise analysis showed that the strength of the defect line decides whether a defect is found. '
  'Small real indentations, repeated and distributed damage, operation at low speed, and healthy bearings that resemble damage remain the main limits of the method.')

# ================================================================== resolve keys
fig_no, tab_no = {}, {}
for b in BLOCKS:
    if b[0] == 'FIG': fig_no[b[1]] = len(fig_no) + 1
    if b[0] == 'TAB': tab_no[b[1]] = len(tab_no) + 1
ROMAN = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV']
ref_order = []


def cite_list(nums):
    new = sorted(nums)
    out, i = [], 0
    while i < len(new):
        j = i
        while j + 1 < len(new) and new[j + 1] == new[j] + 1: j += 1
        out.append(f'[{new[i]}]–[{new[j]}]' if j - i >= 2 else ', '.join(f'[{x}]' for x in new[i:j + 1])); i = j + 1
    return ', '.join(out)


def resolve(s):
    def c(m):
        olds = [int(x) for x in m.group(1).split(',')]
        for o in olds:
            if o not in ref_order: ref_order.append(o)
        return cite_list([ref_order.index(o) + 1 for o in olds])
    s = re.sub(r'\{c:([\d,]+)\}', c, s)
    s = re.sub(r'\{f:(\w+)\}', lambda m: str(fig_no[m.group(1)]), s)
    s = re.sub(r'\{t:(\w+)\}', lambda m: ROMAN[tab_no[m.group(1)]], s)
    assert '{' not in s.replace('{BPFO', '').replace('{H, OR', '').replace('{0.80', '').replace('{1, 2, 3}', '') or True
    return s


def picture(src):
    if isinstance(src, str):
        tmp = doc.add_paragraph(); tmp.add_run().add_picture(src, width=Inches(6.2)); e = tmp._p; e.getparent().remove(e)
        old = e.find(q('pPr'))
        if old is not None: e.remove(old)
        e.insert(0, copy.deepcopy(T_FIGP_PPR)); return e
    return copy.deepcopy(OLDFIG[src])


words = 0
for b in BLOCKS:
    k = b[0]
    if k == 'H1': put(set_runs(copy.deepcopy(T_H1), [(b[1].split(' ', 1)[0] + ' ', False), (b[1].split(' ', 1)[1], False)]))
    elif k == 'H2': put(set_runs(copy.deepcopy(T_H2), [(b[1], False)]))
    elif k == 'P': s = resolve(b[1]); words += len(s.split()); put(set_runs(copy.deepcopy(T_P), [(s, False)]))
    elif k == 'LIST': s = resolve(b[1]); words += len(s.split()); put(set_runs(copy.deepcopy(T_LIST), [(s, False)]))
    elif k == 'RUN': s = resolve(b[2]); words += len(s.split()) + len(b[1].split()); put(set_runs(copy.deepcopy(T_RUN), [(b[1] + ': ', False), (s, False)]))
    elif k == 'EQ': put(copy.deepcopy(EQ[b[1]]))
    elif k == 'FIG':
        put(picture(b[2])); cap = resolve(b[3]); words += len(cap.split())
        put(set_runs(copy.deepcopy(T_CAP), [(f'Fig. {fig_no[b[1]]}. ', False), (cap, False)]))
    elif k == 'TAB':
        _, key, title, header, rows, widths, note, note_hl = b
        put(set_runs(copy.deepcopy(T_TN), [(f'TABLE {ROMAN[tab_no[key]]}', False)])); put(set_runs(copy.deepcopy(T_TT), [(title, False)]))
        tbl = copy.deepcopy(T_TBL); trs = tbl.findall(q('tr')); hdr, mid, last = trs[0], trs[1], trs[-1]
        for tr in trs: tbl.remove(tr)
        grid = tbl.find(q('tblGrid'))
        for g in list(grid): grid.remove(g)
        for w in widths: g = etree.SubElement(grid, q('gridCol')); g.set(q('w'), str(w))
        tbl.find(q('tblPr')).find(q('tblW')).set(q('w'), str(sum(widths)))

        def row(tmpl, vals):
            global words
            tr = copy.deepcopy(tmpl); cells = tr.findall(q('tc')); c0 = cells[0]
            for c_ in cells: tr.remove(c_)
            for v, w in zip(vals, widths):
                c_ = copy.deepcopy(c0); c_.find(q('tcPr')).find(q('tcW')).set(q('w'), str(w))
                hl = isinstance(v, tuple); t = resolve(v[0] if hl else str(v)); words += len(t.split())
                set_runs(c_.find(q('p')), [(t, hl)]); tr.append(c_)
            return tr
        tbl.append(row(hdr, header))
        for i, r in enumerate(rows): tbl.append(row(last if i == len(rows) - 1 else mid, r))
        put(tbl)
        n_ = resolve(note or ''); words += len(n_.split())
        put(set_runs(copy.deepcopy(T_NOTE), [(n_, False)] + ([(' ' + note_hl, True)] if note_hl else [])))

# ------------------------------------------------------------------ references: order of first citation, uncited removed
for e in REFP.values():
    if e.getparent() is not None: e.getparent().remove(e)
prev = el_starting('References')
for new_i, old in enumerate(ref_order, 1):
    e = copy.deepcopy(REFP[old]); ts = list(e.iter(q('t')))
    ts[0].text = re.sub(r'^\[\d+\]', f'[{new_i}]', ts[0].text)
    prev.addnext(e); prev = e
doc.save(OUT)
print('saved', OUT, '| words (text, captions, tables, without abstract and references):', words,
      '| figures', len(fig_no), '| tables', len(tab_no), '| references', len(ref_order), '| dropped refs', sorted(set(REFP) - set(ref_order)))
