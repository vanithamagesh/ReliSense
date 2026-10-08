# Section II of manuscript v3.15: the three modules with their full mathematical description.
# Every equation states what the code of phases 1, 2, 11 and 12 computes (physics64.py, phase2_physics.py,
# phase11_analysis.py, phase12_damage_analysis.py); nothing here is a new experiment.
H1('II. ReliSense')
P('Fig. {f:framework} shows the framework. Module I converts one 4-s vibration recording into 15 kinematic features, Module II classifies them and decides whether the diagnosis is accepted, and Module III tests whether the accuracy depends on the kinematic content. '
  'All settings of Modules I and II were fixed in a preceding study on 15 of the bearings (K001–K005, KA01, KA04, KA05, KA07, KA15, KI01, KI04, KI05, KI07 and KI14) and were not changed afterwards; '
  'the other 14 single-damage bearings and the three further datasets played no part in the design.')
FIG('framework', 3, 'Framework of ReliSense. Module I extracts 15 peak-to-background features at the fault orders located with the measured shaft speed. Module II classifies them and withholds recordings whose confidence lies below a threshold calibrated on training bearings only. Module III tests the role of the kinematic content.')

H2('A. Fault Kinematics and Signal Model')
P('For a bearing with n rolling elements of diameter d, pitch diameter D and contact angle φ, rotating at the shaft frequency f_r, a local defect is struck at a rate fixed by the geometry {c:2}. '
  'The ball-pass frequencies of the outer and inner race (BPFO, BPFI), the ball-spin frequency (BSF) and the fundamental train frequency (FTF) are')
EQT('orders1', 'BPFO = (n/2)·f_r·(1 − (d/D)cosφ),   BPFI = (n/2)·f_r·(1 + (d/D)cosφ)')
EQT('orders2', 'BSF = (D/2d)·f_r·(1 − ((d/D)cosφ)²),   FTF = (1/2)·f_r·(1 − (d/D)cosφ).')
P('Dividing by f_r gives the fault orders k = f/f_r, which depend on the geometry only. For the 6203 bearing of the Paderborn rig (n = 8, d = 6.75 mm, D = 28.55 mm, φ = 0°), they are k_BPFO = 3.054, k_BPFI = 4.946, k_BSF = 1.997 and k_FTF = 0.382. '
  'A local defect can lie on the outer raceway, the inner raceway or a rolling element (Fig. {f:defects}). Each passage of a rolling element over the defect produces a short impact that excites a structural resonance, so that the vibration measured on the housing can be modelled as {c:2}, {c:17}')
EQT('model', 'x(t) = Σ_i A_i·h(t − iT_d − τ_i) + ν(t),   T_d = 1/f_d,')
P('where f_d is the fault frequency (BPFO, BPFI or 2·BSF), h(t) the impulse response of the resonance, τ_i a small random delay caused by rolling-element slip (typically 1%–2% of T_d), and ν(t) the remaining vibration and noise. '
  'For an outer-race defect, which is fixed in the load zone, the amplitudes A_i are nearly constant. An inner-race defect rotates with the shaft and passes through the load zone once per revolution, so that its amplitude follows the load distribution,')
EQT('load', 'A_i = A_0·q(θ_i),   q(θ) = [1 − (1 − cosθ)/(2ε)]^{3/2} for |θ| ≤ θ_max, and 0 otherwise,')
P('where θ_i = 2πf_r·iT_d is the angular position of the defect relative to the load direction and ε the load-distribution factor; the modulation at f_r produces sidebands at BPFI ± m·f_r. A ball defect strikes both raceways once per spin and is modulated at the cage rate FTF. '
  'Two properties of (3) underlie the method. The location of the impact rate f_d is set by geometry and speed alone, independently of how the defect was produced and of the mounting or sensor of the individual bearing. '
  'The strength of the impacts, by contrast, depends on the damage, the excitation, the transmission path h(t) and the noise ν(t). A representation read at the fault orders is therefore expected to describe the damage more closely than the bearing, although not independently of it.')
FIG('defects', NEWDIR + 'fig_defects.png', 'Local defects of the 6203 bearing. (a) Cross-section to scale with the load zone under the radial force F_r and the three defect locations. (b)–(d) Simulated impact trains of model (3) at 1500 rpm: an outer-race defect is struck at BPFO with constant strength; an inner-race defect is struck at BPFI and modulated by the load zone at f_r; a ball defect strikes both races at twice the BSF and is modulated at the cage rate (FTF).')

H2('B. Module I: Kinematic Signature Extraction')
P('Module I maps one recording x(t), sampled at f_s = 64 kHz for 4 s (N = 256 000 samples), to a feature vector z ∈ ℝ^15 in five steps (Fig. {f:pipeline}).')
RUN('1) Band Selection and Demodulation', 'The impacts of (3) are visible as a repetition rate only after demodulation, because each impact excites a resonance of several kilohertz that decays within milliseconds. '
    'With X[m] the discrete Fourier transform of the mean-free recording and f_m = m·f_s/N, the band [f_lo, f_hi] = [2, 12] kHz is retained and the complex envelope is formed as the analytic signal of the band-limited vibration:')
EQT('analytic', 'x_a[t] = IDFT{ 2·X[m]·𝟙(f_lo ≤ f_m < f_hi) },   e[t] = |x_a[t]|².')
P('The band covers the structural resonances of the housing excited by the impacts and excludes the shaft, gear-mesh and electrical lines below 2 kHz. It is fixed for all bearings; the band chosen by the fast kurtogram {c:18} and cepstral pre-whitening {c:21} were tested as alternatives in Section VI-B.')
RUN('2) Envelope Spectrum', 'The squared envelope e[t] contains no information above a few hundred hertz at the fault rates of interest, so it is averaged over blocks of B = 16 samples, which lowers the rate to f_s/B = 4 kHz, its mean is removed, and its magnitude spectrum is computed with a Hann window w[j]:')
EQT('block', 'ē[j] = (1/B)·Σ_{u=0}^{B−1} e[jB + u],   S(f_q) = | Σ_j w[j]·(ē[j] − ⟨ē⟩)·exp(−j2πf_q·jB/f_s) |.')
P('For a 4-s recording, the frequency resolution is Δf = 0.25 Hz. Under (3), the expected value of e[t] is periodic with period T_d, so S(f) contains discrete lines at the harmonics h·f_d, and for an inner-race defect also at h·f_d ± m·f_r {c:2}.')
RUN('3) Order Normalization', 'Because the fault frequencies scale with speed, the spectrum is read at multiples of the shaft frequency of each recording, as in computed order tracking {c:41}. The shaft frequency is f_r = n_sh/60, where n_sh is the median of the measured speed signal in rpm; '
    'the set point is used only if the median deviates by more than 20% from it, which did not occur. On the order axis r = f/f_r, the line of a fault lies at the same position at every speed. For the verification controls of Module III, which need the full spectrum, S is resampled on the order grid r_l = 0.20, 0.22, …, 12.00 (590 bins) and normalized by its median:')
EQT('order', 'O(r_l) = log[ S(r_l·f_r) / median_l S(r_l·f_r) ].')
RUN('4) Peak-to-Background Features', 'For each fault order k ∈ {BPFO, BPFI, BSF, FTF, shaft} and harmonic h ∈ {1, 2, 3}, the expected frequency is f_kh = h·k·f_r. A search band 𝓑_kh and a background band 𝓖_kh are defined around it,')
EQT('bands', '𝓑_kh = { f : |f − f_kh| ≤ Δ_kh },   Δ_kh = max(0.02·f_kh, 1.5·Δf),   𝓖_kh = { f : |f − f_kh| ≤ 10 Hz } \\ 𝓑_kh,')
P('and the feature is the logarithm of the largest spectral value in the search band relative to the median of the background band:')
EQT('feature', 'φ_kh = log[ max_{f∈𝓑_kh} S(f) / median_{f∈𝓖_kh} S(f) ].')
P('The search band absorbs speed errors and rolling-element slip, which shift the observed line by up to about 2%. Because (8) is a ratio within one spectrum, it is invariant to the sensor gain and to the overall vibration level of the bearing; '
  'a healthy bearing gives φ_kh close to zero, and a defect raises the features of its own fault order. The 15 features form the vector')
EQT('vector', 'z = [φ_BPFO,1, φ_BPFO,2, φ_BPFO,3, φ_BPFI,1, …, φ_shaft,3]ᵀ ∈ ℝ^15.')
RUN('5) Properties', 'The feature vector has three properties that a learned representation does not guarantee. Its coordinates are fixed by the geometry, so that the same feature describes the same physical quantity for every bearing and every speed. '
    'It requires the geometry (n, d, D, φ) and the shaft speed of the monitored bearing, but no data of that bearing. And it is compact, so that the classifier of Module II needs few parameters.')
FIG('pipeline', 7, 'Feature chain of Module I for one window: band-pass filtering (2–12 kHz), squared envelope from the Hilbert transform, spectral averaging, conversion to the order axis with the measured shaft speed, and peak-to-background ratios at five fault orders and three harmonics.')

H2('C. Module II: Diagnosis and Calibrated Abstention')
RUN('1) Classifier', 'The features are standardized with the mean μ and standard deviation σ of the training recordings, z̃ = (z − μ) ⊘ σ, and classified by multinomial logistic regression. For the classes k ∈ 𝒦 = {H, OR, IR}, the posterior probability is')
EQT('softmax', 'p(k | z̃) = exp(w_kᵀz̃ + b_k) / Σ_{j∈𝒦} exp(w_jᵀz̃ + b_j),')
P('and the weights are obtained from the N_tr training recordings by minimizing the regularized cross-entropy')
EQT('objective', 'min_{W,b}  − Σ_{i=1}^{N_tr} log p(y_i | z̃_i) + (1/(2C))·Σ_{k∈𝒦} ‖w_k‖²,   C = 0.3.')
P('The model has 3 × 15 weights and 3 biases, that is, 48 trained parameters. A linear model was chosen deliberately: with six or seven bearings per class, every additional degree of freedom is an opportunity to learn the identity of the training bearings instead of the damage. '
  'Because the features of one class lie along one direction of the feature space (the BPFO features for outer-race damage and the BPFI features for inner-race damage), a linear boundary is also sufficient in principle.')
RUN('2) Confidence and Abstention', 'A monitoring system should request another measurement rather than give a confident wrong answer. The confidence of a prediction and the decision rule are')
EQT('decision', 'c(z̃) = max_{k∈𝒦} p(k | z̃),   ŷ = argmax_{k∈𝒦} p(k | z̃) if c(z̃) ≥ τ_b,  ŷ = “refer” otherwise,')
P('where τ_b is the threshold for test bearing b. The threshold must not use any data of b, and it should reflect the confidence that a model produces on a bearing it has never seen, which is lower than on its training bearings. '
  'It is therefore calibrated by an inner leave-one-bearing-out loop over the training bearings T_b (Fig. {f:calib}): every v ∈ T_b is held out in turn, a model is trained on T_b \\ {v}, and its confidences on the recordings ℛ_v of v are pooled. For a target coverage κ,')
EQT('tau', 'τ_b(κ) = Q_{1−κ} { c_{−v,−b}(z̃) : z̃ ∈ ℛ_v, v ∈ T_b },')
P('where Q_{1−κ} is the empirical (1 − κ) quantile and c_{−v,−b} the confidence of the model trained without v and b. With |T_b| = 28 under LOBO, each threshold requires 28 inner models. '
  'On the recordings ℛ_b of the test bearing, the accepted set 𝒜_b = {z̃ ∈ ℛ_b : c(z̃) ≥ τ_b} defines the achieved coverage and the selective accuracy,')
EQT('coverage', 'cov = Σ_b |𝒜_b| / Σ_b |ℛ_b|,   acc_sel = Σ_b Σ_{z̃∈𝒜_b} 𝟙(ŷ = y) / Σ_b |𝒜_b|.')
P('If the inner loop reproduces the situation of an unseen bearing, the achieved coverage should be close to κ; this is tested in Section IV-D {c:42}, {c:43}.')
FIG('calib', 8, 'Calibration of the abstention threshold. For a held-out bearing b, every training bearing v is held out in turn, a model trained without v and b scores the recordings of v, and the pooled confidences give the (1 − κ) quantile τ_b.')

H2('D. Module III: Verification of the Kinematic Content')
P('Module III tests whether the accuracy depends on the kinematic content rather than on an incidental property of the data. All controls were written in one script and fixed before execution; none is used for the diagnosis itself.')
RUN('1) Fault-Order Perturbation', 'If the diagnosis relies on the kinematics, features read at wrong orders should be less accurate. Eleven order features are computed from (6) at harmonics 1–2 of the four fault orders and harmonics 1–3 of the shaft order, '
    'each as the maximum of O within ±max(0.02r, 0.04) of the order r minus its median within ±0.4 order outside that band. The four fault orders are then replaced by')
EQT('perturb', 'k′ = s·k,  s ∈ {0.80, 0.85, 0.90, 0.95, 0.98, 1.02, 1.05, 1.10, 1.15, 1.20},   or   k′ ~ 𝒰(0.3, 5.9) (20 random sets),')
P('which gives 30 wrong order sets, and the same classifier is trained and tested on each. With a_0 the accuracy of the true orders and a_1, …, a_20 those of the random sets, the rank statistic')
EQT('rank', 'p = [1 + Σ_{j=1}^{20} 𝟙(a_j ≥ a_0)] / 21')
P('is the probability of the observed rank if the true orders were exchangeable with random ones; its smallest value is 1/21 ≈ 0.048.')
RUN('2) Representation Ladder and Ablation', 'Five inputs lead from the raw waveform to the fault orders: a wide-kernel convolutional network (WDCNN) {c:6} on raw vibration and on the envelope, logistic regression and a one-dimensional CNN on the order spectrum (6), and the 15 features (9). '
    'If the kinematic content is what transfers, accuracy should rise along this ladder. In the ablation, one element of ReliSense is changed at a time: one fault family (three features) is removed, the harmonics are limited, the band is replaced, C is varied or the classifier is replaced.')
RUN('3) Bearing Identity and Distribution Gap', 'To measure how much a representation encodes the individual bearing, a 29-class logistic regression is trained to name the bearing of a recording, with five-fold cross-validation over recordings stratified by bearing; '
    'a high identity accuracy together with a low accuracy on unseen bearings indicates that the representation separates bearings rather than damage. The gap between artificial and real damage is measured by the maximum mean discrepancy (MMD) {c:64},')
EQT('mmd', 'MMD²(P, Q) = E[k(u, u′)] + E[k(v, v′)] − 2E[k(u, v)],   k(u, v) = exp(−γ‖u − v‖²),')
P('with u, u′ ~ P and v, v′ ~ Q, and γ set to the inverse median of the squared pairwise distances. For each class, P and Q are the standardized training (artificial) and test (real) recordings of the A2R split, and the mean of these three within-class values is divided by the mean MMD between the classes of the test set.')
RUN('4) Defect-Line Contrast', 'To relate accuracy to the strength of the signature, the contrast of a faulty test bearing b of class c is defined against the healthy bearing h(b) of the same type,')
EQT('contrast', 'Γ_b = (1/3)·Σ_{h=1}^{3} [ φ̄_{k_c,h}(b) − φ̄_{k_c,h}(h(b)) ],')
P('where k_c is the fault order of the class (BPFO for outer-race and BPFI for inner-race damage) and φ̄ the mean of (8) over the recordings of a bearing. Γ_b measures how far the defect line rises above that of a healthy bearing of the same design.')
RUN('5) Folding Fields', 'Without any classifier, the signature can be displayed in the time domain. The envelope e(t) of one recording is cut into consecutive fault periods that are stacked as rows,')
EQT('fold', 'F(i, θ) = e((i + θ)·T*) / median_t e(t),   i = 0, …, 119,   0 ≤ θ < 1,')
P('where T* = 1/f* and f* is the envelope-spectrum peak within ±2% of the theoretical fault frequency, which accounts for slip. Impacts that repeat at the fault period form a vertical ridge, whereas a period error of 10% makes them drift diagonally.')
