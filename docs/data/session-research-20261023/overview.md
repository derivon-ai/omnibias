# Omnibias and multi-bias research: complete session findings

This compendium preserves the research trajectory, including useful constructions, favorable results, unfavorable results, corrections, and unresolved claims. It distinguishes **independently checked laboratory evidence**, **results reported by the existing multi-bias paper**, **written mathematical arguments**, and **the author's original hypotheses**. Those evidence categories must not be merged into one claim of novelty or superiority. The canonical reports and retained artifacts remain authoritative for their protocols and individual rows. [Evidence definitions and variant map](/home/vardan/dev/projects/omnibias/docs/multi-bias-research-synthesis.md).

## 1. What the author proposed

The starting concern was that stronger baselines and prior literature removed the claimed novelty of omnibias. The author asked whether to archive the work, open-source it, close the domains, or identify a real contribution. The subsequent investigation pursued narrower claims instead of treating the initial unfavorable assessment as a proof that every package or future architecture is useless. No licensing, domain, service, or stable API change was authorized as a consequence of a negative benchmark. [Campaign assessment](/home/vardan/dev/projects/omnibias/docs/calculus-geometry-campaign.md), [research synthesis](/home/vardan/dev/projects/omnibias/docs/multi-bias-research-synthesis.md).

The author's central hypotheses were:

- Multiple shifted activations can increase a unit's expressive power without expanding its layer's output dimension; a suitable construction should recover the ordinary unit when the bias count is one. [Architecture distinctions](/home/vardan/dev/projects/omnibias/docs/multi-bias-research-synthesis.md).
- Coalescing biases exposes derivatives and therefore a bridge between neurons, calculus, geometry, differential equations and linear algebra. [Bias-collapse derivation](/home/vardan/dev/projects/omnibias/docs/bias-collapse-pde.md), [deep composition](/home/vardan/dev/projects/omnibias/docs/deep-bias-collapse.md).
- Equality detectors should represent hyperplanes themselves, rather than only half-spaces; intersections might support equation solving, physical models and optimization. [Corrected equality semantics](/home/vardan/dev/projects/omnibias/docs/bias-collapse-pde.md), [geometry pilot](/home/vardan/dev/projects/omnibias/docs/calculus-geometry-pilot.md).
- Independent monotone branches might rescue shape-parameter gradients where a fixed non-monotone activation stalls. [Two-bias bell experiment](/home/vardan/dev/projects/omnibias/docs/two-bias-bell-experiment.md).
- One multi-bias unit can implement XOR and, with sufficient complexity, arbitrary Boolean labelings; activation complexity might substitute for some width and depth. [Capacity review](/home/vardan/dev/projects/omnibias/docs/multi-bias-capacity-and-architecture.md).
- Temperature-controlled decisions, exact stochastic objectives, and direct accuracy/F1 optimization might improve hard deployed predictions. [Metric and hardening analysis](/home/vardan/dev/projects/omnibias/docs/bias-collapse-pde.md), [objective audit](/home/vardan/dev/projects/omnibias/docs/unification-research-audit.md).
- Richer per-filter responses might improve CNN amplitude selection, pretrained compression, adaptation, tabular prediction, geometric modeling, clustering, anomalies, logic and maximum-margin bands. [Research opportunities](/home/vardan/dev/projects/omnibias/docs/calculus-geometry-opportunities.md), [learning and margin study](/home/vardan/dev/projects/omnibias/docs/multi-bias-learning-and-margin.md).
- A useful general theory should decide when to enrich a response, refine a direction, add a direction, change interactions or stop growing. [Projection/profile allocation study](/home/vardan/dev/projects/omnibias/docs/projection-profile-allocation.md).

These were research hypotheses, not established outcomes. The laboratory supplied proofs and bounded experiments to distinguish their valid parts from overextensions. A representation theorem does not imply efficient learning, generalization, numerical stability or a universal advantage over other algorithms. [Unification audit](/home/vardan/dev/projects/omnibias/docs/unification-research-audit.md).

## 2. Constructions that must remain distinct

| Construction | Meaning | Important boundary |
|---|---|---|
| Fixed alternating hard thresholds | A scalar union of intervals, with specified initial label and sign convention | Boolean capacity counts depend on this convention |
| Fixed alternating sigmoid bands | Smooth finite-temperature version of that structure | Finite values are not exact Boolean zero/one outputs; even counts do not generally contain a sigmoid |
| Learned-coefficient scalar profile | A weighted sum of shifted activations returning one scalar | Coefficient freedom exceeds that of fixed-sign Boolean bands |
| Vector-valued compression profile | Shared incoming direction with destination-specific output coefficients | Can preserve outgoing responses that one scalar profile cannot |
| Baseline plus residual pairs | Ordinary activation retained, with initially cancelling trainable additions | Exact real-arithmetic initial function preservation does not imply identical training or bit identity |
| Learned score followed by hard bands | Train a smooth score, discard its head, fit an exact interval classifier | Head refitting is not lossless hardening of the original network |
| Exact equality readout | Membership in a zero set | Its ordinary derivative is not supplied by a smooth residual or band |

The distinctions above explain why a positive result in one family does not establish the same result for every other family. [Canonical construction table](/home/vardan/dev/projects/omnibias/docs/multi-bias-research-synthesis.md), [band architecture explanation](/home/vardan/dev/projects/omnibias/docs/multibias-followup-audit.md).

## 3. Corrected starting claims and novelty scope

Deep-network jet propagation is established Taylor-mode AD: its coefficients are derivatives of the represented graph, not a truncated numerical approximation to the requested derivatives. Bell/Faà di Bruno composition and derivative polynomials are established mathematics. Bias differences offer a useful derivation and source of features; independently replacing each deep activation with its derivative does not differentiate the original deep graph. [Deep bias-collapse proof](/home/vardan/dev/projects/omnibias/docs/deep-bias-collapse.md).

The older large derivative speed ratios compare specified specialized workloads against dense/nested controls. They do not establish a comparable advantage over the strongest structure-aware Taylor control, or an end-to-end PINN training speedup. Earlier constant-captured benchmark programs were corrected to accept runtime parameters; compilation, budgets and baseline wins remain disclosed. [Performance protocol and corrections](/home/vardan/dev/projects/omnibias/docs/performance.md).

Soft decisions, annealing, differentiable optimization, learned activation shapes, compatible PDE spaces and validated numerical certificates have substantial prior art. Functional overlap with an existing package is not proof that every API, supported problem, implementation or assurance contract is identical. Conversely, different packaging or framework parity alone does not make an established mathematical method new. The relevant comparisons must be made primitive by primitive and workload by workload. [Claim-to-source table](/home/vardan/dev/projects/omnibias/docs/calculus-geometry-campaign.md), [architecture precedents](/home/vardan/dev/projects/omnibias/docs/multi-bias-capacity-and-architecture.md), [pretrained comparison](/home/vardan/dev/projects/omnibias/docs/multibias-frontier-study.md).

The original broad judgments about commercial value, chemistry, funding and every package's novelty were author-supplied starting assessments; this session did not independently establish a valuation or exhaustive novelty verdict for every package. The canonical research reports support narrower conclusions about the implemented primitives and experiments. A targeted prior-art search is not an exhaustive priority determination. [Campaign contribution boundary](/home/vardan/dev/projects/omnibias/docs/calculus-geometry-campaign.md), [final synthesis classification](/home/vardan/dev/projects/omnibias/docs/multi-bias-research-synthesis.md).

### Original starting-state ledger — author supplied, not independently re-audited here

The following records the author's opening **17 October 2026 assessment**, including its numerical claims and literature comparisons. It is a historical input to this conversation, **not a new verified result of the later research campaigns**. Its cited alternatives establish relevant comparison directions; naming them does not establish exact feature, algorithm, license or assurance equivalence. The linked canonical reports document later scoped work, rather than independently validating every item in this ledger. [Later evaluation boundaries](/home/vardan/dev/projects/omnibias/docs/calculus-geometry-campaign.md), [performance reporting scope](/home/vardan/dev/projects/omnibias/docs/performance.md).

| Starting topic | What the author initially reported | Status retained in this compendium |
|---|---|---|
| A100 derivative baselines | Approximately 4,974×/5,036× figures used dense nested JAX/nested folx, without a structure-aware Taylor baseline. Against the latter, shallow tanh/sine training gains were about 1.1–1.6× at \(D=2\)–3, tying or losing at larger D. Deep example, order 6: JAX Taylor 0.077 ms versus omnibias 0.147 ms. | Author-supplied baseline audit. Later specialized timings remain workload-specific; no general derivative-engine or PINN speed claim follows. |
| A100 grid and Gauss–Newton reductions | For shallow sine/cosine/exponential models, linear operators and tensor grids, reported loss reduction was 3.7× geometric mean versus best per-point code, 20.8× at one million points; Gauss–Newton matrix reduction 8.5×; separable-network reduction 3.29×, with parity near one million and larger gains only at much larger grids. | Author-supplied narrow-regime evidence. Product-grid/separability, exact quadratic structure and comparison to the same reduction algebra are essential; no random-collocation/nonlinear-PDE generalization is asserted. |
| A100 setup and negative results | Poisson LM compile 28.8 s versus 2.3 s; break-even about 5,800 iterations; loss-reduction median break-even 8,209 calls. Near-fit rounding floors, deep-model losses, no robust fused-kernel gain, a failed stable front end, and invalid earlier Burgers VarPro numbers were disclosed. | Historical caveats retained. These A100 jobs/branches and every original row were not rerun or independently re-audited in the later CPU/T1200 campaigns. |
| `omnibias-nphard` | Comparisons named PI-GNN, Erdős Goes Neural, RL4CO and conventional heuristics; an example QAP SOS bound retained about a 10% gap, and LPT reportedly won 5/8 scheduling seeds. | Author-supplied package assessment, not a package-wide novelty proof. Differentiable optimization, decode quality, lower bounds and certificates are separate comparison dimensions. |
| `omnibias-logic` and `omnibias-boolean` | Logic comparisons named SATNet, Scallop, DeepProbLog, LTNtorch and IBM LNN. Product/multilinear Boolean gates overlap standard probability/t-norm formulas and DiffLogic; ANF, GF(2), Walsh transforms and Boolean derivatives are classical. | Relevant prior-art scope, not a finding that each package implements the same problem or interface. Later counting/expectation experiments evaluate specific laboratory models and local controls. |
| `omnibias-symbolic` and `omnibias-tab` | Symbolic comparisons named DeepMoD, PINN-SR, PDE-FIND, SINDy variants, PySINDy and PySR; derivatives of a fitted field retain fitting error. Tabular comparisons named soft trees, NODE, DNDF and TEL. Fractional/integral/asymptotic features were not a demonstrated new general calculus or unique equation-discovery mechanism. | Starting literature map. Later constraint adapters, learned profiles and hard-band classifiers have their own contracts; this session did not exhaustively re-audit all symbolic/tabular packages or infer every capability from their names. |
| Chemistry | NH3 barrier 9.38±1.50 mHa was said to make an apparent 3% splitting match unreliable because of nonlinear sensitivity; earlier high-accuracy NH3 surfaces and FermiNet cyclobutadiene work were named. The author withdrew “Nature-level first” framing. | Original scientific assessment only. This session did not rerun quantum chemistry, independently propagate the quoted uncertainty, or establish a new chemistry result. |
| Licensing and business | The author reported 9 Apache-2.0 packages and 7 AGPL-or-commercial packages, earlier Apache PyPI snapshots, strong free alternatives, doubtful near-term license/sale revenue, uncertain funding and possible employer-rights obligations. | Historical author assessment, not a current legal inventory, valuation, legal opinion or proof that commercialization is impossible. No license, domain or business decision was implemented by these research studies. |

The starting memo also rejected first-in-kind claims for bias/temperature limits, neural derivatives, generic certification and equation discovery. Later work tested narrower opportunities rather than treating that broad judgment as either a proved impossibility or an entitlement to novelty. The positive and negative evidence below belongs to the disclosed later protocols. [Unifying-theory audit](/home/vardan/dev/projects/omnibias/docs/unification-research-audit.md), [multi-bias assessment](/home/vardan/dev/projects/omnibias/docs/multi-bias-research-synthesis.md).

## 4. Calculus, exact geometry and compatible fields

| Finding | Positive result | Negative result or limitation |
|---|---|---|
| Exact affine membership | Rational membership, consistency, projection and distance are computable | Measurement perturbations can leave a zero-thickness set |
| Differentiable affine projection | Reduced QR preserves derivatives through points, matrices and right-hand sides | Numerical feasibility is distinct from exact membership; unsupported rank deficiency must be refused |
| Root sensitivity repair | Regular isolated roots and explicit affine projections have supported derivatives | Underdetermined roots need a selection rule; iteration maps are not nearest-point maps |
| Compatible derivative correction | Polynomial coefficient constraints and reconstruction produce one genuine field | Pointwise PDE residual satisfaction alone need not imply realizability |
| Exact PDE transfer | A fixed supported realization transfers coefficient solutions with boundary obligations | Does not contain every PDE solution or linearize arbitrary nonlinear PDEs |
| Cole–Hopf fixture | Positive finite heat series produce a verified restricted Burgers family | Direct transformed spectral implementation uses the same classical map |

These are completed mathematical/software contracts rather than evidence that an exact equality predicate itself has useful ordinary gradients. [Pilot derivations](/home/vardan/dev/projects/omnibias/docs/calculus-geometry-pilot.md), [campaign proofs](/home/vardan/dev/projects/omnibias/docs/calculus-geometry-campaign.md).

The original pointwise counterexample must be stated correctly: editing derivatives while retaining the value field \(u=x\) made the edited jet incompatible with that field. Its derivative vector integrated to a different field. A separate nonzero-curl example cannot integrate to any scalar field. The successful compatible correction and reconstruction remained valid throughout the audit. [Interpretation correction](/home/vardan/dev/projects/omnibias/docs/unification-research-audit.md).

| Early pilot, fixed wave representation | Held-out \(u\) RMSE | Maximum PDE residual | Setup + solve seconds |
|---|---:|---:|---:|
| Direct characteristic least squares | 0.004845836 | 1.67e-15 | 0.00016232 |
| SVD null-space least squares | 0.004845836 | 8.25e-14 | 0.00518564 |
| Native projected least squares | 0.004845836 | 4.79e-14 | 0.00263937 |
| Ordinary QR projection | 0.004845836 | 8.3e-14 | 0.000655202 |
| Matched sine network, Adam | 0.00492675 | 1.37e-15 | 0.156163 |

All methods share a restricted exact solution family. Direct coefficient algebra already solves it efficiently. These component timings exclude common construction/import work and lack isolated repeated samples, so they do not establish general speed ratios. [Pilot results](/home/vardan/dev/projects/omnibias/docs/calculus-geometry-results.md).

Exact rational affine-subspace recovery correctly classified 58 on-plane and 58 off-plane points. Shifting the positive measurements by exact \(1/1000\) off the plane while keeping latent labels gave zero positive recall. This is a noise-sensitivity result for that detector, not a prohibition on equality after a learned nonanalytic embedding. [Pilot results](/home/vardan/dev/projects/omnibias/docs/calculus-geometry-results.md).

## 5. Scientific verification and physical calibration

| Verification experiment | Outcome | Interpretation |
|---|---|---|
| Adaptive mixed enclosure | 10/12 completed versus 9/12 best uniform | Did not meet the 2× speed or +25-point completion gate |
| Original adaptive total-cost ratio | Best uniform/adaptive: 0.840× core, 0.834× independent | Original adaptive method was slower on common completed cases |
| Heap/order-inheritance follow-up | 1.123× core, 1.121× independent | A small implementation gain, reproduced by same-algebra control |
| Physical sensitivity enclosure | 4/6 meet \(10^{-4}\); all 6 certify a local calibration direction | Local direction is not a finite-step or recovery guarantee |
| Fixed parameter window [32,33] | Certified finite loss change in [−0.0001118, −0.00003285] | One predeclared finite-step demonstration |
| Global energy certificate at coupling 512 | State error below 0.00671 | Works where the previous contraction gate is provably blocked |

The old contraction factor is at least 2.0378 even at zero requested radius for the coupling-512 candidate; spatial refinement cannot repair that sufficient condition. The global monotone-energy proof adds capability using established functional analysis. [Scientific campaign](/home/vardan/dev/projects/omnibias/docs/calculus-geometry-campaign.md).

| Integrated parameter recovery, six discovery cases | Certified cases | Median complete cold cost |
|---|---:|---:|
| Neural proposals, safeguarded | 0/6 | —; all-attempt median 60.08 s |
| Neural proposals, bisection | 0/6 | —; all-attempt median 60.09 s |
| Spectral proposals, safeguarded | 6/6 | 3.884 s |
| Spectral proposals, bisection | 6/6 | 3.059 s |
| Local reduced basis, safeguarded | 6/6 | 5.556 s |
| Local reduced basis, bisection | 6/6 | 3.877 s |

Sparse point-sensor recovery completed 0/6 neural cases and 5/6 for each classical proposal family. All accepted recovery certificates were independently replayed and bound to their requested observation, recipe and resolution. No neural candidate qualified for fresh confirmation. [Physical calibration report](/home/vardan/dev/projects/omnibias/docs/verified-physical-calibration.md).

A post-discovery 64-cell sensitivity cover proves every interior sensor used here is individually injective over [0,512]: derivative magnitude lower bounds are 0.000110018 at the quarter points and 0.000231386 at the center. Reflection symmetry makes the two quarter sensors redundant. The proof is application-specific and does not retrospectively change the primary recovery matrix. [Observability cover](/home/vardan/dev/projects/omnibias/docs/verified-physical-calibration.md).

Higher-order residual enclosure improved some bounds but did not rescue the six fixed diagnostic neural fields: rational point residuals prove a floor above \(10^{-6}\) for the implemented residual-supremum certificate form. This is not a lower bound on their actual state error. A stronger representation or fitting recipe is required before another end-to-end comparison. [Failure diagnosis](/home/vardan/dev/projects/omnibias/docs/verified-physical-calibration.md).

## 6. Singular sources, compression, Hermite structure and numerical accuracy

| Experiment | Supported result | Limitation |
|---|---|---|
| Distributional ridge calculus | ReLU interfaces retain Dirac sources and higher singular derivatives | Arbitrary singular products/surfaces and complete boundary solvers are not supplied |
| Noiseless planar-source recovery | Maximum parameter error <3.8e-9; held-out relative measurement error <5.6e-9 | Local manufactured inverse example, not noisy/global superiority |
| Nearly coincident sources | Original location/strength condition number ~1.59e8; moment design 3.63 | Better-conditioned aggregate coordinates do not recover unidentifiable original locations |
| Positive Poisson source merge | Exact max-error controller and restricted greedy optimality | Positive 1D source family only; weak-source error alone does not bound field error |
| 4,096 uniform sources, tolerance 1e-3 | 16 merged forces /18 spline nodes; adaptive control 17 nodes | Classical representation already compact |
| 4,096 clustered sources, tolerance 1e-3 | 4 merged forces /6 nodes; adaptive control 9 nodes | Not a general multidimensional compression result |

The recorded 10.5–30.8× certificate-pipeline overhead compared exact rational construction/replay with floating classical builders/error checks. It is an **unequal-assurance measurement**, not a proof of intrinsic source-compression inferiority. The counterexample with two nearby Dirac loads correctly separates fourth-order weak probe error from first-order max-norm field error. [Source theory and experiments](/home/vardan/dev/projects/omnibias/docs/bias-collapse-pde.md), [assurance correction](/home/vardan/dev/projects/omnibias/docs/unification-research-audit.md).

| Independent fourth-Hermite contraction rerun, N=2,048 | Pair method ms | Best tensor ms | Tensor/pair |
|---|---:|---:|---:|
| Sign labels, D=8 | 31.465 | 0.631 | 0.020× |
| Sign labels, D=64 | 47.896 | 221.834 | 4.632× |
| RBF labels, D=8 | 35.003 | 1.155 | 0.033× |
| RBF labels, D=64 | 52.529 | 457.264 | 8.705× |

The crossover is reproducible, not contradictory: the pair method has an \(N^2D\) term, while the tensor/moment alternative pays for dimension-dependent combinatorial terms. Both get faster at small dimension, but the tensor comparator improves much more. Sixty-four exact rational pair identities matched an independent tensor oracle. [Hermite audit](/home/vardan/dev/projects/omnibias/docs/unification-research-audit.md).

Hermite/input-density initialization exposed useful signals on selected targets, including even responses that first moments miss. Stronger learned Laplace profiles nevertheless won all six dataset means and 29/30 matched splits in the stronger profile study. Informative derivative order depends on the input law; these were not collapse-specific universal wins. [Full higher-order studies](/home/vardan/dev/projects/omnibias/docs/bias-collapse-pde.md).

The opt-in local JAX tanh Taylor-rule patch repaired a real float32 saturation issue: the stock jet path returned zero derivatives/loss gradients at constructed tail inputs, while patched and existing stable controls did not. All 144 constructed loss configurations passed; maximum sampled relative gradient errors were 3.77e-7 in float32 and 6.44e-15 in float64. Installed JAX and its default path were not changed; high-order underflow at more extreme inputs remains. [Precision study and interval witness](/home/vardan/dev/projects/omnibias/docs/bias-collapse-pde.md).

A separate verified tanh dispatcher met 54/54 point budgets versus 16/54 for the older interval Horner kernel. A stronger multiprecision control also met 54/54 and was faster: candidate matched speed ratio 0.5353×. Improved robustness relative to an old implementation is not superiority over the strongest control. [Verified enclosure study](/home/vardan/dev/projects/omnibias/docs/bias-collapse-pde.md).

## 7. Tabular, Boolean, geometric and LP applications

| Application | Positive evidence | Negative/control evidence |
|---|---|---|
| Artificial hidden equalities, breast cancer | Projection model 97.368% accuracy /0.079670 NLL | Ordinary QR and absorbed-head network identical; CatBoost 98.246% accuracy |
| Correct multi-output constraints | Five fresh seeds: TabPFN MSE reduction 11.45%, MLP 9.53%, ridge 0.82%, CatBoost 0.25% | Produced by ordinary QR adapter around each named learner; wrong constraints increase error |
| Equation-invariant PFN context | Normal projector/anchor invariant to equation recombination | PFN pretraining and learned constraints were not performed |
| Unknown-support exactly-three | Counting block 100%; unrestricted LUT 99.23% | Exhaustive count-support search also 100% |
| Unknown-support parity | Model represents parity | Counting 49.01%, LUT 49.02%; GF(2) and exhaustive search 100% |
| Nonsymmetric target LUT | Local unrestricted LUT 71.28% | Counting prior restricted it to 66.21% |
| Coherent geometric field | Normal-anomaly recall 100% at ~5% false-positive rate | Tangent-rarity recall 2.41%; ordinary AD identical |
| Geometry clustering | Learned field ARI 0.645 | Quadratic fields 0.850 on the synthetic families |
| LP basis proposal | Exact KKT acceptance and sensitivity witnesses | Proposal acceptance 98.44%/69.27%/9.90% at dimension 2/8/32; warm HiGHS stronger |

Counting used 79 parameters versus 136 for the local LUT and seven versus 64 response entries. These were not official DiffLogic/DiffLUT reproductions. Joint logit scaling yielded 36/36 soft-to-hard certificates with probability error at most \(10^{-3}\), preserving the learned circuit's decisions, including its mistakes. [Application campaign](/home/vardan/dev/projects/omnibias/docs/calculus-geometry-campaign.md), [producer attribution](/home/vardan/dev/projects/omnibias/docs/unification-research-audit.md).

For LPs, proposal/check/fallback cost was 0.759/2.126/15.095 ms at dimensions 2/8/32, versus 0.532/1.350/14.016 ms for warm HiGHS with the same exact checker. Active-set sensitivities agreed within 7.93e-13 where valid. Solving the equations after finding a basis does not eliminate the basis-discovery problem. [LP comparison](/home/vardan/dev/projects/omnibias/docs/calculus-geometry-campaign.md).

## 8. Direct metrics, objectives and hardening

Exact stochastic expected accuracy/F1 is differentiable under its specified probability model. It is not the ordinary derivative of a hard deployed metric and need not optimize that metric after thresholding. A symbolic counterexample shows globally optimal expected F1 can yield worse optimally thresholded F1 than a BCE-selected member of the same family. [Objective audit](/home/vardan/dev/projects/omnibias/docs/unification-research-audit.md).

| Metric experiment | Result | Boundary |
|---|---|---|
| Rare-positive expected-F1 learning | Beat BCE 5/30, tied 5, lost 20 | Linear logits; does not rule out all deep/multi-bias models |
| Exact expected-F1 LP bounds | 76 cases replayed, known optima contained | General coupled cases retain positive gaps; LP proxy need not solve expected objective |
| Isolated hardened oracle | Pooled beta≥16 geometric speed ratio 1.9925× | Conditional value/gradient error bounds, not full training speed |
| Parameter-aware optimizer workflow | Full-control/candidate ratio 0.7941× | End-to-end candidate slower despite isolated oracle gain |

The LP bridge exposed and repaired a genuine soundness defect: an interval equality residual containing zero does not prove exact feasibility. The revised certificate checks binary64 input equalities with exact rational arithmetic and refuses merely nearby points. Ninety-two convex-package tests passed, including cancellation/refusal regressions. [Metric experiments and certificate repair](/home/vardan/dev/projects/omnibias/docs/bias-collapse-pde.md).

Reconvergent Boolean circuits require joint information. Naively composing marginal probabilities can disagree with the true expectation, even for \(X\mathbin{\mathrm{xor}}X\). A small dependency-aware implementation works within its declared scope; this does not make arbitrary exact circuit inference tractable. [Boolean semantics](/home/vardan/dev/projects/omnibias/docs/calculus-geometry-campaign.md).

## 9. Multi-bias peak escape, capacity and threshold counting

The two-sigmoid bell has a genuine independently trainable width direction at its peak, although its shared-input derivative remains zero. A fresh scalar experiment reached target 0.2 from the peak with loss ~9.86e-32; a matched fixed-height Gaussian stayed at loss 0.0343527. A Gaussian with trainable amplitude also solved that scalar task, so this is a parameterization/shape result, not evidence that Gaussian models cannot learn. [Bell experiment](/home/vardan/dev/projects/omnibias/docs/two-bias-bell-experiment.md).

XOR is represented with two input weights and two sigmoid biases, versus nine parameters in the specified two-hidden-unit MLP comparison. A trainable-amplitude Gaussian is another compact XOR representation. Boolean completeness requires sufficient profile complexity, valid threshold conventions and precision; it does not establish that arbitrary truth tables are cheap to learn. [XOR and enumeration](/home/vardan/dev/projects/omnibias/docs/two-bias-bell-experiment.md).

| Inputs | Known threshold-function count | Verified worst minimum alternating biases |
|---:|---:|---:|
| 1 | 4 | 1 |
| 2 | 14 | 2 |
| 3 | 104 | 3 |
| 4 | 1,882 | 5 |
| 5 | 94,572 | Only \(7\le B_0(5)\le31\) certified here |

The five-input proposed value eight remains unverified. All 516 fundamental chambers and 1,981,440 projection orders were replayed; one truth table has exact minimum seven. A counting bound gives \(B_0(8)\ge54\), contradicting the suggested Fibonacci extrapolation 34 at eight inputs. [Enumeration evidence](/home/vardan/dev/projects/omnibias/docs/two-bias-bell-experiment.md).

For four inputs, the number of functions representable with at most 1/2/3/4/5 alternating biases is 1,882/11,130/46,342/64,672/65,536. The gain is substantial. The exponential same-fixed-order representation ratio is not an exponential guarantee against the union of all single-threshold projection orders, nor a generalization or efficient-learning theorem. [Capacity definitions and exact counts](/home/vardan/dev/projects/omnibias/docs/multi-bias-capacity-and-architecture.md).

## 10. Width compression, depth feasibility and learning from scratch

| Digits model, three seeds | Accuracy | Export bytes | Teacher file reduction |
|---|---:|---:|---:|
| Teacher, 64 sigmoid units | 97.31% | 20,528 | — |
| Fixed shared profile, 16 directions/64 shifts | 97.50% | 8,548 | 58.36% /2.40× |
| Adaptive shared profile | 97.50% | 8,728 | 57.48% |
| Ordinary pruning to 16 | 97.13% | 6,128 | 70.15% |
| Ordinary pruning to 23 | 97.69% | 8,228 | 59.92% |
| Local merging to 16 | 97.41% | 6,128 | 70.15% |
| Shared spline control | 97.59% | 8,259 | 59.77% |

This is **widthwise direction sharing at unchanged depth**, not the removal of layers. Digits compression inherited teacher parameters and fine-tuned using labels, not teacher-logit distillation. The conventional comparator gate failed; the planned competitive confirmation was therefore not run. Most main fits selected their final allowed checkpoints, so these are fixed-budget outcomes rather than converged optima. [Compression report](/home/vardan/dev/projects/omnibias/docs/activation-profile-compression.md), [clarifications](/home/vardan/dev/projects/omnibias/docs/multi-bias-learning-and-margin.md).

Exactly aligned synthetic profiles achieved zero error versus pruning 0.00186; at direction perturbation 0.05 they achieved 0.00124 versus 0.00260. At perturbation 0.20, pruning won: 0.00956 versus profile 0.01699. Whole-box sharing bounds ranked synthetic errors well (Spearman 0.983) but digits poorly (0.158). Valid bounds were not reliable practical predictors on real data. [Controlled compression](/home/vardan/dev/projects/omnibias/docs/activation-profile-compression.md).

The depth experiment used teacher-output and label loss. A 64→32 teacher reached 97.13%/28,035 bytes; a shallow profile 96.85%/8,548; an ordinary shallow network 97.59%/8,228. An analytic one-projection composition is compressible, but multidirectional counterexamples survive. General inexpensive depth removal is unsupported. [Depth evidence](/home/vardan/dev/projects/omnibias/docs/activation-profile-compression.md).

Early one-seed image experiments completed 21 fits, 19 time-capped. ReLU led the main MNIST/Fashion/CIFAR matrix. Moving a cancelling pair away from tied initialization improved Fashion accuracy 91.22→91.74%, but its NLL slightly worsened 0.24320→0.24421. Initialization mattered, but this was a diagnostic, not repeated-seed superiority. [Image learning study](/home/vardan/dev/projects/omnibias/docs/multi-bias-learning-and-margin.md).

The later controlled three-seed Fashion study found single-bias tanh and adaptive growth both 91.430%; pair LR×10 reached 91.753% versus matched single-bias LR×10 91.633%; translated-ReLU control reached 91.940%, practical ReLU 92.000%. Gap penalties and adaptive growth did not pass advancement. [Frontier discovery](/home/vardan/dev/projects/omnibias/docs/multibias-frontier-study.md), [synthesis table](/home/vardan/dev/projects/omnibias/docs/multi-bias-research-synthesis.md).

## 11. Pretrained compression, filtering and compiled bands

| Pretrained primary matrix | Teacher | Shared profiles | Stronger relevant control |
|---|---:|---:|---:|
| ResNet-18 accuracy | 78.17% | 51.22% | Pruning 77.91% |
| ResNet-50 accuracy | 87.18% | 86.55% | Pruning 86.64%; quality/storage gates still fail |
| GPT-2 perplexity | 25.170 | 29.464 | Pruning 26.153 |

The primary pretrained matrix was incomplete and cannot establish a promoted comparative result. A real scale-handling weakness was found: a retrospective ResNet-18 correction improved 59.465→60.892%, still below pruning 77.783%. A secondary ResNet-18 rollout gave profiles 81.89% versus conventional rank/pruning 78.34%, but profiles were larger (43.684 versus 39.745 MB), and this was a distinct protocol. Both negative and favorable secondary findings must be retained. [Frontier report](/home/vardan/dev/projects/omnibias/docs/multibias-frontier-study.md), [forensic correction](/home/vardan/dev/projects/omnibias/docs/multibias-followup-audit.md), [synthesis](/home/vardan/dev/projects/omnibias/docs/multi-bias-research-synthesis.md).

The analytic filtering failure was substantially repaired. Five fresh amplitude-band problems gave old initialization MSE 0.0381 and 0/5 successes below \(10^{-3}\); moment initialization plus joint fitting achieved 1.56e-6, variable projection 1.01e-7, and a native-band control 6.81e-7, each 5/5. The two-direction nonlinear target remained near 0.044 for all one-direction methods. This confirms a useful initialization/fitting repair while preserving the missing-direction counterexample. [Filtering audit](/home/vardan/dev/projects/omnibias/docs/multibias-followup-audit.md).

The strongest compact-classifier result trains a smooth projection, discards its soft head and refits exact hard intervals. Five fresh problems from one repeated-class synthetic family gave balanced accuracy 92.66% versus compact MLP 87.62%; paired interval +2.84 to +7.25 percentage points. Actual exports were 1,482 versus 2,247 bytes, a 1.52× reduction. It was not a 2× actual-file gain or a result on five unrelated benchmarks. [Band architecture and setup](/home/vardan/dev/projects/omnibias/docs/multibias-followup-audit.md), [comparison](/home/vardan/dev/projects/omnibias/docs/multi-bias-research-synthesis.md).

The score learner had SVM-derived initialization versus random MLP initialization; two selected learners used only one bias; the compiler adds a refitted head; matched projection-pursuit and same-initialization controls are missing. The winning pipeline did not use LP margin refinement. Radial negative control favored RBF SVM/tree over one-axis bands. These conditions prevent attributing the gain uniquely to multi-bias shape. [Band limitations](/home/vardan/dev/projects/omnibias/docs/multi-bias-research-synthesis.md).

## 12. Existing-paper results: useful signals, not independent replications here

| Paper-reported task | Positive result | Negative or comparison boundary |
|---|---|---|
| Binary CIFAR-10, matched topology | Learned K=3 83.79±0.16% versus baseline 81.89±0.33% | Fixed-alternating K=3 80.74±0.66%; learned coefficients matter |
| Binary MNIST depth/width | Shallow K=2 97.70±0.11%, 200,704 binary weights | Architecture/resource tradeoff, not universal depth removal |
| ResNet-50 adaptation | CIFAR-100 K=3 82.31%; inserted K=1 81.78%; Flowers K=3 86.76%, K=1 84.17% | Food101/Pets/ViT rows include losses/ties; LoRA configuration not a universal comparison |
| Localization | Segmentation K=3 IoU94.99% versus BitFit93.91% | Bounding-box LoRA_layer4 stronger than K=3 |
| Tabular | Covertype positive | Adult and wine-quality conventional controls stronger |
| Small character GPT | K=3 best-val 2.231 BPC | ReLU2.230; no reasoning/generation superiority |
| Label noise | No broad favorable pattern | At 40% noise K=3 75.46% versus ReLU77.80% |
| OOD | Mahalanobis AUROC improves in disclosed rows | MSP and in-distribution accuracy worsen |
| Adversarial robustness | Earlier weak-attack appearance investigated | Audited Lipschitz-projected cell AutoAttack robust accuracy0%; do not reassign to another cell |

The session extracted and qualified these existing records rather than independently reproducing the paper's full training suite. Binary weight counts, trainable adapter counts and complete deployed bytes are different quantities. [Paper evidence and caveats](/home/vardan/dev/projects/omnibias/docs/multi-bias-research-synthesis.md).

## 13. Latest projection-versus-profile allocation study

The completed study retained 60/60 controlled cases and 42/42 CNN fits. Focused checks, full numerical replay and independent acceptance audit passed. No dataset passed advancement, so fresh confirmation was not run. Completion of an experiment is not convergence or promotion. [Allocation report](/home/vardan/dev/projects/omnibias/docs/projection-profile-allocation.md).

| Controlled regime, long-fit latent MSE | Continue | Profile | New projection | Tanh + hinges |
|---|---:|---:|---:|---:|
| Ordinary ridge, d8/n512 | 0.000070 | 0.000076 | 0.000149 | 0.000070 |
| High-noise ordinary ridge, d8/n512 | 0.003992 | 0.004213 | 0.008699 | 0.004967 |
| Three bands, d64/n2048 | 0.256933 | 0.014625 | 0.043522 | 0.037446 |
| Two equal components, d8/n512 | 0.046700 | 0.046669 | 0.012004 | 0.046617 |
| Weak second component, d64/n128 | 0.011330 | 0.011376 | 0.013045 | 0.011150 |
| Weak second component, d64/n2048 | 0.004230 | 0.004263 | 0.000314 | 0.004231 |
| Product interaction, d8/n512 | 0.773982 | 0.709436 | 0.427142 | 0.595116 |

In the strongest narrow profile stratum, one direction with nine residual pairs used 405 payload bytes versus 556 for two ordinary directions: 27.16% less payload, but only 3.93% less actual NPZ size. Mean profile MSE was 0.01462451 versus fresh ordinary R2 0.04399638. The target was generated from shifted bands, inference was slower, and important fits were still improving. [Allocation tables and timing](/home/vardan/dev/projects/omnibias/docs/projection-profile-allocation.md).

| CNN policy | Fashion accuracy /NLL | CIFAR-10 accuracy /NLL |
|---|---|---|
| Continue/refine tanh | 91.41% /0.23965 | 72.18% /0.79910 |
| Profile-only policy | 91.41% /0.23965 | 72.24% /0.79850 |
| Projection-only policy | 91.41% /0.23965 | 72.18% /0.79898 |
| Joint selector | 91.41% /0.23965 | 72.22% /0.79836 |
| Conventional spline policy | 91.89% /0.23053 | 77.31% /0.65665 |
| Eligible practical ReLU control | 91.75% /0.22853 | 77.57% /0.64662 |

Fashion's retained tanh exports were identical within each seed across the five tanh policies: growth was rejected, not successfully retained. The CNN profile action often opened a pair on all 112 channels; it did not test every selective subset. Refinement also reset selected filter Adam state, so its effect is not purely geometric. [CNN diagnostics](/home/vardan/dev/projects/omnibias/docs/projection-profile-allocation.md).

Short-versus-long best action rankings reversed in 22/60 cases, with 344/540 branch/control fits flagged optimization-unresolved. Metric local scoring predicted the short profile-versus-projection ordering in 41/60 versus raw gradients 35/60; on common non-tied long cases, span scoring 43/57 exceeded metric 41/57. There is no uniformly best predictor. External CPU contention affected some fit wall times; no isolated training-speed claim follows. [Selection limitations and audit](/home/vardan/dev/projects/omnibias/docs/projection-profile-allocation.md).

## 14. Theory catalog and key equations

**Geometry and calculus.** Separate residual \(r=Ax-b\), set \(M=\{r=0\}\), and hard membership \(E_M=\mathbf1_M\). For full-row-rank \(A\),

\[
\Pi_Mx=x-A^T(AA^T)^{-1}(Ax-b),\qquad
\nabla\!\left[\tfrac12\operatorname{dist}(x,M)^2\right]=x-\Pi_Mx.
\]

Reduced QR is the supported numerical realization; exact membership remains separate. Nonlinear nearest-point derivatives require the curvature-bearing bordered KKT matrix, a regular selected branch and local/global selection assumptions. [Geometry proofs](/home/vardan/dev/projects/omnibias/docs/calculus-geometry-pilot.md).

**Compatible realization.** Linear \(\mathcal LT=TM\) and boundary transfer imply \(Mc=g\Rightarrow\mathcal L(Tc)=Tg\). Nonlinear \(DT(c)F(c)=N(T(c))\) transfers coefficient dynamics by the chain rule. Polynomial jets are realizable exactly when \(\partial_iJ_\alpha=J_{\alpha+e_i}\), including the value field; polynomial Poincaré reconstruction gives a computable potential when compatible. Nonlinear closure, topology, initial/boundary conditions and finite-family coverage remain obligations. [Campaign derivations](/home/vardan/dev/projects/omnibias/docs/calculus-geometry-campaign.md).

**Bias and source calculus.**

\[
D_x^\alpha\phi(w^Tx+b)
=w^\alpha\partial_b^{|\alpha|}\phi(w^Tx+b).
\]

Normalized finite shifts converge to derivative features, with finite-spacing remainders. Distributional \(DT_j=T_{j-1}\) retains interface loads. Positive Poisson merging has exact error \(\tfrac12\sum_i m_i|s_i-c|\), while general weak-source errors need a stability map into the desired solution norm. [Ridge and source derivations](/home/vardan/dev/projects/omnibias/docs/bias-collapse-pde.md).

**Variational verification and inverse physics.** For the stated clamped/nonnegative-coupling problem,

\[
\|u-v\|_\infty
\le C\|v''''+\lambda v^3-384\|_\infty,
\qquad C=\frac1{96\sqrt{15}}.
\]

For \(H=\tfrac14\int u^4\),

\[
H'=-\langle u^3,L^{-1}u^3\rangle<0,\qquad
-H'\ge\frac{I^2}{384\int u+2\lambda I},
\quad I=\int u^4.
\]

Conjugate observables satisfy \(DO=-J^*L^{-1}J\); independent columns imply injectivity under the written smooth/coercive assumptions, not generic identifiability of arbitrary sensors. [Physical theory](/home/vardan/dev/projects/omnibias/docs/verified-physical-calibration.md).

**Representation and learning geometry.** With outputs \(F(\theta)\), Jacobian \(J\), and output loss-gradient \(g\), gradient flow satisfies

\[
\dot F=-JJ^Tg,\qquad \dot L=-\|J^Tg\|^2.
\]

Keeping the old path and appending zero-coefficient features \(\Psi a\) adds \(\Psi\Psi^T\) to this kernel and supplies descent when \(\Psi^Tg\ne0\). Coordinate metric, normalization, feasible signs and finite-step acceptance matter; no global or test-risk guarantee follows. [Unification argument](/home/vardan/dev/projects/omnibias/docs/unification-research-audit.md), [split descent](/home/vardan/dev/projects/omnibias/docs/multi-bias-learning-and-margin.md).

**Profile approximation and oscillations.** A Lipschitz 1D target admits signed sigmoid approximation with

\[
\|P-f\|_\infty
\le\frac{L(b-a)}{2K}+\frac{2L\log 2}{\beta}.
\]

Fixed-temperature unrestricted coefficients are dense on compact intervals but may be ill-conditioned; fixed-amplitude ordered bands have derivative bounded by \(\beta/4\), regardless of count. A common-slope \(K\)-term profile has at most \(K\) threshold intersections and \(K-1\) stationary points; these bounds are tight under its coefficient/offset conventions. [Approximation and oscillation proofs](/home/vardan/dev/projects/omnibias/docs/multi-bias-learning-and-margin.md).

**Exact Boolean counts.** The known threshold count obeys the hyperplane-arrangement identity

\[
T(n)=\sum_{S\subseteq\{0,1\}^n}(-1)^{|S|-r(S)}.
\]

Here \(r(S)\) is the rank of the augmented cube vectors \((x,1)\) for \(x\in S\). For a fixed order of \(N\) scores, alternating far-left-zero units realize \(\sum_{j=0}^{\min(K,N)}\binom Nj\) label sequences; unrestricted coefficient/offset profiles realize \(2\sum_{j=0}^{\min(K,N-1)}\binom{N-1}j\). Optimizing directions involves realizable cube orders and overlaps. Description length and margins prevent interpreting small real-parameter counts as cheap worst-case universality. [Capacity proofs](/home/vardan/dev/projects/omnibias/docs/multi-bias-capacity-and-architecture.md), [oscillation counts](/home/vardan/dev/projects/omnibias/docs/multi-bias-learning-and-margin.md).

**Pruning, filtering and output routing.** The sigmoid-pair supremum is \(\tanh(\beta(b-a)/4)\); actual coefficient changes and downstream amplification must be included in pruning bounds. Shared-kernel input derivatives remain \(\nabla_xp(k^Tx)=p'(k^Tx)k\), giving amplitude selectivity rather than additional spatial directions. Exact replacement of independent response features requires outgoing routing rank, not just one shared incoming direction. [Margin and rank theory](/home/vardan/dev/projects/omnibias/docs/multi-bias-learning-and-margin.md).

**Population allocation decomposition.** Squared risk separates irreducible noise, conditional information loss \(\mathbb E[\operatorname{Var}(m(X)\mid Z)]\), and permitted profile/head approximation error. Cross-entropy similarly separates \(H(Y\mid X)\), \(I(Y;X\mid Z)\), and conditional KL approximation error. Additive response classes can lack interactions even when \(Z\) retains all information. Finite optimization and finite-data selection are additional errors, not part of those population equalities. [Allocation theory](/home/vardan/dev/projects/omnibias/docs/projection-profile-allocation.md).

**Local gain, known-field certificates and budget theory.** Regularized Jacobian QR/SVD prediction depends on a declared metric; equivalent rescaling requires the metric to transform. Known-field Gaussian derivative-energy bounds, finite-degree/spectral information bounds, finite-support counting, PWL approximation budgets, cancellation examples and feasible-cone opening refine the decision conditions. They are conditional mathematical analyses, not reliable population certificates extracted automatically from a fitted CNN. [Full allocation derivations and linked notes](/home/vardan/dev/projects/omnibias/docs/projection-profile-allocation.md).

**Discrete expectations and solver contracts.** Count-probability derivatives equal expected adjacent response differences; a Hamming-tied LUT is the same function class. Soft-to-hard coupling bounds certify agreement, not label accuracy. Exact fixed-score segmentation gives a dynamic program; fixed memberships permit a margin LP; neither solves global joint projection search. Exact primal/dual KKT checks accept LP bases, but basis finding and transitions remain. Continuous-bias pricing gives scoped convex measure certificates, not general neural training optima. [Campaign discrete theory](/home/vardan/dev/projects/omnibias/docs/calculus-geometry-campaign.md), [band contracts](/home/vardan/dev/projects/omnibias/docs/multi-bias-learning-and-margin.md).

## 15. Corrections and proof limitations that must remain visible

The manuscript's Gaussian-half decomposition does not alter the shared-input derivative; its two bias derivatives vanish at the Gaussian peak. The author's different two-sigmoid construction has nonzero individual bias directions there and was positively reproduced. These statements concern different models and are consistent. [Unification correction](/home/vardan/dev/projects/omnibias/docs/unification-research-audit.md), [positive bell experiment](/home/vardan/dev/projects/omnibias/docs/two-bias-bell-experiment.md).

Exact function inclusion does not imply bit-identical evaluation under arbitrary tensor shapes, kernels, reductions or nonfinite inputs. Even-K fixed alternating units have different global limits from a single sigmoid; tied-center odd-K units can have symmetry/conditioning traps. Naive absolute-gap penalties can fail to open coincident pairs; positive mandatory gaps can remove a desired baseline embedding. [Compatibility and learning dynamics](/home/vardan/dev/projects/omnibias/docs/multi-bias-research-synthesis.md), [allocation checks](/home/vardan/dev/projects/omnibias/docs/projection-profile-allocation.md).

The manuscript's stated fixed-lattice spectral theorem had an out-of-band Gaussian counterexample; band projection or a restricted error claim is needed. Finite observed derivatives do not identify the field off-grid, and accurate fitted-field derivatives do not remove fitting error in equation discovery. Written proofs and directed arithmetic are trusted components; no formal proof-assistant verification or exhaustive novelty search was performed. [Spectral audit](/home/vardan/dev/projects/omnibias/docs/unification-research-audit.md), [campaign compatibility](/home/vardan/dev/projects/omnibias/docs/calculus-geometry-campaign.md).

The 2016 concatenated multi-bias CNN and the dimension-preserving summed profile are genuinely different interfaces. The former retains multiple independently routed responses; the latter ties them locally. Dimension preservation is useful, but broad learned scalar-profile ideas have other precedents including APL, splines and projection pursuit. The knots-versus-directions allocation problem has direct MASA precedent. [Architecture comparison](/home/vardan/dev/projects/omnibias/docs/multi-bias-research-synthesis.md), [allocation prior art](/home/vardan/dev/projects/omnibias/docs/projection-profile-allocation.md).

## 16. What to preserve and what the evidence currently supports

Preserve exact geometry and compatible-field primitives, independent certificate replayers, the LP soundness repair, numerical accuracy diagnostics, Boolean oracles, controlled negative examples, profile conversion/growth prototypes, exact band compilation, and the source/data/dependency fingerprints. Keep research prototypes opt-in and labeled by their actual assumptions; no benchmark here justifies deleting useful checked engineering or promoting every prototype to a default. [Campaign assessment](/home/vardan/dev/projects/omnibias/docs/calculus-geometry-campaign.md), [multi-bias continuation](/home/vardan/dev/projects/omnibias/docs/multi-bias-research-synthesis.md).

| Decision | Evidence-based answer |
|---|---|
| Enrich a profile | A useful existing score retains the signal, and independent validation supports a more complex scalar response under the budget |
| Refine a direction | The current projection count can capture the target but its orientation is poor; compare continuation and optimizer-reset controls |
| Add a projection | Useful conditional information is missing, or a mixed direction represents an interaction more economically |
| Preserve additional response channels | Downstream destinations require independent scalar responses even with the same incoming direction |
| Do not grow | Gains are noise-like, validation-unstable, horizon-dependent, unsupported by sample size or outside the storage allowance |

These are conditional, testable answers, not an oracle selector. The strongest next experiment is selective profile growth with adaptive trial lengths, matched long-horizon controls, explicit optimizer-reset ablations and targets not generated from the candidate family. The strongest compact-classifier follow-up needs same-initialization projection-pursuit controls. Verified calibration should preserve the sound inverse-set contract and use spectral/bisection until a stronger neural representation clears the measured precision floor. [Final allocation assessment](/home/vardan/dev/projects/omnibias/docs/projection-profile-allocation.md), [band continuation](/home/vardan/dev/projects/omnibias/docs/multi-bias-research-synthesis.md), [calibration decision](/home/vardan/dev/projects/omnibias/docs/verified-physical-calibration.md).

The surviving contribution is a concrete research program and reproducible evidence about **when activation complexity is useful**, with several narrow positive results and meaningful counterexamples. A universal/default K>1 activation advantage, a new general calculus, a faster generic LP algorithm, universal depth replacement and a simple new threshold-count formula remain unestablished. The reports classify classical mathematics, useful engineering, architectural candidates, disproved overgeneralizations and unresolved priority separately. [Contribution classifications](/home/vardan/dev/projects/omnibias/docs/multi-bias-research-synthesis.md), [allocation boundary](/home/vardan/dev/projects/omnibias/docs/projection-profile-allocation.md).

## 17. Commit scope, verification and remaining publication work

This checkpoint preserves the whole session in two repositories. The LAB initial
snapshot is `10ac0799c925c80a9ea20e5128bc2692e7ff10f6`, on
`codex/session-research-20261023`: 614 canonical source, test, guidance and evidence
files. The core checkpoint includes the affine geometry, locus differentiation,
compatible-jet and distributional-source work, the LP feasibility repair,
cross-backend regressions, research reports, figures and this compendium.
The separate author paper is outside these commits.

The curated LAB allocation archive contains 117 files, about 23.25 MB. All 116
manifest-listed content/support files were verified against both the working
files and the **committed Git blobs**. Original campaign files remain unchanged.
Core research-data Git attributes preserve original byte streams, including CSV
line endings, so checkout does not silently invalidate recorded evidence hashes.
Environments, downloaded datasets, model checkpoints, runtime logs and duplicate
replay workspaces remain outside Git. Full model replay requires the retained
local artifacts and recorded dependencies; copied-report integrity alone is not
numerical replay or independent retraining.

| Commit-phase check | Result and scope |
|---|---|
| Changed primitive, contract and parity tests | 251 passed; CPU, JAX float64 enabled, one thread |
| Selected LAB allocation/compression/frontier regressions | 134 passed in the recorded isolated Python 3.12 environment |
| Executable documentation snippets | 58 passed |
| Compendium code/math/link-preservation regressions | 10 passed; independent reviewer also checked the repaired Hermite expression and all 29 template links |
| API inventory, changed Python Ruff and license markers | Passed for the affected source; new compiler/tests pass Ruff too |
| Strict documentation build | Passed; absolute local reference links are retained with informational messages |
| Full affected-package and parity sweep | **8,663 passed, 13 failed, 15 skipped, 9 deselected**, 890.52 seconds |
| Full Ruff / strict mypy / license-header sweep | Not globally clean: 464 Ruff issues, 1,363 mypy errors in 27 files, and 28 missing tier headers in unchanged experimental JAX code |
| Source-note and archive inventory | 102 complete source notes; source hashes and overview hash checked; 116 curated archive blobs checked |

The 13 broad-suite failures have distinct causes. Eleven certificate/provenance
tests pass in isolation on both current source and the pre-session HEAD, but fail
after importing that HEAD's experimental JAX bounds module: it sets a global
strict-transcendental flag at import time. This collection-order interaction was
reproduced against the old source. One failure concerns unchanged experimental
files' missing license headers. The remaining no-leakage guard flags local
workstation paths and scheduler references. It already had baseline findings,
and the session's intentionally preserved research provenance **adds findings**;
this is an unresolved publication-hygiene issue, not a passing check or an
entirely preexisting failure. A credential-pattern audit found no secret matches;
that does not erase the path-hygiene failure.

These are **local research checkpoints, not release-ready publication commits**.
The guard and certificate checks were not weakened. A publication pass needs
portable, explicitly transformed public reports while retaining original hashed
provenance, plus the existing experimental-code cleanup and import-state repair.
Expected whitespace in frozen patches, generated plots and archived CSV records
was retained rather than silently editing historical evidence.

The structured [checkpoint verification record](/home/vardan/dev/projects/omnibias/docs/data/session-research-20261023/validation.json)
records commands, counts and failure classification. The
[source inventory](/home/vardan/dev/projects/omnibias/docs/data/session-research-20261023/sources.json)
records original source hashes and the LAB commit. Regenerate this document from
the repository root with:

```bash
.venv/bin/python scripts/compile_session_findings.py
.venv/bin/python -m pytest tests/test_session_compendium.py -q
.venv/bin/python -m mkdocs build --strict
```

The compiler preserves its recorded pre-checkpoint core commit across replay;
`--root-base` explicitly starts a new provenance baseline. This commit-phase
verification did not train new models or promote additional scientific claims.
