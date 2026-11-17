# Projection or profile: final evidence assessment

Generated 2026-10-22T22:45:22.704539+00:00. Frozen source `f552253d95e3e6a9112302256704fff13cbd31806e1a0e470084f945ac75da53`. This is a post-timing interpretation of retained artifacts; it performs no training, numerical replay or test-based recipe selection.

## Answer to the research question

**Enrich an existing activation when a useful direction is already available and richer scalar responses deliver a reproducible advantage within the byte budget. Refine or add directions when they expose a residual component, repair orientation, or make an interaction economical. Keep capacity unspent when independent evidence of growth is weak; treat allocation as provisional when optimization remains unresolved.**

The campaign supports conditional decision cases, not a generally superior allocator or a default activation. Its strongest controlled profile result is a deliberately favorable complex one-score response; CNN results must be read against complete conventional-control cohorts and the acceptance audit.

## Practical allocation rule

| Evidence after matching training horizon and storage | Action | Limitation to check |
|---|---|---|
| Useful existing score, genuinely complex scalar response | Enrich the profile | Compare a selective profile bundle with ordinary and hinge/spline controls |
| Better orientation works with the existing count | Refine | Include continuation and a reset-only optimizer control in future attribution studies |
| Useful missing conditional information | Refine or add a projection | Profile-only changes cannot restore information discarded by the score map |
| Scores retain information but the additive head misses interactions | Add a mixed projection or change the head | A useful new projection can lie inside the old linear span |
| Multiple downstream outputs need different scalar responses | Preserve response routing capacity | One scalar response plus a linear readout need not replace a higher-rank vector profile |
| Gain is training-only, uncertain, horizon-dependent, or unaffordable | Retain capacity; use continuation/early stopping | Report uncertainty or budget ineligibility; execution completion is not a capacity certificate |

Zero-byte refinement is compared directly by validation loss. Positive-byte actions pay for incoming and outgoing weights, normalization, active profile parameters and topology. The one-standard-error preference used here is a heuristic, not a formal confidence guarantee.


## Empirical allocation map

The left panel is the frozen short-validation decision. The other panels use latent targets only for post-hoc diagnosis. Each row is three seeds at one registered setting; this is a discrete map, with no inferred continuous phase boundary. Tiny differences and three-seed counts do not establish a universal threshold.

![Allocation decisions and horizon reversals](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/figures-final/empirical_allocation_map.png)

[Predicted versus realized gains](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/figures-final/predicted_vs_realized_relative_gains.png), [rank agreement](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/figures-final/predictor_rank_agreement.png), [all pairwise reversals](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/figures-final/all_pairwise_validation_reversals.png) and their [coordinates and scope](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/figures-final/captions-and-scope.md) disclose local-score failures. The regularized metric predicts the short profile-versus-projection ranking in 41/60 cases; raw-gradient ranking does so in 35/60. On the common 57 non-tied long cases, metric ranking is correct in 41 and span ranking in 43, so no uniformly best predictor is established.

## Completeness, replay and advancement

| Check | Recorded outcome |
| --- | --- |
| Controlled cases | 60/60 retained; {'COMPLETE': 60} |
| Full numerical replay | PASSED |
| Independent acceptance audit | PASSED |
| Promotion blocked by independent audit | False |
| Audit defects / incomplete / false acceptance | 0 / 0 / 0 |
| Short/long best joint-action reversals | 22/60 |
| Optimization-unresolved branch/control fits | 344/540 |

| Dataset | Discovery gate | Qualified | Confirmation gate | Confirmed gate |
| --- | --- | --- | --- | --- |
| fashion | COMPLETE | False | not run | False |
| cifar10 | COMPLETE | False | not run | False |

Independent artifact/cohort acceptance and full replay passed; this validates the recorded assurance checks, not universal superiority. A complete matrix can fail its advancement gate without being a failed numerical certificate. Missing confirmation is not confirmation of an advantage.

## Controlled positive and negative cases

The following are mean long-fit latent MSEs across retained discovery seeds. They compare actual branches, not an oracle population optimum. Noisy observed error is kept separate in the full report. Rows are heterogeneous constructed tasks, so these are not independent benchmark wins.

| Setting | Seeds | Continue | Profile | Projection | Tanh + hinges | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| ordinary; d=8, n=512 | 3 | 0.000070 | 0.000076 | 0.000149 | 0.000070 | Avoid extra capacity |
| ordinary_noisy; d=8, n=512 | 3 | 0.003992 | 0.004213 | 0.008699 | 0.004967 | Noise/estimation control |
| single_band; d=8, n=512 | 3 | 0.150457 | 0.031102 | 0.023392 | 0.029834 | An extra unit may optimize faster |
| three_bands; d=64, n=2048 | 3 | 0.256933 | 0.014625 | 0.043522 | 0.037446 | Complex scalar response favors profile |
| two_equal; d=8, n=512 | 3 | 0.046700 | 0.046669 | 0.012004 | 0.046617 | Distinct components favor direction capacity |
| weak_second; d=64, n=128 | 3 | 0.011330 | 0.011376 | 0.013045 | 0.011150 | High-dimension/small-sample uncertainty |
| weak_second; d=64, n=2048 | 3 | 0.004230 | 0.004263 | 0.000314 | 0.004231 | More data can justify the weak direction |
| interaction; d=8, n=512 | 3 | 0.773982 | 0.709436 | 0.427142 | 0.595116 | Retained coordinates can still lack interactions |
| near_collinear; d=8, n=512 | 3 | 0.002194 | 0.000852 | 0.000729 | 0.001059 | Cheap ordinary oracle exists; optimization confound |

The 200-update acceptance decision and 1,000-update result are different questions. A reversal does not itself reveal a coding error. Continued improvement, separate fixed-grid initialization and different validation horizons prevent an asymptotic-capacity interpretation. The near-collinear ordinary witness is post-hoc analytic/oracle information, not a new trained comparator.

## Strongest narrow profile stratum: three bands, d=64, n=2048

The target is generated by shifted tanh bands along a single useful axis; the candidate basis is well matched to it. This is a favorable structured test, not an out-of-family proof. All listed models obey the same registered byte ceiling.

| Model | Seeds | Latent MSE | Directions | Pairs/hinges | Payload B | NPZ B | Fit s | Unresolved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ordinary continuation | 3 | 0.25693308 | 1 | 0 | 288 | 3,570 | 1.76 | 0/3 |
| Tanh residual profile | 3 | 0.01462451 | 1 | 9 | 405 | 3,687 | 3.67 | 3/3 |
| Added ordinary projection | 3 | 0.04352242 | 2 | 0 | 556 | 3,838 | 1.93 | 3/3 |
| Fresh ordinary R2 | 3 | 0.04399638 | 2 | 0 | 556 | 3,838 | 2.02 | 3/3 |
| Tanh + hinge residual | 3 | 0.03744568 | 1 | 9 | 396 | 3,678 | 2.99 | 3/3 |

Profile payload 405 B versus 556 B for two ordinary directions is 27.16% smaller; actual NPZ reduction is only 3.93%. The spline uses 396 B. The incremental-growth byte ratio is 2.291×; it is not a total-model compression ratio. A residual pair adds two shifted tanh evaluations, so storage does not imply reduced latency.

Mean profile latent MSE is 66.40% below the new-projection branch and 60.94% below the local spline. These are finite-budget, same-fixture descriptive comparisons. The synthetic control adds trainable ReLU-hinge residuals to the ordinary tanh path; it is initialized from ranked profile centers and is not a pure PWL model or an official spline/PPR reproduction. In this stratum fresh R2 training loss still improves 20–22% over its final 100 updates, versus 2.5–5.5% for profiles.

## CNN transfer: quality, deployment bytes and complete fit cost

Rows are means across the three completed seeds; fractional channel and term counts are averages. The practical ReLU control uses affine BatchNorm and storage-matched wider channels; tanh and CNN spline mechanisms use non-affine BatchNorm. Their comparison evaluates complete architectures, rather than isolating the activation alone. Only the exact recorded batch cohort is included below (default batch 128 or its seed-specific override). Complete/test counts expose partial official-test evaluation. Accuracy is percent. Extra terms mean residual pairs for tanh or hinges for spline; they are not projection counts. Zero extra terms means the retained checkpoint did not retain growth, even if discarded growth branches were trained.

| Dataset | Policy | Cohort | Complete; tested | Acc % | NLL | Inference B | Artifact B | Channels | Extra terms | Cold fit s | Test+replay s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fashion | Continue/refine tanh | discover | 3/3; test 3 | 91.41 | 0.23965 | 135,686 | 145,104 | 16.0/32.0/64.0 | 0.0 | 181.0 | 7.5 |
| fashion | Profile policy | discover | 3/3; test 3 | 91.41 | 0.23965 | 135,686 | 145,104 | 16.0/32.0/64.0 | 0.0 | 228.6 | 7.3 |
| fashion | Projection policy | discover | 3/3; test 3 | 91.41 | 0.23965 | 135,686 | 145,104 | 16.0/32.0/64.0 | 0.0 | 188.4 | 7.4 |
| fashion | Joint selector | discover | 3/3; test 3 | 91.41 | 0.23965 | 135,686 | 145,104 | 16.0/32.0/64.0 | 0.0 | 164.6 | 7.5 |
| fashion | Random policy | discover | 3/3; test 3 | 91.41 | 0.23965 | 135,686 | 145,104 | 16.0/32.0/64.0 | 0.0 | 188.3 | 7.4 |
| fashion | Spline policy | discover | 3/3; test 3 | 91.89 | 0.23053 | 136,584 | 145,192 | 16.0/32.0/64.0 | 37.3 | 256.5 | 7.4 |
| fashion | Ordinary ReLU | discover | 3/3; test 3 | 91.75 | 0.22853 | 146,513 | 152,781 | 17.0/34.0/66.0 | 0.0 | 129.8 | 7.3 |
| cifar10 | Continue/refine tanh | discover | 3/3; test 3 | 72.18 | 0.79910 | 136,854 | 146,811 | 16.0/32.0/64.0 | 0.0 | 110.9 | 9.6 |
| cifar10 | Profile policy | discover | 3/3; test 3 | 72.24 | 0.79850 | 138,945 | 148,902 | 16.0/32.0/64.0 | 74.7 | 151.1 | 9.7 |
| cifar10 | Projection policy | discover | 3/3; test 3 | 72.18 | 0.79898 | 138,871 | 148,828 | 16.3/32.3/64.3 | 0.0 | 119.5 | 9.3 |
| cifar10 | Joint selector | discover | 3/3; test 3 | 72.22 | 0.79836 | 140,962 | 150,919 | 16.3/32.3/64.3 | 74.7 | 160.9 | 9.8 |
| cifar10 | Random policy | discover | 3/3; test 3 | 72.21 | 0.79885 | 139,917 | 149,874 | 16.3/32.3/64.3 | 37.3 | 130.7 | 9.3 |
| cifar10 | Spline policy | discover | 3/3; test 3 | 77.31 | 0.65665 | 138,648 | 147,795 | 16.0/32.0/64.0 | 74.7 | 164.8 | 9.2 |
| cifar10 | Ordinary ReLU | discover | 3/3; test 3 | 77.57 | 0.64662 | 147,753 | 154,560 | 17.0/34.0/66.0 | 0.0 | 109.3 | 9.0 |

Completed discovery retained-event actions: {'continue': 49, 'refine': 48, 'profile': 8, 'projection': 3}. CNN fits labeled optimization-unresolved: 37. Profile candidate applied-term range across recorded events: 112–112. Exact per-seed rows, hashes and gate statistics remain in [report.json](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/report.json) and [acceptance-audit.json](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/acceptance-audit.json).

## What the CNN results do and do not diagnose

- The evaluated profile action is one ranked bundle, often opening a pair on **all 112 existing channels**. It does not compare every selective subset. Rejection of that bundle does not prove every individual profile edit is unhelpful.

- `REFINE` both rotates a filter and resets that filter slice's Adam moments/ages. With no reset-only control, gains cannot be attributed purely to geometric rotation.

- Growth policy names do not establish that growth survived. Identical retained model hashes can arise when policies reject their added-capacity branches and choose the same continuation/refinement path.

- Rejected CNN branches do not retain all trained width/center/saturation trajectories. Their short-trial failure does not establish a universal dead-gradient explanation.

- Profile score residualization uses a per-channel bias/gain approximation. Finite validation, not that local heuristic, is the action decision; it is not a population certificate.

- A zero outgoing coefficient initially blocks upstream new-direction gradients; subsequent learning horizon and matched optimizer handling matter. A positive representational possibility is not a convergence guarantee.

- All CNN evidence is limited to this small 16/32/64 architecture, the registered data/augmentation/normalization and the 8% payload ceiling. It cannot establish a default activation for arbitrary deep networks.

## Cost, deployment and assurance limitations

Cold fit process costs include process startup, proposals, discarded forks, training and export; test/replay are separately shown. Controlled branch fit times omit shared proposal/baseline/search work and cannot be treated as complete independent-policy costs. Complete case cost is retained in the original results.

External CPU contention was recorded during fitting; observed fit wall times are complete costs under those conditions, not isolated algorithmic speed comparisons. Inference timing is a separate protocol. Consult [latency_results.csv](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/latency_results.csv) and the full report for device/batch/peak-allocation results; normalized-input timing excludes preprocessing arithmetic. Tiny export NPZ overhead is distinct from inference payload.

A passed replay establishes the checked artifact/numerical contracts. It does not fix an incomplete cohort, unequal candidate search, missing reset-only control or an unmeasured population approximation floor. A failed or incomplete independent audit blocks promotion even if a pooled mean looks favorable.

## Supported continuation

Keep the positive direction narrow: expensive useful input directions plus genuinely complex scalar responses. Test activation families not used to generate the target, optimize conventional spline/PPR dictionaries adequately, compare matched long-horizon learning, and let selective bundles compete with channel growth. Use horizon adaptation and reset-only rotation controls before treating the current selector as reliable.

The practical output should remain `PROFILE`, `REFINE`, `PROJECTION`, or `CONTINUE`, with separate `INCONCLUSIVE` and `BUDGET_INELIGIBLE` reasons. Mathematical information certificates require a known field and independent true-target error assumptions; ordinary learned derivatives do not supply them.

## Detailed evidence and figures

[Decision synthesis](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/decision-theorem.md); [Conditional decision proofs](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/decision-conditions.md); [Approximation/storage budget](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/approximation-budget.md); [Cone-opening failure analysis](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/cone-opening.md); [Cancellation and ordinary witness](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/cluster-cancellation.md); [Finite-support capacity](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/finite-support-capacity.md); [Controlled independent audit](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/synthetic-audit.md); [Controlled interpretation](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/final-synthetic-interpretation.md); [CNN contract audit](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/cnn-contract-audit.md); [Fashion validation audit](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/fashion-validation-audit.md); [CIFAR validation audit](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/cifar-validation-audit.md); [Acceptance audit](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/acceptance-audit.json).

[allocation heatmap](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/allocation_heatmap.png); [fashion learning curves](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/fashion_learning_curves.png); [cifar10 learning curves](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cifar10_learning_curves.png); [fashion paired mistakes](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/fashion_paired_mistakes.png); [cifar10 paired mistakes](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cifar10_paired_mistakes.png).

## Exact reproduction

```bash
cd /home/vardan/dev/projects/omnibias_projects/omnibias-lab
/tmp/omnibias-frontier-env-20261021/bin/python -m projection_profile_research.run \
  --stage all --device cuda --synthetic-device cpu --output <fresh-directory>
/tmp/omnibias-frontier-env-20261021/bin/python -m projection_profile_research.run \
  --replay --device cuda --output /home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02
/tmp/omnibias-frontier-env-20261021/bin/python /home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/source/acceptance_audit.py --output /home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02
/tmp/omnibias-frontier-env-20261021/bin/python /home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/source/write_assessment.py --output /home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02
```

Fresh reproduction must use matching source/dependency/data fingerprints. Frozen-source replay and optimizer/RNG checkpoint resume are different contracts; do not resume a mismatched directory.

## Contribution boundary

Projection/profile modeling and the knots-versus-directions tradeoff have direct precedent in [MASA (1983)](https://epubs.siam.org/doi/10.1137/0904023). The population decompositions, approximation/rank/energy bounds and concentration statements are established mathematics. The present contribution is reproducible resource-aware allocation engineering and bounded evidence about when it succeeds or fails. This run does not establish a generic/default-activation advantage or a new all-purpose deep-learning theory.

## Assessment provenance

- `report.json` SHA-256 `c8fa7c0759c37a3a8856ba0e618ab5f599c03e5a08b87ea032c03761d919edc0`
- `replay.json` SHA-256 `192a8e2f1efdd0de5f00aae3929573079e9552ac3dfc6d822ed9642c05520120`
- `analysis/acceptance-audit.json` SHA-256 `ff1c44ade804c2a85020c4a2a758c98ef7c706408b46220ea612543264c619af`

- Writer SHA-256 `01742627fb42547a839904349ac4c256dcdadbd0abab32a438da7c4718ad9642`. Numeric tables are derived at execution from retained JSON/header metadata; diagnostic prose is not used to override the gates.


## Storage, runtime and actual learned shapes

These timings use the completed three-band, d=64/n=2048 exports. Values below are the mean of three seeds' per-model medians of three timed blocks, in milliseconds. The models have different achieved errors; this is descriptive deployment evidence, not a matched-accuracy speed comparison. Inputs are float32; CPU uses one thread; CUDA blocks are synchronized.

| Device | Batch | Profile, 405 B | Fresh ordinary R2, 556 B | Tanh + hinges, 396 B |
|---|---:|---:|---:|---:|
| cpu | 1 | 0.04444 | 0.01240 | 0.03100 |
| cpu | 128 | 0.05635 | 0.01674 | 0.04001 |
| cuda | 1 | 0.12902 | 0.04632 | 0.10044 |
| cuda | 128 | 0.13424 | 0.04931 | 0.10201 |

The compact-array evaluator evaluates only exported active terms for every control. The frozen research module also loops over dormant slots and incurs additional Python/GPU synchronization overhead. Both implementations passed independent NumPy and mutual forward checks before timing. [All 144 measurements, setup costs and memory results](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/synthetic_latency/REPORT.md) and [raw CSV](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/synthetic_latency/measurements.csv) retain that distinction. The profile uses nineteen tanh responses rather than R2's two. Its storage saving comes with more activation work in this case.

Nineteen CNN exports were timed: twelve primary representatives plus five unique grown exports and two matched same-seed baselines. [CNN measurements](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/latency_results.csv) include CPU/GPU batch1/128, with peak allocation in per-case JSON. Inference storage includes preprocessing arrays; normalized-input runtime excludes preprocessing arithmetic. Batch-averaged blocks do not estimate request-tail latency.

The retained CIFAR-8111 joint/profile export has 224 pairs across 112 channels. Nine channels (8.04%) have both derivative signs on the declared [-4,4] grid. The random-policy export with 112 pairs has none. Joint and profile are byte-identical here, so those two files are not independent replications. Most inspected responses remain monotone on this grid; small shape changes do not establish band-based reasoning. Feature occupancy and global monotonicity were not certified.

![Example learned CNN responses and derivatives](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/profile-shapes/cifar10_8111_joint_b128.png)

[Every-channel shape and gap analysis](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/profile-shapes/report.md) validates the exported formula and derivatives. [Within-event diagnostics](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/cnn_candidate_diagnostics/REPORT.md) compare216 candidates against their own matched 200-update continuation branches. Forty-five improve the training probe while worsening acceptance validation, and78 improve neither at that horizon. These dependent, post-hoc observations distinguish candidate failures without proving a universal cause.

## Conditional capacity allocation: a mathematical answer and a practical rule

- Written UTC: 2026-10-22T21:43:59.791028+00:00.
- Frozen campaign source SHA-256: `f552253d95e3e6a9112302256704fff13cbd31806e1a0e470084f945ac75da53`.
- Status: synthesis of established identities and proved conditional constructions. This note adds no experiment, changes no frozen recipe and makes no claim of a new general theory or superior allocation algorithm.

**Spend on a profile when the useful scores are already adequate and their scalar responses are the economical bottleneck. Refine directions when a better orientation suffices. Add projections when they expose missing information or make a missing interaction economical. Continue without growth when its gain is unsupported or a candidate is unaffordable; unresolved optimization makes the allocation provisional and requires a continuation control.** Neither angle, training error nor a local tangent score alone decides this.

### 1. The population theorem: three structural terms, then finite learning

Let \(Y\in L^2\), \(m(X)=\mathbb E[Y\mid X]\), \(Z=W^\top X\), and let \(\mathcal G_{W,K}\subset L^2(Z)\) be the permitted response/head class. For a fixed score map and nonempty class,

\[
\inf_{g\in\mathcal G_{W,K}}\mathbb E(Y-g(Z))^2
=\underbrace{\mathbb E\operatorname{Var}(Y\mid X)}_{N:\ \text{noise}}
+\underbrace{\mathbb E\operatorname{Var}(m(X)\mid Z)}_{I_W:\ \text{lost information}}
+\underbrace{\inf_{g\in\mathcal G_{W,K}}\mathbb E(\mathbb E[m\mid Z]-g(Z))^2}_{A_{W,K}:\ \text{response/head restriction}}.
\]

Proof: write \(Y-g=(Y-m)+(m-\mathbb E[m\mid Z])+(\mathbb E[m\mid Z]-g)\); the three terms are pairwise orthogonal by conditional expectation. Infimize only the last term. For a trained function \(\widehat g\) add its nonnegative excess above this class infimum. That excess includes finite-data estimation, regularization and imperfect optimization; it must not be attributed solely to optimizer failure. Pure empirical optimization error is defined relative to the infimum of its own empirical objective.

**Consequences.** Profile-only changes at fixed W cannot reduce \(I_W\). Zero \(I_W\) does not establish small \(A_{W,K}\): additive responses can miss interactions even when scores retain all inputs. A changed W can reduce information loss without increasing projection count. An additional direction can help even when it is a combination of existing scores, because it may represent an interaction more cheaply. Noise does not require additional capacity. These are conditional statements, not an oracle for estimating the terms from one finite dataset.

For finite-class classification, let \(p_X=P(Y=\cdot\mid X)\), \(p_Z=\mathbb E[p_X\mid Z]\), and \(q(Z)\) be predicted probabilities, positive wherever needed. Then

\[
\mathbb E[-\log q_Y(Z)]
=H(Y\mid X)
+\underbrace{\mathbb E\operatorname{KL}(p_X\Vert p_Z)}_{I(Y;X\mid Z)}
+\mathbb E\operatorname{KL}(p_Z\Vert q(Z)).
\]

This follows by adding and subtracting \(\log p_{X,Y}\) and \(\log p_{Z,Y}\), then conditioning on X or Z. Infimizing the last term over the response class gives the cross-entropy counterpart. It does not imply that training on a chosen metric finds its population optimum. The scalar squared-error energy bounds below do not become classification-accuracy bounds without further assumptions.

### 2. Known-field information certificates and their limits

For \(X\sim N(0,I_d)\), let P be the orthoprojector onto the retained score span. For a known scalar field with the required Gaussian Sobolev integrability, define

\[
G=\mathbb E\|(I-P)\nabla m\|^2,
\qquad B=\mathbb E\|(I-P)\nabla^2m(I-P)\|_F^2.
\]

Normal Hermite degrees give

\[
\frac{G^2}{G+B}\le I_W\le G,
\qquad \frac GD\le I_W\quad\text{if every normal degree is at most }D.
\]

Use zero lower bound when \(G+B=0\). The first upper bound requires the normal H1 domain; the second-energy lower bound requires finite normal H2 energy; the degree bound requires an actual finite-degree restriction. Proof: normal Hermite degree k contributes \(a^2,ka^2,k(k-1)a^2\) to \(I_W,G,B\); apply Cauchy–Schwarz and \(1\le k\le D\). Rotation is valid here because the standard Gaussian law is rotation invariant.

If an independently verified surrogate satisfies \(\|\widehat m-m\|_2\le\delta\) and rigorous energy calculations bound \(\widehat L\le I_W(\widehat m)\le\widehat U\), contraction of conditional centering gives

\[
(\sqrt{\widehat L}-\delta)_+^2\le I_W(m)
\le(\sqrt{\widehat U}+\delta)^2.
\]

A certified lower bound above the allowed latent-error tolerance rules out all profile-only repairs to the retained scores. A small upper bound merely permits them; it does not prove that the permitted profiles or optimizer attain that error. Derivatives of a learned model, sampled energy estimates, or low training residuals are not by themselves a true-target certificate. A bounded-box uniform field error controls L2 only for a distribution supported there; a Gaussian application additionally needs tail control. The [Gaussian derivation](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/projection_profile_research/DERIVATIVE_ENERGY.md) and [product-uniform extension](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/spectral-extension.md) spell out domains and counterexamples.

Raw derivative-energy reduction is not the actual value of a new score: for \(m=X_2X_3\), retaining only \(X_1\), adding \(X_2\) reduces normal derivative energy but leaves conditional mean zero and information loss unchanged. A proposal must expose a useful conditional response, not merely a large gradient component.

For the finite-degree bound, require D >= 1. Degree zero is handled separately by G = I = 0, without dividing by D.

### 3. Other resource constraints that the information floor does not see

**Response complexity.** For independent centered bounded scores and known additive C2 profiles, K-segment linear splines give a conditional bound envelope \(\sum_j C_j/K_j^4\). With per-segment storage \(c_j\), its fixed-direction continuous optimum has \(K_j\propto(C_j/c_j)^{1/5}\), after paying complete direction setup costs. Adding an omitted independent component trades its variance against its own approximation error and direction cost. This is an established spline approximation calculation, not a tanh rate or an exact trained-error rule. Dependence, interactions and unknown constants invalidate the simple separable allocation. See [the full budget argument](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/approximation-budget.md).

**Outgoing response rank.** For an exact vector profile \(F(z)=C\psi(z)+d\), with components of \(\psi\) linearly independent modulo constants on an interval, the minimum number of scalar response channels followed by a **linear** readout is \(\operatorname{rank}C\). Rank factorization attains it, and the linear span of centered outputs proves necessity. Distinct common-slope sigmoid/tanh shifts satisfy the needed independence. Sharing one input direction therefore need not eliminate many outgoing coefficients. This is a linear-readout statement; a nonlinear head is a different architecture. For probabilities, common logit shifts are unobservable, so raw-logit rank need not be the probability-preserving minimum. See [the output-rank analysis](/home/vardan/dev/projects/omnibias/docs/activation-profile-compression.md).

**Cancellation and margin.** Same-sign nearby directions can have a cheap ordinary-unit witness: the 10-degree tanh pair admits population MSE at most \(1.026\times10^{-4}\) with one ordinary unit. Opposite coefficients growing as inverse separation can instead retain transverse information as directions coincide. Small field error also need not preserve labels when margins shrink. Direction angle alone is insufficient; account for coefficient norms, cancellation and the actual objective. See [cluster and cancellation analysis](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/cluster-cancellation.md) and its [read-through audit](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/derivation-review.md).

**Finite support.** An injective scalar projection of a finite Boolean cube preserves every bit, but arbitrary truth functions can still require exponentially many common-slope profile terms. Structured counting/parity functions are much cheaper. See [the finite-support capacity argument](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/finite-support-capacity.md). Information sufficiency is not cheap representability.

### 4. What can be measured, and what remains oracle information?

| Quantity | Available evidence | What it does not establish |
|---|---|---|
| Exported bytes, adjoining channel weights, normalization state | Exact inference-array/topology accounting | Latency or learning quality |
| Paired finite-horizon validation loss | Estimate for the actually trained candidate list | Global optimum or long-horizon ranking |
| Learning curves, gradients and displacement | Optimization/parameterization diagnostics | Population noise or approximation floor |
| Regularized Jacobian gain with a transported parameter metric | Local training-objective prediction; cone constraints must be respected | Finite-step/generalization improvement |
| Gaussian or box energy brackets | Certificate for a known field with rigorous integration and domain assumptions | Unknown true target unless an independent field-error bound is supplied |
| True conditional mean, noise, best class error, omitted component variance | Analytic fixtures or otherwise separately justified knowledge | Generally identifiable from a small training set |
| Known-direction ordinary witness | Upper bound on representational error | A learned method achieving it |

The CNN per-channel bias/gain score is an explicitly approximate screening device. Finite action forks and acceptance validation must resolve its practical usefulness; calling it a full-network capacity certificate would be incorrect.

### 5. A practical four-action rule

1. **Check eligibility and unresolved training.** Count the full inference representation. Mark unaffordable proposals `BUDGET_INELIGIBLE`. Compare continuation from the current optimizer state; a still-improving ordinary model is an essential control.
2. **Construct from training data only.** Offer `REFINE` (bounded direction changes), `PROFILE` (function-preserving residual pairs), `PROJECTION` (new unit/channel with its adjoining weights), and `CONTINUE`. Include finite direction proposals because first-order scores can vanish at useful finite rotations. Avoid duplicated profile proposals and respect nonnegative width cones.
3. **Compare actual short fits fairly.** Fork the same model, optimizer, random state and minibatches. Train the shortlisted actions within a complete byte allowance; count discarded work. Use a fresh acceptance fold, not the official test set. Report that the chosen action won a finite-horizon comparison, not a globally optimal allocation problem.
4. **Select conservatively.** Prefer profile enrichment when an eligible profile wins independently and the current scores have no established information obstruction. Prefer refinement when it wins with unchanged storage. Prefer a new projection when it wins after full setup cost or when a valid information certificate rules out the current scores. If response/head interactions remain unsupported, report an architecture limitation rather than adding scalar terms indefinitely. If gains are uncertain, return `INCONCLUSIVE` about capacity need and keep the unchanged/continuing model; do not convert uncertainty into a claim that growth is useless.

An information obstruction alone may require changing orientation or adding a direction; it does not decide which. Zero information loss alone does not choose profiles over a more economical multi-directional representation. In every case finite validation, resource constraints and the permitted head remain decisive.

### 6. Optional future confidence control; not the registered one-SE heuristic

Suppose a fixed list of M candidate models is constructed independently of a fresh iid fold of size n. If each per-example loss lies in \([0,L]\), the paired improvement variable lies in \([-L,L]\). Hoeffding and a two-sided union bound give, simultaneously for all candidates with probability at least \(1-\eta\),

\[
\Delta_j\ge\widehat\Delta_j-L\sqrt{\frac{2\log(2M/\eta)}n}.
\]

Indeed \(P(|\widehat\Delta_j-\Delta_j|>\epsilon)\le2e^{-n\epsilon^2/(2L^2)}\); set \(\epsilon=L\sqrt{2\log(2M/\eta)/n}\). Thus a future rule could accept only improvements whose lower bound exceeds a declared meaningful gain. For sequential fresh folds, allocate \(\eta_t\) with \(\sum_t\eta_t\le\eta\) and require each candidate list to be independent of its current fold conditional on the past.

Unbounded squared error or unbounded cross-entropy does not satisfy this contract. For CE, probabilities bounded below by a declared positive epsilon give \(L=-\log\epsilon\); clipping the loss instead changes the certified objective. Reusing a fold to construct or repeatedly adapt candidates requires additional accounting. This is a conservative standard concentration argument, potentially inconclusive at the campaign's small folds. It was **not implemented** as a replacement for the registered one-standard-error rule and must not be presented as a guarantee of that heuristic. See [Hoeffding (1963)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500830).

### 7. Prior-art and contribution boundary

The closest conceptual precedent is [Friedman, Grosse and Stuetzle's MASA (1983)](https://epubs.siam.org/doi/10.1137/0904023), with its [author-hosted paper](https://sites.stat.washington.edu/wxs/Learning-papers/multidimensional-additive-spline.pdf). Pages 292–293 already address fitting directions and spline responses, inadequate knot complexity, repeated nearly identical directions and validation-based term selection. Thus identifying the projection-versus-profile tradeoff is not the new contribution. Function-preserving gradient-guided growth also has precedents such as [GradMax](https://research.google/pubs/gradmax-growing-neural-networks-using-gradient-information/).

The identities, spectral inequalities, rank conditions, Taylor bounds and concentration calculation are established mathematics. The campaign's possible contribution is narrower: a resource-aware finite-fork mechanism, its reproducible controls and supported decision/failure cases. A new competitive algorithm requires completed matched evidence. A broadly superior default activation, a general depth-collapse theorem or a universal capacity oracle has not been established by these derivations.


## Assurance, retained failures and contribution classification

The focused suite passed103 tests; CUDA insertion/gradient checks passed. Full campaign replay passed all60 controlled cases and42 CNN exports. The independent source/data/cohort/storage/gate audit passed with zero remaining defects, incomplete cohorts or false acceptance. Four protected historical inputs, including the paper, retain their registered hashes. Stable APIs and existing environments were preserved.

The first independent audit rejected signed-byte orientation arrays because its own whitelist omitted valid NPY dtype `|i1`. All840 initial errors had that cause. The checker was repaired with signed-byte acceptance and object-dtype rejection regressions; its11 helper checks and full rerun passed. The [failed audit and original checker](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/audit-dtype-failure) and [repair record](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/audit-dtype-repair.json) remain. A report-writer label mismatch initially undercounted unresolved CNN fits; the corrected count is37/42, with [original output and correction](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/reporting-repair) retained. These repairs changed the analysis tools and prose, not frozen training sources or numerical results.

The earlier tiny-ridge GPU run exhausted a fit cap and is retained in [execution adjustment](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/execution_adjustment.json). Controlled fitting then used CPU, while CNNs used the verified T1200 CUDA path. Unavailable official R/SMART/PPR comparators are recorded; local hinge/spline controls are labeled local implementations. No downloads or environment replacements were required.

| Candidate | Assessment |
|---|---|
| Noise/information/response decomposition, energy and rank bounds | Established mathematics, with assumptions and independent fixtures |
| Exact state-preserving insertion, byte accounting and replay | Tested research engineering |
| Structured scalar-profile storage/accuracy tradeoff | Supported finite-budget case; basis-matched targets and unfinished optimization limit generalization |
| Joint finite-fork CNN selector | Implemented architectural candidate; failed registered advancement gates |
| Reliable default multi-bias advantage or unrestricted depth replacement | Unsupported by this study |
| Potential mathematical novelty | No specific new result established by targeted prior-art review |

Closest prior work includes [PPR (1981)](https://doi.org/10.1080/01621459.1981.10477729), [MASA (1983)](https://epubs.siam.org/doi/10.1137/0904023), [GradMax](https://research.google/pubs/gradmax-growing-neural-networks-using-gradient-information/) and [splitting steepest descent](https://proceedings.neurips.cc/paper/2019/file/3a01fc0853ebeba94fde4d1cc6fb842a-Paper.pdf). MASA already discusses spending flexibility on spline knots versus directions. The narrower unresolved opportunity is an allocation procedure that reliably handles finite training horizon, selective channel bundles and complete deployment cost.

The strongest next test is **small selective bundles with adaptive trial length**, against long continuation, reset-only refinement controls and adequately optimized spline/PPR controls. The22/60 synthetic horizon reversals and112-channel CNN bundles directly motivate it. This is a continuation recommendation, not a newly confirmed recipe.

[Complete work/cost accounting](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/analysis/cost-summary.md) includes336,000 retained CNN updates,43,200 discarded fork updates, both initial and fresh replay costs, setup and deployment measurements. Fitting experienced unrelated CPU contention; wall-clock fit ratios do not establish algorithmic speed. The 24-hour allocations were maxima, not a claim of24 hours of measured computation. Seed uncertainty is conditional on these fixed datasets.

# Full registered tables and deployment records

Generated 2026-10-22T22:39:10.636514+00:00. Source fingerprint `f552253d95e3e6a9112302256704fff13cbd31806e1a0e470084f945ac75da53`.

The question is whether the next stored byte should improve an existing scalar response, retain another direction, refine existing directions, or remain unspent. The tables below contain only retained experiment artifacts; missing or incomplete cohorts cannot support superiority.

## Mathematical answer

For fixed retained scores, richer profiles can reduce response-class approximation error. Improving an information-losing representation requires changing the retained directions or adding directions. Additive profiles can also lack interactions even when the retained scores contain all target information. A new projection may therefore help without increasing their linear span. Noise and finite optimization must be distinguished from these representation limits.

The exact conditional-expectation decompositions, regularized linearized gain derivation, assumptions and counterexamples are in [THEORY.md](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/projection_profile_research/THEORY.md). The [derivative-energy supplement](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/projection_profile_research/DERIVATIVE_ENERGY.md) supplies exact Gaussian polynomial information-loss diagnostics, curvature lower bounds, rank conditions and explicit limits for unknown targets. These are established Gaussian analysis. The local CNN score is an explicitly approximate ranking device. Independent validation of finite training forks makes the actual allocation decision.

## Execution adjustment

{
  "status": "RESOURCE_EXECUTION_ADJUSTMENT",
  "utc": "2026-10-22T19:46:11.172174+00:00",
  "prior_attempt": "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_01",
  "prior_source_sha256": "8d703cf2d2a3df2801c6f4caf834bd4307ac3b0cc64d21916e6262c8d5e114fb",
  "original_synthetic_started_utc": "2026-10-22T19:42:20.230098+00:00",
  "reason": "Tiny CUDA ridge fits were dispatch-bound: first ordinary continuation completed 1000 updates in about14.10s, while profile reached571updates in15.02s. Incomplete CUDA attempt retained. All controlled methods move together to CPU; CNN remains CUDA. Seeds/models/updates/fitcap unchanged. Original4h controlled-stage deadline and24h campaign deadline preserved.",
  "old_partial_files": {
    "fingerprint.json": "53bc057fd923e1687b8f3f4d6858b79c7f6a3a6983631a56aa8863b78f572687",
    "theory.json": "434bddcbded814064e5e2ffe428ba777468445e9156f28d2717e93a030a4c50b",
    "checks.json": "e85222a559056e6c568cc77044377ce646af9e0ab068fff0c7f0194c6d30441a",
    "registration.json": "92956ed66cac64a2495776b6e42c953fdf3886840d23e5423369788a4d8dc08d",
    "campaign.json": "2c2c53a313a25d49ded3b24407514496d8bb0832b36bad1bc5d83d8bcc4e68f1",
    "synthetic/contract.json": "7fe055b30d373301733406fedc4728d0547ba844f0480ec065d0ca95fff289e4",
    "synthetic/cases/ordinary_d8_n512_s8101/continue_short.npz": "7b05e4dd0821f9863cc488f7c3956a73710cd5b7321b001ebdd077bce71328ec",
    "synthetic/cases/ordinary_d8_n512_s8101/continue_checkpoint.pt": "01ab03bbe8c9b8adeb5ac01128f6921dbf1807a0c50d374ad16bfaad4b195122",
    "synthetic/cases/ordinary_d8_n512_s8101/profile_checkpoint.pt": "d01855548f116a2b4980d400085b61da264de89c8acdbc529d87079d56b38387",
    "synthetic/cases/ordinary_d8_n512_s8101/baseline_checkpoint.pt": "3db081292ca735f28db83158d5c54a3499b64ad3cda280175c6609fd7e969361",
    "synthetic/cases/ordinary_d8_n512_s8101/continue_long.npz": "e91ece3eb739b6147d3c590f1d462ea8dfe64bd14416a428f33a546bf0033d35",
    "synthetic/cases/ordinary_d8_n512_s8101/data.npz": "7ae4c14434c5d7dffe5996ea7b9b7adcfae2c04d99035d303656d690fd1338c3",
    "synthetic/cases/ordinary_d8_n512_s8101/refine_partial.npz": "82821aa8f4d2263fc6bf606f508b5ca9bd373745bbfb6d5dfbae626df5805970",
    "synthetic/cases/ordinary_d8_n512_s8101/projection_partial.npz": "de16235781bf0a68c9ad6d483fed39480b030b8fa1d8a054b292cc4057f27a20",
    "synthetic/cases/ordinary_d8_n512_s8101/refine_short.npz": "7b05e4dd0821f9863cc488f7c3956a73710cd5b7321b001ebdd077bce71328ec",
    "synthetic/cases/ordinary_d8_n512_s8101/baseline.npz": "0fc0610bef033bb2e3e09227bc0eb59ea07165b9b044e9fbcdf65ec72a1a2035",
    "synthetic/cases/ordinary_d8_n512_s8101/spline_short.npz": "181210be9fea6d6ab2a26812ce335bf5979fa84d7e89ec65511fc743cf498d14",
    "synthetic/cases/ordinary_d8_n512_s8101/partial.json": "50188b03dd3472321a88dceac7b5fb4e3f93184c606a2e315dc6942af41bfc95",
    "synthetic/cases/ordinary_d8_n512_s8101/projection_short.npz": "1b03db713e0a12b9c3a29f47cf469d80f0ba97678b26c49ccadb394c8e3b3255",
    "synthetic/cases/ordinary_d8_n512_s8101/spline_checkpoint.pt": "0beae66aac74d63adb621ed0316dc0750ae3c0c7d58fe1abd7a340489694009d",
    "synthetic/cases/ordinary_d8_n512_s8101/fixed_R1_K1_checkpoint.pt": "5e252ad458104dc6c5f94e68672eb5f547e51381b9b6ca022931cc30cb2eb605",
    "synthetic/cases/ordinary_d8_n512_s8101/configuration.json": "4e24ca7b62b8a5e018f9c462a00a726accfe6ba8a629e81c6cec632a42ea8c38",
    "synthetic/cases/ordinary_d8_n512_s8101/profile_partial.npz": "0695d2c863e761e06eee3cde825db34ae5b1edcd4e027fcaa0f31f7800094d84",
    "synthetic/cases/ordinary_d8_n512_s8101/profile_short.npz": "1c034356d631aec220cb349b0c614f21b5d6058c0cfea08cafbe811a8ca2a1af",
    "synthetic/cases/ordinary_d8_n512_s8101/fixed_R1_K1_partial.npz": "52fb5a20da70cb8121b7b3015f0730a5aaafd0ccc46193faa73a0645754cfbaf",
    "synthetic/cases/ordinary_d8_n512_s8101/fits_partial.json": "0ab97c350d0a962c13dbe20c92339130dac1a0156d97479e5dc9dfee983db247",
    "synthetic/cases/ordinary_d8_n512_s8101/spline_partial.npz": "79db085897230b59be2999c002dd8c00180908b3a3706c98fc11deda69b4981c",
    "synthetic/cases/ordinary_d8_n512_s8101/refine_checkpoint.pt": "ae79f7c5f6806e8081097106dd557f4a3b5bc9888dcebef69a46581d40fde72c",
    "synthetic/cases/ordinary_d8_n512_s8101/projection_checkpoint.pt": "7254aa9a8003a477461fd543fec072c198f7f5dfd4624ef5493ca4bb638b8c7e",
    "synthetic/cases/ordinary_d8_n512_s8101/fixed_R1_K3_checkpoint.pt": "d01755b9aaabe3a160fa14de77614593df5fcd6f9a620cbdc94467d62032b955"
  }
}

## Controlled results

Cases retained/planned: 60/60; statuses: {'COMPLETE': 60}. Joint selected actions: {'profile': 14, 'refine': 7, 'projection': 33, 'continue': 6}. Short/long best-action reversals: 22/60.

| Regime | Dimension | Train n | Policy | Seeds | Latent MSE | Observed MSE | Inference bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hermite4 | 8 | 512 | fixed_grid | 3 | 0.753807 | 0.756005 | 108 |
| hermite4 | 8 | 512 | joint | 3 | 0.640476 | 0.642612 | 103 |
| hermite4 | 8 | 512 | known_direction_oracle | 3 | 0.672855 | 0.67498 | 103 |
| hermite4 | 8 | 512 | no_growth_refine | 3 | 0.920595 | 0.922668 | 64 |
| hermite4 | 8 | 512 | profile_only | 3 | 0.640476 | 0.642612 | 103 |
| hermite4 | 8 | 512 | projection_only | 3 | 0.862069 | 0.864159 | 108 |
| hermite4 | 8 | 512 | random | 3 | 0.640476 | 0.642612 | 103 |
| hermite4 | 8 | 512 | spline | 3 | 0.802905 | 0.805022 | 100 |
| interaction | 8 | 512 | fixed_grid | 3 | 0.405623 | 0.408226 | 152 |
| interaction | 8 | 512 | joint | 3 | 0.750555 | 0.752769 | 108 |
| interaction | 8 | 512 | known_direction_oracle | 3 | 0.737074 | 0.73917 | 152 |
| interaction | 8 | 512 | no_growth_refine | 3 | 0.750555 | 0.752769 | 108 |
| interaction | 8 | 512 | profile_only | 3 | 0.750555 | 0.752769 | 108 |
| interaction | 8 | 512 | projection_only | 3 | 0.750555 | 0.752769 | 108 |
| interaction | 8 | 512 | random | 3 | 0.750555 | 0.752769 | 108 |
| interaction | 8 | 512 | spline | 3 | 0.703294 | 0.7057 | 120 |
| interaction | 64 | 128 | fixed_grid | 3 | 2.14803 | 2.15339 | 556 |
| interaction | 64 | 128 | joint | 3 | 2.13446 | 2.13971 | 556 |
| interaction | 64 | 128 | known_direction_oracle | 3 | 1.91282 | 1.91798 | 824 |
| interaction | 64 | 128 | no_growth_refine | 3 | 2.13446 | 2.13971 | 556 |
| interaction | 64 | 128 | profile_only | 3 | 2.13446 | 2.13971 | 556 |
| interaction | 64 | 128 | projection_only | 3 | 2.13446 | 2.13971 | 556 |
| interaction | 64 | 128 | random | 3 | 2.13446 | 2.13971 | 556 |
| interaction | 64 | 128 | spline | 3 | 2.14785 | 2.1533 | 628 |
| interaction | 64 | 2048 | fixed_grid | 3 | 0.381009 | 0.382821 | 824 |
| interaction | 64 | 2048 | joint | 3 | 0.371757 | 0.373539 | 824 |
| interaction | 64 | 2048 | known_direction_oracle | 3 | 0.735676 | 0.737539 | 824 |
| interaction | 64 | 2048 | no_growth_refine | 3 | 0.698184 | 0.700105 | 556 |
| interaction | 64 | 2048 | profile_only | 3 | 0.59123 | 0.593894 | 790 |
| interaction | 64 | 2048 | projection_only | 3 | 0.371757 | 0.373539 | 824 |
| interaction | 64 | 2048 | random | 3 | 0.59123 | 0.593894 | 790 |
| interaction | 64 | 2048 | spline | 3 | 0.547061 | 0.549428 | 772 |
| misaligned_30 | 8 | 512 | fixed_grid | 3 | 0.0287878 | 0.031364 | 108 |
| misaligned_30 | 8 | 512 | joint | 3 | 0.0333931 | 0.0358816 | 106 |
| misaligned_30 | 8 | 512 | known_direction_oracle | 3 | 0.0316316 | 0.0339877 | 103 |
| misaligned_30 | 8 | 512 | no_growth_refine | 3 | 0.153195 | 0.155707 | 64 |
| misaligned_30 | 8 | 512 | profile_only | 3 | 0.0460785 | 0.0484483 | 103 |
| misaligned_30 | 8 | 512 | projection_only | 3 | 0.0348642 | 0.0374184 | 108 |
| misaligned_30 | 8 | 512 | random | 3 | 0.0460785 | 0.0484483 | 103 |
| misaligned_30 | 8 | 512 | spline | 3 | 0.0293772 | 0.0318887 | 100 |
| near_collinear | 8 | 512 | fixed_grid | 3 | 0.00054964 | 0.00308565 | 108 |
| near_collinear | 8 | 512 | joint | 3 | 0.000729394 | 0.00328908 | 108 |
| near_collinear | 8 | 512 | known_direction_oracle | 3 | 0.000115895 | 0.00263831 | 108 |
| near_collinear | 8 | 512 | no_growth_refine | 3 | 0.00219356 | 0.00478926 | 64 |
| near_collinear | 8 | 512 | profile_only | 3 | 0.00168947 | 0.00427683 | 77 |
| near_collinear | 8 | 512 | projection_only | 3 | 0.000729394 | 0.00328908 | 108 |
| near_collinear | 8 | 512 | random | 3 | 0.00168947 | 0.00427683 | 77 |
| near_collinear | 8 | 512 | spline | 3 | 0.00105925 | 0.00363352 | 100 |
| near_collinear | 64 | 128 | fixed_grid | 3 | 0.0122959 | 0.0148548 | 292 |
| near_collinear | 64 | 128 | joint | 3 | 0.00519181 | 0.00774177 | 405 |
| near_collinear | 64 | 128 | known_direction_oracle | 3 | 0.0148571 | 0.0174067 | 556 |
| near_collinear | 64 | 128 | no_growth_refine | 3 | 0.0127042 | 0.0152724 | 288 |
| near_collinear | 64 | 128 | profile_only | 3 | 0.00519181 | 0.00774177 | 405 |
| near_collinear | 64 | 128 | projection_only | 3 | 0.0201062 | 0.0228343 | 467 |
| near_collinear | 64 | 128 | random | 3 | 0.00519181 | 0.00774177 | 405 |
| near_collinear | 64 | 128 | spline | 3 | 0.00554886 | 0.00814608 | 396 |
| near_collinear | 64 | 2048 | fixed_grid | 3 | 0.000675375 | 0.00312625 | 556 |
| near_collinear | 64 | 2048 | joint | 3 | 0.00115826 | 0.00359499 | 556 |
| near_collinear | 64 | 2048 | known_direction_oracle | 3 | 0.00036711 | 0.00283684 | 556 |
| near_collinear | 64 | 2048 | no_growth_refine | 3 | 0.00218815 | 0.00459882 | 288 |
| near_collinear | 64 | 2048 | profile_only | 3 | 0.000855799 | 0.00335127 | 405 |
| near_collinear | 64 | 2048 | projection_only | 3 | 0.00115826 | 0.00359499 | 556 |
| near_collinear | 64 | 2048 | random | 3 | 0.000855799 | 0.00335127 | 405 |
| near_collinear | 64 | 2048 | spline | 3 | 0.000510885 | 0.00295684 | 396 |
| ordinary | 8 | 512 | fixed_grid | 3 | 5.29938e-05 | 0.00256933 | 68 |
| ordinary | 8 | 512 | joint | 3 | 6.99421e-05 | 0.00258952 | 64 |
| ordinary | 8 | 512 | known_direction_oracle | 3 | 5.51316e-05 | 0.00257154 | 103 |
| ordinary | 8 | 512 | no_growth_refine | 3 | 6.99421e-05 | 0.00258952 | 64 |
| ordinary | 8 | 512 | profile_only | 3 | 6.99421e-05 | 0.00258952 | 64 |
| ordinary | 8 | 512 | projection_only | 3 | 6.99421e-05 | 0.00258952 | 64 |
| ordinary | 8 | 512 | random | 3 | 6.99421e-05 | 0.00258952 | 64 |
| ordinary | 8 | 512 | spline | 3 | 6.99421e-05 | 0.00258952 | 64 |
| ordinary_noisy | 8 | 512 | fixed_grid | 3 | 0.00438546 | 0.25636 | 68 |
| ordinary_noisy | 8 | 512 | joint | 3 | 0.00408572 | 0.256363 | 77 |
| ordinary_noisy | 8 | 512 | known_direction_oracle | 3 | 0.00438216 | 0.256371 | 103 |
| ordinary_noisy | 8 | 512 | no_growth_refine | 3 | 0.00399159 | 0.256188 | 64 |
| ordinary_noisy | 8 | 512 | profile_only | 3 | 0.00408572 | 0.256363 | 77 |
| ordinary_noisy | 8 | 512 | projection_only | 3 | 0.00399159 | 0.256188 | 64 |
| ordinary_noisy | 8 | 512 | random | 3 | 0.00408572 | 0.256363 | 77 |
| ordinary_noisy | 8 | 512 | spline | 3 | 0.00399159 | 0.256188 | 64 |
| orthogonal | 8 | 512 | fixed_grid | 3 | 0.133065 | 0.135613 | 98 |
| orthogonal | 8 | 512 | joint | 3 | 0.0656159 | 0.0680235 | 108 |
| orthogonal | 8 | 512 | known_direction_oracle | 3 | 0.0316316 | 0.0339877 | 103 |
| orthogonal | 8 | 512 | no_growth_refine | 3 | 0.155455 | 0.158017 | 64 |
| orthogonal | 8 | 512 | profile_only | 3 | 0.152353 | 0.154901 | 90 |
| orthogonal | 8 | 512 | projection_only | 3 | 0.0656159 | 0.0680235 | 108 |
| orthogonal | 8 | 512 | random | 3 | 0.152353 | 0.154901 | 90 |
| orthogonal | 8 | 512 | spline | 3 | 0.155455 | 0.158017 | 64 |
| rotated_square | 8 | 512 | fixed_grid | 3 | 0.241588 | 0.244671 | 108 |
| rotated_square | 8 | 512 | joint | 3 | 0.0517512 | 0.0545703 | 93 |
| rotated_square | 8 | 512 | known_direction_oracle | 3 | 0.00013022 | 0.00265024 | 103 |
| rotated_square | 8 | 512 | no_growth_refine | 3 | 0.104338 | 0.107073 | 64 |
| rotated_square | 8 | 512 | profile_only | 3 | 0.104338 | 0.107073 | 64 |
| rotated_square | 8 | 512 | projection_only | 3 | 0.0517512 | 0.0545703 | 93 |
| rotated_square | 8 | 512 | random | 3 | 0.104338 | 0.107073 | 64 |
| rotated_square | 8 | 512 | spline | 3 | 0.104338 | 0.107073 | 64 |
| single_band | 8 | 512 | fixed_grid | 3 | 0.0227201 | 0.025299 | 108 |
| single_band | 8 | 512 | joint | 3 | 0.0233924 | 0.025995 | 108 |
| single_band | 8 | 512 | known_direction_oracle | 3 | 0.0316316 | 0.0339877 | 103 |
| single_band | 8 | 512 | no_growth_refine | 3 | 0.151352 | 0.153843 | 64 |
| single_band | 8 | 512 | profile_only | 3 | 0.0311021 | 0.0335012 | 103 |
| single_band | 8 | 512 | projection_only | 3 | 0.0233924 | 0.025995 | 108 |
| single_band | 8 | 512 | random | 3 | 0.0311021 | 0.0335012 | 103 |
| single_band | 8 | 512 | spline | 3 | 0.0298341 | 0.0323601 | 100 |
| three_bands | 8 | 512 | fixed_grid | 3 | 0.0407474 | 0.0433696 | 108 |
| three_bands | 8 | 512 | joint | 3 | 0.0428208 | 0.0454538 | 108 |
| three_bands | 8 | 512 | known_direction_oracle | 3 | 0.0457051 | 0.0480466 | 103 |
| three_bands | 8 | 512 | no_growth_refine | 3 | 0.253501 | 0.255947 | 64 |
| three_bands | 8 | 512 | profile_only | 3 | 0.0244713 | 0.0270562 | 103 |
| three_bands | 8 | 512 | projection_only | 3 | 0.0428208 | 0.0454538 | 108 |
| three_bands | 8 | 512 | random | 3 | 0.0244713 | 0.0270562 | 103 |
| three_bands | 8 | 512 | spline | 3 | 0.0713109 | 0.0737659 | 100 |
| three_bands | 64 | 128 | fixed_grid | 3 | 0.449979 | 0.452813 | 292 |
| three_bands | 64 | 128 | joint | 3 | 0.449124 | 0.452481 | 405 |
| three_bands | 64 | 128 | known_direction_oracle | 3 | 0.238936 | 0.242029 | 405 |
| three_bands | 64 | 128 | no_growth_refine | 3 | 0.462442 | 0.465285 | 288 |
| three_bands | 64 | 128 | profile_only | 3 | 0.449124 | 0.452481 | 405 |
| three_bands | 64 | 128 | projection_only | 3 | 0.522068 | 0.524471 | 556 |
| three_bands | 64 | 128 | random | 3 | 0.449124 | 0.452481 | 405 |
| three_bands | 64 | 128 | spline | 3 | 0.400681 | 0.403616 | 396 |
| three_bands | 64 | 2048 | fixed_grid | 3 | 0.0439964 | 0.0467549 | 556 |
| three_bands | 64 | 2048 | joint | 3 | 0.0146245 | 0.0171161 | 405 |
| three_bands | 64 | 2048 | known_direction_oracle | 3 | 0.0142864 | 0.0167787 | 405 |
| three_bands | 64 | 2048 | no_growth_refine | 3 | 0.256933 | 0.260191 | 288 |
| three_bands | 64 | 2048 | profile_only | 3 | 0.0146245 | 0.0171161 | 405 |
| three_bands | 64 | 2048 | projection_only | 3 | 0.0435224 | 0.0462935 | 556 |
| three_bands | 64 | 2048 | random | 3 | 0.0146245 | 0.0171161 | 405 |
| three_bands | 64 | 2048 | spline | 3 | 0.0374457 | 0.0401462 | 396 |
| two_equal | 8 | 512 | fixed_grid | 3 | 0.00482313 | 0.00725933 | 108 |
| two_equal | 8 | 512 | joint | 3 | 0.0120037 | 0.014377 | 108 |
| two_equal | 8 | 512 | known_direction_oracle | 3 | 0.000118978 | 0.00262412 | 108 |
| two_equal | 8 | 512 | no_growth_refine | 3 | 0.0503717 | 0.0527326 | 64 |
| two_equal | 8 | 512 | profile_only | 3 | 0.0503717 | 0.0527326 | 64 |
| two_equal | 8 | 512 | projection_only | 3 | 0.0120037 | 0.014377 | 108 |
| two_equal | 8 | 512 | random | 3 | 0.0503717 | 0.0527326 | 64 |
| two_equal | 8 | 512 | spline | 3 | 0.0503717 | 0.0527326 | 64 |
| weak_second | 8 | 512 | fixed_grid | 3 | 0.000110406 | 0.00261979 | 108 |
| weak_second | 8 | 512 | joint | 3 | 0.000135013 | 0.00264114 | 108 |
| weak_second | 8 | 512 | known_direction_oracle | 3 | 0.00286355 | 0.00537861 | 108 |
| weak_second | 8 | 512 | no_growth_refine | 3 | 0.00431297 | 0.00676336 | 64 |
| weak_second | 8 | 512 | profile_only | 3 | 0.00431297 | 0.00676336 | 64 |
| weak_second | 8 | 512 | projection_only | 3 | 0.000135013 | 0.00264114 | 108 |
| weak_second | 8 | 512 | random | 3 | 0.00431297 | 0.00676336 | 64 |
| weak_second | 8 | 512 | spline | 3 | 0.00431297 | 0.00676336 | 64 |
| weak_second | 64 | 128 | fixed_grid | 3 | 0.0114613 | 0.0140905 | 288 |
| weak_second | 64 | 128 | joint | 3 | 0.0110143 | 0.0136253 | 467 |
| weak_second | 64 | 128 | known_direction_oracle | 3 | 0.0484238 | 0.0514131 | 556 |
| weak_second | 64 | 128 | no_growth_refine | 3 | 0.01133 | 0.0139578 | 288 |
| weak_second | 64 | 128 | profile_only | 3 | 0.01133 | 0.0139578 | 288 |
| weak_second | 64 | 128 | projection_only | 3 | 0.0110143 | 0.0136253 | 467 |
| weak_second | 64 | 128 | random | 3 | 0.01133 | 0.0139578 | 288 |
| weak_second | 64 | 128 | spline | 3 | 0.01133 | 0.0139578 | 288 |
| weak_second | 64 | 2048 | fixed_grid | 3 | 0.00032599 | 0.00276469 | 556 |
| weak_second | 64 | 2048 | joint | 3 | 0.00031392 | 0.00276144 | 556 |
| weak_second | 64 | 2048 | known_direction_oracle | 3 | 0.00343916 | 0.00590734 | 556 |
| weak_second | 64 | 2048 | no_growth_refine | 3 | 0.00423004 | 0.00667291 | 288 |
| weak_second | 64 | 2048 | profile_only | 3 | 0.00423004 | 0.00667291 | 288 |
| weak_second | 64 | 2048 | projection_only | 3 | 0.00031392 | 0.00276144 | 556 |
| weak_second | 64 | 2048 | random | 3 | 0.00423004 | 0.00667291 | 288 |
| weak_second | 64 | 2048 | spline | 3 | 0.00423004 | 0.00667291 | 288 |

Whole-case time includes baseline, all candidate forks, fixed-grid/oracle fits, export and evaluation; it is campaign cost rather than the independent cost of each policy. Per-fit times are retained. Short 200-update validation selects each structural policy; reported long metrics follow its selected action to 1,000 updates. Fixed-grid validation selection occurs after 1,200 total retained updates and has a different selection horizon. The known-direction oracle is disclosed and excluded from allocation. Regret is post-hoc against completed long joint actions, excluding spline/fixed/oracle.

## CNN results

| Dataset | Policy | Cohort | Complete/expected | Accuracy % | Log loss | Inference bytes | Artifact bytes | Cold fit seconds | Projection counts | Extra pairs/hinges | Selected actions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cifar10 | joint | discovery | 3/3 | 72.22 | 0.7984 | 140962 | 150919 | 160.9 | [16.333333333333332, 32.333333333333336, 64.33333333333333] | 74.7 | {'continue': 3, 'refine': 3, 'profile': 2, 'projection': 1} |
| cifar10 | no_growth | discovery | 3/3 | 72.18 | 0.7991 | 136854 | 146811 | 110.9 | [16.0, 32.0, 64.0] | 0.0 | {'continue': 6, 'refine': 3} |
| cifar10 | profile | discovery | 3/3 | 72.24 | 0.7985 | 138945 | 148902 | 151.1 | [16.0, 32.0, 64.0] | 74.7 | {'continue': 3, 'refine': 4, 'profile': 2} |
| cifar10 | projection | discovery | 3/3 | 72.18 | 0.7990 | 138871 | 148828 | 119.5 | [16.333333333333332, 32.333333333333336, 64.33333333333333] | 0.0 | {'continue': 5, 'refine': 3, 'projection': 1} |
| cifar10 | random | discovery | 3/3 | 72.21 | 0.7988 | 139917 | 149874 | 130.7 | [16.333333333333332, 32.333333333333336, 64.33333333333333] | 37.3 | {'continue': 4, 'refine': 3, 'profile': 1, 'projection': 1} |
| cifar10 | relu | discovery | 3/3 | 77.57 | 0.6466 | 147753 | 154560 | 109.3 | [17.0, 34.0, 66.0] | 0.0 | {} |
| cifar10 | spline | discovery | 3/3 | 77.31 | 0.6566 | 138648 | 147795 | 164.8 | [16.0, 32.0, 64.0] | 74.7 | {'refine': 5, 'profile': 2, 'continue': 2} |
| fashion | joint | discovery | 3/3 | 91.41 | 0.2396 | 135686 | 145104 | 164.6 | [16.0, 32.0, 64.0] | 0.0 | {'refine': 5, 'continue': 4} |
| fashion | no_growth | discovery | 3/3 | 91.41 | 0.2396 | 135686 | 145104 | 181.0 | [16.0, 32.0, 64.0] | 0.0 | {'refine': 5, 'continue': 4} |
| fashion | profile | discovery | 3/3 | 91.41 | 0.2396 | 135686 | 145104 | 228.6 | [16.0, 32.0, 64.0] | 0.0 | {'refine': 5, 'continue': 4} |
| fashion | projection | discovery | 3/3 | 91.41 | 0.2396 | 135686 | 145104 | 188.4 | [16.0, 32.0, 64.0] | 0.0 | {'refine': 5, 'continue': 4} |
| fashion | random | discovery | 3/3 | 91.41 | 0.2396 | 135686 | 145104 | 188.3 | [16.0, 32.0, 64.0] | 0.0 | {'refine': 5, 'continue': 4} |
| fashion | relu | discovery | 3/3 | 91.75 | 0.2285 | 146513 | 152781 | 129.8 | [17.0, 34.0, 66.0] | 0.0 | {} |
| fashion | spline | discovery | 3/3 | 91.89 | 0.2305 | 136584 | 145192 | 256.5 | [16.0, 32.0, 64.0] | 37.3 | {'profile': 1, 'continue': 6, 'refine': 2} |

Complete inference bytes include active directions, profiles, downstream weights, normalization, topology and shared preprocessing. Actual NPZ/metadata artifact bytes are reported separately. Additional projection counts are channel growth, not pruning. Retained updates and discarded fork work are recorded independently. Cold fit process time includes startup, setup, failed proposals and final export; test/replay costs are separately recorded. Extra terms count residual pairs for tanh and hinges for the spline control; a tanh pair adds two shifted responses.

## Advancement and failures

```json
{
  "discovery_gates": {
    "cifar10": {
      "comparisons": [
        {
          "accuracy_difference": -0.00013333333333335565,
          "accuracy_difference_ci95": [
            -0.0007070203639667134,
            0.0004403536973000021
          ],
          "control": "profile",
          "nll_difference_ci95": [
            -0.0007415978365660588,
            0.00046188959453637667
          ],
          "qualified": false,
          "relative_nll": 0.9998245842401023,
          "storage_eligible": true
        },
        {
          "accuracy_difference": -0.05346666666666666,
          "accuracy_difference_ci95": [
            -0.05700020157464048,
            -0.04993313175869284
          ],
          "control": "relu",
          "nll_difference_ci95": [
            0.13427893353271708,
            0.1692018819365293
          ],
          "qualified": false,
          "relative_nll": 1.234779956769926,
          "storage_eligible": true
        }
      ],
      "confirmation": false,
      "dataset": "cifar10",
      "qualified": false,
      "seeds": [
        8101,
        8111,
        8123
      ],
      "status": "COMPLETE",
      "uncertainty": "paired t interval across independent seed cohorts; small-sample limitation retained"
    },
    "fashion": {
      "comparisons": [
        {
          "accuracy_difference": 0.0,
          "accuracy_difference_ci95": [
            0.0,
            0.0
          ],
          "control": "no_growth",
          "nll_difference_ci95": [
            0.0,
            0.0
          ],
          "qualified": false,
          "relative_nll": 1.0,
          "storage_eligible": true
        },
        {
          "accuracy_difference": -0.0048000000000000265,
          "accuracy_difference_ci95": [
            -0.012560694303242331,
            0.0029606943032422783
          ],
          "control": "spline",
          "nll_difference_ci95": [
            -0.0037518546300172838,
            0.021982799714368808
          ],
          "qualified": false,
          "relative_nll": 1.0395734193160546,
          "storage_eligible": true
        }
      ],
      "confirmation": false,
      "dataset": "fashion",
      "qualified": false,
      "seeds": [
        8101,
        8111,
        8123
      ],
      "status": "COMPLETE",
      "uncertainty": "paired t interval across independent seed cohorts; small-sample limitation retained"
    }
  },
  "recipe": {
    "datasets": {
      "cifar10": {
        "complete": true,
        "complete_policies": {
          "joint": true,
          "no_growth": true,
          "profile": true,
          "projection": true,
          "random": true,
          "relu": true,
          "spline": true
        },
        "conventional": "relu",
        "fixed": "profile",
        "selector": "joint",
        "validation_mean_loss": {
          "joint": 0.7745368620861092,
          "no_growth": 0.7752193327011573,
          "profile": 0.7745252282054005,
          "projection": 0.7751951441711018,
          "random": 0.7750098906485466,
          "relu": 0.6201445218492501,
          "spline": 0.6240129079731345
        }
      },
      "fashion": {
        "complete": true,
        "complete_policies": {
          "joint": true,
          "no_growth": true,
          "profile": true,
          "projection": true,
          "random": true,
          "relu": true,
          "spline": true
        },
        "conventional": "spline",
        "fixed": "no_growth",
        "selector": "joint",
        "validation_mean_loss": {
          "joint": 0.220951610084593,
          "no_growth": 0.220951610084593,
          "profile": 0.220951610084593,
          "projection": 0.220951610084593,
          "random": 0.220951610084593,
          "relu": 0.21218312781127144,
          "spline": 0.20968307140116185
        }
      }
    },
    "frozen_before_test": true,
    "frozen_utc": "2026-10-22T22:12:36.779366+00:00",
    "selection": "discovery checkpoint validation only; no official test consulted",
    "source_sha256": "f552253d95e3e6a9112302256704fff13cbd31806e1a0e470084f945ac75da53"
  },
  "source_sha256": "f552253d95e3e6a9112302256704fff13cbd31806e1a0e470084f945ac75da53",
  "tested": [
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8101_joint_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8101_no_growth_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8101_profile_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8101_projection_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8101_random_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8101_relu_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8101_spline_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8111_joint_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8111_no_growth_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8111_profile_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8111_projection_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8111_random_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8111_relu_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8111_spline_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8123_joint_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8123_no_growth_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8123_profile_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8123_projection_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8123_random_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8123_relu_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/cifar10_8123_spline_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8101_joint_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8101_no_growth_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8101_profile_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8101_projection_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8101_random_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8101_relu_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8101_spline_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8111_joint_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8111_no_growth_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8111_profile_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8111_projection_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8111_random_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8111_relu_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8111_spline_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8123_joint_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8123_no_growth_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8123_profile_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8123_projection_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8123_random_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8123_relu_b128",
    "/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cnn/fashion_8123_spline_b128"
  ],
  "utc": "2026-10-22T22:18:42.412365+00:00"
}
```

Every case result retains optimizer status, actual updates, proposal and failed-trial costs, validation selections, byte ceilings, learning curves and export fingerprints. A continuing decrease in training loss is labeled unresolved optimization rather than convergence.

## Reproduction

```bash
cd /home/vardan/dev/projects/omnibias_projects/omnibias-lab
/tmp/omnibias-frontier-env-20261021/bin/python -m projection_profile_research.run --stage all --device cuda --output <fresh-directory>
/tmp/omnibias-frontier-env-20261021/bin/python -m projection_profile_research.run --replay --output /home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02
```

## Decision procedure

| Observation | Action | Required check |
| --- | --- | --- |
| Retained direction useful; nonlinear scalar response too limited | Enrich profile | Matched finite fork improves fresh validation within storage ceiling |
| Finite rotation improves fit at same size | Refine existing directions | Count optimizer/state handling and search; verify longer fit |
| Another direction supplies useful residual feature or economical interaction | Add projection | Count incoming and outgoing weights; compare profile bundle |
| Training-only gain, noise sensitivity, or unsupported ranking | Continue without growth / inconclusive | Independent validation and budget eligibility |

These are conditional conclusions. The practical selector must earn its recommendation against profile-only, projection-only and conventional controls. The study distinguishes established identities, useful engineering, an architectural candidate and unsupported generalization claims.

## Every CNN seed

| Case | Status | Updates | Validation loss | Test accuracy | Inference bytes | Replay |
| --- | --- | --- | --- | --- | --- | --- |
| cifar10_8101_joint_b128 | COMPLETE | 8000 | 0.7684851672581279 | 0.7218 | 136854 | PASSED |
| cifar10_8101_no_growth_b128 | COMPLETE | 8000 | 0.7684851672581279 | 0.7218 | 136854 | PASSED |
| cifar10_8101_profile_b128 | COMPLETE | 8000 | 0.7684851672581279 | 0.7218 | 136854 | PASSED |
| cifar10_8101_projection_b128 | COMPLETE | 8000 | 0.7684851672581279 | 0.7218 | 136854 | PASSED |
| cifar10_8101_random_b128 | COMPLETE | 8000 | 0.7684851672581279 | 0.7218 | 136854 | PASSED |
| cifar10_8101_relu_b128 | COMPLETE | 8000 | 0.6300558145154419 | 0.7769 | 147753 | PASSED |
| cifar10_8101_spline_b128 | COMPLETE | 8000 | 0.6304043651881345 | 0.7715 | 136856 | PASSED |
| cifar10_8111_joint_b128 | COMPLETE | 8000 | 0.7717312912519537 | 0.7249 | 143126 | PASSED |
| cifar10_8111_no_growth_b128 | COMPLETE | 8000 | 0.7737061375069313 | 0.7235 | 136854 | PASSED |
| cifar10_8111_profile_b128 | COMPLETE | 8000 | 0.7717312912519537 | 0.7249 | 143126 | PASSED |
| cifar10_8111_projection_b128 | COMPLETE | 8000 | 0.7737061375069313 | 0.7235 | 136854 | PASSED |
| cifar10_8111_random_b128 | COMPLETE | 8000 | 0.7731503769392659 | 0.7244 | 139990 | PASSED |
| cifar10_8111_relu_b128 | COMPLETE | 8000 | 0.6035381999316587 | 0.7777 | 147753 | PASSED |
| cifar10_8111_spline_b128 | COMPLETE | 8000 | 0.6296679785349037 | 0.7713 | 139544 | PASSED |
| cifar10_8123_joint_b128 | COMPLETE | 8000 | 0.783394127748246 | 0.72 | 142906 | PASSED |
| cifar10_8123_no_growth_b128 | COMPLETE | 8000 | 0.7834666933384128 | 0.7202 | 136854 | PASSED |
| cifar10_8123_profile_b128 | COMPLETE | 8000 | 0.7833592261061201 | 0.7204 | 136854 | PASSED |
| cifar10_8123_projection_b128 | COMPLETE | 8000 | 0.783394127748246 | 0.72 | 142906 | PASSED |
| cifar10_8123_random_b128 | COMPLETE | 8000 | 0.783394127748246 | 0.72 | 142906 | PASSED |
| cifar10_8123_relu_b128 | COMPLETE | 8000 | 0.6268395511006497 | 0.7725 | 147753 | PASSED |
| cifar10_8123_spline_b128 | COMPLETE | 8000 | 0.6119663801963652 | 0.7765 | 139544 | PASSED |
| fashion_8101_joint_b128 | COMPLETE | 8000 | 0.23855904724050783 | 0.9114 | 135686 | PASSED |
| fashion_8101_no_growth_b128 | COMPLETE | 8000 | 0.23855904724050783 | 0.9114 | 135686 | PASSED |
| fashion_8101_profile_b128 | COMPLETE | 8000 | 0.23855904724050783 | 0.9114 | 135686 | PASSED |
| fashion_8101_projection_b128 | COMPLETE | 8000 | 0.23855904724050783 | 0.9114 | 135686 | PASSED |
| fashion_8101_random_b128 | COMPLETE | 8000 | 0.23855904724050783 | 0.9114 | 135686 | PASSED |
| fashion_8101_relu_b128 | COMPLETE | 8000 | 0.23166347795409828 | 0.9182 | 146513 | PASSED |
| fashion_8101_spline_b128 | COMPLETE | 8000 | 0.22514133813486906 | 0.9182 | 138376 | PASSED |
| fashion_8111_joint_b128 | COMPLETE | 8000 | 0.22050001483316214 | 0.9174 | 135686 | PASSED |
| fashion_8111_no_growth_b128 | COMPLETE | 8000 | 0.22050001483316214 | 0.9174 | 135686 | PASSED |
| fashion_8111_profile_b128 | COMPLETE | 8000 | 0.22050001483316214 | 0.9174 | 135686 | PASSED |
| fashion_8111_projection_b128 | COMPLETE | 8000 | 0.22050001483316214 | 0.9174 | 135686 | PASSED |
| fashion_8111_random_b128 | COMPLETE | 8000 | 0.22050001483316214 | 0.9174 | 135686 | PASSED |
| fashion_8111_relu_b128 | COMPLETE | 8000 | 0.2178579309322896 | 0.9182 | 146513 | PASSED |
| fashion_8111_spline_b128 | COMPLETE | 8000 | 0.21326869580153424 | 0.9186 | 135688 | PASSED |
| fashion_8123_joint_b128 | COMPLETE | 8000 | 0.203795768180109 | 0.9135 | 135686 | PASSED |
| fashion_8123_no_growth_b128 | COMPLETE | 8000 | 0.203795768180109 | 0.9135 | 135686 | PASSED |
| fashion_8123_profile_b128 | COMPLETE | 8000 | 0.203795768180109 | 0.9135 | 135686 | PASSED |
| fashion_8123_projection_b128 | COMPLETE | 8000 | 0.203795768180109 | 0.9135 | 135686 | PASSED |
| fashion_8123_random_b128 | COMPLETE | 8000 | 0.203795768180109 | 0.9135 | 135686 | PASSED |
| fashion_8123_relu_b128 | COMPLETE | 8000 | 0.18702797454742642 | 0.9161 | 146513 | PASSED |
| fashion_8123_spline_b128 | COMPLETE | 8000 | 0.19063918026708224 | 0.9199 | 135688 | PASSED |

## Confirmation assessment

```json
{
  "confirmation_cases": [],
  "confirmation_gates": {},
  "confirmed": false,
  "discovery_diagnostics": [
    {
      "case": "fashion_8101_joint_b128",
      "discarded_updates": 1800,
      "fit_seconds": 103.83607869304251,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "refine",
        "refine",
        "continue"
      ],
      "train_improvement": 0.05152747963338334
    },
    {
      "case": "fashion_8111_projection_b128",
      "discarded_updates": 1200,
      "fit_seconds": 229.3097473490052,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "refine",
        "refine",
        "continue"
      ],
      "train_improvement": 0.04654262338548801
    },
    {
      "case": "fashion_8111_spline_b128",
      "discarded_updates": 1200,
      "fit_seconds": 220.09573262906633,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "refine",
        "continue",
        "refine"
      ],
      "train_improvement": 0.048776588716709555
    },
    {
      "case": "cifar10_8111_random_b128",
      "discarded_updates": 1200,
      "fit_seconds": 143.0730833800044,
      "optimization_status": "FINITE_BUDGET_FIT",
      "selected_actions": [
        "refine",
        "continue",
        "profile"
      ],
      "train_improvement": 0.00782998525714396
    },
    {
      "case": "fashion_8123_random_b128",
      "discarded_updates": 1200,
      "fit_seconds": 87.03051111008972,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "continue",
        "refine",
        "continue"
      ],
      "train_improvement": 0.050628994976774096
    },
    {
      "case": "fashion_8123_no_growth_b128",
      "discarded_updates": 600,
      "fit_seconds": 207.63891873997636,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "continue",
        "refine",
        "continue"
      ],
      "train_improvement": 0.050628994976774096
    },
    {
      "case": "cifar10_8111_profile_b128",
      "discarded_updates": 1200,
      "fit_seconds": 196.74212685693055,
      "optimization_status": "FINITE_BUDGET_FIT",
      "selected_actions": [
        "refine",
        "profile",
        "profile"
      ],
      "train_improvement": 0.00825431403742694
    },
    {
      "case": "cifar10_8101_profile_b128",
      "discarded_updates": 1200,
      "fit_seconds": 125.16108093596995,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "continue",
        "continue",
        "refine"
      ],
      "train_improvement": 0.01678094698258867
    },
    {
      "case": "cifar10_8123_relu_b128",
      "discarded_updates": 0,
      "fit_seconds": 105.88269386708271,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [],
      "train_improvement": 0.03184610551527377
    },
    {
      "case": "cifar10_8111_spline_b128",
      "discarded_updates": 1200,
      "fit_seconds": 168.5641262329882,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "refine",
        "profile",
        "continue"
      ],
      "train_improvement": 0.026319262231597338
    },
    {
      "case": "fashion_8111_no_growth_b128",
      "discarded_updates": 600,
      "fit_seconds": 210.7007384870667,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "refine",
        "refine",
        "continue"
      ],
      "train_improvement": 0.04654262338548801
    },
    {
      "case": "fashion_8123_spline_b128",
      "discarded_updates": 1200,
      "fit_seconds": 90.94284882501233,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "continue",
        "continue",
        "continue"
      ],
      "train_improvement": 0.059263267674099514
    },
    {
      "case": "cifar10_8123_joint_b128",
      "discarded_updates": 1800,
      "fit_seconds": 135.81644774705637,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "refine",
        "continue",
        "projection"
      ],
      "train_improvement": 0.011956124155345416
    },
    {
      "case": "cifar10_8123_no_growth_b128",
      "discarded_updates": 600,
      "fit_seconds": 106.873021367006,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "refine",
        "continue",
        "continue"
      ],
      "train_improvement": 0.01210135181767597
    },
    {
      "case": "fashion_8101_relu_b128",
      "discarded_updates": 0,
      "fit_seconds": 135.7436940560583,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [],
      "train_improvement": 0.06903412983716277
    },
    {
      "case": "cifar10_8123_profile_b128",
      "discarded_updates": 1200,
      "fit_seconds": 124.33975423604716,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "refine",
        "continue",
        "refine"
      ],
      "train_improvement": 0.012052134619584064
    },
    {
      "case": "cifar10_8111_joint_b128",
      "discarded_updates": 1800,
      "fit_seconds": 203.50792426697444,
      "optimization_status": "FINITE_BUDGET_FIT",
      "selected_actions": [
        "refine",
        "profile",
        "profile"
      ],
      "train_improvement": 0.00825431403742694
    },
    {
      "case": "cifar10_8101_projection_b128",
      "discarded_updates": 1200,
      "fit_seconds": 116.69997517298907,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "continue",
        "continue",
        "refine"
      ],
      "train_improvement": 0.01678094698258867
    },
    {
      "case": "cifar10_8123_spline_b128",
      "discarded_updates": 1200,
      "fit_seconds": 194.41095269098878,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "profile",
        "continue",
        "refine"
      ],
      "train_improvement": 0.02115448464957106
    },
    {
      "case": "cifar10_8123_random_b128",
      "discarded_updates": 1200,
      "fit_seconds": 121.00519486400299,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "refine",
        "continue",
        "projection"
      ],
      "train_improvement": 0.011956124155345416
    },
    {
      "case": "fashion_8101_profile_b128",
      "discarded_updates": 1200,
      "fit_seconds": 188.12254995002877,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "refine",
        "refine",
        "continue"
      ],
      "train_improvement": 0.05152747963338334
    },
    {
      "case": "fashion_8123_profile_b128",
      "discarded_updates": 1200,
      "fit_seconds": 227.93504935200326,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "continue",
        "refine",
        "continue"
      ],
      "train_improvement": 0.050628994976774096
    },
    {
      "case": "fashion_8123_joint_b128",
      "discarded_updates": 1800,
      "fit_seconds": 112.98245433100965,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "continue",
        "refine",
        "continue"
      ],
      "train_improvement": 0.050628994976774096
    },
    {
      "case": "cifar10_8101_random_b128",
      "discarded_updates": 1200,
      "fit_seconds": 120.81796945806127,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "continue",
        "continue",
        "refine"
      ],
      "train_improvement": 0.01678094698258867
    },
    {
      "case": "cifar10_8101_no_growth_b128",
      "discarded_updates": 600,
      "fit_seconds": 108.30230810400099,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "continue",
        "continue",
        "refine"
      ],
      "train_improvement": 0.01678094698258867
    },
    {
      "case": "fashion_8111_relu_b128",
      "discarded_updates": 0,
      "fit_seconds": 168.68933368101716,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [],
      "train_improvement": 0.09542450921376479
    },
    {
      "case": "cifar10_8111_relu_b128",
      "discarded_updates": 0,
      "fit_seconds": 105.56505371001549,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [],
      "train_improvement": 0.04641255943335415
    },
    {
      "case": "fashion_8111_joint_b128",
      "discarded_updates": 1800,
      "fit_seconds": 261.7299500940135,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "refine",
        "refine",
        "continue"
      ],
      "train_improvement": 0.04654262338548801
    },
    {
      "case": "cifar10_8101_joint_b128",
      "discarded_updates": 1800,
      "fit_seconds": 136.27864125894848,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "continue",
        "continue",
        "refine"
      ],
      "train_improvement": 0.01678094698258867
    },
    {
      "case": "fashion_8101_random_b128",
      "discarded_updates": 1200,
      "fit_seconds": 221.84798290999606,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "refine",
        "refine",
        "continue"
      ],
      "train_improvement": 0.05152747963338334
    },
    {
      "case": "cifar10_8111_no_growth_b128",
      "discarded_updates": 600,
      "fit_seconds": 110.6876556549687,
      "optimization_status": "FINITE_BUDGET_FIT",
      "selected_actions": [
        "refine",
        "continue",
        "continue"
      ],
      "train_improvement": 0.0075243454425428504
    },
    {
      "case": "cifar10_8123_projection_b128",
      "discarded_updates": 1200,
      "fit_seconds": 118.47810458601452,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "refine",
        "continue",
        "projection"
      ],
      "train_improvement": 0.011956124155345416
    },
    {
      "case": "fashion_8123_projection_b128",
      "discarded_updates": 1200,
      "fit_seconds": 199.60784553701524,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "continue",
        "refine",
        "continue"
      ],
      "train_improvement": 0.050628994976774096
    },
    {
      "case": "fashion_8101_no_growth_b128",
      "discarded_updates": 600,
      "fit_seconds": 104.10679419303779,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "refine",
        "refine",
        "continue"
      ],
      "train_improvement": 0.05152747963338334
    },
    {
      "case": "fashion_8101_spline_b128",
      "discarded_updates": 1200,
      "fit_seconds": 438.993483267026,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "profile",
        "continue",
        "continue"
      ],
      "train_improvement": 0.06320160535647482
    },
    {
      "case": "cifar10_8101_relu_b128",
      "discarded_updates": 0,
      "fit_seconds": 109.95702483598143,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [],
      "train_improvement": 0.029446146474999848
    },
    {
      "case": "cifar10_8101_spline_b128",
      "discarded_updates": 1200,
      "fit_seconds": 124.06848785409238,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "refine",
        "refine",
        "refine"
      ],
      "train_improvement": 0.01944104644786479
    },
    {
      "case": "cifar10_8111_projection_b128",
      "discarded_updates": 1200,
      "fit_seconds": 115.77930711302906,
      "optimization_status": "FINITE_BUDGET_FIT",
      "selected_actions": [
        "refine",
        "continue",
        "continue"
      ],
      "train_improvement": 0.0075243454425428504
    },
    {
      "case": "fashion_8123_relu_b128",
      "discarded_updates": 0,
      "fit_seconds": 67.80886977096088,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [],
      "train_improvement": 0.0643644655317247
    },
    {
      "case": "fashion_8111_random_b128",
      "discarded_updates": 1200,
      "fit_seconds": 240.78646927990485,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "refine",
        "refine",
        "continue"
      ],
      "train_improvement": 0.04654262338548801
    },
    {
      "case": "fashion_8101_projection_b128",
      "discarded_updates": 1200,
      "fit_seconds": 118.31092509801965,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "refine",
        "refine",
        "continue"
      ],
      "train_improvement": 0.05152747963338334
    },
    {
      "case": "fashion_8111_profile_b128",
      "discarded_updates": 1200,
      "fit_seconds": 249.1631002459908,
      "optimization_status": "UNRESOLVED_IMPROVING",
      "selected_actions": [
        "refine",
        "refine",
        "continue"
      ],
      "train_improvement": 0.04654262338548801
    }
  ],
  "failure_investigation": "controlled short/long reversals, selected-action logs, learning curves, local-score approximation and complete search cost; no test retuning",
  "qualified_datasets": [],
  "status": "NO_QUALIFYING_CANDIDATE"
}
```

## Figures

![allocation_heatmap](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/allocation_heatmap.png)

![fashion_learning_curves](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/fashion_learning_curves.png)

![cifar10_learning_curves](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cifar10_learning_curves.png)

![fashion_paired_mistakes](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/fashion_paired_mistakes.png)

![cifar10_paired_mistakes](/home/vardan/dev/projects/omnibias_projects/omnibias-lab/artifacts/projection_profile_20261022_02/cifar10_paired_mistakes.png)

## Deployment measurements

| Case | Device | Batch | Median ms | Peak CUDA bytes |
| --- | --- | --- | --- | --- |
| cifar10_8101_joint_b128 | cpu | 1 | 0.44049 | None |
| cifar10_8101_joint_b128 | cpu | 128 | 53.31428 | None |
| cifar10_8101_joint_b128 | cuda | 1 | 0.38416 | 34132992 |
| cifar10_8101_joint_b128 | cuda | 128 | 1.69319 | 68985856 |
| cifar10_8101_no_growth_b128 | cpu | 1 | 0.39965 | None |
| cifar10_8101_no_growth_b128 | cpu | 128 | 47.49212 | None |
| cifar10_8101_no_growth_b128 | cuda | 1 | 0.38295 | 34132992 |
| cifar10_8101_no_growth_b128 | cuda | 128 | 1.69082 | 68985856 |
| cifar10_8101_profile_b128 | cpu | 1 | 0.43093 | None |
| cifar10_8101_profile_b128 | cpu | 128 | 51.46778 | None |
| cifar10_8101_profile_b128 | cuda | 1 | 0.37562 | 34132992 |
| cifar10_8101_profile_b128 | cuda | 128 | 1.69052 | 68985856 |
| cifar10_8101_projection_b128 | cpu | 1 | 0.42581 | None |
| cifar10_8101_projection_b128 | cpu | 128 | 45.97361 | None |
| cifar10_8101_projection_b128 | cuda | 1 | 0.36684 | 34132992 |
| cifar10_8101_projection_b128 | cuda | 128 | 1.69251 | 68985856 |
| cifar10_8101_relu_b128 | cpu | 1 | 0.37770 | None |
| cifar10_8101_relu_b128 | cpu | 128 | 39.80073 | None |
| cifar10_8101_relu_b128 | cuda | 1 | 0.28034 | 34230784 |
| cifar10_8101_relu_b128 | cuda | 128 | 1.50174 | 53264384 |
| cifar10_8101_spline_b128 | cpu | 1 | 0.37329 | None |
| cifar10_8101_spline_b128 | cpu | 128 | 47.58383 | None |
| cifar10_8101_spline_b128 | cuda | 1 | 0.38679 | 34128896 |
| cifar10_8101_spline_b128 | cuda | 128 | 1.68648 | 68981760 |
| cifar10_8111_joint_b128 | cpu | 1 | 0.78945 | None |
| cifar10_8111_joint_b128 | cpu | 128 | 108.87539 | None |
| cifar10_8111_joint_b128 | cuda | 1 | 0.79439 | 34265088 |
| cifar10_8111_joint_b128 | cuda | 128 | 5.44227 | 85764096 |
| cifar10_8111_no_growth_b128 | cpu | 1 | 0.38232 | None |
| cifar10_8111_no_growth_b128 | cpu | 128 | 46.97031 | None |
| cifar10_8111_no_growth_b128 | cuda | 1 | 0.36884 | 34132992 |
| cifar10_8111_no_growth_b128 | cuda | 128 | 1.69194 | 68985856 |
| cifar10_8111_random_b128 | cpu | 1 | 0.49733 | None |
| cifar10_8111_random_b128 | cpu | 128 | 62.08130 | None |
| cifar10_8111_random_b128 | cuda | 1 | 0.57625 | 34265088 |
| cifar10_8111_random_b128 | cuda | 128 | 3.56115 | 85764096 |
| cifar10_8111_spline_b128 | cpu | 1 | 0.45461 | None |
| cifar10_8111_spline_b128 | cpu | 128 | 53.28758 | None |
| cifar10_8111_spline_b128 | cuda | 1 | 0.49679 | 34194944 |
| cifar10_8111_spline_b128 | cuda | 128 | 2.62466 | 77370880 |
| cifar10_8123_joint_b128 | cpu | 1 | 0.40410 | None |
| cifar10_8123_joint_b128 | cpu | 128 | 51.86292 | None |
| cifar10_8123_joint_b128 | cuda | 1 | 0.37575 | 34239488 |
| cifar10_8123_joint_b128 | cuda | 128 | 1.95327 | 71102464 |
| cifar10_8123_no_growth_b128 | cpu | 1 | 0.42635 | None |
| cifar10_8123_no_growth_b128 | cpu | 128 | 51.81284 | None |
| cifar10_8123_no_growth_b128 | cuda | 1 | 0.37417 | 34132992 |
| cifar10_8123_no_growth_b128 | cuda | 128 | 1.69207 | 68985856 |
| cifar10_8123_spline_b128 | cpu | 1 | 0.44762 | None |
| cifar10_8123_spline_b128 | cpu | 128 | 53.74977 | None |
| cifar10_8123_spline_b128 | cuda | 1 | 0.50105 | 34194944 |
| cifar10_8123_spline_b128 | cuda | 128 | 2.62383 | 77370880 |
| fashion_8101_joint_b128 | cpu | 1 | 0.36687 | None |
| fashion_8101_joint_b128 | cpu | 128 | 42.58816 | None |
| fashion_8101_joint_b128 | cuda | 1 | 0.38565 | 34084864 |
| fashion_8101_joint_b128 | cuda | 128 | 1.40016 | 59948032 |
| fashion_8101_no_growth_b128 | cpu | 1 | 0.36640 | None |
| fashion_8101_no_growth_b128 | cpu | 128 | 35.55781 | None |
| fashion_8101_no_growth_b128 | cuda | 1 | 0.38221 | 34084864 |
| fashion_8101_no_growth_b128 | cuda | 128 | 1.41145 | 59948032 |
| fashion_8101_profile_b128 | cpu | 1 | 0.35616 | None |
| fashion_8101_profile_b128 | cpu | 128 | 40.34586 | None |
| fashion_8101_profile_b128 | cuda | 1 | 0.37302 | 34084864 |
| fashion_8101_profile_b128 | cuda | 128 | 1.39401 | 59948032 |
| fashion_8101_projection_b128 | cpu | 1 | 0.36735 | None |
| fashion_8101_projection_b128 | cpu | 128 | 39.71808 | None |
| fashion_8101_projection_b128 | cuda | 1 | 0.36459 | 34084864 |
| fashion_8101_projection_b128 | cuda | 128 | 1.40388 | 59948032 |
| fashion_8101_relu_b128 | cpu | 1 | 0.31115 | None |
| fashion_8101_relu_b128 | cpu | 128 | 29.79249 | None |
| fashion_8101_relu_b128 | cuda | 1 | 0.28375 | 34214912 |
| fashion_8101_relu_b128 | cuda | 128 | 1.25592 | 47912960 |
| fashion_8101_spline_b128 | cpu | 1 | 0.39940 | None |
| fashion_8101_spline_b128 | cpu | 128 | 39.26568 | None |
| fashion_8101_spline_b128 | cuda | 1 | 0.50496 | 34107392 |
| fashion_8101_spline_b128 | cuda | 128 | 2.13096 | 66366976 |

Latency uses normalized float32 inputs, one CPU thread and synchronized CUDA blocks; input preprocessing arithmetic is excluded. Three block means do not estimate per-request tail latency.
