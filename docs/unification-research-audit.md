# Unifying representations, training geometry and verification

This 20 October 2026 audit revisits the user's objections to the preceding
calculus–geometry summary and the supplied 56-page multi-bias manuscript.
It adds independent checks, corrects two interpretations, and develops a
constructive framework. It does not establish that deep networks dominate every
learning algorithm or that the framework is mathematically new.

Research code: `omnibias-lab/unification_research/audit.py`.
Retained evidence: `omnibias-lab/artifacts/unification_audit_20261020`.
The supplied PDF remains unchanged and is not redistributed; its digest is
recorded in `inputs.json`. The relevant sections inspected are Appendix W,
Figures 17–18, Lemma 1's proof, and Theorem 2/Appendix C.11. The empirical
claims and exhaustive Boolean certificates of the whole manuscript were not
independently reproduced in this audit.

## Corrections and attribution

1. The earlier derivative-coordinate example held the value field fixed while
   editing its derivatives. It proves incompatibility with that retained field.
   A different, curl-based example proves that some edited derivative vectors
   cannot come from any field. Neither counterexample invalidates the successful
   global compatible reconstruction already implemented.
2. The positive-source compression timing charged exact rational construction
   and replay to the candidate, while classical controls used floating builders
   and floating error checks. The recorded 10.5–30.8-fold pipeline overhead is
   not an equal-assurance comparison or a proof of intrinsic inferiority.
3. The TabPFN/MLP/ridge/CatBoost improvements were produced by applying the
   laboratory's ordinary QR output adapter to each named base learner. There was
   no separate omnibias predictor in that table. The counting block was our
   experimental architecture; the unrestricted LUT was a local matched control.
4. The earlier expected-F1 experiment used linear logits, not deep or multi-bias
   networks. Its negative result does not exclude those architectures. It also
   omitted optimization-path checkpoint selection and a no-update BCE fallback.

No frozen numerical source or prior measurement was changed. The affected
documentation now records the assurance and compatibility qualifications.

## 1. Exact decomposition does not change a shared-input derivative

The user's subsequent [two-sigmoid experiment](two-bias-bell-experiment.md)
uses different monotone primitives and supplies a positive result: individual
sigmoid-bias gradients are nonzero at the bell's peak, and training escapes
through its adjustable height/width. The Gaussian-half calculations below
must not be applied to that different construction.

For the manuscript's Gaussian construction,

\[
\phi(z)=e^{-z^2},\qquad
g(z)=e^{-\min(z,0)^2},\qquad h(z)=g(z)-\phi(z),
\]

both branches are nondecreasing, and \(g-h=\phi\) over the reals. But

\[
\frac{d}{dz}[g(z)-h(z)]=g'(z)-h'(z)=\phi'(z).
\]

For a loss \(\ell\), the shared-input gradient remains
\(\ell'(\phi(z))\phi'(z)\). Backpropagation sums the branch contributions
at their shared parent; graph decomposition cannot circumvent the chain rule.
Here \(g'(0)=h'(0)=0\), and the nonzero branch derivative tends to zero in
either tail. Complementary support is not a positive uniform lower bound.

An independently trainable model is a different parameterization:

\[
F(z;s,b)=s_1g(z+b_1)+s_2h(z+b_2).
\]

At \((s_1,s_2,b_1,b_2)=(1,-1,0,0)\), its initial function is the Gaussian.
Its amplitude derivatives are \(g,h\); its bias derivatives are
\(s_1g',s_2h'\). For \(z=0\), target zero and loss \(F^2/2\), the
gradient in this parameter order is exactly \((1,0,0,0)\).
The amplitude can move even though both bias gradients vanish. An ordinary
Gaussian with a trainable amplitude has the same useful direction.

The fresh scalar experiment starts exactly at the peak. Native-bias,
tied-decomposition-bias and independently shifted Gaussian-half branches all
remain at loss 0.5 for 20 gradient steps. Trainable-amplitude native and split
models have identical decreasing loss traces. This isolates the added parameter
direction; it is not a model-quality benchmark. Approaching a peak while trying
to maximize the Gaussian is also not itself an optimization failure: the peak
is the requested optimum.

Real-arithmetic identity does not imply floating-point bit identity. In
binary64 at \(z=10\), the direct Gaussian is approximately
\(3.7201\times10^{-44}\), while evaluating \(1-(1-\phi(z))\) gives zero.
The manuscript's general bit-identity claim needs an implementation-specific
contract. Equal initial bias values also do not make separate parameter
gradients identical to the gradient of a single tied parameter.

### A corrected monotone lift, and its limitation

For any differentiable scalar \(\phi\) with \(|\phi'|\le M\), choose
\(\alpha>M/2\) and define

\[
g_\alpha(z)=\alpha z+\phi(z)/2,\qquad
h_\alpha(z)=\alpha z-\phi(z)/2.
\]

Both derivatives are at least \(\alpha-M/2>0\), and their difference is
\(\phi\). For the Gaussian, \(M=\sqrt{2/e}\). Independent bias shifts
therefore have nonzero local parameter derivatives even at the original peak.
The shared-input derivative still cancels to \(\phi'\).

This construction has a revealing control. With unit opposite coefficients,

\[
g_\alpha(z+b_1)-h_\alpha(z+b_2)
=\tfrac12[\phi(z+b_1)+\phi(z+b_2)]+\alpha(b_1-b_2).
\]

At tied initialization, writing \(v_i=\phi'(z_i)\), the bias-only kernel is
\(2\alpha^2\mathbf1\mathbf1^T+vv^T/2\). The new first-order direction is
an output offset; an explicit output-bias control is essential. This does not
prove recovery of missing input geometry, and the unbounded cancelling linear
branches require numerical care.

### What a better loss can accomplish

A positive equality example can train with a normalized residual loss
\(L=r^2/(2\|w\|^2)\), preserving the exact hyperplane as its zero set.
For a Gaussian-shaped score, \(-\log\phi(r)=r^2\) over the reals avoids the
tail attenuation of a squared peak loss; compute the residual expression
directly to avoid numerical underflow. This is a constructive loss correction.

There is no differentiable \(\ell(\phi(r))\) with finite \(\ell'(1)\)
that gives a nonzero ordinary derivative at \(r=0\), since \(\phi'(0)=0\).
For negative equality examples, the target is exclusion, such as
\(|r|/\|w\|\ge m\). A differentiable symmetric negative loss also has zero
derivative at its symmetry point. Side selection, asymmetric residuals, extra
parameters, second-order steps or a specified nonsmooth rule are distinct
possible mechanisms; renaming the loss does not remove that obligation.

## 2. A unifying model of representation and learning

Use explicit maps

\[
\theta\xrightarrow{T}u_\theta
\xrightarrow{J^q}\text{field and derivatives}
\xrightarrow{\mathcal O}\text{observations/residuals}
\xrightarrow{\ell}\mathbb R.
\]

Discrete readouts are separate maps evaluated from these quantities. A smooth
stochastic expectation can be differentiated according to its probability model;
it is not the ordinary derivative of a hard predicate. Values, derivatives,
constraints, observations and deployment decisions must retain their individual
contracts.

On a fixed finite collection of outputs let \(F(\theta)\in\mathbb R^m\),
\(J=D_\theta F\), and \(g=\nabla_F\mathcal L\). Euclidean gradient flow
gives the exact finite-network identities

\[
\dot F=-JJ^Tg,\qquad
\dot{\mathcal L}=-\|J^Tg\|^2.
\]

Three objects matter: the represented family, the tangent directions \(J\),
and the task gradient \(g\). Equal functions can have different training
geometry. Exact output derivatives do not guarantee that a useful parameter
direction exists. Conversely, adding parameters can improve optimization without
adding a new represented function family.

These identities connect to established
[neural tangent kernel theory](https://papers.neurips.cc/paper_files/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html).
Function-preserving expansion also has direct precedents in
[Network Morphism](https://proceedings.mlr.press/v48/wei16.html).
No infinite-width assumption is needed for the displayed local identities;
global convergence and generalization require separate assumptions.

### A constructive theorem: preserve the function, add useful directions

Retain the old parameters and their Euclidean metric, and append

\[
F_+(\theta,a)=F(\theta)+\Psi a,\qquad a=0.
\]

Then the initial function is unchanged, and

\[
J_+=[J,\Psi],\qquad K_+=JJ^T+\Psi\Psi^T,
\qquad
\dot{\mathcal L}_+-\dot{\mathcal L}
=-\|\Psi^Tg\|^2\le0.
\]

This proves a precise escape condition: if \(J^Tg=0\) but
\(\Psi^Tg\ne0\), the original parameterization is stationary and the lifted
one has a descent direction. It does not assert that arbitrary branch replacement
improves the kernel. It is essential to retain the original parameter path.

For a fixed affine constraint \(CF=d\), if \(CF(\theta)=d\) for every
\(\theta\) and \(C\Psi=0\), then the lifted model remains feasible for all
parameters. In a PDE field this must hold as a global function identity, not
only at training samples. Nonlinear tangent constraints are only first-order
unless a compatible realization or retraction supplies an exact extension.

Rescaling \(\Psi\) rescales the gain, so feature normalization, learning rate,
additional storage and compute must enter a performance comparison. The theorem
does not imply global convergence, better test error, stability of a large SGD
step, or an Adam guarantee. Added features may be orthogonal to the residual.

For an \(L\)-smooth loss in output space and one fixed normalized feature
\(\|\psi\|_2=1\), holding the old model fixed and choosing
\(a=-\langle g,\psi\rangle/L\) gives

\[
\mathcal L(F+a\psi)-\mathcal L(F)
\le-\langle g,\psi\rangle^2/(2L).
\]

This follows directly from the smoothness inequality. It turns the mechanism
into a checked finite-step rule under the stated assumption. Choosing informative
features is still the learning problem. Closely related adaptive constructions
exist in [functional gradient boosting](https://proceedings.mlr.press/v108/nitanda20a.html).

### A compatible PDE-field example that actually unsticks

Let \(u_c(x)=c(x^2-1)\) on \([-1,1]\). Every member satisfies the same
zero boundary data and \(u_c''=2c\). Recover \(c=1\) with
\(\mathcal L=(c-1)^2/2\).

The parameterization \(c=q^3\) represents every real coefficient but stalls
at \(q=0\). The lift \(c=q^3+a\), initialized at \(q=a=0\), has exactly
the same initial field and represented family, while \(\partial_a\mathcal L=-1\).
The implemented fixture escapes and retains all compatibility/boundary identities.
Direct coefficient optimization has exactly the same loss trace. This is a
positive proof of the parameterization mechanism with an honest matched control.

### Where bias collapse enters this framework

For a smooth ridge \(u(x)=\phi(w^Tx+b)\),
\(D_x^\alpha u=w^\alpha\partial_b^{|\alpha|}u\). Properly normalized
finite bias differences generate these derivative features in the limit.
Their useful role is to supply candidate directions \(\psi\) for the lift,
with conditioning and remainder errors checked separately.

There is an exact compatibility rule. If a smooth family satisfies a fixed
linear field constraint \(Cu_b=d\) for every \(b\), and differentiation
commutes with \(C\), then
\(C\partial_b^k u_b=0\) for \(k\ge1\). Those bias derivatives can enrich
the family while preserving that constraint. This applies to fixed boundary
conditions only when the family already preserves them.

For nonlinear equations \(R(u_b)=0\), the first derivative instead satisfies
\(DR(u_b)[\partial_bu_b]=0\): it is a tangent direction to the solution set.
A finite linear addition need not remain a nonlinear solution. Higher
derivatives also involve higher derivatives of \(R\). An exact realization
map or a verified correction is needed to go beyond this local guarantee.
This distinguishes a useful calculus–geometry bridge from the unsupported
claim that every nonlinear PDE becomes a linear coefficient problem.

## 3. Compatibility and compression revisited

For \(u(t,x)=x\), independently replacing derivatives \((0,1)\) by
\((1/2,1/2)\) makes an algebraic equality pass but disagrees with the retained
field. Those derivatives alone integrate to \((t+x)/2\). In contrast,
\((p_x,p_y)=(0,x)\) has curl one and cannot be a gradient on an open rectangle.

The successful existing construction corrects it to \((y/2,x/2)\), then
reconstructs \(u=xy/2\), including global PDE/boundary identities. A general
safe design parameterizes one field first and derives all its outputs. For a
finite linear compatible space, use \(c=c_0+Nz_\theta\), \(CN=0\).

For Dirichlet \(-u''=\delta_{c-h}+\delta_{c+h}\), replacing the source by
\(2\delta_c\) gives the exact triangular solution error

\[
e(x)=\max(0,h-|x-c|),\qquad \|e\|_\infty=h.
\]

The more accurate weak source expansion \(2\delta_c+h^2\delta_c''\)
has solution \(2G(\cdot,c)-h^2\delta_c\), a distribution. Thus a fourth-order
weak probe error and first-order bounded-field error can coexist without any
contradiction. A transfer theorem needs the actual stability estimate
\(\|L^{-1}(f-\tilde f)\|_Y\le C\|f-\tilde f\|_X\).

Nine independent rational Green examples and symbolic compatibility checks passed.
75 focused core tests passed. The old compression timings are retained with the
unequal-assurance qualification above; a new equal-assurance benchmark was not run.

## 4. The Hermite timing crossover is reproducible

The pair contraction and distinct-moment contraction compute the same matrix,
but have different arithmetic costs:

\[
T_{pair}=\Theta(N^2D+ND^2),\quad
T_{unique}=\Theta\left(N\binom{D+3}{4}+D^2\binom{D+2}{3}+ND^2\right).
\]

At \(N=2048\) there are 4,194,304 ordered sample pairs. At \(D=8\),
only 330 distinct fourth moments are needed; at \(D=64\), there are 766,480.
Smaller dimension makes the pair implementation faster in absolute time, but
makes its tensor comparator faster by much more.

| Kernel | D | Pair ms | Best included tensor ms | Tensor/pair |
| --- | ---: | ---: | ---: | ---: |
| Sign | 8 | 31.465 | 0.631 | 0.020 |
| Sign | 64 | 47.896 | 221.834 | 4.632 |
| RBF | 8 | 35.003 | 1.155 | 0.033 |
| RBF | 64 | 52.529 | 457.264 | 8.705 |

This fresh focused rerun used one BLAS thread, one warmup and five interleaved
runtime inputs. It does not replace the original complete grid. 20 existing
tests passed; 64 exact rational Hermite pair identities matched an independent
Kronecker-delta tensor oracle. Timed matrix disagreement was at most
\(4.36\times10^{-14}\). No bug was found in the checked contraction contract.
A measured shape-aware dispatcher is the practical extension.

## 5. Objective optimization, hardening and attribution

Optimizing the exact same deployment objective globally over the same feasible
class cannot be worse on that objective than choosing another member of the
class. Strict improvement is not guaranteed. Finite training, different
stochastic semantics and unseen-data performance are different statements.

A completely explicit counterexample separates stochastic expected F1 from
deterministic thresholded F1, even with global optimization and optimal threshold
selection. For labels \((1,0)\), define on \(0\le t\le1\)

\[
p_1=11/20+t^2/4,\quad p_2=1/10+4t/5,\quad
J(t)=\mathbb E[F1]=p_1(1-p_2/3).
\]

Then

\[
J(t)-J(1)=(t-1)(-40t^2+105t+17)/600\le0.
\]

Expected F1 is globally maximized at \(t=1\), giving \((0.8,0.9)\) and
best thresholded F1 \(2/3\). BCE is minimized by maximizing

\[
p_1(1-p_2)=99/200-11t/25+9t^2/40-t^3/5,
\]

which is strictly decreasing on this interval. It chooses \(t=0\), giving
\((0.55,0.1)\) and hard F1 one. The inequalities, enumeration, existing
expected-F1 gradients and threshold selector agree. This disproves the general
implication; it does not identify the cause of every old training result.

For Boolean parity under independent Bernoulli inputs,
\(q=[1-\prod_i(1-2p_i)]/2\), so
\(\partial_iq=\prod_{j\ne i}(1-2p_j)\). The central soft configuration is
flat for two or more inputs. Representation of parity does not ensure gradient
recovery of its unknown wiring. A hardening certificate proves that a soft
model and its export agree; both can agree on wrong labels.

Seven existing objective/count semantics tests passed. The new exact metric
counterexample was independently checked against enumeration and the existing
implementation. No inspected objective/count formula defect was found. The
previous learning matrices were not retrained.

The following interpretations are now explicit:

| Result | Producer | Legitimate next enhancement |
| --- | --- | --- |
| TabPFN 11.45%, MLP 9.53%, ridge 0.82%, CatBoost 0.25% MSE reductions | Ordinary QR output adapter around each named learner | Validate uncertain/learned constraints; retain a correct-constraint guarantee |
| Exactly-three 100%, parity 49.01%, nonsymmetric 66.21% | Laboratory counting-block candidate | Better wiring search and symmetry discovery against GF(2), enumeration and matched LUT |
| LUT 99.23%, 49.02%, 71.28% | Local unrestricted-LUT control | Not an official DiffLogic or DiffLUT result |
| Normal anomaly detection succeeds, tangent rarity missed | Coherent implicit field, matched by ordinary AD | Add a probability model along the manifold; calibrate joint score on held-out data |
| LP basis proposals often accepted in dimension 2 | Nearest-instance proposal plus exact KKT checker | Cache verified parameter validity regions and compare with warm-basis HiGHS including assurance |

These enhancements are proposed follow-ups, not newly measured performance gains.
For expected-metric training, retain the BCE model as a validation candidate and
select across the training path. This avoids needlessly discarding a better
observed validation checkpoint; it does not guarantee unseen-data improvement.

## 6. An additional flaw in the attached manuscript's spectral theorem

Theorem 2 claims whole-line L2 convergence of a specified fixed-lattice sum.
Its hypotheses are insufficient. Take

\[
\sigma(x)=e^{-x^2},\quad
\widehat\sigma(\omega)=\sqrt\pi e^{-\omega^2/4},\quad
\widehat f=\widehat\sigma\mathbf1_{[-W,W]}.
\]

All stated positivity and boundedness conditions on the finite band hold.
The prescribed coefficients are

\[
s_k=(2W)^{-1}\int_{-W}^W e^{i\omega k\pi/W}\,d\omega
=\mathbf1_{k=0}.
\]

Every truncation containing zero is therefore \(f_K=\sigma\), while

\[
\|f_K-f\|_{L^2(\mathbb R)}^2
=\sqrt{\pi/2}\operatorname{erfc}(W/\sqrt2)>0.
\]

Appendix C.11 line 806 incorrectly discards out-of-band error. The Gaussian
shift sums are not bandlimited. A corrected statement can band-project the
kernel or restrict its error claim to the band, while also fixing the
coefficient/translation sign convention: positive shifts use negative Fourier
coefficient phases. The constant-ratio counterexample avoids that separate sign
issue. This conclusion is about the stated theorem, not every multi-bias
expressivity result or empirical experiment in the paper.

## 7. Scope of a unifying theory

The representation–tangent–objective framework explains why several seemingly
different methods coincide, why an expanded parameterization can train better,
and which compatibility and certification obligations survive a transformation.
Its ingredients are established. A credible new contribution would specify a
useful class of compatible lifts, a selection rule and a demonstrated advantage
after normalization and complete cost, or prove a new guarantee for that class.

Universal representation does not imply universal efficient learning. On finite
data, two target laws can agree on every observed example and disagree at an
unobserved point. No learner can infer which law holds without additional
assumptions. The manuscript itself also requires exponentially many biases in
its general Boolean construction and retains the ridge restriction for one
unit on continuous multidimensional domains. These limits do not rule out
useful broad neural systems; they rule out inferring universal superiority from
expressivity alone.

The strongest constructive direction is to choose normalized additional
features that correlate with the task residual, generate them through a
compatible field or constraint parameterization, and retain an independent
acceptance check. Compare them with ordinary basis enrichment, output-bias
controls, functional boosting and direct coefficient optimization. This is a
specific research program with a proved local mechanism, not a promise to
replace every learning algorithm.

## Reproduction and validation

From the lab root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/omnibias-geometry-comparators/bin/python -m unification_research.audit \
  --output artifacts/reproduction_unification/checks.json
```

The output must be fresh and records source/dependency fingerprints. Seven new
check groups cover shared and independent gradients, a matched scalar training
fixture, the monotone linear-slack construction, tangent-kernel enrichment,
the spectral theorem counterexample, exact metric/compatibility identities and
the compatible PDE-field lift. A scratch test initially compared expanded and
factored SymPy expressions structurally; it was corrected to test that their
difference expands to zero, and the failure is retained.

The independent audits additionally passed 20 Hermite, 75 core geometry/source,
and seven objective/count tests. These 102 existing checks have different
scopes; they are not a new full-repository test run. Exact rational identities,
symbolic proofs, floating comparisons and performance measurements are labeled
separately. No formal proof-assistant verification is claimed.
