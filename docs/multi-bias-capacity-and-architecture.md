# Multi-bias capacity, threshold counting, and architectural opportunity

Research review, 20 October 2026. This review separates freshly checked
identities, earlier exact enumeration, repository-reported experiments, and
research proposals. It does not establish a new all-purpose learning theory.

The strongest application direction is to **replace expensive input
projections with small learned scalar profiles, when the data or trained
model supports that replacement**. The corresponding theory should explain
the tradeoff between projection count, profile complexity, depth, margin,
and numerical precision. The multi-bias repository contains empirical
signals worth pursuing, alongside an initialization trap reproduced below.

The [two-bias experiment](two-bias-bell-experiment.md) contains the earlier
peak-gradient, XOR, and minimum-bias enumeration evidence.

The completed [activation-profile compression study](activation-profile-compression.md)
implements this direction with GPU training, exact error-bound replay, public-data
controls and deployment measurements. It reports the storage gains, competitive
limitations and the next experiment supported by the evidence.

## 1. Optimize the ordering; do not assume arbitrary permutations

For Boolean inputs, write \(z_x=w^\top x\). The ordering
\(z_{x_{(1)}}<\cdots<z_{x_{(N)}}\) depends on \(w\). It is not a fixed binary
ordering. With fixed alternating thresholds starting positive, no output
offset, and labels \(y_i=f(x_{(i)})\), the required number for this ordering is

\[
k_w(f)=y_1+\sum_{i=1}^{N-1}|y_{i+1}-y_i|.
\]

The minimum is over **all realizable projection orders**. Same-label ties
are allowed: perturb a tied block slightly without crossing any opposite
label, obtaining a strict order without increasing this count. Opposite-label
ties cannot be classified by any scalar function of the projection.

For the user's example \(f(x_1,x_2)=1-x_2\), choose \(w=(1,-2)\):

```text
projection       -2         -1        -0.5         0          1
input            01         11      threshold     00         10
label             0          0                    1          1
```

Thus \(f=H(x_1-2x_2+1/2)\): one threshold suffices. Choosing \(w=(0,-1)\)
also works, with two same-label tied groups.

Some proposed orders are impossible. In particular,
\(w_2<0<w_1<w_1+w_2\) contradicts \(w_2<0\). All four scores obey
\(z_{00}+z_{11}=z_{01}+z_{10}\). There are eight generic projection orders
for two inputs, rather than all 24 permutations. Optimizing the weights
must respect these dependencies.

## 2. An exact expression for the number of threshold functions

Let \(T(n)\) count affine threshold functions on \(\{0,1\}^n\), including
constants and functions ignoring some coordinates. For a subset
\(S\subseteq\{0,1\}^n\), define

\[
r(S)=\operatorname{rank}\{(x,1):x\in S\}.
\]

Then the classical hyperplane-arrangement identity gives

\[
\boxed{T(n)=\sum_{S\subseteq\{0,1\}^n}(-1)^{|S|-r(S)}.}
\]

**Derivation.** In parameter space \(\theta=(w,b)\), introduce the hyperplanes
\(H_x=\{\theta:\theta^\top(x,1)=0\}\). Each strict realizable sign pattern is
a nonempty convex chamber. Whitney's subset expression and Zaslavsky's
chamber formula give

\[
\chi(t)=\sum_S(-1)^{|S|}t^{n+1-r(S)},\qquad
T(n)=(-1)^{n+1}\chi(-1).
\]

The displayed rank sum follows. No additional factor of two is needed.
On the finite cube, a threshold function represented with boundary points
can also be represented with strict inequalities by slightly moving the
threshold, so this chamber convention loses no Boolean functions.
See [Brysiewicz, Eble and Kühne (2023)](https://link.springer.com/article/10.1007/s00454-023-00557-2),
especially Sections 2 and 6.2.

Fresh exact arithmetic reproduces:

| Inputs \(n\) | \(T(n)\) |
|---:|---:|
| 0 | 2 |
| 1 | 4 |
| 2 | 14 |
| 3 | 104 |
| 4 | 1,882 |
| 5 | 94,572 |

The implementation groups subsets by their exact rational span. A separate
integer-elimination implementation enumerates every subset for \(n\le3\).
Both agree. This is a reproduction of known values, not a new counting
algorithm claim.

The expression is exact but hides substantial combinatorial computation.
It is not a simple closed form involving only elementary functions of \(n\).
The often tempting general-position expression

\[
2\sum_{j=0}^{n}\binom{2^n-1}{j}
\]

is only an upper bound here. It gives 128 for \(n=3\), while the cube has 104
threshold functions. Cube vertices have affine dependencies.

As of this review, a [24 September 2026 preprint by Xie](https://arxiv.org/abs/2609.29756)
states the asymptotic
\(T(n)=2\binom{2^n-1}{n}(1+O(n^{-99}))\). This is a recent preprint result,
not an exact finite-\(n\) formula or a proof independently audited in this
review. A simpler exact formula remains unestablished by our work.

## 3. What two or three biases actually buy

Fix one strict order of \(N=2^n\) inputs. With at most \(K\) thresholds,
alternating signs beginning positive, and no output offset, the exact count is

\[
\boxed{S(N,K)=\sum_{j=0}^{\min(K,N)}\binom Nj.}
\]

**Proof.** Prepend a zero to the label sequence. Choose its transition slots
among the \(N\) available positions. A sequence with \(j\) transitions needs
exactly \(j\) thresholds, including a possible threshold before the first
sample. Sum over \(j\le K\).

For small fixed \(K\) and large \(N\), this is approximately \(N^K/K!\).
The ratio to the single-threshold count **in the same fixed order** is
approximately \(N^{K-1}/K!\), exponential in the input dimension when
\(N=2^n\) and \(K>1\) is fixed. This supports a precise version of the
user's intuition. It is not that ratio against all threshold functions
over every possible projection, nor an exponential guarantee for
optimization, sample efficiency, or generalization.

After optimizing weights, define \(C(n,K)\) as the number of distinct
representable Boolean functions. The earlier exact four-input enumeration
gives:

| Maximum biases | Distinct functions \(C(4,K)\) | Fraction of all 65,536 |
|---:|---:|---:|
| 1 | 1,882 | 2.87% |
| 2 | 11,130 | 16.98% |
| 3 | 46,342 | 70.71% |
| 4 | 64,672 | 98.68% |
| 5 | 65,536 | 100% |

These values are accumulated from the previously replayed minimum-count
histogram; that enumeration was not rerun during this review.

If \(R_n\) is the number of realizable strict projection orders, then

\[
S(2^n,K)\le C(n,K)\le
\min\{2^{2^n},R_n S(2^n,K)\}.
\]

Orders overlap in the functions they realize, so multiplying their counts
does not give an equality. Since an order changes only on
\(w^\top d=0\), \(d\in\{-1,0,1\}^n\setminus\{0\}\), modulo sign,
for \(n\ge1\),

\[
R_n\le2\sum_{j=0}^{n-1}\binom{(3^n-3)/2}{j}.
\]

Sharp control of the overlaps is a concrete combinatorial research question.
Small-\(n\) minimum counts do not imply subexponential worst-case
universality; the counting obstruction in the earlier report remains.

### Connection to finite sigmoids

For ordered thresholds \(t_1<\cdots<t_K\) and common \(\beta>0\),

\[
F(z)=\sum_{j=1}^K(-1)^{j+1}\sigma(\beta(z-t_j))
\]

lies in \([0,1]\). Alternating partial sums of the decreasing sequence
\(\sigma(\beta(z-t_1)),\ldots,\sigma(\beta(z-t_K))\) prove both bounds.
Its hard readout is \(1\{F(z)\ge1/2\}\); finite sigmoid values should not
be called exact Boolean outputs.

Set \(q=e^{\beta z}>0\), \(a_j=e^{\beta t_j}\). Then

\[
2F(z)-1=
\frac{2\sum_j(-1)^{j+1}q\prod_{\ell\ne j}(q+a_\ell)
      -\prod_j(q+a_j)}
     {\prod_j(q+a_j)}.
\]

The numerator has degree \(K\), nonzero leading coefficient, and negative
constant term; the denominator is positive for \(q>0\). Its positive roots,
counted with multiplicity, bound the initial-positive-plus-transition cost
by \(K\). A tangency may create an isolated positive label at the readout
threshold, but consumes root multiplicity two.
Conversely, sufficiently sharp sigmoids realize each finite strict-margin
threshold labeling, allowing \(\beta\) to grow or an equivalent common
rescaling of unconstrained weights and thresholds. A fixed temperature
combined with bounded weights may realize fewer labelings. Thus the
discrete counting model is relevant to the smooth common-temperature
implementation, with this scale qualification.

## 4. What the multi-bias repository supports

Read-only review of `/home/vardan/dev/projects/omnibias_projects/multi_bias`.
The canonical `paper/main.tex` was not modified. Large-scale training was
not rerun; the following are stored repository summaries, not fresh
confirmations. `docs/SCRATCH_RESULTS.md` locates substantial raw outputs on
remote scratch storage.

| Evidence | Repository report | Interpretation |
|---|---|---|
| Shallow versus deep MNIST | Depth-1 K2: 97.70±0.11%, 200,704 binary weights; depth-4 K1: 97.41±0.09%, 397,312 weights | A useful depth/shape tradeoff signal |
| Matched CIFAR10 CNV topology | Baseline 81.89±0.33%; K3 learned 83.79±0.16%; K3 fixed-alt 80.74±0.66% | A +1.90-point learned-profile signal, not a fixed-sign advantage |
| Progressive bias count | Two-seed progressive K did not clearly improve over K1 on CIFAR100/Pets | Generic adaptive K has already been tried |
| Fully exported logic model | Repository distinguishes `final_acc_hard` from the approximately one-point-lower `acc_discretized` | Exported predictions must be the deployment endpoint |

Sources: `paper/main.tex` around lines 781–845;
`rebuttals/reviewer_c3to_repro/reference/expected_numbers.md`;
`rebuttals/progk_summary.json`;
`rebuttals/reviewer_c3to_repro/reference/baseline_comparison.md`.

Binary weights do not imply binary activations. The BNN implementation
uses sums of hard sign functions, with surrogate gradients; learned
coefficients introduce learned real-valued levels. Threshold/coefficient
storage, intermediate precision, and actual inference operations belong
in the comparison. These experiments do not all test the same primitive
as the user's ordered alternating sigmoid unit.

### Freshly reproduced initialization trap

In `multibias/activation.py`, default even K uses all-positive coefficients,
all-zero biases, and normalization by \(1/K\). With unsorted independent
biases and a symmetric optimizer, equality of the biases is invariant:

\[
\frac1K\sum_{j=1}^K\sigma(z+b_j)=\sigma(z+b)
\quad\text{when all }b_j=b.
\]

Every bias receives the same gradient, so deterministic symmetric updates
preserve this equality. The effective optimization rate can differ from a
single bias; the represented activation still has no additional shape.

A fresh 40-step SGD check reproduced exactly zero bias, gradient, and
prediction discrepancies in four cases: sigmoid/ReLU, each using ordinary
autograd or the custom backward. This finding applies to that default,
not all historical K2 experiments. No library fix was made during this
review. A corrective ablation should compare deliberately separated
initialization and function-preserving growth.

Ordered fixed alternating ReLUs are a different class again: their derivative
alternates between 0 and 1, making the profile monotone. Sigmoid Boolean
completeness cannot be transferred to that ReLU class without changing
coefficients or the architecture.

## 5. A research program with explicit mathematical obligations

The broad representation is

\[
f(x)=\sum_{r=1}^{R}a_r g_r(v_r^\top x).
\]

Separate the number \(R\) of input projections, the complexity \(K_r\) of
each scalar profile, depth, and precision. This connects to established
ridge-function, projection-pursuit, and adaptive-activation mathematics.
The research opportunity is an effective allocation and compression method
with guarantees and measured benefits.

### A. Detect when a complex block can be a scalar profile

For square-integrable \(f(X)\), fixed \(v\), and \(Z=v^\top X\),

\[
\inf_{g\text{ measurable}}\mathbb E[(f(X)-g(Z))^2]
=\mathbb E[\operatorname{Var}(f(X)\mid Z)].
\]

The conditional mean attains the infimum. Orthogonally decompose the error
into this conditional variance and
\(\mathbb E[(\mathbb E[f\mid Z]-g(Z))^2]\).

Increasing the bias count can reduce profile approximation error. It cannot
recover information discarded by \(Z\). This classical identity gives a
precise condition for replacing a block with one projection. In continuous
domains, a smooth scalar ridge function also has Hessian rank at most one;
functions such as \(x_1x_2\) cannot equal one such ridge on an open set.

For an injective projection of the finite Boolean cube, unrestricted
measurable \(g\) can memorize every truth table and the conditional variance
is zero. The same issue occurs on a finite training set with distinct
projected samples. The useful restricted identity is

\[
\inf_{g\in\mathcal G_K}\mathbb E[(f-g(Z))^2]
=\mathbb E[\operatorname{Var}(f\mid Z)]
+\inf_{g\in\mathcal G_K}
 \mathbb E[(\mathbb E[f\mid Z]-g(Z))^2].
\]

Here \(\mathcal G_K\) must enforce the actual profile complexity, margin,
or regularity budget. Unrestricted empirical conditional variance alone
cannot identify useful compression.

Research target: a reliable finite-sample criterion selecting which blocks
or channel groups admit this compression, with a bound on its estimation
error. This is more actionable than claiming every deep model can be
flattened cheaply.

### B. Certify a shared-projection replacement

Suppose \(\sigma\) is \(L\)-Lipschitz and \(\|x\|_2\le R_x\). Using Euclidean
norms below,

\[
\left|\sum_j a_j\sigma(w_j^\top x+b_j)
-\sum_j a_j\sigma(v^\top x+b_j)\right|
\le L R_x\sum_j|a_j|\|w_j-v\|.
\]

Apply the Lipschitz inequality termwise and then Cauchy–Schwarz. A downstream
Lipschitz constant multiplies the bound. A scalar classifier whose output
margin exceeds that resulting perturbation retains its labels.

Equal directions give exact sharing. Close directions yield a computable
error bound. Arbitrary scalar coefficients belong to the general profile
class; a strict fixed-sign Boolean unit is a smaller class. Vector-valued
hidden outputs require either factoring their output weights or retaining
separate scalar profiles sharing the same projection.

The inequality is elementary, not claimed novel. A new method would need
to choose and compile the groups effectively, preserve accuracy after
quantization, and improve measured cost against identical-algebra controls.

### C. Grow useful shape directions without changing the initial function

Let \(F(\theta)\in\mathbb R^m\) denote predictions and introduce
\(\widetilde F(\theta,\gamma)\) with
\(\widetilde F(\theta,0)=F(\theta)\) for every \(\theta\). Retain the original
parameter coordinates and Euclidean gradient-flow metric. For
\(L=\frac12\|F-y\|^2\), at insertion,

\[
-\dot L_{\mathrm{extended}}
=-\dot L_{\mathrm{original}}+\|J_\gamma^\top(F-y)\|^2.
\]

**Proof.** The extended Jacobian is \([J_\theta,J_\gamma]\), so its squared
loss-gradient norm is the sum of the two squared norms. This is a local
training statement, not a global convergence or generalization theorem.

A concrete no-op insertion into a real-valued activation is

\[
\sigma(z+c+h)-\sigma(z+c-h),\qquad h=0.
\]

The forward contribution is zero, while its \(h\)-derivative is
\(2\sigma'(z+c)\). A new center \(c\) can add a residual-aligned shape
direction. Reusing an existing center can merely duplicate its tangent
direction. Original input gradients need not become nonzero for an added
shape parameter to help.

The fresh analytic fixture has identical initial predictions and old
gradients, with squared-loss descent rates 0.0480524 and 0.0635998;
the difference 0.0155474 equals the new shape-gradient norm squared.
This illustrates the identity, not an empirical speed benchmark.

Unrestricted insertion is for real-valued activation blocks: it need not
preserve a Boolean unit's \([0,1]\) range after training. Ordered Boolean
growth must additionally enforce its sign/order constraints and account
for the feasible parameter directions. Ordinary SGD after a coordinate
change also changes its metric unless corrected.

Research target: choose new centers by residual correlation per added
deployment cost, then derive finite-step guarantees and test whether this
criterion predicts held-out improvement. Function-preserving growth and
tangent-kernel identities themselves have prior art.

### D. Prune pairs with an exact error bound

For \(a<b\),

\[
\sup_z\{\sigma(\beta(z-a))-\sigma(\beta(z-b))\}
=\tanh\!\left(\frac{\beta(b-a)}4\right).
\]

Center the coordinate at \((a+b)/2\). The difference equals
\(\sinh d/(\cosh t+\cosh d)\), where
\(d=\beta(b-a)/2\); its maximum is at \(t=0\).

Deleting adjacent opposite-sign pairs preserves alternating order and the
remaining unit's \([0,1]\) range. For disjoint deleted pairs, sum these
suprema to bound the uniform perturbation. Output margins larger than
the bound preserve hard predictions. Include downstream amplification
when this is an internal activation.

Three fresh numerical fixtures check the midpoint value, zero derivative,
and consistency with a dense grid. The proof, not that grid, establishes
the supremum. A small raw gap is insufficient for pruning at a high
temperature: the product \(\beta(b-a)\) matters.

### E. Capacity must be paired with margins and precision

If projected samples occupy a range of length \(R_z\), have \(q\) interior
label transitions, and lie at least \(\delta\) from each separating
threshold, their transition gaps are disjoint and

\[
2q\delta\le R_z.
\]

For cube inputs with \(\|w\|_1=1\), \(R_z\le1\). More transitions constrain
the attainable normalized geometric margin. A sufficient finite-sigmoid
hardening bound is \(\beta\delta>\log 3\), as derived in the earlier report;
this is not a necessary condition for every individual correct classifier.

A finite-class generalization bound also makes the tradeoff explicit.
For fixed \(K\), IID sample size \(m\), and zero-one loss, with probability
at least \(1-\eta\), simultaneously for every function in this class,

\[
R(f)\le\widehat R(f)+
\sqrt{\frac{\log C(n,K)+\log(1/\eta)}{2m}}.
\]

This follows from Hoeffding plus a union bound. Adaptive choice of \(K\)
requires allocating confidence across choices or a corresponding model
selection argument. This is classical, potentially loose, and confined
here to the finite Boolean domain.

For quantized models, total description length matters too: \(B\) bits can
encode at most \(2^B\) models. Representing all \(2^{2^n}\) Boolean functions
therefore requires at least \(2^n\) bits in the worst case. Real parameter
count can conceal this cost.

The sharper theory target is a joint frontier:

\[
(\text{projections},\text{biases},\text{depth},
 \text{margin},\text{weight precision},\text{approximation error}).
\]

An exact unresolved count, a genuinely sharper bound on this frontier, or
a proven compression condition beyond the elementary cases could be a
mathematical contribution.

## 6. Prior art and what would constitute a contribution

| Proposed ingredient | Relevant primary source | Remaining contribution target |
|---|---|---|
| Multiple intervals on one learned projection | [Duch, K-Separability (2006)](https://is.umk.pl/~duch/pubs/06-k-rep-LNCS.pdf) | A sharper capacity/margin result or superior verified algorithm |
| Trainable scalar activation shape | [Agostinelli et al., APL (2014)](https://arxiv.org/abs/1412.6830) | Specific growth/pruning or compression advantage |
| Activation regularization and knot selection | [Unser (2019)](https://jmlr.org/papers/v20/18-418.html), [Lipschitz-constrained learned activations (2024)](https://jmlr.org/papers/v25/22-1347.html) | A new guarantee beyond generic complexity penalties |
| Function-preserving growth | [Net2Net](https://arxiv.org/abs/1511.05641) | Finite-step, cost-aware shape selection with evidence |
| Added Jacobian directions | [Neural Tangent Kernel](https://arxiv.org/abs/1806.07572) | New conclusions beyond the PSD kernel increment |
| Ridge/spline representation theory | [Parhi and Nowak (2021)](https://jmlr.org/papers/v22/20-583.html) | Precise conditions and algorithms for useful compression |
| Depth cannot always be cheaply removed | [Telgarsky (2016)](https://proceedings.mlr.press/v49/telgarsky16.html) | Delineate compressible classes; respect activation assumptions |

The suggested main study is a depth/width/profile-complexity comparison
anchored in the existing MNIST and CIFAR CNV results. Correct the default
symmetry trap in a separate implementation change with its contract tests.
Compare profile growth against ordinary widening, a tied-weight network
implementing the same function, and shared-projection spline/APL controls.
Use validation for all choices. Measure final exported accuracy, actual
bytes, intermediate precision, and measured inference cost.

Proposed advancement criterion: independently reproduce at least a twofold
reduction in a specified deployment cost at matched held-out accuracy,
including thresholds, coefficients, and activation precision. This is a
research gate, not a result already obtained. Do not infer inference speed
from counts of binary weights.

## 7. Reproduction and evidence classification

The [learning, image and margin follow-up](multi-bias-learning-and-margin.md)
adds a constructive one-dimensional approximation bound, exact-nesting
initializations, an independently checked maximum-margin XOR example, and
from-scratch image diagnostics. It also qualifies the fixed-order
“exponential” capacity comparison by margin and by its projection-order scope.

Fresh research source under
`/home/vardan/dev/projects/omnibias_projects/omnibias-lab`:

```bash
python -B -m boolean_bias_research.threshold_count_audit \
  --output /tmp/threshold-counts-fresh.json

python -B -m boolean_bias_research.bias_profile_audit \
  --repository /home/vardan/dev/projects/omnibias_projects/multi_bias \
  --output /tmp/bias-profile-fresh.json
```

Recorded environment:
`/tmp/omnibias-geometry-comparators/bin/python`, Python 3.12.13.
The counting program uses only the standard library. The profile audit uses
Torch 2.14.1, float64, one CPU thread, deterministic fixtures without random
initialization. Recorded outputs:

```text
artifacts/multi_bias_theory_20261020/threshold_counts.json
artifacts/multi_bias_theory_20261020/profile_checks.json
```

Both record source/dependency information and reject existing output paths.
The profile audit fingerprints the inspected package sources. The count
audit records diagnostic timings; no competitive timing claim is made.
Both audits pass. Ruff checks and formatting pass for both research scripts.

Earlier exact capacity evidence remains under:

```text
boolean_bias_research/order_audit/fixed_sign_results/fixed_sign_summary.json
boolean_bias_research/order_audit/fixed_sign_results/fixed_sign_replay.json
```

| Finding | Classification |
|---|---|
| Exact threshold rank sum and known counts | Established mathematics, reproduced |
| Fixed-order capacity formula and large low-K gains | Supported combinatorial result; no novelty claim |
| Current default even-K symmetry trap | Freshly reproduced implementation limitation |
| Extra shape-direction descent identity | Established-style local result, checked |
| Exact pair pruning and elementary direction-sharing bound | Supported mathematical tools; novelty unestablished |
| Existing shallow/CNV empirical gains | Repository-reported architectural signals; fresh confirmation needed |
| Adaptive profiles with certified compression and actual deployment gains | Research candidate |
| Simple exact all-n threshold-count formula | Not established |
| Universal cheap replacement of depth or guaranteed generalization improvement | Unsupported |
