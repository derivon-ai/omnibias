# Two independently trainable sigmoid biases: a positive gradient result

Follow-up: [capacity, exact threshold counting, repository review, and
architecture opportunities](multi-bias-capacity-and-architecture.md).

The user's specific unit

\[
f(x)=\sigma(wx+b_1)-\sigma(wx+b_2)
\]

does avoid a peak-stationarity failure of a fixed-height Gaussian. A fresh
20 October 2026 experiment reproduces this. The earlier Gaussian-half
decomposition in the [unification audit](unification-research-audit.md) used
different primitives; its zero individual bias gradients at the peak do not
apply to these sigmoid branches.

The improvement is an independently trainable shape direction. It does not
require the common input gradient to become nonzero, and the same function
written as one compact expression learns identically under the same parameters.

## The mathematical mechanism

Let \(t=wx+m\), \(m=(b_1+b_2)/2\), and \(a=(b_1-b_2)/2>0\). Then

\[
f(t,a)=\frac{\sinh a}{\cosh t+\cosh a},\qquad f(0,a)=\tanh(a/2).
\]

The two derivatives are

\[
f_t=-\frac{\sinh a\sinh t}{(\cosh t+\cosh a)^2},\qquad
f_a=\frac{1+\cosh a\cosh t}{(\cosh t+\cosh a)^2}.
\]

At the peak, \(f_t=f_m=0\), but
\(f_a=1/(1+\cosh a)>0\). In the original parameters,

\[
f_{b_1}=\sigma'(wx+b_1)>0,\qquad
f_{b_2}=-\sigma'(wx+b_2)<0.
\]

The shared input receives their sum; the two independent biases receive their
individual derivatives. Gradient descent can move the biases in opposite
directions, changing the separation and therefore both width and peak height.

There is a stronger exact statement. For one finite-input observation and
\(L=(f-y)^2/2\),

\[
\|\nabla_{b_1,b_2}L\|^2
=(f-y)^2\{[\sigma'(wx+b_1)]^2+[\sigma'(wx+b_2)]^2\}>0
\]

whenever \(f\ne y\). Thus independently trainable sigmoid biases eliminate
finite single-example stationary points of this type in real arithmetic.
This theorem does not claim a uniform gradient lower bound, feasibility of
every target, or convergence at a prescribed rate. Derivatives approach zero
in saturation, floating arithmetic can produce zeros, and gradients from
different examples can cancel.

## Frozen small experiment

The configuration was saved before evaluating results. CPU, one thread,
float64. The peak experiment uses ordinary SGD with learning rate 0.5 and
1,000 steps; the separate curve fits use Adam at 0.03 for 1,000 steps.

At the peak, use input \(x=1\), shared weight \(w=1\), center bias \(m=-1\),
and half-gap \(a=1\). The initial prediction is
\(H=\tanh(1/2)=0.46211715726\); the target is 0.2. The matched Gaussian is
\(H\exp[-\kappa(wx+b)^2]\), with
\(\kappa=1/[2(1+\cosh1)]\). It matches the sigmoid bell's initial peak
height and local curvature. The entire profiles are different functions.
An unscaled unit-height Gaussian has the same exact peak-stationarity argument,
but would start with a different error; the scaled control avoids that confound.

| Model | Initial gradient | Final prediction | Final half squared error |
| --- | --- | ---: | ---: |
| Fixed-height Gaussian, train weight and center | (0, 0) | 0.462117 | 0.0343527 |
| User's two-sigmoid unit, train weight and two biases | (0, 0.0515354, -0.0515354) | 0.200000 | 9.86e-32 |
| Same sigmoid bell, compact formula and identical parameters | Same to roundoff | 0.200000 | 8.67e-32 |
| Gaussian with trainable amplitude | (0, 0, 0.262117) | 0.200000 | 0 |
| Peak-normalized sigmoid bell | (0, 0, 0) | 0.462117 | 0.0343527 |
| Sigmoid bell with fixed bias separation | (0, 0) | 0.462117 | 0.0343527 |

The learned sigmoid biases are approximately \(-0.594535\) and \(-1.405465\).
Their midpoint remains -1. The shared weight remains one. The method succeeds
by changing the gap and peak height, without relocating the center.

### Why normalization is an informative control

The peak-normalized bell is

\[
q(t,a)=H\frac{1+\cosh a}{\cosh t+\cosh a}.
\]

Its value at the peak is always H, and both \(q_t\) and \(q_a\) vanish
there. The useful escape direction disappears when peak height is fixed.

More generally, if a differentiable family has a smooth interior maximizer
\(x_*(\theta)\) and constant maximum height M, differentiating
\(F(x_*(\theta),\theta)=M\) gives \(\partial_\theta F=0\) at that maximum
because \(\partial_xF=0\). This requires the stated differentiability and
interior-selection assumptions. Allowing height to vary avoids this obstruction.

Independent-bias SGD also differs from naive center/gap SGD: learning rate
\(\eta\) for \(b_1,b_2\) corresponds to \(\eta/2\) for m,a. The compact
control retained the original parameters, so it did not introduce this metric
confound. No general invariance under coordinate changes is claimed for Adam.

## Whole-curve fitting

Twenty-four additional fits use two noiseless target families and three fixed
seeds, 2303, 2311 and 2333. Each fit has 81 training points on [-4,4] and 400
distinct evaluation points between them. No hyperparameter or checkpoint was
selected on the evaluation grid. This is function approximation, not a
statistical generalization benchmark.

The targets are

\[
y_G(x)=H e^{-((x-0.7)/1.15)^2},\qquad
y_S(x)=\sigma(1.4(x+0.4)+1.4)-\sigma(1.4(x+0.4)-1.4).
\]

All methods start with the same shared weight and center for each seed; the
Gaussian and bell initially match peak height and local curvature. A trainable
Gaussian amplitude supplies a three-parameter control against the
three-parameter sigmoid bell. The fixed-height Gaussian has only two parameters.

| Model | Gaussian target mean off-grid RMSE | Sigmoid-bell target mean off-grid RMSE |
| --- | ---: | ---: |
| Fixed-height Gaussian | 5.47e-17 | 0.0618531 |
| Trainable-amplitude Gaussian | 6.46e-17 | 0.0069470 |
| Two-sigmoid unit | 0.0063965 | 7.85e-17 |
| Compact identical sigmoid bell | 0.0063965 | 9.12e-17 |

The Gaussian learns its complete target curve from all three initializations;
nonmonotonicity alone does not cause universal failure. Each matched function
family fits its own target accurately. The sigmoid unit also learns its full
curve. The split and compact implementations agree, so the result is consistent
with the independently trainable parameter mechanism rather than a computation
graph exception to the chain rule.

Elapsed fit times are recorded as diagnostics, not as a repeated competitive
timing comparison. No broader model accuracy or speed claim follows from these
small noiseless fixtures.

## Reproduction and checks

Research source: `omnibias-lab/two_bias_research/run.py`.
Results: `omnibias-lab/artifacts/two_bias_peak_20261020`.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/omnibias-geometry-comparators/bin/python -m two_bias_research.run \
  --output artifacts/reproduction_two_bias
```

Run from the lab root. Existing output directories are refused. Registration
records configuration, source hash and dependency versions before fitting.
The analytic oracle, ordinary autograd and central differences agree on 15
input/gap cases. Checks also cover the center/gap metric, an exactly cancelling
batch, all six peak controls, and split/compact agreement on all six curve
tasks. These are mathematical and floating-point checks, not interval proofs.

The supported conclusion is positive and specific: **independently trainable
sigmoid shifts provide useful shape directions that a fixed-height Gaussian
lacks at its peak.** Extending that into an architecture advantage requires
tasks where these directions are useful across examples, with ordinary
trainable-amplitude and same-function controls retained.

## XOR and the minimum Boolean bias count

A separate 20 October 2026 follow-up tests the four-parameter unit on all four
XOR inputs and derives its Boolean representation count. This uses **fixed
alternating coefficients**, ordered thresholds, and a fixed final readout at
1/2. The sigmoid values are in (0,1); the readout supplies exact Boolean labels.
Finite sigmoids are not claimed to output exact real zeros and ones.

### XOR construction and training

Let s = x1 + x2 for binary inputs. The unit

\[
F(x)=\sigma(\beta(s-1/2))-\sigma(\beta(s-3/2))
\]

classifies XOR exactly when \(\beta>2\log3\): its value at s=1 is
\(\tanh(\beta/4)>1/2\), and its values at s=0,2 are below 1/2.
The four parameters are two weights and two biases. A conventional 2-2-1 MLP
with biases has nine parameters. This comparison concerns those architectures;
it is not a lower bound on every possible XOR model.

At weights (10,10) and biases (-5,-15), the outputs in order 00,01,10,11 are
approximately (0.006693, 0.986614, 0.986614, 0.006693).

A fresh fixed experiment uses seeds 2401 through 2420, full-batch squared error,
Adam at 0.03, and 1,500 steps. Initial weights are independent standard normal;
the common center bias is minus half their sum. The two-sigmoid half-gap starts
at one. Gaussian controls use the same weights and center; the amplitude
control starts at amplitude one. No failed seeds were removed.

| Model | Trainable parameters | Fits classifying all four inputs correctly |
| --- | ---: | ---: |
| Two sigmoids, two independent biases | 4 | 20/20 |
| Gaussian, fixed amplitude | 3 | 15/20 |
| Gaussian, trainable amplitude | 4 | 18/20 |

These are training successes on a complete four-row truth table, not
held-out generalization results. The controls are explicit initialization
recipes, not a claim of equal function-space initialization. No comparative
timing claim is made. A Gaussian also represents XOR: \(e^{-4(s-1)^2}\)
has outputs approximately (0.018316,1,1,0.018316).

There is still an exact batch stationary point for the two-sigmoid model:
\(w=(0,0), b_1=\log3, b_2=-\log3\) yields constant output 1/2 and zero
full-batch gradient on balanced XOR, with half squared error 1/8. The experiment
checks this cancellation. The one-example nonstationarity theorem above
therefore must not be extended to every training dataset and initialization.

### Exact minimum for a specified Boolean function

Let \(N=2^n\), and sort the cube vertices by a generic projection:

\[
w^\mathsf{T}x_{(1)}<\cdots<w^\mathsf{T}x_{(N)},\qquad
T_f(w)=\sum_{i=1}^{N-1}|f(x_{(i+1)})-f(x_{(i)})|.
\]

For a model with a fixed output complement or initial Boolean level available,

\[
\boxed{K_{\min}(f)=\min_w T_f(w).}
\]

For the stricter model with coefficients beginning +1, then -1, and no output
constant or complement, the exact count is instead

\[
\boxed{K_{\min}^{0}(f)=\min_w\{T_f(w)+f(x_{(1)})\}.}
\]

The extra term supplies a threshold below all samples when the first desired
label is one. It is a threshold cost, not another learned output coefficient.
Forcing an even number of terms and zero limits at both tails would impose
another convention; the formulas here allow either even or odd K.

Proof for steps: each label change needs a threshold, and placing one between
each adjacent opposite-label pair achieves the bound. For the no-constant
model an initial one also requires a step before the first sample. Ties are
admissible only between equal labels; a sufficiently small generic perturbation
breaks those ties without adding transitions.

The same minimum holds for **finite common-slope sigmoid classification**.
For any K-term shifted-sigmoid sum, set q=exp(z). After multiplying
\(F(z)-1/2\) by the positive denominator
\(\prod_j(1+e^{b_j}q)\), its numerator is a polynomial of degree at most K.
It cannot have more than K distinct threshold crossings. The no-constant
model starts below 1/2 at minus infinity, so an initial positive label costs
one additional crossing. Conversely, sufficiently steep sigmoids implement
any finite separated step labeling. This argument concerns classifications,
not interpolation of exact real values 0 and 1.

The universal requirement is a different quantity:

\[
\boxed{B(n)=\max_{f:\{0,1\}^n\to\{0,1\}}K_{\min}(f).}
\]

Define \(B_0(n)\) analogously for the no-constant convention. The maximum
cannot be replaced by the bias count of parity or of one sampled function.

### Valid outputs and a sharp separation guarantee

Ordered thresholds imply
\(1>q_1>\cdots>q_K>0\), where \(q_j=\sigma(\beta(z-t_j))\).
Pairing adjacent terms proves that the fixed alternating sum stays strictly
between zero and one for finite inputs and K>=1. Thus no negative output
coefficients need to be learned to obtain valid probability-range outputs.

If z lies between thresholds t_r and t_(r+1), its error relative to the hard
alternating step is bounded by the two nearest tails:

\[
|F(z)-F_{\rm hard}(z)|
\le\sigma[-\beta(z-t_r)]+\sigma[-\beta(t_{r+1}-z)].
\]

Omit a term when there is no threshold on that side. To prove the bound,
write each left sigmoid as one minus its tail. The alternating left-tail
sum is bounded by the nearest left tail; the alternating right-tail sum is
bounded by the nearest right tail. More distant shifts do not add an extra
factor of K.

If every input score has distance at least delta from every threshold, then

\[
|F-F_{\rm hard}|\le\frac{2}{1+e^{\beta\delta}},\qquad
\boxed{\beta\delta>\log3\ \Longrightarrow\ \text{correct hard labels}.}
\]

This is a sufficient condition uniform in K, and its constant is sharp:
at the midpoint of two thresholds separated by gap g, the bell height is
\(\tanh(\beta g/4)\), exceeding 1/2 precisely when
\(\beta g>2\log3\). Separation alone is insufficient if a sample lies
too near a threshold; the score-to-threshold margin is the relevant quantity.
These are representation guarantees, not lower bounds on every batch gradient.

### Completeness, special cases, and asymptotic bounds

The binary encoding \(z=\sum_{i=1}^n2^{i-1}x_i\) gives distinct scores
0 through N-1. Placing a threshold wherever consecutive labels change proves
Boolean completeness with at most N-1 thresholds when an initial level is
available. That is the sharp worst-case count for this **fixed encoding**,
not necessarily the optimum after choosing weights for the target function.
The no-constant convention also needs at most N-1: if the transition count
is odd, reverse the projection if necessary to start with label zero; if
it is even, it is at most N-2, leaving room for a startup threshold. No
learned coefficients are needed.

Parity is much easier: equal weights give the Hamming count and exactly n
thresholds suffice. Fewer cannot suffice: a K-threshold classifier is the sign
of a degree-K polynomial \(\prod_j(w^\mathsf{T}x-t_j)\); on the Boolean
cube, parity is orthogonal to every polynomial of degree less than n. A
polynomial strictly matching parity's sign would have strictly positive
correlation, a contradiction. Thus parity has minimum n, not N-1.

A counting argument also rules out a universal linear-in-n count. Changes of
projection order occur on the central hyperplanes \(d^\mathsf{T}w=0\),
where \(d\in\{-1,0,1\}^n\setminus\{0\}\), identified up to sign.
There are \(m=(3^n-1)/2\) such hyperplanes. The number of order chambers is
at most

\[
R_n\le\overline R_n=2\sum_{j=0}^{n-1}\binom{m-1}{j}.
\]

For one ordering, at most \(2\sum_{j=0}^K\binom{N-1}{j}\) truth tables
have at most K changes. Consequently universality requires

\[
2\overline R_n\sum_{j=0}^K\binom{N-1}{j}\ge2^N.
\]

The binomial tail bound implies, with q=N-1,

\[
B(n)\ge\max\left\{n,\left\lceil q/2-
\sqrt{q\log\overline R_n/2}\right\rceil\right\},\qquad
B(n)\le2^n-1.
\]

In particular \(B(n)\ge2^{n-1}-O(n2^{n/2})\), and both conventions have
\(B(n),B_0(n)=\Theta(2^n)\). This does not determine an exact closed form
for every finite n. It explains why structured functions can be compact while
unrestricted truth tables eventually need exponentially many shifts.

For the fixed-sign no-constant convention, a sharper direct count is useful:
there are exactly \(\sum_{k=0}^K\binom{N}{k}\) label strings with at most K
changes when the initial label is also counted. Hence a necessary condition is

\[
\overline R_n\sum_{k=0}^K\binom{2^n}{k}\ge2^{2^n}.
\]

Exact integer evaluation gives **B0(8)>=54**. The Fibonacci extrapolation of
1,2,3,5,8 would give 34 at n=8, and therefore cannot be the general minimum
formula. This counting bound is not asserted to be attained.

### Exhaustive small-input verification

An independent computation enumerates **all** projection-order chambers for
n<=4, using rational feasibility witnesses and rational infeasibility
certificates, then examines all truth tables up to input/output symmetries.
A separate standard-library replayer recomputes the certificates and minima.

| Inputs n | Projection orders | Boolean functions | Worst minimum, either convention |
| --- | ---: | ---: | ---: |
| 1 | 2 | 4 | 1 |
| 2 | 8 | 16 | 2 |
| 3 | 96 | 256 | 3 |
| 4 | 5,376 | 65,536 | 5 |

The chamber graph is connected within the convex sector
0<w1<...<wn. Checking every one-sign-flip neighbor either reaches another
listed chamber or produces an exact infeasibility certificate, so this is
an exhaustive proof by computation, not a random-direction search. All other
sectors follow by signed coordinate permutations. The detailed algorithm and
certificate format are preserved with the experiment.

A separate audit removes output-complement symmetry and checks the startup
cost explicitly. It gives the same worst values 1,2,3,5 for fixed alternating
signs with no output constant. For n=4, 864 functions require five biases in
this stricter convention, versus 192 when a fixed output complement is free.

For n=5, all 516 fundamental chambers and 1,981,440 orders were independently
replayed. One fresh truth table, integer 3350833497 under the documented bit
indexing, has exact minimum seven. Thus **7<=B0(5)<=31** is certified here;
equality to eight remains unverified. The search checked 50 random target
functions, not all 2^32 functions. An initial portability failure and its
corrected fresh run are disclosed with the retained artifacts.

### Five-input claim in the supplied manuscript

The printed witness on page 25 does **not** establish that eight thresholds
are necessary. In the paper's own convention, x_k is +1 when bit k-1 of the
vertex index is set and -1 otherwise, the weights

\[
w=(971,578,753,979,17)
\]

and three thresholds

\[
(-1775,175,2125)
\]

reproduce all 32 printed labels with the fixed +,-,+ pattern. Every score is
at least 9 away from every threshold. Thus this is also a robust finite-sigmoid
counterexample to the alleged eight-threshold requirement for that witness.
The exact integer check and input-index conversion are retained. This does
**not** disprove B(5)=8; it invalidates that particular claimed lower-bound
certificate. A different hard witness and a universal upper-bound proof are
needed to establish equality.

The random-direction procedure described in Table 13 also cannot certify
its sampled minima as lower bounds. Minimizing over a subset of directions
can overestimate a function's true minimum. Exhausting truth tables alone
does not repair missing projection directions. The n<=4 result above avoids
this issue by enumerating every chamber.

### Direct prior art and interpretation

[Bohossian and Bruck, Multiple Threshold Neural Logic, 1997](https://proceedings.neurips.cc/paper_files/paper/1997/file/0966289037ad9846c5e994be2a91bafa-Paper.pdf)
uses a shared weighted sum with multiple transitions, explicitly explains
unrestricted Boolean completeness using binary weights, and gives the
Hamming-count parity construction.

[Duch, K-Separability, 2006](https://is.umk.pl/~duch/pubs/06-k-rep-LNCS.pdf)
explicitly gives the difference of two sigmoids on one shared projection and
the n+2 versus 2(n+1)+3 parameter comparison. For n=2 this is the same four
versus nine comparison. Its projection-order viewpoint also matches the
transition-count problem. These are direct precedents for this architecture
and completeness mechanism; the successful XOR experiment remains valid.

The fresh contributions here are reproducible checks, a precise contract for
bias counting, and a correction of the printed five-input certificate. No
new closed form for the unrestricted optimized minimum has been established.

### Follow-up reproduction and retained evidence

From `omnibias-lab`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/omnibias-geometry-comparators/bin/python -m boolean_bias_research.run \
  --output artifacts/reproduction_boolean_xor
/tmp/omnibias-geometry-comparators/bin/python -m boolean_bias_research.gap_checks \
  --output artifacts/reproduction_boolean_gap_checks.json
```

The frozen XOR registration and 60 fits are in
`artifacts/boolean_bias_xor_20261020/`; all 912 scalar gap checks passed.
The self-contained exact enumeration bundle is
`boolean_bias_research/order_audit/`. Its `README.md` supplies generation and
independent replay commands for both output conventions, the five-input
witness, and the printed-witness counterexample. `bundle_manifest.json`
fingerprints all 28 retained files. The counting bound is reproducible using
`counting_lower_bound.py`. No stable package API was changed.
