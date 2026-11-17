# Calculus and geometry research opportunities

The four-hour pilot validates fixed representations and their derivatives. It does not evaluate the whole proposed family of learned geometric models. A useful architecture can combine established mathematics in a new way, but the connection between neuron derivatives, geometric sets and differential equations does not itself establish architectural novelty or a learning advantage. This assessment distinguishes those questions and identifies experiments that could answer them.

## What the pilot can establish

| Experiment | Actual construction | What remains outside its scope |
| --- | --- | --- |
| Wave system | Fixed characteristic sine/cosine features; fit output coefficients | Learned directions, biases, derivative dictionaries, interfaces and nonlinear realization maps |
| Noisy tabular data | Fixed artificial affine constraints immediately before a free affine prediction head | Task-conditioned constraints, nonlinear distance features, mixtures of manifolds and PFN pretraining |
| LP and QP | Existing solvers plus sensitivities conditional on the discovered active set | Learned presolvers, active-set proposals and amortized optimization across related problems |

There is no obvious arithmetic defect explaining the reported equivalences. For a fixed affine projection \(h\mapsto Ph+q\) and a free affine head,

\[
W(Ph+q)+d=(WP)h+(Wq+d).
\]

That is an algebraic equivalence. The factorization can change optimization or regularization relative to an unconstrained parameterization, but the independent control can reproduce those choices too. The identity does not eliminate nonlinear distance features, input-dependent projections, or constraints on the actual prediction target.

The wider repository has already tested learned band profiles, beyond this pilot. The stronger [profile controls](benchmarks/profile_controls_study.json) include 360 selected evaluations: learned Laplace profiles beat the learned collapsed profile on all six dataset means and 29 of 30 matched splits. Those results argue against treating the particular collapsed profile as intrinsically superior. They do not evaluate geometric PFN priors, cardinality circuits or coherent manifold models.

The precise conclusion is: **the tested constructions did not demonstrate a new mathematical principle or a broad advantage over matched controls; the broader proposed architectures remain research questions.**

## The underlying objects have direct precedents

An exact equality neuron means precisely

\[
E_{w,b}(x)=\mathbf 1\{w^Tx=b\}.
\]

This definition is preserved. It is not replaced by a sigmoid band. [Koiran and Sontag, 1997, printed page 196, Lemma 6](https://www.sontaglab.org/FTPDIR/quadratic-vc.pdf) explicitly use linear and equality gates, with equality output one only at zero. [Kilic, Sima and Bruck, 2024](https://arxiv.org/html/2402.08751v1) define exactly \(\mathbf1\{w^TX=b\}\), compose such gates into circuits, and study geometric representations of \(AX=b\) on Boolean inputs. These sources rule out invention of the predicate itself, without deciding the value of a particular training method.

The calculus connection is also explicit in prior work. For a unit normal w, distributionally,

\[
\Delta\operatorname{ReLU}(w^Tx-b)=\delta_H,
\qquad H=\{x:w^Tx=b\},
\]

where \(\delta_H\) integrates test functions over the hyperplane. For general nonzero w the factor is \(\|w\|\delta_H\). [Parhi and Nowak, 2021, Figure 3](https://jmlr.org/papers/volume22/20-583/20-583.pdf) explicitly maps a ReLU network to hyperplane impulse sheets, then to atoms encoding orientations and biases. A surface measure and an equality indicator share a set but are different mathematical objects.

Normalized bias differences also have a precise scope. For \(\sigma\in C^k\) on a neighborhood of z,

\[
\lim_{h\to0}h^{-k}\sum_{j=0}^{k}(-1)^{k-j}{k\choose j}
\sigma(z+jh)=\sigma^{(k)}(z).
\]

The signed coefficients and normalization matter. Merely making k+1 neurons' biases coincide gives their summed activation, not automatically its kth derivative. These are classical finite differences and confluent representations. A useful implementation might improve conditioning of nearly cancelling neurons, but must compare to stable divided differences and direct derivative-activation features.

For PDE representations, [EPGP](https://proceedings.mlr.press/v202/harkonen23a.html) constructs priors that exactly satisfy systems of linear constant-coefficient PDEs. [Local extreme learning machines](https://arxiv.org/abs/2012.02895) fix hidden parameters and solve output coefficients. These are directly relevant scientific-ML precedents, rather than merely applications of unrelated mathematics.

## A guaranteed useful prediction application

Suppose a vector target y is known to satisfy a nonempty affine constraint set M. For any prediction \(\mu\), orthogonal projection obeys

\[
\boxed{\|\mu-y\|^2=\|\Pi_M\mu-y\|^2+\|\mu-\Pi_M\mu\|^2.}
\]

**Proof.** The correction is normal to M, while \(\Pi_M\mu-y\) is tangent. Expand the squared norm and the cross term vanishes. Thus enforcing a correct target constraint cannot worsen squared prediction error and improves it strictly when the original prediction violates the constraint.

This provides a real use case for conservation-constrained vector regression, compositional measurements and sensor reconstruction. It differs from arbitrary hidden constraints: the equality supplies information about the true target. It is established projection geometry, not a classification-accuracy theorem. If the supplied constraints are wrong, projection can increase error.

For a genuine joint Gaussian prediction \(N(\mu,\Sigma)\), with \(\Sigma\) positive definite and A full row rank, equality conditioning gives

\[
\mu_c=\mu-\Sigma A^T(A\Sigma A^T)^{-1}(A\mu-b),\qquad
\Sigma_c=\Sigma-\Sigma A^T(A\Sigma A^T)^{-1}A\Sigma.
\]

This is a possible integration with tabular foundation models. Separate scalar predictive distributions do not provide a calibrated joint covariance automatically. With uncertain constraints, a noise model should replace hard conditioning. The Euclidean nonincrease theorem above concerns orthogonal projection; a weighted correction instead has the corresponding weighted-metric guarantee.

The first experiment should compare unmodified prediction, ordinary QR correction, nullspace prediction and uncertainty-aware conditioning on tasks with independently justified constraints. Measure squared error, calibration, feasibility, incorrect-constraint sensitivity and total cost. An improvement shared with ordinary QR is a successful application of constraints, but not evidence that bias derivatives supplied the improvement.

## Geometry inside a TabPFN style model

TabPFN learns an inference procedure from synthetic tasks. A plausible research route is to change those tasks and the information available during inference. The [official repository](https://github.com/PriorLabs/TabPFN) currently defaults to TabPFN-3.5; the [extensions](https://github.com/PriorLabs/tabpfn-extensions) already include embeddings, outlier detection, generation and imputation. These capabilities must be controls rather than claimed first applications.

Consider a task prior

\[
x=g_\omega(z)+\epsilon,\qquad
r_\omega(g_\omega(z))=0,\qquad y=h_\omega(z).
\]

Vary codimension, curvature, orientation, disconnected regimes, measurement noise, missingness and the predictive mechanism. A transformer could infer a task-specific manifold and use normal distances, tangent coordinates or derivative summaries as context. This differs from adding a fixed projection to a final head.

There is an exact representation rule for affine constraints. For full-row-rank A define

\[
P_N=A^T(AA^T)^{-1}A,\qquad q=A^T(AA^T)^{-1}b.
\]

The pair \((P_N,q)\) determines M by \(P_Nx=q\) and is unchanged by \((A,b)\mapsto(SA,Sb)\) for invertible S. Conversely, the same nonempty affine set has the same normal projector and minimum-norm anchor, so this pair is a complete invariant of that set in fixed Euclidean coordinates. This is established linear algebra. Using these quantities as task context could prevent sensitivity to arbitrary rescaling or recombination of the same equations. It does not provide invariance to arbitrary changes of input units; those require an explicit metric. A dense projector costs quadratic storage, so a scalable model needs an appropriate factorization or invariant contractions.

**Hypothesis:** a prior over noisy geometric mechanisms, with equation-invariant context, improves sample efficiency and robustness on related unseen tasks. Compare the same architecture and training budget with: the same synthetic tasks without geometric context; ordinary prior tasks; ordinary AD computing identical context; and current TabPFN with feature engineering. Hold out entire geometric families, not only samples from seen generators. If the same prior without derivative features gets the same gain, attribute the result to the prior.

Learning a full PFN is an expensive later stage. First test a constrained output adapter and a small task-conditioned model. Affine correction needs no bias derivatives; nonlinear jet extraction is where omnibias could supply a measurable implementation contribution.

## Manifold learning clustering and anomaly detection

Let \(r_\theta(x)\in\mathbb R^m\) define a regular learned zero set with full-row-rank Jacobian J. The score

\[
s_\perp(x)=r_\theta(x)^T(JJ^T)^{-1}r_\theta(x)
\]

is exact squared distance for affine residuals and a local approximation for curved ones. On the manifold,

\[
P_T=I-J^T(JJ^T)^{-1}J
\]

gives tangent directions. Second derivatives provide curvature. Derive these quantities from one field; independent normal and Hessian outputs need compatibility checks.

**Hypothesis:** coherent curvature and normal geometry help distinguish touching manifolds or detect faults that move across a normal operating manifold. Useful stress cases include equal tangent planes with different curvature, intersecting operating regimes, small normal faults and large tangent displacements. A residual that learns to be identically zero is a degenerate solution, so reconstruction or contrastive information and rank controls are necessary.

A distance-only anomaly score misses rare points that lie on the manifold. It needs a density model along the manifold and a noise scale in the normal directions. Multiple charts also need regime probabilities and overlap handling. A sum of convenient scores should not be advertised as a normalized likelihood without deriving its change-of-variables and volume factors.

Existing methods include the [Manifold Tangent Classifier](https://proceedings.neurips.cc/paper_files/paper/2011/hash/d1f44e2f09dc172978a4d3151d11d63e-Abstract.html), [Deep SVDD](https://proceedings.mlr.press/v80/ruff18a.html), [DROCC](https://proceedings.mlr.press/v119/goyal20c.html), and [Neural Manifold Clustering and Embedding](https://arxiv.org/abs/2201.10000). Compare local PCA and mixtures of probabilistic PCA as well as learned methods. Measure anomaly recall at fixed false-positive rates, clustering quality, calibration, rank failures and full derivative cost. Compare identical fields evaluated with ordinary AD and omnibias; representation gains and derivative-engine gains are separate findings.

The existing consumer `omnibias-geometry` already contains learned-chart metrics, tangent/normal projectors and curvature machinery. Reuse it. The next research contribution would need a specific training rule, reliable geometric certificate or measured cost/accuracy gain beyond these controls.

## Exact equality on Boolean data

The measure-zero objection for continuous inputs does not apply to the Boolean cube. The equality

\[
E_k(x)=\mathbf1\left\{\sum_{i=1}^n x_i=k\right\}
\]

selects an entire Hamming-weight class. In particular, \(E_1(x_1,x_2)\) is XOR.

There is a direct finite bias-stencil construction:

\[
\boxed{\tau(z)=\operatorname{ReLU}(z+1)-2\operatorname{ReLU}(z)
+\operatorname{ReLU}(z-1)=\max(0,1-|z|).}
\]

Therefore \(\tau(z)=\mathbf1\{z=0\}\) for every **integer** z. The proof follows by evaluating its four linear pieces and their endpoints. Affine residuals with integer weights and integer threshold/bias thus have an exact three-ReLU equality realization on Boolean inputs. On real noninteger inputs it is a tent, not an equality detector. Integrality, or a separately proved gap between zero and other attainable residuals, is essential. Learning unrestricted real weights does not preserve that condition.

This classical B-spline identity gives a concrete model candidate: small trainable counting blocks, with an exact hard implementation and a well-defined smooth expectation during training. For independent Bernoulli inputs,

\[
F_k(p)=\mathbb E[E_k(X)]=[t^k]\prod_i(1-p_i+p_it),
\]

\[
\partial_i F_k=\Pr(S_{-i}=k-1)-\Pr(S_{-i}=k).
\]

**Proof.** Condition on \(X_i\); the expectation is affine in \(p_i\). A count dynamic program evaluates a specified k in \(O(n(k+1))\) arithmetic operations. Every symmetric Boolean function has the representation \(\sum_{k=0}^n a_k E_k(x)\), using n+1 count-response entries instead of an unrestricted \(2^n\)-entry LUT. That saving applies to symmetric functions; learning which inputs form a useful group remains part of the task.

The expectation is a differentiable function of probabilities. It does not assign an ordinary derivative to the hard equality predicate. Exact expected derivatives can also vanish: parity has zero first derivatives at the fully symmetric probabilities for n greater than one.

Cardinality inference is established; [Tarlow et al., 2012](https://proceedings.mlr.press/r10/tarlow12a.html) gives fast exact methods. A fair comparison includes an independent count DP, hard popcount plus lookup, optimized counting circuits, [DiffLogic](https://github.com/Felix-Petersen/difflogic), [convolutional DiffLogic](https://arxiv.org/abs/2411.04732), and [DiffLUT-Net](https://arxiv.org/abs/2609.09254), submitted 8 September 2026 with learnable six-input LUT functions and connectivity. Charging a counting block as one free operation would hide its hardware cost.

**Hypothesis:** learned small-support counting blocks improve sample efficiency or synthesized area/latency on unknown-support cardinality tasks, while generic LUT blocks handle nonsymmetric interactions. Include noisy counting, parity, count intervals and deliberately nonsymmetric tasks. Report hard accuracy, sample efficiency, training cost, hardening gap, synthesized size and latency. A benefit confined to counting tasks is a useful scoped result.

### A further calculus and discrete difference identity

For a sufficiently differentiable scalar profile, successive Boolean differences of a ridge are finite bias differences. Repeated use of the fundamental theorem of calculus gives

\[
\sum_{S\subseteq\{1,\ldots,k\}}(-1)^{k-|S|}
\phi\left(z+\sum_{i\in S}w_i\right)
=\left(\prod_{i=1}^k w_i\right)
\int_{[0,1]^k}\phi^{(k)}\left(z+\sum_iw_it_i\right)dt.
\]

This is an exact bridge between calculus and Boolean interactions. At finite spacing the integral matters; substituting one point derivative is generally wrong. It suggests derivative-based interaction diagnostics for structured ridge models. It does not remove the difficulty of general high-dimensional integration or arbitrary Boolean optimization.

### Shared inputs require joint information

Local multilinear gates can lose the global independent-input expectation interpretation when branches reconverge:

\[
\operatorname{softXOR}(p,p)=2p(1-p),\qquad
\mathbb E[X\mathbin{\mathrm{xor}}X]=0.
\]

Likewise soft AND gives \(p^2\), while \(\mathbb E[X\land X]=p\). These are limitations of that probabilistic interpretation, not errors in a deliberately chosen vertex-exact relaxation. For distinct Bernoulli variables, the AND discrepancy is their covariance; for XOR it is twice the covariance. Propagating pairwise covariances alone does not in general close a deep circuit's moment equations.

A candidate is to detect shared inputs and evaluate selected small subcircuits exactly, with cardinality DP or truth-table contraction. A resource-bound certificate could track the remaining approximation. Compare [DeepProbLog](https://proceedings.neurips.cc/paper/2018/hash/dc5d637ed5e62c36ecb73b654b05ba2a-Abstract.html), [Scallop](https://github.com/scallop-lang/scallop), weighted model counting and Boolean-quotient simplification. Exact inference for unrestricted circuits remains hard; a tractable structural restriction is part of the proposed result.

## Nonlinear PDEs can have finite nonlinear solution families

The failure of finite linear spaces to close under arbitrary nonlinear products does not forbid finite **nonlinear** solution manifolds. For \(\nu>0\), let

\[
v(x,t)=c_0+\sum_{k=1}^K e^{-\nu k^2t}
\bigl[a_k\cos(kx)+b_k\sin(kx)\bigr],
\quad c_0>\sum_k(|a_k|+|b_k|).
\]

Then \(v_t=\nu v_{xx}\) and \(v\ge\delta>0\) for \(t\ge0\). Set

\[
u=-2\nu v_x/v.
\]

Writing \(\ell=\log v\) gives \(\ell_t=\nu(\ell_{xx}+\ell_x^2)\). Differentiate once in x and substitute \(u=-2\nu\ell_x\) to obtain

\[
u_t+u u_x=\nu u_{xx}.
\]

Linear coefficient dynamics thus produce a nonlinear Burgers solution, with exact periodicity and an explicit denominator certificate. The represented initial data are restricted to those obtained from this positive finite heat family; the construction does not contain every Burgers solution. This is classical Cole–Hopf, and learned linearizing coordinates already appear in [Gin et al.](https://arxiv.org/abs/1911.02710).

A worthwhile compiler target is a nonlinear realization map accompanied by domain, boundary, derivative-compatibility and error-transfer obligations. Its control must include the direct transformed spectral solver. A more general certified transformation, a new closure class, or better adaptive representation could be a result; reproducing Cole–Hopf alone is not one.

## Optimization and large geometric models

For related LP instances, a neural model could propose an active set, followed by a linear solve and primal/dual feasibility checks. A valid optimality certificate can accept a correct proposal; otherwise a conventional solver supplies a fallback. Measure proposal cost, certificate cost, fallback frequency and total solve time against HiGHS with appropriate warm starts. Encoding constraints as hyperplanes does not discover the active set or remove Boolean integrality. Weighted cardinality with unrestricted integer weights also recovers subset-sum structure; bounded-weight DP is pseudopolynomial.

For a large geometric model, specify the task: reconstruct shapes, predict physical fields on shapes, or infer constraint systems. These need different data and architectures. [LaGeM](https://arxiv.org/abs/2410.01295) already studies large generative geometric models; [GATr](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6f6dd92b03ff9be7468a6104611c9187-Abstract-Conference.html) provides an equivariant geometric transformer. Neural zero-set geometry and derivative-based shape learning also appear in [IGR](https://proceedings.mlr.press/v119/gropp20a.html) and [Exploring Differential Geometry in Neural Implicits](https://arxiv.org/abs/2201.09263).

An omnibias contribution could be a decoder producing one field with coherent values, normals and curvatures, plus checked local regularity and supported constraints. The proposed gain is reliable constrained adaptation or cheaper high-order evaluation at matched error. Large-scale pretraining is premature until this decoder has an advantage over ordinary implicit fields and the same derivative algebra.

## Recommended research sequence

| Priority | Question | Evidence required to continue |
| --- | --- | --- |
| 1 | Do equality/counting blocks help Boolean learning? | Gains against identical DP and current logic/LUT models at matched hard accuracy and hardware cost |
| 2 | Can true output constraints improve tabular predictions reliably? | Target-relevant constraints, calibrated uncertainty and robustness to constraint error; distinguish ordinary projection gains |
| 3 | Do coherent curvature features help manifold faults or clustering? | Gains over local geometry and learned controls at matched false-positive rate and total cost |
| 4 | Can a geometric PFN prior transfer beyond its synthetic generators? | Held-out mechanism families, matched priors/architectures, current TabPFN controls and a clear role for derivatives |
| 5 | Can a compiler certify nonlinear PDE realizations or continuous-parameter dictionaries? | A complete restricted theorem and comparison to the direct transformed solver or identical optimization algebra |

These are proposed studies, not results from the four-hour pilot. No blanket conclusion about all omnibias projects follows from the fixed-projection experiment. The strongest theory targets are a precisely stated representational separation, a stability/sample-complexity theorem under explicit assumptions, or a new tractable certificate. The strongest engineering targets are a measurable advantage in a defined task and a useful implementation of a guarantee that users can independently check.
