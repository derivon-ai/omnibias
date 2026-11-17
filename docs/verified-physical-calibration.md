# Verified physical calibration: evidence report

This bounded CPU study implements guaranteed scalar parameter-set recovery for the stated clamped nonlinear boundary-value problem. All measurements are synthetic, generated from separately certified reference solutions. It compares neural, spectral and local reduced-basis proposals under the same mathematical assurance. The earlier [calculus–geometry campaign](calculus-geometry-campaign.md) is preserved separately.

The integrated matrix recorded 108/108 trials; the point-sensor matrix recorded 54/54. Discovery promotion gate: **NOT MET**. Full details and exact uncertainty are retained in the [machine-readable summary](benchmarks/verified_physical_calibration.json).

The primary neural recipe certified **0/6 case/noise groups**, each requiring three successful repetitions. The fastest fully completed conventional recipe was **spectral / bisection**, at **3.06 seconds median cold cost**. The useful outcome is a replayable physical-parameter recovery contract and explicit identifiability/conditioning arguments. The registered evidence does not establish a neural competitive advantage.

## Mathematical contract

The study assumes the one-dimensional boundary-value problem

\[
 u''''+\lambda u^3=384,\qquad
 u(0)=u'(0)=u(1)=u'(1)=0,\qquad0\le\lambda\le512.
\]

For each parameter, the strictly convex coercive functional
\(E_\lambda(u)=\tfrac12\int|u''|^2+\tfrac\lambda4\int u^4-384\int u\)
on \(H^2_0(0,1)\) has a unique minimizer. The direct method and the
one-dimensional Sobolev embedding give existence; strict convexity gives
uniqueness. Its Euler equation and regularity yield the classical solution.
It is nonzero because the forcing is nonzero.

The earlier [global energy proof](calculus-geometry-campaign.md#a-global-monotone-energy-alternative)
gives, for any exactly clamped \(C^4\) candidate \(v\),

\[
 \|u-v\|_\infty\le C\|v''''+\lambda v^3-384\|_\infty,
 \qquad C=\frac1{96\sqrt{15}}.
\]

For completeness, the unperturbed clamped Green kernel is, for \(x\le y\),
\(G(x,y)=x^2(1-y)^2[3y-x(1+2y)]/6\), extended by symmetry.
The bracket is nonnegative because its minimum on \(0\le x\le y\)
is \(2y(1-y)\). Direct rational identities check the boundary conditions,
continuity through order two, unit third-derivative jump,
\(\max_xG(x,x)=1/192\), and \(\iint G=1/720\).
The reproducing-kernel inequality yields
\(\|e\|_\infty\le\|e''\|_2/\sqrt{192}\).
For \(f=\operatorname{sign}(e)\), Cauchy–Schwarz in the clamped energy
space and positivity of this unperturbed kernel give
\(\|e\|_1=\langle f,e\rangle\le\|e''\|_2
\sqrt{\langle f,(D^4)^{-1}f\rangle}\le\|e''\|_2/\sqrt{720}\).
Testing \(e''''+q e=-r\), \(q\ge0\), against \(e\) gives
\(\|e''\|_2^2\le\|r\|_\infty\|e\|_1\).
Combining the inequalities proves the displayed constant, including the
zero-error case by continuity. The same proof applies to nonnegative bounded
potentials without assuming positivity of their Green kernels.

The argument applies to the error equation with nonnegative potential
\(\lambda(u^2+uv+v^2)\). The inverse of
\(L_\lambda=D^4+3\lambda u^2\) has the same bound. The implementation
uses a rational upper bound verified by \(138240C_{up}^2\ge1\).
No positivity of the perturbed Green kernel is assumed. Parameter-interval
certificates enclose the residual for every coupling in the stated interval.

### Identifiability from an integrated observation

Define the quartic-potential observable

\[
 H(\lambda)=\frac14\int_0^1u(x;\lambda)^4\,dx.
\]

The implicit function theorem applies because \(L_\lambda\) is coercive
and invertible. Derivatives at the prior boundary are interpreted one-sided. The physical sensitivity \(s=\partial_\lambda u\)
satisfies \(L_\lambda s=-u^3\) with homogeneous clamped boundaries.
Therefore

\[
 H'(\lambda)=-\langle u^3,L_\lambda^{-1}u^3\rangle<0.
\]

Strict negativity follows from positive definiteness and \(u^3\ne0\).
Thus \(H\) is continuous and strictly decreasing and has a unique inverse
on its attainable range. This is a deliberately chosen distributed observable.
The assumed measurement error bound is on the integral itself; it is not an
inferred error bound from an arbitrary sparse sampling of a noisy field.

Let \(I=\int u^4\). Cauchy–Schwarz in the \(L_\lambda\) energy inner
product gives

\[
 -H'(\lambda)\ge\frac{\langle u^3,u\rangle^2}{\langle u,L_\lambda u\rangle}
 =\frac{I^2}{384\int u+2\lambda I}.
\]

The denominator identity follows by testing the original equation with \(u\).
It is strictly positive. A certified lower bound on \(I\) and upper bound on
the denominator therefore give a computable slope bound.

For a parameter bracket \([a,b]\), any certified observation witness at
\(q\ge b\) supplies \(I(\lambda)\ge4H(q)\). The energy estimate with
candidate zero gives the unconditional bound \(\|u\|_\infty\le B_0=C_{up}384\).
Consequently the implemented local bound

\[
 m_{[a,b]}=\frac{(4H(q)_{lo})^2}{384B_0+2bB_0^4}
 \le -H'(\lambda)\quad(\lambda\in[a,b])
\]

is sound whenever the numerator is positive. Every use records and replays
its right-hand witness. No empirical derivative estimate justifies exclusion.

### State, quadrature and physical-sensitivity uncertainty

If \(\|u-v\|_\infty\le\delta\) and both fields have magnitude at most
\(B\), then

\[
 |H(u)-H(v)|\le B^3\delta.
\]

Polynomial candidate integrals are exact rational integrals. Neural candidate
integrals use midpoint Taylor quadrature: on a cell of width \(h\),

\[
 \int g=h g(c)+\frac{h^3}{24}g''(c)+R,
 \qquad R\in\frac{h^5}{1920}\,g^{(4)}([c-h/2,c+h/2]).
\]

Here \(g=v^4/4\); all derivatives belong to the same field. Directed interval
arithmetic encloses the remainder over the whole cell. Quadrature uncertainty
and state error are stored separately. A state target alone is insufficient
for an accurate observation and therefore cannot terminate the observation
refinement ladder.

For any clamped sensitivity candidate \(z\), let
\(\eta\ge\|z''''+3\lambda v^2z+v^3\|_\infty\) and
\(S\ge\|z\|_\infty\). Residual subtraction gives the independently
replayed physical-sensitivity bound

\[
 \|s-z\|_\infty\le C_{up}
 [\eta+6\lambda_{max}B\delta S+3B^2\delta].
\]

It concerns the physical solution map. The derivative of the numerical fitting
algorithm is a different object. Floating sensitivity approximations may propose
queries; only interval observations and proved slope bounds accept updates.

### Recovering a complete parameter set

For observation interval \(Y=[y_-,y_+]\), monotonicity makes its feasible
preimage an interval, possibly empty or clipped at the prior boundaries. The
algorithm brackets the clipped inverse of each endpoint. A certified point
observation strictly above or below a threshold excludes the appropriate side.
When the threshold lies inside the observation enclosure, a valid slope bound
gives \(|\lambda_* - q|\le\max|H(q)-y|/m\). Rational outward rounding
preserves containment.

The outer feasible interval uses the outer endpoint brackets; an inner interval
is reported only where both inverse brackets prove membership. A separate
attainable-range test proves feasibility before the integrated recovery can be
`CERTIFIED`. Numerical endpoint uncertainty must total at most \(10^{-3}\)
in coordinates \(\lambda/512\). Measurement uncertainty can still leave a
wide physical parameter interval. Budget or precision exhaustion is
`INCONCLUSIVE`; it is never evidence that no solution exists.

Sparse sensors at \(1/4,1/2,3/4\) use complete parameter subdivision with
uniform field bounds. A cell is excluded only when one sensor range is disjoint
from its measurement interval. A cell is included only when all its ranges are
contained. All unresolved components remain. Replacing a parent requires two
valid children. `CERTIFIED` for this separate cover means that the unresolved
parameter width meets the tolerance; existence is reported separately and is
not inferred from a nonempty outer cover. Interior-sensor monotonicity is not
assumed. Reflection sends a solution to another solution with the same forcing
and clamped boundaries; uniqueness therefore proves u(x)=u(1-x). The quarter
and three-quarter sensors are redundant in this symmetric fixture. Three sensor
locations do not imply three independent pieces of information.

### Two-parameter extension and its assumptions

For \(u''''+\lambda u^3=a\), \(a>0\), define

\[
 V(\lambda,a)=\min_u\left[\tfrac12\|u''\|_2^2+
 \tfrac\lambda4\int u^4-a\int u\right].
\]

The envelope theorem and implicit differentiation give

\[
 \nabla V=(\tfrac14\int u^4,-\int u),\qquad
 D^2V=-J^*L^{-1}J,\qquad J=[u^3,-1].
\]

The two columns are independent: if \(\alpha u^3-\beta=0\) everywhere,
continuity and the clamped endpoint give \(\beta=0\), and the nonzero field
then gives \(\alpha=0\). Hence the Hessian is strictly negative definite.
Integrating its quadratic form along the segment between two parameters proves
strict monotonicity of the gradient and global injectivity on a convex domain
with \(\lambda\ge0,a>0\). This establishes identifiability for this pair
of conjugate observables. It supplies neither a quantitative uniform conditioning
certificate nor an implemented two-dimensional recovery method.

The same calculation applies conditionally to a real Hilbert-space energy
\(E(u;\theta)=E_0(u)+\sum_i\theta_iF_i(u)\): assume a unique smooth
minimizer, a coercive invertible state Hessian \(L\), and twice continuously Fréchet-differentiable
functionals \(F_i\). Its conjugate observables \(O_i=F_i(u(\theta))\)
satisfy \(DO=-J^*L^{-1}J\), where \(J\eta=\sum_i\eta_iDF_i\).
This follows by differentiating state stationarity, then applying the chain
rule. Independent columns of \(J\) along a convex parameter domain give the
same injectivity proof. These are explicit sufficient conditions, not a claim
for arbitrary neural networks or PDEs. The envelope step is established theory;
see [Milgrom and Segal](https://milgrom.people.stanford.edu/wp-content/uploads/2002/03/Envelope-Theorems.pdf).
Computable quantitative conditioning and measurable observables remain separate
application requirements.

The assumptions matter. At zero forcing, \(u=0\) for every coupling, so coupling
is unidentifiable. A clamped-endpoint point sensor is likewise uninformative.
One scalar observable cannot generally identify two unknown parameters: locally,
\(H_\lambda\ne0\) permits a curve of \((\lambda,a)\) with the same
\(H\). For \(h(t)=(t-1/2)^2\) and exact measurement \(1/16\),
the squared observation loss vanishes at both \(t=1/4\) and \(t=3/4\).
Reducing that loss cannot distinguish the two parameters. These counterexamples prevent extending the recovery conclusion
to an arbitrary sensor, uncertain physics or an optimization trajectory alone.

A useful exact regularity counterexample is the clamped Green function
\(G(x,1/2)\). It is nonzero and piecewise cubic, with matching derivatives
through order two and a unit jump in its third derivative. Its fourth derivative
is zero inside each open cell but equals a Dirac load distributionally.
Ignoring the interface would therefore falsely certify the homogeneous problem.
The accepted witness types are global polynomials and analytic caged fields;
a SciPy spline cannot enter by supplying only its cell-interior derivatives.

## Relation to existing work

| Construction | Prior work and attribution |
| --- | --- |
| Monotone biharmonic existence and uniqueness | [Dunninger and Locker](https://msp.org/pjm/1975/60-1/pjm-v60-n1-p05-p.pdf); established functional analysis |
| Certified parameter identification and reduced models | [Kartmann et al.](https://link.springer.com/article/10.1007/s44207-024-00002-z); certified reduced-basis and error-aware inverse algorithms already exist |
| Rigorous polynomial boundary-value proposals | [Validated Chebyshev methods](https://research.vu.nl/en/publications/rigorous-numerics-for-odes-using-chebyshev-series-and-domain-deco/); the local coefficient bound uses classical Chebyshev algebra |
| Numerical collocation comparison | [SciPy solver contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html); its C1 cubic spline is a numerical reference, not directly a C4 scalar witness |
| This study | A complete scalar recovery/replay contract, a derived conditioning bound and matched implementation evidence on the stated problem; mathematical priority and general competitive advantage require separate evidence |

The source review is targeted. The energy identities and conjugate-observable
argument are deductions from established variational theory. No new general
calculus or universal inverse-problem method is asserted.

## Experimental protocol and cost accounting

Discovery uses couplings 24, 96 and 320 with seeds 1301, 1423 and 1543. Measurements carry certified reference uncertainty, with a separate bounded 0.1% uncertainty case. The reference is an independently fitted polynomial with 90-digit search and a 10^30 rational coefficient denominator; competing proposals use the fixed 10^12 denominator. Its whole-domain error certificate, not the floating collocation value, creates the observation interval. Generating parameters are absent from worker inputs.

Neural widths are 12 then 24, with 100 readout steps on seeded paired tanh features; the hidden weights and biases remain fixed. Conclusions about neural proposals apply to that recipe. Spectral controls use 7 then 12 Chebyshev modes and eight Newton steps. The local POD control uses seven fixed polynomial snapshots and energy-weighted SVD, retaining singular values above 1e-12 of the largest. It is not official pyMOR. SciPy is a numerical consistency comparator; CAPD, Arb and pyMOR were unavailable. No new comparator downloads were required.

Cold runs use isolated processes and include import/configuration, fitting, all accepted and rejected refinement attempts, certificate construction, serialization and replay. POD snapshot/basis setup is added to each cold calibration. Actual persistent-process batches charge shared initialization and family setup once. Their driver is selected on completed discovery recipes; confirmation freezes that choice. Persistent batches reuse a process warmed by correctness/control preparation; template caches are warm, while measured dependency-import/configuration startup and POD construction are charged once. The warm batches do not measure a fresh interpreter launch; cold calibrations separately charge the complete subprocess launch. Cold runs separately include fresh process and template initialization. Acquisition/reference-generation cost is common experimental fixture preparation, recorded separately from solving an inverse problem with supplied measurements.

The parent terminates each isolated calibration after 60 seconds. Measured cost includes input serialization and process termination overhead, so timeout rows may slightly exceed 60 seconds. Three repetitions are required for a completed case. Every candidate observation is replayed before it can exclude a parameter. Hard resource limits, precision floors and unavailable proofs remain explicit. Rows may report a resolved broad feasible interval when measurement uncertainty is large; outer width is never conflated with numerical endpoint uncertainty.

The additional full-campaign reproduction audit took approximately 1160 seconds from the recorded UTC boundaries. It repeats the already charged per-calibration field/decision replay and is recorded as an extra campaign audit, rather than silently adding a second assurance pass to one proposal family. Fixture preparation and every stage's measured duration are retained in the summary.

## Integrated-observation results

| Proposal | Driver | Certified cases / 6 | Median completed cold cost (s) | Median cost, all attempts (s) | Median certified fields |
| --- | --- | ---: | ---: | ---: | ---: |
| neural | safeguarded | 0/6 | — | 60.08 | 7 |
| neural | bisection | 0/6 | — | 60.09 | 8 |
| spectral | safeguarded | 6/6 | 3.884 | 3.884 | 27 |
| spectral | bisection | 6/6 | 3.059 | 3.059 | 13 |
| reduced_basis | safeguarded | 6/6 | 5.556 | 5.538 | 26 |
| reduced_basis | bisection | 6/6 | 3.877 | 3.859 | 13 |

Completed cases require all three repetitions. Costs on the completed subset do not establish superiority over a method with different completion. The promotion gate uses matched resolution, complete matrices, cold cost and actual measured batch cost against the strongest eligible named controls.

| Proposal | Actual batch repetitions | Median batch including setup (s) | Median batch body (s) |
| --- | ---: | ---: | ---: |
| neural | 0 | — | — |
| spectral | 3 | 7.436 | 5.725 |
| reduced_basis | 3 | 11.77 | 10.04 |

Only recipes that completed every discovery case enter persistent-process batches. Each measured batch contains the six discovery case/noise combinations. An absent batch is not assigned an extrapolated speed.

| Proposal | Projected total for 10 calibrations (s) | Projected total for 100 calibrations (s) |
| --- | ---: | ---: |
| neural | — | — |
| spectral | 11.25 | 97.13 |
| reduced_basis | 18.46 | 169 |

These are amortization projections, not measured 10/100-case batches: one measured setup plus N times measured six-case body cost divided by six. They assume the same case mixture, unchanged resources and linear scaling; they do not establish a speed advantage.

| True coupling | Relative measurement uncertainty | Proposal | Status | Outer parameter width | Numerical endpoint uncertainty / 512 |
| ---: | ---: | --- | --- | ---: | ---: |
| 24 | 0 | neural | INCONCLUSIVE | 512 | — |
| 24 | 0 | spectral | CERTIFIED | 0.1617 | 0.0006317 |
| 24 | 0 | reduced_basis | CERTIFIED | 0.1617 | 0.0006317 |
| 24 | 1/1000 | neural | INCONCLUSIVE | 512 | — |
| 24 | 1/1000 | spectral | CERTIFIED | 0.5558 | 0.000631 |
| 24 | 1/1000 | reduced_basis | CERTIFIED | 0.5558 | 0.000631 |
| 96 | 0 | neural | INCONCLUSIVE | 512 | — |
| 96 | 0 | spectral | CERTIFIED | 0.1636 | 0.0006389 |
| 96 | 0 | reduced_basis | CERTIFIED | 0.1388 | 0.000542 |
| 96 | 1/1000 | neural | INCONCLUSIVE | 512 | — |
| 96 | 1/1000 | spectral | CERTIFIED | 0.725 | 0.0006399 |
| 96 | 1/1000 | reduced_basis | CERTIFIED | 0.6991 | 0.000541 |
| 320 | 0 | neural | INCONCLUSIVE | 512 | — |
| 320 | 0 | spectral | CERTIFIED | 0.1789 | 0.0006989 |
| 320 | 0 | reduced_basis | CERTIFIED | 0.2045 | 0.0007987 |
| 320 | 1/1000 | neural | INCONCLUSIVE | 512 | — |
| 320 | 1/1000 | spectral | CERTIFIED | 1.215 | 0.0006895 |
| 320 | 1/1000 | reduced_basis | CERTIFIED | 1.239 | 0.0008018 |

This table shows the first deterministic repetition for each safeguarded recipe; all repetitions remain in the artifacts. Raw exact endpoints, retained witnesses, failed precision attempts and complete costs are replayable.

The fastest cold recipe completing every discovery case was **spectral / bisection**, with median completed cold cost 3.059 seconds. This names the measured result on the fixed matrix; it is not a general solver ranking.

On the fixed integrated matrix, compare drivers using both complete cost and the number of retained field certificates. Sensitivity-based proposals are an empirical scheduling choice, and the same assurance contract does not imply that they beat bisection.

| Proposal | Median retained fit (s) | Median retained certificate construction (s) | Median retained independent replay (s) |
| --- | ---: | ---: | ---: |
| neural | 1.229 | 38.06 | 17.26 |
| spectral | 0.02918 | 0.4541 | 0.4938 |
| reduced_basis | 0.04988 | 0.9948 | 1.082 |

This diagnostic uses the retained prefix of each integrated run. It excludes interrupted final work, unretained attempts and other overhead; its component medians are not additive total costs. Full costs remain the cold/batch measurements above.

## Sparse point sensors

| Proposal | Resolved cases / 6 | Median completed cold cost (s) | Median cost, all attempts (s) |
| --- | ---: | ---: | ---: |
| neural | 0/6 | — | 60.09 |
| spectral | 5/6 | 8.716 | 9.501 |
| reduced_basis | 5/6 | 10.09 | 11.61 |

These results concern rigorous parameter covers. Existence, inner components, outer components and unresolved parameter width are stored separately. A narrow outer cover alone is not a proof that inconsistent multi-sensor measurements have a physical solution. No unproved point-sensor monotonicity is used.

Conventional point-cover failures remain explicit: case/noise IDs [('case_2', '1/1000')], retained field counts [128], unresolved normalized widths ['11/8192']. The frozen 128-field query budget can stop subdivision before the 0.001 cover target even when the 60-second time budget is not exhausted. This is a cover-resolution failure, not an empty feasible-set conclusion. The post-discovery sign/conditioning check tests a possible basis for a future more selective inverse driver; it does not rerun or upgrade these cases.

## Confirmation, failures and assessment

Discovery completion and speed evidence: `{"cases": 6, "cold_cost_ratio_control_over_neural": null, "common_cases": 0, "completion": {"neural": 0.0, "reduced_basis": 1.0, "spectral": 1.0}, "completion_advantage": -1.0, "measured_batch_ratio_control_over_neural": null, "qualified": false, "reason": "registered gates on the complete frozen matrix; strongest eligible named controls", "strongest_control_completion": 1.0}`.

No candidate qualified for fresh-seed confirmation. Confirmation observations were not generated. The reserved stage instead records fixed precision diagnostics at smoke/failure-analysis couplings 37, 200 and 500; those are not held-out confirmation results.

| Diagnostic coupling | Proposal | Width / modes | Numerical grid residual RMSE | Proved state error | Requested field target met |
| ---: | --- | ---: | ---: | ---: | --- |
| 37 | neural | 12 | 0.01791 | 0.0001336 | False |
| 37 | neural | 24 | 0.009678 | 0.0001022 | False |
| 37 | spectral | 7 | 0.0002633 | 1.196e-06 | False |
| 37 | spectral | 12 | 1.308e-07 | 1.482e-09 | True |
| 200 | neural | 12 | 0.07835 | 0.0006083 | False |
| 200 | neural | 24 | 0.03105 | 0.0003669 | False |
| 200 | spectral | 7 | 0.003579 | 1.61e-05 | False |
| 200 | spectral | 12 | 7.482e-07 | 8.667e-09 | True |
| 500 | neural | 12 | 0.121 | 0.0009171 | False |
| 500 | neural | 24 | 0.06567 | 0.0006681 | False |
| 500 | spectral | 7 | 0.0101 | 4.49e-05 | False |
| 500 | spectral | 12 | 1.4e-07 | 1.507e-09 | True |

The grid residual is a floating optimization diagnostic on the original readout, and is not itself a rigorous error bound for the snapped field. The state column comes from the replayed continuum certificate. Targets here are state error 1e-6 and candidate-integral quadrature width 1e-6. These fixed diagnostics supply no held-out or speed claim.

## Uniform sparse-sensor identifiability check

A separate bounded post-discovery sensitivity cover returned **CERTIFIED** over [0,512], using 64 rational parameter cells. Each cell encloses the entire physical sensitivity field, including state error and variational residual. Independent replay checks the full parameter cover and every point-derivative bound. This does not replace or modify the primary calibration matrix.

| Sensor | Proved lower magnitude of coupling derivative | Inverse Lipschitz upper bound |
| --- | ---: | ---: |
| 1/4 | 0.000110018 | 9089.42 |
| 1/2 | 0.000231386 | 4321.78 |
| 3/4 | 0.000110018 | 9089.42 |

The exact rational bounds are stored in the artifact; displayed values are rounded summaries. If -du(x_j;lambda)/dlambda is at least m_j>0 throughout the prior, the fundamental theorem of calculus gives |u(x_j;b)-u(x_j;a)| >= m_j|b-a|. Thus each listed sensor is injective for the stated problem. Interval-measurement preimages and their intersection are intervals or empty. Existence for a particular noisy vector still requires a compatibility proof. This is a verified application-specific observability result, not a general monotonicity theorem for point sensors or a neural performance advantage.

## Bounded post-discovery enclosure probe

The reserved failure analysis also tests a separate classical Taylor residual enclosure through order four, requiring field derivatives through order eight. It retains the frozen primary campaign and fits only the fixed diagnostic couplings 37, 200 and 500 with seed 997. At a cell center, residual derivative terms are combined with a whole-cell Taylor remainder; an independent derivative implementation replays every enclosure. This is a verification/observation-enclosure test on the same fitted fields, with state target 1e-6 and quadrature target 1e-8. It supplies no end-to-end calibration or training-speed claim.

| Coupling | Width | Original target met | Higher-order target met | Higher-order proved state error | Floor for the fixed energy-certificate form |
| ---: | ---: | --- | --- | ---: | ---: |
| 37 | 12 | False | False | 0.0001197 | 0.0001197 |
| 37 | 24 | False | False | 4.524e-05 | 4.477e-05 |
| 200 | 12 | False | False | 0.0005654 | 0.0005654 |
| 200 | 24 | False | False | 0.0001456 | 0.0001412 |
| 500 | 12 | False | False | 0.0008536 | 0.0008536 |
| 500 | 24 | False | False | 0.0002987 | 0.0002902 |

For 6/6 diagnostic fields, a rational point-residual enclosure proves that any certificate of the implemented form C_up times a residual-supremum upper bound must exceed 1e-6. Indeed, every such upper bound is at least |r(x0)|, whose lower magnitude is independently enclosed. This is a floor for this certificate form and fitted field; it is not a lower bound on the actual state error. Improving only the spatial enclosure cannot overcome that floor.


Any improvement is attributed to the classical enclosure order and its implementation. It does not alter the failed or passing primary gates, and requires a newly frozen end-to-end comparison before promotion.

Process outcomes: `{'TIMEOUT': 54, 'FINISHED': 108}`. Termination reasons: `{'parent_60_second_cap': 54, 'resolution_met': 72, 'complete_cover': 30, 'atomic_split_budget': 6}`. Incomplete stage matrices, if any, are labeled above and cannot promote a candidate.

Before source freeze, engineering smoke used coupling 37 and seed 997, disjoint from discovery and confirmation. It exposed avoidable Bernstein residual-subdivision cost; the stronger exact coefficient-norm control was adopted before any discovery result. A quadrature-width test initially demanded an unjustified tight width at 16 cells; containment already passed. Interface/cache regressions during implementation were corrected before freezing the matrix. No false accepted certificate or parameter exclusion was observed. Missing optional proof/solver packages remain unavailable rather than being represented by numerical substitutes.

The final reporting regression also caught a formatter/runtime Python-version mismatch in the supplementary report loader. It was corrected with an explicit Python-3.12 formatting/lint target and revalidated. The initial failed report/test logs are retained; no frozen numerical source or certificate was changed. HTML review also exposed missing math rendering in the documentation configuration; the corrected build preserves the equations and has a structural rendering gate.

| Candidate | Classification |
| --- | --- |
| Global energy stability and physical sensitivity bounds | Established mathematics with explicit implementation contracts |
| Strict integrated-observation monotonicity and conditioning | Proved problem-specific deduction from variational theory; priority not established |
| Two-parameter conjugate-observable injectivity | Complete argument under positive forcing; numerical Jacobian check, no 2D recovery |
| Complete scalar inverse brackets and independent replay | Useful research engineering |
| Neural competitive advantage | Evaluated only for the registered recipe; promotion requires both discovery and fresh confirmation gates |
| Arbitrary sensors or uncertain PDEs inherit identifiability | Unsupported; explicit counterexamples and separate cover semantics apply |
| Uniform interior-sensor identifiability | Verified application-specific observability result; mathematical priority not established |
| Higher-order residual enclosure | Established Taylor mathematics; post-discovery engineering probe, no end-to-end competitive claim |

For the present problem, preserve the recovery/checker contract and use spectral proposals with bisection, the fastest fully completed recipe. The fixed neural recipe should not advance on these results. A stronger neural readout/representation must first clear the observed residual-certificate floor under a newly frozen protocol.

The next bounded implementation study should reuse the proved point-sensor sensitivity signs and inverse-Lipschitz constants to bracket each sensor interval's inverse endpoints, then intersect the resulting certified parameter sets. Compare its full proof-construction, replay and recovery cost with the frozen uniform subdivision control, including the noisy coupling-320 case and measured batches. The concrete unresolved question is whether reusable observability proofs amortize enough to improve completion and total cost across repeated calibrations. This is a proposed engineering study, not a demonstrated speed gain. Broader observation/physics classes remain a subsequent gate.

## Reproduction and validation

From the laboratory root, use the environment and sibling paths documented in `verified_calibration_research/README.md`:

```bash
python -m pytest --import-mode=importlib -p no:cacheprovider verified_calibration_research/test_checks.py -q
python -m verified_calibration_research.run --stage all --output artifacts/reproduction_verified_calibration
python -m verified_calibration_research.run --replay --output artifacts/reproduction_verified_calibration
python research_verified_calibration_audit.py artifacts/reproduction_verified_calibration
python -m reports.finalize_verified_calibration artifacts/reproduction_verified_calibration --probe
```

The additional experiment-chain audit is **PASS**. It binds each self-contained mathematical certificate to the requested observation, resolution, proposal family and recipe. This separate audit preserves the frozen producer and closes an integration obligation that a certificate for its own declared observation alone does not satisfy. Reproduction therefore requires both numerical replay and the binding audit; the finalizer requires numerical replay and runs the binding audit before publishing the report.

16 focused tests passed before discovery, including actual worker integration, exact state/sensitivity fixtures, independent integration, coverage, noise, malformed proofs and immediate stop on checker rejection. All retained campaign stages underwent independent replay. The two-parameter finite-difference discrepancy was 8.81e-13; it is a numerical consistency check, not an interval Hessian proof. Independent replay uses separately implemented derivative/recurrence algebra and exact coefficient reconstruction; it shares the interval-arithmetic foundation and the written stability theorem. The audit is a software consistency check within this study. Written mathematical arguments and directed arithmetic are trusted components; no formal proof-assistant build is claimed.

An additional 11 focused regressions passed after the primary matrices, covering the supplemental audit and failure probes. The complete validation record also includes lint, strict documentation build, source archives and dependency snapshots.

