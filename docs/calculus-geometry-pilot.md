# Calculus–geometry pilot: exact sets, solution maps and realizable jets

This pilot studies the exact equality interpretation
\(E_M(x)=\mathbf 1\{x\in M\}\). The hard readout, the defining residual,
and a differentiable geometric loss have separate contracts. The mathematical
identities below are established mathematics or elementary consequences proved
here. They are not presented as inventions. The open research question is
whether a specific representation and implementation yields an independently
demonstrated advantage.

## 1. Exact equality and affine geometry

Let \(A\in\mathbb R^{m\times d}\) have full row rank and let

\[
r(x)=Ax-b,\qquad M=\{x:Ax=b\}.
\]

Minimize \(\tfrac12\|y-x\|^2\) subject to \(Ay=b\). Stationarity and
feasibility give \(y-x+A^T\lambda=0\) and
\(AA^T\lambda=Ax-b\). Therefore

\[
\Pi_M(x)=x-A^T(AA^T)^{-1}(Ax-b),
\qquad P_N=A^T(AA^T)^{-1}A,\quad P_T=I-P_N.
\]

The correction lies in the normal space and is orthogonal to every feasible
displacement. Hence, for any \(z\in M\),

\[
\|x-z\|^2=\|x-\Pi_Mx\|^2+\|\Pi_Mx-z\|^2.
\]

This proves both nearest-point optimality and uniqueness. In particular,

\[
L_M(x)=\tfrac12 r(x)^T(AA^T)^{-1}r(x),\qquad
\nabla_xL_M=x-\Pi_Mx,\qquad
\nabla_x^2L_M=P_N,\qquad D_x\Pi_M=P_T.
\]

For a single plane \(a^Tx=b\), this reduces to
\(L=(a^Tx-b)^2/(2\|a\|^2)\), with \(a\ne0\).
The normal eigenvalues of the distance Hessian equal one; tangent eigenvalues
equal zero. This corrects anisotropic *input-space* conditioning of the raw
loss \(\tfrac12\|Ax-b\|^2\). It does not remove the factorization cost or
guarantee well-conditioned derivatives with respect to a learned matrix A.

For an invertible equation recombination \(S\), the pairs \((A,b)\) and
\((SA,Sb)\) define exactly the same distance and projection. This follows
either from the unique minimization problem or from
\((SAA^TS^T)^{-1}=S^{-T}(AA^T)^{-1}S^{-1}\). No assumption about gradient
descent is required.

The inverse above specifies the mathematics. Tensor implementations use a
reduced QR factorization of \(A^T\) and triangular solves. Exact rational
arithmetic instead supports redundant equations: elimination produces an
anchor \(a\) and nullspace basis N, then

\[
P_T=N(N^TN)^{-1}N^T,\qquad a_0=(I-P_T)a,
\qquad \Pi_Mx=a_0+P_Tx.
\]

The nullspace Gram matrix is positive definite because the basis columns are
independent. Rational LDLT solves avoid square roots. An inconsistent affine
system has no projection; exhausting an arithmetic budget says nothing about
consistency.

The equality detector satisfies \(E_M(x)=1\iff L_M(x)=0\). Its ordinary
derivative is zero away from M, and it is discontinuous transversely to M.
Training through the smooth residual/distance retains information discarded by
the Boolean readout. It does not differentiate the Boolean readout itself.
Numerical feasibility tolerances also do not define exact membership.

The location of equality matters in a deep model. A fixed affine hyperplane
has zero probability under a full-dimensional absolutely continuous input
distribution. This conclusion does not cover arbitrary learned embeddings:
\(\mathbf1\{\operatorname{ReLU}(g(x))=0\}=\mathbf1\{g(x)\le0\}\),
whose positive class can have positive volume. A nonidentically-zero scalar
residual built entirely from real-analytic operations, such as a finite tanh
network followed by an affine residual, does have a measure-zero zero set on
a connected open domain; this uses the real-analytic assumption, not merely
smoothness. A proof is given by [Mityagin, *The Zero Set of a Real Analytic
Function*](https://arxiv.org/abs/1512.07276). ReLU flat regions evade that
assumption. These are established representation facts; the ReLU construction
recovers an ordinary inequality decision and its equality readout remains
hard. Floating saturation is a separate numerical issue and cannot prove any
of these exact identities for a general trained model.

These projection and equality-constrained quadratic constructions are classical
convex optimization. See [Boyd and Vandenberghe, equality-constrained
optimization](https://stanford.edu/~boyd/cvxbook/). Existing neural
implementations include [KKT-hPINN](https://github.com/li-group/kkthpinn), whose
architecture projects network predictions onto linear equality constraints.
Consequently, placing this projection inside a network does not establish
method novelty.

## 2. Nonlinear roots, iterations and nearest points

Let \(r(y,\theta)\in\mathbb R^m\) be C² in y and sufficiently continuously
differentiable in the parameters, including the displayed mixed derivatives,
and let J denote its y
Jacobian. At a regular root the tangent space is \(\ker J\). A minimum-norm
linearized correction is

\[
\delta=-J^T(JJ^T)^{-1}r.
\]

This solves \(J\delta=-r\). It need not reach a root in one step, and
iterating it does not generally select the nearest root to the original seed.
For example, for \(r(y)=\|y\|^2-1\), from \(y=(3,4)\) one linearized
step yields \((39/25,52/25)\), whereas the nearest point is \((3/5,4/5)\).

A local nearest point to a datum z obeys the augmented KKT equations

\[
y-z+J(y,\theta)^T\lambda=0,\qquad r(y,\theta)=0.
\]

Write

\[
H=I+\sum_i\lambda_i\nabla_y^2r_i,
\qquad K=\begin{pmatrix}H&J^T\\J&0\end{pmatrix}.
\]

If J has full row rank and H is positive definite on \(\ker J\), K is
nonsingular. To see this, suppose \(Hv+J^T\mu=0\), \(Jv=0\).
Then \(v^THv=0\) implies v=0, and full row rank then implies \(\mu=0\).
The implicit function theorem gives the branch sensitivities

\[
\begin{pmatrix}dy\\d\lambda\end{pmatrix}
=K^{-1}\left[
\begin{pmatrix}dz\\0\end{pmatrix}
-\begin{pmatrix}\partial_\theta(J^T)\lambda\\\partial_\theta r\end{pmatrix}d\theta
\right].
\]

These are local sensitivities of the selected regular KKT branch. Global
nearest-point uniqueness needs additional assumptions. At z=y, the multiplier
is zero and the datum derivative is the tangent projector. Away from the
manifold, curvature matters. On the unit circle, for nonzero z,

\[
y=z/\|z\|,\qquad D_zy=(I-yy^T)/\|z\|.
\]

At z=(2,0), its derivative in the vertical direction is 1/2, rather than the
on-manifold tangent factor 1. A solver that only finds a root cannot infer this
nearest-point derivative without a selection rule. For an isolated regular
root, its local derivative with respect to a seed is zero because the root
itself is unchanged; an underdetermined manifold generally retains seed
dependence.

KKT/implicit differentiation for neural optimization layers predates this
pilot; [OptNet](https://proceedings.mlr.press/v70/amos17a.html) explicitly
develops parameter derivatives through quadratic optimization layers. The
curvature term above is retained to avoid applying an affine projector formula
to a curved nearest-point problem.

## 3. Residual-coordinate changes: invariance and a local bound

Let \(\widetilde r(x,\theta)=S(x,\theta)r(x,\theta)\), with S invertible
throughout a declared neighborhood. The zero sets are identical there. At a
root, all terms containing r vanish, so

\[
J_{\widetilde r}=S J_r,\qquad
\partial_\theta\widetilde r=S\partial_\theta r.
\]

For a square regular system,
\(-J_{\widetilde r}^{-1}\partial_\theta\widetilde r
=-J_r^{-1}\partial_\theta r\). Thus the selected solution and its true local
parameter sensitivity are preserved. For an underdetermined system an
additional selection rule remains necessary. A nearest-point objective fixes
such a selection; its local primal solution is unchanged by an invertible
residual reexpression. Its multipliers change according to
\(\widetilde\lambda=S^{-T}\lambda\).

This does **not** make the off-root iteration invariant. For a square
invertible J, write

\[
E_{ij}=\sum_k(\partial_j S_{ik})r_k,
\qquad G=S^{-1}E,
\qquad J_{\widetilde r}=S(J+G).
\]

The Newton steps are \(\delta=-J^{-1}r\) and
\(\widetilde\delta=-(J+G)^{-1}r\). If
\(\eta=\|J^{-1}G\|<1\), the Neumann bound gives

\[
\boxed{\|\widetilde\delta-\delta\|
\le\frac{\|J^{-1}\|^2\|G\|\|r\|}{1-\eta}.}
\]

**Proof.** Use the resolvent identity
\(J^{-1}-(J+G)^{-1}=(J+G)^{-1}GJ^{-1}\), and
\(\|(J+G)^{-1}\|\le\|J^{-1}\|/(1-\|J^{-1}G\|)\).
For the operator norm of DS as a map from residual vectors to E,
\(\|E\|\le\|DS\|\|r\|\). Uniform bounds
\(\|J^{-1}\|\le K\), \(\|S^{-1}\|\le M\), \(\|DS\|\le C\)
and \(KMC\|r\|\le\eta_0<1\) therefore imply

\[
\|\widetilde\delta-\delta\|
\le \frac{K^2MC}{1-\eta_0}\|r\|^2.
\]

These constants can be enclosed on a fixed box with verified arithmetic.
For example, if \(\|J(x)-J(x_0)\|\le\varepsilon_J<\sigma_{\min}(J(x_0))\)
and \(\|S(x_0)^{-1}(S(x)-S(x_0))\|\le\varepsilon_S<1\), then

\[
K\le[\sigma_{\min}(J(x_0))-\varepsilon_J]^{-1},\qquad
M\le\|S(x_0)^{-1}\|/(1-\varepsilon_S).
\]

The bound certifies a declared neighborhood when all its inequalities are
verified; it does not locate a root or prove a speed improvement.
Constant S has E=0 and identical exact Newton steps. State-dependent S need
not: for \(r(x)=x\), \(S(x)=e^{kx}\), the transformed step is
\(-x/(1+kx)\), which has a pole at \(-1/k\) although S never vanishes.

For full-row-rank rectangular J, the local quadratic measure
\(\tfrac12r^T(JJ^T)^{-1}r\) has the normal-projector Hessian at a root.
It approximates squared distance locally; it is not generally the exact
distance or globally convex. The derivatives of a state-dependent metric may
not be discarded away from the root.

The invariance statements and perturbation bound are consequences of standard
linear algebra, implicit differentiation and the inverse perturbation identity.
No novelty claim is made for them. Affine invariance of Newton methods has an
extensive existing theory, catalogued on [Deuflhard's ZIB research
page](https://www.zib.de/userpage/deuflhard/research/algorithm/ainewton.en.html).

## 4. When calculus really transfers to linear algebra

Let \(T_U:C\to U\) realize coefficient vectors as functions, and let
\(T_F:F_c\to F\) realize forcing coefficients. An exact representation
of a linear differential operator is the identity

\[
\mathcal L T_U=T_F M.
\]

If the forcing is \(T_Ff\), then \(Mc=f\) implies
\(\mathcal L(T_Uc)=T_Ff\). Conversely, a represented function solves the PDE
exactly if \(Mc-f\in\ker T_F\). The coefficient equation is equivalent to
the PDE only when \(T_F\) is injective, or after explicitly quotienting its
kernel. If the same invariant space is used on both sides, this is the
familiar \(\mathcal LT=TM\).

Boundary conditions need a separate exact identity
\(\mathcal BT_U=T_BB_c\). The combined coefficient system is then
\(Mc=f\), \(B_cc=g\), with the same injectivity caveat. Existence in this
representation says nothing about solutions outside its image. Learned
frequencies or basis parameters make M depend on parameters and generally
restore a nonlinear optimization problem.

The existing [ridge/profile compiler](profile-pde.md) implements a particular
exact operator transfer. On profiles annihilated by \(R(D_z)\),

\[
D_i[e^{a\cdot x}\psi(w\cdot x+b)]
=e^{a\cdot x}(a_i+w_iD_z)\psi(w\cdot x+b),
\]

so operator composition becomes polynomial composition modulo R. The transfer
preserves the declared differential equations exactly. It is not a quotient
rule for multiplying the *functions* themselves.

For example, \(V=\operatorname{span}\{1,x\}\) is invariant under
differentiation, but \(x\cdot x=x^2\notin V\). Declaring \(x^2=0\) in a
coefficient quotient would change the realized function, not solve the original
nonlinear PDE. Fourier products similarly generate sum frequencies. A
nonlinearity can sometimes close on a special family, but closure must be
proved for that family and operation.

There is a broad elementary obstruction: a finite-dimensional real algebra of
continuous functions on a connected domain contains only constant functions.
For any f in such an algebra, its powers satisfy a nonzero scalar polynomial
identity p(f)=0 by finite dimensionality (use 1,f,… if the algebra is unital,
or f,f²,… otherwise). Thus f takes values in the finite root set of p.
Continuity and connectedness force f to be constant. This rules out a fixed,
nontrivial finite continuous-function space closed under unrestricted repeated
pointwise multiplication. It does not rule out growing bases, approximations,
disconnected domains, or specially restricted nonlinear dynamics. This is an
elementary algebraic limitation, not a proposed new theorem.

Exact PDE representations already have substantial prior art.
[EPGP](https://proceedings.mlr.press/v202/harkonen23a.html) constructs Gaussian
process priors supported on solutions of constant-coefficient linear PDE
systems, using the Ehrenpreis–Palamodov principle. Consequently, an exact
linear-PDE representation alone is not a first-of-its-kind claim.

## 5. A computable exact jet-realizability check

A PDE relation \(F(x,u,p_\alpha)=0\) defines a geometric subset of jet space.
A solution also needs contact compatibility

\[
\partial_i p_\alpha=p_{\alpha+e_i},\qquad |\alpha|<k,
\qquad p_0=u.
\]

For instance, the section u=x, p=x lies in the relation p=u, but p differs
from the derivative of u. Projecting these coordinates onto p=u cannot by
itself solve u'=u. In two variables, the demands \(u_x=0\), \(u_y=x\)
violate mixed-partial compatibility. These are the classical contact conditions
in geometric PDE theory; see [del Pino's jet-space exposition, §1.3](https://alvarodelpino.com/wp-content/uploads/2019/10/coimbra2018notes.pdf).

**Restricted exact certificate.** Suppose every coordinate p_alpha is a
rational polynomial, every index through order k is present, and every contact
residual polynomial is identically zero. Then \(p_\alpha=D^\alpha p_0\)
globally. Proof is induction on |alpha| using a predecessor index and its
contact identity. The polynomial p0 is therefore an explicit realizing field.
Any supplied linear polynomial-coefficient PDE identities and coordinate-plane
boundary identities can then be checked by exact coefficient equality.

This answers the first discovery question in a restricted, computable class.
The mathematical criterion is classical. The implementation contributes exact
failure witnesses and explicit resource refusals. Sampling cannot provide this
certificate: a nonzero polynomial can vanish on all points in a finite sample.
The checker does not establish that a general proposed jet section is
realizable, approximate the nearest realizable section, or certify a neural
field outside the polynomial representation.

```python
from omnibias.core.jet_geometry import (
    LinearJetEquation, PolynomialJetBoundary,
    check_polynomial_jet, polynomial_jet,
)
from omnibias.core.realization.polynomial import SparsePolynomial

x = SparsePolynomial.variable(2, 0)
t = SparsePolynomial.variable(2, 1)
section = polynomial_jet((x - 2*t)**3, order=3)  # raw partial derivatives
checked = check_polynomial_jet(
    section,
    equations=(LinearJetEquation({(0, 2): 1, (2, 0): -4}, label="wave"),),
    boundaries=(PolynomialJetBoundary((0, 0), 1, 0, x**3, "initial data"),),
)
assert checked.valid and checked.holonomic
```

Checking a finite representation is different from solving for one. Projection
in coefficient space preserves realizability when the coefficient-to-field map
and its derivative maps are kept together. Independent pointwise projection of
derivative coordinates generally discards that property. A useful next
research target is an efficient correction *within* a declared realizable
function representation, compared with ordinary coefficient least squares and
nullspace elimination.

## 6. Exact Python contracts and current status

```python
from fractions import Fraction
from omnibias.core.affine_geometry import exact_affine_geometry

geometry = exact_affine_geometry(((2, 1), (4, 2)), (1, 2))
assert geometry.consistent and geometry.rank == 1
assert geometry.project((0, 0)) == (Fraction(2, 5), Fraction(1, 5))
assert geometry.squared_distance((0, 0)) == Fraction(1, 5)
assert geometry.contains(geometry.project((0, 0)))
```

`exact_affine_geometry` accepts only int/Fraction values, excluding bool.
It returns a sealed immutable object with exact consistency, rank,
minimum-norm anchor and tangent projector. It supports empty constraints with
an explicit dimension, redundant equations and proven inconsistency. Projection
or distance on an inconsistent set raises `InconsistentAffineConstraints`.
`AffineBudget` bounds dimensions, rows, rational slots, charged work and
coefficient bit lengths. Each operation starts a fresh meter; exhaustion raises
`AffineBudgetExceeded`, never a false inconsistency verdict.

`polynomial_jet` and `check_polynomial_jet` reuse the existing rational
`SparsePolynomial` representation. A complete jet uses raw derivatives.
Nonzero residual polynomials are retained as failure witnesses.
`PolynomialJetBudget` bounds coordinates, degrees, coefficient bits, cumulative
intermediate terms and work. The reused polynomial engine can additionally
raise `AlgebraBudgetExceeded`. A certificate fingerprint records content
integrity; it does not mean a theorem prover accepted the result.

| Candidate | Mathematical assessment | Evidence needed for a stronger claim |
|---|---|---|
| Exact affine projection and normalized distance loss | Established linear algebra and constrained optimization | Useful software quality or workload advantage, with equivalent algebra controls |
| Differentiating a selected regular root or KKT solution | Established implicit differentiation | Correct branch/selection contracts and independent gradient tests |
| Invertible residual reexpression | Preserves roots/sensitivities; changes off-root iterations | A specific conditioning or convergence improvement including transformation cost |
| Global polynomial contact checker | Classical criterion, implemented as bounded exact algebra | Useful coverage/diagnostics compared with existing symbolic algebra tools |
| Finite-basis calculus transfer | Exact only with the stated realization/closure conditions | A new representation, theorem or measured advantage over established exact methods |
| Generic nonlinear PDEs reduced to a fixed linear system | Unsupported; unrestricted finite nonlinear closure fails | Additional structure, approximation analysis or a proved special case |

The focused tests exercise independent plane/line distance formulas,
Pythagorean optimality, equation recombination, redundancy, inconsistency,
resource failures, raw derivative normalization, global wave/boundary
identities, contact failures and finite-sample false passes. Scientific-ML,
tabular and LP measurements are recorded separately by the pilot driver; these
mathematical identities alone do not establish a learning or performance gain.

## 7. Further exact transfers and their scope

### Moving hyperplanes turn affine transport into linear ODEs

For the whole-space transport equation

\[
u_t+(A(t)x+c(t))\cdot\nabla u=0,
\]

set \(u(x,t)=\phi(w(t)^Tx+b(t))\), with a fixed C¹ profile and C¹ parameters.
The chain rule gives the residual

\[
\phi'(w^Tx+b)\{[w'+A^Tw]^Tx+b'+c^Tw\}.
\]

Consequently the linear ODE system

\[
\boxed{w'=-A^Tw,\qquad b'=-c^Tw}
\]

transfers its solution to an exact PDE solution with the prescribed ridge
initial field. Sums of such transported profiles also solve the linear PDE.
This is a concrete calculus–geometry map: the normal and offset of each
hyperplane evolve instead of independently fitting the field at every point.
The construction is the classical method of characteristics, described in
[Mishra's transport notes](https://metaphor.ethz.ch/x/2019/hs/401-4671-00L/literature/mishra_hyperbolic_pdes.pdf).
It is not a new solution method. Arbitrary initial fields still need a suitable
representation, boundary/inflow data remain separate, and nonaffine velocities
generally bend planes outside this finite geometric family.

For shear velocity \(v=(y,0)\), \(w(t)=(1,-t)\) and constant bias give
\(u(x,y,t)=\phi(x-ty)\). The following exact polynomial check includes the
variable PDE coefficient and the complete initial-field identity:

```python
from omnibias.core.jet_geometry import (
    LinearJetEquation, PolynomialJetBoundary,
    check_polynomial_jet, polynomial_jet,
)
from omnibias.core.realization.polynomial import SparsePolynomial as P

x, y, t = (P.variable(3, axis) for axis in range(3))
shear = check_polynomial_jet(
    polynomial_jet((x-t*y)**3, 1),
    equations=(LinearJetEquation({(0, 0, 1): 1, (1, 0, 0): y}),),
    boundaries=(PolynomialJetBoundary((0, 0, 0), 2, 0, x**3),),
)
assert shear.valid
```

### Fractional operators need the domain and integration limits

For a nonzero ridge direction w and an appropriate bounded smooth profile,
the whole-space fractional Laplacian has the profile transfer

\[
(-\Delta_x)^{\alpha/2}\phi(w^Tx+b)
=\|w\|^\alpha\bigl((-D_z^2)^{\alpha/2}\phi\bigr)(w^Tx+b),\qquad0<\alpha<2.
\]

The Fourier multiplier \(\|\xi\|^\alpha\) restricted to the ridge's frequency
line proves the identity, with distributional interpretations where necessary.
For a bounded domain, a zero exterior extension contributes an additional
exterior integral; it is generally not another ridge profile. Caputo and
fractional integral transfers similarly require their lower terminals to
transform with the coordinate. Thus a fractional bias operator cannot silently
replace an initial-boundary-value spatial operator. The definitions and exterior
conditions are explained in [Lischke et al.'s comparative review](https://arxiv.org/abs/1801.09767).
Analytical fractional profile/RBF methods already exist, including
[Burkardt, Wu and Zhang](https://arxiv.org/abs/2009.10811). No fractional kernel
or fractional-PDE performance claim is added by this pilot.

## 8. Continuous neuron parameters and an exact LP dual certificate

Fix direction one and atoms \(\psi_b(x)=(x-b)_+^2\), with bias b in a compact
interval. Finite point observations define a linear map from signed measures
over biases to data. Consider

\[
\inf_\mu\|\mu\|_{\mathrm{TV}}
\quad\text{subject to}\quad
\int\psi_b(x_i)\,d\mu(b)=y_i.
\]

Any finite neural atom list that is exactly feasible supplies an upper bound
\(\sum_j|a_j|\), after combining duplicate biases. For any vector q with
\(\sup_b|\sum_iq_i\psi_b(x_i)|\le1\), weak duality gives the lower bound
\(q^Ty\). Strong duality need not be assumed to check these bounds or to
establish optimality when they match.

For rational data and q, the correlation is a rational quadratic polynomial on
each bias interval between data knots. Its absolute maximum is attained at an
endpoint or an in-cell stationary point. Those stationary points are rational,
so a finite exact enumeration checks **every permitted continuous bias**. If
the exact maximum is U>0, q/U is feasible. U=0 with nonzero target pairing is
instead an infeasibility witness, not a valid finite optimum certificate.

For \(x=(1,2)\), \(y=(1/4,9/4)\), b in [0,1], the endpoint dictionary has
optimum 3/2. The interior atom at b=1/2 fits with coefficient cost one. The
endpoint-feasible dual q=(-3,1) has correlation
\(-2b^2+2b+1\), whose actual maximum is 3/2. The scaled q=(-2,2/3) proves the
matching lower bound one. Thus finite-grid dual feasibility cannot establish a
continuous-dictionary certificate.

This measure/spline connection is established; see
[Bach 2017](https://jmlr.org/papers/v18/14-546.html),
[Parhi and Nowak 2021](https://jmlr.org/papers/v22/20-583.html) and
[Pilanci and Ergen 2020](https://proceedings.mlr.press/v119/pilanci20a.html).
Distributionally, \(D_x^3\psi_b=2\delta_b\) connects the representation to
sparse sources. The current pilot fits point moments, not a PDE, and supplies
no unobserved-field error bound. It covers a fixed direction, squared-ReLU
profile and bounded bias interval, not arbitrary deep networks or variable
directions. [Retained experimental results](calculus-geometry-results.md)
include dense-grid controls, unresolved gaps, infeasible initial dictionaries
and known-generator fallback upper bounds.
