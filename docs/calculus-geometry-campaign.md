# Calculus–geometry campaign: mathematics and evidence

This report accompanies the bounded research campaign of 19 October 2026. The
experiment implementations belong to the adjacent `omnibias-lab` consumer.
The [earlier pilot](calculus-geometry-results.md) remains a separate, frozen
experiment. The constructions below preserve exact equality as a hard readout;
their training derivatives belong to a specified smooth computation.

The strongest result is a working **verified physical-calibration prototype**:
global field bounds, physical sensitivity enclosures and a proved finite loss
decrease on a fixed parameter interval. A separately derived energy certificate
also succeeds on a candidate for which the old contraction gate is provably
blocked. Its foundation is classical monotone-operator theory; the contribution
demonstrated here is the checked computational capability.

Correct output constraints reduced public TabPFN-v2 synthetic regression error
by **11.45% on five fresh confirmation seeds**, using ordinary QR projection.
The primary adaptive-speed and Boolean-learning candidates did not meet their
registered promotion gates. These findings support focused continuation in
verified calibration and constraint-aware prediction; they do not establish a
general theory or collapse-specific advantage. Section 5 retains all controls,
missed targets, setup limitations and replay results.

## 1. The maps and their obligations

Write a neural realization, its jet, and its residual as

\[
\theta\xrightarrow{R}u_\theta
\xrightarrow{j^q}(u_\theta,Du_\theta,\ldots,D^qu_\theta)
\xrightarrow{C}(\mathcal N[u_\theta],\mathcal B[u_\theta]).
\]

The zero set of the last map describes represented solutions. A transformation
of coordinates is useful only when its effect on these maps is established.
The following propositions are established mathematics, with proofs included
to make the computational contracts explicit.

An arbitrary assignment of derivative coordinates is often called a formal
jet; one obtained by differentiating a single field is holonomic. A PDE
constrains jet coordinates, while compatibility constrains which coordinate
assignments come from a field. These are distinct requirements. The useful
compiler contract is to preserve both, together with boundary information.

### Affine geometry and target constraints

Let \(A\in\mathbb R^{m\times n}\) have full row rank and
\(M=\{x:Ax=b\}\). The minimum-distance problem has Lagrangian
\(\tfrac12\|y-x\|^2+\lambda^T(Ay-b)\). Stationarity gives
\(y=x-A^T\lambda\), and feasibility gives
\(AA^T\lambda=Ax-b\). Consequently

\[
\Pi_Mx=x-A^T(AA^T)^{-1}(Ax-b),\quad
L_M(x)=\tfrac12\|x-\Pi_Mx\|^2,
\]
\[
\nabla L_M=x-\Pi_Mx,\qquad
D\Pi_M=I-A^T(AA^T)^{-1}A.
\]

For reduced QR \(A^T=QR\), compute the correction as
\(Q R^{-T}(Ax-b)\). This uses a triangular solve and never forms a Gram
inverse. Numerical rank checks and numerical feasibility are distinct from
exact rational membership.

A linear system can also be differentiated with respect to its parameters.
For a nonsingular square system \(A(\theta)x(\theta)=b(\theta)\),
\(dx=A^{-1}(db-(dA)x)\). Linearity in the unknown does not make these
parameter derivatives zero. For an underdetermined system the derivative also
depends on the selection rule; the affine projection retains the starting
point's tangent component.

If the true vector target \(y\in M\), orthogonality proves

\[
\|\mu-y\|^2=\|\Pi_M\mu-y\|^2+\|\mu-\Pi_M\mu\|^2.
\]

Thus a correct affine constraint cannot increase squared prediction error.
This is an application guarantee, independent of a derivative engine. If the
target is \((0,0)\) and a wrong constraint forces its coordinates to sum to one,
projection produces \((1/2,1/2)\), increasing coordinate-mean squared error to
\(1/4\). There is no corresponding unconditional classification theorem.

The normal projector and minimum-norm anchor

\[
P_N=QQ^T,\qquad q=QR^{-T}b
\]

are unchanged under \((A,b)\mapsto(SA,Sb)\) for invertible \(S\).
They specify the same affine set by \(P_Nx=q\), and hence are a complete
invariant of a nonempty affine set in a fixed Euclidean coordinate system.
Changing measurement units requires an explicit metric. Supplying this context
to a PFN supplies task information; it does not demonstrate that the PFN
inferred the constraints.

### Nonlinear roots, iteration maps and nearest points

For a regular residual \(r(y,\theta)\), the least-norm solution of
\(J(y)\delta=-r(y)\) is a correction to the *linearization*. Iterating these
corrections defines an iteration map whose derivative includes its dependence
on the seed. A root alone supplies no selection rule on an underdetermined
solution manifold.

A nearest-point branch instead satisfies the augmented equations

\[
y-x+J(y,\theta)^T\lambda=0,\qquad r(y,\theta)=0.
\]

Its differential is determined by

\[
\begin{bmatrix}
I+\sum_i\lambda_i\nabla_y^2r_i&J^T\\J&0
\end{bmatrix}
\begin{bmatrix}dy\\d\lambda\end{bmatrix}
=
\begin{bmatrix}dx-\partial_\theta(J^T)\lambda\,d\theta\\
-\partial_\theta r\,d\theta\end{bmatrix}.
\]

LICQ and a positive definite reduced Hessian make this bordered matrix
nonsingular and select a local stationary minimum branch. Calling it the
nearest point additionally requires the chosen neighborhood to have a unique
nearest point, or a justified global selection. Omitting the curvature term
changes the derivative. On a circle, a tangent-plane correction and radial
nearest projection already differ away from the circle.

### Exact transfers between coefficient systems and PDEs

Let \(T:C\to V\) be linear and satisfy \(\mathcal LT=TM\). If
\(f=Tg\), then \(Mc=g\) implies \(\mathcal L(Tc)=f\). Boundary conditions
must also transfer: solve \((\mathcal BT)c=h\). This gives all solutions in
the represented family only when the coefficient equations correctly account
for \(\ker T\); an injective realization avoids that ambiguity. It gives every
PDE solution only if the representation contains every admissible solution,
which the finite pilots do not claim.

For nonlinear evolution \(u_t=\mathcal N(u)\), a differentiable realization
\(u=T(c)\) transfers coefficient dynamics \(\dot c=F(c)\) whenever

\[
DT(c)F(c)=\mathcal N(T(c)).
\]

The proof is the chain rule along the coefficient trajectory. Initial data,
boundary conditions, existence intervals and the domain of \(T\) remain
obligations. Finite linear polynomial families generally fail nonlinear
closure: squaring a nonzero quadratic generates a fourth-degree term.

For the concrete nonlinear example, take \(\nu>0\) and

\[
v=c_0+\sum_{k=1}^K e^{-\nu k^2t}
[a_k\cos kx+b_k\sin kx],\qquad
c_0>\sum_k(|a_k|+|b_k|).
\]

Then \(v\ge\delta>0\) for \(t\ge0\), \(v_t=\nu v_{xx}\), and
\(u=-2\nu v_x/v\) satisfies \(u_t+uu_x=\nu u_{xx}\). To prove it, put
\(\ell=\log v\), use
\(\ell_t=\nu(\ell_{xx}+\ell_x^2)\), and differentiate once in space.
This is the classical Cole–Hopf map, with exact periodic boundaries and a
restricted initial-data family. The direct transformed spectral solver is the
same mathematical control.

## 2. Certificates for geometric transformations

### Derivative compatibility

Suppose all proposed jet components \(J_\alpha\), including \(J_0\), are
polynomials. The finite conditions

\[
\partial_iJ_\alpha=J_{\alpha+e_i}\quad (|\alpha|<q)
\]

are necessary and sufficient for the entire jet to equal \(j^qJ_0\).
Necessity follows from differentiation; sufficiency follows by induction on
\(|\alpha|\). Checking equality of every coefficient is a computable exact
certificate. Polynomial PDE and boundary conditions add finite coefficient
equations. A correction in coefficient space can therefore retain one genuine
field; the certificate covers the chosen polynomial family.

Without a supplied value component, curl-free first derivatives yield a
potential on a simply connected domain, up to an additive constant. Topology
matters: \((-y/(x^2+y^2),x/(x^2+y^2))\) has zero curl on the punctured plane
but integrates to \(2\pi\) around the unit circle. Pointwise equation
constraints alone are insufficient: proposed derivatives \(p_x=0,p_y=x\)
violate mixed-partial compatibility.

There is also an explicit reconstruction map, rather than only a test. On a
domain star-shaped about zero, for a closed vector field \(p\), define

\[
T(p)(x)=u_0+\int_0^1p(tx)\mathbin{\cdot}x\,dt.
\]

To prove \(\partial_kT(p)=p_k\), differentiate the integral and use
\(\partial_kp_i=\partial_ip_k\). Its integrand becomes
\(p_k(tx)+t\sum_i x_i\partial_ip_k(tx)
=\tfrac{d}{dt}[t p_k(tx)]\). Integrating proves the claim. For a monomial
\(p_i(x)=c x^\alpha\), the contribution is exactly
\(c x^{\alpha+e_i}/(|\alpha|+1)\). Polynomial reconstruction therefore uses
only rational linear algebra.

The additional Python prototype projects two degree-two derivative polynomials
onto their three curl constraints, then reconstructs a degree-three potential.
For the incompatible proposal \((0,x)\), it returns \((y/2,x/2)\) and
\(u=xy/2\). Exact symbolic identities verify \(\Delta u=0\) on the square
and its prescribed boundary values on all four edges. The maximum principle
then gives uniqueness among classical solutions for these boundary data.
This corrects the derivative field and derives the value field; it does not
preserve an independently supplied incompatible value field. The projection
uses a Euclidean coefficient metric, with no assertion that this is the nearest
field in a physical norm. This is the polynomial Poincaré construction;
[finite element exterior calculus](https://arxiv.org/abs/0906.4325) provides a
much broader established theory of compatible differential spaces and
projections.

Even valid observed jets do not establish generalization. For finitely many
observations \(x_i\), adding
\(C\prod_i(x-x_i)^{q+1}\) preserves derivatives through order \(q\) at every
observation while changing the field elsewhere. A well-posed PDE with complete
boundary/initial data can add information that these jets do not contain.

### Bias-cluster compression

For a common direction \(w\), write

\[
u(x)=\sum_j a_j\sigma(w^Tx+b+\epsilon_j),\quad
v(x)=\sum_{k=0}^p m_k\sigma^{(k)}(w^Tx+b),\quad
m_k=\frac{\sum_j a_j\epsilon_j^k}{k!}.
\]

Taylor's theorem applied to \(\sigma^{(|\alpha|)}\) gives

\[
\|D^\alpha(u-v)\|_\infty\le
|w^\alpha|\frac{\sum_j|a_j||\epsilon_j|^{p+1}}{(p+1)!}
\sup_I|\sigma^{(p+|\alpha|+1)}|=:e_\alpha,
\]

provided the entire segments between each shifted argument and the central
argument lie in \(I\). The interval fixture explicitly checks this domain.
It compares direct evaluation, derivative features and an 80-digit Newton
divided-difference reference. Different precision precludes a speed conclusion.

For a boundary cage \(g\), the derivative error transfers by Leibniz:
\(\|D^k[g(u-v)]\|\le\sum_{j=0}^k {k\choose j}
\|D^{k-j}g\|e_j\). Structural clamped boundary conditions are preserved.
For \(R(u)=u''''+\lambda u^3-f\) and \(\|u\|\le B\),

\[
\|R(v)-R(u)\|\le e_4+
|\lambda|(3B^2e_0+3Be_0^2+e_0^3).
\]

If a validated solution lies within \(\rho\) of \(u\), it lies within
\(\rho+e_0\) of \(v\). Meeting the original tolerance requires the error
budget to accommodate this increase, or a fresh residual certificate for \(v\).
These statements do not preserve parameter or selected-solution sensitivities
automatically; those require derivative bounds on the transformation and a
conditioned implicit solution map.

### A constructive physical-sensitivity certificate

For the clamped problem \(u''''+\lambda u^3=384\), suppose a field certificate
proves \(\|u-v\|_\infty\le\delta\). Let \(B\) bound both fields and
\(M=1/384\) be the clamped Green-operator norm. If
\(\kappa=3|\lambda|B^2M<1\), Neumann inversion gives

\[
\|L_u^{-1}\|_{\infty\to\infty}\le R=\frac{M}{1-\kappa},
\qquad L_u=D^4+3\lambda u^2.
\]

The implicit function theorem then gives the physical sensitivity
\(L_u u_\lambda=-u^3\), with homogeneous clamped boundary data. For any
clamped sensitivity surrogate \(s\), enclose

\[
\eta\ge\|s''''+3\lambda v^2s+v^3\|_\infty,\qquad
S\ge\|s\|_\infty.
\]

Subtracting this equation from the true variational equation and using
\(|u^2-v^2|\le2B\delta\) and \(|u^3-v^3|\le3B^2\delta\) proves

\[
\boxed{\|u_\lambda-s\|_\infty\le
R\,[\eta+6|\lambda|B\delta S+3B^2\delta].}
\]

Every quantity on the right is computable from whole-domain enclosures. The
prototype fits a separate sensitivity readout by linear least squares, snaps
it to rational parameters, and checks its variational residual independently.
It does not identify the derivative of the network fitting procedure with the
physical solution derivative. At an observation \(x_0\), the intervals
\(u(x_0)\in v(x_0)+[-\delta,\delta]\) and
\(u_\lambda(x_0)\in s(x_0)+[-E,E]\) enclose the calibration gradient
\((u(x_0)-y)u_\lambda(x_0)\). Excluding zero certifies its sign at the stated
parameter. A finite optimization step requires additional neighborhood control.
This is an application of standard a posteriori residual and inverse bounds,
with an explicit scientific-learning use case.

For a finite step, repeat both enclosures on an entire parameter interval
\([a,b]\). If its calibration-gradient enclosure has upper endpoint
\(-\gamma<0\), the fundamental theorem of calculus proves
\(J(b)-J(a)\le-\gamma(b-a)<0\). A separate bounded follow-up tests the
fixed window \([32,33]\), with one frozen width-12 field, measurement \(9/10\)
and field radius \(1/100\). It must certify the whole window; a pointwise
gradient sign alone is insufficient.

### A global monotone-energy alternative

For the clamped problem on \([0,1]\), a positive Green kernel gives a
stronger stability contract than the preceding contraction test. For
\(0\le x\le y\le1\), write

\[
G(x,y)=\frac{x^2(1-y)^2}{6}\,[3y-x(1+2y)],\qquad G(y,x)=G(x,y).
\]

It satisfies the clamped boundary conditions, is twice continuously
differentiable across the diagonal, and has unit jump in its third derivative.
The bracket is nonnegative because it is at least \(2y(1-y)\). Direct
integration and diagonal maximization give

\[
\sup_xG(x,x)=\frac1{192},\qquad
\int_0^1\!\int_0^1G(x,y)\,dx\,dy=\frac1{720}.
\]

Equip \(H^2_0(0,1)\) with energy norm \(\|z''\|_2\). Its evaluation
representer is \(G(x,\cdot)\), hence
\(\|z\|_\infty\le\|z''\|_2/\sqrt{192}\). To bound the one-norm,
let \(s=\operatorname{sign}z\). Green representation, Cauchy–Schwarz and
positivity give

\[
\|z\|_1=\langle s,z\rangle
\le\langle s,Gs\rangle^{1/2}\|z''\|_2
\le\frac{\|z''\|_2}{\sqrt{720}}.
\]

For \(z''''+q(x)z=f\) with clamped boundaries and bounded \(q\ge0\),
testing the weak equation with \(z\) yields

\[
\|z''\|_2^2+\int qz^2
\le\|f\|_\infty\|z\|_1,
\qquad
\boxed{\|z\|_\infty\le C\|f\|_\infty,
\quad C=\frac1{96\sqrt{15}}.}
\]

The zero-energy case is immediate; otherwise divide by \(\|z''\|_2\).
This proof does **not** assert positivity of the inverse of \(D^4+q\).
It uses positivity of the unperturbed one-dimensional Green kernel only.
General clamped biharmonic domains need not have that property.

For \(u''''+\lambda u^3=f\), \(\lambda\ge0\), existence and uniqueness
follow by minimizing the coercive, strictly convex energy
\(\frac12\int|u''|^2+\frac\lambda4\int u^4-\int fu\) on
\(H^2_0\). Weak lower semicontinuity and the one-dimensional Sobolev embedding
give a minimizer; its Euler equation and continuous \(f\) give a classical
\(C^4\) solution. For any clamped \(C^4\) candidate \(v\), subtracting
equations sets \(q=\lambda(u^2+uv+v^2)\ge0\). Therefore

\[
\boxed{\|u-v\|_\infty\le
C\|v''''+\lambda v^3-f\|_\infty.}
\]

No smallness assumption on \(\lambda\), the candidate amplitude or its error
radius is needed. A directed rational upper bound on \(C\), validated by
\(138240\,C_{\rm up}^2\ge1\), makes this a computable certificate when
paired with the existing whole-domain residual enclosure. The physical
sensitivity operator \(D^4+3\lambda u^2\) has the same inverse bound;
the earlier sensitivity proof may consequently replace its Neumann constant
with \(C\). Field approximation, residual enclosure and differentiating the
physical solution remain separate obligations.

This is an application of classical energy, Green-kernel and monotone-operator
arguments. [Dunninger and Locker (1975)](https://msp.org/pjm/1975/60-1/pjm-v60-n1-p05-p.pdf)
already establish existence and uniqueness for nonlinear monotone biharmonic
problems. [Grunau and Robert](https://arxiv.org/abs/0705.3301) explain the
limitations of positivity for general clamped domains. The explicit bound above
is proved here for this verification contract; a new mathematical principle
or optimal constant is not claimed.

### Residual coordinates and a verifiable neighborhood

Let \(r(x,\theta)=0\) have a selected regular isolated solution and set
\(q(x,\theta)=S(x,\theta)r(x,\theta)\). If
\(\sigma_{\min}(S(x_*,\theta_*))=s>0\) and a proved variation bound is
\(\|S(x,\theta)-S_*\|\le L\rho<s\), then \(S\) is invertible throughout
that ball. Thus \(q=0\) and \(r=0\) have exactly the same zeros there.
At a root, \(q_x=S r_x\) and \(q_\theta=S r_\theta\); the terms multiplied
by \(r\) vanish. Therefore

\[
-q_x^{-1}q_\theta=-r_x^{-1}r_\theta.
\]

For constant \(S=r_x(x_*,\theta_*)^{-1}\), a bound
\(\|r_x(x,\theta_*)-r_x(x_*,\theta_*)\|\le L_J\rho\) gives
\(\|q_x-I\|\le\|S\|L_J\rho\). When this is \(\eta<1\),
\(\kappa_2(q_x)\le(1+\eta)/(1-\eta)\). This is classical local
preconditioning with an explicit sufficient neighborhood. Building and
applying \(S\) costs work; changing the metric can also change off-root
iterations. Preserving zeros does not preserve every training trajectory.

For a fully explicit nonlinear example, let
\(q_i(x,\theta)=x_i+x_i^2/4-\theta_i\),
\(r=\operatorname{diag}(10^{-6},1)q\), and
\(S=\operatorname{diag}(10^6,1)\). On \(\|x\|_\infty\le1/4\),
\(7/8\le\partial_iq_i\le9/8\), so
\(\kappa_2(q_x)\le9/7\). The scalar map sends the interval endpoints to
\(-15/64\) and \(17/64\), respectively; it is strictly increasing there.
Thus each \(\theta_i\in[-1/8,1/8]\) selects exactly one root in that box.
Both residual coordinates give the same sensitivity
\(\partial x_i/\partial\theta_i=(1+x_i/2)^{-1}\). The unscaled Jacobian is
ill-conditioned while the selected solution and its derivatives are unchanged.
This proves a concrete conditioning improvement with an explicit neighborhood;
it is classical residual scaling, rather than a new linearization principle.

## 3. Equality, Boolean expectations and optimization

The exact equality gate is \(E(z)=\mathbf1\{z=0\}\). On integer residuals,

\[
\operatorname{ReLU}(z+1)-2\operatorname{ReLU}(z)+
\operatorname{ReLU}(z-1)=\max(0,1-|z|)=E(z).
\]

The four linear pieces prove the identity. At \(z=1/2\), the value is
\(1/2\), establishing the integrality restriction. Integer weights and bias,
or a proved attainable-residual gap with corresponding scaling, are essential.

For independent Bernoulli variables with probabilities \(p_j\), the count PMF
is the coefficient vector of \(\prod_j(1-p_j+p_jt)\). For response table
\(a_0,\ldots,a_m\),

\[
q=\mathbb E[a_S],\qquad
\partial_{p_j}q=\mathbb E[a_{S_{-j}+1}-a_{S_{-j}}].
\]

Conditioning on one variable proves the derivative. Enumerating all assignments
shows exact equality with a full LUT whose entries are tied by Hamming weight.
The campaign uses deterministic observed inputs and independently randomized
wire choices, which makes these conditional probabilities well-defined even
if the observed features are correlated. Hard argmax wires may duplicate;
replay evaluates their actual deterministic circuit. Marginal composition
does not in general retain global joint semantics: soft XOR of \((p,p)\)
is \(2p(1-p)\), whereas \(X\operatorname{xor}X=0\).

The optimized expected-accuracy objective is exactly
\(N^{-1}\sum_i[y_iq_i+(1-y_i)(1-q_i)]\) for that stochastic model. Its
derivative is not an ordinary derivative of the deterministic hard predicate.
Parity can have vanishing expected gradients at symmetric probabilities.

For one learned block, let \(p_j^*\) be the probability of its selected hard
wire and let \(e\) bound the largest response-table rounding error. Coupling
the stochastic block with the hard block gives the uniform bound

\[
\sup_{x\in\{0,1\}^d}|q(x)-h(x)|
\le 1-\left(\prod_j p_j^*\right)(1-e).
\]

**Proof.** All sampled wires equal their selected hard wires with probability
\(\prod_jp_j^*\). On that event the expected response differs from the hard
response by at most \(e\); on its complement the difference is at most one.
The argument is valid for every deterministic input, including duplicated
wires. Directed exponential and sigmoid enclosures plus rational arithmetic
can bound the right side for the exact real stochastic model defined by the
saved parameters. Floating framework evaluation is checked separately. This
is a classical coupling bound; it can be too loose to be useful and supplies
no ordinary derivative of the hard circuit.

If the right side is strictly below \(1/2\), thresholding the soft probability
at \(1/2\) gives exactly the hard decision on **every Boolean input**. A
post-training procedure scales both routing and response logits by a common
\(\beta\), doubles \(\beta\) until a directed enclosure is below \(10^{-3}\),
and otherwise refuses at its budget. Strictly positive routing gaps and
nonzero response logits ensure eventual convergence of this sufficient bound.
Ties can make it inconclusive. A second evaluator uses directed mpmath
expressions independently of the core interval wrappers. The hard model is
unchanged; this certifies approximation to that model, not its accuracy against
unknown labels.

The Boolean domain is essential here. A continuous function cannot uniformly
approximate a continuous-space equality indicator with error strictly below
\(1/2\) in a neighborhood containing the surface and arbitrarily close
off-surface points: continuity would force its limiting off-surface value to
equal its on-surface value. Exact membership remains a separate hard readout.
The finite-domain certificate does not replace that intended definition.

### A concrete statistical reason to test counting units

For a *known* support of size \(m\), arbitrary hard symmetric Boolean
responses have \(2^{m+1}\) possibilities, compared with \(2^{2^m}\) for a
general LUT. If the support is unknown but consists of distinct inputs, the
counting hypothesis class has at most
\({d\choose m}2^{m+1}\) functions. Because the implemented decoder allows
duplicate wires, its correct upper bound is instead

\[
|\mathcal H_{\rm count}|\le {d+m-1\choose m}2^{m+1}.
\]

**Proof.** A count ignores wire order. There are
\({d+m-1\choose m}\) multisets of \(m\) wires from \(d\) inputs and
\(2^{m+1}\) binary response tables. Different descriptions may define the
same function, so this is an upper bound. For unrestricted LUTs the coarse
bound \(d^m2^{2^m}\) suffices. For a finite class and independent observations
with zero-one loss, a union bound applied to Hoeffding's inequality gives,
simultaneously with probability at least \(1-\delta\),

\[
|R(h)-\widehat R(h)|\le
\sqrt{\frac{\log(2|\mathcal H|/\delta)}{2N}}
\quad\text{for every }h\in\mathcal H.
\]

This classical argument supplies a scoped reason for a counting prior to help
with a structured target. It neither guarantees that gradient optimization
finds the best model nor favors counting on nonsymmetric targets. It also
explains why support recovery and an exhaustive small-instance control matter.
The hard class bound does not automatically extend to every stochastic soft
training model or to arbitrarily composed deep circuits.

Here \(R\) is ordinary classification risk. The experiment samples a finite
cube without replacement and also reports balanced accuracy; the displayed
bound supplies a capacity argument rather than a numerical confidence claim
for those reported pilot scores.

### An exact bridge from derivatives to Boolean interactions

Repeated application of the fundamental theorem of calculus gives

\[
\sum_{S\subseteq[k]}(-1)^{k-|S|}
\phi\!\left(z+\sum_{i\in S}w_i\right)
=\left(\prod_iw_i\right)
\int_{[0,1]^k}\phi^{(k)}\!\left(z+\sum_iw_it_i\right)dt.
\]

For \(k=1\), this is the usual integral formula for a finite difference;
induction applies it once more in the next coordinate. Thus an interval bound
on \(\phi^{(k)}\) over the full argument interval certifies the size and,
when it excludes zero, the sign of this Boolean interaction. Replacing the
integral with a single derivative at finite spacing is generally incorrect.
This is a classical discrete-calculus identity with a useful implementation
contract: derivative enclosures can certify interaction restrictions without
pretending that general Boolean optimization has become easy.

For LPs \(\min c^Tx\) subject to \(Ax\le b\), a proposed basis \(I\)
provides the candidate equations \(A_Ix=b_I\) and
\(A_I^T\lambda_I=-c\). Exact primal feasibility, \(\lambda_I\ge0\),
stationarity and complementarity prove optimality by weak duality. Finding a
valid basis remains part of the algorithm. The experiment charges archive
construction, proposal, exact checks and conventional fallback.

For constant \(A\) and affine parameterized \(b,c\), a nonsingular basis gives
affine formulas for \(x(\theta)\) and \(\lambda_I(\theta)\). Strictly positive
inactive slacks and basis multipliers give a verifiable neighborhood preserving
the active set: in the infinity norm, take the minimum of each positive slack
or multiplier divided by the one-norm of its parameter derivative. A zero
denominator imposes no restriction. A zero margin refuses a positive-radius
claim. At a transition, a globally differentiable LP solution map need not
exist; quadratic regularization defines a different optimization problem.

For mixed value and first-derivative observations of
\(\psi_b(x)=(x-b)_+^2\), dual pricing is

\[
c(b)=\sum_i q_i^{(0)}(x_i-b)_+^2+
2\sum_jq_j^{(1)}(x_j-b)_+.
\]

It is a continuous piecewise quadratic in \(b\). Data knots, interval
endpoints and interior stationary points exhaust its absolute maximum.
Rational inputs permit exact pricing and a checked continuous-dictionary dual
bound. An independent checker reconstructs each quadratic from three exact
values. Second-derivative observations at knots require an additional
convention and are excluded. A moment certificate gives no error bound away
from the observed functionals.

## 4. Claim-to-source comparison

| Candidate | Relevant established work | What this campaign can add |
| --- | --- | --- |
| Exact linear-PDE representation | [EPGP](https://proceedings.mlr.press/v202/harkonen23a.html) constructs exact priors for constant-coefficient linear PDE systems | Checked realization and boundary contracts on a restricted family |
| Hard prediction constraints | [KKT-hPINN](https://github.com/li-group/kkthpinn) uses linear equality projection; [ENFORCE](https://github.com/process-intelligence-research/ENFORCE) uses adaptive nonlinear corrections | Reliable application, same-algebra comparison and explicit failure states |
| Neural residual certification | [∂-CROWN](https://arxiv.org/abs/2305.10157) bounds PINN residual errors | A measured adaptive enclosure policy with independently replayable leaf certificates |
| Residual-specific derivatives | [Explicit derivative kernels](https://arxiv.org/abs/2606.29702) discuss partial-jet closure and verified kernels | A particular verified implementation benefit, if the matched control supports it |
| Equality/counting units | [Equality gates](https://arxiv.org/abs/2402.08751) and [exact cardinality inference](https://proceedings.mlr.press/r10/tarlow12a.html) precede this study | Evidence about learning unknown wiring and response tables under bounded resources |
| Symmetry and learned logic | [SymSATNet](https://papers.neurips.cc/paper_files/paper/2022/hash/5642b9811a9ac5281be1cc84c275f251-Abstract-Conference.html) ties parameters using rule symmetries and also studies discovering symmetries; [Predict and Constrain](https://proceedings.mlr.press/v80/brukhim18a.html) integrates cardinality into deep structured prediction | A scoped implementation and learning comparison; symmetry reduction or differentiable cardinality alone is not a first |
| Collision/compression coordinates | [Confluent Prony representations](https://www.journalofsing.org/volume10/batenkov-yomdin.pdf), Taylor remainders and divided differences | Certified adaptive grouping or sensitivity guarantees, if a further argument succeeds |
| Tabular foundation integration | [Official TabPFN](https://github.com/PriorLabs/TabPFN) | Constrained output application and equation-invariant context feasibility; no PFN pretraining claim |
| Differentiable optimization | [CVXPYlayers](https://github.com/cvxpy/cvxpylayers) and standard active-set/KKT theory | End-to-end measured amortization with exact acceptance and explicit fallback |
| Active-set validity regions | [Multiparametric LP geometry](https://doi.org/10.1023/B:JOTA.0000004869.66331.5c) studies parameter-space regions and degeneracy | Exact restricted witnesses and sensitivity checks; the region construction itself is established |
| Certified scientific outputs | [Certified reduced-basis methods](https://link.springer.com/article/10.1186/2190-5983-1-3) already combine residual bounds, stability constants and parameterized outputs | A checked neural-field sensitivity and calibration prototype; no first certified-surrogate claim |

The source review is targeted, not an exhaustive novelty search. A combination
becomes a potentially new result only after a specific claim, complete argument
or reproducible advantage survives the directly relevant comparison.

## 5. Computational evidence

The final CPU campaign completed every planned discovery stage. Its source and
full case matrices passed independent replay. An earlier run is retained with
its source snapshot: integrated replay exposed a missing stage adapter, which
was repaired with an integration regression before the final rerun. Candidate
recipes were unchanged. Discovery seeds were 401, 503 and 607; confirmation used
809, 907, 1009, 1103 and 1201.

The public [machine-readable summary](benchmarks/calculus_geometry_research.json)
contains aggregate values and evidence digests. Local raw artifacts retain
models, complete partitions, failures, timings and dependency fingerprints.

Registration discloses an engineering-smoke overlap: before the full matrix,
one seed-401 counting/exactly-three run's test accuracy was seen. No candidate
recipe was tuned to it; the LUT starting function was strengthened before the
discovery matrix. This limits how strictly that discovery result can be called
untouched. The five confirmation seeds remained fresh.

Comparator setup used an isolated environment, about 1.6 GB at setup, and reused
the existing CPU-accessible Torch. An initial resolver began unnecessary CUDA
downloads and was stopped. Official TabPFN-v2 public weights (44,390,977 bytes)
were available; the current default weights were gated and not used. The
installed TabPFN library was 9.2.0. Official DiffLogic/DiffLUT CUDA comparisons
were unavailable. CVXPYlayers and actual HiGHS warm-basis controls were available.
The local LUT control is never labeled an official learned-logic benchmark.

### Scientific verification


The report was generated 124.6 minutes after implementation began, within the eight-hour ceiling. Stage deadlines are maxima; unused time was used for failure analysis and the separate certificate constructions.


| Derivative oracle | Adaptive completed cases | Best uniform completed cases | Best uniform / adaptive total cost |
| --- | ---: | ---: | ---: |
| core | 10/12 | 9/12 | 0.840× |
| independent | 10/12 | 9/12 | 0.834× |


156/216 verification attempts passed; all accepted certificates replayed.
Each case has three repetitions. Costs include certificate construction and
independent replay; ratios use cases where both methods completed. Candidate
fitting is recorded separately because this comparison starts with a supplied
surrogate. The adaptive policy did not meet the registered 2× or 25-percentage-
point completion gate. Agreement with the independent derivative implementation
attributes any shared gain to enclosure strategy.


A separate follow-up removes repeated full-partition scans with max-heaps and
retains a selected derivative order in refined children. A regression with order
inheritance disabled reproduces the original partition and gate exactly. The
follow-up reruns every frozen candidate and all controls serially; it does not
refit candidates or replace the original results.

| Oracle | Best control / heap total cost | Common completed cases |
| --- | ---: | ---: |
| core | 1.123× | 10/12 |
| independent | 1.121× | 10/12 |

These are engineering measurements of bookkeeping and enclosure reuse. The original classical solver results remain separate end-to-end controls; verification speed is not training or PDE-solution speed.


### Physical parameter gradients

A separate variational readout approximates the physical coupling sensitivity.
The derived global error bound combines the certified field error, a rigorous
sensitivity-residual enclosure and the inverse bound. Every enclosure and
calibration-gradient interval replayed with the independent derivative oracle.

| Seed | Width | Global sensitivity error bound | 1e-4 target | Calibration direction |
| --- | ---: | ---: | --- | --- |
| 401 | 6 | 0.000159 | INCONCLUSIVE | increase coupling |
| 401 | 12 | 3.29e-05 | PASS | increase coupling |
| 503 | 6 | 9.93e-05 | PASS | increase coupling |
| 503 | 12 | 6.39e-05 | PASS | increase coupling |
| 607 | 6 | 0.000119 | INCONCLUSIVE | increase coupling |
| 607 | 12 | 6.79e-05 | PASS | increase coupling |

Four of six cases meet the requested error target. All six sound enclosures certify a negative loss derivative for the synthetic calibration observation at lambda=32. The two missed accuracy targets remain INCONCLUSIVE. A separate SciPy collocation/finite-difference reference lies inside the sensitivity intervals; it is a numerical consistency check. The certificate concerns the physical solution derivative, not the fitted network optimizer's derivative. It certifies a local direction and gives no finite-step guarantee. This is a concrete verified scientific-ML use case; comparative speed and mathematical novelty are not established.


The separate fixed-window calibration test over lambda [32,33] returned **PASS**. This is a single predeclared seed-401/width-12 demonstration, not a selected best window. Its endpoint collocation reference is only a consistency check.


A uniform negative physical gradient proves a finite objective change between -0.0001118 and -3.285e-05 for the synthetic observation. The fundamental theorem of calculus supplies the finite-step guarantee.


### Global monotone-energy certificate

The separately proved energy bound removes the contraction smallness condition
for this clamped, nonnegative-coupling cubic problem. This is a post-discovery
capability study with fixed seed 401, width 12 and 100 readout steps; the coupling
values are fixed at 32, 128 and 512. Both derivative implementations independently
replay every whole-domain residual enclosure. No comparative training or speed
claim follows from this small fixture.

| Coupling | Oracle | Certified error upper bound | Requested 0.01 error | Original local contraction Z at that radius |
| ---: | --- | ---: | --- | ---: |
| 32 | core | 0.0034512 | PASS | 0.235 |
| 32 | independent | 0.0034512 | PASS | 0.235 |
| 128 | core | 0.0035064 | PASS | 0.7832 |
| 128 | independent | 0.0035064 | PASS | 0.7832 |
| 512 | core | 0.0067095 | PASS | 2.132 |
| 512 | independent | 0.0067095 | PASS | 2.132 |

A contraction factor at least one prevents the original local contraction proof; it does not imply that the PDE is ill-posed. The energy proof supplies a different sufficient condition. This supports a useful verified-solver capability, while its monotone-operator foundation is classical. Broader domains and negative couplings are outside the proof.


An independent midpoint enclosure proves the old contraction factor is at least 2.0378 even at zero error radius for the coupling-512 candidate. Consequently, refining its partition or reducing its requested radius cannot rescue that contraction gate. The new certificate's error bound is below 0.00671. This demonstrates an added certification capability for the same candidate, rather than a coarse-grid artifact.


### Supported transformations

The positive heat-series/Cole–Hopf fixture has denominator lower bound
1.625. Its analytic PDE residual,
ordinary-AD residual and direct transformed-spectral disagreement are all below
2e-17. These are numerical checks of the separately derived restricted identity;
they do not show faster nonlinear training. Bias-cluster compression is checked
through four spatial derivatives against uniform Taylor-remainder bounds and
80-digit divided-difference references. The fourth-derivative grid discrepancy
is 5.27e-09; the rigorous bound
is 1.22e-06. Independent
central-difference integral checks retain cancellation counterexamples. Formal
jet incompatibility, wrong constraints, correlated branches and finite-jet
ambiguity are retained as exact counterexamples.


### Boolean learning

The unknown-support study contains 36 bounded fits and nine exhaustive-search
controls. Both learned families start with the same routing, response function
and minibatch stream. Objectives are selected using validation hard balanced
accuracy; no hyperparameter is selected on test data.

| Target | Counting block | Unrestricted LUT | Exhaustive count-support search |
| --- | ---: | ---: | ---: |
| exactly_three | 100.00% | 99.23% | 100.00% |
| parity | 49.01% | 49.02% | 100.00% |
| random_lut | 66.21% | 71.28% | 57.89% |


The count block uses 79 trainable parameters versus 136 for the unrestricted
LUT, and seven versus 64 response entries. It learns exactly-three perfectly on
all discovery seeds, but does not clear the required learning margin over the
matched LUT across both structured tasks. Both gradient learners struggle on
parity; ordinary training-only GF(2) recovery achieves 100.00% mean
balanced accuracy there. The nonsymmetric target exposes the count model's
restriction. No claim against official DiffLogic or DiffLUT is supported.

Exact expected accuracy is a well-defined objective for the stochastic model;
it does not solve its nonconvex optimization or guarantee a better trained hard
model. Hard exports replay on every one of the 4096 assignments. The additional
dependency-aware block gives the correct probability for a reconvergent circuit;
naive marginal gate composition disagrees. Its explicit six-variable scope limit
prevents a claim of general tractable exact inference.

Directed soft-to-hard bounds were also checked for routing temperatures 0.3,
0.1 and 0.03, with response temperature fixed at one. They bound the exact real
stochastic model, while floating predictions receive separate consistency
checks. Bounds can remain loose even when observed hard accuracy is good.


Joint scaling of routing and response logits produced 36/36 directed certificates with global probability error at most 1e-3. Each passing bound was independently replayed and all 4096 floating threshold decisions matched the hard export. This preserves the learned circuit's decisions; it does not improve its label accuracy.


The constructive field prototype passes: a 12-coordinate derivative representation is corrected and integrated into one polynomial field. The full PDE/boundary system has rank 12; its boundary-amplitude derivative at the test point is 0.030000, while the uniquely selected solution has zero starting-point dependence to numerical tolerance. Exact symbolic PDE and boundary identities and paired Torch/JAX gradients are checked. This is a restricted, classical realization certificate.


### Constrained predictions and PFN feasibility

The true target satisfies `y3 = y1 + y2`. Models observe noisy training and
validation outputs; evaluation uses held-out latent targets. Values below are
coordinate-mean squared errors averaged over discovery seeds.

| Learner | Free prediction | Ordinary QR correction | Nullspace model | Wrong RHS 0.5 |
| --- | ---: | ---: | ---: | ---: |
| catboost | 0.127264 | 0.126699 | 0.126897 | 0.154477 |
| mlp | 0.021664 | 0.019429 | 0.020372 | 0.047207 |
| ridge | 0.049398 | 0.048747 | 0.048747 | 0.076525 |
| tabpfn | 0.033101 | 0.027920 | 0.036296 | 0.055698 |


The named output-adapter application met the plan's 10% discovery gate on at
least one declared learner. The entire fixed learner matrix was therefore
confirmed on five untouched seeds, without changing settings:

| Learner | Free MSE | QR-corrected MSE | Relative reduction |
| --- | ---: | ---: | ---: |
| catboost | 0.104329 | 0.104065 | 0.25% |
| mlp | 0.019450 | 0.017597 | 9.53% |
| ridge | 0.041382 | 0.041042 | 0.82% |
| tabpfn | 0.024173 | 0.021405 | 11.45% |


This is a useful application of supplied true constraints. Ordinary QR provides
the same correction and mathematical guarantee. TabPFN uses the public official
v2 regression checkpoint with library 9.2.0; it is an older comparator than the
current gated default. Separate scalar regressors do not supply a joint
covariance. The PFN generator/context prototype passes row-recombination tests;
PFN pretraining and inferred-constraint learning were not performed.

### Geometric model components

All learned values, normals and curvatures derive from one scalar field.
Ordinary AD agrees with the analytic derivatives. Quadratic fields receive the
same noisy off-surface supervision; PCA and KMeans are unsupervised context
controls. Mean clustering ARI is:


| Learned fields | Quadratic fields | KMeans |
| ---: | ---: | ---: |
| 0.645 | 0.850 | 0.000 |


Anomaly detection on held-out points:

| Score | Test false-positive rate | Normal-fault recall | Rare tangent recall |
| --- | ---: | ---: | ---: |
| omnibias | 5.08% | 100.00% | 2.41% |
| AD | 5.08% | 100.00% | 2.41% |
| quadratic | 5.21% | 100.00% | 5.99% |
| local_pca | 6.05% | 99.80% | 100.00% |


Thresholds use the 95th validation percentile. Rare tangent points remain on
the true manifold; their detection can reflect density or approximation error,
so it is not evidence of a globally correct geometric likelihood. The strong
quadratic control and independent derivative-algebra control prevent a claim
of general geometric-model superiority. Large-model training was outside this
component study.

### LPs, sensitivities and continuous dictionaries

The 576 LP queries include exact rational acceptance checks and numerical
HiGHS fallback. Times include the archive cost amortized over 64 queries.
Actual HiGHS simplex-basis warm starts provide the stronger solver control.

| Dimension | Exact proposal acceptance | Proposal + checks + fallback (ms) | SciPy HiGHS (ms) | Warm HiGHS (ms) | Warm HiGHS + same exact check (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 98.44% | 0.759 | 1.062 | 0.056 | 0.532 |
| 8 | 69.27% | 2.126 | 1.389 | 0.087 | 1.350 |
| 32 | 9.90% | 15.095 | 2.119 | 0.423 | 14.016 |


Solver-only timings provide numerical solutions. The last column also pays for
the same exact KKT checker as the proposal method, so the assurance work is
explicit. Exact-certificate success rates are retained in the machine-readable
summary; a failed exact check is never relabeled a proof.

Exact KKT witnesses replay without an LP optimizer. Independently computed
positive active-set neighborhoods are checked against finite differences;
maximum gradient discrepancy is 7.93e-13.
Zero margins refuse a positive-radius sensitivity claim. The CVXPYlayers and
existing omnibias comparison uses an explicitly regularized QP; it does not
claim a differentiable LP solution at an active-set transition. The mixed
value/first-derivative continuous-bias fixture closes its exact primal/dual gap
to zero; it is a supplied-witness observation problem, not equation discovery.

## 6. Assessment and continuation

| Candidate | Assessment |
| --- | --- |
| Calculus–geometry transfer identities, projection, counting expectations and capacity bounds | Established mathematics, proved here with explicit computational obligations |
| Field, physical-sensitivity and hardening certificates with independent replay | Useful engineering with explicit assumptions and completed correctness checks |
| Global nonnegative-coupling cubic stability certificate | Complete proof and checked implementation using established energy/monotone theory |
| Adaptive bookkeeping/order reuse | Measured engineering follow-up; attribution belongs to the algorithm and same-algebra controls |
| Unknown-support counting learner | Architectural candidate with an exact hard implementation; broad competitive advantage not established |
| True-constraint TabPFN adapter | Useful application, with fresh-seed confirmation and ordinary QR attribution |
| Geometric PFN priors and a large coherent geometry model | Feasibility components only; learning benefit remains unsupported |
| Equality alone turning LPs into a faster general linear solve | Unsupported as an algorithm claim; active-set discovery and strong solver controls remain necessary |
| A new mathematical principle or a general theory resolving optimization and generalization | Not established by this campaign |

Preserve the checked affine/jet primitives, certificate replayers, exact Boolean
oracles and this bounded experiment suite. Keep the broader prototypes as
research components whose status is explicit. No stable public API, license,
service or domain was changed by this campaign.

The strongest continuation is **verified physical parameter calibration** for
problems with a provable stability contract. This campaign now supplies a global
state bound, physical-sensitivity bounds and a fixed finite-step demonstration.
The next evidence gate is a matched-accuracy, end-to-end comparison with strong
collocation, validated-numerics and certified reduced-basis controls on a broader
parameter family, including enclosure and replay costs. A wider automatically
checkable stability/realization class would be a specific mathematical target.

**Constraint-aware prediction with independently justified constraints** is a
second practical direction. Test uncertain constraints and real multi-output
tasks before any PFN pretraining. The strongest unresolved transformation question
is when a computable certificate can preserve both the selected solution and
its sensitivities beyond the restricted families proved here.
For Boolean learning, the next useful question is robust unknown-support
recovery at larger input dimension and with noise, against algebraic,
combinatorial and current learned-logic controls. These are concrete research
questions; this pilot supplies no basis for promising their outcome.

## 7. Reproduction and validation

From the laboratory root, with its documented comparator environment and sibling
source paths available:

```bash
python -m pytest --import-mode=importlib -p no:cacheprovider calculus_geometry_research/test_checks.py -q
python -m calculus_geometry_research.run --stage all --output artifacts/reproduction_geometry_research
python -m calculus_geometry_research.run --replay --output artifacts/reproduction_geometry_research
python research_campaign_audit.py artifacts/reproduction_geometry_research --output artifacts/reproduction_geometry_research/audit.json
python research_adaptive_heap.py artifacts/reproduction_geometry_research/science.json --output artifacts/reproduction_geometry_research/adaptive_heap.json
python research_adaptive_heap.py artifacts/reproduction_geometry_research/science.json --output artifacts/reproduction_geometry_research/adaptive_heap.json --replay
python research_pde_sensitivity.py artifacts/reproduction_geometry_research/science.json --output artifacts/reproduction_geometry_research/sensitivity.json
python research_pde_sensitivity.py artifacts/reproduction_geometry_research/science.json --output artifacts/reproduction_geometry_research/sensitivity.json --replay
python research_pde_calibration_step.py artifacts/reproduction_geometry_research/sensitivity.json --output artifacts/reproduction_geometry_research/calibration_step.json
python research_pde_calibration_step.py artifacts/reproduction_geometry_research/sensitivity.json --output artifacts/reproduction_geometry_research/calibration_step.json --replay
python research_monotone_energy.py artifacts/reproduction_geometry_research/science.json --output artifacts/reproduction_geometry_research/monotone_energy.json
python research_monotone_energy.py artifacts/reproduction_geometry_research/science.json --output artifacts/reproduction_geometry_research/monotone_energy.json --replay
python research_energy_gate_review.py artifacts/reproduction_geometry_research/monotone_energy.json --output artifacts/reproduction_geometry_research/energy_gate_review.json
python research_energy_gate_review.py artifacts/reproduction_geometry_research/monotone_energy.json --output artifacts/reproduction_geometry_research/energy_gate_review.json --replay
python research_lp_certificate_controls.py artifacts/reproduction_geometry_research/optimization.json --output artifacts/reproduction_geometry_research/lp_certificate_controls.json
python research_lp_certificate_controls.py artifacts/reproduction_geometry_research/optimization.json --output artifacts/reproduction_geometry_research/lp_certificate_controls.json --replay
```

Eight campaign tests, six general follow-up tests, two physical-sensitivity tests and three monotone-energy tests pass, including integrated replay, missing-matrix refusal, hardening tampering and compatibility reconstruction.
The earlier focused run passed 55 checks covering campaign fixtures and exact,
Torch, JAX, batching, compilation, parameter and second-derivative regressions.
Those commands overlap and are not additive package-wide test counts. Complete
matrix replay, exact rational witnesses, tampered-certificate refusal, budget
exhaustion, saturation and counterexamples add the evidence reported above.
The first monotone-energy regression incorrectly demanded a bit-exact zero from
outward-rounded interval arithmetic. Its failure record is retained. The
corrected check compares the enclosure with an independent analytic zero; no
numerical implementation changed for that repair. No false certificate was
accepted. The derivations are written proofs; no formal proof-assistant build
is claimed.

The subsequent [verified physical-calibration study](verified-physical-calibration.md) extends the finite-step demonstration to certified parameter-set recovery, with assurance-matched spectral and reduced-basis controls.
