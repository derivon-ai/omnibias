# Actual vacuum conditional gaps in an arbitrary finite exterior

`su2_ambient_conditional_gap` certifies a strictly positive conditional
Poincaré bound for the actual ground-state density of a finite SU\(2\)
Wilson-word Hamiltonian, at every positive coupling and every fixed exterior.
Its constants use only the number of block links and touching interaction
terms. They do not grow when disjoint exterior links or plaquettes are added.
The bound is small at weak coupling; it is not a bulk spectral-gap theorem.

The implementation is
`omnibias.geometry.gauge.transfer.ambient_conditional`. The analytic proof
below is a written proof; canonical rational replay does not formalize its
semigroup, manifold or spectral arguments.

```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer.ambient_conditional import (
    replay_su2_ambient_conditional_certificate,
    su2_ambient_conditional_gap,
)

result = su2_ambient_conditional_gap(
    Fraction(1, 64),
    block_link_ids=[1],
    plaquette_words=[
        [1, 2, -3, -4], [-1, 5, 6, -7],
        [8, 9, -10, -1], [1, -11, 12, 13],
    ],
    n_links=13,
)
assert result["status"] == "PASS"
assert result["arithmetic"]["selected_heat_time_ratio"] == "4/7"
assert result["quantum_gap_lower"]["prefactor"] == "3/4096"
assert result["quantum_gap_lower"]["negative_exponent"] == "63744/7"
assert result["factored_dyadic_quantum_gap_lower"]["negative_integer_exponent"] == 13663
assert replay_su2_ambient_conditional_certificate(result["certificate"])
assert not result["global_hamiltonian_gap_verified"]
```

## 1. Defined operator and actual conditional measure

On \(G^E\), \(G=\mathrm{SU}\(2\)\), use product normalized Haar measure and
the bi-invariant metric with fundamental Casimir \(3/4\). Thus

\[
C_e=-\Delta_e,\qquad
H=\frac{\kappa}{2}\sum_{e\in E}C_e+
  \frac{2}{\kappa}\sum_{p=1}^{P}\bigl(2-\operatorname{Tr}U_{w_p}\bigr),
\qquad \kappa>0. \tag{1}
\]

Each input word has four signed link IDs. Negative IDs mean inverse links;
repeated IDs, backtracking and duplicate words are allowed. Duplicates are
separate summands in \(1\). Such words always give smooth real potentials in
the interval \([0,8/\kappa]\). No vertex data are supplied, so the API does
not verify that these words are closed graph cycles. Its unconditional
statement is on the full scalar Hilbert space. If closed-loop gauge
invariance is established independently, the conditional inequality also
holds on the appropriate internal-Gauss subspace, with arbitrary boundary
representations retained.

The connected compact product manifold and strictly elliptic kinetic term
give compact resolvent. Its heat semigroup with the smooth bounded real
potential is positivity improving. Consequently the lowest eigenvalue is
simple and has a smooth strictly positive real eigenfunction \(\psi\).
When a compact gauge group acts and preserves \(1\), uniqueness makes
\(\psi\) gauge invariant. Neither an input vacuum approximation nor an
input spectral gap is used here.

Choose nonempty \(X\subset E\), let \(Y=E\setminus X\), and set

\[
b=|X|,\qquad
p=\#\{j:w_j\text{ contains a link of }X\},\qquad
d\mu_X^y(x)=\frac{\psi(x,y)^2\,dx}{\int_{G^X}\psi(z,y)^2\,dz}.
\tag{2}
\]

The count in \(2\) uses the original word, without cancellation or
deduplication. This can overestimate the interactions that truly depend
on \(X\), but cannot omit one. Every \(y\in G^Y\) has a well-defined smooth
positive conditional density.

**Theorem.** For every \(s>0\), every exterior \(y\), and every complex
\(f\in H^1(G^X)\),

\[
\frac{\kappa}{2}\int|\nabla_Xf|^2d\mu_X^y
\ \ge\ \gamma_X\operatorname{Var}_{\mu_X^y}(f),\qquad
\gamma_X\ge\frac{3\kappa}{8}\,2^{-3b}
 \exp\left[-\frac{32sp}{\kappa^2}-\frac{1936b}{49s}\right]>0.
\tag{3}
\]

If \(p=0\), the conditional measure is exactly Haar and its exact quantum
gap is \(3\kappa/8\), independent of \(b\ge1\). The Poincaré gap without
the kinetic coefficient is \(2\gamma_X/\kappa\).

The operator corresponding to the left side of (3) is
\(-\kappa(\Delta_X+2\nabla_X\log\psi\cdot\nabla_X)/2\), with constants
as its vacuum. It is the ground-state-transformed **actual conditional**
operator. It is not the frozen bare Wilson operator with a substituted
isolated ground energy.

## 2. Local Feynman–Kac comparison

Put \(\alpha=\kappa/2\) and split exactly

\[
H=H_0+V_X,\qquad H_0=\alpha C_X+H_Y,\qquad
H_Y=\alpha C_Y+V_Y,\qquad 0\le V_X\le M=8p/\kappa.
\tag{4}
\]

Here \(V_X\) is the sum of every touching word term, while \(V_Y\) contains
the rest and depends only on \(y\). For every \(T>0\) and nonnegative \(f\),

\[
e^{-MT}e^{-TH_0}f\ \le\ e^{-TH}f\ \le\ e^{-TH_0}f.
\tag{5}
\]

These inequalities are in the cone of nonnegative functions. They are
not an appeal to monotonicity of the operator exponential in
self-adjoint operator order. To prove (5), the Feynman–Kac integrand gains
the factor \(\exp[-\int_0^T V_X(X_t,Y_t)dt]\in[e^{-MT},1]\).
Equivalently, each bounded-potential Trotter multiplier is between
\(e^{-MT/n}\) and \(1\); positivity of the other semigroup factor gives
the bounds after every product, and the strong limit preserves the cone.
The bounded-potential Feynman–Kac formula and this Trotter passage are
given in Bär–Pfäffle, §6, Theorems 6.1–6.2. Their sign convention is
converted here to \(e^{-TH}\).
[Primary source](https://arxiv.org/abs/1108.5082).

Let \(k_s^X\) be the product heat kernel for \(e^{-sC_X}\), with
\(s=\alpha T\). Apply (5) to the actual eigenfunction and define

\[
F_y(z)=\bigl(e^{-TH_Y}\psi(z,\cdot)\bigr)(y)>0.
\]

Since \(H_Y\) acts only in the exterior, this is the **same positive
function of \(z\)** for every starting \(x\). Using
\(e^{-TH}\psi=e^{-TE_0}\psi\), (5) becomes

\[
e^{TE_0-MT}\int k_s^X(x,z)F_y(z)dz
\le\psi(x,y)\le
e^{TE_0}\int k_s^X(x,z)F_y(z)dz.
\tag{6}
\]

The intermediate comparisons hold almost everywhere from the semigroup
argument; smoothness makes (6) valid everywhere. The empty-exterior case
uses the identity operator for \(e^{-TH_Y}\).

Write \(K_s\) for the central single-link heat kernel, and
\(R_s=\max_G K_s/\min_G K_s\). Its strict positivity and compactness give
finite \(R_s\). Comparing (6) at two block points cancels \(E_0\), the
exterior semigroup and all its possible volume dependence:

\[
\frac{\max_x\psi(x,y)}{\min_x\psi(x,y)}
 \le e^{MT}R_s^b. \tag{7}
\]

No derivative of the exterior potential or of a conditional vacuum is
needed. The common \(F_y\), rather than unrelated upper and lower
exterior bounds, is what makes (7) local in the stated sense.

## 3. SU\(2\) heat-kernel ratio and density comparison

The Casimir convention in \(1\) identifies the group with the round
three-sphere of radius \(2\), whose diameter is \(2\pi\) and Ricci
curvature is nonnegative. The Li–Yau differential estimate for positive
heat solutions is

\[
|\nabla\log u|^2-\partial_t\log u\le\frac{3}{2t}.
\tag{8}
\]

Integrate it on a constant-speed minimizing geodesic between times
\(s/2\) and \(s\). Completing the gradient square gives

\[
u(s/2,I)\le 2^{3/2}
 \exp\left[\frac{d(I,U)^2}{2s}\right]u(s,U).
\tag{9}
\]

This is the nonnegative-Ricci heat Harnack estimate of Li–Yau,
*On the parabolic kernel of the Schrödinger operator* (1986), Theorem 1.1
and the path-integration argument leading to Theorem 2.3 (take zero Ricci
lower-bound loss and let its parameter decrease to one).
[Primary source](https://doi.org/10.1007/BF02399203);
[full paper, pp. 157–158 and 167–169](https://scispace.com/pdf/on-the-parabolic-kernel-of-the-schrodinger-operator-26njhmohgk.pdf).

The convolution identity and Cauchy–Schwarz imply
\(K_s(U)\le K_s(I)\). Also
\(K_s(I)=\sum_{n\ge0}(n+1)^2e^{-sn(n+2)/4}\) decreases in \(s\).
Apply (9) to the heat kernel, so that

\[
R_s\le2^{3/2}\exp(2\pi^2/s). \tag{10}
\]

If \(w=d\mu_X^y/dx\), (7) gives
\(\max w/\min w\le e^{2MT}R_s^{2b}\). For completeness, normalized
Haar tensorizes with scalar gap \(3/4\), and

\[
\operatorname{Var}_{\mu_X^y}(f)
\le (\max w)\operatorname{Var}_{dx}(f)
\le\frac43\frac{\max w}{\min w}\int|\nabla_Xf|^2d\mu_X^y.
\tag{11}
\]

The first inequality follows by choosing the Haar mean in the infimum
over constants defining weighted variance. The second uses the Haar
gap and the minimum of \(w\). Smooth functions are dense in the weighted
form domain because \(w\) is smooth, positive and bounded above and
below on each compact fiber. Thus the estimate covers all complex
Sobolev functions, without a spin cutoff.

Equations (7), (10) and (11) prove

\[
\gamma_X\ge\frac{3\kappa}{8}2^{-3b}
 \exp\left[-\frac{32sp}{\kappa^2}-\frac{4\pi^2b}{s}\right].
\tag{12}
\]

Use \(\pi<22/7\) to get (3). For \(p=0\), exact tensor factorization in
(4) makes \(\psi\) constant in \(x\), yielding the sharper exact Haar
claim instead of (12).

## 4. Exact time selection and positive representations

For \(s=q\kappa\), the exponential loss in (3) is

\[
\Omega(q)=\frac{32qp+1936b/(49q)}{\kappa}.
\tag{13}
\]

The default searches only
\(q\in\{1/4,1/2,4/7,1,2,4\}\), with the first minimum in this ordered
list selected on a tie. Every candidate is valid; finite-grid optimality
is the only optimization claim. For \(b=1,p=4\), \(q=4/7\) gives
\(\Omega=996/(7\kappa)\). An explicit positive rational `heat_time`
overrides this selection. Integer, `Fraction`, and rational-string inputs
are accepted; binary floats and Booleans are refused. The exact-Haar
branch still validates an explicitly supplied heat time before ignoring it.

The primary bound records a rational prefactor and rational \(\Omega\)
in the expression \(P\exp(-\Omega)\). No exponential is evaluated or
converted to a very large rational denominator. An optional second
representation is entirely factored:

\[
N=\lceil3\Omega/2\rceil,\qquad
\gamma_X\ge\frac{3\kappa}{8}\,2^{-(3b+N)}.
\tag{14}
\]

Indeed,

\[
\log2-\frac23
=\int_1^2\left(\frac1x-\frac43+\frac{4x}{9}\right)dx
=\int_1^2\frac{(2x-3)^2}{9x}\,dx\ge0,
\]

so \(N\log2\ge\Omega\). The integer \(3b+N\) is stored directly, without
constructing the denominator \(2^{3b+N}\). For the Haar branch all loss
exponents are zero. Both representations stay positive even when an
ordinary floating-point exponential would underflow.

## 5. Scope and remaining interaction problem

Canonical replay reconstructs the exact operator words, touched-word
count, time selection, all bounds and all flags. Resealing a changed
bound or a strengthened scope does not make replay pass. The record
earns the written-analytic actual conditional theorem, while
`theorem_prover_verified`, `mathlib_verified` and
`analytic_proof_formally_verified` remain false.

For each fixed pair \(b,p\), (3) is uniform over every finite exterior,
every exterior configuration and every positive coupling. Enlarging the
block, increasing its touching count, or sending \(\kappa\) to zero can
make this lower bound arbitrarily small. It is not uniform in those
parameters. A genuine cubic family with independently verified incidence
at most four has \(p\le4b\), but raw word IDs alone do not earn that
geometric premise.

The theorem supplies one block conditional inequality. If \(Y\ne\varnothing\),
any nonconstant function of \(y\) has zero block energy. Thus no global
Poincaré or Hamiltonian gap follows from this inequality alone: a
covering argument controlling correlations or conditional projections
is still required. There is no continuum, weak-coupling bulk-gap or
Yang–Mills reconstruction claim.
