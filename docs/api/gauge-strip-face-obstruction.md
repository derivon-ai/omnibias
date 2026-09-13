# Local face vacua, boundary charges and strip frustration

For an isolated open SU(2) strip with \(n\ge2\) plaquettes and any finite
\(\kappa>0\), the exact decomposition into overlapping face Hamiltonians
has two properties:

1. A face with its boundary representations retained has a gap **at most**
   \(3\kappa/16\), witnessed by an exact normalized boundary-charge trial.
2. Neighboring local ground spaces have zero intersection. Subtracting
   their individual ground energies does not make the strip Hamiltonian
   frustration free.

These are statements about the specified local terms. The trial used for
the first is not globally gauge invariant. Neither statement is a gap
upper bound, or a gaplessness claim, for the full physical strip.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer.strip_face_obstruction import (
    su2_strip_face_obstruction,
    replay_su2_strip_face_obstruction_certificate,
)

row = su2_strip_face_obstruction(Q(1, 64), n_plaquettes=3)
assert row["status"] == "PASS"
assert row["local_boundary_flux_gap_upper"] == "3/1024"
assert [face["total_electric_weight"] for face in row["local_faces"]] == [
    "7/2", "3", "7/2",
]
assert row["strict_local_energy_frustration_verified_in_written_analysis"]
assert not row["global_physical_gap_upper_verified"]
assert not row["actual_ambient_conditional_gap_verified"]
assert replay_su2_strip_face_obstruction_certificate(row["certificate"])
```

## 1. Exact original-link decomposition

Let \(B_i,T_i\), \(1\le i\le n\), be bottom and top links directed
right, and \(V_j\), \(0\le j\le n\), be upward rungs. Set

\[
P_i=B_iV_iT_i^{-1}V_{i-1}^{-1},\qquad
A_i=2-\operatorname{Tr}P_i,
\quad
w_j=\begin{cases}1,&j=0,n,\\1/2,&1\le j\le n-1.\end{cases}
\]

With \(C_e=-\Delta_e\), fundamental Casimir \(3/4\), define

\[
h_i=\frac\kappa2\left(C_{B_i}+C_{T_i}
          +w_{i-1}C_{V_{i-1}}+w_iC_{V_i}\right)
       +\frac2\kappa A_i.
\tag{1}
\]

Every horizontal Casimir appears once; each internal rung appears twice
with weight \(1/2\), and each outer rung appears once with weight one.
Every potential appears once. Consequently

\[
H_n=\frac\kappa2\sum_{e=1}^{3n+1}C_e+
              \frac2\kappa\sum_i A_i=\sum_{i=1}^n h_i.
\tag{2}
\]

This is an identity of closed semibounded forms on the original tensor
product and on its globally gauge-invariant subspace. Each individual
term commutes with every original vertex gauge action. Different local
faces overlap in one rung precisely when they are neighbors.

The total electric weight of an interior face is three. An end face
has total weight \(7/2\). For example, the three-face strip has weights
\(7/2,3,7/2\), on the actual ten original links. The three-link path on
one side of a shared rung is not assigned the same weight as that rung.

The local Hilbert space may be taken to be full \(L^2(SU(2)^4)\), or
restricted only by gauge actions at vertices whose entire **original
strip star** lies in that face. For end faces, the two exterior-end
vertices are internal in this sense. An interior face has no such
internal vertices. Both endpoints of every shared rung are boundary
vertices in either neighboring face: their original stars contain a
horizontal link of the other face. No Gauss constraint at these boundary
vertices is imposed in the local source.

This is the sector needed to retain possible boundary charges during
gluing. It is different from declaring all four vertices neutral in
each face separately. The full constrained strip Hilbert space is not
the tensor product of those locally neutral Hilbert spaces.

## 2. The actual local ground state

All four local electric weights are strictly positive. The local scalar
operator is elliptic on a connected compact product, with smooth bounded
potential. Its positivity-improving semigroup gives a unique normalized
positive smooth ground state, denoted \(\phi_i\), of energy \(e_i\).
Every independent face-vertex gauge action commutes with \(h_i\).
Uniqueness and positivity therefore make \(\phi_i\) invariant under all
four such actions, even though the local state space allows boundary
charges. Gauge reduction of a single cycle implies

\[
\phi_i=\varphi_i(P_i),\qquad \varphi_i(gUg^{-1})=\varphi_i(U).
\tag{3}
\]

The same ground state is present in the internal-Gauss subspace and has
the same energy there. It is nonconstant for every finite positive
coupling: a constant function has zero electric energy density, whereas
\((2/\kappa)A_i\) is a nonconstant function and hence cannot be its
eigenvalue. On (3) each individual edge Casimir becomes the one-holonomy
Casimir. Its radial Hamiltonian has coefficient equal to the total
weight in Section 1; this fact is not a license to replace the full
boundary state space by radial functions.

Let \(s\) be a shared rung and let \(R\) be the complementary three-edge
path, with orientations chosen so the holonomy is \(sR\) up to inversion
and conjugation. Product Haar integration gives

\[
\int |\varphi_i(sR)|^2\,dH^{\otimes3}(R)=
\int |\varphi_i(U)|^2\,dH(U)=1.
\tag{4}
\]

Thus the **probability** marginal of the actual local vacuum density on
the shared rung is exactly normalized Haar. The complementary three-edge
probability marginal is also product Haar, by integrating that rung.
Equation (4) concerns this actual local ground, not the conditional law
of the full-strip vacuum after exterior links have been frozen.

## 3. A boundary-charge trial with exact shifted energy

Take \(f=\chi_{1/2}(s)=\operatorname{Tr}s\). The Haar identities are

\[
\int f\,dH=0,\qquad\int f^2\,dH=1,\qquad
\int|\nabla f|^2\,dH=\frac34.
\tag{5}
\]

By (4), the vector \(\phi_i f\) has norm one and is orthogonal to
\(\phi_i\). It obeys all required internal Gauss constraints: only the
two endpoints of \(s\) can change \(f\), and both are boundary vertices.
For instance a center transformation at one such endpoint changes its
sign, displaying its nontrivial boundary charge.

The exact ground-state form identity, on smooth functions and then by
closure, is

\[
\langle\phi_i f,(h_i-e_i)\phi_i f\rangle
=\frac\kappa2\int\sum_{e\in i}w_{ie}|\nabla_e f|^2
                          \phi_i^2\,dH^{\otimes4}.
\tag{6}
\]

Only the shared rung contributes. Its weight is \(1/2\). Equations
(4)–(6) therefore give the exact Rayleigh quotient

\[
\langle\phi_i f,(h_i-e_i)\phi_i f\rangle
=\frac\kappa2\frac12\frac34=\frac{3\kappa}{16}.
\tag{7}
\]

Min–max proves the announced upper bound in the full or internal-Gauss
local sector. There is no claim that this trial is an eigenfunction.
It is inadmissible as a scalar physical excitation of the full strip
without a matching exterior charge. Multiplying it by an exterior
object changes both the state and its normalization; (7) is not then a
global Rayleigh quotient.

The constant is sharp among bounds of the form \(c\kappa\) valid for
all positive couplings. Indeed the weighted electric operator has first
excited energy \(3\kappa/16\), while the constant Haar trial has
potential expectation \(4/\kappa\). Positivity of the potential and
min–max give

\[
\max\left(0,\frac{3\kappa}{16}-\frac4\kappa\right)
\le\operatorname{gap}(h_i)\le\frac{3\kappa}{16}.
\tag{8}
\]

These comparisons hold also after imposing the allowed internal Gauss
constraints, since both the ground and (7)'s trial remain admissible.
Dividing by \(\kappa\) and taking \(\kappa\to\infty\) proves the
sharpness assertion. Equation (8) does not assert a useful positive
weak-coupling lower bound.

## 4. Exact Schmidt obstruction to common local ground states

There is an important difference between the Haar probability marginals
in (4) and the quantum reduced state on the shared rung. The latter is
mixed. Here is an all-representation proof.

Expand the normalized central function in orthonormal SU(2) characters,

\[
\varphi_i(U)=\sum_{j\in\{0,1/2,1,\ldots\}}a_j\chi_j(U),
\qquad\sum_j|a_j|^2=1.
\]

In the product \(\varphi_i(sR)\), the normalized matrix-element bases
\(\sqrt{d_j}D^j_{ab}\), \(d_j=2j+1\), give exactly \(d_j^2\)
Schmidt singular values \(|a_j|/d_j\) for each \(j\). The private path
product \(R\) is Haar, and its other two variables are Haar spectators,
so this is also the Schmidt decomposition across the original
one-edge versus three-edge split. Inversion or conjugation conventions
only change the unitary bases. This argument is a complete Peter–Weyl
decomposition, not a finite-spin sample.

Since \(\varphi_i\) is positive, \(a_0>0\). Since it is nonconstant,
some \(a_j\ne0\) for \(j>0\). Its Schmidt rank exceeds one. Equivalently,
the shared-edge quantum reduced state has eigenvalues
\(|a_j|^2/d_j^2\), each repeated \(d_j^2\) times, and is not pure.
An elementary alternative uses (4): a positive factorization
\(\varphi_i(sR)=a(s)b(R)\), together with the Haar diagonal marginals
on both factors, would make both factors constant, a contradiction.

Now let two adjacent faces have private link sets \(A,C\) and shared
rung \(s\). A vector in the ground space of the first local term must
factor as \(\phi_{As}\otimes\eta_{C,\mathrm{rest}}\), because its
local ground is one-dimensional. Its reduced quantum state on \(sC\)
is therefore the product \(\rho_s\otimes\rho_C\). If it were also in
the second local ground space, this reduced state would have to be the
pure state \(|\phi_{sC}\rangle\langle\phi_{sC}|\). A product density
matrix is pure only if both factors are pure, contradicting the mixed
\(\rho_s\) just established. Hence

\[
\ker(h_i-e_i)\cap\ker(h_{i+1}-e_{i+1})=\{0\}.
\tag{9}
\]

This is an identity in the original full tensor-product space, with
identity factors on other links, and consequently remains true after
any global Gauss restriction. Compactness gives an attained global
ground energy. If \(E_0(H_n)=\sum_i e_i\), a global ground vector would
have zero expectation in every nonnegative term \(h_i-e_i\), and hence
belong to both kernels in (9). Therefore

\[
E_0(H_n)>\sum_{i=1}^n e_i\qquad(n\ge2,\ 0<\kappa<\infty).
\tag{10}
\]

No numerical lower bound for the difference in (10) is supplied. The
strict statement itself is exact and uses every representation.

## 5. The finite-size implication and scope

The standard Knabe-type local-gap criterion concerns frustration-free
local Hamiltonians. For example, the assumptions and local/global
inequality in [Gosset–Mozgunov, *Local gap threshold for frustration-free
spin systems*, Theorem 1](https://arxiv.org/pdf/1512.00088) require that
setting each local term to its minimum leaves a common ground space.
Equation (9) rules out precisely that construction for (1). A spectral
flattening to the local excited projectors does not restore their
missing common kernel.

The actual global ground-state transform does produce nonnegative terms
annihilating its vacuum. If \(\Psi_n>0\) is that vacuum, they have
first-order factors \(\nabla_e-\nabla_e\log\Psi_n\). Their coefficient
functions need not depend only on neighboring links. Similarly, actual
conditional heatbath projections are not known to commute at separated
faces. Thus frustration freeness of the transformed operator does not
by itself recover the local hypotheses of a finite-size criterion.
This is an identified missing theorem, not an assumed Markov property
of the equal-time vacuum law.

The certificate rebuilds the original graph, every electric share, every
allowed internal vertex and trial endpoint, the rational Rayleigh quotient
and all scope fields. Digest resealing cannot make an altered payload
canonical. `PASS` distinguishes the exact scalar/graph replay and these
written implications from a formal proof of positivity, Schmidt theory or
the full operator. No continuum, global physical gap, ambient conditional,
or theorem-prover flag is earned.
