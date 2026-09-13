# Static SU(2) sources: exact graph bounds

`omnibias.geometry.gauge.transfer.static_sources` constructs and replays
rational bounds on a **charged-minus-vacuum ground energy**. This differs
from a neutral excitation gap. All link spins are included in the analytic
argument; no representation cutoff defines the model.

The Hamiltonian framework and electric flux interpretation originate with
[Kogut and Susskind, Phys. Rev. D 11, 395 (1975)](https://doi.org/10.1103/PhysRevD.11.395).
The elementary variational arguments below are supplied in full. We do not
claim their mathematical novelty or a solution of bulk confinement.

## Model and boundary conditions

Let Γ be a finite connected graph, with no self-loops (parallel edges are
allowed). On the product Haar space of one SU(2) variable per edge, define

\[
\mathcal H= aH =\frac\kappa2\sum_e w_e C_e
 +\frac2\kappa\sum_p v_p(2-\operatorname{Tr} U_p),\qquad
 C_e=-\sum_{A=1}^3 X_{e,A}^2.
\]

Here κ>0, w_e>0, v_p≥0. The code requires exact rational parameters.
The trace is the unnormalized fundamental trace, and each p is a simple
closed graph cycle traversed once. Thus every magnetic summand is between
0 and 4 before multiplying by 2v_p/κ. The external source rest energies are
zero by convention and identical vacuum constants are subtracted.
There is no additional source self-energy subtraction. A renormalized
continuum static potential can use an additive convention in which this
nonnegative bare-energy bound does not apply.

The neutral space consists of gauge-invariant scalar functions. The charged
space consists of matrix-valued functions with covariance

\[
\Psi(g\cdot U)=g_s\Psi(U)g_t^{-1},\qquad s\ne t,
\]

using the Hilbert–Schmidt norm on the two color indices. Equivalently one
includes a fundamental source and its dual at the endpoints in Gauss's law.
Gauss's law is imposed at **every vertex, including boundary vertices**;
there is no dynamical fundamental matter and no unaccounted flux exit.
An orientation reversal on an edge replaces its matrix by its inverse.

Write \(\mathcal V_\Gamma(s,t)=E_c-E_v\) for the two ground energies in
this dimensionless convention. A supplied graph has no inferred spatial
dimension or continuum interpretation.

## The electric identity

In a Peter–Weyl spin network, C_e has eigenvalue j_e(j_e+1). Acting by the
central element −I at a vertex shows that the number of incident
half-integer edges is even, except at s and t where it is odd. Consequently
the half-integer edge subgraph contains a path connecting s and t: in each
connected component the number of odd-degree vertices is even.

Every such edge has Casimir at least 3/4. For positive weights,

\[
\sum_e w_e j_e(j_e+1)\ge\frac34d_w(s,t),
\]

where d_w is the weighted graph distance. This is an operator form lower
bound on the complete charged space because spin networks form a complete
electric eigenbasis. Center parity is only a necessary condition for
general admissibility, but a simple fundamental path attains the bound:
put j=1/2 on its edges and j=0 elsewhere. Every internal path vertex and
each source has a singlet intertwiner. The vacuum has all spins zero.
Hence, when all magnetic weights vanish,

\[
\boxed{\mathcal V_\Gamma(s,t)=\frac{3\kappa}{8}d_w(s,t).}
\]

The shortest-path computation uses rational arithmetic, and its path and
distance labels are included in the certificate. Replaying reconstructs
the whole model and the proof inputs, rather than trusting a hash alone.

## Exact cancellation of vacuum energy

The scalar operator on the full product Haar space is elliptic on a compact
connected manifold with smooth bounded real potential. Its positivity
improving heat semigroup gives a unique strictly positive normalized scalar
groundstate ψ₀. Gauge transformations commute with the operator, and
uniqueness makes ψ₀ gauge invariant; its energy is therefore E_v.

For any smooth matrix F, integration by parts and the groundstate equation
give the groundstate transform

\[
\langle\psi_0F,(\mathcal H-E_v)\psi_0F\rangle
 =\frac\kappa2\int\psi_0^2
       \sum_{e,A}w_e\|X_{e,A}F\|_{\rm HS}^2\,dU.
\]

This extends to the quadratic-form domain. Since ψ₀ is smooth and strictly
positive on a fixed compact graph, multiplication and division by ψ₀
preserve that domain. The identity gives \(\mathcal V_\Gamma\ge0\).

For an oriented simple path γ from s to t, set F=U_γ/√2. It has unit norm
at every configuration. Inserting any generator on a used edge gives

\[
\sum_A\|X_{e,A}F\|_{\rm HS}^2=3/4;
\]

unused edges contribute zero. Inverse traversal gives the same Casimir.
The variational principle therefore gives

\[
\boxed{0\le\mathcal V_\Gamma(s,t)\le\frac{3\kappa}{8}d_w(s,t).}
\]

This upper bound is independent of the surrounding volume and of the
magnetic strength. No approximation to ψ₀ is used in the proof or code.
It holds for any smooth real scalar gauge-invariant potential in this
finite-graph Hamiltonian. It is an upper bound, not positive string tension.

## A lower bound from separating bridges

Let B_st be the graph bridges whose removal separates s and t, and put
\(d_{\rm br}=\sum_{e\in B_{st}}w_e\). A simple cycle cannot use a bridge.
Removing all bridges therefore splits the edge configuration variables
into independent magnetic blocks and free bridge rotors. Each block's full
scalar groundstate is gauge invariant by the positivity argument above.
Their product, constant on bridges, lies in the neutral space and attains
the sum of the block ground energies. Thus this sum equals E_v.

On each block, its Hamiltonian minus its scalar ground energy is
nonnegative, also when tensored with external color factors. Each
source-separating bridge must have half-integer spin: multiply the center
Gauss constraints on one component of its deletion. It follows that

\[
\boxed{\frac{3\kappa}{8}d_{\rm br}
 \le\mathcal V_\Gamma(s,t)\le\frac{3\kappa}{8}d_w(s,t).}
\]

This lower bound is valid for the stated simple-cycle potential; arbitrary
nonlocal potentials need not split over the bridge decomposition. The
code rejects nonsimple plaquettes.

If a shortest source path consists entirely of separating bridges, the
bounds coincide. In particular a chain with arbitrary interacting loop
blocks attached at single chain vertices has **exactly** the pure electric
source energy, at every κ>0. A block entered and exited at different
vertices can contribute additional energy; its internal distance cannot
be counted as bridge distance. Ordinary multidimensional cubic boxes
have no separating bridges. This theorem does not give their string tension.

## Finite-volume bound and refusal to extrapolate

Positivity of the magnetic potential gives \(E_c\ge3\kappa d_w/8\).
The neutral constant Haar trial has zero electric energy and zero mean
fundamental trace for every simple plaquette. Consequently
\(E_v\le4\sum_pv_p/\kappa\). The returned enclosure is

\[
\max\left\{\frac{3\kappa}{8}d_{\rm br},
             \frac{3\kappa}{8}d_w-\frac4\kappa\sum_pv_p\right\}
\le\mathcal V_\Gamma(s,t)\le\frac{3\kappa}{8}d_w.
\]

The Haar term grows with total plaquette weight. On cubic boxes this can
destroy a positive lower bound when the volume increases. Such a failed
positivity gate is retained. `status=PASS` means the displayed interval is
sound; its lower endpoint may be zero. `string_tension_claim`,
`continuum_claim` and `yang_mills_claim` remain false. The infinite-spin and
groundstate arguments are written proofs; no Lean or Mathlib tier is earned.

## The next mathematical obligation

Set \(d\mu_0=\psi_0^2dU\). The exact transformed problem is

\[
\mathcal V_\Gamma(s,t)=\frac\kappa2
 \inf_{F\ne0,\ F\text{ source-covariant}}
 \frac{\int\sum_{e,A}w_e\|X_{e,A}F\|^2d\mu_0}
      {\int\|F\|^2d\mu_0}.
\]

A confinement proof needs a positive lower bound for this quotient
proportional to source separation, uniformly over actual bulk lattices.
The unknown interacting vacuum measure remains in μ₀; the transform has
not bounded its correlations or supplied a finite-block implication.
A global maximum/minimum density comparison that deteriorates with volume
does not meet the requirement. Neither a neutral spectral gap nor the
bridge identity supplies this missing estimate.

If a dimensionless lower bound is τ(a)r at lattice distance r=R/a, its
physical string tension is τ(a)/a². Continuum confinement additionally
needs a controlled scale trajectory and limit with positive physical
string tension; the present bounds make no such assertion.
In particular the bare path upper bound is ultraviolet divergent at fixed
physical separation along usual weak-coupling trajectories. It does not
bound a renormalized continuum potential uniformly in the cutoff.

## Use

```python
from omnibias.geometry.gauge.transfer.static_sources import (
    su2_static_source_bounds, replay_su2_static_source_certificate,
)

result = su2_static_source_bounds(
    4, [(0, 1), (1, 2), (2, 3), (3, 0)], 0, 2,
    kappa=4, plaquettes=[(1, 2, 3, 4)],
)
assert result["witness"]["static_energy_enclosure"] == ["2", "3"]
assert replay_su2_static_source_certificate(result["certificate"])
assert not result["string_tension_claim"]
```

The companion `ensemble-laws` report evaluates decorated chains and actual
open cubic boxes with every elementary plaquette. It keeps exact bounds,
full graph data, signed plaquettes, path witnesses and every replay result.
The charged-cycle spectral module provides much tighter bounds on one
finite cycle by directly enclosing both all-spin ground energies.
