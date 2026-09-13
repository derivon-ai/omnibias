# Actual cubic Wilson conditional-block bounds

12 September 2026. This note transfers an earned local Fourier bound on
the actual Wilson vacuum to every exterior conditional of a finite block.
The estimates involve the block size, local plaquette incidence and the
local correction norm; they contain no factor equal to the total lattice
volume. A separate conditional Schur estimate can remove even the block
size from the final gap. No isolated two-plaquette gap is substituted for
a conditional of the coupled system.

The bounded-density comparison and conditional Poincare mechanisms are
classical. Primary context is
[Holley–Stroock (1987)](https://link.springer.com/article/10.1007/BF01011161)
and [Menz, Theorems 2.3 and 2.7](https://arxiv.org/abs/1402.5160).
The elementary comparison is proved below; the compact-group conditional
Schur argument, including its exact-one-form restriction, is proved in
[the compact-group conditional Schur proof](gauge-marginal-majorant.md).
This is a finite-volume, all-spin theorem, not a continuum or novelty claim.


```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer import (
    su2_wilson_polar_vacuum,
    su2_wilson_conditional_block,
    replay_su2_wilson_conditional_block_certificate,
)

source = su2_wilson_polar_vacuum(
    15, family="cubic", correction_radius=Fraction(1, 10),
)
result = su2_wilson_conditional_block(
    source["certificate"], block_size=7, exponent_steps=4,
)
assert result["status"] == "PASS"
assert result["uniform_over_all_exteriors_verified"]
assert result["conditional_poincare_lower"] == "367/1350"
assert result["conditional_energy_units_lower"] == "367/180"
assert replay_su2_wilson_conditional_block_certificate(result["certificate"])
assert not result["frozen_wilson_hamiltonian_claim"]
assert not result["continuum_claim"]
assert not result["yang_mills_mass_gap_claim"]
```

```text
su2_wilson_conditional_block(source_certificate, *, block_size, exponent_steps=4)
replay_su2_wilson_conditional_block_certificate(certificate) -> bool
```

The source must be a canonical cubic certificate of one of these types:

- `su2_wilson_residual_vacuum_v1`
- `su2_wilson_linear_vacuum_v1`
- `su2_wilson_polar_vacuum_v1`

The radius, coupling, actual groundstate and graph-family scope are
inherited by full replay. Unsupported sources, forged certificates and
strip charts raise `ValueError`. A canonical source whose nonlinear
construction failed yields `INCONCLUSIVE`. An earned actual source can
be consumed even when its global sufficient spectral tests failed.

`block_size` is a strictly positive integer upper bound. The certificate
covers every nonempty original-edge subset of at most that size, in every
source-family box, and every exterior configuration. This is a quantified
family implication; it is not a membership test of a caller-supplied graph.

The output keeps the following fields separate:

| Field | Meaning |
|---|---|
| `conditional_poincare_lower` | Best diffusion floor for the specified block-size bound. |
| `conditional_energy_units_lower` | That floor multiplied by the microscopic factor `kappa/2`. |
| `witness.arithmetic.all_cardinalities_conditional_poincare_lower` | Separate block-size-independent floor, or `None`. |
| `actual_block_conditionals_verified` | The actual nonlinear source and every-exterior comparison passed. |
| `all_cardinalities_conditional_gap_verified` | A positive original-coordinate curvature or Schur route passed. |
| `spatial_correction_tail_verified` | The actual source has `decay_base>1`; the support-tail formula in the witness applies. |

The energy-scaled conditional number does not identify a new Hamiltonian
with frozen Wilson terms. In particular, a larger one-edge conditional
bound is not a lower bound for the whole-box spectrum. The separate
cardinality-independent field carries the latter uniform diffusion
implication. The complete analytic proof follows.

The actual new source is documented in
[the complete polar Wilson source](gauge-wilson-polar-source.md).
Its earlier variants retain their own certificate schemas and budgets.

## 1. Actual source and original coordinates

Let the electric graph be a finite open three-dimensional rectangular
cubic box. Orient each edge in its positive coordinate direction. Use
unit electric weights, elementary square Wilson terms, and Gauss
constraints at every vertex. Write
\[
 aH=\frac\kappa2 C+\frac2\kappa\sum_p(2-\chi_p),\qquad
 C=\sum_e(-\Delta_e),\qquad g=4/\kappa^2,\qquad
 S_*=\frac g3\sum_p\chi_p.
 \tag{1}
\]
The required input is a complete, canonically replayed actual-vacuum
construction
\[
 \psi_0=Z^{-1/2}e^{S_*+U},\qquad
 d\mu=\psi_0^2\,dH,\qquad \mathcal N_b(U)\le r,
 \quad b\ge1,\quad r<\infty.                           \tag{2}
\]
In particular, (2) is not an ansatz or the definition of a different
Hamiltonian. The all-spin fixed point must have identified the positive
groundstate of (1). The earlier exact residual and linear source and the
new polar-fusion source can provide this premise when their own nonlinear
gates pass.

For a product representation label \(\mathbf j\), let
\[
 E_{\mathbf j}=\sum_ej_e(j_e+1),\quad
 X_{\mathbf j}=\{e:j_e>0\},\quad
 w_{\mathbf j}=b^{\operatorname{diam}X_{\mathbf j}},\qquad
 U=\sum_{\mathbf j\ne0}\operatorname{Tr}
        [F_{\mathbf j}\rho_{\mathbf j}].
\]
Distances are ambient line-graph distances on the original electric
graph. The local norm is
\[
 \mathcal N_b(U)=\max_e\sum_{\mathbf j\ne0}
          j_eE_{\mathbf j}w_{\mathbf j}\|F_{\mathbf j}\|_1.
 \tag{3}
\]
The trace norm in (3) is the original-edge Fourier coefficient norm.
No invariance under partial transpose or an edge-orientation change is
assumed.

Gauge invariance at every vertex and girth four imply
\(E_{\mathbf j}\ge3\) for every nonzero coefficient. Indeed its active
support cannot have a vertex of degree one, so it contains a cycle with
at least four active edges; every active edge has Casimir at least
\(3/4\). This statement concerns the original invariant coefficient
before conditioning. The conditional itself need not be gauge invariant.

The proof below also works on any finite simple invariant SU(2) graph
with an earned nonconstant electric floor \(E_{\min}>0\), with
\(3\) replaced by \(E_{\min}\).

## 2. Block-local correction oscillation

Let \(B\) be a nonempty set of original edges and fix arbitrary exterior
values \(z\in SU(2)^{B^c}\). Terms with \(X_{\mathbf j}\cap B=\varnothing\)
are constant in the free block variables and cancel from differences.
For every other coefficient, unitarity gives
\[
 |\operatorname{Tr}[F_{\mathbf j}\rho_{\mathbf j}(x_B,z)]|
 \le\|F_{\mathbf j}\|_1.
\]
Consequently
\[
 \operatorname{osc}_{x_B}(2U(x_B,z))
 \le4\sum_{\mathbf j:X_{\mathbf j}\cap B\ne\varnothing}
                         \|F_{\mathbf j}\|_1.          \tag{4}
\]
For any specified edge \(e\), \(j_e\ge1/2\) on its active coefficients.
Thus, retaining the support weight,
\[
 \sum_{\mathbf j:e\in X_{\mathbf j}}
       w_{\mathbf j}\|F_{\mathbf j}\|_1
 \le\frac{2}{E_{\min}}\mathcal N_{b,e}(U)
 \le\frac{2r}{E_{\min}}.                               \tag{5}
\]
A union bound over the block gives the completely explicit result
\[
 \boxed{\quad
 \operatorname{osc}_{B}(2U\mid z)
 \le\frac{8|B|r}{E_{\min}}
 =\frac{8|B|r}{3}\quad\hbox{on cubic boxes, for every }z.
 \quad}                                               \tag{6}
\]
Each coefficient is charged only to edges in the specified block.
No sum over the exterior volume appears. Repeated counting of a
coefficient touching several block edges makes this an upper bound.

If additional exact coefficient information is available, the union
bound can be replaced by a local nonnegative dual. Any weights
\(\lambda_e\ge0\), \(e\in B\), satisfying
\[
 \sum_{e\in B}\lambda_ej_eE_{\mathbf j}w_{\mathbf j}\ge1
 \quad\hbox{for every allowed coefficient touching }B
\]
give the right side of (4) at most \(4r\sum_{e\in B}\lambda_e\).
The generic choice \(\lambda_e=2/E_{\min}\) proves (6).
The theta-specific \(13/72\) dual does not satisfy this different
all-graph constraint merely because the block resembles a theta.

There is also a genuine spatial tail statement. The portion of \(U\)
whose support touches \(B\) and has diameter at least \(R\ge0\) has
\[
 \operatorname{osc}_{B}(2U_{\ge R}\mid z)
       \le\frac{8|B|r}{E_{\min}}b^{-R}.                 \tag{7}
\]
In particular it controls coefficients reaching farther than distance
\(R\) from \(B\). Equation (7) can bound a finite-neighborhood conditional
approximation once its retained coefficients are known and verified.
At \(b=1\) it asserts no spatial decay.

## 3. The true conditional and an explicit reference comparison

The actual block conditional is
\[
 d\mu_B^z=Z_B(z)^{-1}
       e^{2S_*(x_B,z)+2U(x_B,z)}\,dH_B.                \tag{8}
\]
Its same-exterior reference is
\[
 d\nu_B^z=Z_{*,B}(z)^{-1}e^{2S_*(x_B,z)}\,dH_B.       \tag{9}
\]
It includes every plaquette touching \(B\), including boundary plaquettes
with some fixed exterior links. It is generally interacting and does
not equal the isolated-block vacuum or a product of such vacua.

Suppose a bound \(\gamma_{*,B}>0\) has been proved for every reference
conditional (9), with the original block metric
\(\Gamma_B=\sum_{e\in B}|\nabla_e\,\cdot\,|^2\). Then
\[
 \boxed{\quad
 \gamma_B(\mu_B^z)
 \ge\gamma_{*,B}\exp\!\left(-\frac{8|B|r}{3}\right),
 \quad\hbox{uniformly in }z.
 \quad}                                               \tag{10}
\]
To prove this without confusing means, if \(d\mu=w\,d\nu\) and
\(m\le w\le M\), then
\[
 \operatorname{Var}_\mu f
 =\inf_c\int|f-c|^2w\,d\nu
 \le M\operatorname{Var}_\nu f
 \le\frac{M}{m\gamma_\nu}\int\Gamma_B(f)\,d\mu.
\]
For (8)–(9), normalization cancels from \(M/m\) and (6) applies.
There is no assumption that conditional normalizations or means agree.

A reference-free sufficient floor follows directly from product Haar.
Let \(p_B\) be the number of elementary plaquettes touching \(B\).
Each contributes at most \(8g/3\) to the oscillation of \(2S_*\), since
\(\chi_p\in[-2,2]\). Product Haar on any nonempty edge block has gap
\(3/4\), independent of the block cardinality. Thus
\[
 \boxed{\quad
 \gamma_B(\mu_B^z)\ge
 \frac34\exp\!\left[-\frac83(gp_B+|B|r)\right]
 \ge\frac34\exp\!\left[-\frac{8|B|}{3}(d_*g+r)\right],
 \quad}                                               \tag{11}
\]
where \(d_*=4\) for cubic boxes. Boundary plaquettes are counted, and
\(p_B\le d_*|B|\) is valid even at finite-volume boundaries.
This floor is positive for every finite block and every finite earned
radius, including when a curvature bound is negative.

For rational \(\Omega\ge0\), take any integer \(m>\Omega\).
Then \(e^{-\Omega}\ge(1-\Omega/m)^m>0\). Choosing
\(m=\max(m_{\rm requested},\lfloor\Omega\rfloor+1)\) ensures the domain
and yields a pure rational certificate of (11).

Smoothness of (2) follows from the original elliptic eigenfunction
equation. All estimates are uniform pointwise in exterior configurations
before integration. Smooth approximation extends the conditional
Poincare bounds to the full block Sobolev form domains. No quotient
chart or invariant-only conditional function class is used.

## 4. Cardinality-independent conditional Schur route

For one original edge the preceding result gives
\[
 \gamma_1\ge\frac34e^{-\Omega_1},\qquad
 \Omega_1=\frac83(d_*g+r).                              \tag{12}
\]
For distinct original edges, let \(c_{ef}\) uniformly bound
\(\|\operatorname{Hess}_{ef}(S_*+U)\|_{\rm op}\).
The elementary-square seed has each Hessian block at most \(g/6\).
There are three other edges per incident plaquette. The correction
norm gives the original mixed row bound
\[
 \sum_{f\ne e}c_{ef}\le c_*^{\rm mix}
 :=\frac{d_*g}{2}+\frac{2r}{3}.                         \tag{13}
\]
The symbol \(c_*^{\rm mix}\) is a derivative bound, not a Casimir.

If
\[
 \delta=\gamma_1-2c_*^{\rm mix}>0,                     \tag{14}
\]
the actual conditional Poincare comparison matrix has a positive
uniform diagonal-dominance margin. Freezing any exterior simply
restricts its nonnegative off-diagonal majorant to a principal
submatrix. The compact-group exact-one-form Poisson proof therefore
gives, for every nonempty block regardless of cardinality,
\[
 \boxed{\quad
\gamma_B(\mu_B^z)\ge\delta,\qquad \hbox{all }B,z.
 \quad}                                               \tag{15}
\]
An exact rational lower bound in (12) suffices in (14).

Here is the essential compact-group argument. On a fixed conditional
block solve its centered weighted Poisson equation \(Au=f-\mu_B^zf\).
Such an inverse exists on every finite compact positive-density block
by bounded-density comparison; no uniform inverse is assumed.
For \(\alpha_e=d_eu\), the one-coordinate weighted Weitzenbock energy,
including its connection, Ricci and diagonal Hessian terms, equals
\(\|A_eu\|_2^2\ge\gamma_1\|d_eu\|_2^2\) after disintegration.
The scalar conditional gap justifies this bound on these exact forms;
it is not asserted for arbitrary cotangent fields.
Other-coordinate connection energies are nonnegative. Testing the full
one-form Poisson identity componentwise and applying the mixed-Hessian
bounds gives \(\mathsf Hx\le v\), where
\(x_e=\|d_eu\|_2\), \(v_e=\|d_ef\|_2\),
\(\mathsf H_{ee}=\gamma_1\), and
\(\mathsf H_{ef}=-2c_{ef}\). Positive diagonal dominance implies
\(\mathsf H^{-1}\ge0\) entrywise and
\(\mathsf H\ge\delta I\). Therefore
\(\operatorname{Var}f=\langle df,du\rangle
\le v^T\mathsf H^{-1}v\le\delta^{-1}\|df\|_2^2\), proving (15).

This is stronger in scope than inserting a global gap into a
conditional: a global gap alone does not prove (15). The local premises
were established for every exterior, and the matrix proof keeps those
quantifiers. The same theorem also follows true marginals by an exact
Schur update; conditioning and marginalization are distinct operations.

The original-coordinate curvature bound is another available route:
\[
 \rho=\frac12-\frac{4d_*g}{3}-\frac{4r}{3}.             \tag{16}
\]
When positive, restriction to any block preserves the same Hessian
majorant and gives \(\gamma_B\ge\rho\). A consumer may report the maximum
of the independently earned direct-Haar, Schur and curvature floors.
No derivative bound from a different gauge chart is accepted in (16).

Multiplying these diffusion floors by \(\kappa/2\) expresses them in
the microscopic energy units of (1). For a proper conditioned subset
this does not identify its generator with an independently constructed
Wilson Hamiltonian. Taking the entire edge set recovers a neutral
spectral consequence; only a cardinality-independent route supplies a
volume-independent spectral floor.

## 5. Numerical consequences and limits of this comparison

The existing cubic source at \(\kappa=16,r=1/8\) has
\[
 g=1/64,\qquad \Omega_1=1/2,\qquad
 c_*^{\rm mix}=11/96,\qquad \rho=1/4.
\]
Thus (15) gives the diffusion floor
\( (3/4)e^{-1/2}-11/48\), while (16) gives the stronger \(1/4\).
The already earned dimensionless physical floor is therefore \(2\).
Merely running the oscillation calculation again does not enlarge its
source window.

The independently audited polar-fusion improvement of the source
bilinear bound now supplies the canonically replayed cubic witness
\(\kappa=15,r=1/10\). Its original-coordinate curvature floor is
\[
 \rho=367/1350,\qquad \frac{\kappa\rho}{2}=367/180>2.
\]
Every one-edge exterior conditional has the stronger four-step direct
floor \(31970155204/69198046875\), approximately \(0.4620\) in
diffusion units. That one-edge value is not a whole-box spectral bound.
For blocks of seven and sixty-four edges, the implementation instead
selects the cardinality-independent \(367/1350\) route. Both bounds
replay from the same actual source with their different quantifiers.

A second replayed source uses \(\kappa=15,r=1/8,b=33/32\), earning a
spatial correction tail as well. Here
\[
 g=4/225,\qquad
 \Omega_1=353/675,\qquad
 c_*^{\rm mix}=107/900,\qquad
 \rho=161/675,\qquad
 \frac{\kappa\rho}{2}=\frac{161}{90}.
\]
The source improvement, rather than the elementary comparison, earns
the new coupling. These local conditional
bounds then attach every-exterior quantifiers to the same actual
source.

A seven-edge theta embedded in the interior of a cubic box illustrates
the missing boundary issue. It has two internal squares, six further
coplanar plaquettes touching its outer boundary and fourteen
perpendicular plaquettes: twenty crossing plaquettes. Comparing its
actual conditional to an isolated theta vacuum must therefore also
control these twenty interactions.

For example, if the isolated theta has a proved full-scalar diffusion
floor \(\gamma_{\theta}\) and logarithmic correction \(U_\theta\) with
radius \(r_\theta\), a valid comparison is only
\[
 \gamma_{\mu_B^z}\ge\gamma_{\theta}
 \exp\!\left[-\frac{160g}{3}-\frac{56r}{3}
                    -\frac{13r_\theta}{18}\right].
 \tag{17}
\]
The final term uses the proved theta coefficient dual. The middle term
is the ambient local norm bound (6) for seven edges; the first is the
unremoved crossing-plaquette oscillation. An isolated neutral gap would
not suffice even for this comparison, since arbitrary exterior
conditionals require all scalar block functions.

At the existing \(\kappa=16,r=1/8\), the first two exponents in (17)
already total \(19/6\). This deliberately conservative transfer can lose
most of the isolated gap. It exposes why assembling isolated passing
spectra, without boundary-uniform reference control or tighter
cross-block corrections, is not a scalable proof.

## 6. A verified literature option, not a production premise

For a single SU(2) link and fixed exterior, the Wilson reference
conditional (9) is especially simple. Write that link as a unit
quaternion. Every incident fundamental character is linear in its four
real quaternion coordinates, including inverse orientations. Their
weighted sum is a single linear field. After a rotation its density is
therefore \(e^{h q_0}\) relative to uniform measure on the unit \(S^3\).

[Li–Ma–Zhang, Theorem 1.2, *On the spectral gap of Boltzmann measures on
the unit sphere* (2021)](https://doi.org/10.1016/j.spl.2020.108963)
gives the unit-\(S^{n-1}\) lower bound \(n-2\), uniformly in the field.
For \(n=4\), scaling to the SU(2) radius-two metric gives the full-scalar
conditional reference floor \(1/2\). This is not the sharper radial
one-dimensional bound from their Proposition 1.1; angular modes remain
part of the required full space.

It follows from the published theorem and (6) that an optional stronger
reference choice is
\[
 \gamma_1\ge e^{-8r/3}
       \max\left\{\frac12,\frac34e^{-8d_*g/3}\right\}.
 \tag{18}
\]
The all-field branch prevents exponential deterioration from a large
linear seed field. It is weaker than the elementary Haar branch for
the actual cubic \(\kappa=15,16\) witnesses. Accordingly no production
certificate needs that external theorem here. In either branch, a small
actual correction and inter-edge derivative control are still required
to obtain a useful Schur margin.

## 7. What a scalable next test must measure

A bounded next target is to certify the full-scalar gap of a coupled
block reference uniformly over its compact exterior-staple parameter
set, together with its mixed derivatives into neighboring blocks.
The block reference must contain the boundary terms of (9). The local
norm supplies (6) for the actual correction and, when \(b>1\), the
neighborhood-tail control (7). This specifies the missing finite versus
analytic inputs without pretending that a finite grid covers the
exterior quantifier.

The successful comparison must reduce the cross-block matrix norm
enough to leave a positive conditional Schur margin. Increasing block
size while retaining only the union bound (6) worsens the oscillation
exponent, so increasing block size alone is not evidence of progress.
Existing actual finite-family sources, their local bounds, and every
new certificate remain at fixed microscopic coupling. No continuum,
all-scale renormalization, infinite-volume reconstruction or Clay
claim follows from these conditional estimates.

## 8. Executable conditional certificates

The source-consuming implementation is
`omnibias.geometry.gauge.transfer.wilson_conditional_block`.
It accepts only canonical cubic residual, linear or polar Wilson source
certificates. The radius is inherited, never supplied. Arbitrary graph
membership is not inferred from a family source. Strip gauge coordinates
and fixed-theta source schemas are explicitly refused by this original
cubic-edge consumer.

For each requested positive block-size bound it records the local
oscillation, every-exterior quantifier, exact exponential domain, three
separate conditional routes, and the best applicable diffusion floor.
The energy-scaled conditional result is explicitly not a new Hamiltonian
with frozen Wilson terms. The cardinality-independent floor is a
different field and is absent if both Schur and curvature routes fail.

As a regression of that distinction, the actual polar source at
\(\kappa=100,r=1/2\) has an unnecessarily loose but passing nonlinear
radius. Both of its global sufficient gap routes fail. The direct local
comparison still earns a positive every-exterior conditional floor for
each specified finite block size. It earns no cardinality-independent
floor in that case. A failed nonlinear source, by contrast, stays
`INCONCLUSIVE` and cannot borrow a conditional bound from a different
radius. Independent finite algebra and canonical tamper regressions
check these cases; they do not replace the all-spin proof above.

