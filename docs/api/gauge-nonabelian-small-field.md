# Nonabelian trace jets and actual weak-field block localization

These exact-rational APIs concern SU(2) Wilson plaquettes and the actual
finite periodic-cubic Hamiltonian vacuum. They provide local Taylor,
geometry and localization bounds. They do not construct a continuum theory.

## Ordered noncommuting trace jets

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import (
    su2_plaquette_trace_jet,
    replay_su2_nonabelian_small_field_certificate,
)

result = su2_plaquette_trace_jet(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], degree=12,
    parameter_radius=Q(1, 2),
)
assert result["witness"]["arithmetic"]["coefficients"][:4] == ["2", "0", "-3/4", "1/4"]
assert replay_su2_nonabelian_small_field_certificate(result["certificate"])
assert not result["continuum_claim"]
```

The four ordered, oriented log vectors define

\[
\chi(t)=\operatorname{Tr}\prod_{j=1}^4\exp(i t A_j\cdot\sigma/2).
\]

Coordinates, `parameter_radius` and optional `norm_bounds` are integers
or fractions; floats and booleans are refused. Supplied rational bounds
are checked by \(|A_j|^2\le b_j^2\); the default uses the safe vector
\(\ell^1\) norm. `degree` is any nonnegative integer. Coefficients are
Taylor coefficients, including the factorial divisor, computed by exact
quaternion jet convolution. The convention \(q_0I+i\mathbf q\cdot\sigma\)
has a **minus** vector cross term in quaternion multiplication.

For \(R=\sum_j b_j/2\), multinomial Leibniz and real-path unitarity give
\(|\chi^{(n)}(t)|\le2R^n\). Therefore, at every real \(|t|\le T\),

\[
\left|\chi(t)-\sum_{j=0}^N c_jt^j\right|
\le\frac{2(RT)^{N+1}}{(N+1)!}.
\]

This is a closed-form jet with a certified Taylor remainder, without nested
autodifferentiation. The endpoint enclosure also uses \(-2\le\chi\le2\).
Numerical grid and random-sample regressions are independent diagnostics;
the all-parameter enclosure follows from the derivative bound.

## Magnetic remainder and unreduced electric chart

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import su2_wilson_small_field_budget

t = Q(1, 4096)
result = su2_wilson_small_field_budget(t**5, t**2)
a = result["witness"]["arithmetic"]
assert Q(a["magnetic_cubic_absolute_upper_per_plaquette"]) == 2*t
assert Q(a["magnetic_remainder_after_cubic_upper_per_plaquette"]) == Q(8, 3)*t**3
assert not result["physical_gauge_fixed_electric_metric_verified"]
```

For four log vectors of norm at most \(\varepsilon\le2\), put
\(B=\sum_j A_j\) and \(T=\sum_{i<j<k}A_i\cdot(A_j\times A_k)\).
Pauli multiplication and the degree-three remainder prove

\[
\chi=2-|B|^2/4+T/4+R_4,\quad |T|\le4\varepsilon^3,
\quad |R_4|\le4\varepsilon^4/3.
\]

For the convention
\(aH=\kappa\sum_e C_e/2+2\sum_p(2-\chi_p)/\kappa\), the cubic
magnetic term is at most \(2\varepsilon^3/\kappa\) in absolute value,
and the remainder is at most \(8\varepsilon^4/(3\kappa)\).
The orthogonal commutator loop \((A,B,-A,-B)\) has zero linear curl
and zero cubic term but action \(4\sin^4(\varepsilon/2)>0\).
It forbids a relative bound by quadratic curl on the whole unfixed chart.

For one original link with \(r=|A|<2\pi\), Haar density in these
coordinates is proportional to \(h^2\), with
\(h=\sin(r/2)/(r/2)\). The inverse metric is
\(G^{-1}=P_r+h^{-2}P_t\). Half-density conjugation gives the exact
unreduced scalar Dirichlet-form identity

\[
hCh^{-1}=-\operatorname{div}(G^{-1}\nabla)-1/4.
\]

Indeed \(G^{-1}\nabla h=\nabla h\) and \(\Delta h=-h/4\).
Since \(1-\varepsilon^2/24\le h\le1\), the relative derivative
error is at most \((1-\varepsilon^2/24)^{-2}-1\); the electric scalar
shift is \(-\kappa/8\) per original edge. This localized chart is not
a gauge-invariant domain. Its metric cannot be assigned to a tree-gauged
physical Hamiltonian without a separate calculation.

## Actual finite-block localization

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import su2_wilson_block_small_field

t = Q(1, 2**20)
result = su2_wilson_block_small_field(t**5, t**2, block_side=2)
a = result["witness"]["arithmetic"]
assert Q(a["good_localized_vacuum_norm_squared_lower"]) > 0
assert a["good_normalized_energy_above_vacuum_upper"] is not None
assert result["actual_gauge_invariant_vacuum_localization_verified"]
assert result["all_state_localization_operator_bound_verified"]
```

The family is every isotropic periodic cubic SU(2) lattice of integer side
\(L\ge\max(3,b+1)\) and every embedded unwrapped block with vertices
\(\{0,\ldots,b\}^3\). The axial tree takes all z-links, y-links at
z=0 and x-links at y=z=0. Each remaining link is a ribbon product of at
most \(2b\) plaquettes. Bi-invariant distance obeys
\(r\le\pi\sqrt{2-\chi}\le(22/7)\sqrt{2-\chi}\).
Consequently, with

\[
n=3b^2(b+1),\qquad s=(7\varepsilon/(44b))^2,
\]

all block plaquette actions at most \(s\) imply tree-gauged link logs
at most \(\varepsilon\). The nested
[actual-vacuum moment certificate](gauge-wilson-large-field.md) gives
\(\mathbb E A_p\le m=\min(3\kappa/2,2)\), hence

\[
\Pr(\text{bad block})\le\min(1,nm/s)
\le\min(1,C_b\kappa/\varepsilon^2),\quad
C_b=8712b^4(b+1)/49.
\]

For physical localization put \(S=\sum_{p\in B}A_p\). Let
\(\theta(S)\) increase linearly from zero at \(s/2\) to \(\pi/2\)
at \(s\), constant outside, and set
\(\chi_g=\cos\theta\), \(\chi_b=\sin\theta\).
This is a gauge-invariant Sobolev partition. Its good closed support lies
in \(S\le s\), so the tree-patch implication applies there.

For the actual vacuum \(\psi_0\), Markov and the groundstate form give

\[
\|\chi_g\psi_0\|^2\ge\max(0,1-2nm/s),\qquad
\mathcal E_0(\chi_g)+\mathcal E_0(\chi_b)
\le\left(\frac{22}{7s}\right)^2
       \frac\kappa2 n\min(4,n)m(4-m).
\]

Here \(\mathcal E_0(F)=\langle F\psi_0,(aH-E_0)F\psi_0\rangle\).
The gradient identity \(\sum_e|\nabla_e A_p|^2=4A_p-A_p^2\),
at most four plaquettes incident on an edge, and Cauchy--Schwarz prove
the cost bound. If the norm floor \(v_g\) is positive, cost divided by
\(v_g\) bounds the normalized localized state's energy above the vacuum.
It is a vacuum approximation, not an orthogonal excitation.

There is also a universal single-block IMS bound. Pointwise,
\(\Gamma(S)\le4\min(4,n)S\), and \(\theta'\) is supported where
\(s/2<S<s\). Therefore the localization multiplication error obeys

\[
\frac\kappa2\bigl(\Gamma(\chi_g)+\Gamma(\chi_b)\bigr)
\le\frac{968\kappa\min(4,n)}{49s}.
\]

This bounds the partition error for **every** physical state, not only
the vacuum. On the bad cutoff's closed support, \(S\ge s/2\), so the
local magnetic potential is at least \(s/\kappa\). This is an
uncentered potential bound; subtracting the ambient vacuum energy is
a separate step.

For every fixed \(b\), the exact substitution
\(\kappa=t^5,\varepsilon=t^2,0<t\le1\) gives bad probability
\(O_b(t)\), retained norm \(1-O_b(t)\), vacuum localization cost
\(O_b(t^2)\), cubic magnetic error \(2t\), and quartic remainder
\(8t^3/3\). No sampling or extrapolation supplies these exponents.
The constants depend on block size. The physical energy cost also requires
division by spatial spacing, which this substitution does not control.

For integer \(b\ge2\), choosing \(\kappa=b^{-30}\) and
\(\varepsilon=b^{-12}\) instead gives explicit growing-block bounds:
patch failure at most \(17424/(49b)\), retained norm at least
\(1-34848/(49b)\), summed magnetic error at most
\(12/b^3+16/b^{15}\), vacuum localization cost at most
\([(34848/49)(44/7)^4]/b^5\), and universal IMS error
\([(3872/49)(44/7)^2]/b^4\). The bad-support local potential floor
is \((7/44)^2b^4\). The separately proved
[isolated kinetic comparison](gauge-isolated-block-kinetic.md) has error
at most \(166/b^6\). Its isolated operator is not the conditional
operator of the embedded block in the vacuum bound.

All three certificate types have canonical replay, including nested moment
and chart sources. A valid but vacuous probability or norm bound remains
explicit. The universal single-block IMS flag is earned only by the block
producer. Parent, conditional polymer, continuum and formal flags remain
false. Exterior-conditioned physical metric, cross-block control,
renormalized Schwinger functions and the continuum mass gap remain separate
obligations.
