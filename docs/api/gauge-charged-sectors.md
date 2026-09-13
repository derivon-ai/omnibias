# Exact SU(2) charged-sector dimensions

`omnibias.geometry.gauge.transfer.charged_sectors` counts invariant tensors on
a finite labelled graph. This supplies an electric-sector admissibility and
basis-dimension check. It does not construct magnetic matrix elements or certify
an energy, a mass gap, or a continuum theory.

```python
from omnibias.geometry.gauge.transfer.charged_sectors import (
    charged_spin_network_dimension,
    su2_singlet_multiplicity,
)

assert su2_singlet_multiplicity([1, 1, 1, 1, 1, 1]) == 5
assert charged_spin_network_dimension(
    3, [(0, 1), (1, 2)], [1, 1], {0: [1], 2: [1]}
) == 1
assert charged_spin_network_dimension(3, [(0, 1), (1, 2)], [1, 1]) == 0
```

Every spin label is the nonnegative Python integer `two_j = 2*j`. Floats and
bools are rejected. Vertices are numbered from zero. Parallel links remain
distinct labelled links; a self-loop supplies two incident representation
factors. An external charge is a labelled representation factor, and each
vertex can carry any finite sequence of charges. These factors include their
color spaces; they are not selected fixed color vectors. Orienting a link the
other way does not change its SU(2) invariant dimension, since irreducible
SU(2) representations are self-dual.

For fixed edge spins, Peter–Weyl decomposition gives one factor
`V_j tensor V_j*` per edge. Regrouping these factors with the external charge
spaces at each vertex makes the local gauge actions independent. Therefore

\[
\dim\mathcal H_{\{j_e\},\{q\}}^{\mathrm{inv}}
=\prod_v\dim\operatorname{Inv}_{SU(2)}
\left(\bigotimes_{e\ni v}V_{j_e}\otimes\bigotimes_{q\text{ at }v}V_q\right),
\]

where self-loops occur twice in the incidence product. This is the usual
fixed-graph spin-network decomposition; see
[Baez, *Spin Network States in Gauge Theory*](https://arxiv.org/abs/gr-qc/9411007).
No division by representation dimensions or permutations of identical labels
is made. An empty graph and isolated neutral vertices contribute dimension one.

Local multiplicities use exact integer Clebsch–Gordan fusion: twice-spins `a`
and `b` produce each channel `abs(a-b), abs(a-b)+2, ..., a+b` once.
Multiplicity counts add when channels coincide. Trivalent admissibility is
the triangle and parity test; vertices of larger valence can have several
intertwiners. The implementation prunes channels that the remaining factors
cannot cancel, and uses a channel-intersection formula for four factors.
Its general cost can grow with the numerical spin labels, so this is not a
polynomial-time bound in their binary encoding length.

Regression checks compare fusion with an independent Laurent-weight character
calculation: for a product of SU(2) characters, singlet multiplicity equals the
coefficient of weight zero minus the coefficient of weight two. Tests cover
all labels zero through three at valence up to six, randomized larger
valences, the 100-fundamental Catalan count, charge endpoints, orientation
reversal, parallel edges, self-loops, and the six-valent periodic `2x2x2` cube.
The all-fundamental cube sector has dimension `5**8`, whereas adding a
fundamental charge at two vertices without changing the edge labels violates
local parity and gives dimension zero. Changing the connecting edge to spin
zero restores admissibility. These are exact finite representation checks.
