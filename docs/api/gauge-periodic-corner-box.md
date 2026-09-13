# A quantitative open-box lower bound for the periodic corner tilt

This note proves a lower energy-density comparison for the actual SU(2)
Hamiltonian on every periodic cubic lattice of side \(N\ge\ell\). It keeps
all spins and the original link electric form. For \(\ell=8192\), the explicit
constants below give, uniformly for \(0<\kappa\le2^{-330}\) and
\(0\le t\le1/4\),

\[
 \frac{E_0(H_{N,t})}{N^3}
 \ge e^{\rm q}_{N,t}-\frac1{64}.                    \tag{1}
\]

The reference on the right is the identity-background periodic quadratic
energy density, with zero frequencies contributing zero. No toron-minimum
assumption enters the lower bound. This is an energy-density statement,
not a conditional excitation or mass-gap theorem. The finite rational
certificate replays the constants in this written proof; it does not
formalize the analytic argument.

## 1. Hamiltonian and a boundary replacement that retains every face

Use positively oriented original links on the periodic cubic lattice and
the normalization \(C_{1/2}=3/4\). For each vertex \(x\), take the three
coherently oriented corner faces \(xy,yz,zx\). Their action sum is
\(S_x\), and their transported product has the six-edge boundary word
\(D_x\). Write \(A(U)=2-\operatorname{Tr}U\). Each elementary plaquette
occurs in exactly one \(S_x\), and

\[
 H_{N,t}=\frac\kappa2\sum_e C_e+
 \frac1\kappa\sum_x V_{x,t},\qquad
 V_{x,t}=(2+t/2)S_x-(t/2)A(D_x).                   \tag{2}
\]

The unitary chord triangle inequality implies
\(A(D_x)\le3S_x\). Consequently

\[
 V_{x,t}\ge(2-t)S_x\ge\frac74 S_x
 \quad(0\le t\le1/4).                             \tag{3}
\]

Partition each coordinate circle into consecutive intervals with vertex
lengths between \(\ell\) and \(2\ell-1\). This exists for every
\(N\ge\ell\): write \(N=q\ell+r\), \(0\le r<\ell\), and use one
interval of length \(\ell+r\) and \(q-1\) of length \(\ell\).
Products of these intervals are the boxes. Retain only their ordinary,
nonwrapping internal links. In particular, if there is just one interval,
the edge crossing its chosen periodic seam is still removed; endpoint
membership alone does not define an internal link.

For a wholly contained corner cell retain its exact potential \(V_{x,t}\).
For a crossing cell use (3), retain \((2-t)A(U_p)\) for each of its wholly
internal plaquettes, and discard its other nonnegative terms. Keep every
internal original electric term and discard the remaining positive electric
terms. The resulting operator is a sum of independent open-box operators:

\[
 H_{N,t}\ge\sum_B H_{B,t}.                         \tag{4}
\]

Each internal face has a positive coefficient, either in a retained full
cell or in the boundary replacement. There is no double allocation because
each face has one designated corner cell. Merely dropping crossing cells
would not suffice: already for one cube, three retained corner faces leave
two unconstrained cycle holonomies.

The form inequality (4) holds before imposing Gauss invariance. Although
fibers of a global physical state need not be neutral box states, the scalar
ground energy of each compact open-box operator equals its neutral ground
energy: its positive ground state is unique and its gauge transforms have
the same normalization and energy. Thus
\(E_0(H_{N,t})\ge\sum_B E_0(H_{B,t})\). No isolated-box conditional
ground-state substitution is made.

## 2. An explicit tree and global control of the small well

For a box with vertex side lengths \(n_i\in[\ell,2\ell-1]\), use the
comb tree consisting of all \(x\)-links, the \(y\)-links at \(x=0\),
and the \(z\)-links at \(x=y=0\). Root paths first move in \(z\), then
in \(y\), then in \(x\). Let \(S_B\) be the sum of all internal face
actions. Uniform integer bounds are

\[
 V_B\le V_*:=8\ell^3,\qquad E_B\le E_*:=24\ell^3,
 \qquad C_*:=96\ell^4.                             \tag{5}
\]

A \(y\)-chord at coordinate \(x>0\) closes a rectangle filled by \(x\)
elementary faces. A \(z\)-chord closes the folded union of an \(xz\)
rectangle of width \(x\) and a \(yz\) rectangle of width \(y\), with at
most \(x+y\le4\ell\) faces. Successive plaquette gluing expresses each
chord holonomy as a product of these face holonomies conjugated by transport
paths. Multiplicativity of the unit-quaternion norm and Cauchy--Schwarz give

\[
 A(U_c)\le4\ell S_B,\qquad
 \sum_{c\ {
m chord}}A(U_c)\le C_*S_B.             \tag{6}
\]

Tree gauge fixing sends Haar measure to product Haar on the chord variables;
the removed tree coordinates integrate to one. The residual root action is
simultaneous conjugation. Since every internal face is retained, zero box
potential means flatness on every elementary face. The open cubical box is
simply connected, so a flat connection is pure gauge. After tree fixing the
only minimum is that every chord equals \(I\).

On \(S_B\le s:=r^2/C_*\), write
\(U_c=(\sqrt{1-|z_c|^2},z_c)\). Equation (6) ensures the positive
hemisphere and \(Z^2:=\sum_c|z_c|^2\le r^2\). Up to a constant that
cancels from Rayleigh quotients, Haar density satisfies

\[
 w(z)=\prod_c(1-|z_c|^2)^{-1/2},\qquad
 1\le w\le1+4r^2\quad(r\le1/2).                  \tag{7}
\]

Indeed \(\prod_c(1-|z_c|^2)\ge1-Z^2\); the final scalar inequality
\((1-r^2)^{-1/2}\le1+4r^2\) holds for \(r\le1/2\).

## 3. Original-link metric and finite-word potential remainders

At a tree-gauge representative, an original derivative acts on any chord
by a left field, a right field, their signed difference, or zero. The two
possible occurrences are the two endpoint tree paths. No field occurs more
than twice. In quaternion coordinates the field matrices are
\((\sqrt{1-|z_c|^2}I\pm[z_c]_\times)/2\). Each differs from \(I/2\)
by at most \(|z_c|\) in operator norm.

Write the complete derivative matrix as \(B(z)\) and its metric as
\(G=B^TB\). Block row/column norm bounds give
\(\|B_0\|\le E_*\) and \(\|B-B_0\|\le2E_*r\).
Each chord's own original electric term contributes \(I/4\) at zero,
so \(G_0\ge I/4\). Hence

\[
 (1-c_G r)G_0\le G\le(1+c_G r)G_0,
 \qquad c_G=32E_*^2,
 \quad r\le1.                                    \tag{8}
\]

For clarity, a uniform finite-word estimate used for the potential is

\[
 \left|A\!\left(\prod_{j=1}^k U_{c_j}^{\epsilon_j}\right)
          -\left|\sum_{j=1}^k\epsilon_j z_{c_j}\right|^2\right|
 \le192r\sum_{j=1}^k|z_{c_j}|^2,
 \quad k\le6,\ r\le1/32,                         \tag{9}
\]

where the chord occurrences are distinct. To prove it, set
\(T=\sum_j|z_{c_j}|^2\) and
\(h_j=U_{c_j}^{\epsilon_j}-I\). Then
\(\|h_j\|\le2|z_{c_j}|\),
\(\sum_j\|h_j\|\le2\sqrt6r<1/2\), and the scalar linear remainder
is at most \(T\). The sum of products of order at least two is at most
\((\sum_j\|h_j\|)^2\le24T\), by the finite elementary-symmetric
expansion: if \(L=\sum_j\|h_j\|\le1/2\), then
\(\sum_{m\ge2}L^m/m!\le L^2\), for example using
\(m!\ge2\cdot3^{m-2}\). Thus the difference from
\((0,\sum_j\epsilon_jz_{c_j})\) has norm at most \(25T\).
Comparing squared norms bounds the error by
\((125+625r)rT\le192rT\), proving (9).

Elementary faces have four distinct original links, and the corner boundary
has six. Tree links contribute identities. A chord appears in at most four
elementary faces and two translated corner-boundary words. Using (9),
\(2+t/2\le17/8\), and \(t/2\le1/8\), the complete box potential
\(\mathcal V_B\) and its quadratic part \(\mathcal V_{B,0}\) obey

\[
 |\mathcal V_B-\mathcal V_{B,0}|\le2048r Z^2.       \tag{10}
\]

The sharper coefficient from this counting is \(1680\); \(2048\) is used
uniformly. Boundary replacement coefficients are at most two, so the same
bound covers those faces. Linearizing (6) gives
\(Z^2\le C_*S_{B,0}\). By (3),
\(\mathcal V_{B,0}\ge(7/4)S_{B,0}\ge Z^2/C_*\). Therefore

\[
 \mathcal V_B\ge(1-c_Vr)\mathcal V_{B,0},\qquad c_V=2048C_* . \tag{11}
\]

This also proves nondegeneracy of the rooted well. No lower eigenvalue is
estimated by a sampled spectrum or an unquantified compactness argument.

## 4. Quantitative IMS lower bound

Let \(E_B^{\rm q}\) denote the exact original-metric quadratic ground
energy of the box. Equivalently,
\(E_B^{\rm q}=(3/2)\operatorname{Tr}_+\sqrt{K_B}\), where \(K_B\)
is the original-link curl quadratic matrix. Linear tree elimination changes
coordinates but not its positive oscillator frequencies; the removed
gradients have zero curl. The plaquette coefficient matrix has norm at most
\(17/16\), and the cubic curl has squared norm at most twelve. Thus

\[
 0\le K_B\le\frac{51}{4}I<16I,
 \qquad E_B^{\rm q}\le18V_B.                      \tag{12}
\]

Here \(V_B\) denotes the number of box vertices, as in (5); the potential
function is \(\mathcal V_B\).

For \(R=\sqrt{S_B}\), each original edge meets at most four internal
plaquettes. Since a four-link face has
\(\sum_e\Gamma_e A_p=4A_p-A_p^2\),
\(\Gamma S_B\le16S_B\) and \(\Gamma R\le4\) almost everywhere.
Use the cosine/sine partition with angle zero for \(R\le\sqrt s/2\),
linear to \(\pi/2\) at \(R=\sqrt s\), and constant thereafter.
The original-link IMS cost is at most
\(2\pi^2\kappa/s<20\kappa/s\).

Put \(a=\max(c_G,c_V)r\), \(b=4r^2\), and
\(f=(1-a)/(1+b)\). Apply the partition to the unique positive scalar ground
state of the box. This state is gauge invariant, as explained in Section 1;
the cutoffs depend only on \(S_B\), so both localized pieces remain gauge
invariant. Tree elimination and the reduced metric apply to these pieces.
The good piece, extended by zero from the chord chart to its Euclidean
tangent space, has Rayleigh floor \(fE_B^{\rm q}\) by (7)--(11). The bad
piece has \(S_B\ge s/4\), so (3) gives the conservative floor
\(s/(4\kappa)\). IMS on this actual ground state proves

\[
 E_0(H_{B,t})\ge
 \min\left\{fE_B^{\rm q},\frac{s}{4\kappa}\right\}
 -\frac{20\kappa}{s}.                             \tag{13}
\]

Having bounded the scalar ground energy, its lower bound is a lower bound
for every scalar form value by the variational definition of that energy.
This is what licenses the arbitrary scalar fibers in (4); tree elimination
was not applied to an arbitrary nonneutral fiber.

Choose the following exact dyadic constants:

\[
 \begin{split}
 C&=c_G+c_V,\qquad
 j=\left\lceil\log_2(16384C)\right\rceil,\quad r=2^{-j},\quad s=r^2/C_*,\\
 p_0&=2j+\left\lceil\log_2(8192C_*V_*)\right\rceil,\qquad
 p=3\left\lceil p_0/3\right\rceil,\quad\kappa_*=2^{-p}.
 \end{split}                                      \tag{14}
\]

All ceilings are computed with integer bit lengths, not floating logarithms.
They ensure \(a\le1/16384\), \(b\le1/16384\),
\(f\ge1-1/8192\), and
\(\kappa_*\le s/(8192V_*)\). In particular the bad branch exceeds
\(18V_*\), hence the minimum in (13) is its good branch. Uniformly for
every allowed box and \(0<\kappa\le\kappa_*\),

\[
 \frac{E_0(H_{B,t})}{V_B}
 \ge\frac{E_B^{\rm q}}{V_B}-\varepsilon_B,
 \quad
 \varepsilon_B=18(1-f)+\frac{20\kappa_*}{s\ell^3}
 \le\frac1{128}.                                 \tag{15}
\]

For example, (14) gives a crude universal upper bound
\(18/8192+20/8192=19/4096<1/128\) on (15). The implementation retains
the sharper exact rational expression. No unspecified small-coupling
remainder or caller-supplied error is an input.

## 5. Surface comparison with the periodic quadratic density

Embed the direct sum of box quadratic matrices into the original periodic
link space by zero on removed links, calling it \(K_{\rm split}\).
For a crossing cell, its exact quadratic coefficient is
\[
 Q_t=I+\frac t4(I-\mathbf1\mathbf1^T)
 \ge(1-t/2)I.
\]
Its retained boundary coefficient is \((1-t/2)P\), where \(P\) selects
its internal faces. Therefore its difference is positive semidefinite and
has rank at most three after composition with its three curl rows. If
\(n_c\) cells cross boxes,

\[
 0\le K_{\rm split}\le K_{N,t}\le\frac{51}{4}I,
 \qquad\operatorname{rank}(K_{N,t}-K_{\rm split})\le3n_c.
 \tag{16}
\]

For positive matrices with spectra in \([0,M]\), a positive rank-\(q\)
perturbation changes \(\operatorname{Tr}\sqrt{K}\) by at most
\(q\sqrt M\): eigenvalue interlacing telescopes the ordered square roots.
This argument includes zero modes. Applying it to (16) gives

\[
 0\le N^3e_{N,t}^{\rm q}-\sum_B E_B^{\rm q}\le18n_c.
 \tag{17}
\]

A box of vertex side lengths \(n_1,n_2,n_3\) contains exactly
\((n_1-1)(n_2-1)(n_3-1)\) complete corner cells. Consequently

\[
 \frac{n_c}{N^3}
 \le\max_B\sum_{i=1}^3\frac1{n_i}\le\frac3\ell.
 \tag{18}
\]

Combining (4), (15), and (17)--(18) proves the explicit lower density bound

\[
 \boxed{\frac{E_0(H_{N,t})}{N^3}
 \ge e_{N,t}^{\rm q}-\frac{54}{\ell}-\varepsilon_B.}
 \tag{19}
\]

At \(\ell=8192\), the surface term is \(27/4096<1/128\),
\(j=107\), \(p_0=328\), and \(p=330\). Equations (15) and (19)
therefore imply (1) for every \(N\ge8192\). The proof treats global
torons and all remaining link variables through the unrestricted lower
operator inequality; it does not condition on or minimize over a chosen
flat holonomy sector. A normalized actual periodic vacuum is used only
when another theorem later applies the resulting ground-energy bound.

## 6. Source contract and remaining scope

```python
from omnibias.geometry.gauge.transfer.periodic_corner_box import (
    su2_periodic_corner_box_lower,
    replay_su2_periodic_corner_box_lower_certificate,
)

result = su2_periodic_corner_box_lower()
assert result["status"] == "PASS"
assert result["arithmetic"]["kappa_exponent"] == 330
assert result["actual_nonlinear_lower_verified_in_written_analysis"]
assert replay_su2_periodic_corner_box_lower_certificate(result["certificate"])
```

The implementation is
`omnibias.geometry.gauge.transfer.periodic_corner_box`.
`su2_periodic_corner_box_lower(box_side=8192, kappa_exponent=None)` computes
the fixed radius and exact budgets above. A supplied exponent selects the
interval \(0<\kappa\le2^{-p}\); it is checked against the actual branch
and error inequalities. Exponents are positive multiples of three, so
\(\tau=\kappa^{1/3}\) has a rational dyadic endpoint.
`replay_su2_periodic_corner_box_lower_certificate` reconstructs all inputs,
constants, flags and metadata canonically.

Only passing sources assert
`actual_nonlinear_lower_verified_in_written_analysis`. Failed budgets are
inconclusive about the mathematical theory. The source does not assert a
conditional spectral gap, arbitrary Gauss-sector comparison, a continuum
limit, or a Yang--Mills mass gap. A matching upper density estimate and a
separate variational argument are required for the intended vacuum moment
inequality. IMS and harmonic-well comparison are established techniques;
the present proof supplies the explicit finite-geometry constants instead
of invoking an unspecified asymptotic error.


The dedicated regressions independently check the exact dyadic budget,
all eight retained-face subsets in the crossing-cell quadratic comparison,
comb-tree connectivity and full face-curl ranks, the removed seam when the
partition has only one interval, and rational quaternion word remainders.
They also check canonical replay after rehashed source and scope mutations.
These finite algebra checks supplement the written theorem; they do not
formalize its analytic steps.
