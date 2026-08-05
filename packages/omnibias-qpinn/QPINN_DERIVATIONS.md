# QPINN derivations

Per-equation derivations of the split-real residuals shipped by
`omnibias-qpinn`. All equations below are written in atomic units
(:math:`\hbar = 1`, :math:`m_e = 1`); the constants reappear when the
user passes ``hbar`` and ``mass`` to the residual.

## Conventions

A complex wavefunction is encoded as two real components,

\[
    \psi(x, t) = \psi_R(x, t) + i\,\psi_I(x, t).
\]

The kinetic operator is :math:`\hat T = -(1/2m)\,\nabla^2`. The
external potential :math:`V(x, t)` is assumed real (Hermitian).
A residual :math:`R(x, t)` is the LHS of the PDE; for a well-trained
network :math:`R \to 0` pointwise.

The metric convention for the relativistic equations is **mostly
minus** :math:`\eta_{\mu\nu} = \text{diag}(+1, -1, -1, -1)`. With this
choice :math:`(\gamma^0)^2 = +I`, :math:`(\gamma^i)^2 = -I` and
:math:`\Box = -\partial_t^2 + \nabla^2`.

## 1. Time-independent Schrodinger (TISE)

The eigenvalue problem :math:`\hat H \psi = E \psi`, with
:math:`\hat H = \hat T + V`, becomes

\[
    R_R(x) = (\hat H\psi)_R - E\,\psi_R, \qquad
    R_I(x) = (\hat H\psi)_I - E\,\psi_I.
\]

Since the Hamiltonian is real, :math:`(\hat H\psi)_R = -(1/2m)\,\nabla^2
\psi_R + V\,\psi_R` and similarly for the imaginary channel. The
factor of :math:`i` does **not** appear in TISE; both channels evolve
independently.

The variational energy estimate uses the Rayleigh quotient

\[
    E_{\text{est}} = \frac{\int\big(\psi_R\,(\hat H\psi)_R
                          + \psi_I\,(\hat H\psi)_I\big)\,dx}
                        {\int|\psi|^2\,dx}.
\]

Implementation: `omnibias.qpinn.torch.equations.tise` /
`omnibias.qpinn.jax.equations.tise`.

## 2. Time-dependent Schrodinger (TDSE)

The TDSE is

\[
    i\,\partial_t\psi(x, t) = \hat H\psi(x, t).
\]

Writing :math:`\psi = \psi_R + i\,\psi_I`,

\[
    i\,(\partial_t\psi_R + i\,\partial_t\psi_I)
    = (\hat H\psi)_R + i\,(\hat H\psi)_I.
\]

Separating real / imaginary parts,

\[
    -\partial_t\psi_I = (\hat H\psi)_R,
    \qquad
    +\partial_t\psi_R = (\hat H\psi)_I.
\]

Multiplying by ``hbar`` for general units, the residual is

\[
    R_R(x, t) = -\hbar\,\partial_t\psi_I - (\hat H\psi)_R,
    \qquad
    R_I(x, t) = +\hbar\,\partial_t\psi_R - (\hat H\psi)_I.
\]

## 3. Nonlinear Schrodinger / Gross-Pitaevskii

The NLSE adds a density-dependent term :math:`g\,|\psi|^2` to the
potential:

\[
    \hat H_{\text{NL}}\psi = \hat H\psi + g\,|\psi|^2\,\psi.
\]

Since :math:`g\,|\psi|^2` is real, it acts as a local real potential
on the two channels separately:

\[
    R_R = -\hbar\,\partial_t\psi_I - (\hat H\psi)_R - g\,|\psi|^2\,\psi_R,
\]

\[
    R_I = +\hbar\,\partial_t\psi_R - (\hat H\psi)_I - g\,|\psi|^2\,\psi_I.
\]

## 4. Helmholtz

The (homogeneous) Helmholtz equation is :math:`(\nabla^2 + k^2)\psi
= 0` (steady wave equation):

\[
    R_R = \nabla^2\psi_R + k^2\,\psi_R, \qquad
    R_I = \nabla^2\psi_I + k^2\,\psi_I.
\]

For an inhomogeneous medium the wavenumber is :math:`k(x)`; the
residual extends term-by-term. A non-zero source
:math:`s(x) = (s_R, s_I)` is added to the residual (this is how
`omnibias-qpinn` encodes :math:`(\nabla^2 + k^2)\psi = -s` for
scattering problems).

## 5. Klein-Gordon (real scalar)

The Klein-Gordon equation for a real scalar field :math:`\phi(x^\mu)`
is

\[
    (\Box + m^2)\,\phi + \lambda\,\phi^3 = 0,
\]

with :math:`\Box = -\partial_t^2 + \nabla^2`. The
:math:`\lambda\,\phi^3` term arises from a :math:`(\lambda / 4)\,\phi^4`
self-interaction in the Lagrangian. The residual is therefore

\[
    R(x^\mu) = \Box\phi - m^2\,\phi - \lambda\,\phi^3.
\]

In `omnibias-qpinn` this is real (single-component), so we use the
plain `ComponentSpec(("phi",))` rather than the split-complex
encoding.

## 6. Dirac

The free Dirac equation is

\[
    (i\,\gamma^\mu\,\partial_\mu - m)\,\psi = 0.
\]

On a 4-spinor :math:`\psi^a = \psi^a_R + i\,\psi^a_I`, the matrix
multiplication :math:`(\gamma^\mu \partial_\mu \psi)^a = \sum_b
\gamma^\mu_{ab}\,\partial_\mu\psi^b` is the only non-trivial piece;
the entries :math:`\gamma^\mu_{ab}` can be complex (specifically the
:math:`\gamma^2` entries pick up factors of :math:`\pm i` from
:math:`\sigma_y`).

For a constant 4x4 complex matrix
:math:`M^a{}_b = M^{re}_{ab} + i\,M^{im}_{ab}` acting on a complex
spinor :math:`\psi^b`:

\[
    (M\psi)^a_R = \sum_b\big(M^{re}_{ab}\,\psi^b_R - M^{im}_{ab}\,\psi^b_I\big),
\]

\[
    (M\psi)^a_I = \sum_b\big(M^{re}_{ab}\,\psi^b_I + M^{im}_{ab}\,\psi^b_R\big).
\]

This is the formula in :func:`omnibias.qpinn._core.spinor.apply_gamma_matrix`.
The residual of :math:`i\,\gamma^\mu\partial_\mu\psi - m\psi = 0`
splits into

\[
    R^a_R = -\big[\gamma^\mu\partial_\mu\psi\big]^a_I - m\,\psi^a_R,
\]

\[
    R^a_I = +\big[\gamma^\mu\partial_\mu\psi\big]^a_R - m\,\psi^a_I,
\]

following the standard rule that multiplying by :math:`i` swaps real
and imaginary parts with a sign flip on the new real channel.

## 7. Bloch-periodic cage

The Bloch wavefunction has the form :math:`\psi_k(x) = e^{i k\cdot
x}\,u_k(x)`, where :math:`u_k(x)` is periodic with the lattice. In
split-real,

\[
    \psi_R(x) = \cos(k\cdot x)\,u_R(x) - \sin(k\cdot x)\,u_I(x),
\]

\[
    \psi_I(x) = \sin(k\cdot x)\,u_R(x) + \cos(k\cdot x)\,u_I(x).
\]

For the first derivative along axis :math:`a`,

\[
    \partial_a\psi_R = \cos(k\cdot x)\,\partial_a u_R - \sin(k\cdot x)\,\partial_a u_I
                       - k_a\,\psi_I,
\]

\[
    \partial_a\psi_I = \sin(k\cdot x)\,\partial_a u_R + \cos(k\cdot x)\,\partial_a u_I
                       + k_a\,\psi_R.
\]

For the second derivative (which Schrodinger / Helmholtz need),

\[
    \partial_a^2\psi_R = \cos(k\cdot x)\,\partial_a^2 u_R - \sin(k\cdot x)\,\partial_a^2 u_I
                         - 2 k_a\,\big[\sin(k\cdot x)\,\partial_a u_R + \cos(k\cdot x)\,\partial_a u_I\big]
                         - k_a^2\,\psi_R,
\]

\[
    \partial_a^2\psi_I = \sin(k\cdot x)\,\partial_a^2 u_R + \cos(k\cdot x)\,\partial_a^2 u_I
                         + 2 k_a\,\big[\cos(k\cdot x)\,\partial_a u_R - \sin(k\cdot x)\,\partial_a u_I\big]
                         - k_a^2\,\psi_I.
\]

Higher orders follow from the Leibniz / Newton-binomial expansion of
:math:`\partial_a^n[e^{i k\cdot x}\,u]`; v0.0.1 supports up to second
order (which covers Schrodinger / Helmholtz). Mixed partials and
fourth order (biharmonic) are slated for v0.0.2.

## 8. Norm-conservation cage

The hard cage parameterises

\[
    \psi(x) = \tilde\psi(x) / N,
    \quad
    N = \sqrt{\sum_q w_q\,|\tilde\psi(x_q)|^2 + \varepsilon},
\]

on a fixed quadrature grid :math:`\{x_q, w_q\}`. Since :math:`N` is
independent of :math:`x`, the derivatives are scaled by :math:`1/N`
without any extra terms; in particular :math:`\nabla^2\psi =
\nabla^2\tilde\psi / N` exactly, preserving the closed-form path.

The corresponding soft loss is

\[
    L_{\text{norm}}(\psi)
    = \Big(\sum_q w_q\,|\psi(x_q)|^2 - 1\Big)^2,
\]

useful when one prefers a regulariser over the hard projection.

## 9. Molecular local energy (Born-Oppenheimer)

For fixed nuclei the electronic Hamiltonian in atomic units is

\[
    \hat H = -\tfrac12\sum_j\nabla_{r_j}^2
             - \sum_{j,a}\frac{Z_a}{\lVert r_j - R_a\rVert}
             + \sum_{j<k}\frac{1}{\lVert r_j - r_k\rVert}
             + \sum_{a<b}\frac{Z_a Z_b}{\lVert R_a - R_b\rVert}.
\]

The **local energy** :math:`E_L = \hat H\psi/\psi` is the quantity a variational
Monte-Carlo run averages. Writing the kinetic term through
:math:`\log|\psi|` gives the drift ("log-derivative") form that avoids the
:math:`1/\psi` blow-up:

\[
    \frac{\nabla^2\psi}{\psi}
    = \nabla^2\log|\psi| + \lVert\nabla\log|\psi|\rVert^2
    \quad\Longrightarrow\quad
    T_L = -\tfrac12\big(\nabla^2\log|\psi| + \lVert\nabla\log|\psi|\rVert^2\big),
\]

so :math:`E_L = T_L + V(R, r)`. omnibias computes :math:`(\nabla, \nabla^2)\log|\psi|`
in **closed form** from the multivariate jet tower of an MLP
:math:`\log|\psi|` (`log_psi_derivatives`, via `mlp_jet_mv`), never by autodiff or
finite differences. The bare potential :math:`V` is `coulomb_potential`.
Implementation: `omnibias.qpinn.{torch,jax}.molecular`.

**Hydrogen-atom oracle.** The 1s orbital :math:`\psi = e^{-Z r}` has
:math:`\log|\psi| = -Z r`, so in 3-D :math:`\nabla\log|\psi| = -Z\,\hat r`,
:math:`\nabla^2\log|\psi| = -2Z/r`, and

\[
    T_L = -\tfrac12\!\left(-\tfrac{2Z}{r} + Z^2\right) = \tfrac{Z}{r} - \tfrac{Z^2}{2},
    \qquad V = -\tfrac{Z}{r},
    \qquad E_L = -\tfrac{Z^2}{2},
\]

constant over the whole domain (zero local-energy variance) — the exact ground
state eigenvalue.

**Harmonic-trap oracle.** For :math:`\log|\psi| = -\tfrac{\omega}{2} r^2` in
:math:`D` dimensions, :math:`\nabla\log|\psi| = -\omega r`,
:math:`\nabla^2\log|\psi| = -\omega D`, and with :math:`V = \tfrac12\omega^2 r^2`
the :math:`r^2` terms cancel exactly, leaving :math:`E_L = \tfrac{D\omega}{2}`.

## 10. Nuclear-cusp cage

The Kato cusp condition requires the spherically-averaged wavefunction to
satisfy :math:`\psi'(0)/\psi(0) = -Z_a` as an electron approaches nucleus
:math:`a`. The hard cage multiplies any base ansatz by

\[
    C(r) = \exp\!\Big(\sum_a u_a(s_a)\Big),
    \qquad
    u_a(s) = -\frac{Z_a\,s}{1 + b_a\,s},
    \qquad s_a = \lVert r - R_a\rVert,
\]

so that :math:`\psi = C(r)\,\psi_{\text{base}}(r)`. The Padé form gives
:math:`u_a'(0) = -Z_a` (the cusp slope) and saturates to a finite shift at large
:math:`s`, so it does not distort the tail. Because :math:`\log C = \sum_a u_a`
is additive, its derivatives compose with the base through the Leibniz product
rule; the cage supplies the closed-form value, gradient, Laplacian, and mixed
partials up to second order (`NuclearCuspField`). With a constant base
(:math:`\psi_{\text{base}} = 1`, single nucleus, :math:`b = 0`) the cage reduces
to the hydrogenic :math:`e^{-Z s}` and reproduces :math:`E_L = -Z^2/2`.

## 11. Padé-Jastrow correlation factor

Electron correlation is captured by a symmetric factor :math:`e^{J}` additive to
:math:`\log|\psi|`, with the two-body Padé form

\[
    J = \sum_{j<k} \frac{a_{\sigma}\,r_{jk}}{1 + b\,r_{jk}}
        + \sum_{j,a} \frac{c\,s_{ja}}{1 + d\,s_{ja}},
    \qquad r_{jk} = \lVert r_j - r_k\rVert .
\]

The electron-electron slope :math:`a_\sigma` is fixed by the Kato e-e cusp:
:math:`a_\sigma = \tfrac12` for antiparallel spins and :math:`\tfrac14` for
parallel (the extra factor of two from the Pauli node). The e-n term carries the
:math:`-Z` slope. Since :math:`J` is additive, the local kinetic energy of the
correlated ansatz :math:`\psi = e^{J}\,\det M` is

\[
    T_L = -\tfrac12\Big(\nabla^2\log|\psi| + \lVert\nabla\log|\psi|\rVert^2\Big),
    \qquad
    \nabla\log|\psi| = \nabla J + \nabla\log|\det M|,
\]

so the cross term :math:`2\,\nabla J\cdot\nabla\log|\det M|` must be retained.
`jastrow_slater_local_kinetic_energy` computes it from the closed-form
:math:`(\nabla J, \nabla^2 J)` (`jastrow_value_grad_laplacian`) and the separated
:math:`(\nabla, \nabla^2)\log|\det M|` (`tier2_grad_laplacian_log_psi`).
Implementation: `omnibias.ferminet.jastrow`.
