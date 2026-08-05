# 2-D SQG steady vortex (single-Riesz / Poisson substrate)

The [2-D Euler steady vortex](euler2d-vortex.md) certificate left one frontier
open: genuine **surface quasi-geostrophic (SQG)** velocity

\[
u = R^\perp\theta = \nabla^\perp(-\Delta)^{-1/2}\theta = (-R_2\theta,\,R_1\theta)
\]

needs the *single* Riesz transform / half-Laplacian \(|\xi|^{-1}\), which is **not
elementary** on the Euler blob basis \(f_a=a^2/(\pi D^2)\) (only the *composite*
\(R_iR_k\) is). This page discharges that obligation by changing the basis to the
one on which the half-Laplacian is *itself* diagonal in closed form: the **2-D
Poisson kernel**.

## Why the Poisson kernel unlocks SQG

For a scale \(a>0\) with \(D=x^2+y^2+a^2\) set

\[
\theta_a(x)=\frac{1}{2\pi}\,\frac{a}{D^{3/2}},
\qquad
\hat\theta_a(\xi)=e^{-a|\xi|}.
\]

Because the symbol is \(e^{-a|\xi|}\), *every* half-power of \(-\Delta\) acts by a
plain multiplication and stays elementary. In particular the stream function and
the **single Riesz transform** are closed form:

\[
\psi_a=(-\Delta)^{-1/2}\theta_a=\frac{1}{2\pi}\,\frac{1}{D^{1/2}},
\qquad
R_j\theta_a=\partial_j\psi_a=-\frac{1}{2\pi}\,\frac{x_j}{D^{3/2}},
\qquad
u=\nabla^\perp\psi_a=\frac{1}{2\pi}\,\frac{(y,-x)}{D^{3/2}}.
\]

The new substrate [`omnibias.core.verified.sqg`](../api/core.md) provides all of
these as outward-rounded intervals (the half-power \(D^{1/2}\) uses the rigorous
`Interval.sqrt`). The closed forms are pinned down two independent ways in the
tests: a direct `mpmath` **Hankel transform** of the symbol (the radial 2-D
inverse Fourier transform, agreeing to \(\sim10^{-18}\)) and `mpmath` derivatives
of \(\psi_a\).

!!! note "Scope — genuine SQG, but an exact steady state (not the blow-up problem)"
    The single Riesz transform is **closed form** here, so this is *genuine* SQG —
    not an Euler stand-in. It is, however, an **exact steady state**, so it says
    nothing about the celebrated open **SQG finite-time singularity** question (a
    steady solution is trivially global). `honesty.blowup_claim`, `unproven_claim` and
    `three_d_claim` are all `False`; the open singularity problem is recorded in
    `open_obligations`.

## The exact steady state

For a radial \(\theta=\sum_i c_i\theta_{a_i}\) the velocity \(u\) is *tangential*
(\(\parallel(y,-x)\)) and \(\nabla\theta\) is *radial* (\(\parallel(x,y)\)), so
their dot product is the **zero polynomial** \(y\,x-x\,y\equiv0\):

\[
u\cdot\nabla\theta
  = \Big(\tfrac{1}{2\pi}\textstyle\sum_i\tfrac{c_i}{D_i^{3/2}}\Big)
    \Big(\textstyle\sum_j\tfrac{-3 c_j a_j}{2\pi D_j^{5/2}}\Big)\,
    \big[y\,x - x\,y\big] \equiv 0 .
\]

The active scalar is transported by its own \(R^\perp\) velocity — which here is
fully closed form — and the residual is an **exact 0** over the whole plane.

```python
from omnibias.pinn.certified import certified_sqg_steady_vortex

cert = certified_sqg_steady_vortex(
    coeffs=[1.0, 0.4, -0.2], scales=[0.6, 1.3, 2.1],
)
assert cert["steady_residual_certified_sup"] == 0.0
assert cert["honesty"]["sqg_velocity_closed_form"] is True
```

## Certified whole-plane norms

By radial symmetry the velocity, temperature and strain *magnitudes* collapse to
**1-D functions of \(r\)** with half-integer powers:

\[
|u| = \frac{r\,|S(r)|}{2\pi},\quad
|\theta| = \frac{\big|\sum_i c_i a_i D_i^{-3/2}\big|}{2\pi},\quad
\lVert\nabla u\rVert_F^2 = \frac{2S^2 - 6r^2 S T + 9 r^4 T^2}{(2\pi)^2},
\]

with \(S=\sum_i c_i D_i^{-3/2}\), \(T=\sum_i c_i D_i^{-5/2}\) (from \(S'=-3rT\)).
Each is enclosed by per-cell [`TaylorModel`](../api/core.md)s on \([0,R]\) —
assembling \(D_i^{-3/2}=\text{invd}\cdot\sqrt{\text{invd}}\) from the rigorous
`TaylorModel.reciprocal` **and the new `TaylorModel.sqrt`** — plus an explicit
far-field tail (\(|rS|\le S_0/r^2\), \(|\theta|\le S_a/r^3\),
\(M\le 17 S_0^2/r^6\)), giving genuine *whole-plane* sups. For the 3-blob example:

| quantity | value | meaning |
|---|---|---|
| `steady_residual_certified_sup` | **`0.0`** | exact (perpendicularity identity) |
| `velocity_sup` \(\lVert u\rVert_\infty\) | `0.1817` | certified whole-plane |
| `temperature_sup` \(\lVert\theta\rVert_\infty\) | `0.4787` | certified whole-plane |
| `strain_sup` \(\lVert\nabla u\rVert_{F,\infty}\) | `1.1057` | certified whole-plane |
| `total_temperature` \(\sum_i c_i\) | `1.2` | exact (unit-mass blobs) |
| `steady_residual_grid_max` | `8.3e-18` | substrate re-confirmation of the zero |
| `divergence_grid_max` | `2.1e-16` | \(\nabla\cdot u=0\) |
| `riesz_perp_identity_grid_max` | `4.6e-16` | \(u=R^\perp\theta\) (two routes agree) |

Because \(u\sim 1/r^2\) at infinity (faster than Euler's \(1/r\)), the **kinetic
energy \(\int|u|^2\) is finite for any coefficients**; `kinetic_energy_finite` is
`True`.

## Independent replay

The numpy-only twin in [`omnibias.symbolic.sqg`](../api/symbolic.md) imports
nothing from `omnibias.pinn.certified`. It recomputes the total temperature, re-confirms the
steady state / divergence / \(u=R^\perp\theta\) identities from *its own* closed
forms, and — as an anti-faking guard — **densely samples** the three radial
magnitudes to check the certified sups genuinely dominate them:

```python
from omnibias.symbolic import verify_sqg_steady_vortex

rep = verify_sqg_steady_vortex(cert)
assert rep["replay_match"] is True
assert rep["sup_dominates_samples"] is True   # certified sup >= dense sample
assert rep["steady_residual_is_zero"] is True
```

A forged certificate that understates a sup is caught (`sup_dominates_samples`
becomes `False`).

## Toward the singularity: the certified obstruction

The steady vortex is global by construction. The genuinely hard, *open* question
is finite-time blow-up. SQG is **scale invariant**
(\(\theta_\lambda(x,t)=\theta(\lambda x,\lambda t)\)), so a self-similar
singularity at \(t=T\) would take the form \(\theta(x,t)=\Theta(y)\),
\(y=x/(T-t)\), with the profile solving the *stationary* equation

\[
F[\Theta] := (y + R^\perp\Theta)\cdot\nabla\Theta = 0,
\qquad
V := y + R^\perp\Theta,\quad \nabla\!\cdot V = \nabla\!\cdot y = 2
\]

(\(R^\perp\Theta\) is divergence free, so all of \(\nabla\!\cdot V\) comes from the
self-similar drift \(y\)). Pairing \(F\) with \(\Theta\) and integrating by parts
gives an **exact, basis-independent identity** — and with it a hard obstruction:

\[
\langle F[\Theta],\Theta\rangle
  = -\tfrac12\!\int(\nabla\!\cdot V)\,\Theta^2
  = -\lVert\Theta\rVert_2^2
\quad\Longrightarrow\quad
\boxed{\;\lVert F[\Theta]\rVert_2 \ge \lVert\Theta\rVert_2 > 0\;}
\]

by Cauchy–Schwarz. So **no nontrivial _localized_ exact self-similar profile
exists**: the self-similar drift forces the profile residual to stay bounded below
by the profile's own \(L^2\) norm. `certified_sqg_selfsimilar_blowup_attempt`
certifies this lower bound for \(\Theta=\sum_i c_i\theta_{a_i}\): the squared norm
**diagonalises in closed form** on the Poisson basis,

\[
\langle\theta_a,\theta_b\rangle_{L^2(\mathbb R^2)} = \frac{1}{2\pi(a+b)^2},
\qquad
\lVert\Theta\rVert_2^2 = \sum_{ij}\frac{c_i c_j}{2\pi(a_i+a_j)^2},
\]

(verified by `omnibias.core.verified.sqg.sqg_blob_l2_inner`, checked against an
`mpmath` quadrature), and the substrate grid confirms
\((R^\perp\Theta)\cdot\nabla\Theta\equiv0\) so that \(F=y\cdot\nabla\Theta\)
pointwise for these radial profiles.

```python
from omnibias.pinn.certified import certified_sqg_selfsimilar_blowup_attempt

cert = certified_sqg_selfsimilar_blowup_attempt(
    coeffs=[1.0, 0.4, -0.2], scales=[0.6, 1.3, 2.1],
)
assert cert["exact_selfsimilar_profile_exists"] is False         # certified
assert cert["selfsimilar_residual_l2_lower_bound"] > 0.0          # ||F|| >= ||Theta||
lo, hi = cert["selfsimilar_residual_inner_product"]
assert hi < 0.0                                                   # <F,Theta> = -||Theta||^2
assert cert["divergence_of_selfsimilar_drift"] == 2.0
```

### Why this isn't the end — and what the one open lemma is

The obstruction kills the *exact* route; the live program is **approximate**
self-similarity plus nonlinear stability. The certificate also records the
linear-stability diagnostic that frames precisely what is missing. In the natural
\(L^2\) energy the rescaled drift is **destabilizing**,

\[
\langle -y\cdot\nabla W,\,W\rangle = +\lVert W\rVert_2^2
\qquad(\nabla\!\cdot y = 2),
\]

so coercivity (hence a *conditional* blow-up) cannot come from the drift — it must
come from the nonlocal stretching term measured in a **weighted / higher-regularity
norm**. That single infinite-dimensional coercivity inequality is the one open
analytic lemma, recorded crisply in `open_obligations`:

- approximate self-similar profile with a certified small rescaled residual (2-D
  box Taylor models),
- **infinite-dimensional coercivity / spectral gap** of the linearized rescaled
  operator in a weighted high-regularity norm — *the blocker*,
- nonlinear remainder closure (radii polynomial) in that norm,
- *front-type* (non-decaying) profiles evade the localized \(L^2\) identity
  (Córdoba 1998 rules out the simplest separately).

!!! warning "What is and isn't claimed"
    This is a **rigorous negative result** (an obstruction) plus an honest map of
    the conditional pipeline — **not** a blow-up proof. `honesty.blowup_claim`,
    `unproven_claim` and `three_d_claim` are all `False`;
    `exact_selfsimilar_profile_exists` is the *certified* `False`. If the named
    coercivity lemma is ever proven, the conditional pipeline closes; until then it
    is isolated as exactly one inequality.

The numpy twin re-derives the whole obstruction from an **independent radial
quadrature** — \(\lVert\Theta\rVert_2^2\), \(\langle F,\Theta\rangle\) and
\(\lVert F\rVert_2\) — and checks the no-go identity and the lower bound directly:

```python
from omnibias.symbolic import verify_sqg_selfsimilar_blowup_attempt

rep = verify_sqg_selfsimilar_blowup_attempt(cert)
assert rep["replay_match"] is True
assert rep["nogo_identity_holds"] is True     # <F,Theta> = -||Theta||^2 by quadrature
assert rep["obstruction_holds"] is True       # ||F|| >= ||Theta|| > 0
```

## Narrowing the lemma: the certified coercivity engine

The obstruction says coercivity must come from the **stretching** term, not the
drift. The next certificate makes the *linear* part of that statement quantitative
and exposes the exact inequality that remains open.

For a background \(\bar\Theta=\sum_i c_i\theta_{a_i}\) the operator that governs a
perturbation \(W\) of the rescaled flow is

\[
\mathcal L W = -\,y\cdot\nabla W \;-\;(R^\perp\bar\Theta)\cdot\nabla W
              \;-\;(R^\perp W)\cdot\nabla\bar\Theta .
\]

Its \(L^2\) self-adjoint part has a **fully certified** Rayleigh lower bound from
three exact facts: the self-similar drift contributes \(+1\)
(\(\langle -y\cdot\nabla W,W\rangle=\lVert W\rVert_2^2\)); the divergence-free
background transport contributes \(0\); and the stretching term is controlled by
the **Riesz isometry** \(\lVert R^\perp W\rVert_2=\lVert W\rVert_2\) (symbol
\(R_1^2+R_2^2=1\)), giving \(|\langle (R^\perp W)\!\cdot\!\nabla\bar\Theta,W\rangle|
\le\lVert\nabla\bar\Theta\rVert_\infty\lVert W\rVert_2^2\). Hence (Weyl)

\[
\langle \mathcal L W, W\rangle \;\ge\;
\bigl(1-\lVert\nabla\bar\Theta\rVert_\infty\bigr)\lVert W\rVert_2^2,
\]

an \(L^2\) spectral gap \(\ge 1-\lVert\nabla\bar\Theta\rVert_\infty\), with the
whole-plane \(\lVert\nabla\bar\Theta\rVert_\infty\) certified by the same radial
`TaylorModel.sqrt` machinery:

```python
from omnibias.pinn.certified import certified_sqg_linearized_coercivity_attempt

cert = certified_sqg_linearized_coercivity_attempt(coeffs=[0.15, 0.05], scales=[1.2, 2.0])
assert cert["drift_self_adjoint_coefficient"] == 1.0     # div y = 2
assert cert["riesz_isometry_constant"] == 1.0            # R_1^2 + R_2^2 = 1
# The gap is rounded outward (down), so it sits at or below the naive difference.
assert cert["l2_coercivity_gap_lower"] <= 1.0 - cert["grad_theta_sup"]
assert cert["l2_coercive"] is True
```

The number is also produced by a **general, reusable** primitive,
[`certified_block_operator_gap`](../api/core.md) — the finite-section + tail
*Schur* bound

\[
\lambda_{\min}(S)\ \ge\ \tfrac12\Bigl[(a+d)-\sqrt{(a-d)^2+4b^2}\Bigr],
\]

for a self-adjoint operator split by a projection into a finite block (gap \(a\),
computed here), a coupling \(b\), and a **tail gap \(d\) that is an explicit
hypothesis**. Coercivity holds iff \(d>b^2/a\) — *one scalar inequality*. This is
the bridge from a finite computer computation to an infinite-dimensional theorem:
the higher/weighted-norm coercivity that the singularity program actually needs is
exactly this primitive fed by a weighted-norm assembly, with the tail bound as the
sole open input.

!!! warning "This is an L² *linear diagnostic*, not stability"
    `certified_sqg_linearized_coercivity_attempt` certifies a property of the
    linearized operator's \(L^2\) self-adjoint part. It is **not** a stability or
    blow-up result: the background is *not* an exact profile (none exists), and
    finite-time singularity is governed by a higher/weighted norm controlling
    \(\lVert\nabla\theta\rVert\) in which the stretching term *loses a derivative*.
    `blowup_claim`, `unproven_claim`, `three_d_claim` and `stability_claim` are all
    `False`; `what_this_does_not_prove` and `open_obligations` say so explicitly.

The numpy twin densely re-samples \(\lVert\nabla\bar\Theta\rVert_\infty\) (the
certified whole-plane sup must dominate it), re-derives the Weyl and block-engine
gaps, and checks the honesty flags:

```python
from omnibias.symbolic import verify_sqg_linearized_coercivity_attempt

rep = verify_sqg_linearized_coercivity_attempt(cert)
assert rep["replay_match"] is True
assert rep["grad_sup_dominates_samples"] is True   # anti-faking the sup
assert rep["block_gap_match"] is True              # general engine == Weyl number
```

## Where this sits

- Certificate: `omnibias.pinn.certified.certified_sqg_steady_vortex`.
- Verified substrate: [`omnibias.core.verified.sqg`](../api/core.md) (single Riesz
  transform, stream function and velocity on the Poisson blob basis, checked vs an
  `mpmath` Hankel transform) and [`TaylorModel`](../api/core.md) `reciprocal` +
  `sqrt` (radial norm sups with half-powers).
- Independent twins: `omnibias.symbolic.sqg.verify_sqg_steady_vortex` and
  `verify_sqg_selfsimilar_blowup_attempt` (independent radial quadrature).
- 2-D Euler sibling: [2-D Euler steady vortex](euler2d-vortex.md) (whose recorded
  SQG open obligation this page discharges).
- Self-similar obstruction: `omnibias.pinn.certified.certified_sqg_selfsimilar_blowup_attempt`
  — proves no *localized exact* self-similar profile exists and isolates the single
  open coercivity lemma for the conditional blow-up program.
- Coercivity engine: `omnibias.pinn.certified.certified_sqg_linearized_coercivity_attempt`
  (certified \(L^2\) gap \(\ge 1-\lVert\nabla\bar\Theta\rVert_\infty\); twin
  `omnibias.symbolic.verify_sqg_linearized_coercivity_attempt`) built on the general
  `omnibias.core.verified.certified_block_operator_gap` finite-section + tail bound.
- **Next frontier (open):** the genuine **SQG finite-time singularity** question.
  The exact self-similar route is now *certified closed* and the *linear* \(L^2\)
  coercivity is *certified*; the live route is approximate self-similarity + the one
  recorded infinite-dimensional coercivity lemma in a weighted high-regularity norm
  (the open input to the block-operator gap engine), which needs *time-dependent* /
  non-radial 2-D box Taylor models.
