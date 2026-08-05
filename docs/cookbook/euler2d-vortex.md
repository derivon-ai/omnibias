# 2-D Euler steady vortex (Riesz/Leray substrate)

The [CCF-on-the-line calibration](ccf-line-calibration.md) lives on the exact
*line* Hilbert pair — the 1-D nonlocal-transport model of 3-D Euler. This page is
its **2-D companion**: a certificate built on the verified *planar* substrate
[`omnibias.core.verified.riesz`](../api/core.md), which provides the
Biot–Savart map \(u=\nabla^\perp\Delta^{-1}\omega\) and the second-order Riesz /
Leray building blocks in closed form on the radial blob basis
\(f_a = a^2/(\pi D^2)\), \(D=x^2+y^2+a^2\).

For a radial vorticity field \(\omega=\sum_i c_i f_{a_i}\) (every blob centred at
the origin) the Biot–Savart velocity is *tangential* and \(\nabla\omega\) is
*radial*:

\[
u=\frac{1}{2\pi}\,(-y,x)\sum_i\frac{c_i}{D_i},
\qquad
\nabla\omega=(x,y)\sum_i\frac{-4c_i a_i^2}{\pi D_i^3}.
\]

`certified_euler2d_steady_vortex` turns the three consequences into a
theorem-grade, replayable certificate.

!!! warning "Scope — this is 2-D **Euler**, not SQG, and not a blow-up"
    The substrate's *single* Riesz transform of a radial blob is **not
    elementary** (it carries the half-Laplacian \(|\xi|^{-1}\)); only the
    *composite* \(R_iR_k\) and the Biot–Savart \(\nabla^\perp\Delta^{-1}\) are
    closed form. So genuine SQG velocity \(u=R^\perp\theta=\nabla^\perp(-\Delta)^{-1/2}\theta\)
    is **out of reach on this basis** and is recorded as an open obligation. What
    *is* certified here is 2-D incompressible Euler, which is globally regular
    (Beale–Kato–Majda) — this is an **exact regular steady state**, the opposite
    of a singularity. `honesty.unproven_claim`, `three_d_claim`, `blowup_claim` and
    `sqg_claim` are all `False`.

## The exact steady state

A tangential field dotted with a radial field is the **zero polynomial**:

\[
u\cdot\nabla\omega
  = \Big(\tfrac{1}{2\pi}\textstyle\sum_i\tfrac{c_i}{D_i}\Big)
    \Big(\textstyle\sum_j\tfrac{-4 c_j a_j^2}{\pi D_j^3}\Big)\,
    \big[(-y)\,x + x\,y\big] \equiv 0 .
\]

There is no floating-point error to bound — \((-y)x+xy=0\) for *all* real
\(x,y\) — so the reported `steady_residual_certified_sup` is an **exact 0** over
the whole plane, not a sampled estimate. The certificate then re-confirms it by
evaluating \(u\) and \(\nabla\omega\) *independently through the substrate* on a
grid: `steady_residual_grid_max` \(\sim 10^{-17}\).

```python
from omnibias.pinn.certified import certified_euler2d_steady_vortex

cert = certified_euler2d_steady_vortex(
    coeffs=[1.0, 0.4, -0.2], scales=[0.6, 1.3, 2.1],
)
assert cert["steady_residual_certified_sup"] == 0.0
assert cert["honesty"]["exact_steady_state"] is True
```

## Certified whole-plane norms

By radial symmetry the velocity, vorticity and strain *magnitudes* collapse to
**1-D functions of \(r\)**:

\[
|u| = \frac{r\,|Q(r)|}{2\pi},\quad
|\omega| = \frac{|\Omega_p(r)|}{\pi},\quad
\lVert\nabla u\rVert_F^2 = \frac{\Omega_p(r)^2 + r^4 W(r)^2}{2\pi^2},
\]

with \(Q=\sum_i c_i/D_i\), \(\Omega_p=\sum_i c_i a_i^2/D_i^2\),
\(W=\sum_i c_i/D_i^2\). Each is enclosed by per-cell
[`TaylorModel`](../api/core.md)s on \([0,R]\) (assembling \(1/D_i\) from a
rigorous `TaylorModel.reciprocal`) plus an explicit far-field tail
(\(|rQ|\le S_0/r\), \(|\Omega_p|\le S_2/r^4\), …), giving genuine *whole-plane*
sups. For the 3-blob example:

| quantity | value | meaning |
|---|---|---|
| `steady_residual_certified_sup` | **`0.0`** | exact (perpendicularity identity) |
| `velocity_sup` \(\lVert u\rVert_\infty\) | `0.1482` | certified whole-plane |
| `vorticity_sup` \(\lVert\omega\rVert_\infty\) | `0.9614` | certified whole-plane |
| `strain_sup` \(\lVert\nabla u\rVert_{F,\infty}\) | `0.6796` | certified whole-plane |
| `circulation` \(\Gamma=\sum_i c_i\) | `1.2` | exact (unit-mass blobs) |
| `steady_residual_grid_max` | `1.6e-17` | substrate re-confirmation of the zero |
| `divergence_grid_max` | `2.7e-16` | \(\nabla\cdot u=0\) |
| `riesz_trace_identity_grid_max` | `4.5e-15` | \(R_{11}\omega+R_{22}\omega=-\omega\) |
| `leray_divergence_grid_max` | `1.1e-15` | Leray projection is div-free |

Because \(u\sim\Gamma/(2\pi r)\) at infinity, the **kinetic energy is finite only
when \(\Gamma=0\)**; `kinetic_energy_finite` records this honestly (a
zero-circulation vortex pair has finite energy, a net-circulation vortex does
not).

## Independent replay

The numpy-only twin in [`omnibias.symbolic.euler2d`](../api/symbolic.md) imports
nothing from `omnibias.pinn.certified`. It recomputes the circulation, re-confirms the steady
state / divergence / Calderón–Zygmund identities from *its own* closed forms, and
— as an anti-faking guard — **densely samples** the three radial magnitudes to
check the certified sups genuinely dominate them:

```python
from omnibias.symbolic import verify_euler2d_steady_vortex

rep = verify_euler2d_steady_vortex(cert)
assert rep["replay_match"] is True
assert rep["sup_dominates_samples"] is True   # certified sup >= dense sample
assert rep["steady_residual_is_zero"] is True
```

A forged certificate that understates a sup is caught (`sup_dominates_samples`
becomes `False`), exactly as for the CCF continuum-operator twin.

## Where this sits

- Certificate: `omnibias.pinn.certified.certified_euler2d_steady_vortex`.
- Verified substrate: [`omnibias.core.verified.riesz`](../api/core.md) (Biot–Savart,
  second-order Riesz, Leray, checked vs `mpmath`) and
  [`TaylorModel`](../api/core.md) (radial norm sups).
- Independent twin: `omnibias.symbolic.euler2d.verify_euler2d_steady_vortex`.
- 1-D sibling on the line: [CCF-on-the-line calibration](ccf-line-calibration.md).
- **Next frontier (open):** a certified single Riesz / half-Laplacian of a blob
  would lift this from 2-D Euler to genuine 2-D **SQG**.
