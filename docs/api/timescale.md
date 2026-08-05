# omnibias-timescale

Time-scale (Hilger) calculus: one differential/difference calculus over an arbitrary
closed subset of the reals, unifying the continuous and discrete registers.

- **`TimeScale`** — the reals `R`, the uniform grid `hZ`, the quantum scale `q^Z ∪ {0}`,
  and finite sets, with jump operators `sigma` / `rho` and graininess `mu` / `nu`.
- **Delta / nabla derivatives** — `delta_derivative` (`f^Δ`) and `nabla_derivative`
  (`f^∇`), and `delta_derivative_tower`, which **dispatches** to the closed-form derivative
  tower on `R`, to the forward difference on `hZ`, and to the Jackson q-derivative on the
  quantum scale.
- **Delta integral** — graininess-weighted on discrete scales, quadrature on `R`.
- **Hilger exponential** `e_p(t, s)` and the regressive `circle-plus` group; **linear
  dynamic equations** `y^Δ = p(t) y + q(t)`.
- **Bit-identical torch / jax twins** of the delta derivative.

!!! note "The founding collapse, generalized (`mu -> 0`)"
    The graininess `mu` measures how discrete a time scale is; `mu -> 0` recovers the
    continuum, and the delta derivative collapses to the closed-form derivative tower.
    This is the founding `delta -> 0` bias-collapse of `omnibias-difference`
    **generalized** to an arbitrary domain — a **distinct** limit from the `q -> 1`
    collapse of `omnibias-qcalculus` and the `beta -> inf` **temperature collapse** feasibility penalty elsewhere,
    never conflated.

Honesty labels: **closed-form** for the derivative on `R` (the tower) and the discrete
delta/nabla differences; **numerical** for the callable operators, quadrature, and Hilger
exponential on `R`. Smoke: `docs/examples/timescale_validate.py`.

## Public API

::: omnibias.timescale
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source

Status: Alpha (`0.1.0a1`).
