---
name: omnibias-verified-primitive
description: Add or modify a rigorous primitive in omnibias.core.verified (Interval, affine zonotopes, TaylorModel(MV), enclosure engines, eig / eig_operator / invariant_subspace / dirichlet / lohner / kantorovich / frobenius). Use when writing sound, outward-rounded numerics that enclose the true value.
---

# Writing a rigorous primitive

The rigorous register trades speed for certainty: an enclosure contains the
true object. A silently-too-tight bound is a correctness bug.

## Why nested AD fails

Autodiff returns a point. Interval arithmetic bolted onto a float graph
without outward rounding is unsound. Generic ODE/eigenvalue libraries do not
ship QR-Lohner, radii polynomials, or Lehmann-Maehly-Goerisch lower bounds
on the same jet that the differentiable register uses.

## What only this tower unlocks

Pure-Python `Interval` (outward-rounded), affine zonotopes, TaylorModel(MV),
sequence-space tails, Kantorovich, Lohner, eig_operator, dirichlet. Every
enclosure contains a dense deterministic grid **and** a random sample of true
values. This is the shared substrate for verify, dynamics, sos, and the Lean
bridge.

## Use

- Home: `omnibias.core.verified`. Pure Python; never imports a backend.
- `kantorovich.radii_polynomial_certificate(Y0, Z0, Z1, Z2)` certifies a
  unique zero in an explicit ball. `radii_spectral` / `radii_series` assemble
  that shape for Fourier / one-sided series. `BandedLinearPart` /
  `tail_inverse_bound_from_banded` is the additive banded path.
- Laguerre sibling: `laguerre_basis`. Cohn-Elkies 1-D: `cohn_elkies_hermite_bound`.
  Named de Bruijn–Newman attempt: `attempt_named_lambda_bound`. Hubbard GS
  sandwich: `hubbard_half_filled_ground_state`.
- Lohner / TM validated flow: `lohner`, `jet_flow`.
- Eigenvalue *lower* bounds: `eig_operator` (Lehmann-Maehly-Goerisch).
- Dirichlet / zeta / L / Jacobi-theta on `Re(s) > 1`: `dirichlet`.
- Enclosure Collapse is `width -> 0` of a *sound enclosure* (a point plus a proof): `enclosure_collapse`.

Contain a grid and a random sample. Exhausted search returns the unresolved
region. `theorem_prover_verified` / `mathlib_verified` after a real
`lake build` — pair with `omnibias-certificate-lean` / `omnibias-formal-agent`.

## Extend

```bash
python -m pytest packages/omnibias-core/tests -q -k verified
```

Compose with `omnibias-verify`, `omnibias-dynamics`, `omnibias-sos`,
`omnibias-formal`. New primitives stay in `omnibias.core.verified` unless they
earn a package (`omnibias-new-package`).

## Next invention

A new enclosure engine (e.g. a banded Lohner step or a tighter Dirichlet
rectangle) whose grid-and-random test passes and whose finite rational core
seals on the Lean kernel.

## Further references

- `docs/api/enclosure_collapse.md`
- `docs/scope-and-guarantees.md`
