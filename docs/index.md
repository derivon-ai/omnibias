# omnibias documentation

> **Numerically stable closed-form n-th derivative primitives for scientific
> machine learning** — one `sigma` call per order, bit-identical across
> PyTorch, JAX, and Keras 3.

!!! tip "AI agents start here"
    Read [`AGENTS.md`](https://github.com/derivon-ai/omnibias/blob/main/AGENTS.md)
    for the repository layout, the build / test / lint commands, the
    closed-form derivative-tower contract, and the do / don't lists. The
    [**Discovery & Calculus Handbook**](handbook/index.md) is the AI-friendly,
    book-length reference; the [AI quickstart](handbook/ai-quickstart.md) is
    the dense cheat-sheet. Runnable demos live in
    [`docs/examples/`](https://github.com/derivon-ai/omnibias/tree/main/docs/examples).

## Headline results

All numbers below are float64; methods agree to `≤ 10⁻¹⁵` — the wins are
bit-for-bit, not accuracy trades. **CPU numbers are regenerable from this
repo** ([`benchmarks/`](https://github.com/derivon-ai/omnibias/tree/main/benchmarks));
GPU numbers are an off-band tier. Full derivation in
[`complexity.md`](complexity.md); licensing in [`licensing.md`](licensing.md).

| Win | Number | Tier |
|---|---|---|
| Laplacian flat in `D` (CPU, `H=32`, `B=64`) | ~0.004 ms from `D=3` to `D=60` | CPU, reproducible |
| Same Laplacian vs `jax.hessian` / `torch.func.hessian` at `D=60` | **211× / 923×** faster | CPU, reproducible |
| Polylaplacian `Δ⁴` vs folx-nested / dense-nested | **4,660× / 181,000×** faster | CPU, reproducible |
| Laplacian at `D=240` (GPU, `H=256`, `B=4096`) | **68× / 199×** vs jax/torch; **63× / 108×** less memory | GPU, off-band |
| Cross-backend bit-identity (PyTorch / JAX / Keras 3) | float64-ULP-equal; 509 parity tests in `tests/` | CI |
| Closed-form derivative orders | unbounded `n` for the Riccati class | — |

This is the regime where omnibias decisively beats general-purpose autodiff:
high-order PDEs (4th-order biharmonic, Kuramoto–Sivashinsky, Cahn–Hilliard),
relativistic kinetic-energy corrections `Δᵏψ`, neural-VMC local kinetic
energies, and any pipeline that nests Laplacians more than once.

## Release tracks

omnibias ships **42 distributions** on two release tracks. Every package carries
its own version and `Development Status` classifier, ships tests, and has its own
CI job; the full grouped index is [`packages.md`](packages.md).

| Track | Packages | Use when |
| --- | --- | --- |
| **Curated public core** (8) | `omnibias-core`, `omnibias-torch`, `omnibias-jax`, `omnibias-ferminet`, `omnibias-fields`, `omnibias-pinn`, `omnibias-geometry`, `omnibias-keras` | You need the frozen public surface and the breaking-change policy in [`stability.md`](stability.md). |
| **Extended set** (34, Alpha) | The calculus registers, differentiable + certified optimization, verification / dynamics, and the learning primitives -- see [`packages.md`](packages.md) | You want a scientific register whose public surface may still shift between alpha releases. |

Track is a *release* decision, independent of maturity: `omnibias-keras` is Alpha
yet sits in the curated core, because the Keras 3 unified backend is part of the
published front door.

[Scope & guarantees](scope-and-guarantees.md) is the canonical page for
what is closed-form, what is autodiff-exact, what is numerical, and what the
certified stacks do (and don't) claim.

## Start here

- **[Discovery & Calculus Handbook](handbook/index.md)** — book-length,
  AI-friendly tour: 1-D neural jets, vector calculus & PDE discovery,
  differential geometry, exterior calculus, information theory, optimal
  transport, information geometry. Every function carries a
  *What · When · Theory · Example · Returns* card.
- **[Theory](theory.md)** — closed-form derivative towers, activation
  classification, and the Riccati / Eulerian / Hermite recurrences.
- **[Operator surface](operator-surface.md)** — canonical capability matrix:
  the six `OperatorBlock` roles (`identity / grad / laplacian / derivative /
  band / integral`, including the closed-form antiderivative `integral`), the
  three distinct senses of "integral", and honest closed-form / autodiff /
  numerical / certified labels. **Check here before claiming a capability.**
- **[Activations dictionary](activations.md)** — 23 + 3 real activations and
  their per-order support.
- **[Complexity](complexity.md)** — measured asymptotic wins with full
  derivation.
- **[Stability matrix](stability.md)** — per-symbol contract level.
- **[Cookbook](https://github.com/derivon-ai/omnibias/tree/main/docs/cookbook)**
  — PINN heat / Navier–Stokes, FermiNet kinetic energy, geometry on the
  sphere, proof-carrying PDE and fluid-dynamics certificates, gauge-theory
  primitives, and the certified fluid-dynamics walkthroughs
  (certified finite-model evidence, **not** proofs of global regularity).
  The four PINN capability gaps (causality, SDF geometry, parametric
  operators, spectral bias) are acceptance-gated; see
  [`benchmarks/pinn_four_gap_matrix.md`](benchmarks/pinn_four_gap_matrix.md).
- **[API reference](https://github.com/derivon-ai/omnibias/tree/main/docs/api)**
  — per-package autodoc.

## Core idea

Most frameworks compute higher derivatives by repeatedly applying automatic
differentiation. omnibias instead evaluates activation derivative towers
`sigma^(n)(z)` in **closed form**, sharing the same pure-Python recurrence
across PyTorch, JAX, and Keras 3. That gives stable Laplacians, Hessians,
high-order PDE operators, and curvature primitives at **one forward-pass
cost per order**, with **bit-identical** numerics across backends, without
making every downstream experiment part of the stable core.
