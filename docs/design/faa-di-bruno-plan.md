# Faà di Bruno multi-layer jet composition — design

> Companion design note for the exact higher-order *directional jet* primitive
> added to `omnibias-core` / `omnibias-jax` / `omnibias-torch`. The live progress
> tracker is the plan file `faa_di_bruno_jets`; the math reference shipped with
> the code is
> [`packages/omnibias-core/FAA_DI_BRUNO_DERIVATIONS.md`](../../packages/omnibias-core/FAA_DI_BRUNO_DERIVATIONS.md).

## 1. Motivation

omnibias already computes the **exact** activation derivative tower
\(\sigma^{(k)}(z)\) in closed form (Eulerian / Legendre / Hermite recurrences in
[`packages/omnibias-core/src/omnibias/core/polynomials.py`](../../packages/omnibias-core/src/omnibias/core/polynomials.py)).
For a *single* layer the pre-activation \(z = Wx+\beta\) is affine in the input,
so every higher partial collapses to \(\sigma^{(n)}(z)\,W_j^{\,n}\) — no chain
rule beyond a power of the weight (see `nth_partial` in
[`one_layer.py`](../../packages/omnibias-pinn/src/omnibias/pinn/jax/fields/one_layer.py)).

The gap is **composition of two or more nonlinear layers**. The \(n\)-th
derivative of \(\sigma_L\circ A_L\circ\cdots\circ\sigma_1\circ A_1\) requires
Faà di Bruno's formula. Because omnibias supplies *exact* \(\sigma^{(k)}\), the
composition can be carried out *exactly* and *cheaply* by propagating a
truncated Taylor jet through the network — with no nested autodiff and no finite
differences, both of which lose precision rapidly at order \(\ge 4\).

## 2. Scope

In scope (this round):

- **Directional (1-D path) jets.** Restrict to a scalar parametrisation
  \(x(t) = x_0 + t\,v\). The output tower is
  \(\big(\tfrac{d^k}{dt^k} f(x_0+t v)\big|_{t=0}\big)_{k=0}^{N}\).
- Pure-Python combinatorics (Bell polynomials / Faà di Bruno coefficients) in
  `omnibias-core`.
- Backend jet ops (compose / affine / layer / mlp + conversions) as
  bit-identical jax and torch twins.

Non-goals (deferred):

- **Multivariate multi-index Faà di Bruno** (full higher-order tensors). The
  full Hessian / higher tensors are recoverable from directional jets by
  polarization; the dense multi-index machinery is heavier combinatorially and
  is parked for a later round.
- A tracked multi-layer `DeepMLP` field/module. We ship *primitives* plus
  demos; the existing one-layer fields and the curvature package stay unchanged.

## 3. Math

Represent a scalar function of the path parameter \(t\) by its Taylor
coefficients \(a_k = f^{(k)}(0)/k!\). A *jet* is the array
\(a = (a_0,\dots,a_N)\); we store the order along the leading axis and broadcast
over trailing axes (hidden units, output components, batch).

### 3.1 Affine layer

\(u = Wz + b\) is linear in \(t\), so jets transform per-order:

\[
  u^{[k]} = W\,z^{[k]}\ \ (k=0,\dots,N),\qquad u^{[0]} \mathrel{+}= b.
\]

### 3.2 Activation layer (the composition)

For \(b(t) = \sigma(u(t))\) with \(u^{[0]} = u(0)\), two equivalent kernels:

**Bell-polynomial form (oracle, exact rationals, pure Python).** With the
partial (incomplete) exponential Bell polynomials \(B_{n,k}\),

\[
  b^{(n)} \;=\; \sum_{k=1}^{n} \sigma^{(k)}\!\big(u^{[0]}\big)\,
    B_{n,k}\!\big(u^{(1)},u^{(2)},\dots,u^{(n-k+1)}\big),
\]

where \(u^{(j)} = j!\,u^{[j]}\) are the *derivatives* of \(u\) (not Taylor
coefficients). \(B_{n,k}\) is a sum over partitions of \(n\) into \(k\) parts.

**Shifted-power composition (backend kernel, stable, \(O(N^2)\)).** Let
\(w(t) = u(t) - u^{[0]}\) (so \(w^{[0]} = 0\)). Then

\[
  \sigma(u(t)) \;=\; \sum_{k=0}^{N}
    \frac{\sigma^{(k)}\!\big(u^{[0]}\big)}{k!}\, w(t)^k,
\]

truncated at order \(N\). Because \(w\) has no constant term, \(w^k\) starts at
order \(k\), so the sum is naturally triangular and only \(k\le N\) contribute.
Series powers are formed by truncated convolution
\((a\star b)[n] = \sum_{i=0}^{n} a[i]\,b[n-i]\). This is the production path; it
avoids enumerating partitions and the large integer Bell coefficients, and is
numerically stable in float64. The Bell form validates it.

### 3.3 Deep directional tower

Input jet for the path \(x_0 + t v\): \(x^{[0]}=x_0\), \(x^{[1]}=v\), \(x^{[k]}=0\)
for \(k\ge 2\). Apply (affine, activation) per layer; the final readout is a pure
affine (no activation). The result is the exact directional derivative tower of
the deep network.

### 3.4 Hessian via polarization (demo, not a shipped op)

The directional second derivative is \(v^\top H v\). The full Hessian follows
from \(O(d)\) (diagonal + mixed) or \(O(d^2)\) directional 2-jets via the
polarization identity \(2\,v^\top H w = (v+w)^\top H (v+w) - v^\top H v - w^\top H w\).
Shown in the notebook / benchmark, not added as a core op this round.

## 4. API

### `omnibias.core.bell` (pure Python, `@cache`, no torch/jax)

- `bell_partial(n, k) -> dict[tuple[int, ...], float]` — \(B_{n,k}\) as a map
  from a multiplicity signature to its (integer-valued) coefficient.
- `bell_complete(n) -> ...` — complete Bell polynomial \(B_n = \sum_k B_{n,k}\);
  evaluation at all-ones gives the Bell number.
- `faa_di_bruno_terms(n) -> list[...]` — the \((k, \text{partition}, \text{mult})\)
  decomposition used by the oracle and the derivations doc.

### `omnibias.jax.jet` and `omnibias.torch.jet` (identical signatures)

- `compose_jet(u_jet, sigma_tower)` — \(\sigma\circ u\) via shifted-power
  convolution. `u_jet` shape `(N+1, ...)`; `sigma_tower[k]` holds
  \(\sigma^{(k)}(u^{[0]})\) with matching trailing shape.
- `affine_jet(z_jet, W, b)` — per-order `W @` plus bias on order 0.
- `layer_jet(z_jet, W, b, spec, order)` — affine then activation, building the
  tower from `spec.fastpath`.
- `mlp_jet(x0, v, layers, order)` — full deep directional tower. `layers` is a
  sequence of `(W, b, spec_or_None)`; `spec=None` is a pure affine readout.
- `tower_to_jet(d)` / `jet_to_tower(a)` — \(k!\) scaling between the
  derivative-tower and Taylor-jet conventions. `jet_to_tower` matches the
  `jax.experimental.jet` series convention used in the oracle test.

### Order caps

The Riccati-class activations (`tanh`, `sigmoid`, `softplus`, `gaussian`,
`exp`, `sin`, `cos`, `sinh`, `cosh`) support every order. Bounded-order
activations (`silu`, `gelu`, `relu`, `mish` at \(n\ge2\); `arctan`, `log1pu2`,
`softabs`, `smooth_sign` at \(n\ge3\); `tan`, `cot`, `coth`, `sech`,
`log_cosh` at \(n\ge4\)) raise `NotImplementedError` in their fastpath. The
tower builder wraps that into a clear `ValueError` naming the activation and the
requested order so a too-deep jet on a bounded activation fails loudly.

## 5. File layout

| File | Role |
| --- | --- |
| `packages/omnibias-core/src/omnibias/core/bell.py` | Bell / Faà di Bruno combinatorics (new) |
| `packages/omnibias-core/src/omnibias/core/__init__.py` | export `bell_partial`, `bell_complete`, `faa_di_bruno_terms` |
| `packages/omnibias-jax/src/omnibias/jax/jet.py` | jax jet ops (new) |
| `packages/omnibias-torch/src/omnibias/torch/jet.py` | torch jet ops (new, twin) |
| `packages/omnibias-{jax,torch}/src/omnibias/{jax,torch}/__init__.py` | export jet ops; version bump |
| `packages/omnibias-core/FAA_DI_BRUNO_DERIVATIONS.md` | math reference |
| `notebooks/23_faa_di_bruno_jets.ipynb` + `docs/cookbook/faa-di-bruno-jets.md` | demo |
| `omnibias_experiments/benchmarks/faa_di_bruno/*` (submitted via that project's cluster wrapper) | off-band GPU benchmarks |

## 6. Validation matrix (float64)

| Check | Oracle | Tolerance |
| --- | --- | --- |
| Bell identities (`B_{n,1}=x_n`, `B_{n,n}=x_1^n`, \(\sum_k B_{n,k}(1,\dots)=\text{Bell}(n)\)) | hand values / Bell numbers | exact (rational/int) |
| single-layer reduction | \(\sigma^{(k)}(z)(w\cdot v)^k\) via existing fastpath | `rtol=1e-12` |
| deep MLP tower | `jax.experimental.jet` | `rtol=1e-10` |
| deep MLP tower | nested `jax.grad` / `torch.func.jvp`, \(k\le6\) | `rtol=1e-9` |
| Bell vs shifted-power | the two internal kernels | `rtol=1e-10` |
| cross-backend | jax vs torch | `rtol=1e-12` |
| golden regression | pinned `tests/data/faa_di_bruno_mlp_golden.npz` | `rtol=1e-12` |
| order-cap error | bounded activation beyond cap raises `ValueError` | n/a |

## 7. Heavy benchmarks → GPU cluster

Unit tests are light (small nets, CPU) and run in CI. The scaling/throughput
benchmarks live off-band under `benchmarks/faa_di_bruno/` in the separate
`omnibias_experiments` project and are submitted to a GPU node through that
project's cluster batch wrapper.

Planned (run as a follow-up): accuracy vs order/depth (exact jet vs nested-AD
vs finite-difference), jit/vmap throughput vs width and batch, and a deep
closed-form Hessian-vector demo via directional 2-jets + polarization.

## 8. Gates

Closed-form on torch + jax; analytic + oracle + cross-backend + pinned
regression tests in float64 with documented tolerances; `ruff check` clean on
new files; `mkdocs build --strict` clean; leakage grep clean; SPDX dual-license
header on every new source file; docstrings + `docs/api` + derivations +
cookbook + `CHANGELOG.md` `[Unreleased]`. No git commit until explicitly
requested.
