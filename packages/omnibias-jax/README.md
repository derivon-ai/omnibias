# omnibias-jax

JAX backend for the omnibias closed-form n-th derivative framework.

## Why this is fast

All numbers float64, identical answers to autodiff up to `≤ 10⁻¹⁵`. Full
derivation in [`docs/complexity.md`](../../docs/complexity.md).

- Closed-form Laplacian overhead is **`O(1)` in input dimension `D`** —
  independent of `D` because the reduction collapses the inner sum once.
- At `D = 240`, **68× faster than `jax.hessian` + trace** and **63× less
  memory**.
- Iterated Laplacian `Δᵏ` is **480× faster than folx-nested at `k = 3`**;
  folx-nested OOMs at `k = 4` while omnibias finishes in ~0.1 ms.
- Bit-identical to `omnibias-torch` and `omnibias-keras` for every
  `(activation, order)` pair.

## Install

```bash
pip install omnibias-jax
```

`omnibias-jax` depends on `omnibias-core` (pure-Python math) and `jax>=0.4.30`.

## What is in here

- The same closed-form derivative kernels as `omnibias-torch`, written in
  JAX (`jax.numpy`) so they JIT-compile cleanly inside FermiNet, vmc_jax,
  DeepQMC, and similar stacks. Polynomial coefficients are imported from
  `omnibias-core`, so a JAX `sigma^(n)(z)` is *bit-identical* to the
  torch `sigma^(n)(z)` for every `n` (this is the contract validated by
  `tests/test_cross_backend_parity.py`).
- A backend-specific activation dictionary registered via the same
  ``ActivationSpec`` protocol as torch (`get_activation`, `list_activations`,
  `register_activation`).
- ``neural_field_laplacian`` / ``neural_field_hessian`` / family: closed-form
  Laplacian and full Hessian for a one-layer scalar field
  ``f(x) = b + sum_h c_h sigma(W_h . x + b_h)`` on ``R^D``. These are the
  primitives the FermiNet bridge in `omnibias-ferminet` calls when composing
  through coordinate transformations.
- Born-Oppenheimer derivative kernels (`coulomb_potential`,
  `make_local_energy`, `make_bo_force`, `make_bo_hessian`,
  `vibrational_frequencies`) used to build analytic nuclear Hessians of
  neural-VMC energies.

## Public API

```python
from omnibias.jax import (
    JaxActivationSpec, get_activation, list_activations,
    register_activation, is_registered,
    neural_field_laplacian, neural_field_value_grad_hessian,
    coulomb_potential, make_local_energy,
    make_bo_force, make_bo_hessian, vibrational_frequencies,
    BankSpec, init_bias_scan, bias_scan, init_multipack, multipack_apply,
)
```

The FermiNet bridge (`folx`-compatible API, Tier-2 restricted FermiNet,
multiblock primitives) lives in the separate `omnibias-ferminet`
package; importing `omnibias.jax` does **not** trigger a FermiNet import,
so the JAX core remains useful when FermiNet is absent.

Gated Wave-1 twins (not shipped): `init_multipack` / `multipack_apply` (01-01)
and `init_bias_scan` / `bias_scan` / `BankSpec` (01-02). Same honesty as the
torch modules: interior shift along `w`, `gamma` is not `delta -> 0`.

Gated Wave-3 twins (not shipped): Scan-Net (`init_scan_net` / `scan_net_apply`;
on-lattice equivariance), Jet-KAN (`init_jet_kan` / `jet_kan_apply`;
model-jet exactness, KA theorem does not justify), Hermite ladder
(`hermite_basis` / `ladder_apply`; Rodrigues reweight required),
equivariant scan (gaussian-family steering; discrete `C_L`), and
`hierarchical_scan` (1-D offsets).

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
