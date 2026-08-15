# omnibias-partition

**Status: Alpha (0.1.0a1).**

A light, **certified soft partition-of-unity** primitive -- the keystone shared by four
downstream bridges (discontinuity-capturing PINNs, region-wise Riemannian atlases,
per-region symbolic law discovery, and certified decision layers).

A hard partition of `R^d` into regions is a set of indicator functions `1[region l]`.
omnibias makes it a **soft partition** built from oblique split gates
`g(x) = sigmoid(beta·(w·x − t))`: `depth` gates route an input into `2**depth` regions with
weights `w_l(x)` that are

- **non-negative** and **sum to one** for every `x` (a genuine partition of unity), and
- **harden** to a crisp `{0,1}` partition as `beta → ∞`.

On top of the weights it ships:

1. **`partition_weights`** -- the numpy reference plus **bit-identical torch / jax twins**
   (parity `~1e-9`, float64) and a **keras.ops** twin (`[keras]` extra), so the same
   partition trains under any backend.
2. **`hard_assignment` / `hardened_rules`** -- the crisp region index and the exported
   human-readable `if w·x > t` boundaries (`axis` mode gives single-feature rules).
3. **A sound certificate** (`certify_partition_gap`): an outward-rounded
   [`Interval`](https://omnibias.ai/api/core/) enclosure of the soft→hard
   membership gap, plus the closed-form `log(n_regions)/beta` Gibbs bound -- a well-posed
   **yes-if** object (bounds hold; the *optimal* hard partition is not claimed).
4. **`RegionModels`** -- a per-region model registry whose single
   `combine(X, beta, region_outputs) = Σ_l w_l · out_l` engine is what every bridge calls.

`split_kind ∈ {"oblique", "axis", "sparse"}`: axis-aligned and L1-sparse splits are the
interpretable / heterogeneous-robust lever, available from day one.

Gated arrangement geometry (`omnibias.partition.arrangement`) is the many-normal
generalisation: the binary tree is the special case that agrees with
`partition_weights`. Cell membership is temperature collapse; sampling is a
subgraph, never a complete face lattice.

Terminology: the gate's `beta → ∞` hardening is **temperature collapse**, the
feasibility sense (a soft indicator becoming a 0/1 step) -- **not** the
founding bias collapse (the multi-bias `delta → 0` limit to the closed-form derivative
`sigma^(K-1)`).

## Install

```bash
pip install -e packages/omnibias-partition            # numpy core + certificate
pip install -e "packages/omnibias-partition[torch]"   # + torch weight twin
pip install -e "packages/omnibias-partition[jax]"      # + jax weight twin
pip install -e "packages/omnibias-partition[keras]"    # + keras.ops weight twin
```

## Scope / honesty

- The soft→hard membership gap is **sound** (outward-rounded intervals; a looser bound only
  widens the certified gap). The partition parameters themselves are trained by the
  downstream bridges (autodiff); partition only provides the primitive + certificate.
- Products of sigmoids are differentiated by autodiff in the bridges, **not** the closed-form
  derivative tower (the "closed-form" brand does not auto-extend to products).

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
