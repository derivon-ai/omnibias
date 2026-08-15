# omnibias-struct

**Status: Alpha (0.1.0a1).**

Certified **differentiable dynamic programming** on the omnibias tower: soft
**Viterbi**, **shortest-path**, and **CTC** layers, on **PyTorch** and **JAX**
(bit-identical twins, float64).

The whole package rests on keeping two limits apart -- getting this wrong is the
contagious mistake, so it is stated up front and locked by a test:

- **`beta -> inf` (temperature / relaxation).** The hard `max` combine of the DP
  recursion is replaced by the smooth log-sum-exp
  `lse_beta(a) = beta^-1 log sum_i exp(beta a_i)`. Since `lse_beta >= max` and
  `lse_beta -> max` as `beta -> inf`, the soft DP anneals to exact hard
  Viterbi / shortest-path / CTC. This is the *feasibility / temperature* sense of
  "collapse" (a soft object hardened to a discrete one), **not** bias collapse.
- **`delta -> 0` (the founding bias-collapse tower): the exact differentiation
  engine.** The soft combine is differentiated **exactly** by the closed-form
  derivative tower, never by conflating it with the annealing. Pairwise,
  `lse_beta(a, b) = a + beta^-1 softplus(beta (b - a))`, and `softplus` is Riccati
  with the closed-form tower `softplus^(n) = sigma^(n-1)` shipped in
  `omnibias.core`. Its beta-tempered tower propagated through `compose_jet`
  (`omnibias.{torch,jax}.jet`) gives the closed-form log-sum-exp / softmax jets;
  the first-order sensitivity is the softmax marginal, and the soft-DP gradient is
  the forward-backward path marginal assembled from it.

## The honest object (yes-if)

Exact hard DP is not differentiable (its `argmax` gradient is a.e. zero). The
sound differentiable object is a **relaxation + a certified gap**, never an
exactness claim:

| Function | Meaning |
|----------|---------|
| `lse_beta >= max` | the soft value is an upper bound on the hard optimum |
| `lse_beta <= max + log(N) / beta` | closed-form gap (`N` = exact path count) |

so `V* <= V_beta <= V* + log(N)/beta` is a **certified sandwich** that shrinks as
`beta` grows, self-checked against the exact brute-force optimum on small
instances.

## Quick start

```python
import torch
from omnibias.struct.torch import soft_viterbi, soft_viterbi_marginals
from omnibias.struct import ChainTrellis, viterbi, certify_soft_dp, count_paths

emissions = torch.randn(5, 3, dtype=torch.float64)   # (T=5 steps, S=3 states)
transitions = torch.randn(3, 3, dtype=torch.float64)

soft = soft_viterbi(emissions, transitions, beta=8.0)      # differentiable scalar
gamma = soft_viterbi_marginals(emissions, transitions, beta=8.0)  # (T, S) marginals

# certify the relaxation against the exact hard optimum
trellis = ChainTrellis(emissions.numpy(), transitions.numpy())
hard, _path = viterbi(trellis)
cert = certify_soft_dp(hard, float(soft), count_paths(trellis), beta=8.0)
assert cert.is_sound            # V* <= V_beta <= V* + log(N)/beta
```

```python
import jax.numpy as jnp
from omnibias.struct.jax import soft_viterbi   # bit-identical twin
soft = soft_viterbi(jnp.asarray(emissions.numpy()), jnp.asarray(transitions.numpy()), beta=8.0)
```

## Install

```bash
pip install omnibias-struct[torch]   # or [jax], [all]
```

## Public API

- Backend-agnostic (`omnibias.struct`): `ChainTrellis`, `DAG`, `CTCLattice`,
  `count_paths`; hard DP `viterbi`, `shortest_path`, `ctc_best` and the
  brute-force oracles `brute_force_viterbi`, `brute_force_shortest_path`,
  `brute_force_ctc`; `DPGapCertificate`, `certify_soft_dp`, `logsumexp_gap_bound`.
- Backends (`omnibias.struct.torch` / `omnibias.struct.jax`): `logsumexp_beta`,
  `softmax_beta`, `logsumexp_beta_jacobian`, `logsumexp_beta_hessian`,
  `pairwise_lse`, `pairwise_lse_jet`; `soft_viterbi`, `soft_viterbi_marginals`,
  `soft_shortest_path`, `soft_shortest_path_marginals`, `soft_ctc`.
- Gated tropical homotopy (`omnibias.struct._core.tropical`): reuses
  `logsumexp_gap_bound`; `beta -> inf` is temperature collapse; large
  `(n, D)` refused.

## Tests

```bash
python -m pytest packages/omnibias-struct/tests -q
```

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
