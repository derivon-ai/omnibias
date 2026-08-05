# Operator-block discovery

The OperatorBlock primitive carries an explicit `op` argument:

```python
from omnibias.torch import OperatorBlock

identity_block   = OperatorBlock(channels=8, op="identity", base="tanh")
grad_block       = OperatorBlock(channels=8, op="grad", base="sigmoid")
laplacian_block  = OperatorBlock(channels=8, op="laplacian", base="gaussian")
derivative_block = OperatorBlock(channels=8, op="derivative", base="tanh", derivative_order=4)
band_block       = OperatorBlock(channels=8, op="band", base="sigmoid")
integral_block   = OperatorBlock(channels=8, op="integral", base="sigmoid")
```

There are **six** operator roles (full capability matrix in the canonical
[operator-surface](../operator-surface.md) page):

- `identity` (K=1): the literal `sigma(z + b)` (Lemma identity).
- `grad` (K=2) / `laplacian` (K=3): closed-form `sigma'` / `sigma''` at the
  bias mean, from the bias-collapse fast path.
- `derivative` (K=n+1): closed-form `sigma^(n)` at arbitrary order `n`
  (`grad` / `laplacian` are the `n = 1, 2` aliases).
- `band` (K=2): the literal window difference `sigma(z + b_hi) - sigma(z + b_lo)`.
- `integral` (K=2): the **closed-form antiderivative** window
  `S(z + b_hi) - S(z + b_lo)` with `S' = sigma` -- the fundamental-theorem twin
  of the derivative tower, not a difference of `sigma` values.

The `derivative` / `grad` / `laplacian` / `integral` roles correspond to the
canonical use of the underlying multi-bias unit in the closed-form collapse
limit. Every base activation in the omnibias dictionary has an `operator_role`
field documenting which roles it is canonical for (see
[`docs/stability.md`](../stability.md)); `grad` / `laplacian` / `derivative`
require a `fastpath` kernel and `integral` requires an antiderivative kernel.

This makes the framework *typed* in a Bayesian-prior sense: a softmax
over operator roles is a typed object you can train as part of the
model. That idea is realized in two places:

- [`omnibias-symbolic`](../api/symbolic.md) (Alpha) ships the library-free
  SINDy variant `NeuralJetDiscoverer`, which recovers closed-form
  differential identities from exact omnibias jets.
- [`omnibias.torch.architectures`](../api/torch.md) ships the learnable
  `fit_joint_operator_regressor` / `JointOperatorRegressor`
  (`omnibias.torch.architectures.joint_operator`), which gates a typed
  operator dictionary end-to-end (see notebooks 13 and 18).

For a worked through-the-stack example, see the multi-bias paper's
Section 4 or open an issue on the GitHub repository.
