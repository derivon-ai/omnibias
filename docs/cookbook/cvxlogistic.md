# Convex layer: logistic regression

`omnibias.torch.architectures.CvxLogistic` solves binary logistic
regression as a depth-`T` *unrolled* gradient descent. Its sigmoid link is
the `K = 2` bias collapse of `softplus`, evaluated through an
`OperatorBlock` with the `"grad"` operator role. The whole solve is
differentiable, so the per-step sizes are learnable parameters.

`forward(X, y)` runs the solver on the batch and returns the fitted weight
vector; `predict_proba(X, w)` turns those weights back into probabilities.

```python
import torch
from omnibias.torch.architectures import CvxLogistic

# Cookbook snippets pin float64 so they stay independent of process-wide
# ``torch.set_default_dtype`` changes left by other docs examples.
torch.set_default_dtype(torch.float64)

clf = CvxLogistic(n_features=64, T=20)
x = torch.randn(32, 64)
y = (x.sum(dim=-1) > 0).to(dtype=x.dtype)

w = clf(x, y)                    # (64,) fitted weights
probs = clf.predict_proba(x, w)  # (32,) probabilities
loss = torch.nn.functional.binary_cross_entropy(probs, y)
```

Because the OperatorBlock is parameterised by an `ActivationSpec` and
the Riccati-class sigmoid has *closed-form derivatives at every order*,
the closed-form Hessian gives an exact diagonal-block Fisher
information without autograd. `omnibias-curvature` exposes that surface
generally (see [curvature](../api/curvature.md)); `CvxLogistic` is the
demonstration scaled to a single layer.

See [`packages/omnibias-torch/examples/cvxlayer_lasso.py`](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-torch/examples/cvxlayer_lasso.py)
for the convex Lasso variant.
