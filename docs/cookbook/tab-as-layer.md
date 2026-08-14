# Tab layers as neural heads

A tab SoftTree / Arrangement / Boosted model is a drop-in neural layer: build it
with `as_head(z, kind)` so it already lives on `z.device` / `z.dtype`, then
compose with any encoder. `forward` is tensor-in / tensor-out on `(..., d)` with
logits `(..., k)` (`k=1` for binary). Tabular `fit_*` trainers are optional
pretrain, not required.

Runnable examples:
[`docs/examples/tab_as_layer.py`](../examples/tab_as_layer.py) (PyTorch) and
[`docs/examples/tab_as_layer_jax.py`](../examples/tab_as_layer_jax.py) (JAX).

```python
import torch
from omnibias.tab.torch.plugin import as_head
from torch import nn

torch.manual_seed(0)
X = torch.randn(8, 4, dtype=torch.float64)
probe = torch.zeros(1, 8, dtype=X.dtype, device=X.device)
head = as_head(probe, "arrangement", n_hyperplanes=2)
enc = nn.Linear(4, 8).to(dtype=X.dtype)
logits = head(enc(X))
assert logits.shape == (8, 1)          # trailing class axis always present
assert head.W.dtype == X.dtype
```

The same probe builds a SoftTree or a boosted arrangement. Joint Adam on
`encoder.parameters()` plus `head.parameters()` updates both sides -- see the
PyTorch example.

```python
soft = as_head(probe, "softtree", n_trees=2, depth=1, task="binary", seed=0)
boosted = as_head(probe, "boosted", n_members=2, n_hyperplanes=1)
assert soft(enc(X)).shape == (8, 1)
assert boosted(enc(X)).shape == (8, 1)
```

## Keras: `learnable_beta` is member-`beta`

Keras 3 twins live in `omnibias.tab.keras` (not `omnibias-keras`) and use
`keras.ops`. `ArrangementBoosted(learnable_beta=True)` forwards trainable
`beta` to **members**; ensemble `learning_rate` / `base` stay frozen.
`ArrangementBoosted.build()` builds members so Keras 3 does not warn on first
call.

<!-- docs-test: skip reason="optional [keras] extra; member-beta is documented in docs/api/tab.md" -->
```python
from omnibias.tab.keras import ArrangementBoosted

layer = ArrangementBoosted(
    n_features=8, n_hyperplanes=2, n_members=2, learnable_beta=True
)
# layer.members[i].beta is trainable; layer.learning_rate / layer.base are not
```

## Optional Equinox wrappers

`omnibias.tab.jax.equinox_head` (`ArrangementHead` / `SoftTreeHead` /
`BoostedHead`) is an optional `[equinox]` extra so `import omnibias.tab.jax`
stays Equinox-free. Tab CI **fails** if the extra is missing when `CI` is set;
local runs still `importorskip`. CUDA AMP stays skip-if-no-CUDA.

<!-- docs-test: skip reason="optional [equinox] extra; tab CI requires it, local runs importorskip" -->
```python
from omnibias.tab.jax.equinox_head import ArrangementHead
```
