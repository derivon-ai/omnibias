# CmbNet on MNIST

`omnibias.torch.architectures.CmbNet` is a small MNIST classifier that
uses `cmbConv2d` (drop-in for `nn.Conv2d` with an inline OperatorBlock)
in place of `Conv2d -> ReLU` blocks.

Runnable example:
[`packages/omnibias-torch/examples/cmbnet_mnist.py`](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-torch/examples/cmbnet_mnist.py).

```python
import torch
from omnibias.torch.architectures import CmbNet

net = CmbNet(in_channels=1, num_classes=10, width=(16, 32, 64))
batch = torch.randn(8, 1, 28, 28)
logits = net(batch)  # (8, 10)
```

CmbNet's classification accuracy on MNIST is competitive with a
size-matched ReLU CNN; the win is that every block is a *typed*
operator (gradient / Laplacian / integral / identity) so the network
is interpretable as a learned-coefficient differential operator, not a
black box.
