# Closed-form / differentiable convolution

A standard convolution layer *samples* its kernel: a Gaussian blur of width
`sigma` is approximated by reading `exp(-x^2/2 sigma^2)` at the integer taps and
re-normalising. That carries two errors -- point-sampling instead of pixel-area
integration (aliasing), and a fixed discrete `sigma` that cannot be learned with
a gradient. omnibias builds the taps in **closed form** instead, from the gaussian
activation and its shared derivative tower, so the kernel is area-exact,
anti-aliased, and differentiable in a *continuous* scale.

```python
import torch
from omnibias.torch.blocks import AnalyticGaussianConv2d

x = torch.randn(2, 3, 32, 32)              # (B, C, H, W)
blur = AnalyticGaussianConv2d(channels=3, kernel_size=7, sigma_init=1.5)  # learnable sigma
y = blur(x)                                # (B, 3, H, W) -> (B, 3, H, W), depthwise
```

The Keras twins (`AnalyticGaussianConv1D` / `AnalyticGaussianConv2D`) are
bit-identical in float64; the only difference is the channels-last layout.

## The theory: exact cell integration

Write the unit-area Gaussian as `G(x) = g(x/sigma) / (sigma sqrt(2 pi))` with
`g(u) = exp(-u^2/2)`. Tap `j` is **not** `G(j)` -- it is the integral of `G` over
the pixel cell `[j - 1/2, j + 1/2]`:

\[
w_j^{(0)} \;=\; \int_{j-1/2}^{j+1/2} G(x)\,dx
        \;=\; \tfrac12\!\left[\operatorname{erf}\frac{j+1/2}{\sigma\sqrt2}
                              - \operatorname{erf}\frac{j-1/2}{\sigma\sqrt2}\right].
\]

That `erf` is exactly the gaussian antiderivative `spec.integral` -- the same
`multibias_integral_window_forward` machinery used by
`OperatorBlock(op="integral", base="gaussian")`. The result is *area-exact*: there
is no sampling error in the kernel, and the taps integrate to one (up to the
truncated tail) for any real `sigma`.

For a **derivative-of-Gaussian** of order `n >= 1`, integrating the `n`-th
derivative over a cell telescopes to a difference of the `(n-1)`-th derivative,
which is the closed-form gaussian tower `g^(k) = (-1)^k He_k(u) g(u)`:

\[
w_j^{(n)} \;=\; \int_{j-1/2}^{j+1/2} G^{(n)}(x)\,dx
        \;=\; \frac{g^{(n-1)}(b_j) - g^{(n-1)}(a_j)}{\sigma^{n}\sqrt{2\pi}},
\qquad a_j=\tfrac{j-1/2}{\sigma},\; b_j=\tfrac{j+1/2}{\sigma}.
\]

So **derivative-of-Gaussian *is* the omnibias sigma tower** -- the same
`gaussian_nth_derivative` (probabilist's Hermite) coefficients that drive
`OperatorBlock(op="grad"|"laplacian", base="gaussian")`, now used to synthesise
convolution kernels rather than to act pointwise. Because `G * G` is again a
Gaussian (`sigma^2 = sigma_1^2 + sigma_2^2`) and `d/dx (G * f) = (dG/dx) * f`,
the blur, gradient and Laplacian responses are all one closed-form family indexed
by `derivative_order`.

## Building the taps directly

`analytic_gaussian_taps` is exported so you can inspect or reuse the kernel:

```python
import torch
from omnibias.torch.blocks import analytic_gaussian_taps

sigma = torch.tensor([1.0, 2.0])           # one scale per channel
taps0 = analytic_gaussian_taps(7, sigma, order=0)   # (2, 7) blur taps, rows sum ~ 1
taps1 = analytic_gaussian_taps(7, sigma, order=1)   # (2, 7) DoG taps, antisymmetric, sum ~ 0
```

`order=0` taps are symmetric and partition unity; `order=1` taps are
antisymmetric with zero mean (a derivative kills the DC component) -- both exact
consequences of the closed form, not numerical accidents.

## Learnable continuous scale

`sigma` is a real-valued parameter, so the network can *learn how much to blur*:
the gradient flows through `erf` / the Hermite tower analytically.

```python
from omnibias.torch.blocks import AnalyticGaussianConv1d

signal = torch.randn(4, 8, 64)             # (B, C, L)
target = torch.zeros_like(signal)

blur1d = AnalyticGaussianConv1d(channels=8, kernel_size=11, sigma_init=2.0)
loss = (blur1d(signal) - target).pow(2).mean()
loss.backward()
blur1d.sigma.grad         # exact d(output)/d(sigma), no finite differences
```

Set `learnable_sigma=False` to freeze the scale (the parameter becomes a buffer).
A 2-D Laplacian-of-Gaussian is the sum of a `(2, 0)` and a `(0, 2)` layer; a
gradient magnitude uses `(1, 0)` and `(0, 1)`.

## Relationship to `OperatorBlock` and `cmbConv*`

`cmbConv1d/2d` keep a *trained* `nn.Conv` kernel and post-compose an
`OperatorBlock` (an activation-tower transform of the response). The analytic
layers are the complementary idea: the **kernel itself** is the closed-form
gaussian / tower, with the learnable degree of freedom collapsed to the single
physical scale `sigma`. Use `cmbConv*` when you want a free learned filter bank;
use `AnalyticGaussianConv*` when you want an exact, anti-aliased, scale-steerable
Gaussian-family filter (scale-space, edge/blob detection, sampling-free
down/upstream filtering) trained end-to-end.
