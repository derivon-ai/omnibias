# Train-then-certify: a sealed network certificate

You trained a small network. Gradient descent (Adam, L-BFGS, our curvature-aware
optimizers) gives you a *good* set of weights — but no **proof** about the trained
model's behaviour over an input region. `certify_trained_network` closes that gap:
it rigorously encloses the network's minimum over an input box and **seals** the
result into a tamper-evident certificate.

```python
from omnibias.verify import certify_trained_network
from omnibias.core.proof.certificate import decode_interval

# `net` is any trained JetMLP-like model (exposes `_layer_specs()`), or a raw
# `(W, b, name)` layer list. Here: u(x, y) = -exp(-x^2/2) - exp(-y^2/2), min -2.
net = [
    ([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], "gaussian"),
    ([[-1.0, -1.0]], [0.0], None),
]
box = [(-2.0, 2.0), (-2.0, 2.0)]

nc = certify_trained_network(net, box, tol=1e-3, flatness=True)

assert nc.verified                      # recomputed digest matches the sealed body
assert nc.converged                     # certified gap f_upper - f_lower <= tol
enc = decode_interval(nc.certificate["payload"]["interval"])
assert enc.lo <= -2.0 <= enc.hi         # proved enclosure of the global minimum
```

## What the certificate carries

`certify_trained_network` returns a `NetworkCertificate` that bundles:

- **`result`** — the rigorous `GlobalMinResult`: `f_lower <= min_{x in box} net(x) <= f_upper`
  holds **unconditionally**, computed by the interval branch-and-bound driven by the
  closed-form verified jet (value, exact gradient, and — with `flatness`/`strict_local_min`
  — Hessian);
- **`certificate`** — the sealed v1 certificate (canonical, hash-sealed JSON). Its `meta`
  records the ingested-weight digest (via `verified_layer_bundle`), the input box, the
  argmin, the certified gap, the honest `converged` flag, and any bundled flatness;
- **`flatness`** (optional) — a certified enclosure of the extreme Hessian eigenvalues over
  the box, the exact-curvature basin-sharpness read-out;
- **`strict_local_min`** (optional) — the interval `LDLᵀ` inertia certificate that the
  Hessian is positive definite over the box.

The two properties you actually check:

- **`nc.verified`** recomputes the certificate digest, so any post-hoc edit to a bound (or
  to a `meta` field) is detected — a forged, tighter `f_lower` breaks the seal;
- **`nc.converged`** reports honestly whether the certified gap reached `tol`. Under a
  starved `max_boxes` budget it returns `False` (and `honesty["global_min_certified"]` is
  `False`) — the enclosure is *still* sound, it just isn't tight yet.

## Certified output envelope

Because the bridge takes the network itself, you can certify the minimum directly and the
maximum by minimising the negated readout, giving a *proof* envelope `[lo, hi]` the trained
model's output cannot leave anywhere in the region:

```python
lo = certify_trained_network(net, box, tol=1e-2).result.f_lower
neg = [*net[:-1], ([[-v for v in row] for row in net[-1][0]],
                   None if net[-1][1] is None else [-v for v in net[-1][1]], net[-1][2])]
hi = -certify_trained_network(neg, box, tol=1e-2).result.f_lower
# net(x) in [lo, hi] for every x in box -- unconditionally.
```

## Runnable demo

```bash
python examples/train_then_certify.py
```

The demo trains a small `tanh` `JetMLP` for a few Adam steps (falling back to a fixed
analytic net if `omnibias-torch` is not installed), certifies its read-out over an input
box, prints the sealed enclosure + certified curvature, and confirms the digest.

!!! warning "Honest scope"
    This inherits the interval branch-and-bound curse of dimensionality **and** interval
    dependency overestimation that grows with box width and network depth. It is for
    **small** networks over **low-dimensional** input boxes with `tanh` / `sigmoid` /
    `gaussian` activations — a certified read-out over an input region, **not**
    million-parameter training and **not** a continuum / global-regularity-grade statement.
