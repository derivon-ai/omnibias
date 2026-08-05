# omnibias-verify

**Status: Alpha (0.1.0a1).**

Certified neural-network verification built on the omnibias rigorous derivative
tower (`omnibias.core.verified`). Push a *rigorous enclosure* of an input box
through a trained network and read off a **theorem-grade** property certificate.

Two layers:

- **Exact pure-Python `_core`** (no torch / jax): a backend-neutral `Network`
  description and rigorous forward propagation. Smooth activations reuse the
  closed-form interval enclosures `tanh_iv` / `sigmoid_iv`; nonsmooth ones
  (ReLU, and GELU / max-pool in the propagation engine) get their own sound
  enclosures. Interval bound propagation (IBP) is the baseline; the
  Taylor-model engine and branch-and-bound (added on top) tighten it.
- **Backend frontends** (`omnibias.verify.torch`, `omnibias.verify.jax`):
  ingest a trained `nn.Sequential` / a JAX `(W, b)` stack into the neutral
  `Network`, so verification never touches the framework again and the two
  frontends agree bit-for-bit on equal weights.

## What this is and is not

- Every enclosure is **sound by construction**: the output box provably contains
  `net(x)` for *every* `x` in the input box. A robustness "certified" verdict is
  a proof; an "unknown" verdict is honest incompleteness, never an over-claim.
- It is **not complete**: branch-and-bound may exhaust its budget and return
  `UNKNOWN` rather than decide a borderline property.
- This is verification of a **fixed, trained** network — not training, not
  attack search.

## Public API (pure core)

```python
from omnibias.verify import (
    # network description
    Network, affine_layer,
    ReLULayer, TanhLayer, SigmoidLayer, GELULayer, MaxPoolLayer,
    # propagation: interval baseline, Taylor-model engine, branch-and-bound
    interval_propagate, BoundResult,
    taylor_propagate, taylor_output_bounds,
    scalar_readout_range, output_range, RangeResult,
    # property certificates
    certify_robustness, RobustnessCertificate,
    lipschitz_bound, interval_jacobian,
    monotonicity, MonotonicityCertificate,
    reachable_box,
)
```

Frontends: `omnibias.verify.torch.network_from_sequential`,
`omnibias.verify.jax.network_from_params`.

For scientific models exposing `_layer_specs()` (for example omnibias `JetMLP`),
`omnibias.verify.certify_pinn_aposteriori` extracts certified-jet layers, records
model provenance, evaluates a rigorous PDE residual, and returns a sealed
a-posteriori error certificate. See `docs/cookbook/proof-carrying-pde.md` and
`examples/proof_carrying_pde/`.

## Tighter than IBP, never looser

The Taylor-model engine keeps the polynomial shape in the input variables through
every layer, so correlated terms cancel instead of compounding (as they do in
interval bound propagation). Where a neuron's input range leaves the activation's
radius of convergence the engine falls back to the interval enclosure, and the
reported bound is always intersected with an IBP pass — so it is **provably never
looser than IBP**, and usually much tighter. Branch-and-bound subdivides the
input box to drive the enclosure to any tolerance.

## Tests

```bash
python -m pytest packages/omnibias-verify/tests -q
```

## License

Dual-licensed: AGPL-3.0-or-later OR a commercial licence from Derivon
(`LicenseRef-omnibias-Commercial`). See [`LICENSE`](LICENSE),
[`../../LICENSING.md`](../../LICENSING.md), and
[`../../COMMERCIAL-LICENSE.md`](../../COMMERCIAL-LICENSE.md). Contact
info@derivon.ai for commercial terms.
