# Certified policy gradients (exact adjoint + a sound horizon + a sound bias bound)

`omnibias-control`'s adjoint stack trains a policy through a declared
`DifferentiableEnvironment` with an **exact** policy gradient (closed-form
`dpi/dy` from the jet tower, not a finite difference), then lets you certify
two things a short-horizon trainer otherwise picks by hand: how long the
training window needs to be, and how much bias that window costs. See
[omnibias.control adjoint](../api/adjoint_control.md) for the full reference,
and [omnibias-control](../api/control.md) for the pre-existing CBF-QP safety
filter this stack composes with.

## Train with the exact adjoint

`actor_adjoint_step` differentiates through the full closed-loop rollout via
the discrete adjoint (costate) recursion -- exact, not truncated:

```python
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from omnibias.control.jax.envs import DoubleGyrePointMass
from omnibias.control.jax.policy import actor_adjoint_step

env = DoubleGyrePointMass()
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
layers = [
    (0.1 * jax.random.normal(k1, (6, 2)), jnp.zeros(6), "tanh"),
    (0.1 * jax.random.normal(k2, (2, 6)), jnp.zeros(2), None),
]
y0 = jnp.array([0.3, 0.4])
train_horizon = 10
for _ in range(15):
    result = actor_adjoint_step(env, layers, y0, train_horizon, lr=0.05)
    layers = result.layers
print(float(result.total_cost))
```

This prints `7.5134...`, down from an initial rollout cost of `~26` -- these
15 steps at this fixed learning rate were chosen because they are the
**largest budget measured stable across 5 random policy inits**
(`benchmarks/actor_adjoint_control.py`); a larger budget tried first
diverged on one seed. `omnibias.control.torch.policy.actor_adjoint_step` is
the bit-identical twin.

## Certify the training horizon -- on the actual trained policy

The certificate composes the *real* per-step closed-loop Jacobians
`M_k = A_k + B_k dpi/dy_k` (assembled the same way
`actor_adjoint_gradient` does internally) into an interval matrix product,
then brackets its spectral radius:

```python
from omnibias.control.jax.adjoint import env_step_jacobians, policy_jacobian_dy, policy_forward
from omnibias.core.adjoint import closed_loop_jacobian
from omnibias.control.horizon import certified_horizon

y = y0
m_seq = []
for _ in range(train_horizon):
    u = policy_forward(layers, y)
    a, b = env_step_jacobians(env, y, u)
    dpi_dy = policy_jacobian_dy(layers, y)
    m_seq.append(closed_loop_jacobian(np.asarray(a), np.asarray(b), np.asarray(dpi_dy)))
    y = env.step(y, u)

real_result = certified_horizon(m_seq, radius=0.01, tol=0.5, max_horizon=train_horizon)
print(real_result.certified)
```

This prints `False`. That is the certificate working *correctly*, not a
bug: on this double-gyre task the closed loop near the target is only
**marginally** stable (eigenvalue magnitudes cluster near `1.0`, a few
just above), so no window in this trajectory actually contracts by the
declared `tol=0.5` within a `0.01` Jacobian ball. A sound certificate must
refuse here rather than fabricate a horizon -- see
[`docs/benchmarks/certified_horizon_smoke.json`](../benchmarks/certified_horizon_smoke.json)
for the >= 1000-trial coverage sweep on systems that *are* declared
contracting.

## Certify the training horizon -- on a system that does contract

For contrast, a genuinely contracting linear system gets a real certified
window:

```python
m_toy = [np.diag([0.5, 0.6])] * 20
toy_result = certified_horizon(m_toy, radius=0.01, tol=0.05)
print(toy_result.certified, toy_result.horizon)
```

This prints `True 7`: the smallest horizon whose enclosed monodromy has a
spectral-radius upper bound `<= 0.05`, one step more conservative than the
point-matrix reference (`ceil(log(0.05) / log(0.6)) = 6`) because the
enclosure is a sound *outer* bound around the declared `0.01` ball, not the
bare point trajectory.

## Bound what a short-horizon shortcut costs

If a trainer substitutes a learned terminal adjoint at truncation (PEARL-
style, `actor_adjoint_jet_step`), `truncation_bias_bound` sums the exact
per-step sensitivity terms into a sound bound on the resulting gradient
error, given a caller-supplied bound on the terminal substitution's error:

```python
from omnibias.control.certified.gradient_bias import truncation_bias_bound, terminal_adjoint_error_bound
from omnibias.verify import LinearLayer, Network
from omnibias.core.verified.interval import Interval

error_net = Network([LinearLayer(weight=((0.5, 0.0), (0.0, 0.5)), bias=(0.0, 0.0))])
box = [Interval(-0.1, 0.1), Interval(-0.1, 0.1)]
terminal_err = terminal_adjoint_error_bound(error_net, box, point_residual=0.01)
print(terminal_err)

dpi_dtheta_seq = [np.array([[0.3, 0.0]])] * 2
b_seq = [np.array([[1.0], [0.0]])] * 2
m_seq_toy = [np.diag([0.8, 0.8])] * 2
bias_report = truncation_bias_bound(dpi_dtheta_seq, b_seq, m_seq_toy, terminal_err)
print(bias_report.bound)
```

`terminal_err` is `0.06` (`= 0.01` measured residual at the box centre
`+ 0.5` Lipschitz constant `* 0.1` box radius), and `bias_report.bound` is
`0.0324`. Both numbers are only as sound as `error_net` genuinely modelling
`lambda_pred - lambda_true` -- a caller responsibility, not verified by this
function; state which case applies whenever you report the number.

## Assemble the proof-carrying bundle

`build_bundle` composes whichever certificates you have; a missing slot
downgrades the verdict, never upgrades it:

```python
from omnibias.control.bundle import build_bundle, summary

bundle = build_bundle(theta=None, gradient_bias=bias_report, horizon=toy_result, recoverable=None)
print(bundle.verdict)
print(summary(bundle))
```

This prints `partial` and
`ControllerBundle[partial]: gradient-bias<=0.0324, horizon=7 (model-relative; not certified safe control)`
-- the recoverable-set slot is empty here, so the bundle honestly declines
to claim `"certified"`, even though the two present certificates both hold.
Attach `omnibias.control.certify.certify_recoverable` (see the
[certified-safe-control cookbook](certified-safe-control.md)) to fill that
slot.

## Honest scope

- `actor_adjoint_step` finds a stationary point of the rollout cost, not a
  global optimum; no HJB equation is solved.
- `certified_horizon` is sound **only within the caller-declared** per-step
  Jacobian radius. It does not derive that radius from a Lipschitz bound
  automatically. The "certify on the actual trained policy" section above is
  the honest failure mode: a real closed loop can simply not be contracting
  enough for the declared tolerance, and the certificate says so instead of
  guessing.
- `truncation_bias_bound` is **conditional** on the supplied
  `terminal_error_bound` genuinely representing the true terminal-adjoint
  error; this module does not check that assumption.
- No PPO / TD3 / SHAC / PEARL reimplementation ships in this repository;
  `omnibias.control.jax.policy.zero_order_step` is a minimal model-free
  stand-in for comparison, not a drop-in replacement for those algorithms
  (see [`benchmarks/actor_adjoint_control.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/actor_adjoint_control.py)).
- Two founding limits appear across `omnibias-control` and must not be
  conflated: the CBF-QP safety filter sharpens with **temperature**
  (`beta -> inf`, feasibility); every `dpi/dy` in this cookbook is founding
  **bias collapse** (`delta -> 0`, the closed-form tower).
