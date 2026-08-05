# Certified-safe learned control (CBF-QP filter you can train through)

`omnibias-control` combines two things that safe-control tooling usually keeps apart:
a **differentiable** control-barrier-function (CBF) safety filter you can train a policy
*through*, and a **rigorous certificate** of the region where that filter is guaranteed
feasible. This cookbook builds the flagship end to end -- on dynamics that are *learned*
from data -- following
[`docs/examples/control_learned_lagrangian_cbf.py`](https://github.com/derivon-ai/omnibias/blob/main/docs/examples/control_learned_lagrangian_cbf.py).

## The safety filter

A CBF **safety filter** takes a nominal (task) action `a_nom` and returns the nearest
action that keeps the system inside a safe set:

\[
a^\star(s) = \arg\min_a \tfrac12\lVert a - a_{\text{nom}}\rVert^2
\quad\text{s.t.}\quad G(s)\,a \le h(s).
\]

For a disc obstacle `b(p) = ||p - c||^2 - r^2` under a point robot, the barrier has
**relative degree 2**, so we use a 2nd-order (exponential) CBF. With `psi0 = b`,
`psi1 = b_dot + a1 psi0`, the condition `psi1_dot + a2 psi1 >= 0` is *linear in the
acceleration* and becomes one row `G_obs(s) a <= h_obs(s)`; the actuator box
`||a||_inf <= a_max` adds `2 d` more rows.

`cbf_filter` solves this projection for a whole **batch** of per-sample polytopes
`(G(s), h(s))` at once, by accelerated (Nesterov) gradient descent on the hard-hinge
penalty `mu/2 sum relu(G a - h)^2` -- the temperature-collapse unit whose gradient is
`(a - a_nom) + mu G^T relu(G a - h)`. No pivoting, no factorization: it is one `jit`-ed
matmul loop, fully **differentiable**, with a bit-identical torch twin.

```python
import jax.numpy as jnp
from omnibias.control import CBFSpec, FilterSchedule
from omnibias.control.jax import cbf_filter, control_affine_cbf_rows

# Point robot: state x = [px, py, vx, vy], action = acceleration.
def f(x):
    return jnp.concatenate([x[2:], jnp.zeros(2)])

def g(x):
    return jnp.concatenate([jnp.zeros((2, 2)), jnp.eye(2)], axis=0)

def barrier(x):                                     # disc obstacle at (2, 2), radius 1
    return jnp.sum((x[:2] - jnp.array([2.0, 2.0])) ** 2) - 1.0

x = jnp.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.5, 0.5, 0.5]])   # (B, 4) states
a_nom = jnp.zeros((2, 2))                                      # (B, 2) nominal actions

spec = CBFSpec(gains=(2.0, 2.0), a_max=2.5)        # exponential CBF + actuator box
G, h = control_affine_cbf_rows(f, g, barrier, x, spec)   # autodiff Lie derivatives
a_safe = cbf_filter(a_nom, G, h, FilterSchedule())        # batched projection
```

## Learned dynamics: a Lagrangian Neural Network feeds the CBF

The builder is **generic control-affine** `x_dot = f(x) + g(x) a`, so the `f, g` may come
from a *learned* model. Here the robot has an unknown, coupled inertia `M_true` (pushing
in x also accelerates y). We recover it with a **Lagrangian Neural Network** on
`omnibias-variational` -- fit `L_theta(q, qdot) = 1/2 qdot^T M(theta) qdot` by the
inverse-dynamics residual `|| inverse_dynamics(L_theta; q,qdot,qddot,t) - tau ||^2` -- then
turn the learned Lagrangian into CBF rows with `g = M_hat^{-1} B`:

<!-- docs-test: skip reason="needs the trained Lagrangian from the linked runnable example" -->
```python
from omnibias.control.jax import lagrangian_cbf_rows
G, h = lagrangian_cbf_rows(learned_L, B, barrier, q, qdot, t, spec)
```

`lagrangian_cbf_rows` assembles `f = [qdot; M^{-1}F]`, `g = [0; M^{-1}B]` from
`omnibias.variational` (`mass_matrix`, `acceleration`) and defers to
`control_affine_cbf_rows`; for a constant learned mass it is *identical* to the analytic
`g = M_hat^{-1}B` rows (the example asserts this).

## Train the policy *through* the filter

`safe_rollout` closes the loop differentiably (`policy -> rows -> cbf_filter -> step`), so
the task loss backpropagates through the safety filter -- the policy learns to *anticipate*
it (steer around the obstacle) instead of stalling against it:

<!-- docs-test: skip reason="needs the policy / step / rows_fn pipeline from the linked runnable example" -->
```python
from omnibias.control.jax import safe_rollout, min_barrier
X, A, resid = safe_rollout(policy, step, rows_fn, x0, horizon=T, schedule=FilterSchedule.fast())
```

On 256 held-out starts under the *true* coupled dynamics (single seed of the example):

| method | collisions | success (reach) | min barrier |
|---|--:|--:|--:|
| nominal (no filter) | 0.11 | 1.00 | −0.56 (crashes) |
| filter@test (naive identity mass) | 0.00\* | 0.60 | +0.36 |
| filter@test (learned model) | 0.00 | 0.38 | +0.09 |
| **ours (trained through filter)** | **0.00** | **0.66** | +0.07 |

Every filtered method is collision-free on this seed; **ours** reaches the goal best
because it was trained through the filter. Across seeds (see the internal multi-seed
study) the CBF built on the *naive* identity-mass model can still collide under the true
coupled dynamics -- learning the dynamics is what makes the filter sound (\*this single
seed happened not to trigger it).

## The rigorous recoverable-set certificate

The obstacle CBF row is satisfiable by an actuator-admissible action iff the
recoverability margin

\[
\varphi(s) = h_{\text{obs}}(s) + a_{\max}\lVert G_{\text{obs}}(s)\rVert_1 \ge 0 .
\]

`certify_disc_recoverable` builds a sound interval extension of `phi` (closed form, no NN
propagation) and runs `omnibias-verify` interval branch-and-bound. A rigorous
`f_lower >= 0` proves the whole state box is recoverable; sweeping the speed gives a
**certified safe speed limit**:

```python
from omnibias.control import certify_disc_recoverable

# `g` is the learned M^{-1}B; omit it for the identity-mass case.
g_hat = jnp.array([[1.0, 0.25], [0.25, 1.0]])

cert = certify_disc_recoverable((0, 0), 1.0, (2.0, 2.0), 2.5,
                                [(-1.5, 1.5), (-2.5, -1.5)], vmax=1.0, g=g_hat)
# certified is True with f_lower > 0: v_max <= 1.0 is proved for this whole corridor.
cert.certified, cert.f_lower
```

Because the filter enforces exactly that CBF row, on the certified region the barrier
stays `>= 0` for **all** states -- forward invariance is a theorem, matching the zero
empirical collisions.

## Honest scope

- The certificate is **model-relative**: rigorous for the model matrix `g` you pass. For a
  *learned* model, pass the learned `g = M_hat^{-1}B` and report its empirical error
  separately (here `||M_hat - M_true|| ~ 1e-7`, so it transfers). It is *not* a robustness
  guarantee against arbitrary model mismatch.
- The CBF conservativeness is set by the design-chosen class-K `gains`; the certified safe
  speed is honest but not the true viability kernel.
- The exterior hard-hinge penalty leaves an `O(1/mu)` residual, so use the eval-quality
  `FilterSchedule()` (not `.fast()`) at deployment; the residual at eval is `~1e-3`.
