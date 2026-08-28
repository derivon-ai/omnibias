# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""``DifferentiableEnvironment`` adapters for wrapped robotics physics engines
(theory 10-02, Phase 5).

omnibias contributes the optimizer (:mod:`omnibias.control.jax.policy` /
:mod:`omnibias.control.torch.policy`) and the certificates
(:mod:`omnibias.control.horizon`, :mod:`omnibias.control.certified.
gradient_bias`, :mod:`omnibias.control.certify`) -- **not** a rigid-body
physics engine. This module is a thin adapter layer: it wraps an existing
differentiable engine's state and step function behind the
:class:`omnibias.control.ocp.DifferentiableEnvironment` protocol so the
same trainers used on :mod:`omnibias.control.jax.envs` run unchanged on a
Brax or MuJoCo-MJX system.

Both engines are optional, heavy, compiled dependencies (Brax pulls in
JAX-native rigid-body dynamics; MuJoCo-MJX bundles MuJoCo's model
compiler). Neither is imported at module load time -- constructing an
adapter without the corresponding package installed raises a clear
:class:`ImportError` naming the missing extra;
:func:`brax_available` / :func:`mjx_available` let a caller check first
(the same "optional dependency degrades, never crashes at import time"
contract as :func:`omnibias.control.certify.certify_recoverable`, except
here there is no meaningful null adapter to degrade to -- a robotics
environment cannot function without its physics engine).

**Honesty note (gate G8).** Neither ``brax`` nor ``mujoco`` is installed
in the environment this program was developed in, so gate G8 (cost parity
vs SHAC on one hardware class, per
``theory/10-control/02-jet-adjoint-policy-optimization.md``) has **not**
been measured against a real robotics benchmark -- it is leftover-recorded,
exactly as the plan flagged it as "expected to be the hardest gate." The
adapters below are implemented against each engine's documented API and
exercised by a fake in-repo stand-in
(:class:`FakeArticulatedEnvironment`) so the *seam* itself is tested; a
:mod:`pytest.importorskip`-gated test exercises the real adapters if the
engine happens to be present.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

CostFn = Callable[[Any, Any], Any]


def brax_available() -> bool:
    """Whether the ``brax`` package can be imported."""
    try:
        import brax  # noqa: F401
    except ImportError:
        return False
    return True


def mjx_available() -> bool:
    """Whether ``mujoco.mjx`` can be imported."""
    try:
        from mujoco import mjx  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class BraxEnvironmentAdapter:
    r"""Wraps a ``brax.envs.Env`` behind :class:`omnibias.control.ocp.DifferentiableEnvironment`.

    ``env`` is an already-constructed Brax environment (e.g.
    ``brax.envs.get_environment("cartpole")``); ``cost_fn`` / ``terminal_cost_fn``
    map a Brax ``State`` (and, for ``cost_fn``, the action) to a scalar --
    Brax's own ``state.reward`` is a *reward*, and this seam is cost-minimizing,
    so the adapter does not assume a sign convention for you.
    """

    env: Any
    cost_fn: CostFn
    terminal_cost_fn: Callable[[Any], Any]
    state_dim: int
    action_dim: int

    def __post_init__(self) -> None:
        if not brax_available():
            raise ImportError(
                "BraxEnvironmentAdapter requires the 'brax' package; install it "
                "with `pip install brax` (or the 'robotics' extra once wired) "
                "before constructing this adapter. Check brax_available() first "
                "if you want to degrade instead of raising."
            )

    def step(self, y: Any, u: Any) -> Any:
        """``y`` is a Brax pipeline state's flattened observation-compatible array.

        Brax's native step operates on ``brax.base.State``, not a bare array;
        callers wanting the full Brax state should drive ``self.env`` directly
        and use this adapter only for the trainer-facing array view via a
        supplied ``qp_to_array`` / ``array_to_qp`` pair at construction time
        in a subclass, or via :func:`brax_state_environment` below.
        """
        raise NotImplementedError(
            "BraxEnvironmentAdapter.step is engine-specific; use "
            "brax_state_environment() to build a concrete array-in/array-out "
            "adapter around a specific Brax System."
        )

    def cost(self, y: Any, u: Any) -> Any:
        return self.cost_fn(y, u)

    def terminal_cost(self, y: Any) -> Any:
        return self.terminal_cost_fn(y)


def brax_state_environment(
    system: Any,
    *,
    cost_fn: CostFn,
    terminal_cost_fn: Callable[[Any], Any],
) -> Any:
    r"""Build a concrete :class:`omnibias.control.ocp.DifferentiableEnvironment` from a
    Brax ``System`` using ``brax.generalized.pipeline`` (or the ``spring`` /
    ``positional`` pipelines), operating directly on Brax's ``State.q`` /
    ``State.qd`` pytree via ``jax.tree_util.tree_flatten`` so the returned
    object's ``step`` matches the flat-array signature the rest of this
    package expects.

    Requires ``brax``; raises :class:`ImportError` immediately if absent.
    """
    if not brax_available():
        raise ImportError(
            "brax_state_environment requires the 'brax' package; install it "
            "with `pip install brax`."
        )
    import jax.numpy as jnp
    from brax.generalized import pipeline

    @dataclass(frozen=True)
    class _BraxFlatEnv:
        state_dim: int
        action_dim: int

        def _unflatten(self, y: Any) -> Any:
            n_q = system.q_size()
            q, qd = y[:n_q], y[n_q:]
            return pipeline.init(system, q, qd)

        def step(self, y: Any, u: Any) -> Any:
            state = self._unflatten(y)
            next_state = pipeline.step(system, state, u)
            return jnp.concatenate([next_state.q, next_state.qd])

        def cost(self, y: Any, u: Any) -> Any:
            return cost_fn(y, u)

        def terminal_cost(self, y: Any) -> Any:
            return terminal_cost_fn(y)

    n_q = system.q_size()
    n_qd = system.qd_size()
    n_u = system.act_size()
    return _BraxFlatEnv(state_dim=n_q + n_qd, action_dim=n_u)


@dataclass(frozen=True)
class MjxEnvironmentAdapter:
    r"""Wraps a ``mujoco.mjx`` ``(Model, Data)`` pair behind
    :class:`omnibias.control.ocp.DifferentiableEnvironment`.

    ``model`` is an ``mjx.Model`` (from ``mjx.put_model(mujoco.MjModel...)``);
    the adapter's flat state is ``concat(data.qpos, data.qvel)``, and the
    action is written to ``data.ctrl`` before one ``mjx.step``.
    """

    model: Any
    cost_fn: CostFn
    terminal_cost_fn: Callable[[Any], Any]

    def __post_init__(self) -> None:
        if not mjx_available():
            raise ImportError(
                "MjxEnvironmentAdapter requires 'mujoco' with MJX support; "
                "install a recent `mujoco` wheel (>=3.0) before constructing "
                "this adapter. Check mjx_available() first if you want to "
                "degrade instead of raising."
            )

    @property
    def state_dim(self) -> int:
        return int(self.model.nq + self.model.nv)

    @property
    def action_dim(self) -> int:
        return int(self.model.nu)

    def _unflatten(self, y: Any) -> Any:
        from mujoco import mjx

        n_q = self.model.nq
        data = mjx.make_data(self.model)
        return data.replace(qpos=y[:n_q], qvel=y[n_q:])

    def step(self, y: Any, u: Any) -> Any:
        import jax.numpy as jnp
        from mujoco import mjx

        data = self._unflatten(y).replace(ctrl=u)
        next_data = mjx.step(self.model, data)
        return jnp.concatenate([next_data.qpos, next_data.qvel])

    def cost(self, y: Any, u: Any) -> Any:
        return self.cost_fn(y, u)

    def terminal_cost(self, y: Any) -> Any:
        return self.terminal_cost_fn(y)


@dataclass(frozen=True)
class FakeArticulatedEnvironment:
    r"""A dependency-free stand-in for a wrapped robotics engine.

    Exercises the *adapter seam* (a flat-array ``DifferentiableEnvironment``
    wrapping an externally-owned stepping rule) without requiring Brax or
    MuJoCo-MJX to be installed. A 2-link double-pendulum-like linear
    surrogate: not a physics claim, purely a test fixture for
    ``test_robotics.py``.
    """

    state_dim: int = 4
    action_dim: int = 2
    dt: float = 0.02

    def step(self, y: Any, u: Any) -> Any:
        import jax.numpy as jnp

        n = self.state_dim // 2
        q, qd = y[:n], y[n:]
        qdd = -q + u
        qd_next = qd + self.dt * qdd
        q_next = q + self.dt * qd_next
        return jnp.concatenate([q_next, qd_next])

    def cost(self, y: Any, u: Any) -> Any:
        import jax.numpy as jnp

        return jnp.sum(y * y) + 0.1 * jnp.sum(u * u)

    def terminal_cost(self, y: Any) -> Any:
        import jax.numpy as jnp

        return 2.0 * jnp.sum(y * y)


__all__ = [
    "BraxEnvironmentAdapter",
    "FakeArticulatedEnvironment",
    "MjxEnvironmentAdapter",
    "brax_available",
    "brax_state_environment",
    "mjx_available",
]
