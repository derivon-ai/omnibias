# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Inverse problems (torch): recover PDE coefficients from observations.

A forward solve fits a field to a *known* PDE. An inverse solve fits the field
**and** the unknown coefficients at once, to a three-term objective

``mean(r_pde^2) + condition_weight * mean(r_bc/ic^2) + data_weight * mean(r_data^2)``

where ``r_data`` is the misfit against measured solution values. That third term
is what makes the coefficients identifiable: the PDE residual alone is satisfied
by many ``(field, coefficient)`` pairs, so with no data the recovered value would
be wherever the optimiser drifted.

The gradient reaches a coefficient because
:class:`~omnibias.pinn.solver._core.unknowns.Unknown` resolves through a binding
that the driver fills with live tensors, and each coefficient is parameterised by
an *unconstrained* raw variable through its transform -- so a positivity or box
constraint holds by construction with no projection step and no clipping.

Everything else is shared with :func:`~omnibias.pinn.solver.torch.solve_optimize`:
the exact-curvature optimisers, residual-adaptive refinement and gradient-norm
balancing all work here because both drivers run the *same* loop, with the
unknowns simply joining the trainable parameter vector.

Honest scope
------------
* **torch-only**, like the rest of the second-order solver surface. There is no
  ``jax`` twin (``omnibias.pinn.solver.jax`` has no optimisation driver at all).
* Identifiability is the caller's responsibility, and the failure is silent by
  nature. A coefficient the data cannot see is not recovered -- it merely stops
  moving. Two known structural limits are documented per builder: ``wave`` sees
  only ``speed ** 2`` (the sign is unidentifiable), and a coefficient multiplying
  a term that vanishes on the observed region (a diffusivity where the solution is
  linear, say) has no gradient. Check ``InverseSolution.recovered`` against a
  synthetic study before trusting it on real data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import torch
from omnibias.pinn.solver._core.hard import plan_hard_conditions
from omnibias.pinn.solver._core.observations import Observations, check_observations
from omnibias.pinn.solver._core.sampling import CollocationSpec, RefinementSpec
from omnibias.pinn.solver._core.system import System
from omnibias.pinn.solver._core.unknowns import bind_unknowns
from omnibias.pinn.solver.torch._solution import FieldSolution
from omnibias.pinn.solver.torch.assemble import residual_norm, to_tensor
from omnibias.pinn.solver.torch.fields import build_field
from omnibias.pinn.solver.torch.steady import (
    _check_optimizer,
    _hard_diagnostics,
    _optimize,
    _Unknowns,
)


@dataclass
class InverseSolution(FieldSolution):
    """A fitted field plus the coefficients recovered alongside it.

    ``recovered`` maps each :class:`~omnibias.pinn.solver._core.unknowns.Unknown`
    name to its final value in *physical* units (the transform is already applied),
    and ``data_misfit`` is the RMS of the observation residual, which is the number
    to look at when deciding whether the fit is worth believing.
    """

    recovered: dict[str, float] = dataclass_field(default_factory=dict)
    data_misfit: float = 0.0

    def __getitem__(self, name: str) -> float:
        return self.recovered[name]


def solve_inverse(
    system: System,
    observations: Sequence[Observations],
    *,
    hidden: int = 64,
    activation: str = "tanh",
    weight_init_scale: float | None = None,
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
    collocation: CollocationSpec | None = None,
    optimizer: str = "cubic_gauss_newton",
    iters: int = 200,
    lr: float = 1.0,
    adam_iters: int = 500,
    adam_lr: float = 1e-2,
    condition_weight: float = 10.0,
    data_weight: float = 10.0,
    loss_balancing: str = "none",
    balance_every: int = 10,
    balance_alpha: float = 0.9,
    refinement: RefinementSpec | None = None,
    optimizer_kwargs: dict[str, Any] | None = None,
    hard_conditions: str = "none",
) -> InverseSolution:
    """Recover ``system.unknowns`` from ``observations`` while fitting the field.

    Parameters
    ----------
    system
        A system built with at least one
        :class:`~omnibias.pinn.solver._core.unknowns.Unknown` coefficient. Passing a
        purely forward system is an error rather than a no-op, because it almost
        always means the caller forgot to wrap a coefficient.
    observations
        Measured solution values (see
        :class:`~omnibias.pinn.solver._core.observations.Observations`). Use
        :func:`~omnibias.pinn.solver._core.observations.sample_observations` to build
        synthetic ones from a forward solve.
    optimizer
        Defaults to ``"cubic_gauss_newton"`` rather than the forward driver's
        ``"lbfgs"``, and the gap is not marginal. Recovering a known coefficient
        from a 3x-wrong initial guess at a small budget (see
        ``docs/benchmarks.md``): Gauss-Newton lands within 0.02% (heat), 0.07%
        (wave) and 2.6% (Burgers) of the truth, where L-BFGS at the same budget
        reaches 17%, 155% and 72% -- it barely moves the coefficient at all. A
        single scalar coefficient has curvature on a completely different scale
        from the network weights, and a method with one shared step size cannot
        serve both; the exact Gauss-Newton metric rescales each direction by its
        own curvature. Any member of
        :data:`~omnibias.pinn.solver.torch.OPTIMIZERS` is still accepted.
    data_weight
        Weight on the mean-squared data misfit. The default deliberately matches
        ``condition_weight``: the observations are as hard a constraint as the
        boundary data, and under-weighting them is the usual reason a recovery
        stalls at the initial guess.
    adam_iters
        A longer warmup than the forward driver's, because the coefficient starts
        at a guess: the second-order optimisers converge fast but need the field to
        be roughly right before the coefficient gradient is meaningful.

    The remaining arguments have exactly the meaning they do in
    :func:`~omnibias.pinn.solver.torch.solve_optimize`, including the full
    exact-curvature ``optimizer`` set, ``loss_balancing="grad_norm"`` (which now
    balances three terms) and residual-adaptive ``refinement``.
    """
    if not system.unknowns:
        raise ValueError(
            "solve_inverse needs at least one Unknown coefficient; this system is "
            "fully specified, so use solve_optimize (or wrap a coefficient in "
            "Unknown(...) to recover it)"
        )
    _check_optimizer(optimizer, loss_balancing, balance_every)
    obs = check_observations(
        observations,
        components=system.component_names(),
        ndim=system.domain.ndim,
    )
    spec = collocation or CollocationSpec()
    hard = plan_hard_conditions(system, mode=hard_conditions)
    field = build_field(
        system,
        hidden=hidden,
        activation=activation,
        weight_init_scale=weight_init_scale,
        dtype=dtype,
        seed=seed,
        hard_conditions=hard,
    )
    unknowns = _Unknowns(system.unknowns, dtype=dtype)
    blocks = [
        (
            obs_i.component,
            to_tensor(obs_i.coords, field),
            to_tensor(obs_i.values, field),
            float(obs_i.weight),
        )
        for obs_i in obs
    ]

    def misfit(current: Any) -> torch.Tensor:
        """Per-observation residual, each block scaled by its own ``weight``.

        The scaling is ``sqrt(weight)`` on the *residual* so that squaring it
        reproduces a ``weight``-weighted misfit whichever way the caller's
        optimiser consumes the rows (scalar loss or residual vector).
        """
        rows = []
        for component, coords, values, weight in blocks:
            state = current(coords)
            predicted = state.ops.value(state, component)
            rows.append((predicted - values) * weight**0.5)
        return torch.cat(rows)

    diagnostics = _optimize(
        field=field,
        system=system,
        spec=spec,
        unknowns=unknowns,
        optimizer=optimizer,
        iters=iters,
        lr=lr,
        adam_iters=adam_iters,
        adam_lr=adam_lr,
        condition_weight=condition_weight,
        loss_balancing=loss_balancing,
        balance_every=balance_every,
        balance_alpha=balance_alpha,
        refinement=refinement,
        opt_kwargs=dict(optimizer_kwargs or {}),
        misfit=misfit,
        data_weight=data_weight,
        hard=hard,
    )
    recovered = unknowns.recovered()
    diagnostics.update(_hard_diagnostics(hard))
    diagnostics["hidden"] = hidden
    diagnostics["n_observations"] = sum(len(o) for o in obs)
    diagnostics["recovered"] = recovered

    # Every downstream evaluation of this system needs the coefficients, so the
    # reported residual is measured under the recovered binding.
    with bind_unknowns(recovered), torch.no_grad():
        norm = residual_norm(field, system, spec)
        data_rms = float(torch.sqrt(torch.mean(misfit(field) ** 2)))

    return InverseSolution(
        field=field,
        system=system,
        residual_norm=norm,
        method=f"inverse:{optimizer}",
        diagnostics=diagnostics,
        recovered=recovered,
        data_misfit=data_rms,
    )


__all__ = ["InverseSolution", "solve_inverse"]
