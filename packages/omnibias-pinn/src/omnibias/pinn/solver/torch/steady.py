# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Steady / boundary-value drivers (torch).

Two honest paths:

* :func:`solve_least_squares` -- for a **linear** PDE with a frozen-feature
  ansatz the residual is *affine* in the readout weights, so the collocation
  operator is assembled column-by-column from the **closed-form** differential
  operators (no autodiff) and the system is solved with a single least-squares
  solve (**numerical**).
* :func:`solve_optimize` -- a general residual-minimisation driver. The
  differential operators are still **closed-form**; the *parameter* gradients use
  backend **autodiff**. Beyond the default Adam warmup + L-BFGS (quasi-Newton) it
  accepts the exact-curvature optimisers of :mod:`omnibias.torch.optim`, whose
  Hessian / Gauss-Newton products are matrix-free double-backward passes over the
  closed-form operators.

Second-order scope note: the curvature path is **torch-only**. The JAX solver
(:mod:`omnibias.pinn.solver.jax.steady`) has no ``solve_optimize``, and
:mod:`omnibias.jax.optim` ships the functional ``gauss_newton_*`` /
``natural_gradient_*`` primitives rather than optimiser objects.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import torch
from omnibias.pinn.solver._core.hard import HardConditionPlan, plan_hard_conditions
from omnibias.pinn.solver._core.sampling import (
    CollocationSpec,
    RefinementSpec,
    candidate_points,
    select_refinement_points,
)
from omnibias.pinn.solver._core.system import System
from omnibias.pinn.solver._core.taxonomy import Linearity
from omnibias.pinn.solver._core.unknowns import Unknown, bind_unknowns
from omnibias.pinn.solver.torch._solution import FieldSolution
from omnibias.pinn.solver.torch.assemble import (
    build_plan,
    condition_residual,
    default_interior,
    eval_plan_rows,
    interior_residual,
    residual_norm,
    to_tensor,
)
from omnibias.pinn.solver.torch.fields import build_field, freeze_features
from omnibias.pinn.solver.torch.readout import (
    readout_dtype,
    readout_size,
    set_readout,
)
from omnibias.torch.optim import (
    CubicGaussNewton,
    CubicNewton,
    GaussNewton,
    GradNormBalancer,
    JetSubspaceTensor,
    NaturalGradient,
    TrustRegionNewtonCG,
    functional_residual_fn,
    gauss_newton_fisher,
)


class _ClosureOptimizer(Protocol):
    """An optimiser whose ``step`` is driven by a closure returning a tensor.

    The curvature optimisers own differentiation, so the closure recomputes the
    objective (scalar loss, or residual vector) from the live parameters with its
    graph intact and never calls ``backward()``.
    """

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor: ...


#: Exact-curvature optimisers driven by a **scalar-loss** closure, exactly like
#: :class:`torch.optim.LBFGS`. ``KFAC`` is deliberately absent: it installs
#: forward / backward hooks on :class:`torch.nn.Linear` to build its Kronecker
#: factors, which the closed-form jet forward of a field never triggers, so it
#: would silently precondition with stale identity factors.
_SCALAR_OPTIMIZERS: dict[str, Callable[..., _ClosureOptimizer]] = {
    "cubic_newton": CubicNewton,
    "jet_subspace_tensor": JetSubspaceTensor,
    "natural_gradient": NaturalGradient,
    "trust_region_newton_cg": TrustRegionNewtonCG,
}

#: Optimisers that consume the **residual vector** instead of a scalar loss, so
#: term weighting has to travel in the rows (see :func:`_weighted_rows`).
_RESIDUAL_OPTIMIZERS = frozenset({"cubic_gauss_newton", "gauss_newton"})

#: Every accepted ``optimizer=`` value.
OPTIMIZERS: frozenset[str] = frozenset(
    {"adam", "lbfgs"} | set(_SCALAR_OPTIMIZERS) | set(_RESIDUAL_OPTIMIZERS)
)


def _scaled(rows: torch.Tensor, weight: float) -> torch.Tensor:
    """Scale a residual block so ``sum(scaled^2) == weight * mean(rows^2)``."""
    return rows * math.sqrt(weight / max(int(rows.numel()), 1))


def _weighted_rows(
    field: Any,
    system: System,
    coords: torch.Tensor,
    spec: CollocationSpec,
    condition_weight: float,
    extra: torch.Tensor | None = None,
    extra_weight: float = 1.0,
    hard: HardConditionPlan | None = None,
) -> torch.Tensor:
    r"""The residual vector whose ``sum(r^2)`` *is* the fused scalar loss.

    Row ``i`` of the interior block is scaled by ``1/sqrt(n_interior)`` and row
    ``j`` of the condition block by ``sqrt(condition_weight / n_condition)``, so

    ``sum(r^2) == mean(r_interior^2) + condition_weight * mean(r_condition^2)``,

    i.e. exactly the scalar objective the L-BFGS / Adam paths minimise. The
    residual-vector optimisers form ``0.5 * mean(r^2)``, a fixed positive multiple
    of that, which leaves the minimiser and the Gauss-Newton directions unchanged.

    ``extra`` is an optional third block (the data misfit of an inverse solve),
    scaled the same way by ``extra_weight``.
    """
    rows = [_scaled(interior_residual(field, system, coords), 1.0)]
    cond = condition_residual(field, system, spec, hard)
    if cond.numel():
        rows.append(_scaled(cond, condition_weight))
    if extra is not None and extra.numel():
        rows.append(_scaled(extra, extra_weight))
    return torch.cat(rows)


class _Unknowns(torch.nn.Module):
    """The unconstrained *raw* parameters standing behind unknown coefficients.

    Holding them in a :class:`torch.nn.Module` is what lets the whole optimiser
    surface work unchanged on an inverse problem: they show up in
    ``named_parameters()``, so :func:`functional_residual_fn` folds them into the
    same flat vector as the field weights, and
    :func:`torch.func.functional_call` substitutes them during a Jacobian pass.

    The binding is rebuilt on every call rather than cached, because
    :meth:`Unknown.from_raw` is part of the graph -- caching it would freeze the
    coefficient at its value from the previous iteration.
    """

    def __init__(self, unknowns: Sequence[Unknown], *, dtype: torch.dtype) -> None:
        super().__init__()
        self.descriptors = tuple(unknowns)
        self.raw = torch.nn.ParameterDict(
            {
                u.name: torch.nn.Parameter(
                    torch.tensor(u.initial_raw(), dtype=dtype)
                )
                for u in self.descriptors
            }
        )

    def binding(self) -> dict[str, Any]:
        return {u.name: u.from_raw(self.raw[u.name]) for u in self.descriptors}

    def recovered(self) -> dict[str, float]:
        with torch.no_grad():
            return {
                u.name: float(u.from_raw(self.raw[u.name])) for u in self.descriptors
            }


class _Collocation:
    """Mutable holder for the interior collocation tensor.

    Residual-adaptive refinement rewrites the interior point set *during* the
    optimisation, and both the scalar closures and the functional residual module
    have to see the new points. Reading them through one shared holder means the
    refinement never has to rebuild an optimiser or a residual function.
    """

    __slots__ = ("coords",)

    def __init__(self, coords: torch.Tensor) -> None:
        self.coords = coords


class _ResidualModule(torch.nn.Module):
    """Expose ``field`` + ``system`` as a module whose forward is the residual vector.

    This is the bridge to the *functional* optimisers: :func:`functional_residual_fn`
    walks ``named_parameters()`` and rebuilds the forward with
    :func:`torch.func.functional_call`, so the residual becomes a pure function of a
    flat parameter vector that :func:`torch.func.jacrev` can differentiate. The
    unknown coefficients ride along in that vector, which is why an inverse solve
    can use the Gauss-Newton path without any extra plumbing.
    """

    def __init__(
        self,
        field: Any,
        system: System,
        points: _Collocation,
        spec: CollocationSpec,
        condition_weight: float,
        unknowns: _Unknowns,
        misfit: Callable[[Any], torch.Tensor] | None = None,
        data_weight: float = 1.0,
        hard: HardConditionPlan | None = None,
    ) -> None:
        super().__init__()
        self.field = field
        self.unknowns = unknowns
        self._system = system
        self._points = points
        self._spec = spec
        self._condition_weight = float(condition_weight)
        self._misfit = misfit
        self._data_weight = float(data_weight)
        self._hard = hard

    def forward(self) -> torch.Tensor:
        with bind_unknowns(self.unknowns.binding()):
            extra = None if self._misfit is None else self._misfit(self.field)
            return _weighted_rows(
                self.field,
                self._system,
                self._points.coords,
                self._spec,
                self._condition_weight,
                extra,
                self._data_weight,
                self._hard,
            )


def _hard_diagnostics(hard: HardConditionPlan) -> dict[str, Any]:
    """Report what the hard-condition plan absorbed, and what it declined and why."""
    return {
        "hard_conditions": hard.summary(),
        "hard_absorbed": len(hard.conditions),
        "hard_declined": tuple(str(d) for d in hard.declined),
    }


def solve_least_squares(
    system: System,
    *,
    hidden: int = 128,
    activation: str = "tanh",
    weight_init_scale: float | None = 2.0,
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
    collocation: CollocationSpec | None = None,
    ridge: float = 1e-8,
    hard_conditions: str = "none",
    basis: str = "mlp",
    K: int = 8,
    L: float | tuple[float, ...] = 2.0 * math.pi,
    time_hidden: int | None = None,
    time_depth: int = 1,
) -> FieldSolution:
    """Exact-operator linear collocation: one least-squares solve.

    Only valid for linear systems. The feature map is frozen so the linear
    readout is the only unknown (one-layer: the hidden ``W`` / ``beta``;
    spectral / Chebyshev: the temporal MLP ``W_t`` / ``beta_t`` / inner layers);
    the collocation matrix is assembled from the closed-form operators applied
    to each frozen feature.

    ``hard_conditions="auto"`` embeds every condition it can certify into the
    ansatz and drops those rows from the system, leaving a smaller least-squares
    problem that satisfies them identically. It is opt-in: the default ``"none"``
    reproduces the previous behaviour bit for bit.
    """
    if system.linearity is not Linearity.LINEAR:
        raise ValueError(
            "solve_least_squares requires a LINEAR system; use solve_optimize "
            "for nonlinear problems"
        )
    system.require_bound_coefficients("solve_least_squares")
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
        basis=basis,
        K=K,
        L=L,
        time_hidden=time_hidden,
        time_depth=time_depth,
    )
    freeze_features(field)
    _, _, n_unknowns = readout_size(field)

    with torch.no_grad():
        # Build the collocation states ONCE (frozen features -> sigma constant),
        # then read off the affine collocation matrix column-by-column from the
        # closed-form operators (no autodiff). The cage is affine in the readout,
        # so this stays a linear solve when hard conditions are absorbed.
        plan = build_plan(field, system, spec, hard)
        e_k = torch.zeros(n_unknowns, dtype=readout_dtype(field))
        set_readout(field, e_k)
        r0 = eval_plan_rows(plan)
        n_rows = r0.shape[0]
        mat = torch.zeros(n_rows, n_unknowns, dtype=r0.dtype)
        for k in range(n_unknowns):
            e_k.zero_()
            e_k[k] = 1.0
            set_readout(field, e_k)
            mat[:, k] = eval_plan_rows(plan) - r0
        rhs = (-r0).unsqueeze(1)
        if ridge > 0.0:
            gram = mat.T @ mat + ridge * torch.eye(n_unknowns, dtype=mat.dtype)
            theta = torch.linalg.solve(gram, mat.T @ rhs)
        else:
            theta = torch.linalg.lstsq(mat, rhs, driver="gelsd").solution
        set_readout(field, theta.squeeze(1))

    return FieldSolution(
        field=field,
        system=system,
        residual_norm=residual_norm(field, system, spec),
        method="least_squares",
        diagnostics={
            "n_unknowns": n_unknowns,
            "n_rows": int(n_rows),
            **_hard_diagnostics(hard),
        },
    )


def solve_optimize(
    system: System,
    *,
    hidden: int = 64,
    activation: str = "tanh",
    weight_init_scale: float | None = None,
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
    collocation: CollocationSpec | None = None,
    optimizer: str = "lbfgs",
    iters: int = 200,
    lr: float = 1.0,
    adam_iters: int = 200,
    adam_lr: float = 1e-2,
    condition_weight: float = 10.0,
    loss_balancing: str = "none",
    balance_every: int = 10,
    balance_alpha: float = 0.9,
    refinement: RefinementSpec | None = None,
    optimizer_kwargs: dict[str, Any] | None = None,
    hard_conditions: str = "none",
    basis: str = "mlp",
    K: int = 8,
    L: float | tuple[float, ...] = 2.0 * math.pi,
    time_hidden: int | None = None,
    time_depth: int = 1,
) -> FieldSolution:
    """General residual-minimisation driver (Adam warmup, then ``optimizer``).

    Parameters
    ----------
    optimizer
        One of :data:`OPTIMIZERS`. ``"lbfgs"`` (default) and ``"adam"`` are the
        first-order / quasi-Newton baselines. The rest are the exact-curvature
        optimisers of :mod:`omnibias.torch.optim`, whose Hessian / Gauss-Newton
        products come from matrix-free double-backward passes over the closed-form
        operators:

        * ``"cubic_newton"`` -- adaptive cubic regularisation (ARC) on the exact
          full Hessian: globally convergent and saddle-escaping, no learning rate.
        * ``"trust_region_newton_cg"`` -- trust-region Newton-CG.
        * ``"jet_subspace_tensor"`` -- a higher-order tensor model in a Krylov
          subspace.
        * ``"natural_gradient"`` -- Fisher scoring. Defaults to the closed-form
          Gauss-Newton Fisher ``F = (1/N) J^T J`` of *this* residual; pass
          ``optimizer_kwargs={"metric": ...}`` for another Riemannian metric, or
          ``{"metric": None}`` for plain backtracked gradient descent.
        * ``"cubic_gauss_newton"`` -- ARC on the PSD Gauss-Newton metric; the
          natural fit for a least-squares residual, and the recommended
          second-order default.
        * ``"gauss_newton"`` -- Levenberg-Marquardt over the flat parameter vector
          (functional; bridged through :func:`functional_residual_fn`). Prefer
          ``optimizer_kwargs={"solver": "qr"}`` or ``"cgls"`` to avoid squaring the
          conditioning of a stiff operator.

        ``KFAC`` is deliberately not offered: it builds its Kronecker factors from
        hooks on :class:`torch.nn.Linear`, which the closed-form jet forward never
        triggers, so it would precondition with stale factors.
    iters
        Iterations of ``optimizer`` (for ``"lbfgs"``, its ``max_iter``).
    loss_balancing
        ``"none"`` (default) or ``"grad_norm"``. The latter drives a
        :class:`~omnibias.torch.optim.GradNormBalancer` over the two loss terms
        ``[interior_mse, condition_mse]``, refreshed every ``balance_every``
        iterations, so the *weighted* per-term gradient norms match and the
        gradient-pathology stiffness of a multi-term PINN loss goes away.
        ``condition_weight`` still applies on top of the balanced weight, so set it
        to ``1.0`` unless you want to deliberately over-weight the conditions.
        Only the scalar-loss optimisers can use it -- the residual-vector
        optimisers carry term weights in the rows instead.
    refinement
        Optional :class:`~omnibias.pinn.solver.RefinementSpec` enabling
        residual-adaptive refinement (RAR): every ``refinement.every`` iterations
        of ``optimizer``, fresh candidate points are scored by residual magnitude
        under ``no_grad`` and the selected ones are appended to the interior set.
        The Adam warmup does not refine -- the residual of a near-converged field
        is what carries the signal. This is the ``optimize`` path only:
        :func:`solve_least_squares` caches a ``CollocationPlan`` whose
        frozen-feature states are built once, so its point set cannot move.
    optimizer_kwargs
        Extra keyword arguments forwarded to the chosen optimiser's constructor.
    hard_conditions
        ``"none"`` (default) keeps every condition in the loss, exactly as
        before. ``"auto"`` embeds each condition the planner can certify into the
        ansatz and removes it from the loss, so the multi-term stiffness those
        rows create disappears along with the need to tune ``condition_weight``
        against them. Absorption is partial: whatever cannot be certified stays
        soft, and ``solution.diagnostics["hard_declined"]`` says why.
    """
    _check_optimizer(optimizer, loss_balancing, balance_every)
    system.require_bound_coefficients("solve_optimize")
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
        basis=basis,
        K=K,
        L=L,
        time_hidden=time_hidden,
        time_depth=time_depth,
    )
    diagnostics = _optimize(
        field=field,
        system=system,
        spec=spec,
        unknowns=_Unknowns((), dtype=dtype),
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
        hard=hard,
    )
    diagnostics.update(_hard_diagnostics(hard))
    diagnostics["hidden"] = hidden
    return FieldSolution(
        field=field,
        system=system,
        residual_norm=residual_norm(field, system, spec),
        method=f"optimize:{optimizer}",
        diagnostics=diagnostics,
    )


def _check_optimizer(
    optimizer: str, loss_balancing: str, balance_every: int
) -> None:
    """Validate the optimiser / balancing combination before any work is done."""
    if optimizer not in OPTIMIZERS:
        raise ValueError(
            f"unknown optimizer {optimizer!r}; expected one of {sorted(OPTIMIZERS)}"
        )
    if loss_balancing not in ("none", "grad_norm"):
        raise ValueError(
            f"loss_balancing must be 'none' or 'grad_norm', got {loss_balancing!r}"
        )
    if loss_balancing != "none" and optimizer in _RESIDUAL_OPTIMIZERS:
        raise ValueError(
            f"loss_balancing={loss_balancing!r} needs a scalar loss, but optimizer "
            f"{optimizer!r} consumes the residual vector; weight its rows through "
            "condition_weight instead"
        )
    if balance_every < 1:
        raise ValueError(f"balance_every must be >= 1, got {balance_every}")


def _optimize(
    *,
    field: Any,
    system: System,
    spec: CollocationSpec,
    unknowns: _Unknowns,
    optimizer: str,
    iters: int,
    lr: float,
    adam_iters: int,
    adam_lr: float,
    condition_weight: float,
    loss_balancing: str,
    balance_every: int,
    balance_alpha: float,
    refinement: RefinementSpec | None,
    opt_kwargs: dict[str, Any],
    misfit: Callable[[Any], torch.Tensor] | None = None,
    data_weight: float = 1.0,
    hard: HardConditionPlan | None = None,
) -> dict[str, Any]:
    """The one optimisation loop behind both the forward and inverse drivers.

    Everything that varies between them is a parameter: ``unknowns`` contributes
    extra optimisation variables and the coefficient binding, and ``misfit``
    contributes the data term. A forward solve passes an empty ``_Unknowns`` and no
    misfit, so it pays only an empty context-manager per residual evaluation --
    which is why inverse problems get the curvature optimisers, residual-adaptive
    refinement and gradient-norm balancing for free rather than as a second
    implementation that could drift.
    """
    points = _Collocation(default_interior(field, system, spec))
    n_uniform = int(points.coords.shape[0])
    dtype = readout_dtype(field)
    trainable = list(field.parameters()) + list(unknowns.parameters())
    rows = _ResidualModule(
        field,
        system,
        points,
        spec,
        condition_weight,
        unknowns,
        misfit,
        data_weight,
        hard,
    )

    def loss_terms() -> list[torch.Tensor]:
        with bind_unknowns(unknowns.binding()):
            r = interior_residual(field, system, points.coords)
            terms = [torch.mean(r ** 2)]
            cr = condition_residual(field, system, spec, hard)
            if cr.numel():
                terms.append(torch.mean(cr ** 2))
            if misfit is not None:
                terms.append(torch.mean(misfit(field) ** 2))
        return terms

    #: Fixed weight of each term, before any balancing.
    weights = [1.0, condition_weight, data_weight]
    n_terms = len(loss_terms())
    balancer: GradNormBalancer | None = None
    if loss_balancing == "grad_norm":
        if n_terms < 2:
            raise ValueError(
                "loss_balancing='grad_norm' needs at least two loss terms, but this "
                "system has no boundary or initial conditions to balance against"
            )
        balancer = GradNormBalancer(n_terms=n_terms, alpha=balance_alpha)
        balancer.weights = balancer.weights.to(dtype)
    balance = [1.0] * n_terms

    def loss() -> torch.Tensor:
        terms = loss_terms()
        value = balance[0] * weights[0] * terms[0]
        for k in range(1, len(terms)):
            value = value + balance[k] * weights[k] * terms[k]
        return value

    def refresh_balance(it: int) -> None:
        if balancer is None or it % balance_every:
            return
        updated = balancer.update(loss_terms(), trainable)
        for k in range(n_terms):
            balance[k] = float(updated[k])

    # Monotone count of refinement rounds performed. The round index seeds the
    # candidate draw, so it must never restart -- deriving it from a per-phase
    # iteration counter would redraw an earlier round's points verbatim.
    rounds_done = 0

    def refine(it: int) -> None:
        """Append the highest-residual fresh candidates to the interior set."""
        nonlocal rounds_done
        if refinement is None or it == 0 or it % refinement.every:
            return
        rounds_done += 1
        candidates = candidate_points(
            system.domain, spec, refinement, round_index=rounds_done
        )
        with torch.no_grad(), bind_unknowns(unknowns.binding()):
            scored = to_tensor(candidates, field)
            rows = interior_residual(field, system, scored)
            n_pts = int(scored.shape[0])
            if rows.numel() % n_pts:
                raise ValueError(
                    "residual-adaptive refinement needs one residual value per "
                    f"point per equation; got {rows.numel()} rows for {n_pts} points"
                )
            per_point = rows.reshape(-1, n_pts).abs().amax(dim=0)
        extra = select_refinement_points(
            candidates,
            per_point.detach().cpu().numpy(),
            refinement,
            n_existing=int(points.coords.shape[0]),
            round_index=rounds_done,
        )
        if extra.shape[0]:
            points.coords = torch.cat(
                [points.coords, to_tensor(extra, field)], dim=0
            )

    def step_hooks(it: int) -> None:
        refresh_balance(it)
        refine(it)

    def run_adam(
        n: int, step_lr: float, hooks: Callable[[int], None] = step_hooks
    ) -> None:
        adam = torch.optim.Adam(trainable, lr=step_lr)
        for it in range(n):
            hooks(it)
            adam.zero_grad()
            value = loss()
            value.backward()
            adam.step()

    # The warmup deliberately does *not* refine: the residual of a near-random
    # field says nothing about where the converged solution is hard, so refining
    # there would spend the `max_points` budget on noise. Refinement starts once
    # the warmup has produced a meaningful residual landscape.
    if adam_iters > 0:
        run_adam(adam_iters, adam_lr, hooks=refresh_balance)

    diagnostics: dict[str, Any] = {"optimizer": optimizer}

    if optimizer == "lbfgs":
        refresh_balance(0)
        inner = iters if refinement is None else min(refinement.every, iters)
        lbfgs = torch.optim.LBFGS(
            trainable,
            lr=lr,
            max_iter=inner,
            line_search_fn="strong_wolfe",
            **opt_kwargs,
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            value = loss()
            value.backward()
            return value

        if refinement is None:
            lbfgs.step(closure)
        else:
            # L-BFGS owns its inner iteration loop inside one .step(), so RAR needs
            # an explicit outer loop: `every` inner iterations per round, refining
            # between rounds.
            rounds = max(1, iters // refinement.every)
            for round_index in range(1, rounds + 1):
                lbfgs.step(closure)
                if round_index < rounds:
                    refine(round_index * refinement.every)
    elif optimizer == "adam":
        run_adam(iters, adam_lr)
    elif optimizer in _SCALAR_OPTIMIZERS:
        if optimizer == "natural_gradient" and "metric" not in opt_kwargs:
            # Fisher scoring on the closed-form Gauss-Newton metric of *this*
            # residual; with no metric at all NaturalGradient is only backtracked
            # gradient descent, which would be a misleading default here.
            _, metric_fn = functional_residual_fn(rows)

            def gn_metric(flat: torch.Tensor) -> torch.Tensor:
                fisher, _ = gauss_newton_fisher(metric_fn, flat)
                return fisher

            opt_kwargs["metric"] = gn_metric
        curvature = _SCALAR_OPTIMIZERS[optimizer](trainable, **opt_kwargs)
        for it in range(iters):
            step_hooks(it)
            curvature.step(loss)
    elif optimizer == "cubic_gauss_newton":
        cgn = CubicGaussNewton(trainable, **opt_kwargs)

        def residual_closure() -> torch.Tensor:
            return rows()

        for it in range(iters):
            refine(it)
            cgn.step(residual_closure)
    else:  # "gauss_newton" -- functional over the flat parameter vector
        flat, residual_fn = functional_residual_fn(rows)
        gn = GaussNewton(**opt_kwargs)
        info: Any = None
        params = list(rows.parameters())
        for it in range(iters):
            if refinement is not None and it and it % refinement.every == 0:
                # This optimiser is functional, so the live module parameters lag
                # behind `flat`; sync them first or the candidates get scored with
                # the pre-optimisation field.
                with torch.no_grad():
                    torch.nn.utils.vector_to_parameters(flat, params)
                refine(it)
            flat, info = gn.step(residual_fn, flat)
        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(flat, params)
        diagnostics["gn_damping"] = gn.damping
        if info is not None:
            diagnostics["gn_accepted"] = info.accepted

    if balancer is not None:
        diagnostics["loss_balancing"] = loss_balancing
        diagnostics["balance_weights"] = tuple(balance)
    if refinement is not None:
        diagnostics["n_interior_uniform"] = n_uniform
        diagnostics["n_interior_final"] = int(points.coords.shape[0])
        diagnostics["n_refinement_rounds"] = rounds_done
    return diagnostics


def solve_steady(
    system: System, *, method: str = "auto", **kwargs: Any
) -> FieldSolution:
    """Dispatch a steady solve: exact linear collocation or optimisation.

    ``method`` is ``"auto"``, ``"least_squares"`` / ``"lstsq"``, ``"optimize"``, or
    any member of :data:`OPTIMIZERS` (which then selects that optimiser).
    """
    if method == "auto":
        method = (
            "least_squares"
            if system.linearity is Linearity.LINEAR
            else "optimize"
        )
    if method in ("least_squares", "lstsq"):
        return solve_least_squares(system, **kwargs)
    if method == "optimize" or method in OPTIMIZERS:
        if method in OPTIMIZERS:
            kwargs.setdefault("optimizer", method)
        return solve_optimize(system, **kwargs)
    raise ValueError(f"unknown steady method {method!r}")


__all__ = [
    "OPTIMIZERS",
    "solve_least_squares",
    "solve_optimize",
    "solve_steady",
]
