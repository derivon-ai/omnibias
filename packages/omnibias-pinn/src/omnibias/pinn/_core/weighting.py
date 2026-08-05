# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Stateful multi-term loss weighting: the EMA and the update cadence.

A multi-term PINN loss ``L = sum_k lambda_k L_k`` (residual + boundary + initial
+ data) trains badly with fixed ``lambda_k``, because the terms' gradients differ
by orders of magnitude -- the *gradient pathology* of Wang, Teng & Perdikaris
(2021). The cure is to re-estimate ``lambda_k`` during training, and every
published recipe has the same two-part shape:

1. a **target** ``lhat_k`` computed from a measurement of the current step, and
2. an **exponential moving average** ``lambda_k <- alpha lambda_k +
   (1 - alpha) lhat_k`` that keeps the weights from chasing minibatch noise.

Only part 1 differs between recipes, so it is the only thing a subclass writes.
Parts 2 and 3 -- the EMA and the *update cadence* (re-estimating every step is
usually wasteful, since each estimate costs one extra backward pass per term) --
live here once.

Why this module is pure Python
------------------------------
The weighters hold host-side floats, never tensors. The backend-specific work is
*measuring* the statistics (:func:`omnibias.pinn.torch.losses.grad_stats` and its
jax twin); the state machine that consumes them is shared, so torch and jax
weights are bit-identical by construction rather than by test. It also means a
weighter is trivially picklable, loggable, and usable outside a training loop.

The measurement is deliberately not done here: a jax user computes it inside a
``jit``-ed step and reads it out, a torch user calls ``autograd.grad``, and
neither belongs in ``_core``.

Reference
---------
Wang, Teng & Perdikaris, *Understanding and mitigating gradient pathologies in
physics-informed neural networks*, arXiv:2001.04536 (2021).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GradStats:
    """Magnitude summary of one loss term's parameter gradient.

    Both statistics are over the *flattened, concatenated* parameter gradient
    ``dL_k / dtheta``, which is why they are plain floats: the weighter never
    needs the gradient itself.

    Attributes
    ----------
    max_abs:
        ``max_theta |dL_k / dtheta|``. The gradient-annealing target uses this
        for the reference term, because a stiff residual is characterised by a
        few very large entries rather than a large average.
    mean_abs:
        ``mean_theta |dL_k / dtheta|``, used for the balanced terms.
    """

    max_abs: float
    mean_abs: float

    def __post_init__(self) -> None:
        if self.max_abs < 0.0 or self.mean_abs < 0.0:
            raise ValueError(
                f"gradient magnitudes must be non-negative, got "
                f"max_abs={self.max_abs}, mean_abs={self.mean_abs}"
            )


class LossWeighter:
    """Base class: named loss weights with an EMA and an update cadence.

    Subclasses implement :meth:`targets` -- the one-step estimate of where each
    weight should go. Everything else (smoothing, cadence, clamping, combining)
    is inherited.

    Parameters
    ----------
    keys:
        The loss-term names. Order fixes the iteration order of
        :attr:`weights`; membership is checked on every :meth:`update`.
    alpha:
        EMA **retention** of the previous weights, in ``[0, 1]``. ``alpha = 0``
        takes each new target outright; ``alpha = 1`` freezes the weights. The
        update is ``lambda <- alpha lambda + (1 - alpha) lhat``, i.e. the usual
        ``(1 - a) lambda + a lhat`` with ``a = 1 - alpha``. This is the
        convention of :class:`omnibias.torch.optim.GradNormBalancer`.
    every:
        Re-estimate the targets every ``every`` calls to :meth:`update`; the
        intervening calls return the current weights untouched. Each estimate
        costs one extra backward pass per term, so ``every > 1`` is the usual
        setting.
    init:
        Initial weight for every key.
    floor, ceiling:
        Clamp applied after the EMA. The floor keeps a term from being switched
        off entirely by a transient measurement; ``ceiling=None`` is no upper
        clamp.
    """

    def __init__(
        self,
        keys: Iterable[str],
        *,
        alpha: float = 0.9,
        every: int = 1,
        init: float = 1.0,
        floor: float = 0.0,
        ceiling: float | None = None,
    ) -> None:
        key_tuple = tuple(keys)
        if not key_tuple:
            raise ValueError("keys must be non-empty")
        if len(set(key_tuple)) != len(key_tuple):
            raise ValueError(f"keys must be unique, got {key_tuple!r}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        if floor < 0.0:
            raise ValueError(f"floor must be >= 0, got {floor}")
        if ceiling is not None and ceiling < max(floor, 0.0):
            raise ValueError(f"ceiling {ceiling} must be >= floor {floor}")
        if not floor <= init <= (math.inf if ceiling is None else ceiling):
            raise ValueError(f"init {init} must lie within [{floor}, {ceiling}]")
        self.keys = key_tuple
        self.alpha = float(alpha)
        self.every = int(every)
        self.floor = float(floor)
        self.ceiling = None if ceiling is None else float(ceiling)
        self._weights = dict.fromkeys(key_tuple, float(init))
        self._step = 0
        self._n_updates = 0

    # -- state ----------------------------------------------------------

    @property
    def weights(self) -> dict[str, float]:
        """The current weights, as a fresh dict (mutating it is a no-op)."""
        return dict(self._weights)

    @property
    def step(self) -> int:
        """How many times :meth:`update` has been called."""
        return self._step

    @property
    def n_updates(self) -> int:
        """How many of those calls actually re-estimated the targets."""
        return self._n_updates

    def __getitem__(self, key: str) -> float:
        return self._weights[key]

    def __repr__(self) -> str:
        shown = ", ".join(f"{k}={v:.4g}" for k, v in self._weights.items())
        return (
            f"{type(self).__name__}({shown}, alpha={self.alpha}, "
            f"every={self.every}, step={self._step})"
        )

    # -- the recipe hook ------------------------------------------------

    def targets(self, stats: Mapping[str, Any]) -> Mapping[str, float]:
        """One-step target weights from a measurement. Implemented by subclasses."""
        raise NotImplementedError(f"{type(self).__name__} must implement targets()")

    # -- driving --------------------------------------------------------

    def update(self, stats: Mapping[str, Any]) -> dict[str, float]:
        """Advance one step and return the current weights.

        On a cadence step this measures :meth:`targets` and EMA-blends them in;
        otherwise ``stats`` is ignored entirely, so a caller may skip the
        expensive measurement on non-cadence steps and pass ``{}``.
        """
        self._step += 1
        if (self._step - 1) % self.every != 0:
            return self.weights
        self._n_updates += 1
        self.blend(self.targets(stats))
        return self.weights

    def blend(self, targets: Mapping[str, float]) -> dict[str, float]:
        """EMA the given targets into the weights, unconditionally."""
        missing = set(self.keys) - set(targets)
        if missing:
            raise ValueError(f"targets missing keys {sorted(missing)!r}")
        unknown = set(targets) - set(self.keys)
        if unknown:
            raise ValueError(f"targets has unknown keys {sorted(unknown)!r}")
        for k in self.keys:
            t = float(targets[k])
            if not math.isfinite(t):
                # A zero-gradient term gives an infinite target; keeping the old
                # weight is the only defensible response.
                continue
            w = self.alpha * self._weights[k] + (1.0 - self.alpha) * t
            w = max(w, self.floor)
            if self.ceiling is not None:
                w = min(w, self.ceiling)
            self._weights[k] = w
        return self.weights

    def combine(self, losses: Mapping[str, Any]) -> Any:
        """Return ``sum_k lambda_k L_k`` for backend tensors ``L_k``.

        Duck-typed on ``float * tensor`` and ``tensor + tensor``, so it works
        unchanged on torch and jax. The weights enter as Python floats, so no
        gradient flows into them -- that is the point of a *weighting* scheme,
        as opposed to the self-adaptive pointwise weights, which are trained.
        """
        if set(losses) != set(self.keys):
            raise ValueError(
                f"losses keys {sorted(losses)!r} do not match "
                f"weighter keys {sorted(self.keys)!r}"
            )
        total: Any = None
        for k in self.keys:
            term = self._weights[k] * losses[k]
            total = term if total is None else total + term
        return total


class ConstantWeighter(LossWeighter):
    """Fixed weights: the no-op baseline that makes the seam testable.

    Useful for ablations -- swap it in wherever an adaptive weighter goes and
    the training loop is unchanged, so any difference is the weighting.
    """

    def __init__(self, weights: Mapping[str, float]) -> None:
        if not weights:
            raise ValueError("weights must be non-empty")
        super().__init__(weights.keys(), alpha=1.0, every=1)
        for k, v in weights.items():
            if float(v) < 0.0:
                raise ValueError(f"weight for {k!r} must be >= 0, got {v}")
            self._weights[k] = float(v)

    def targets(self, stats: Mapping[str, Any]) -> Mapping[str, float]:
        """The current weights: constant by construction."""
        return dict(self._weights)


class GradNormWeighter(LossWeighter):
    r"""Gradient-norm annealing (Wang, Teng & Perdikaris 2021).

    The target for every non-reference term is

    .. math::

        \hat\lambda_k = \frac{\max_\theta |\partial_\theta L_r|}
                             {\mathrm{mean}_\theta\,|\partial_\theta L_k|}

    where ``r`` is the reference term (the PDE residual). It equalises the
    *scales* of the terms' gradients: the reference contributes its largest
    entry, the others their average, so a boundary term whose gradient is
    uniformly tiny next to a spiky residual gets scaled up rather than ignored.
    The reference term's own weight is pinned to 1.

    Contrast with :class:`omnibias.torch.optim.GradNormBalancer`, which targets
    the ratio of L2 *norms*. Same pathology, different statistic: the max/mean
    form is the annealing variant, deliberately more aggressive on stiff terms.

    Parameters
    ----------
    keys:
        Loss-term names; must include ``reference``.
    reference:
        The term whose ``max_abs`` sets the scale, usually the PDE residual.
    eps:
        Guards the division when a term's gradient vanishes.
    alpha, every, init, floor, ceiling:
        As for :class:`LossWeighter`.
    """

    def __init__(
        self,
        keys: Iterable[str],
        *,
        reference: str,
        alpha: float = 0.9,
        every: int = 1,
        init: float = 1.0,
        eps: float = 1e-12,
        floor: float = 0.0,
        ceiling: float | None = None,
    ) -> None:
        super().__init__(
            keys, alpha=alpha, every=every, init=init, floor=floor, ceiling=ceiling
        )
        if reference not in self.keys:
            raise ValueError(
                f"reference {reference!r} not among keys {list(self.keys)!r}"
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.reference = reference
        self.eps = float(eps)
        self._weights[reference] = 1.0

    def targets(self, stats: Mapping[str, Any]) -> Mapping[str, float]:
        """Annealing targets from per-term :class:`GradStats`."""
        if set(stats) != set(self.keys):
            raise ValueError(
                f"stats keys {sorted(stats)!r} do not match "
                f"weighter keys {sorted(self.keys)!r}"
            )
        ref_max = float(stats[self.reference].max_abs)
        out: dict[str, float] = {}
        for k in self.keys:
            if k == self.reference:
                out[k] = 1.0
            else:
                out[k] = ref_max / (float(stats[k].mean_abs) + self.eps)
        return out

    def blend(self, targets: Mapping[str, float]) -> dict[str, float]:
        """EMA as usual, then re-pin the reference weight to exactly 1."""
        super().blend(targets)
        self._weights[self.reference] = 1.0
        return self.weights


class NTKWeighter(LossWeighter):
    r"""NTK-trace balancing, made stateful.

    The target is the geometric-mean balance already implemented statelessly by
    :func:`omnibias.pinn.torch.losses.ntk_balanced_loss`:
    ``lhat_k = exp(mean_j log T_j - log T_k)``, so a term with a large NTK trace
    (a term the network already learns fast) is scaled down. Wrapping it in the
    EMA and cadence of :class:`LossWeighter` is what makes it usable in a real
    loop -- the stateless version recomputes from scratch every step and
    inherits all of that step's noise.

    Parameters
    ----------
    keys:
        Loss-term names.
    eps:
        Floor applied to each trace before the logarithm.
    alpha, every, init, floor, ceiling:
        As for :class:`LossWeighter`.
    """

    def __init__(
        self,
        keys: Iterable[str],
        *,
        alpha: float = 0.9,
        every: int = 1,
        init: float = 1.0,
        eps: float = 1e-12,
        floor: float = 0.0,
        ceiling: float | None = None,
    ) -> None:
        super().__init__(
            keys, alpha=alpha, every=every, init=init, floor=floor, ceiling=ceiling
        )
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.eps = float(eps)

    def targets(self, stats: Mapping[str, Any]) -> Mapping[str, float]:
        """Geometric-mean balance of the per-term NTK traces."""
        if set(stats) != set(self.keys):
            raise ValueError(
                f"stats keys {sorted(stats)!r} do not match "
                f"weighter keys {sorted(self.keys)!r}"
            )
        log_t = {k: math.log(max(float(stats[k]), self.eps)) for k in self.keys}
        mean_log = sum(log_t.values()) / len(log_t)
        return {k: math.exp(mean_log - log_t[k]) for k in self.keys}


__all__ = [
    "ConstantWeighter",
    "GradNormWeighter",
    "GradStats",
    "LossWeighter",
    "NTKWeighter",
]
