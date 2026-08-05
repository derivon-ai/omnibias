# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Beta-annealing schedules for the tanh quantization surrogates (pure-Python).

The hard quantizer *forward* is independent of ``beta``; only the smooth
``tanh(beta z)`` *backward* surrogate depends on it. Annealing ``beta`` upward
over training realises the ``beta -> inf`` homotopy: the surrogate gradient
``beta * (1 - tanh(beta z)^2)`` concentrates ever more tightly at the decision
boundary (its integral is fixed, so it approaches the ``2 * delta`` Dirac limit),
interpolating from an easy, well-conditioned early objective to the exact
straight-through limit. This is the same saturation captured by the
``ActivationSpec`` limit metadata (the ``lim`` operator of Phase 1).

Pure-Python and backend-agnostic, so the identical scheduler drives torch and jax
training loops; it mirrors the ``step() -> scalar`` contract of
:class:`omnibias.torch.training.k_scheduler.KGrowthScheduler`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["BetaAnnealScheduler"]

_SCHEDULES = ("linear", "exp", "cosine")


@dataclass
class BetaAnnealScheduler:
    """Anneal the surrogate sharpness ``beta`` from ``beta_start`` to ``beta_end``.

    Parameters
    ----------
    beta_start, beta_end : float
        Endpoint sharpness values; both must be ``> 0``. ``beta_end > beta_start``
        sharpens (the usual soft-to-hard curriculum), but either ordering works.
    num_steps : int
        Number of steps over which to anneal. ``value(step)`` is clamped to the
        endpoints outside ``[0, num_steps]``.
    schedule : {"linear", "exp", "cosine"}
        Interpolation law. ``"linear"`` is linear in ``beta``; ``"exp"`` is
        geometric (linear in ``log beta``, natural for a multiplicative sharpness);
        ``"cosine"`` eases in and out.
    """

    beta_start: float
    beta_end: float
    num_steps: int
    schedule: str = "linear"
    _step: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.beta_start <= 0.0 or self.beta_end <= 0.0:
            raise ValueError("beta_start and beta_end must be > 0")
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")
        if self.schedule not in _SCHEDULES:
            raise ValueError(f"unknown schedule {self.schedule!r}; choose from {_SCHEDULES}")

    def value(self, step: int) -> float:
        """Beta at an arbitrary step (clamped to the endpoints; stateless)."""
        frac = min(max(step / self.num_steps, 0.0), 1.0)
        if self.schedule == "linear":
            return self.beta_start + frac * (self.beta_end - self.beta_start)
        if self.schedule == "exp":
            log_beta = math.log(self.beta_start) + frac * (
                math.log(self.beta_end) - math.log(self.beta_start)
            )
            return math.exp(log_beta)
        # cosine ease-in/out: monotone 0 -> 1 as frac 0 -> 1.
        ease = 0.5 * (1.0 - math.cos(math.pi * frac))
        return self.beta_start + ease * (self.beta_end - self.beta_start)

    def step(self) -> float:
        """Return the current beta and advance the internal step counter."""
        beta = self.value(self._step)
        self._step += 1
        return beta

    @property
    def current_step(self) -> int:
        """Number of times :meth:`step` has been called."""
        return self._step

    def reset(self) -> None:
        """Rewind the internal step counter to zero."""
        self._step = 0
