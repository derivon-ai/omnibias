# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact edge-of-stability (EoS) learning-rate controller.

Gradient descent on a quadratic is stable iff the step size obeys
``eta < 2 / lambda_max(H)``. The *edge of stability* (Cohen et al., 2021,
arXiv:2103.00065) is the empirical regime where full-batch GD self-organises so
that ``lambda_max(H) ~ 2/eta`` -- the sharpness rides the stability boundary.

Ordinary schedulers cannot *target* that boundary because ``lambda_max`` is
expensive to estimate. omnibias makes the **exact** top Hessian eigenvalue cheap
via the matrix-free Hessian-vector product (:func:`omnibias.curvature.torch.top_eigenvalue`),
so we can set the step size to sit at a chosen fraction ``c`` of the *exact*
stability limit, re-measured online:

.. math::

    \eta_t = \mathrm{clamp}\!\left(c \cdot \frac{2}{\lambda_{\max}(H_t)},\ \eta_{\min},\ \eta_{\max}\right).

This is a curvature-set first-order step: an *enhancement* wrappable around any
GD/SGD update, and -- unlike SAM -- it needs no ascent step, only the top
eigenvalue this study already instruments. The controller is deliberately tiny
and framework-plain; the omnibias-specific part is the exact, cheap
``lambda_max`` it consumes.

Prototyped here in the example; if it earns its keep it can graduate into
``omnibias.torch.optim`` as a first-class controller.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from omnibias.curvature.torch import top_eigenvalue
from torch import Tensor

_TINY = 1e-12


class EdgeOfStabilityLR:
    r"""Online learning-rate controller that tracks the exact edge of stability.

    Each call to :meth:`rate` (re-)estimates the exact top Hessian eigenvalue of
    the supplied ``loss`` (every :attr:`measure_every` steps, EMA-smoothed) and
    returns the target learning rate ``eta = c * 2 / lambda_max`` clamped to
    ``[eta_min, eta_max]``, along with telemetry for logging.

    For heavy-ball momentum ``beta`` the linear-stability limit widens to
    ``eta < 2(1 + beta) / lambda_max``, so the controller targets
    ``eta = c * 2(1 + beta) / lambda_max`` and the held product
    ``lambda_max * eta`` sits at :attr:`target` ``= 2c(1 + beta)`` (``= 2c`` for plain GD).

    Parameters
    ----------
    c:
        Target fraction of the stability limit (``c < 1`` stays just inside the edge;
        ``c = 1`` sits on it; ``c > 1`` deliberately rides *past* the linear edge --
        stable only because the true landscape is non-quadratic).
    momentum:
        Heavy-ball momentum of the GD step the rate feeds; widens the target to
        ``2c(1 + beta)``. Must match the optimizer's momentum.
    eta_min, eta_max:
        Clamps on the returned rate (guards the flat-init ``lambda_max -> 0`` blow-up
        and keeps a floor once the landscape sharpens).
    probe_iters:
        Power-iteration count for the matrix-free ``lambda_max`` estimate.
    measure_every:
        Re-probe ``lambda_max`` every ``k`` steps; reuse the last estimate in between
        (amortises the HVP cost on wider nets). ``1`` re-probes every step.
    ema:
        Exponential-moving-average factor on ``lambda_max`` (``0`` disables smoothing);
        damps the step-to-step jitter of the power-iteration estimate.
    seed:
        Base seed for the power-iteration probe vectors (advanced per step).
    """

    def __init__(
        self,
        *,
        c: float = 0.9,
        momentum: float = 0.0,
        eta_min: float = 1e-4,
        eta_max: float = 1.0,
        probe_iters: int = 12,
        measure_every: int = 1,
        ema: float = 0.5,
        seed: int = 0,
    ) -> None:
        if not 0.0 < c <= 2.0:
            raise ValueError(f"c must be in (0, 2], got {c}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if not 0.0 <= ema < 1.0:
            raise ValueError(f"ema must be in [0, 1), got {ema}")
        self.c = float(c)
        self.momentum = float(momentum)
        self.target = 2.0 * self.c * (1.0 + self.momentum)
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)
        self.probe_iters = int(probe_iters)
        self.measure_every = max(int(measure_every), 1)
        self.ema = float(ema)
        self.seed = int(seed)
        self._lam: float | None = None
        self._step = 0
        self.last_eta: float = float("nan")
        self.last_lambda_max: float = float("nan")

    def _generator(self, device: torch.device) -> torch.Generator:
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed + self._step)
        return gen

    def rate(self, loss: Tensor, params: Iterable[Tensor]) -> dict[str, float]:
        """Return ``{eos_eta, eos_lambda_max, eos_lambda_eta, eos_target}`` for ``loss``.

        ``loss`` must carry a live autograd graph (it is differentiated twice for
        the Hessian-vector product). ``eos_lambda_eta = eta * lambda_max`` is the
        stability product the controller holds at :attr:`target` ``= 2c(1 + beta)``.
        """
        plist = list(params)
        if not plist:
            raise ValueError("params is empty")
        if self._lam is None or self._step % self.measure_every == 0:
            gen = self._generator(plist[0].device)
            lam = max(float(top_eigenvalue(loss, plist, iters=self.probe_iters, generator=gen)), 0.0)
            if self._lam is None or self.ema <= 0.0:
                self._lam = lam
            else:
                self._lam = self.ema * self._lam + (1.0 - self.ema) * lam
        self._step += 1
        lam_eff = max(self._lam, _TINY)
        eta = min(max(self.target / lam_eff, self.eta_min), self.eta_max)
        self.last_eta = float(eta)
        self.last_lambda_max = float(self._lam)
        return {
            "eos_eta": self.last_eta,
            "eos_lambda_max": self.last_lambda_max,
            "eos_lambda_eta": self.last_eta * self.last_lambda_max,
            "eos_target": self.target,
        }


__all__ = ["EdgeOfStabilityLR"]
