# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Sharpness-aware optimizer built on the exact-curvature torch primitives.

:class:`ExactSAM` is the reusable, amortised packaging of the
:mod:`omnibias.curvature.torch.sharpness` functionals as a drop-in
``torch.optim.Optimizer``. It targets **test error / flat minima** -- the axis
where Adam is *not* optimal -- rather than per-step training loss.

Motivation
----------
The ``mnist1d_double_descent`` study (finding **H4**) shows that adding an
*exact* sharpness **penalty** to the objective suppresses the model-wise
double-descent test-error peak more cleanly than Adam *or* an exact
Sharpness-Aware-Minimisation (SAM) ascent step. Classic SAM pays ``2x`` (an
ascent forward-backward plus the real step) to *estimate* the sharpness
direction; omnibias already has the sharpness in closed form
(``sigma'''`` rides the penalty gradient exactly), so ExactSAM instead **adds
the exact sharpness gradient to the step** and **amortises** the expensive
curvature probe over ``probe_every`` steps -- targeting ``<= SAM``'s cost. The
realised cost is a wall-clock question (each probe is several double-backward
HVPs), so it must be *measured*, not assumed.

Update
------
Every step uses the cheap loss gradient ``grad L`` (one backward); every
``probe_every`` steps it additionally computes the exact sharpness gradient
``grad S`` (``penalty`` mode) or the exact second-order SAM-gap gradient
(``ascent`` mode) and caches it, reusing the stale direction in between::

    g      = grad L + (cached) w * grad S
    m      = momentum * m + g            # one heavy-ball buffer
    theta -= lr * (sign(m) if sign_momentum else m)

``penalty`` mode (default) differentiates ``lam * curvature_sharpness(measure)``
with ``measure in {"frobenius", "trace", "top_eig"}`` (Frobenius is the H4
winner); ``ascent`` mode differentiates the exact ``sam_sharpness_gap(rho)``.

Honesty
-------
* The sharpness gradient is **exact** (closed-form ``sigma'''`` via
  reverse-over-reverse-over-reverse autograd); the Hutchinson measures are
  unbiased stochastic estimators (variance ``~ 1/n_samples``).
* This regularises curvature; it does **not** assert that flatter always
  generalises (problem-dependent).
* The base step is a self-contained heavy-ball SGD (mirroring how SAM wraps
  SGD) -- a deliberate choice, so ExactSAM is one buffer plus the amortised
  cached sharpness direction.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Literal

import torch
from omnibias.curvature.torch.sharpness import curvature_sharpness, sam_sharpness_gap
from torch import Tensor

Closure = Callable[[], Tensor]


class ExactSAM(torch.optim.Optimizer):
    r"""Amortised exact-sharpness-aware optimizer (generalisation-first, ``<= SAM`` cost).

    A drop-in ``torch.optim.Optimizer`` whose step is the exact-sharpness-augmented
    gradient descended with a selectable ``base`` optimiser (heavy-ball SGD, Adam, or a
    memory-lean per-tensor-RMS preconditioner). The ``closure`` returns
    the scalar loss with its autograd graph intact and does **not** call
    ``.backward()`` -- the optimiser owns differentiation (the ``torch.optim.LBFGS``
    convention)::

        opt = ExactSAM(model.parameters(), lr=1e-2, lam=1e-3, measure="frobenius")
        for _ in range(n_steps):
            opt.step(lambda: loss_fn(model))   # scalar loss, graph intact

    Parameters
    ----------
    lr:
        Step size of the heavy-ball base.
    lam:
        Weight of the exact sharpness penalty (``penalty`` mode). ``lam=0`` recovers
        plain heavy-ball SGD (the ablation baseline) and skips the probe. When
        ``lam_auto=True`` this is instead the *upper bound* on the effective weight.
    lam_auto:
        Register/fit-aware penalty (``penalty`` mode only). Each step, cap the effective
        weight at the largest value that keeps the *combined* step first-order
        loss-decreasing: ``lam_eff = min(lam, lam_safety * ||grad L||^2 / |<grad L, grad S>|)``
        (uncapped when the penalty does not oppose the loss gradient). Where the exact
        sharpness gradient chronically fights the fit -- e.g. ``tanh``/MSE regression, the
        register where a fixed penalty *underfits* -- ``lam_eff`` collapses toward ``0`` and
        the optimiser degrades to its ``base``; where it does not, ``lam_eff`` rides at ``lam``.
        The cap is exact for ``base="sgd"`` and a raw-coordinate heuristic for the adaptive bases.
    lam_safety:
        Fraction ``rho in (0, 1]`` of the first-order fit-descent budget the penalty may
        consume when ``lam_auto=True``.
    mode:
        ``"penalty"`` -- add ``lam * grad curvature_sharpness(measure)`` (H4 winner);
        ``"ascent"`` -- add the exact second-order SAM-gap gradient ``grad sam_gap(rho)``.
    measure:
        Sharpness functional for ``penalty`` mode: ``"frobenius"`` / ``"trace"``
        (Hutchinson) or ``"top_eig"`` (power iteration).
    rho:
        SAM radius for ``ascent`` mode.
    momentum:
        Heavy-ball coefficient in ``[0, 1)``.
    sign_momentum:
        Descend ``sign(m)`` (Lion-like) instead of ``m`` (only valid with ``base="sgd"``).
    base:
        Base optimiser applied to the sharpness-augmented gradient ``g = grad L + lam grad S``:
        ``"sgd"`` (heavy-ball, one buffer -- the SAM-wraps-SGD default), ``"adam"`` (per-coordinate
        adaptivity, adds a second-moment buffer ``v``), or ``"frugal"`` (memory-lean per-tensor RMS:
        one momentum buffer + ``O(#tensors)`` scalars). ``adam`` / ``frugal`` use ``momentum`` as the
        first-moment EMA ``beta1``.
    beta2:
        Second-moment EMA for ``base in {"adam", "frugal"}``.
    eps:
        Denominator floor for ``base in {"adam", "frugal"}``.
    probe_every:
        Refresh the exact sharpness gradient every ``probe_every`` steps and reuse it
        in between -- the amortisation that targets ``<= SAM`` cost.
    n_samples:
        Hutchinson probes for the ``"frobenius"`` / ``"trace"`` measures.
    iters:
        Power-iteration count for ``"top_eig"`` / the SAM gap's ``lambda_max``.
    seed:
        Seed of the probe's random generator (Rademacher / power-iteration starts).
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 1e-2,
        lam: float = 1e-3,
        lam_auto: bool = False,
        lam_safety: float = 0.5,
        mode: Literal["penalty", "ascent"] = "penalty",
        measure: Literal["frobenius", "trace", "top_eig"] = "frobenius",
        rho: float = 0.05,
        momentum: float = 0.9,
        sign_momentum: bool = False,
        base: Literal["sgd", "adam", "frugal"] = "sgd",
        beta2: float = 0.999,
        eps: float = 1e-8,
        probe_every: int = 1,
        n_samples: int = 4,
        iters: int = 20,
        seed: int = 0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if lam < 0.0:
            raise ValueError(f"lam must be >= 0, got {lam}")
        if mode not in ("penalty", "ascent"):
            raise ValueError(f"mode must be 'penalty' or 'ascent', got {mode!r}")
        if measure not in ("frobenius", "trace", "top_eig"):
            raise ValueError(f"measure must be 'frobenius', 'trace' or 'top_eig', got {measure!r}")
        if rho <= 0.0:
            raise ValueError(f"rho must be > 0, got {rho}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if base not in ("sgd", "adam", "frugal"):
            raise ValueError(f"base must be 'sgd', 'adam' or 'frugal', got {base!r}")
        if sign_momentum and base != "sgd":
            raise ValueError("sign_momentum is only supported with base='sgd'")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if lam_auto and mode != "penalty":
            raise ValueError("lam_auto requires mode='penalty' (there is no lambda knob in ascent mode)")
        if lam_auto and lam <= 0.0:
            raise ValueError("lam_auto treats lam as the upper bound, so lam must be > 0")
        if not 0.0 < lam_safety <= 1.0:
            raise ValueError(f"lam_safety must be in (0, 1], got {lam_safety}")
        if probe_every < 1:
            raise ValueError(f"probe_every must be >= 1, got {probe_every}")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if iters < 1:
            raise ValueError(f"iters must be >= 1, got {iters}")
        super().__init__(params, {"lr": lr})
        self._params: list[Tensor] = [
            p for group in self.param_groups for p in group["params"] if p.requires_grad
        ]
        if not self._params:
            raise ValueError("optimizer requires at least one parameter with requires_grad=True")
        self.lr = float(lr)
        self.lam = float(lam)
        self.lam_auto = bool(lam_auto)
        self.lam_safety = float(lam_safety)
        self.mode = mode
        self.measure = measure
        self.rho = float(rho)
        self.momentum = float(momentum)
        self.sign_momentum = bool(sign_momentum)
        self.base = base
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.probe_every = int(probe_every)
        self.n_samples = int(n_samples)
        self.iters = int(iters)
        self.seed = int(seed)
        self._m: list[Tensor] | None = None
        self._v: list[Tensor] | None = None
        self._c: list[Tensor] | None = None
        self._gS: list[Tensor] | None = None
        self._lam_eff: float | None = None  # last effective lambda (telemetry / the money plot)
        self._t = 0
        self._gen: torch.Generator | None = None

    def _sharpness_grad(self, loss: Tensor) -> list[Tensor]:
        r"""Exact gradient of the (weighted) sharpness term at the current params.

        Called only on refresh steps. ``penalty`` mode differentiates
        ``lam * S(measure)``; ``ascent`` mode differentiates the exact second-order
        SAM gap ``rho||grad L|| + 1/2 rho^2 max(lambda_max, 0)``. Returned detached --
        it becomes a cached, stale direction reused between refreshes.
        """
        if self.mode == "penalty":
            s = curvature_sharpness(
                loss, self._params, measure=self.measure, n_samples=self.n_samples,
                iters=self.iters, generator=self._gen, differentiable=True,
            )
            weight = self.lam
        else:
            s = sam_sharpness_gap(
                loss, self._params, rho=self.rho, iters=self.iters,
                generator=self._gen, differentiable=True,
            )
            weight = 1.0
        grad_s = torch.autograd.grad(s, self._params)
        return [weight * g.detach() for g in grad_s]

    def step(self, closure: Closure | None = None) -> Tensor:  # type: ignore[override]
        r"""One sharpness-aware step; ``closure`` returns the scalar loss (graph intact)."""
        if closure is None:
            raise ValueError("ExactSAM.step requires a closure returning the scalar loss")
        if self._gen is None:
            self._gen = torch.Generator(device=self._params[0].device).manual_seed(self.seed)
        loss = closure()
        refresh = (self._t % self.probe_every) == 0
        if refresh:
            grad_l = torch.autograd.grad(loss, self._params, retain_graph=True)
            if self.mode == "penalty" and self.lam == 0.0:
                self._gS = [torch.zeros_like(p) for p in self._params]
            else:
                self._gS = self._sharpness_grad(loss)
        else:
            grad_l = torch.autograd.grad(loss, self._params)
        assert self._gS is not None
        if self.lam_auto:
            # Fit-preservation cap: the largest ratio r in [0, 1] that keeps the *combined*
            # step first-order loss-decreasing, since d/dt L(theta - t (gL + r gS))|_0 =
            # -(||gL||^2 + r <gL, gS>). If <gL, gS> >= 0 the penalty does not fight the
            # fit (r = 1); otherwise r = min(1, safety ||gL||^2 / -<gL, gS>). Because gS
            # already carries lam, the effective penalty is lam_eff = r lam =
            # min(lam, safety ||gL||^2 / |<gL, grad S>|). Recomputed every step (cheap:
            # two reductions, no extra backward), so it adapts even between probe refreshes.
            # Exact for base="sgd"; a raw-coordinate heuristic for the adaptive bases.
            gl_det = [gl.detach() for gl in grad_l]
            gl_sq = torch.stack([(g_ * g_).sum() for g_ in gl_det]).sum()
            dot = torch.stack([(g_ * gs).sum() for g_, gs in zip(gl_det, self._gS, strict=True)]).sum()
            cap = self.lam_safety * gl_sq / dot.neg().clamp_min(1e-30)
            r = torch.where(dot >= 0, torch.ones_like(cap), cap.clamp_max(1.0))
            g = [gl + r * gs for gl, gs in zip(gl_det, self._gS, strict=True)]
            self._lam_eff = float(r) * self.lam
        else:
            g = [gl.detach() + gs for gl, gs in zip(grad_l, self._gS, strict=True)]
            if self.mode == "penalty":
                self._lam_eff = self.lam

        if self._m is None:
            self._m = [torch.zeros_like(p) for p in self._params]
        step_num = self._t + 1
        with torch.no_grad():
            if self.base == "sgd":
                # Heavy-ball SGD (the SAM-wraps-SGD default): one momentum buffer.
                for p, m, gi in zip(self._params, self._m, g, strict=True):
                    m.mul_(self.momentum).add_(gi)
                    direction = m.sign() if self.sign_momentum else m
                    p.sub_(self.lr * direction)
            elif self.base == "adam":
                # Adam preconditioning of the sharpness-augmented gradient (adds a v buffer):
                # decouples *fit* (per-coordinate adaptivity) from *flatten* (the penalty).
                if self._v is None:
                    self._v = [torch.zeros_like(p) for p in self._params]
                bc1 = 1.0 - self.momentum**step_num
                bc2 = 1.0 - self.beta2**step_num
                for p, m, v, gi in zip(self._params, self._m, self._v, g, strict=True):
                    m.mul_(self.momentum).add_(gi, alpha=1.0 - self.momentum)
                    v.mul_(self.beta2).addcmul_(gi, gi, value=1.0 - self.beta2)
                    denom = (v / bc2).sqrt_().add_(self.eps)
                    p.addcdiv_(m / bc1, denom, value=-self.lr)
            else:
                # Memory-lean adaptive base: one momentum buffer + a per-tensor RMS scalar
                # (O(#tensors)), the FrugalCurvature philosophy applied to the augmented grad.
                if self._c is None:
                    self._c = [p.new_zeros(()) for p in self._params]
                bc1 = 1.0 - self.momentum**step_num
                bc2 = 1.0 - self.beta2**step_num
                for p, m, c, gi in zip(self._params, self._m, self._c, g, strict=True):
                    m.mul_(self.momentum).add_(gi, alpha=1.0 - self.momentum)
                    c.mul_(self.beta2).add_(gi.pow(2).mean(), alpha=1.0 - self.beta2)
                    denom = (c / bc2).sqrt() + self.eps
                    p.sub_((self.lr / bc1) * m / denom)
        self._t += 1
        return loss.detach()


__all__ = ["ExactSAM"]
