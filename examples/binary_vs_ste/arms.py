# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The benchmark arms: one hard ``sign`` forward, many different backwards.

Every arm produces a *quantizer* ``q(z) -> {-1,+1}`` (used for both weights and
activations) with the exact hard ``sign`` forward; arms differ only in the
**backward** -- the surrogate-gradient kernel and how its bandwidth ``beta`` is
set (fixed, annealed, or learned). Three backends:

* ``ste`` -- the straight-through estimator baseline (compact-support box kernel);
* ``omnibias`` -- the shipped library ``omnibias.binary.binarize`` (exact
  ``beta * tanh'(beta z)`` backward), used to reproduce both the mis-scaled
  ``beta = 10`` default *and* the correctly scaled ``beta = 1`` setting;
* ``kernel`` -- the single-hyperplane kernel menu from
  :mod:`examples.binary_vs_ste.kernels` (``tanh`` / ``logistic`` / ``gaussian`` /
  ``cauchy``), peak-normalised so ``beta`` controls only the window width;
* ``curvature`` -- the jet-STE backward ``s'(z) + (h^2/6) s'''(z)``
  (:func:`omnibias.binary.torch.ops.binarize_curvature`), a 4th-order-accurate
  windowed-average slope that the closed-form ``s'''`` makes free.

The default ``beta`` is ``~1`` (matching the post-BatchNorm activation scale), not
``10`` -- a too-large ``beta`` shrinks the gradient window below the data scale and
starves most units (the dead-unit failure mode of a sharp surrogate / the STE box).

``scale_aware`` standardises the pre-activation by its own (detached) std before the
kernel, so ``beta`` measures the window in *units of the tensor's own scale*. This
makes the surrogate scale-free like the STE box -- crucial because one global
``beta`` otherwise can't fit both the unit-scale BatchNorm activations and the
``~0.03``-scale kaiming weights at once (the weight path would collapse to plain STE).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from omnibias.binary.torch import ops as q
from torch import Tensor

from examples.binary_vs_ste.kernels import KERNELS, NORMALIZATIONS, binarize_kernel
from examples.binary_vs_ste.ste import binarize_ste

#: A quantizer maps a real tensor to ``{-1, +1}`` (hard forward, surrogate backward).
Quantizer = Callable[[Tensor], Tensor]

_BACKENDS = ("ste", "omnibias", "kernel", "curvature")
_BETA_MODES = ("fixed", "anneal", "learnable")
#: floor on the per-tensor std used by the scale-aware surrogate (avoid /0).
_SCALE_EPS = 1e-5


def _scale_aware_quantizer(
    beta: float | Tensor, kernel: str, normalize: str
) -> Quantizer:
    """Kernel quantizer on the per-tensor standardised pre-activation ``z / std(z)``.

    ``beta`` is divided by the (detached) per-call std, so the effective window is a
    multiple of the tensor's own spread; a learnable ``beta`` Tensor still receives a
    gradient (the std is a detached constant of the division).
    """

    def quant(z: Tensor) -> Tensor:
        scale = z.detach().std(unbiased=False).clamp_min(_SCALE_EPS)
        return binarize_kernel(z, beta / scale, kernel=kernel, normalize=normalize)

    return quant


@dataclass(frozen=True)
class Arm:
    """One benchmark arm: how to build the per-step quantizer.

    Parameters
    ----------
    name : str
        Short identifier used in result tables.
    backend : {"ste", "omnibias", "kernel"}
        Which family supplies the backward (see module docstring).
    kernel : str
        Surrogate density (one of :data:`~examples.binary_vs_ste.kernels.KERNELS`);
        used by the ``kernel`` backend and recorded for ``ste``/``omnibias``.
    normalize : {"peak", "exact"}
        ``"peak"`` -> unit-height kernel (``beta`` sets width only);
        ``"exact"`` -> ``beta * k(beta z)`` (the exact smooth-surrogate derivative,
        what the shipped ``binarize`` uses).
    beta_mode : {"fixed", "anneal", "learnable"}
        How ``beta`` evolves: constant, annealed ``beta -> beta_end``, or a trained
        scalar parameter (initialised at ``beta``).
    beta : float
        Fixed / initial / annealing-start bandwidth.
    beta_end : float
        Annealing end bandwidth (used only when ``beta_mode == "anneal"``).
    scale_aware : bool
        Standardise the pre-activation by its own (detached) std before the kernel
        (``kernel`` backend only), making ``beta`` scale-free across weights and
        activations.
    description : str
        Human-readable one-liner.
    """

    name: str
    backend: str
    kernel: str
    normalize: str
    beta_mode: str
    beta: float
    beta_end: float
    description: str
    scale_aware: bool = False

    def __post_init__(self) -> None:
        if self.backend not in _BACKENDS:
            raise ValueError(f"unknown backend {self.backend!r}; choose from {_BACKENDS}")
        if self.kernel not in KERNELS:
            raise ValueError(f"unknown kernel {self.kernel!r}; choose from {KERNELS}")
        if self.normalize not in NORMALIZATIONS:
            raise ValueError(f"unknown normalize {self.normalize!r}; choose from {NORMALIZATIONS}")
        if self.beta_mode not in _BETA_MODES:
            raise ValueError(f"unknown beta_mode {self.beta_mode!r}; choose from {_BETA_MODES}")
        if self.backend == "ste" and self.beta_mode != "fixed":
            raise ValueError("the STE baseline has no beta to anneal or learn")
        if self.backend == "curvature" and self.beta_mode == "learnable":
            raise ValueError("the curvature backward carries no beta gradient; use fixed/anneal")
        if self.scale_aware and self.backend != "kernel":
            raise ValueError("scale_aware applies to the kernel backend only")
        if self.beta_mode == "anneal" and self.beta_end <= 0.0:
            raise ValueError("anneal arms need beta_end > 0")

    def make_quantizer(self, beta: float | Tensor) -> Quantizer:
        """Build the quantizer ``q(z)`` for the current ``beta`` (float or scalar Tensor)."""
        if self.backend == "ste":
            return binarize_ste
        if self.backend == "omnibias":
            return partial(q.binarize, beta=beta)
        if self.backend == "curvature":
            return partial(q.binarize_curvature, beta=beta)
        if self.scale_aware:
            return _scale_aware_quantizer(beta, self.kernel, self.normalize)
        return partial(binarize_kernel, beta=beta, kernel=self.kernel, normalize=self.normalize)


def _arm(
    name: str,
    backend: str,
    description: str,
    *,
    kernel: str = "tanh",
    normalize: str = "peak",
    beta_mode: str = "fixed",
    beta: float = 1.0,
    beta_end: float = 3.0,
    scale_aware: bool = False,
) -> Arm:
    return Arm(
        name=name,
        backend=backend,
        kernel=kernel,
        normalize=normalize,
        beta_mode=beta_mode,
        beta=beta,
        beta_end=beta_end,
        description=description,
        scale_aware=scale_aware,
    )


_ARMS: dict[str, Arm] = {
    "ste": _arm(
        "ste", "ste", "straight-through estimator (compact box kernel, beta=1)", kernel="box"
    ),
    "omnibias_b10": _arm(
        "omnibias_b10",
        "omnibias",
        "shipped binarize, exact beta*tanh'(beta z), beta=10 (mis-scaled control)",
        normalize="exact",
        beta=10.0,
    ),
    "omnibias_b1": _arm(
        "omnibias_b1",
        "omnibias",
        "shipped binarize, exact beta*tanh'(beta z), beta=1 (correctly scaled)",
        normalize="exact",
        beta=1.0,
    ),
    "tanh": _arm("tanh", "kernel", "tanh sech^2 kernel, peak-normalised, beta=1", kernel="tanh"),
    "logistic": _arm(
        "logistic", "kernel", "logistic 4 s(1-s) kernel, peak-normalised, beta=1", kernel="logistic"
    ),
    "gaussian": _arm(
        "gaussian", "kernel", "gaussian exp(-u^2/2) kernel, peak-normalised, beta=1", kernel="gaussian"
    ),
    "cauchy": _arm(
        "cauchy",
        "kernel",
        "cauchy 1/(1+u^2) kernel (heavy tails, no dead units), peak-normalised, beta=1",
        kernel="cauchy",
    ),
    "anneal": _arm(
        "anneal",
        "kernel",
        "tanh kernel, beta annealed 0.5 -> 3 (soft-to-hard curriculum, peak)",
        kernel="tanh",
        beta_mode="anneal",
        beta=0.5,
        beta_end=3.0,
    ),
    "learnable_beta": _arm(
        "learnable_beta",
        "kernel",
        "tanh kernel with a trained bandwidth beta (init 1, peak)",
        kernel="tanh",
        beta_mode="learnable",
        beta=1.0,
    ),
    "curvature": _arm(
        "curvature",
        "curvature",
        "jet-STE: s'(z)+(h^2/6)s'''(z) windowed-average slope, beta=1 (4th-order)",
        beta=1.0,
    ),
    "scaled": _arm(
        "scaled",
        "kernel",
        "tanh kernel on per-tensor standardised z (scale-free like STE), beta=1",
        kernel="tanh",
        beta=1.0,
        scale_aware=True,
    ),
    "scaled_anneal": _arm(
        "scaled_anneal",
        "kernel",
        "scale-free tanh kernel, beta annealed 0.5 -> 3 (GNC + scale-free)",
        kernel="tanh",
        beta_mode="anneal",
        beta=0.5,
        beta_end=3.0,
        scale_aware=True,
    ),
}

#: Canonical arm order for sweeps and tables (baseline + library controls first).
ARMS: tuple[str, ...] = tuple(_ARMS)


def get_arm(name: str) -> Arm:
    """Look up an :class:`Arm` by name (one of :data:`ARMS`)."""
    try:
        return _ARMS[name]
    except KeyError:
        raise ValueError(f"unknown arm {name!r}; choose from {ARMS}") from None


__all__ = ["ARMS", "Arm", "Quantizer", "get_arm"]
