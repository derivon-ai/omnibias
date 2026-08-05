# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Learnable-temperature activation layers (Keras backend).

``keras.layers.Layer`` twins of :mod:`omnibias.torch.tempered_blocks`: they hold
a differentiable temperature ``beta`` (or negative slope ``alpha``) and call the
closed-form tempered / piecewise tower each forward, so the same weight trains on
TensorFlow, JAX, or PyTorch through ``keras.ops``.

* :class:`TemperedActivation` -- ``softplus(beta z)/beta -> relu``,
  ``sigmoid(beta z) -> step``, ``tanh(beta z) -> sign`` (pick the base + scale).
* :class:`LearnablePReLU` -- leaky ReLU with a learnable negative slope on the
  almost-everywhere tower.

``beta`` (and ``alpha``) should stay positive; add your own reparameterisation
if you need to guarantee it.
"""

from __future__ import annotations

from typing import Any

from omnibias.core.spec import ActivationSpec, make_tempered_fastpath
from omnibias.keras.activations.registry import get_activation

from keras import initializers, layers, ops


class TemperedActivation(layers.Layer):
    """Beta-tempered smooth surrogate of a base activation's tower.

    Parameters
    ----------
    base : str or :class:`ActivationSpec`, default ``"softplus"``
        Base activation carrying a fastpath. The surrogate tower is
        ``beta**(n - p) * base.fastpath(beta * z, n)``.
    beta : float, default 1.0
        Initial temperature (larger = sharper; ``beta -> inf`` approaches the
        hard activation).
    scale : {"one_over_beta", "unit"}, default ``"one_over_beta"``
        ``"one_over_beta"`` (``p = 1``) for ``softplus(beta z)/beta -> relu``;
        ``"unit"`` (``p = 0``) for bounded surrogates like ``sigmoid`` / ``tanh``.
    learnable_beta : bool, default False
        If True, ``beta`` is a trainable weight; otherwise it is frozen.
    """

    def __init__(
        self,
        base: str | ActivationSpec[Any] = "softplus",
        beta: float = 1.0,
        *,
        scale: str = "one_over_beta",
        learnable_beta: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        spec = base if isinstance(base, ActivationSpec) else get_activation(base)
        if spec.fastpath is None:
            raise ValueError(
                f"TemperedActivation requires a base with a fastpath; {spec.name!r} has none."
            )
        if scale == "one_over_beta":
            self._scale_power = 1
        elif scale == "unit":
            self._scale_power = 0
        else:
            raise ValueError(f"scale must be 'unit' or 'one_over_beta', got {scale!r}")
        self._base_name = base if isinstance(base, str) else spec.name
        self.scale = scale
        self.init_beta = float(beta)
        self.learnable_beta = learnable_beta
        self._base_fastpath = spec.fastpath
        self.beta = self.add_weight(
            name="beta",
            shape=(),
            initializer=initializers.Constant(float(beta)),
            trainable=learnable_beta,
            dtype=self.variable_dtype,
        )
        self.built = True

    def fastpath(self, z: Any, n: int) -> Any:
        """Closed-form ``n``-th derivative of the tempered surrogate at current ``beta``."""
        kernel = make_tempered_fastpath(
            self._base_fastpath, self.beta, scale_power=self._scale_power
        )
        return kernel(z, n)

    def call(self, z: Any) -> Any:
        return self.fastpath(z, 0)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "base": self._base_name,
                "beta": self.init_beta,
                "scale": self.scale,
                "learnable_beta": self.learnable_beta,
            }
        )
        return config


class LearnablePReLU(layers.Layer):
    """Leaky ReLU with a (learnable) negative slope ``alpha`` on the a.e. tower.

    ``call(z) = where(z > 0, z, alpha * z)``; ``fastpath`` gives the
    almost-everywhere tower (``n = 1`` step-with-slope, ``n >= 2 -> 0``).
    """

    def __init__(
        self,
        negative_slope: float = 0.25,
        *,
        learnable: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.init_negative_slope = float(negative_slope)
        self.learnable = learnable
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=initializers.Constant(float(negative_slope)),
            trainable=learnable,
            dtype=self.variable_dtype,
        )
        self.built = True

    def fastpath(self, z: Any, n: int) -> Any:
        if n < 0:
            raise ValueError(f"order n must be >= 0, got {n}.")
        if n == 0:
            return ops.where(z > 0, z, self.alpha * z)
        if n == 1:
            return ops.where(z > 0, ops.ones_like(z), self.alpha * ops.ones_like(z))
        return ops.zeros_like(z)

    def call(self, z: Any) -> Any:
        return self.fastpath(z, 0)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "negative_slope": self.init_negative_slope,
                "learnable": self.learnable,
            }
        )
        return config


__all__ = ["LearnablePReLU", "TemperedActivation"]
