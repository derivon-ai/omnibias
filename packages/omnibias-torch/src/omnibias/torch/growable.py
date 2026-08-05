# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""GrowableOperatorMultiBiasUnit: an OMBU that can grow its bias arity ``K``
during training without changing its output shape.

Two safe growth strategies, both preserving the unit's output bit-identically
(or to a chosen approximation tolerance) at the moment of growth:

- ``"pair"`` (default; works for every base activation in the dictionary).
  Adds two new biases at an existing bias location ``b_j``, with signs
  ``(+eta, -eta)``. By Lemma 1's sum-of-signs identity (``1 + eta - eta = 1``)
  the unit's literal forward is unchanged at the moment of addition. Both new
  biases have non-zero gradients from step 1 because the two ``eta`` terms
  decouple as soon as either bias drifts.

- ``"saturate"`` (only valid for activations whose value vanishes as
  ``z -> -infinity``: sigmoid, softplus, gaussian, exp, ReLU, huber). Adds a
  single new bias at ``b_new = -big``; ``sigma(z + b_new)`` is then
  approximately zero so the new term contributes nothing. Cheaper (one new
  parameter instead of two) but the new unit's gradient is approximately zero
  at activation, so a learning-rate boost is required to wake it up.

Implementation strategy: pre-allocate biases and signs at ``K_max`` and
maintain an ``active_K`` counter. Forward only sums over the first
``active_K`` columns; inactive columns receive no gradient and stay at their
init values until ``grow()`` activates them. This avoids any optimizer-state
surgery (Adam's moments for inactive entries stay at zero, ready to learn
from scratch the moment the column becomes active).

A fresh ``GrowableOperatorMultiBiasUnit`` with ``init_K=1`` is bit-identical
to ``sigma(z + init_bias)`` and matches the K=1 OMBU in cost (because the
forward only touches one column).
"""

from __future__ import annotations

from typing import Literal

from omnibias.torch.activations.registry import ActivationSpec, get_activation
from omnibias.torch.fastpath.dispatch import multibias_literal_forward
from omnibias.torch.identity_init import identity_signs

import torch
import torch.nn as nn
from torch import Tensor

GrowStrategy = Literal["pair", "saturate"]


_SATURATE_FRIENDLY: frozenset[str] = frozenset(
    {"sigmoid", "softplus", "gaussian", "exp", "relu", "huber"}
)


class GrowableOperatorMultiBiasUnit(nn.Module):
    """A multi-bias unit whose arity ``K`` can grow during training.

    Parameters
    ----------
    num_channels : int
    init_K : int, default 1
        Initial active arity. Must be in ``[1, K_max]``.
    K_max : int, default 8
        Pre-allocated maximum arity. The biases / signs tensors are created
        at this width; only the first ``active_K`` columns are live in the
        forward pass.
    base : str or :class:`ActivationSpec`, default ``"sigmoid"``
    init_bias : float, default 0.0
        Initial value of the active bias entries. The inactive (reserve)
        bias entries are filled with ``inactive_bias_value`` (default
        ``init_bias``) so they do not surprise the user if read directly.
    learnable_signs : bool, default True
        Required for the ``"pair"`` growth strategy (the new ``+eta``,
        ``-eta`` signs need to be trainable).
    pair_eta : float, default 0.05
        Magnitude of the symmetric sign pair injected by ``"pair"`` growth.
    saturate_big : float, default 25.0
        Magnitude of the saturating bias offset injected by ``"saturate"``
        growth (sigmoid(-25) is below float32 epsilon).
    """

    spec: ActivationSpec[Tensor]
    biases: Tensor
    signs: Tensor
    _active_K: Tensor

    def __init__(
        self,
        num_channels: int,
        init_K: int = 1,
        K_max: int = 8,
        base: str | ActivationSpec[Tensor] = "sigmoid",
        init_bias: float = 0.0,
        learnable_biases: bool = True,
        learnable_signs: bool = True,
        pair_eta: float = 0.05,
        saturate_big: float = 25.0,
    ) -> None:
        super().__init__()
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")
        if K_max < 1:
            raise ValueError(f"K_max must be >= 1, got {K_max}")
        if not (1 <= init_K <= K_max):
            raise ValueError(f"init_K must be in [1, K_max={K_max}], got {init_K}")
        if pair_eta <= 0:
            raise ValueError(f"pair_eta must be > 0, got {pair_eta}")
        if saturate_big <= 0:
            raise ValueError(f"saturate_big must be > 0, got {saturate_big}")

        self.num_channels = num_channels
        self.K_max = K_max
        self.spec = base if isinstance(base, ActivationSpec) else get_activation(base)
        self.pair_eta = float(pair_eta)
        self.saturate_big = float(saturate_big)

        biases_init = torch.full((num_channels, K_max), float(init_bias))
        signs_init = torch.zeros((num_channels, K_max))
        signs_init[:, :init_K] = identity_signs(init_K).unsqueeze(0).expand(num_channels, init_K)

        if learnable_biases:
            self.biases = nn.Parameter(biases_init)
        else:
            self.register_buffer("biases", biases_init)

        if learnable_signs:
            self.signs = nn.Parameter(signs_init)
        else:
            self.register_buffer("signs", signs_init)

        # active_K is a Python int wrapped in a buffer for state_dict round-trip.
        self.register_buffer("_active_K", torch.tensor(init_K, dtype=torch.long))

    # ----- properties ------------------------------------------------------

    @property
    def active_K(self) -> int:
        """Current number of live bias terms (channel-wise)."""
        return int(self._active_K.item())

    @property
    def can_grow(self) -> bool:
        return self.active_K < self.K_max

    # ----- forward ---------------------------------------------------------

    def forward(self, z: Tensor) -> Tensor:
        """Literal multi-bias forward over the active K columns only."""
        K = self.active_K
        biases = self.biases[:, :K]
        signs = self.signs[:, :K]
        return multibias_literal_forward(z, biases, signs, self.spec.forward)

    # ----- growth ----------------------------------------------------------

    def grow(
        self,
        strategy: GrowStrategy = "pair",
        *,
        channel: int | None = None,
        anchor_value: float | Tensor | None = None,
    ) -> int:
        """Grow ``active_K`` according to the chosen strategy.

        Parameters
        ----------
        strategy : ``"pair"`` (default) or ``"saturate"``
        channel : int, optional
            Which existing bias column ``j`` to anchor the new entries to,
            for the ``"pair"`` strategy. Ignored if ``anchor_value`` is
            supplied. Defaults to the first column (``j=0``).
        anchor_value : float or 1-D tensor of length ``num_channels``,
            optional. When provided, the new pair is anchored at this
            user-specified bias location instead of at an existing bias
            column. This is the recommended way to spread new capacity
            into regions of the input distribution that are not yet
            covered (e.g. by sampling from the activation's input
            quantiles). For ``strategy="pair"`` the safe-add invariant
            still holds (signs ``(+eta, -eta)`` sum to zero, so the
            unit's output is bit-identical at the moment of growth
            *regardless of where the new biases are placed*).

        Returns
        -------
        int
            The number of newly-activated bias columns (1 for ``saturate``,
            2 for ``pair``).

        Raises
        ------
        RuntimeError
            If the requested growth would exceed ``K_max``.
        ValueError
            If ``strategy="saturate"`` is requested for an activation that
            does not vanish at ``z -> -infinity``, or if growth is requested
            on a unit built with ``learnable_signs=False`` (the injected sign
            columns could never train).
        """
        K = self.active_K
        if strategy in ("pair", "saturate") and not isinstance(self.signs, nn.Parameter):
            raise ValueError(
                f"grow(strategy={strategy!r}) requires learnable signs, but this "
                "unit was built with learnable_signs=False (signs is a frozen "
                "buffer). The injected sign column(s) would never train: 'pair' "
                "would leave the (+eta, -eta) split permanently frozen and "
                "'saturate' would leave the new column permanently zero (a dead, "
                "untrained column). Reconstruct with learnable_signs=True to grow."
            )
        if strategy == "pair":
            if K + 2 > self.K_max:
                raise RuntimeError(f"grow(pair) would exceed K_max={self.K_max}: active_K={K}")
            with torch.no_grad():
                if anchor_value is not None:
                    if isinstance(anchor_value, Tensor):
                        if anchor_value.shape != (self.num_channels,):
                            raise ValueError(
                                f"anchor_value tensor must have shape "
                                f"({self.num_channels},), got {tuple(anchor_value.shape)}"
                            )
                        anchor = anchor_value.to(self.biases.device, self.biases.dtype)
                    else:
                        anchor = torch.full(
                            (self.num_channels,),
                            float(anchor_value),
                            device=self.biases.device,
                            dtype=self.biases.dtype,
                        )
                else:
                    j = 0 if channel is None else int(channel)
                    if not (0 <= j < K):
                        raise ValueError(f"channel must be in [0, {K}), got {j}")
                    anchor = self.biases[:, j].clone()
                self.biases[:, K] = anchor
                self.biases[:, K + 1] = anchor
                self.signs[:, K] = +self.pair_eta
                self.signs[:, K + 1] = -self.pair_eta
                self._active_K += 2
            return 2

        if strategy == "saturate":
            if K + 1 > self.K_max:
                raise RuntimeError(f"grow(saturate) would exceed K_max={self.K_max}: active_K={K}")
            if self.spec.name not in _SATURATE_FRIENDLY:
                raise ValueError(
                    f"strategy='saturate' is only safe for activations that vanish "
                    f"at z -> -infinity: one of {sorted(_SATURATE_FRIENDLY)}. "
                    f"Got {self.spec.name!r}; use strategy='pair' instead."
                )
            with torch.no_grad():
                self.biases[:, K] = -self.saturate_big
                # Sign sum constraint: pre-existing signs already sum to one
                # (Lemma 1) and the new term contributes ~0, so we can leave
                # signs[:, K] at its init (zero) -- it will learn freely.
                self.signs[:, K] = 0.0
                self._active_K += 1
            return 1

        raise ValueError(f"Unknown strategy {strategy!r}; expected 'pair' or 'saturate'.")

    # ----- diagnostics -----------------------------------------------------

    def active_parameters(self) -> dict[str, Tensor]:
        """Return views into the currently-active bias / sign columns."""
        K = self.active_K
        return {"biases": self.biases[:, :K], "signs": self.signs[:, :K]}

    def extra_repr(self) -> str:
        return (
            f"num_channels={self.num_channels}, active_K={self.active_K}/{self.K_max}, "
            f"base={self.spec.name!r}, "
            f"pair_eta={self.pair_eta:g}, saturate_big={self.saturate_big:g}"
        )


__all__ = ["GrowStrategy", "GrowableOperatorMultiBiasUnit"]
