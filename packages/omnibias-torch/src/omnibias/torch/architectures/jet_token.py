# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet-token mix (torch; theory 09-02).

Tokens are N-jets mixed by ``compose_jet``, not a softmax of dots.
The founding bias collapse (``delta -> 0``) supplies ``sigma^(n)``.
Temperature collapse (``beta -> inf``, feasibility) does not appear
in the default mix. Do not conflate the two.
"""

from __future__ import annotations

from omnibias.core import jet_token as core
from omnibias.torch.activations.registry import get_activation
from omnibias.torch.jet import compose_jet

import torch
from torch import Tensor

DISCLAIMER = core.DISCLAIMER
JetTokenConfig = core.JetTokenConfig
honesty_payload = core.honesty_payload
worked_example = core.worked_example


def jet_token_forward(tokens: Tensor, *, config: JetTokenConfig | None = None) -> Tensor:
    """``tokens`` is ``(2, seq)`` for the order-1 worked mix, or a pair of 1-jets.

    The mix is an affine jet combination followed by ``compose_jet`` of
    ``tanh``. Softmax-of-values is refused.
    """
    cfg = JetTokenConfig() if config is None else config
    core.check_full_width(cfg, n_params=max(int(cfg.width) * 4, 1))
    spec = get_activation("tanh")
    if spec.fastpath is None:
        raise TypeError("tanh must expose a fastpath")
    mixed = 0.5 * tokens[0] + 0.5 * tokens[1]
    u0 = mixed[0]
    tower = torch.stack((spec.forward(u0), spec.fastpath(u0, 1)), dim=0)
    u_jet = mixed.reshape(2)
    return compose_jet(u_jet, tower)


def worked_compose_jet() -> Tensor:
    """Worked example through the live ``compose_jet`` kernel."""
    tokens = torch.tensor(((1.0, 0.5), (0.0, 1.0)), dtype=torch.get_default_dtype())
    return jet_token_forward(tokens, config=JetTokenConfig(width=2, allow_full=True))


__all__ = [
    "DISCLAIMER",
    "JetTokenConfig",
    "honesty_payload",
    "jet_token_forward",
    "worked_compose_jet",
    "worked_example",
]
