# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""PyTorch twin of the Cauchy-Hardy dictionary (theory 01-12).

``H`` is a signed permutation of coefficients: no quadrature. Line Hilbert
only; commutation needs ``alpha > 0``.
"""

from __future__ import annotations

from omnibias.core.conjugate import HardyDictionary, hardy_p_deriv_n, hardy_q_deriv_n

import torch
from torch import Tensor


def hardy_atoms(y: Tensor, dictionary: HardyDictionary) -> Tensor:
    """Evaluate every atom at ``y``. Result shape ``(*y.shape, n_atoms)``."""
    y_list = y.detach().cpu().reshape(-1).tolist()
    n_atoms = len(dictionary.atoms)
    flat: list[list[float]] = []
    for yi in y_list:
        row: list[float] = []
        for atom in dictionary.atoms:
            if atom.parity == "even":
                row.append(hardy_p_deriv_n(float(yi), atom.scale, atom.exponent, atom.order))
            else:
                row.append(hardy_q_deriv_n(float(yi), atom.scale, atom.exponent, atom.order))
        flat.append(row)
    out = torch.tensor(flat, dtype=y.dtype, device=y.device)
    return out.reshape(*y.shape, n_atoms)


def hilbert_coeffs(coeffs: Tensor, dictionary: HardyDictionary) -> Tensor:
    """Apply the signed permutation along the last axis. No quadrature."""
    perm = dictionary.hilbert_permutation()
    n = coeffs.shape[-1]
    if n != len(perm):
        raise ValueError("coeffs last axis must match the dictionary")
    pieces = []
    for j in range(n):
        acc = torch.zeros(coeffs.shape[:-1], dtype=coeffs.dtype, device=coeffs.device)
        for i, (target, sign) in enumerate(perm):
            if target == j:
                acc = acc + float(sign) * coeffs[..., i]
        pieces.append(acc)
    return torch.stack(pieces, dim=-1)


def evaluate_expansion(coeffs: Tensor, y: Tensor, dictionary: HardyDictionary) -> Tensor:
    """``sum_k c_k atom_k(y)``."""
    atoms = hardy_atoms(y, dictionary)
    return (atoms * coeffs).sum(dim=-1)


__all__ = ["evaluate_expansion", "hardy_atoms", "hilbert_coeffs"]
