#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Check the fixed-epsilon scale generator used by the resonance argument.

This exercises live joint jets on the actual path omega*exp(-s), u*exp(s).
It checks the necessary path acceleration and replays finite rational
operands in Lean. It does not estimate the actual ODE passage remainder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from omnibias.core.proof.lean_check import check_certificate
from omnibias.core.proof.realization_replay import (
    polynomial_evaluation_certificate,
    source_digest,
    verify_replay_certificate,
)
from omnibias.core.realization import SparsePolynomial
from omnibias.jax.realization import affine_joint_jet as jax_affine
from omnibias.torch.realization import affine_joint_jet as torch_affine


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(*, lean: bool) -> dict[str, object]:
    # Axes are omega, u. The coefficients represent Taylor coefficients,
    # so the acceleration in the order-two row is divided by 2.
    omega = SparsePolynomial.variable(2, 0)
    u = SparsePolynomial.variable(2, 1)

    def generator(p: SparsePolynomial) -> SparsePolynomial:
        return u * p.derivative(1) - omega * p.derivative(0)

    polynomial = omega**3 + 2 * omega * u + u**2 + omega
    hessian_part = (
        omega**2 * polynomial.derivative(0).derivative(0)
        - 2 * omega * u * polynomial.derivative(0).derivative(1)
        + u**2 * polynomial.derivative(1).derivative(1)
    )
    acceleration = omega * polynomial.derivative(0) + u * polynomial.derivative(1)
    require(
        not (generator(generator(polynomial)) - hessian_part - acceleration).terms,
        "full curved-path second derivative identity",
    )
    require(not generator(omega * u).terms, "epsilon is invariant along the path")
    require(
        not (generator(generator(polynomial)) - (9 * omega**3 + 4 * u**2 + omega)).terms,
        "independent monomial-weight calculation",
    )

    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    points = [(Fraction(1, 2**i), Fraction(1, 2**j)) for i, j in ((4, 4), (8, 4), (4, 8))]
    samples: list[dict[str, object]] = []
    for w, v in points:
        x = torch.tensor([[float(w)], [-float(w)], [float(w / 2)]], dtype=torch.float64,
                         requires_grad=True)
        weight = torch.tensor([[[float(v)]], [[float(v)]], [[float(v / 2)]]],
                              dtype=torch.float64, requires_grad=True)
        actual = torch_affine(x, weight)
        expected = np.array([[float(w * v)], [0.0], [0.0]])
        np.testing.assert_array_equal(actual.detach().numpy(), expected)
        twin = jax_affine(jnp.asarray(x.detach().numpy()), jnp.asarray(weight.detach().numpy()))
        np.testing.assert_array_equal(np.asarray(twin), expected)
        gradients = torch.autograd.grad(actual[0].sum(), (x, weight))
        require(all(bool(torch.isfinite(g).all()) for g in gradients), "finite live gradients")
        require(all(bool(torch.count_nonzero(g)) for g in gradients), "live parameter graphs")

        # Omitting the quadratic path rows really changes the answer.
        wrong_x = x.detach().clone()
        wrong_weight = weight.detach().clone()
        wrong_x[2] = 0
        wrong_weight[2] = 0
        wrong = torch_affine(wrong_x, wrong_weight)
        require(float(2 * wrong[2, 0]) == float(-2 * w * v), "missing acceleration detected")
        samples.append({"omega": str(w), "u": str(v), "epsilon": str(w * v),
                        "correct_second_derivative": "0",
                        "straight_path_second_derivative": str(-2 * w * v)})

    # The source polynomial and rational point both appear in the generated
    # Lean proposition. This is a finite operand replay, not a proof that
    # the Python derivative compiler is sound for every polynomial.
    point = points[1]
    operands = [hessian_part, acceleration, generator(generator(polynomial))]
    expected_values = [p.evaluate(point) for p in operands]
    certificate = polynomial_evaluation_certificate(operands, point, expected_values)
    expected_digest = source_digest([p.to_payload() for p in operands])
    require(verify_replay_certificate(certificate, expected_source_digest=expected_digest),
            "exact operand-bound replay")
    altered = polynomial_evaluation_certificate([p + 1 for p in operands], point,
                                               [value + 1 for value in expected_values])
    require(not verify_replay_certificate(altered, expected_source_digest=expected_digest),
            "re-sealed operand substitution rejected")
    result: dict[str, object] | None = None
    if lean:
        repository = Path(__file__).resolve().parents[1]
        source = repository / "formal" / "omnibias-verified-kernel"
        with tempfile.TemporaryDirectory(prefix="hilbert16-scale-") as work:
            workspace = Path(work)
            shutil.copytree(source, workspace / "formal" / "omnibias-verified-kernel",
                            ignore=shutil.ignore_patterns(".lake", ".git"))
            checked = check_certificate(certificate, start=workspace, timeout=180)
            require(checked.available and checked.verified, f"Lean operand replay: {checked.detail}")
            result = asdict(checked)

    return {
        "schema": "hilbert16-fixed-epsilon-scale-jets-v1",
        "exact_polynomial_identities": 3,
        "paired_backend_samples": samples,
        "path_acceleration_negative_check": True,
        "resealed_source_substitution_rejected": True,
        "finite_operand_certificate": certificate,
        "lean_result": result,
        "scope": {
            "actual_singular_remainder_derivatives_established": False,
            "neural_approximation_used": False,
            "uniform_varying_detuning_cycle_bound_established": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lean", action="store_true")
    args = parser.parse_args()
    payload = json.dumps(run(lean=args.lean), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
