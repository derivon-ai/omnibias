# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Recover a planar source and measure the loss of two-source identifiability.

This is an untimed manufactured experiment, not a solver speed benchmark or
novelty claim. Both the library and independent control evaluate the SAME
Gaussian weak measurements. No finite-width delta surrogate is fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import torch
from omnibias.core.ridge_distribution import RidgeDistribution, collapse_dirac_cluster
from omnibias.torch.ridge_distribution import gaussian_weak

ROOT = Path(__file__).resolve().parents[1]
PROFILE = RidgeDistribution((1,), ((-1, (1,)),))
WIDTH = 0.18
TRUTH = (0.55, -0.17, 1.7)  # angle, signed offset, source strength


def measurements(parameters: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Unit-normal source; normalized full-space Gaussian probes."""
    angle, bias, strength = parameters.unbind()
    normal = torch.stack((torch.cos(angle), torch.sin(angle)))
    return strength * gaussian_weak(PROFILE, centers @ normal, WIDTH**2, bias)[..., 0]


def direct_control(parameters: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Independent closed-form normal-marginal density (standard baseline)."""
    angle, bias, strength = parameters.unbind()
    distance = centers[:, 0]*torch.cos(angle) + centers[:, 1]*torch.sin(angle) + bias
    return strength * torch.exp(-distance**2/(2*WIDTH**2))/(math.sqrt(2*math.pi)*WIDTH)


def experiment(seeds: list[int]) -> dict[str, Any]:
    torch.set_num_threads(1)
    axis = torch.linspace(-1, 1, 9, dtype=torch.float64)
    centers = torch.cartesian_prod(axis, axis)
    truth = torch.tensor(TRUTH, dtype=centers.dtype)
    target = direct_control(truth, centers).detach()
    generator = torch.Generator().manual_seed(4901)
    held_out = 2*torch.rand((121, 2), dtype=centers.dtype, generator=generator)-1
    held_target = direct_control(truth, held_out)
    runs = []
    for seed in seeds:
        generator.manual_seed(seed)
        initial = truth + torch.tensor([0.35, 0.12, 0.5], dtype=truth.dtype) * (
            2*torch.rand(3, generator=generator, dtype=truth.dtype)-1)
        parameters = initial.clone().requires_grad_()
        optimizer = torch.optim.LBFGS([parameters], max_iter=100, max_eval=150,
                                       tolerance_grad=1e-12, tolerance_change=1e-15,
                                       line_search_fn="strong_wolfe")
        evaluations = 0

        def closure(optimizer=optimizer, parameters=parameters) -> torch.Tensor:
            nonlocal evaluations
            optimizer.zero_grad()
            loss = (measurements(parameters, centers)-target).square().mean()
            loss.backward()
            evaluations += 1
            return loss

        optimizer.step(closure)
        prediction = measurements(parameters, held_out)
        value = measurements(parameters, centers)
        control = direct_control(parameters, centers)
        gradient = torch.autograd.grad(value.square().mean(), parameters)[0]
        control_gradient = torch.autograd.grad(control.square().mean(), parameters)[0]
        runs.append({
            "seed": seed, "initial": initial.tolist(), "recovered": parameters.detach().tolist(),
            "max_parameter_abs_error": float((parameters-truth).abs().max().detach()),
            "held_out_relative_l2": float((prediction-held_target).norm().detach()/held_target.norm()),
            "control_max_value_difference": float((value-control).abs().max().detach()),
            "control_max_gradient_difference": float((gradient-control_gradient).abs().max().detach()),
            "closure_evaluations": evaluations,
        })

    normal = torch.tensor([math.cos(TRUTH[0]), math.sin(TRUTH[0])], dtype=centers.dtype)
    projected = centers @ normal
    # Columns parameterize scaled derivative moments, NOT separate source
    # locations. Their conditioning must not be sold as restored resolution.
    jet = RidgeDistribution((1,), tuple((-k-1, tuple(F(9, 50)**k if j == k else F(0)
                                                   for j in range(4))) for k in range(4)), outputs=4)
    jet_matrix = gaussian_weak(jet, projected, WIDTH**2)
    jet_singular = torch.linalg.svdvals(jet_matrix)
    jet_condition = float(jet_singular[0]/jet_singular[-1])
    conditioning = []
    for separation in (0., 0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
        # Dimensionless unknowns: the two offsets divided by probe width and
        # their two strengths. The two sources have the same fixed normal.
        p = torch.tensor([-separation/(2*WIDTH), separation/(2*WIDTH), 1., 1.],
                         dtype=centers.dtype, requires_grad=True)

        def two_sources(q: torch.Tensor) -> torch.Tensor:
            return sum(q[j+2] * gaussian_weak(PROFILE, projected, WIDTH**2, WIDTH*q[j])[..., 0]
                       for j in range(2))

        jacobian = torch.autograd.functional.jacobian(two_sources, p)
        singular = torch.linalg.svdvals(jacobian)
        threshold = torch.finfo(centers.dtype).eps * max(jacobian.shape) * singular[0]
        rank = int((singular > threshold).sum())
        half = F(str(separation))/2
        collapse = collapse_dirac_cluster((1,), (-half, half), (1, 1),
                                          max_order=3, min_probe_std=F(9, 50))
        original = two_sources(p)
        collapsed = gaussian_weak(collapse.field, projected, WIDTH**2)[..., 0]
        conditioning.append({
            "separation": separation, "separation_over_probe_width": separation/WIDTH,
            "dimensionless_singular_values": singular.tolist(), "numerical_rank": rank,
            "condition_number": float(singular[0]/singular[-1]) if rank == 4 else None,
            "rank_is_floating_diagnostic_not_certificate": True,
            "collapsed_four_moment_condition_number": jet_condition,
            "sampled_collapse_max_error": float((original-collapsed).abs().max().detach()),
            "analytic_uniform_weak_error_bound_exact": str(collapse.weak_error_bound),
            "analytic_bound_does_not_enclose_float_rounding": True,
        })
    sources = [Path(__file__).relative_to(ROOT),
               Path("packages/omnibias-core/src/omnibias/core/ridge_distribution.py"),
               Path("packages/omnibias-torch/src/omnibias/torch/ridge_distribution.py")]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {
        "scope": "Manufactured noiseless one-sheet inverse problem; local two-sheet identifiability",
        "not_established": ["new mathematics", "speed advantage", "noisy-data robustness",
                            "general PDE solver", "global identification guarantee", "rigorous enclosure"],
        "truth": dict(zip(("angle", "bias", "strength"), TRUTH, strict=True)),
        "probe_width": WIDTH, "training_probes": len(centers), "held_out_probes": len(held_out),
        "seeds": seeds, "runs": runs, "two_source_conditioning": conditioning,
        "source_commit": commit,
        "source_worktree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "source_sha256": {str(path): hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in sources},
        "environment": {"python": platform.python_version(), "device": "cpu", "dtype": "float64",
                        "torch_threads": torch.get_num_threads()},
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "omnibias-core")},
        "all_recovery_checks_passed": all(row["max_parameter_abs_error"] < 1e-6
                                          and row["held_out_relative_l2"] < 1e-6 for row in runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    if not args.seeds:
        parser.error("at least one seed is required")
    result = experiment(args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"all_recovery_checks_passed": result["all_recovery_checks_passed"],
                      "max_parameter_error": max(row["max_parameter_abs_error"] for row in result["runs"]),
                      "max_held_out_relative_l2": max(row["held_out_relative_l2"] for row in result["runs"]),
                      "two_source_conditions": [(row["separation"], row["condition_number"])
                                                for row in result["two_source_conditioning"]]}, indent=2))
    if not result["all_recovery_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
