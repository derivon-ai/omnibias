# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Emit one self-contained ``run_demo`` command per sweep cell (scheduler-neutral).

Each printed line is a complete command that trains + instruments one
``(register, arm, seed, noise, width-block)`` slice and writes its per-run JSONs under
``<scratch-base>/<tag>/``. ``submit.sh`` wraps each line with a site-supplied submission
command; on its own this script just prints the commands (a dry run / manifest).

Invalid ``(arm, register)`` combinations (e.g. the Gauss-Newton family outside
``mse_tanh``) are skipped automatically via :func:`arms_for_register`.
"""

from __future__ import annotations

import argparse
import sys

from examples.mnist1d_double_descent.arms import ARMS, CORE_ARMS, SHARPNESS_ARMS, arms_for_register
from examples.mnist1d_double_descent.experiment import DEFAULT_WIDTHS
from examples.mnist1d_double_descent.models import REGISTERS

_ARM_GROUPS = {"core": CORE_ARMS, "all": ARMS, "sharpness": SHARPNESS_ARMS}


def _resolve_arms(names: list[str]) -> tuple[str, ...]:
    if len(names) == 1 and names[0] in _ARM_GROUPS:
        return _ARM_GROUPS[names[0]]
    return tuple(names)


def _chunk(seq: list[int], size: int) -> list[list[int]]:
    if size <= 0:
        return [list(seq)]
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def generate(args: argparse.Namespace) -> list[str]:
    """Build the list of run_demo command strings for the requested grid."""
    python = args.python or sys.executable
    arm_names = _resolve_arms(args.arms)
    width_blocks = _chunk(list(args.widths), args.width_block)
    extra = []
    if args.synthetic:
        extra.append("--synthetic")
    if args.no_pip:
        extra.append("--no-pip")
    suffix = (" " + " ".join(extra)) if extra else ""

    cmds: list[str] = []
    for register in args.registers:
        for arm in arms_for_register(register, arm_names):
            for noise in args.noises:
                for seed in args.seeds:
                    for bi, block in enumerate(width_blocks):
                        tag = f"{register}__{arm}__noise{noise:g}__seed{seed}__wb{bi}"
                        scratch = f"{args.scratch_base.rstrip('/')}/{tag}"
                        widths = " ".join(str(w) for w in block)
                        cmds.append(
                            f"{python} -m examples.mnist1d_double_descent.run_demo "
                            f"--register {register} --arms {arm} --widths {widths} "
                            f"--seeds {seed} --noise {noise:g} --steps {args.steps} "
                            f"--device {args.device} --scratch-dir {scratch}{suffix}"
                        )
    return cmds


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate scheduler-neutral sweep jobs.")
    p.add_argument("--registers", nargs="+", default=list(REGISTERS), choices=list(REGISTERS))
    p.add_argument("--arms", nargs="+", default=["all"], help="arm names or a group: core|all|sharpness")
    p.add_argument("--widths", nargs="+", type=int, default=list(DEFAULT_WIDTHS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--noises", nargs="+", type=float, default=[0.0, 0.15])
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--width-block", type=int, default=4, help="widths per job (0 = all in one)")
    p.add_argument("--device", default="auto")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--no-pip", action="store_true")
    p.add_argument("--python", default=None, help="python executable (default: this interpreter)")
    p.add_argument(
        "--scratch-base",
        default="artifacts/omnibias_mnist1d",
        help="base dir for per-job records",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    for cmd in generate(_parse_args(argv)):
        print(cmd)


if __name__ == "__main__":
    main()
