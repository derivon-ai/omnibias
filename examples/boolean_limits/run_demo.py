# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Print the factoring accuracy cliff (a negative result).

Run::

    python -m examples.boolean_limits.run_demo

It prints the soft-gate solver's success rate per factor bit-width. The rate
collapses with size: the differentiable relaxation does not shrink the factoring
search space. This is a limitation study, not an RSA attack.
"""

from __future__ import annotations

from examples.boolean_limits.experiment import success_vs_bitlength


def main() -> None:
    print("Soft-gate factoring success vs factor bit-width (the accuracy cliff):")
    rates = success_vs_bitlength(widths=(2, 3, 4), max_per_width=6, restarts=24)
    for width, rate in rates.items():
        bar = "#" * int(round(rate * 40))
        print(f"  width={width}  n={2 * width:2d} vars  success={rate:5.1%}  {bar}")
    print(
        "\nThe number of latent bits equals the preimage entropy; annealing "
        "beta -> inf\ndoes not reduce it. See docs/cookbook/rsa-limitation.md."
    )


if __name__ == "__main__":
    main()
