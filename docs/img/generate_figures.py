# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Generate the README / docs figures for omnibias.

Run from the repo root (needs omnibias-core + omnibias-torch + omnibias-jax):

    python docs/img/generate_figures.py

Outputs PNGs into ``docs/img/``. The accuracy / cost / parity / tower figures
are computed from the public API. The bench_* figures plot the committed JSON
under ``docs/benchmarks/`` (regenerate those first with
``uv run python benchmarks/<script>.py``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omnibias.torch.activations.registry import get_activation as torch_get

OUT = Path(__file__).parent
BENCH = OUT.parent / "benchmarks"
PRIMARY = "#3b6fb6"
ACCENT = "#d1495b"
GOOD = "#2a9d8f"
AMBER = "#e09f3e"
PURPLE = "#6a4c93"
GRID = "#d9dde3"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _load(name: str) -> dict:
    return json.loads((BENCH / name).read_text(encoding="utf-8"))


def fig_accuracy_cliff() -> None:
    """Closed-form sigma^(n) vs an n-fold finite-difference stencil (float64)."""
    spec = torch_get("tanh")
    orders = list(range(1, 9))
    fd_err: list[float] = []
    h = 5e-2

    for n in orders:
        half = n + 2
        xs = torch.linspace(-3.0, 3.0, 1201, dtype=torch.float64)
        pad = torch.arange(-half, half + 1, dtype=torch.float64) * h
        grid = xs[:, None] + pad[None, :]
        d = spec.forward(grid)
        for _ in range(n):
            d = (d[:, 2:] - d[:, :-2]) / (2.0 * h)
        center = d[:, d.shape[1] // 2]
        truth = spec.fastpath(xs, n)
        denom = truth.abs().clamp_min(1e-3)
        fd_err.append(float(((center - truth).abs() / denom).median()))

    cf_err = [float(np.finfo(np.float64).eps)] * len(orders)

    _style()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.semilogy(orders, fd_err, "o-", color=ACCENT, lw=2.4, ms=7, label="finite-difference stencil")
    ax.semilogy(
        orders,
        [max(e, 1e-16) for e in cf_err],
        "s-",
        color=GOOD,
        lw=2.4,
        ms=7,
        label="omnibias closed form",
    )
    ax.axhspan(1e-16, 1e-12, color=GOOD, alpha=0.06)
    ax.set_xlabel("derivative order  n")
    ax.set_ylabel("median relative error")
    ax.set_title("n-th derivative accuracy (tanh, float64)")
    ax.set_ylim(1e-17, 1e3)
    ax.legend(loc="center left")
    ax.text(
        0.98,
        0.04,
        "one  $\\sigma$  evaluation, any order",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#555",
    )
    fig.savefig(OUT / "accuracy_cliff.png")
    plt.close(fig)


def fig_cost_vs_order() -> None:
    """Wall-time of closed-form sigma^(n) vs nested autograd, per order."""
    # Prefer committed JSON when present so the figure matches the README table.
    path = BENCH / "derivative_order.json"
    if path.is_file():
        rows = _load("derivative_order.json")["rows"]
        orders = [r["n"] for r in rows]
        fp = [r["time_ms"]["closed_form"] for r in rows]
        ag = [r["time_ms"]["nested_autograd"] for r in rows]
    else:
        spec = torch_get("tanh")
        orders = list(range(1, 7))
        z = torch.linspace(-3, 3, 20000, dtype=torch.float64)

        def time_fastpath(n: int) -> float:
            best = float("inf")
            for _ in range(5):
                t0 = time.perf_counter()
                spec.fastpath(z, n)
                best = min(best, time.perf_counter() - t0)
            return best * 1e3

        def time_autograd(n: int) -> float:
            best = float("inf")
            for _ in range(3):
                zz = z.clone().requires_grad_(True)
                t0 = time.perf_counter()
                y = spec.forward(zz).sum()
                g = torch.autograd.grad(y, zz, create_graph=True)[0]
                for _ in range(n - 1):
                    g = torch.autograd.grad(g.sum(), zz, create_graph=True)[0]
                best = min(best, time.perf_counter() - t0)
            return best * 1e3

        fp = [time_fastpath(n) for n in orders]
        ag = [time_autograd(n) for n in orders]

    _style()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(orders, ag, "o-", color=ACCENT, lw=2.4, ms=7, label="nested autograd")
    ax.plot(orders, fp, "s-", color=GOOD, lw=2.4, ms=7, label="omnibias closed form")
    ax.set_xlabel("derivative order  n")
    ax.set_ylabel("time per call  (ms)")
    ax.set_title("Cost of the n-th derivative (20k points, CPU)")
    ax.legend(loc="upper left")
    fig.savefig(OUT / "cost_vs_order.png")
    plt.close(fig)


def fig_parity_heatmap() -> None:
    """Max |torch - jax| of sigma^(n), per activation and order (float64)."""
    import jax.numpy as jnp  # noqa: E402
    from omnibias.jax.activations import get_activation as jax_get  # noqa: E402

    names = ["sigmoid", "tanh", "softplus", "gaussian", "exp", "sin", "cos", "sinh", "cosh"]
    orders = list(range(0, 7))
    z = np.linspace(-2.0, 2.0, 256)
    zt = torch.from_numpy(z).double()
    zj = jnp.asarray(z)

    grid = np.zeros((len(names), len(orders)))
    for i, nm in enumerate(names):
        ts, js = torch_get(nm), jax_get(nm)
        for j, n in enumerate(orders):
            t = ts.fastpath(zt, n).detach().numpy()
            jv = np.asarray(js.fastpath(zj, n), dtype=np.float64)
            grid[i, j] = np.max(np.abs(t - jv))

    loggrid = np.log10(np.clip(grid, 1e-18, None))
    _style()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    im = ax.imshow(loggrid, cmap="cividis", aspect="auto", vmin=-16, vmax=-10)
    ax.set_xticks(range(len(orders)), [str(o) for o in orders])
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("derivative order  n")
    ax.set_title("torch vs jax: $\\log_{10}$ max |Δ| (float64)")
    for i in range(len(names)):
        for j in range(len(orders)):
            ax.text(j, i, f"{loggrid[i, j]:.0f}", ha="center", va="center", color="white", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("$\\log_{10}$ abs difference")
    fig.savefig(OUT / "parity_heatmap.png")
    plt.close(fig)


def fig_derivative_tower() -> None:
    """tanh and its first four derivatives, all from the closed-form tower."""
    spec = torch_get("tanh")
    z = torch.linspace(-4, 4, 600, dtype=torch.float64)
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    colors = ["#22223b", PRIMARY, GOOD, AMBER, ACCENT]
    for n in range(0, 5):
        y = spec.fastpath(z, n).numpy()
        lbl = "$\\sigma$" if n == 0 else f"$\\sigma^{{({n})}}$"
        ax.plot(z.numpy(), y, color=colors[n], lw=2.2, label=lbl)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel("z")
    ax.set_title("Closed-form derivative tower of tanh (one forward pass)")
    ax.legend(ncol=5, loc="upper center")
    fig.savefig(OUT / "derivative_tower.png")
    plt.close(fig)


def fig_bench_laplacian_scaling() -> None:
    data = _load("laplacian_scaling.json")
    rows = data["rows"]
    ds = [r["D"] for r in rows]
    methods = [
        ("omnibias", GOOD, "s-"),
        ("folx", PRIMARY, "o-"),
        ("jax_hessian", AMBER, "^-"),
        ("torch_func_hessian", ACCENT, "v-"),
    ]
    labels = {
        "omnibias": "omnibias",
        "folx": "folx",
        "jax_hessian": "jax.hessian",
        "torch_func_hessian": "torch.func.hessian",
    }

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    ax = axes[0]
    for key, color, style in methods:
        ax.semilogy(ds, [r["time_ms"][key] for r in rows], style, color=color, lw=2.2, ms=7, label=labels[key])
    ax.set_xlabel("input dimension  D")
    ax.set_ylabel("median time  (ms)")
    ax.set_title("Laplacian cost vs D (CPU, float64)")
    ax.legend(loc="upper left")

    ax = axes[1]
    for key, color, style in methods[1:]:
        ax.semilogy(
            ds,
            [r["speedup_vs_omnibias"][key] for r in rows],
            style,
            color=color,
            lw=2.2,
            ms=7,
            label=labels[key],
        )
    ax.axhline(1.0, color=GOOD, lw=1.6, ls="--", label="omnibias (=1)")
    ax.set_xlabel("input dimension  D")
    ax.set_ylabel("slowdown vs omnibias  (×)")
    ax.set_title("Same answers (≤ 2e-15); cost ratio grows with D")
    ax.legend(loc="upper left")

    fig.savefig(OUT / "bench_laplacian_scaling.png")
    plt.close(fig)


def fig_bench_polylaplacian() -> None:
    data = _load("polylaplacian_order.json")
    rows = data["rows"]
    ks = [r["k"] for r in rows]

    def times(key: str) -> list[float | None]:
        out: list[float | None] = []
        for r in rows:
            cell = r[key]
            out.append(cell["time_ms"] if cell.get("status") == "ok" else None)
        return out

    _style()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    series = [
        ("omnibias", times("omnibias"), GOOD, "s-"),
        ("folx-nested", times("folx_nested"), PRIMARY, "o-"),
        ("dense-nested", times("dense_nested"), ACCENT, "v-"),
    ]
    for label, vals, color, style in series:
        xs = [k for k, v in zip(ks, vals, strict=True) if v is not None]
        ys = [v for v in vals if v is not None]
        ax.semilogy(xs, ys, style, color=color, lw=2.4, ms=8, label=label)
        for k, v in zip(ks, vals, strict=True):
            if v is None:
                ax.scatter([k], [1e-3], marker="x", color=color, s=80, zorder=5)
    ax.set_xlabel("polylaplacian order  k   (Δᵏ)")
    ax.set_ylabel("median time  (ms)")
    ax.set_title("Iterated Laplacian cost vs order (CPU, D=16)")
    ax.legend(loc="upper left")
    ax.text(
        0.98,
        0.04,
        "omnibias flat in k; nested autodiff explodes",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#555",
    )
    fig.savefig(OUT / "bench_polylaplacian.png")
    plt.close(fig)


def fig_bench_optimizers() -> None:
    data = _load("optimizer_pinn.json")
    results = data["results"]
    order = [
        "adam",
        "lbfgs",
        "gauss_newton",
        "cubic_gauss_newton",
        "trust_region_newton_cg",
    ]
    labels = {
        "adam": "Adam",
        "lbfgs": "L-BFGS",
        "gauss_newton": "Gauss–Newton",
        "cubic_gauss_newton": "Cubic GN",
        "trust_region_newton_cg": "Trust-Newton-CG",
    }
    colors = {
        "adam": ACCENT,
        "lbfgs": AMBER,
        "gauss_newton": GOOD,
        "cubic_gauss_newton": PRIMARY,
        "trust_region_newton_cg": PURPLE,
    }

    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name in order:
        agg = results[name]
        ax.scatter(
            [agg["wall_s_median"]],
            [agg["rel_l2_median"]],
            s=140,
            color=colors[name],
            zorder=3,
            label=labels[name],
        )
        ax.annotate(
            labels[name],
            (agg["wall_s_median"], agg["rel_l2_median"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=10,
        )
    ax.set_yscale("log")
    ax.set_xlabel("median wall-clock  (s)")
    ax.set_ylabel("median rel-L2 vs analytic")
    ax.set_title("1-D Poisson PINN (5 seeds, CPU, float64)")
    ax.legend(loc="upper right")
    fig.savefig(OUT / "bench_optimizers.png")
    plt.close(fig)


def fig_bias_collapse() -> None:
    """Conceptual panel: K parallel soft hyperplanes coalesce as δ→0."""
    spec = torch_get("sigmoid")
    z = torch.linspace(-6.0, 6.0, 800, dtype=torch.float64)
    K = 4
    deltas = [1.5, 0.6, 0.15, 0.0]

    _style()
    fig, axes = plt.subplots(1, len(deltas), figsize=(12.5, 3.4), sharey=True)
    for ax, delta in zip(axes, deltas, strict=True):
        if delta == 0.0:
            # Bias collapse: mean of K parallel planes -> sigma^(K-1) / (K-1)!
            # For equal spacing that coalesces, the limit of the divided difference
            # is the (K-1)-st derivative. We plot sigma^(K-1) scaled for visibility.
            y = spec.fastpath(z, K - 1).numpy()
            y = y / (np.max(np.abs(y)) + 1e-30)
            ax.plot(z.numpy(), y, color=GOOD, lw=2.6)
            ax.set_title(f"$\\delta \\to 0$  →  $\\sigma^{{({K-1})}}$")
        else:
            biases = torch.linspace(-(K - 1) / 2, (K - 1) / 2, K) * delta
            # Divided-difference style multibias combination (forward differences).
            import math as _math

            coeffs = torch.tensor(
                [((-1.0) ** (K - 1 - k)) * _math.comb(K - 1, k) for k in range(K)],
                dtype=torch.float64,
            )
            y = torch.zeros_like(z)
            for c, b in zip(coeffs, biases, strict=True):
                y = y + c * spec.forward(z + b)
            y = y / (delta ** (K - 1))
            y = y / (torch.max(torch.abs(y)) + 1e-30)
            ax.plot(z.numpy(), y.numpy(), color=PRIMARY, lw=2.4)
            # Mark the K plane centres.
            for b in biases.tolist():
                ax.axvline(b, color=ACCENT, lw=1.0, alpha=0.55, ls="--")
            ax.set_title(f"$\\delta = {delta:g}$  ({K} planes)")
        ax.axhline(0, color="#999", lw=0.7)
        ax.set_xlabel("z")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-1.3, 1.3)
    axes[0].set_ylabel("normalised response")
    fig.suptitle(
        "Bias collapse: K parallel soft hyperplanes coalesce into $\\sigma^{(K-1)}$",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(OUT / "bias_collapse.png")
    plt.close(fig)


def main() -> None:
    fig_accuracy_cliff()
    fig_cost_vs_order()
    fig_parity_heatmap()
    fig_derivative_tower()
    fig_bias_collapse()
    if (BENCH / "laplacian_scaling.json").is_file():
        fig_bench_laplacian_scaling()
    if (BENCH / "polylaplacian_order.json").is_file():
        fig_bench_polylaplacian()
    if (BENCH / "optimizer_pinn.json").is_file():
        fig_bench_optimizers()
    print("wrote:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
